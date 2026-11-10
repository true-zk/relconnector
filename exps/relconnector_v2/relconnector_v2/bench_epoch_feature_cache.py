"""CPU-only, ID-stable multi-epoch feature-cache replay on real SQLite rows.

The column files model a persistent ID-keyed encoded-feature store. They are
intentionally simple torch-serialized blocks, not the production text cache.
The same sampled IDs recur in each simulated epoch, so warm results are an
upper bound when training requests vary or the cache cannot cover all IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch_frame import TensorFrame, cat, stype
from torch_frame.data.multi_tensor import _MultiTensor

from relconnector_v2.connector import PandasDatabaseReader
from relconnector_v2.feature_read_profiles import tuned_feature_read_profile
from relconnector_v2.features.schema import (
    TableFeatureSchema,
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
)
from relconnector_v2.features.sql import SqlFeatureFetcher
from relconnector_v2.features.text import GloveTextEmbedder
from relconnector_v2.profile_feature_processing import (
    _sample_ids,
    _schema_from_existing_stats,
)


def _fetch(
    reader: PandasDatabaseReader,
    schema: TensorFrameFeatureSchema,
    table: str,
    ids: list[int],
    window_batches: int,
    threshold: float,
    batch_rows: int,
) -> tuple[pd.DataFrame, float]:
    fetcher = SqlFeatureFetcher(
        reader,
        feature_columns=schema.feature_columns,
        block_size=4096,
        block_density_threshold=threshold,
        cache_bytes=0,
    )
    window_rows = window_batches * batch_rows
    frames = []
    start = time.perf_counter()
    for offset in range(0, len(ids), window_rows):
        frame, _ = fetcher._fetch_table(
            table, torch.tensor(ids[offset : offset + window_rows], dtype=torch.long)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), time.perf_counter() - start


def _only_columns(
    schema: TensorFrameFeatureSchema, table: str, columns: set[str]
) -> TensorFrameFeatureSchema:
    spec = schema.tables[table]
    kept = tuple(col for col in spec.columns if col.name in columns)
    reduced = TableFeatureSchema(
        kept,
        {name: kind for name, kind in spec.col_to_stype.items() if name in columns},
        {name: stats for name, stats in spec.col_stats.items() if name in columns},
    )
    return TensorFrameFeatureSchema({table: reduced}, "cpu-cache-replay")


def _cache_path(root: Path, dataset: str, table: str, column: str) -> Path:
    digest = hashlib.sha256(f"{table}\0{column}".encode()).hexdigest()[:16]
    return root / dataset / f"{digest}.pt"


def _load(path: Path, rows: int) -> Any:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if len(value) != rows:
        raise AssertionError(f"Wrong cached row count: {path}")
    return value


def _column_value(frame: TensorFrame, name: str, kind: stype) -> Any:
    parent = kind.parent
    position = frame.col_names_dict[parent].index(name)
    feature = frame.feat_dict[parent]
    if isinstance(feature, dict):
        return {key: value[:, position] for key, value in feature.items()}
    if isinstance(feature, _MultiTensor):
        return feature[:, position]
    return feature[:, position].unsqueeze(1)


def _compact(value: Any) -> Any:
    """Detach a column slice from shared backing storage before saving it."""
    if isinstance(value, torch.Tensor):
        return value.contiguous().clone()
    if isinstance(value, _MultiTensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    raise TypeError(f"Unsupported encoded column: {type(value).__name__}")


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype.is_floating_point:
            return bool(torch.allclose(left, right, rtol=0, atol=0, equal_nan=True))
        return torch.equal(left, right)
    if isinstance(left, _MultiTensor) and isinstance(right, _MultiTensor):
        return (
            type(left) is type(right)
            and _same(left.values, right.values)
            and torch.equal(left.offset, right.offset)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    return False


def _restore(
    table: str,
    columns: dict[str, Any],
    cache_dir: Path,
    dataset: str,
    rows: int,
    uncached: TensorFrame | None = None,
) -> TensorFrame:
    parts = []
    for name, item in columns.items():
        kind = stype(item["type"]).parent
        if uncached is not None and not item["cache_selected"]:
            value = _column_value(uncached, name, kind)
        else:
            value = _load(_cache_path(cache_dir, dataset, table, name), rows)
        parts.append(TensorFrame({kind: cast(Any, value)}, {kind: [name]}))
    restored = cat(parts, dim=1)
    if len(restored) != rows or restored.num_cols != len(columns):
        raise AssertionError(f"Wrong restored TensorFrame: {dataset}/{table}")
    return restored


def _median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-dir", type=Path, default=Path("exps/data/relbench"))
    parser.add_argument(
        "--schema-cache-root",
        type=Path,
        default=Path(
            "exps/benchmarks/tuned-component-ablation-all10-100x5-20261007-v1"
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("exps/benchmarks/feature-processing-20261027/all10-rows512.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-table", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--batch-rows", nargs="+", type=int, default=[32, 64, 128, 256, 512]
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.rows_per_table, args.epochs, args.repeats, *args.batch_rows) <= 0:
        parser.error("sizes, epochs, and repeats must be positive")
    if args.rows_per_table != 512:
        parser.error("column policy currently requires the matching 512-row profile")

    torch.set_num_threads(1)
    profile = json.loads(args.profile.read_text())
    cache_dir = args.output.parent / "id-feature-cache-blocks"
    paths = sorted(args.database_dir.glob("rel-*.sqlite"))
    if args.datasets:
        wanted = set(args.datasets)
        paths = [path for path in paths if path.stem in wanted]
        if {path.stem for path in paths} != wanted:
            parser.error("A requested dataset is missing")
    output: dict[str, Any] = {
        "status": "running",
        "design": {
            "cpu_only": True,
            "rows_per_table": args.rows_per_table,
            "epochs": args.epochs,
            "repeats": args.repeats,
            "batch_rows": args.batch_rows,
            "cache_format": "one torch-serialized encoded column per sampled ID block",
            "warm_hit_assumption": "identical sampled IDs in every epoch; all cached blocks hit",
            "policy": "cache if (epochs-1)*(column map+prepare - disk read) > disk write and map+prepare > disk read",
            "sql_read_policy": "frozen per-dataset phase-one window/density; no in-memory row cache",
            "page_cache": "ordinary OS page cache; no fsync or drop_caches",
        },
        "datasets": {},
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("design") != output["design"]:
            parser.error("Cannot resume with a different experiment design")
        output = previous
        output["status"] = "running"
    embedder = GloveTextEmbedder(cache_bytes=0, execution="direct")
    embedder(["warmup"])
    try:
        for path in paths:
            dataset = path.stem
            if dataset in output["datasets"]:
                print(f"[{dataset}] already complete", flush=True)
                continue
            print(f"[{dataset}] starting", flush=True)
            read_profile = tuned_feature_read_profile(dataset)
            reader = PandasDatabaseReader(path)
            schema, source = _schema_from_existing_stats(
                reader, dataset, args.schema_cache_root, args.database_dir
            )
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            previous = profile["datasets"][dataset]["tables"]
            result: dict[str, Any] = {
                "read_profile": {
                    "window_batches": read_profile.window_batches,
                    "density_threshold": read_profile.density_threshold,
                },
                "schema_source": source,
                "tables": {},
            }
            try:
                for table, spec in schema.tables.items():
                    if not spec.columns:
                        continue
                    ids = _sample_ids(
                        connection,
                        table,
                        reader.schemas()[table].primary_key,
                        args.rows_per_table,
                    )
                    if not ids:
                        continue
                    batch_rows = max(1, (args.rows_per_table + 31) // 32)
                    frame, first_read = _fetch(
                        reader,
                        schema,
                        table,
                        ids,
                        read_profile.window_batches,
                        read_profile.density_threshold,
                        batch_rows,
                    )
                    encoder = TensorFrameEncoder(schema, embedder, text_batch_size=256)
                    encode_runs = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        encoded = encoder.encode(table, frame)
                        encode_runs.append(time.perf_counter() - start)
                    full_read_runs = [first_read]
                    for _ in range(args.repeats - 1):
                        _, elapsed = _fetch(
                            reader,
                            schema,
                            table,
                            ids,
                            read_profile.window_batches,
                            read_profile.density_threshold,
                            batch_rows,
                        )
                        full_read_runs.append(elapsed)

                    columns: dict[str, Any] = {}
                    for column, kind in spec.col_to_stype.items():
                        if column == "__const__":
                            continue
                        file = _cache_path(cache_dir, dataset, table, column)
                        file.parent.mkdir(parents=True, exist_ok=True)
                        value = _compact(_column_value(encoded, column, kind))
                        start = time.perf_counter()
                        torch.save(value, file)
                        write_s = time.perf_counter() - start
                        read_runs = []
                        for repeat in range(args.repeats):
                            start = time.perf_counter()
                            loaded = _load(file, len(frame))
                            read_runs.append(time.perf_counter() - start)
                            if repeat == 0 and not _same(value, loaded):
                                raise AssertionError(
                                    f"Cache roundtrip changed {dataset}/{table}/{column}"
                                )
                        process_s = previous[table]["columns"][column]["median_total_s"]
                        disk_read_s = _median(read_runs)
                        saving_s = process_s - disk_read_s
                        cache = saving_s > 0 and (args.epochs - 1) * saving_s > write_s
                        columns[column] = {
                            "type": kind.value,
                            "rows": len(frame),
                            "process_s_from_phase2": process_s,
                            "disk_write_s": write_s,
                            "disk_read_s": disk_read_s,
                            "cache_bytes": file.stat().st_size,
                            "cache_selected": cache,
                            "break_even_epochs_after_first": (
                                write_s / saving_s if saving_s > 0 else None
                            ),
                        }

                    uncached = {
                        name
                        for name, item in columns.items()
                        if not item["cache_selected"]
                    }
                    reduced_schema = (
                        _only_columns(schema, table, uncached) if uncached else None
                    )
                    reduced_encoder = (
                        TensorFrameEncoder(
                            reduced_schema, embedder, text_batch_size=256
                        )
                        if reduced_schema is not None
                        else None
                    )
                    selective_runs = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        uncached_frame = None
                        if reduced_schema is not None:
                            raw, _ = _fetch(
                                reader,
                                reduced_schema,
                                table,
                                ids,
                                read_profile.window_batches,
                                read_profile.density_threshold,
                                batch_rows,
                            )
                            assert reduced_encoder is not None
                            uncached_frame = reduced_encoder.encode(table, raw)
                        _restore(
                            table,
                            columns,
                            cache_dir,
                            dataset,
                            len(frame),
                            uncached_frame,
                        )
                        selective_runs.append(time.perf_counter() - start)

                    full_cache_runs = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        _restore(table, columns, cache_dir, dataset, len(frame))
                        full_cache_runs.append(time.perf_counter() - start)

                    batch_sweep: dict[str, float] = {}
                    for size in args.batch_rows:
                        start = time.perf_counter()
                        for offset in range(0, len(frame), size):
                            encoder.encode(table, frame.iloc[offset : offset + size])
                        batch_sweep[str(size)] = time.perf_counter() - start

                    result["tables"][table] = {
                        "rows": len(frame),
                        "sample_ids_sha256": hashlib.sha256(
                            json.dumps(ids).encode()
                        ).hexdigest(),
                        "columns": columns,
                        "first_read_s": first_read,
                        "median_full_read_s": _median(full_read_runs),
                        "median_full_encode_s": _median(encode_runs),
                        "median_selective_warm_s": _median(selective_runs),
                        "median_full_cache_warm_s": _median(full_cache_runs),
                        "batch_encode_s": batch_sweep,
                    }
                    print(
                        f"[{dataset}] {table}: {len(frame)} rows, {len(columns)} columns",
                        flush=True,
                    )
            finally:
                connection.close()

            tables = result["tables"].values()
            direct = sum(
                t["median_full_read_s"] + t["median_full_encode_s"] for t in tables
            )
            first_selective = direct + sum(
                col["disk_write_s"]
                for t in tables
                for col in t["columns"].values()
                if col["cache_selected"]
            )
            first_all = direct + sum(
                col["disk_write_s"] for t in tables for col in t["columns"].values()
            )
            warm_selective = sum(t["median_selective_warm_s"] for t in tables)
            warm_all = sum(t["median_full_cache_warm_s"] for t in tables)
            result["summary"] = {
                "tables": len(result["tables"]),
                "columns": sum(len(t["columns"]) for t in tables),
                "selected_columns": sum(
                    col["cache_selected"]
                    for t in tables
                    for col in t["columns"].values()
                ),
                "selected_by_type": {
                    kind: sum(
                        col["cache_selected"] and col["type"] == kind
                        for t in tables
                        for col in t["columns"].values()
                    )
                    for kind in sorted(
                        {col["type"] for t in tables for col in t["columns"].values()}
                    )
                },
                "selected_cache_bytes": sum(
                    col["cache_bytes"]
                    for t in tables
                    for col in t["columns"].values()
                    if col["cache_selected"]
                ),
                "all_cache_bytes": sum(
                    col["cache_bytes"] for t in tables for col in t["columns"].values()
                ),
                "direct_epoch_s": direct,
                "selective_first_epoch_s": first_selective,
                "selective_warm_epoch_s": warm_selective,
                "all_first_epoch_s": first_all,
                "all_warm_epoch_s": warm_all,
                "direct_total_s": args.epochs * direct,
                "selective_total_s": first_selective
                + (args.epochs - 1) * warm_selective,
                "all_total_s": first_all + (args.epochs - 1) * warm_all,
                "batch_encode_s": {
                    str(size): sum(t["batch_encode_s"][str(size)] for t in tables)
                    for size in args.batch_rows
                },
                "sampled_rows_total": sum(t["rows"] for t in tables),
            }
            output["datasets"][dataset] = result
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(output, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"[{dataset}] done", flush=True)
    finally:
        embedder.close()
    output["status"] = "complete"
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
