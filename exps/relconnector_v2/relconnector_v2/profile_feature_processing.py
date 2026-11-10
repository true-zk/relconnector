"""Profile CPU feature conversion by column with frozen SQL read settings.

The experiment samples real rows from every feature-bearing table. SQL reads
use the phase-one profile, but read time is reported separately from conversion.
Text embeddings use the production GloVe CPU path with reuse caches disabled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch_frame import stype
from torch_frame.data.stats import StatType

from relconnector_v2.artifacts import decode_json_value
from relconnector_v2.connector import PandasDatabaseReader
from relconnector_v2.connector.base import quote_identifier
from relconnector_v2.feature_read_profiles import tuned_feature_read_profile
from relconnector_v2.features.schema import (
    TableFeatureSchema,
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
    _fingerprint,
)
from relconnector_v2.features.sql import SqlFeatureFetcher
from relconnector_v2.features.text import GloveTextEmbedder, normalize_text
from relconnector_v2.observability import OperationMetrics

TEXT_STAGES = (
    "prepare_text",
    "text_normalize",
    "text_tokenize",
    "text_lookup",
    "text_pool",
    "text_model",
    "text_expand",
)


def _schema_from_existing_stats(
    reader: PandasDatabaseReader,
    dataset: str,
    cache_root: Path,
    database_dir: Path,
) -> tuple[TensorFrameFeatureSchema, dict[str, str | int]]:
    proposal = json.loads((database_dir / f"{dataset}.stypes.json").read_text())[
        "tables"
    ]
    expected = {
        (table, column) for table, kinds in proposal.items() for column in kinds
    }
    source_dir = cache_root / dataset / "cache" / "schema"
    best: tuple[int, Path, dict[str, Any]] | None = None
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text()).get("payload", {})
        cached = payload.get("tables", {})
        present = {
            (table, column)
            for table, entry in cached.items()
            for column in entry.get("stypes", {})
        }
        score = len(present & expected)
        if score and (best is None or score > best[0]):
            best = score, path, payload
    if best is None:
        raise FileNotFoundError(f"No compatible feature statistics in {source_dir}")
    _, source, encoded_payload = best
    payload = decode_json_value(encoded_payload)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid cached feature statistics in {source}")
    specs: dict[str, TableFeatureSchema] = {}
    for table, schema in reader.schemas().items():
        if schema.kind != "data":
            continue
        entry = payload["tables"].get(table)
        if entry is None:
            raise ValueError(f"No cached feature statistics for {dataset}/{table}")
        kinds = {column: stype(value) for column, value in entry["stypes"].items()}
        columns = tuple(column for column in schema.columns if column.name in kinds)
        if not columns and "__const__" not in kinds:
            raise ValueError(f"No benchmarkable feature columns in {dataset}/{table}")
        stats = {
            column: {StatType(kind): value for kind, value in values.items()}
            for column, values in entry["stats"].items()
        }
        specs[table] = TableFeatureSchema(columns, kinds, stats)
    schema = TensorFrameFeatureSchema(specs, _fingerprint(specs))
    metadata: dict[str, str | int] = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_columns": sum(len(spec.columns) for spec in specs.values()),
    }
    return schema, metadata


def _sample_ids(
    connection: sqlite3.Connection,
    table: str,
    primary_key: str | None,
    count: int,
) -> list[int]:
    quoted_table = quote_identifier(table)
    key = quote_identifier(primary_key) if primary_key else "rowid"
    low, high = connection.execute(
        f"SELECT MIN({key}), MAX({key}) FROM {quoted_table}"
    ).fetchone()
    if low is None or high is None:
        return []
    anchors = 4
    chunk = max(1, (count + anchors - 1) // anchors)
    ids: list[int] = []
    for part in range(anchors):
        start = int(low) + (int(high) - int(low)) * part // anchors
        rows = connection.execute(
            f"SELECT {key} FROM {quoted_table} WHERE {key} >= ? ORDER BY {key} LIMIT ?",
            (start, chunk),
        ).fetchall()
        ids.extend(int(row[0]) - (0 if primary_key else 1) for row in rows)
    return list(dict.fromkeys(ids))[:count]


def _timers(metrics: OperationMetrics) -> dict[tuple[str, str], float]:
    return {
        (str(item["operation"]), str(item["column"])): float(item["total_s"])
        for item in metrics.report()["timers"]
    }


def _profile_table(
    reader: PandasDatabaseReader,
    connection: sqlite3.Connection,
    table: str,
    schema: TensorFrameFeatureSchema,
    embedder: GloveTextEmbedder,
    *,
    window_batches: int,
    density_threshold: float,
    rows_per_table: int,
    repeats: int,
) -> dict[str, Any]:
    table_schema = reader.schemas()[table]
    spec = schema.tables[table]
    ids = _sample_ids(connection, table, table_schema.primary_key, rows_per_table)
    if not ids:
        return {"status": "empty", "columns": {}}
    fetcher = SqlFeatureFetcher(
        reader,
        feature_columns=schema.feature_columns,
        block_size=4096,
        block_density_threshold=density_threshold,
        cache_bytes=0,
    )
    frames = []
    read_started = time.perf_counter()
    rows_per_batch = max(1, (rows_per_table + 31) // 32)
    rows_per_window = window_batches * rows_per_batch
    for offset in range(0, len(ids), rows_per_window):
        frame, _ = fetcher._fetch_table(
            table,
            torch.tensor(ids[offset : offset + rows_per_window], dtype=torch.long),
        )
        frames.append(frame)
    import pandas as pd

    frame = pd.concat(frames, ignore_index=True)
    read_s = time.perf_counter() - read_started
    columns: dict[str, Any] = {}
    table_runs: list[dict[str, float]] = []
    for name, kind in spec.col_to_stype.items():
        if name == "__const__":
            continue
        item: dict[str, Any] = {"type": kind.value, "rows": len(frame)}
        if kind == stype.text_embedded:
            texts = [normalize_text(value) for value in frame[name]]
            item.update(
                total_chars=sum(map(len, texts)),
                max_chars=max(map(len, texts), default=0),
                unique_texts=len(set(texts)),
                empty_texts=sum(not value for value in texts),
            )
        columns[name] = item
    for _ in range(repeats):
        metrics = OperationMetrics(True)
        embedder.metrics = metrics
        encoder = TensorFrameEncoder(
            schema,
            embedder,
            text_batch_size=256,
            metrics=metrics,
        )
        started = time.perf_counter()
        encoded = encoder.encode(table, frame)
        elapsed = time.perf_counter() - started
        if len(encoded) != len(frame):
            raise AssertionError(f"Feature row mismatch: {table}")
        times = _timers(metrics)
        for name, item in columns.items():
            kind = item["type"]
            key = (f"map_{kind}", name)
            if key not in times:
                raise AssertionError(f"Missing column timer: {table}/{name}")
            stages = {stage: times.get((stage, name), 0.0) for stage in TEXT_STAGES}
            item.setdefault("runs", []).append(
                {
                    "map_s": times[key],
                    "text_prepare_s": stages["prepare_text"],
                    "text_stages_s": stages if kind == "text_embedded" else {},
                }
            )
        del encoded, encoder
        table_runs.append(
            {
                "encode_s": elapsed,
                "frame_prepare_s": times.get(("frame_prepare", ""), 0.0),
            }
        )
    for item in columns.values():
        runs = item["runs"]
        item["median_map_s"] = statistics.median(run["map_s"] for run in runs)
        item["median_prepare_s"] = statistics.median(
            run["text_prepare_s"] for run in runs
        )
        item["median_total_s"] = item["median_map_s"] + item["median_prepare_s"]
        if item["type"] == "text_embedded":
            item["median_stages_s"] = {
                stage: statistics.median(run["text_stages_s"][stage] for run in runs)
                for stage in TEXT_STAGES
            }
    return {
        "status": "complete",
        "rows": len(frame),
        "rows_per_virtual_batch": rows_per_batch,
        "read_s": read_s,
        "median_encode_s": statistics.median(run["encode_s"] for run in table_runs),
        "table_runs": table_runs,
        "columns": columns,
    }


def _summary(tables: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"columns": 0, "sampled_values": 0, "median_column_s_sum": 0.0}
    )
    text_columns = []
    for table, item in tables.items():
        for column, detail in item["columns"].items():
            kind = detail["type"]
            group = groups[kind]
            group["columns"] += 1
            group["sampled_values"] += detail["rows"]
            group["median_column_s_sum"] += detail["median_total_s"]
            if kind == "text_embedded":
                text_columns.append(
                    {
                        "table": table,
                        "column": column,
                        "median_s": detail["median_total_s"],
                        "rows": detail["rows"],
                        "total_chars": detail["total_chars"],
                    }
                )
    for group in groups.values():
        group["us_per_value"] = (
            1e6 * group["median_column_s_sum"] / group["sampled_values"]
        )
    text_columns.sort(key=lambda item: item["median_s"], reverse=True)
    return {
        "types": dict(groups),
        "top_text_columns": text_columns[:10],
        "median_table_encode_s_sum": sum(
            item["median_encode_s"] for item in tables.values()
        ),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_columns_csv(path: Path, payload: dict[str, Any]) -> None:
    columns = [
        "dataset",
        "table",
        "column",
        "type",
        "rows",
        "median_total_ms",
        "median_map_ms",
        "median_prepare_ms",
        "total_chars",
        "unique_texts",
        *[f"{stage}_ms" for stage in TEXT_STAGES],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for dataset, result in payload["datasets"].items():
            for table, table_result in result["tables"].items():
                for column, item in table_result["columns"].items():
                    row = {
                        "dataset": dataset,
                        "table": table,
                        "column": column,
                        "type": item["type"],
                        "rows": item["rows"],
                        "median_total_ms": item["median_total_s"] * 1000,
                        "median_map_ms": item["median_map_s"] * 1000,
                        "median_prepare_ms": item["median_prepare_s"] * 1000,
                        "total_chars": item.get("total_chars", ""),
                        "unique_texts": item.get("unique_texts", ""),
                    }
                    row.update(
                        {
                            f"{stage}_ms": value * 1000
                            for stage, value in item.get("median_stages_s", {}).items()
                        }
                    )
                    writer.writerow(row)


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-table", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    if args.rows_per_table <= 0 or args.repeats <= 0:
        parser.error("rows-per-table and repeats must be positive")
    torch.set_num_threads(1)
    paths = sorted(args.database_dir.glob("rel-*.sqlite"))
    if args.datasets:
        requested = set(args.datasets)
        paths = [path for path in paths if path.stem in requested]
        if {path.stem for path in paths} != requested:
            parser.error("A requested dataset is missing")
    output: dict[str, Any] = {
        "status": "running",
        "design": {
            "rows_per_table": args.rows_per_table,
            "repeats": args.repeats,
            "model": GloveTextEmbedder.model_name,
            "text_execution": "direct",
            "text_cache": "off",
            "encoded_cache": "off",
            "torch_threads": 1,
            "cpu_only": True,
            "sampling": "four ID-range anchors per feature-bearing table",
        },
        "datasets": {},
    }
    embedder = GloveTextEmbedder(cache_bytes=0, execution="direct")
    embedder(["warmup"])
    try:
        for path in paths:
            dataset = path.stem
            profile = tuned_feature_read_profile(dataset)
            reader = PandasDatabaseReader(path)
            schema, source = _schema_from_existing_stats(
                reader, dataset, args.schema_cache_root, args.database_dir
            )
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            tables: dict[str, Any] = {}
            try:
                for table, spec in schema.tables.items():
                    if not spec.columns:
                        continue
                    print(
                        f"[{dataset}] {table}: {len(spec.columns)} columns", flush=True
                    )
                    tables[table] = _profile_table(
                        reader,
                        connection,
                        table,
                        schema,
                        embedder,
                        window_batches=profile.window_batches,
                        density_threshold=profile.density_threshold,
                        rows_per_table=args.rows_per_table,
                        repeats=args.repeats,
                    )
            finally:
                connection.close()
            output["datasets"][dataset] = {
                "read_profile": {
                    "window_batches": profile.window_batches,
                    "density_threshold": profile.density_threshold,
                },
                "schema_source": source,
                "tables": tables,
                "summary": _summary(tables),
            }
            _write(args.output, output)
            types = output["datasets"][dataset]["summary"]["types"]
            print(
                f"[{dataset}] done; text {types.get('text_embedded', {}).get('median_column_s_sum', 0):.3f}s",
                flush=True,
            )
    finally:
        embedder.close()
    output["status"] = "complete"
    _write(args.output, output)
    _write_columns_csv(args.output.with_suffix(".columns.csv"), output)


if __name__ == "__main__":
    main()
