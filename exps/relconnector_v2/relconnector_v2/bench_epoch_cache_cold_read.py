"""Stress-test encoded-column reads after asking Linux to evict file pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch

from relconnector_v2.bench_epoch_feature_cache import (
    _cache_path,
    _fetch,
    _only_columns,
    _restore,
)
from relconnector_v2.connector import PandasDatabaseReader
from relconnector_v2.feature_read_profiles import tuned_feature_read_profile
from relconnector_v2.features.schema import TensorFrameEncoder
from relconnector_v2.features.text import GloveTextEmbedder
from relconnector_v2.profile_feature_processing import (
    _sample_ids,
    _schema_from_existing_stats,
)


def _evict(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selective", action="store_true")
    parser.add_argument("--database-dir", type=Path, default=Path("exps/data/relbench"))
    parser.add_argument(
        "--schema-cache-root",
        type=Path,
        default=Path(
            "exps/benchmarks/tuned-component-ablation-all10-100x5-20261007-v1"
        ),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = json.loads(args.input.read_text())
    cache_dir = args.input.parent / "id-feature-cache-blocks"
    output: dict[str, Any] = {
        "status": "running",
        "method": "fsync then POSIX_FADV_DONTNEED per file before one read; eviction is a hint",
        "datasets": {},
    }
    embedder = (
        GloveTextEmbedder(cache_bytes=0, execution="direct") if args.selective else None
    )
    if embedder is not None:
        embedder(["warmup"])
    for dataset, result in source["datasets"].items():
        reader = (
            PandasDatabaseReader(args.database_dir / f"{dataset}.sqlite")
            if args.selective
            else None
        )
        schema = (
            _schema_from_existing_stats(
                reader, dataset, args.schema_cache_root, args.database_dir
            )[0]
            if reader is not None
            else None
        )
        connection = (
            sqlite3.connect(
                f"file:{(args.database_dir / f'{dataset}.sqlite').resolve()}?mode=ro",
                uri=True,
            )
            if reader is not None
            else None
        )
        read_profile = tuned_feature_read_profile(dataset) if reader else None
        tables = {}
        for table, item in result["tables"].items():
            paths = {
                name: _cache_path(cache_dir, dataset, table, name)
                for name in item["columns"]
            }
            for path in paths.values():
                _evict(path)
            start = time.perf_counter()
            _restore(table, item["columns"], cache_dir, dataset, item["rows"])
            elapsed = time.perf_counter() - start
            tables[table] = {"cold_full_restore_s": elapsed}
            if (
                args.selective
                and reader is not None
                and schema is not None
                and connection is not None
                and read_profile is not None
                and embedder is not None
            ):
                uncached_names = {
                    name
                    for name, info in item["columns"].items()
                    if not info["cache_selected"]
                }
                ids = _sample_ids(
                    connection,
                    table,
                    reader.schemas()[table].primary_key,
                    source["design"]["rows_per_table"],
                )
                if (
                    hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    != item["sample_ids_sha256"]
                ):
                    raise AssertionError(f"Sample IDs changed: {dataset}/{table}")
                for name, info in item["columns"].items():
                    if info["cache_selected"]:
                        _evict(paths[name])
                start = time.perf_counter()
                uncached_frame = None
                if uncached_names:
                    reduced_schema = _only_columns(schema, table, uncached_names)
                    raw, _ = _fetch(
                        reader,
                        reduced_schema,
                        table,
                        ids,
                        read_profile.window_batches,
                        read_profile.density_threshold,
                        max(1, (source["design"]["rows_per_table"] + 31) // 32),
                    )
                    uncached_frame = TensorFrameEncoder(
                        reduced_schema, embedder, text_batch_size=256
                    ).encode(table, raw)
                _restore(
                    table,
                    item["columns"],
                    cache_dir,
                    dataset,
                    item["rows"],
                    uncached_frame,
                )
                tables[table]["cold_selective_restore_s"] = time.perf_counter() - start
        if connection is not None:
            connection.close()
        total = sum(item["cold_full_restore_s"] for item in tables.values())
        output["datasets"][dataset] = {
            "cold_full_restore_s": total,
            "warm_full_restore_s": result["summary"]["all_warm_epoch_s"],
            "cold_selective_restore_s": (
                sum(item["cold_selective_restore_s"] for item in tables.values())
                if args.selective
                else None
            ),
            "warm_selective_restore_s": result["summary"]["selective_warm_epoch_s"],
            "tables": tables,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"[{dataset}] cold full restore {total:.3f}s", flush=True)
    output["status"] = "complete"
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if embedder is not None:
        embedder.close()


if __name__ == "__main__":
    main()
