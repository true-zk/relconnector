"""CPU-only, paired microbenchmark for the SQL feature read policy.

The requests are deterministic synthetic windows over real SQLite tables.  This
isolates SQL access and pandas decoding; it does not estimate training speed.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import torch

from relconnector_v2.connector import PandasDatabaseReader, TableSchema
from relconnector_v2.connector.base import quote_identifier
from relconnector_v2.features.sql import SqlFeatureFetcher
from relconnector_v2.observability import OperationMetrics


def _limits(connection: sqlite3.Connection, schema: TableSchema) -> tuple[int, int]:
    expression = (
        quote_identifier(schema.primary_key)
        if schema.primary_key is not None
        else "rowid - 1"
    )
    table = quote_identifier(schema.name)
    # Separate aggregates let SQLite use each end of the PK index.
    low = connection.execute(f"SELECT MIN({expression}) FROM {table}").fetchone()[0]
    high = connection.execute(f"SELECT MAX({expression}) FROM {table}").fetchone()[0]
    return int(low), int(high)


def _tables(
    reader: PandasDatabaseReader, connection: sqlite3.Connection
) -> list[tuple[TableSchema, int, int, int]]:
    fetcher = SqlFeatureFetcher(reader)
    candidates = []
    for schema in reader.schemas().values():
        if schema.kind != "data":
            continue
        width = len(fetcher.feature_columns(schema))
        if not width:
            continue
        low, high = _limits(connection, schema)
        if high - low + 1 >= 8192:
            candidates.append((schema, low, high, width))
    if not candidates:
        raise ValueError("No feature-bearing table with at least 8192 IDs")
    largest = max(candidates, key=lambda item: (item[2] - item[1], item[3]))
    widest = max(candidates, key=lambda item: (item[3], item[2] - item[1]))
    return [largest] if widest[0].name == largest[0].name else [largest, widest]


def _make_windows(
    low: int,
    high: int,
    *,
    scenario: str,
    seed: int,
    windows: int,
    batches: int,
    batch_size: int,
    block_size: int,
) -> list[list[torch.Tensor]]:
    rng = random.Random(seed)
    result = []
    first_block = (low + block_size - 1) // block_size
    last_block = (high - block_size + 1) // block_size
    if last_block < first_block:
        raise ValueError("Table does not contain a complete block")
    for _ in range(windows):
        block = rng.randint(first_block, last_block)
        dense_start = block * block_size
        dense_pool = range(dense_start, dense_start + block_size // 2)
        if scenario == "saturated":
            count = batches * batch_size
            unique_count = min(block_size, count, int(count * 0.875))
            unique_ids = rng.sample(
                range(dense_start, dense_start + block_size), unique_count
            )
            ids = unique_ids + rng.choices(unique_ids, k=count - unique_count)
            rng.shuffle(ids)
            result.append(
                [
                    torch.tensor(ids[offset : offset + batch_size], dtype=torch.long)
                    for offset in range(0, count, batch_size)
                ]
            )
            continue
        window = []
        for _batch in range(batches):
            ids = []
            for _row in range(batch_size):
                if scenario == "dense" or (scenario == "mixed" and rng.random() < 0.8):
                    ids.append(rng.choice(dense_pool))
                else:
                    ids.append(rng.randint(low, high))
            window.append(torch.tensor(ids, dtype=torch.long))
        result.append(window)
    return result


def _run_policy(
    reader: PandasDatabaseReader,
    table: str,
    windows: list[list[torch.Tensor]],
    policy: str,
    block_size: int,
    density_threshold: float = 0.25,
) -> tuple[dict[str, float | int], list[list[pd.DataFrame]]]:
    metrics = OperationMetrics(True)
    fetcher = SqlFeatureFetcher(
        reader,
        block_size=block_size,
        block_density_threshold=density_threshold,
        cache_bytes=0,
        read_policy=cast(
            Literal["hybrid", "sparse", "dense"],
            "sparse" if policy == "per_batch_sparse" else policy,
        ),
        metrics=metrics,
    )
    returned_rows = 0
    outputs = []
    started = time.perf_counter()
    for window in windows:
        if policy == "per_batch_sparse":
            frames = []
            for ids in window:
                frame, rows = fetcher._fetch_table(table, ids)
                returned_rows += rows
                frames.append(frame)
        else:
            combined = torch.cat(window)
            frame, rows = fetcher._fetch_table(table, combined)
            returned_rows += rows
            frames = []
            offset = 0
            for ids in window:
                frames.append(
                    frame.iloc[offset : offset + len(ids)].reset_index(drop=True)
                )
                offset += len(ids)
        outputs.append(frames)
    elapsed = time.perf_counter() - started
    sql_timers = [
        item
        for item in metrics.report()["timers"]
        if item["operation"] == "sql_execute"
    ]
    return (
        {
            "elapsed_s": elapsed,
            "sql_s": sum(float(item["total_s"]) for item in sql_timers),
            "queries": sum(int(item["count"]) for item in sql_timers),
            "returned_rows": returned_rows,
        },
        outputs,
    )


def _check_equal(
    expected: list[list[pd.DataFrame]], actual: list[list[pd.DataFrame]]
) -> None:
    for expected_window, actual_window in zip(expected, actual, strict=True):
        for expected_batch, actual_batch in zip(
            expected_window, actual_window, strict=True
        ):
            pd.testing.assert_frame_equal(expected_batch, actual_batch)


def _measure(
    reader: PandasDatabaseReader,
    table: str,
    windows: list[list[torch.Tensor]],
    *,
    scenario: str,
    block_size: int,
    repeats: int,
) -> dict[str, Any]:
    policies = ["per_batch_sparse", "sparse", "hybrid"]
    if scenario in {"dense", "saturated"}:
        policies.append("dense")
    observations: dict[str, list[dict[str, float | int]]] = {
        policy: [] for policy in policies
    }
    for repeat in range(repeats + 1):
        outputs = {}
        # Rotate order to reduce systematic advantage from warm OS page cache.
        order = policies[repeat % len(policies) :] + policies[: repeat % len(policies)]
        for policy in order:
            result, frames = _run_policy(reader, table, windows, policy, block_size)
            outputs[policy] = frames
            if repeat:
                observations[policy].append(result)
        if repeat == 0:
            for policy in policies[1:]:
                _check_equal(outputs["per_batch_sparse"], outputs[policy])
    summary = {
        policy: {
            "median_elapsed_s": statistics.median(
                float(row["elapsed_s"]) for row in values
            ),
            "median_sql_s": statistics.median(float(row["sql_s"]) for row in values),
            "queries": values[0]["queries"],
            "returned_rows": values[0]["returned_rows"],
            "runs": values,
        }
        for policy, values in observations.items()
    }
    baseline = float(summary["per_batch_sparse"]["median_elapsed_s"])
    for policy in policies:
        summary[policy]["speedup_vs_per_batch_sparse"] = baseline / float(
            summary[policy]["median_elapsed_s"]
        )
    sparse = float(summary["sparse"]["median_elapsed_s"])
    summary["hybrid"]["speedup_vs_window_sparse"] = sparse / float(
        summary["hybrid"]["median_elapsed_s"]
    )
    requested = sum(len(ids) for window in windows for ids in window)
    unique = sum(len(torch.unique(torch.cat(window))) for window in windows)
    return {
        "scenario": scenario,
        "requested_rows": requested,
        "unique_rows": unique,
        "duplicate_factor": requested / unique,
        "policies": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-dir", type=Path, default=Path("exps/data/relbench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("sparse", "dense", "mixed", "saturated"),
        default=("sparse", "dense", "mixed"),
    )
    args = parser.parse_args()
    if (
        min(args.windows, args.batches, args.batch_size, args.block_size, args.repeats)
        <= 0
    ):
        parser.error("All sizes and repeats must be positive")
    paths = sorted(args.database_dir.glob("rel-*.sqlite"))
    if args.datasets is not None:
        selected = set(args.datasets)
        paths = [path for path in paths if path.stem in selected]
        if {path.stem for path in paths} != selected:
            parser.error("A requested dataset is missing")
    payload: dict[str, Any] = {
        "design": {
            "windows": args.windows,
            "batches_per_window": args.batches,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "density_threshold": 0.25,
            "repeats_after_warmup": args.repeats,
            "seed": args.seed,
            "cache_bytes": 0,
            "reader": "PandasDatabaseReader",
            "note": "Synthetic requests over real SQLite rows; CPU feature reads only.",
        },
        "datasets": {},
    }
    for path in paths:
        print(f"[{path.stem}] selecting tables", flush=True)
        reader = PandasDatabaseReader(path)
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            tables = _tables(reader, connection)
        finally:
            connection.close()
        dataset_result = {
            "database_bytes": path.stat().st_size,
            "database_mtime_ns": path.stat().st_mtime_ns,
            "tables": {},
        }
        for schema, low, high, width in tables:
            print(
                f"[{path.stem}] {schema.name}: IDs {low}..{high}, {width} columns",
                flush=True,
            )
            table_result: dict[str, Any] = {
                "id_min": low,
                "id_max": high,
                "feature_columns": width,
            }
            for scenario in args.scenarios:
                batches = (
                    max(args.batches, 8) if scenario == "saturated" else args.batches
                )
                windows = _make_windows(
                    low,
                    high,
                    scenario=scenario,
                    seed=args.seed,
                    windows=args.windows,
                    batches=batches,
                    batch_size=args.batch_size,
                    block_size=args.block_size,
                )
                table_result[scenario] = _measure(
                    reader,
                    schema.name,
                    windows,
                    scenario=scenario,
                    block_size=args.block_size,
                    repeats=args.repeats,
                )
                table_result[scenario]["batches_per_window"] = batches
                result = table_result[scenario]["policies"]["hybrid"]
                print(
                    f"  {scenario}: hybrid {result['median_elapsed_s']:.3f}s, "
                    f"{result['speedup_vs_window_sparse']:.2f}x vs window IN",
                    flush=True,
                )
            dataset_result["tables"][schema.name] = table_result
        payload["datasets"][path.stem] = dataset_result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
