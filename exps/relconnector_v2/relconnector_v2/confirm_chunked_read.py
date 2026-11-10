"""Compare selected CPU read policies with an explicit exact-ID baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from relconnector_v2.connector import PandasDatabaseReader

from .tune_chunked_read import _batches, _benchmark_table, _write


def _candidate(name: str) -> tuple[int, float]:
    window, threshold = name.split("-t", maxsplit=1)
    return int(window[1:]), float(threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, default=Path("exps/data/relbench"))
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--baseline",
        choices=("default", "no_window_exact"),
        default="no_window_exact",
        help="Comparison policy: current default or one exact-ID IN query per batch.",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    tuning = json.loads(args.tuning.read_text())
    if tuning["status"] != "complete":
        parser.error("tuning output is incomplete")
    if args.seed in {
        tuning["design"]["tuning_seed"],
        tuning["design"]["validation_seed"],
    }:
        parser.error("confirmation seed must be new")
    torch.set_num_threads(1)
    result: dict[str, Any] = {
        "status": "running",
        "tuning_file": str(args.tuning),
        "seed": args.seed,
        "repeats_after_warmup": args.repeats,
        "baseline": args.baseline,
        "baseline_definition": (
            "One exact-ID IN query per batch, with no cross-batch window"
            if args.baseline == "no_window_exact"
            else "Production default window of four batches and threshold 0.25"
        ),
        "datasets": {},
    }
    baseline = (1, 1.0) if args.baseline == "no_window_exact" else (4, 0.25)
    baseline_key = "w1-t1" if args.baseline == "no_window_exact" else "w4-t0.25"
    for dataset, original in tuning["datasets"].items():
        path = args.database_dir / f"{dataset}.sqlite"
        if (
            path.stat().st_size != original["database_bytes"]
            or path.stat().st_mtime_ns != original["database_mtime_ns"]
        ):
            raise ValueError(f"Database changed since tuning: {dataset}")
        reader = PandasDatabaseReader(path)
        selected = original["best"]
        candidates = [baseline, _candidate(selected)]
        tables = {}
        for table, metadata in original["tables"].items():
            batches = _batches(
                metadata["id_min"],
                metadata["id_max"],
                tuning["design"]["block_size"],
                args.seed,
            )
            tables[table] = _benchmark_table(
                reader,
                table,
                batches,
                candidates,
                block_size=tuning["design"]["block_size"],
                repeats=args.repeats,
                order_seed=args.seed,
            )
            if args.baseline == "no_window_exact":
                actual_queries = tables[table][baseline_key]["queries"]
                expected_queries = len(batches)
                if actual_queries != expected_queries:
                    raise AssertionError(
                        f"{dataset}/{table}: expected {expected_queries} exact-ID "
                        f"queries, got {actual_queries}"
                    )
        baseline_s = sum(
            float(table[baseline_key]["median_elapsed_s"]) for table in tables.values()
        )
        selected_s = sum(
            float(table[selected]["median_elapsed_s"]) for table in tables.values()
        )
        result["datasets"][dataset] = {
            "baseline": baseline_key,
            "selected": selected,
            "baseline_s": baseline_s,
            "selected_s": selected_s,
            "speedup": baseline_s / selected_s,
            "tables": tables,
        }
        _write(args.output, result)
        print(
            f"[{dataset}] {selected}: {baseline_s / selected_s:.2f}x "
            f"({baseline_s:.3f}s to {selected_s:.3f}s)",
            flush=True,
        )
    result["status"] = "complete"
    result["median_speedup"] = statistics.median(
        item["speedup"] for item in result["datasets"].values()
    )
    _write(args.output, result)


if __name__ == "__main__":
    main()
