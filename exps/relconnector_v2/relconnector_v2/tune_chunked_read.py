"""CPU-only grid search for fixed SQL feature windows and block thresholds.

This uses controlled request streams over real database rows.  Each candidate
reads the same batches and returns the same ordered feature values.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
from pathlib import Path
from typing import Any

import torch

from relconnector_v2.connector import PandasDatabaseReader

from .bench_chunked_read import _check_equal, _make_windows, _run_policy, _tables


def _batches(low: int, high: int, block_size: int, seed: int) -> list[torch.Tensor]:
    """Thirty-two fixed batches spanning four distinct density regimes."""
    groups: list[list[torch.Tensor]] = []
    for offset, scenario in enumerate(("sparse", "mixed", "dense")):
        groups.extend(
            _make_windows(
                low,
                high,
                scenario=scenario,
                seed=seed + 1000 * offset,
                windows=2,
                batches=4,
                batch_size=512,
                block_size=block_size,
            )
        )
    groups.extend(
        _make_windows(
            low,
            high,
            scenario="saturated",
            seed=seed + 3000,
            windows=1,
            batches=8,
            batch_size=512,
            block_size=block_size,
        )
    )
    return [batch for group in groups for batch in group]


def _partition(batches: list[torch.Tensor], size: int) -> list[list[torch.Tensor]]:
    return [batches[i : i + size] for i in range(0, len(batches), size)]


def _name(window: int, threshold: float) -> str:
    return f"w{window}-t{threshold:g}"


def _benchmark_table(
    reader: PandasDatabaseReader,
    table: str,
    batches: list[torch.Tensor],
    candidates: list[tuple[int, float]],
    *,
    block_size: int,
    repeats: int,
    order_seed: int,
) -> dict[str, dict[str, Any]]:
    baseline_windows = _partition(batches, 1)
    _, expected = _run_policy(reader, table, baseline_windows, "sparse", block_size)
    partitioned = {window: _partition(batches, window) for window, _ in candidates}
    observations: dict[str, list[dict[str, float | int]]] = {
        _name(*candidate): [] for candidate in candidates
    }
    for repeat in range(repeats + 1):
        order = list(candidates)
        random.Random(order_seed + repeat).shuffle(order)
        for window, threshold in order:
            metrics, actual = _run_policy(
                reader,
                table,
                partitioned[window],
                "hybrid",
                block_size,
                threshold,
            )
            if repeat == 0:
                _check_equal(
                    [[frame for group in expected for frame in group]],
                    [[frame for group in actual for frame in group]],
                )
            else:
                observations[_name(window, threshold)].append(metrics)
            del actual
    return {
        _name(window, threshold): {
            "window_batches": window,
            "density_threshold": threshold,
            "median_elapsed_s": statistics.median(
                float(run["elapsed_s"])
                for run in observations[_name(window, threshold)]
            ),
            "median_sql_s": statistics.median(
                float(run["sql_s"]) for run in observations[_name(window, threshold)]
            ),
            "queries": observations[_name(window, threshold)][0]["queries"],
            "returned_rows": observations[_name(window, threshold)][0]["returned_rows"],
            "runs": observations[_name(window, threshold)],
        }
        for window, threshold in candidates
    }


def _score(tables: dict[str, dict[str, Any]], field: str) -> dict[str, float]:
    keys = next(iter(tables.values()))[field]
    return {
        key: sum(
            float(table[field][key]["median_elapsed_s"]) for table in tables.values()
        )
        for key in keys
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-dir", type=Path, default=Path("exps/data/relbench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.25, 0.4, 0.6, 0.8, 1.0)
    )
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=43)
    args = parser.parse_args()
    if (
        args.block_size <= 0
        or args.repeats <= 0
        or args.validation_repeats <= 0
        or any(window <= 0 or 32 % window for window in args.windows)
        or any(not 0 < threshold <= 1 for threshold in args.thresholds)
    ):
        parser.error("Invalid sizes or thresholds; windows must divide 32")
    torch.set_num_threads(1)
    candidates = list(
        dict.fromkeys((w, t) for w in args.windows for t in args.thresholds)
    )
    default = (4, 0.25)
    if default not in candidates:
        parser.error(
            "The grid must include the current default: window 4, threshold 0.25"
        )
    paths = sorted(args.database_dir.glob("rel-*.sqlite"))
    if args.datasets is not None:
        selected = set(args.datasets)
        paths = [path for path in paths if path.stem in selected]
        if {path.stem for path in paths} != selected:
            parser.error("A requested dataset is missing")
    output: dict[str, Any] = {
        "status": "running",
        "design": {
            "batch_count": 32,
            "batch_size": 512,
            "block_size": args.block_size,
            "scenario_batches": {
                "sparse": 8,
                "mixed": 8,
                "dense": 8,
                "saturated": 8,
            },
            "windows": args.windows,
            "thresholds": args.thresholds,
            "repeats_after_warmup": args.repeats,
            "validation_repeats_after_warmup": args.validation_repeats,
            "tuning_seed": args.seed,
            "validation_seed": args.validation_seed,
            "cache_bytes": 0,
            "note": "Controlled CPU read-path tuning; synthetic requests over real SQLite rows.",
        },
        "datasets": {},
    }
    for path in paths:
        print(f"[{path.stem}] selecting tables", flush=True)
        reader = PandasDatabaseReader(path)
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            selected_tables = _tables(reader, connection)
        finally:
            connection.close()
        tables: dict[str, dict[str, Any]] = {}
        for schema, low, high, width in selected_tables:
            print(f"[{path.stem}] tuning {schema.name} ({width} columns)", flush=True)
            batches = _batches(low, high, args.block_size, args.seed)
            grid = _benchmark_table(
                reader,
                schema.name,
                batches,
                candidates,
                block_size=args.block_size,
                repeats=args.repeats,
                order_seed=args.seed,
            )
            tables[schema.name] = {
                "id_min": low,
                "id_max": high,
                "feature_columns": width,
                "grid": grid,
            }
            print(f"[{path.stem}] finished {schema.name}", flush=True)
        training_scores = _score(tables, "grid")
        finalists = [
            next(candidate for candidate in candidates if _name(*candidate) == key)
            for key in sorted(training_scores, key=lambda key: training_scores[key])[:3]
        ]
        for window in args.windows:
            by_window = [
                candidate for candidate in candidates if candidate[0] == window
            ]
            best_for_window = min(
                by_window,
                key=lambda candidate: training_scores[_name(*candidate)],
            )
            if best_for_window not in finalists:
                finalists.append(best_for_window)
        if default not in finalists:
            finalists.append(default)
        for schema, low, high, _ in selected_tables:
            batches = _batches(low, high, args.block_size, args.validation_seed)
            tables[schema.name]["validation"] = _benchmark_table(
                reader,
                schema.name,
                batches,
                finalists,
                block_size=args.block_size,
                repeats=args.validation_repeats,
                order_seed=args.validation_seed,
            )
        validation_scores = _score(tables, "validation")
        best_key = min(validation_scores, key=lambda key: validation_scores[key])
        default_key = _name(*default)
        output["datasets"][path.stem] = {
            "database_bytes": path.stat().st_size,
            "database_mtime_ns": path.stat().st_mtime_ns,
            "tables": tables,
            "training_scores_s": training_scores,
            "validation_scores_s": validation_scores,
            "best": best_key,
            "validated_speedup_vs_default": (
                validation_scores[default_key] / validation_scores[best_key]
            ),
        }
        _write(args.output, output)
        print(
            f"[{path.stem}] best {best_key}, validation "
            f"{output['datasets'][path.stem]['validated_speedup_vs_default']:.2f}x",
            flush=True,
        )
    output["status"] = "complete"
    _write(args.output, output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
