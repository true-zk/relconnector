"""Screen individual v2 feature-supply improvements on fixed replays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .run_cache_iteration import execute, save

CASES = {
    "rel-amazon": (
        "item-churn",
        "benchmarks/cache-roi-amazon-ratebeer-100x5-v1/rel-amazon/replay",
    ),
    "rel-ratebeer": (
        "beer_ratings-total_score",
        "benchmarks/cache-roi-amazon-ratebeer-100x5-v1/rel-ratebeer/replay",
    ),
}

PROFILES = (
    "vanilla_sync_official",
    "v2_sync_official",
    "v2_sync_direct",
    "v2_async_direct_win1",
    "v2_async_direct_win4",
    "v2_process_w1_strict",
    "v2_process_w4_strict",
    "v2_process_w4_ready",
)

IMPROVEMENTS = {
    "vanilla_sync_official": "baseline",
    "v2_sync_official": "v2_feature_path",
    "v2_sync_direct": "direct_glove",
    "v2_async_direct_win1": "thread_pipeline_overlap",
    "v2_async_direct_win4": "four_batch_windows",
    "v2_process_w1_strict": "process_isolation",
    "v2_process_w4_strict": "four_process_workers",
    "v2_process_w4_ready": "ready_order_no_head_of_line_blocking",
}


def _read_result(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f"expected one result record in {path}")
    result = json.loads(lines[0])
    if result.get("status") != "ok":
        raise ValueError(f"run did not report success: {path}")
    return result


def _operation_totals(report: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for timer in report.get("timers", []):
        operation = str(timer["operation"])
        totals[operation] = totals.get(operation, 0.0) + float(timer["total_s"])
    return totals


def _process_operations(pipeline: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for worker in pipeline["workers"]:
        if worker.get("stage") != "assembly":
            continue
        operations = worker.get("diagnostics", {}).get("operations", {})
        for operation, value in _operation_totals(operations).items():
            totals[operation] = totals.get(operation, 0.0) + value
    return totals


def _summarize(
    dataset: str,
    profile: str,
    result: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    process = profile.startswith("v2_process")
    if process:
        pipeline = result["pipeline"]
        training_s = sum(float(epoch["wall_duration_s"]) for epoch in pipeline["epochs"])
        operations = _process_operations(pipeline)
        ready_wait_s = float(pipeline["ready_wait_s"])
        fingerprint = pipeline["replay"]["fingerprint"]
        init_cache = result["initialization_cache"]
        workers = int(profile.split("_w", 1)[1].split("_", 1)[0])
        window_batches = int(result["effective_config"]["feature_window_batches"])
        executor = "process"
        text_execution = result["effective_config"]["text_execution"]
        training_order = result["effective_config"]["process_training_order"]
    else:
        training_s = float(result["training_s"])
        operations = _operation_totals(result["operation_metrics"])
        ready_wait_s = None
        fingerprint = result["fingerprint"]
        init_cache = result["initialization_cache"]
        workers = 0
        window_batches = int(result.get("window_batches", 1))
        executor = str(result["executor"])
        text_execution = str(result.get("text_execution", "official"))
        training_order = "in_order"
    batches = int(result["completed_batches"])
    direct_text_s = sum(
        operations.get(name, 0.0)
        for name in ("text_tokenize", "text_lookup", "text_pool")
    )
    return {
        "dataset": dataset,
        "profile": profile,
        "improvement": IMPROVEMENTS[profile],
        "executor": executor,
        "workers": workers,
        "window_batches": window_batches,
        "text_execution": text_execution,
        "training_order": training_order,
        "completed_batches": batches,
        "fingerprint": fingerprint,
        "training_s": training_s,
        "training_s_per_batch": training_s / batches,
        "ready_wait_s": ready_wait_s,
        "official_text_model_s": operations.get("text_model", 0.0),
        "direct_text_compute_s": direct_text_s,
        "map_text_embedded_s": operations.get("map_text_embedded", 0.0),
        "frame_convert_s": operations.get("frame_convert", 0.0),
        "initialization_cache": init_cache,
        "peak_main_rss_mb": execution["memory"]["peak_main_rss_mb"],
        "peak_tree_pss_mb": execution["memory"]["peak_tree_pss_mb"],
        "elapsed_s": execution["elapsed_s"],
    }


def _effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effects = []
    for dataset in CASES:
        by_profile = {row["profile"]: row for row in rows if row["dataset"] == dataset}
        baseline = by_profile.get(PROFILES[0])
        previous = None
        for profile in PROFILES:
            current = by_profile.get(profile)
            if current is None:
                continue
            effects.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "improvement": IMPROVEMENTS[profile],
                    "training_s_per_batch": current["training_s_per_batch"],
                    "incremental_reduction_pct": (
                        None
                        if previous is None
                        else 100
                        * (
                            1
                            - current["training_s_per_batch"]
                            / previous["training_s_per_batch"]
                        )
                    ),
                    "cumulative_reduction_pct": (
                        None
                        if baseline is None
                        else 100
                        * (
                            1
                            - current["training_s_per_batch"]
                            / baseline["training_s_per_batch"]
                        )
                    ),
                }
            )
            previous = current
    return effects


def _write_reports(output: Path, rows: list[dict[str, Any]]) -> None:
    save(output / "summary.json", rows)
    effects = _effect_rows(rows)
    save(output / "effects.json", effects)
    for name, records in (("summary.csv", rows), ("effects.csv", effects)):
        if not records:
            continue
        flat = [
            {key: value for key, value in record.items() if key != "initialization_cache"}
            for record in records
        ]
        temporary = output / f"{name}.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        temporary.replace(output / name)


def _common(
    root: Path,
    dataset: str,
    task: str,
    replay_dir: Path,
    cache_dir: Path,
    epochs: int,
    max_batches: int,
) -> list[str]:
    return [
        "--dataset",
        dataset,
        "--task",
        task,
        "--database-dir",
        str(root / "data/relbench"),
        "--replay-dir",
        str(replay_dir),
        "--cache-dir",
        str(cache_dir),
        "--epochs",
        str(epochs),
        "--max-batches",
        str(max_batches),
        "--batch-size",
        "512",
        "--num-neighbors",
        "128,128",
        "--channels",
        "128",
        "--seed",
        "42",
        "--device",
        "cuda:0",
        "--torch-num-threads",
        "1",
        "--text-batch-size",
        "256",
    ]


def _command(
    profile: str,
    common: list[str],
    result_path: Path,
) -> list[str]:
    if profile == "vanilla_sync_official":
        return [
            sys.executable,
            "-m",
            "relconnector_v2.fixed_vanilla_benchmark",
            *common,
            "--executor",
            "sync",
            "--output",
            str(result_path),
        ]
    if profile.startswith("v2_process"):
        workers = int(profile.split("_w", 1)[1].split("_", 1)[0])
        order = profile.rsplit("_", 1)[1]
        return [
            sys.executable,
            "-m",
            "relconnector_v2.benchmark",
            *common,
            "--output",
            str(result_path),
            "--cache-strategy",
            "off",
            "--feature-cache-mb",
            "0",
            "--text-cache-mb",
            "0",
            "--encoded-cache-mb",
            "0",
            "--disk-text-cache-mode",
            "off",
            "--text-execution",
            "direct",
            "--window-batches",
            "4",
            "--assembly-processes",
            str(workers),
            "--initial-active-processes",
            str(workers),
            "--no-adaptive-workers",
            "--training-order",
            order,
            "--no-cross-epoch-prefetch",
            "--progress-interval-s",
            "30",
        ]
    executor = "sync" if profile.startswith("v2_sync") else "async"
    text_execution = "official" if profile.endswith("official") else "direct"
    window_batches = 4 if profile.endswith("win4") else 1
    return [
        sys.executable,
        "-m",
        "relconnector_v2.fixed_v2_benchmark",
        *common,
        "--output",
        str(result_path),
        "--executor",
        executor,
        "--text-execution",
        text_execution,
        "--window-batches",
        str(window_batches),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=14400)
    args = parser.parse_args(argv)
    if min(args.epochs, args.max_batches, args.timeout_s) <= 0:
        parser.error("epochs, max batches and timeout must be positive")

    root = Path(
        os.environ.get("RELCONNECTOR_PROJECT_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "relconnector_v2"),
        "RELCONNECTOR_PROJECT_ROOT": str(root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    plan = [f"{dataset}/{profile}" for dataset in args.datasets for profile in PROFILES]
    state_path = output / "status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        rows = json.loads((output / "summary.json").read_text()) if (output / "summary.json").exists() else []
    else:
        state = {
            "pid": os.getpid(),
            "status": "planned",
            "started_unix": time.time(),
            "current": None,
            "plan": plan,
            "completed": [],
            "failures": [],
        }
        rows = []
        save(state_path, state)
        save(
            output / "manifest.json",
            {
                "purpose": "stepwise component ablation; each profile adds one v2 capability",
                "datasets": args.datasets,
                "profiles": list(PROFILES),
                "improvements": IMPROVEMENTS,
                "epochs": args.epochs,
                "max_batches": args.max_batches,
                "batch_size": 512,
                "num_neighbors": [128, 128],
                "channels": 128,
                "device": "cuda:0",
                "application_caches": "off",
                "initialization_cache": "warm",
                "os_page_cache": "not cleared",
                "screening_repeats": 1,
            },
        )

    for dataset in args.datasets:
        task, replay_relative = CASES[dataset]
        replay_dir = root / replay_relative
        cache_dir = (
            root
            / "benchmarks/ready-worker-scaling-amazon-ratebeer-100x5-20261005-v1"
            / dataset
            / "cache"
        )
        replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
        for profile in PROFILES:
            key = f"{dataset}/{profile}"
            run_dir = output / dataset / profile
            result_path = run_dir / (
                "result.jsonl" if profile.startswith("v2_process") else "result.json"
            )
            if key in state["completed"] and result_path.exists():
                continue
            state.update(status="running", current=key, pid=os.getpid())
            save(state_path, state)
            common = _common(
                root,
                dataset,
                task,
                replay_dir,
                cache_dir,
                args.epochs,
                args.max_batches,
            )
            record = execute(
                _command(profile, common, result_path),
                run_dir,
                root,
                environment,
                args.timeout_s,
            )
            try:
                if record["returncode"] or not record["clean_process_exit"]:
                    raise RuntimeError(
                        f"returncode={record['returncode']}, reason={record['reason']}, "
                        f"lingering={record['lingering_processes']}"
                    )
                result = _read_result(result_path)
                row = _summarize(dataset, profile, result, record)
                expected = args.epochs * args.max_batches
                if row["completed_batches"] != expected:
                    raise RuntimeError("incomplete batch coverage")
                if row["fingerprint"] != replay_manifest["fingerprint"]:
                    raise RuntimeError("replay fingerprint mismatch")
                if row["initialization_cache"] != {"schema": True, "topology": True}:
                    raise RuntimeError(
                        f"initialization cache was not warm: {row['initialization_cache']}"
                    )
                rows = [
                    old
                    for old in rows
                    if not (old["dataset"] == dataset and old["profile"] == profile)
                ]
                rows.append(row)
                state["completed"].append(key)
                _write_reports(output, rows)
            except Exception as exc:  # noqa: BLE001 - retain failure and stop matrix.
                state["failures"].append({"case": key, "error": str(exc), **record})
                state.update(status="failed", current=None)
                save(state_path, state)
                return 1
            save(state_path, state)

    state.update(status="completed", current=None, finished_unix=time.time())
    save(state_path, state)
    _write_reports(output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
