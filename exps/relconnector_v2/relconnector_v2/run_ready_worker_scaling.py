"""Run fixed-replay vanilla versus cache-free ready-order worker scaling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
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
WORKERS = (1, 2, 4, 8)


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


def _summarize(
    dataset: str,
    profile: str,
    result: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    if profile == "vanilla_sync":
        training_s = float(result["training_s"])
        operations = _operation_totals(result["operation_metrics"])
        ready_wait_s = None
        fingerprint = result["fingerprint"]
        batches = int(result["completed_batches"])
        init_cache = result["initialization_cache"]
        workers = 0
    else:
        pipeline = result["pipeline"]
        epochs = pipeline["epochs"]
        training_s = sum(float(epoch["wall_duration_s"]) for epoch in epochs)
        operations = {}
        for worker in pipeline["workers"]:
            if worker.get("stage") != "assembly":
                continue
            worker_operations = worker.get("diagnostics", {}).get("operations", {})
            for operation, value in _operation_totals(worker_operations).items():
                operations[operation] = operations.get(operation, 0.0) + value
        ready_wait_s = float(pipeline["ready_wait_s"])
        fingerprint = pipeline["replay"]["fingerprint"]
        batches = int(result["completed_batches"])
        init_cache = result.get("initialization_cache")
        workers = int(profile.rsplit("w", 1)[1])
    return {
        "dataset": dataset,
        "profile": profile,
        "workers": workers,
        "completed_batches": batches,
        "fingerprint": fingerprint,
        "training_s": training_s,
        "training_s_per_batch": training_s / batches,
        "ready_wait_s": ready_wait_s,
        "text_model_s": operations.get("text_model", 0.0),
        "frame_convert_s": operations.get("frame_convert", 0.0),
        "map_text_embedded_s": operations.get("map_text_embedded", 0.0),
        "initialization_cache": init_cache,
        "peak_main_rss_mb": execution["memory"]["peak_main_rss_mb"],
        "peak_tree_pss_mb": execution["memory"]["peak_tree_pss_mb"],
        "elapsed_s": execution["elapsed_s"],
    }


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    save(output / "summary.json", rows)
    flat = [
        {key: value for key, value in row.items() if key != "initialization_cache"}
        for row in rows
    ]
    if flat:
        temporary = output / "summary.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        temporary.replace(output / "summary.csv")


def _seed_initialization_cache(source: Path, target: Path) -> None:
    if target.exists():
        return
    shutil.copytree(source, target, copy_function=os.link)


def _gpu_state() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    source_cache = (
        root / "benchmarks/vanilla-sync-warm-init-100x5-20260930-v1/cache"
    )
    profiles = ["vanilla_sync", *(f"ready_w{worker}" for worker in WORKERS)]
    plan = [f"{dataset}/{profile}" for dataset in args.datasets for profile in profiles]
    state_path = output / "status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        rows = json.loads((output / "summary.json").read_text()) if (output / "summary.json").exists() else []
    else:
        state = {
            "pid": os.getpid(),
            "status": "planned",
            "started_unix": time.time(),
            "gpu_at_start": _gpu_state(),
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
                "datasets": args.datasets,
                "profiles": profiles,
                "epochs": args.epochs,
                "max_batches": args.max_batches,
                "batch_size": 512,
                "num_neighbors": [128, 128],
                "channels": 128,
                "device": "cuda:0",
                "training_order": "ready",
                "cross_epoch_prefetch": False,
                "application_caches": "off",
                "initialization_cache": "warm",
                "os_page_cache": "not cleared",
                "worker_adaptation": False,
                "replays": {dataset: replay for dataset, (_, replay) in CASES.items()},
            },
        )

    for dataset in args.datasets:
        task, replay_relative = CASES[dataset]
        replay_dir = root / replay_relative
        dataset_dir = output / dataset
        cache_dir = dataset_dir / "cache"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _seed_initialization_cache(source_cache, cache_dir)

        prewarm_key = f"{dataset}/prewarm_v2"
        prewarm_marker = dataset_dir / "prewarm-v2.ok"
        if not prewarm_marker.exists():
            state.update(status="running", current=prewarm_key, pid=os.getpid())
            save(state_path, state)
            command = [
                sys.executable,
                "-m",
                "relconnector_v2.benchmark",
                "--dataset",
                dataset,
                "--task",
                task,
                "--database-dir",
                str(root / "data/relbench"),
                "--cache-dir",
                str(cache_dir),
                "--device",
                "cpu",
                "--max-batches",
                str(args.max_batches),
                "--prepare-only",
            ]
            record = execute(command, dataset_dir / "prewarm-v2", root, environment, args.timeout_s)
            if record["returncode"] or not record["clean_process_exit"]:
                state["failures"].append({"case": prewarm_key, **record})
                state.update(status="failed", current=None)
                save(state_path, state)
                return 1
            prewarm_marker.write_text("ok\n")

        for profile in profiles:
            key = f"{dataset}/{profile}"
            run_dir = dataset_dir / profile
            result_path = run_dir / ("result.json" if profile == "vanilla_sync" else "result.jsonl")
            if key in state["completed"] and result_path.exists():
                continue
            state.update(status="running", current=key, pid=os.getpid())
            save(state_path, state)
            common = [
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
                str(args.epochs),
                "--max-batches",
                str(args.max_batches),
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
            if profile == "vanilla_sync":
                command = [
                    sys.executable,
                    "-m",
                    "relconnector_v2.fixed_vanilla_benchmark",
                    *common,
                    "--executor",
                    "sync",
                    "--output",
                    str(result_path),
                ]
            else:
                workers = int(profile.rsplit("w", 1)[1])
                command = [
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
                    "--assembly-processes",
                    str(workers),
                    "--initial-active-processes",
                    str(workers),
                    "--no-adaptive-workers",
                    "--training-order",
                    "ready",
                    "--no-cross-epoch-prefetch",
                    "--progress-interval-s",
                    "30",
                ]
            record = execute(command, run_dir, root, environment, args.timeout_s)
            try:
                if record["returncode"] or not record["clean_process_exit"]:
                    raise RuntimeError(
                        f"returncode={record['returncode']}, reason={record['reason']}, "
                        f"lingering={record['lingering_processes']}"
                    )
                result = _read_result(result_path)
                row = _summarize(dataset, profile, result, record)
                if row["completed_batches"] != args.epochs * args.max_batches:
                    raise RuntimeError("incomplete batch coverage")
                replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
                if row["fingerprint"] != replay_manifest["fingerprint"]:
                    raise RuntimeError("replay fingerprint mismatch")
                rows = [
                    old
                    for old in rows
                    if not (old["dataset"] == dataset and old["profile"] == profile)
                ]
                rows.append(row)
                state["completed"].append(key)
                _write_summary(output, rows)
            except Exception as exc:  # noqa: BLE001 - retain failure and stop matrix.
                state["failures"].append({"case": key, "error": str(exc), **record})
                state.update(status="failed", current=None)
                save(state_path, state)
                return 1
            save(state_path, state)

    state.update(status="completed", current=None, finished_unix=time.time())
    save(state_path, state)
    _write_summary(output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
