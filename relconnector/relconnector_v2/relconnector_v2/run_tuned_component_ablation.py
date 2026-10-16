"""Tune per-dataset process supply, then run the full ten-dataset ablation.

The experiment owns its output and text-cache directories.  Schema/topology
artifacts stay warm, while text embeddings are removed before and after every
profile.  Disk text caching is therefore cold at profile start but can pay back
inside the five-epoch run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .run_cache_iteration import clear_text_cache, execute, save
from .run_validation_matrix import CASES


PROFILE_ORDER = (
    "vanilla_row_sync_official",
    "vanilla_hybrid_sync_official",
    "v2_sync_official",
    "v2_sync_direct",
    "v2_async_direct_win1",
    "v2_async_direct_tuned_window",
    "v2_process_tuned_strict",
    "v2_process_tuned_ready",
    "v2_process_tuned_ready_disk",
)

IMPROVEMENTS = {
    "vanilla_row_sync_official": "baseline_single_row_sql",
    "vanilla_hybrid_sync_official": "hybrid_dense_sparse_sql",
    "v2_sync_official": "v2_feature_path",
    "v2_sync_direct": "direct_glove",
    "v2_async_direct_win1": "thread_pipeline_overlap",
    "v2_async_direct_tuned_window": "tuned_multi_batch_window",
    "v2_process_tuned_strict": "tuned_process_workers",
    "v2_process_tuned_ready": "ready_order_no_head_of_line_blocking",
    "v2_process_tuned_ready_disk": "cold_disk_text_cache",
}

PRIORITY = (
    "rel-f1",
    "rel-trial",
    "rel-salt",
    "rel-arxiv",
    "rel-stack",
    "rel-amazon",
    "rel-avito",
    "rel-hm",
    "rel-ratebeer",
    "rel-event",
)

REPLAY_SOURCES = {
    "rel-amazon": Path(
        "benchmarks/cache-roi-amazon-ratebeer-100x5-v1/rel-amazon/replay"
    ),
    "rel-ratebeer": Path(
        "benchmarks/cache-roi-amazon-ratebeer-100x5-v1/rel-ratebeer/replay"
    ),
}

INITIALIZATION_SOURCES = {
    "rel-amazon": Path(
        "benchmarks/ready-worker-scaling-amazon-ratebeer-100x5-20261005-v1/"
        "rel-amazon/cache"
    ),
    "rel-ratebeer": Path(
        "benchmarks/ready-worker-scaling-amazon-ratebeer-100x5-20261005-v1/"
        "rel-ratebeer/cache"
    ),
}


def _link_tree(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, copy_function=os.link)


def _subset_replay(source: Path, target: Path, maximum: int) -> None:
    if target.exists():
        return
    manifest = json.loads((source / "manifest.json").read_text())
    records = manifest["batches"][:maximum]
    target.mkdir(parents=True)
    for item in records:
        for key in ("plan", "seed"):
            os.link(source / item[key], target / item[key])
    identity = dict(manifest["identity"])
    identity["max_batches"] = maximum
    subset = {"version": 1, "identity": identity, "batches": records}
    subset["fingerprint"] = hashlib.sha256(
        json.dumps(subset, sort_keys=True).encode()
    ).hexdigest()
    (target / "manifest.json").write_text(json.dumps(subset, indent=2) + "\n")


def _gpu_uuid(index: int) -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        number, uuid = (part.strip() for part in line.split(",", 1))
        if int(number) == index:
            return uuid
    raise RuntimeError(f"CUDA device {index} was not reported by nvidia-smi")


def _gpu_processes(uuid: str) -> list[dict[str, object]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4 and parts[0] == uuid:
            rows.append(
                {
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mb": int(parts[3]),
                }
            )
    return rows


def _wait_for_idle_gpu(
    uuid: str,
    state: dict[str, Any],
    status_path: Path,
    *,
    poll_s: float,
    stable_s: float,
) -> None:
    idle_since: float | None = None
    while True:
        processes = _gpu_processes(uuid)
        now = time.time()
        if processes:
            idle_since = None
            state.update(status="waiting_gpu", gpu_processes=processes)
        else:
            idle_since = now if idle_since is None else idle_since
            state.update(
                status="waiting_gpu_stability",
                gpu_processes=[],
                gpu_idle_s=now - idle_since,
            )
            if now - idle_since >= stable_s:
                state.pop("gpu_processes", None)
                state.pop("gpu_idle_s", None)
                save(status_path, state)
                return
        save(status_path, state)
        time.sleep(poll_s)


def _result(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"expected one result in {path}, found {len(lines)}")
    result = json.loads(lines[0])
    if result.get("status") != "ok":
        raise ValueError(f"unsuccessful result in {path}")
    return result


def _training_metrics(result: dict[str, Any]) -> tuple[float, list[float]]:
    pipeline = result.get("pipeline", {})
    epochs = pipeline.get("epochs", [])
    walls = [float(epoch["wall_duration_s"]) for epoch in epochs if "wall_duration_s" in epoch]
    if walls:
        return sum(walls), walls
    return float(result["training_s"]), []


def _disk_metrics(result: dict[str, Any]) -> dict[str, object]:
    active = 0
    hits = 0
    misses = 0
    reasons: set[str] = set()
    for worker in result.get("pipeline", {}).get("workers", []):
        if worker.get("stage") != "assembly":
            continue
        disk = worker.get("diagnostics", {}).get("text_embedding", {}).get("disk", {})
        active += int(bool(disk.get("active")))
        hits += int(disk.get("hits", 0))
        misses += int(disk.get("misses", 0))
        if disk.get("activation_reason"):
            reasons.add(str(disk["activation_reason"]))
    return {
        "disk_active_workers": active,
        "disk_hits": hits,
        "disk_misses": misses,
        "disk_activation_reasons": sorted(reasons),
    }


def _row(
    dataset: str,
    profile: str,
    result: dict[str, Any],
    execution: dict[str, Any],
    *,
    workers: int,
    window: int,
) -> dict[str, Any]:
    training_s, epoch_wall = _training_metrics(result)
    completed = int(result["completed_batches"])
    fingerprint = result.get("fingerprint") or result.get("pipeline", {}).get(
        "replay", {}
    ).get("fingerprint")
    memory = execution["memory"]
    cold = epoch_wall[0] if epoch_wall else None
    warm = sum(epoch_wall[1:]) / len(epoch_wall[1:]) if len(epoch_wall) > 1 else None
    return {
        "dataset": dataset,
        "profile": profile,
        "improvement": IMPROVEMENTS[profile],
        "workers": workers,
        "window_batches": window,
        "completed_batches": completed,
        "fingerprint": fingerprint,
        "training_s": training_s,
        "training_s_per_batch": training_s / completed,
        "cold_epoch_s": cold,
        "warm_epoch_mean_s": warm,
        "mean_loss": result["mean_loss"],
        "initialization_cache": result.get("initialization_cache"),
        "peak_main_rss_mb": memory["peak_main_rss_mb"],
        "peak_tree_pss_mb": memory["peak_tree_pss_mb"],
        "peak_tree_rss_mb": memory["peak_tree_rss_mb"],
        "elapsed_s": execution["elapsed_s"],
        **_disk_metrics(result),
    }


def _write_reports(output: Path, rows: list[dict[str, Any]]) -> None:
    order = {name: position for position, name in enumerate(PROFILE_ORDER)}
    rows.sort(key=lambda row: (PRIORITY.index(row["dataset"]), order[row["profile"]]))
    effects = []
    for dataset in PRIORITY:
        group = [row for row in rows if row["dataset"] == dataset]
        if not group:
            continue
        baseline = group[0]["training_s_per_batch"]
        previous = None
        for row in group:
            current = row["training_s_per_batch"]
            effects.append(
                {
                    "dataset": dataset,
                    "profile": row["profile"],
                    "improvement": row["improvement"],
                    "workers": row["workers"],
                    "window_batches": row["window_batches"],
                    "training_s_per_batch": current,
                    "incremental_reduction_pct": (
                        None if previous is None else 100 * (1 - current / previous)
                    ),
                    "cumulative_reduction_pct": 100 * (1 - current / baseline),
                }
            )
            previous = current
    save(output / "summary.json", rows)
    save(output / "effects.json", effects)
    for name, records in (("summary.csv", rows), ("effects.csv", effects)):
        if not records:
            continue
        flattened = [
            {
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in record.items()
            }
            for record in records
        ]
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)


def _base_args(
    python: str,
    dataset: str,
    task: str,
    root: Path,
    replay_dir: Path,
    cache_dir: Path,
    *,
    epochs: int,
    max_batches: int,
    device: str,
) -> list[str]:
    return [
        python,
        "-m",
        "relconnector_v2.benchmark",
        "--dataset",
        dataset,
        "--task",
        task,
        "--database-dir",
        str(root / "data" / "relbench"),
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
        device,
        "--torch-num-threads",
        "1",
        "--text-batch-size",
        "256",
        "--no-adaptive-workers",
        "--no-cross-epoch-prefetch",
        "--max-inflight-windows",
        "8",
        "--progress-interval-s",
        "30",
    ]


def _process_command(
    base: list[str],
    output: Path,
    *,
    workers: int,
    window: int,
    order: str,
    disk: bool,
) -> list[str]:
    return [
        *base,
        "--output",
        str(output),
        "--assembly-processes",
        str(workers),
        "--initial-active-processes",
        str(workers),
        "--window-batches",
        str(window),
        "--training-order",
        order,
        "--feature-cache-mb",
        "0",
        "--encoded-cache-mb",
        "0",
        "--text-cache-mb",
        "0",
        "--text-cache-scope",
        "local",
        "--text-execution",
        "direct",
        "--disk-text-cache-mode",
        "on" if disk else "off",
        "--cache-strategy",
        "legacy" if disk else "off",
        "--validate-categorical-inputs",
    ]


def _single_command(
    python: str,
    module: str,
    dataset: str,
    task: str,
    root: Path,
    replay: Path,
    cache: Path,
    output: Path,
    *,
    epochs: int,
    max_batches: int,
    device: str,
    extras: list[str],
) -> list[str]:
    return [
        python,
        "-m",
        module,
        "--dataset",
        dataset,
        "--task",
        task,
        "--database-dir",
        str(root / "data" / "relbench"),
        "--replay-dir",
        str(replay),
        "--cache-dir",
        str(cache),
        "--output",
        str(output),
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
        device,
        "--torch-num-threads",
        "1",
        "--text-batch-size",
        "256",
        *extras,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--tune-epochs", type=int, default=2)
    parser.add_argument("--tune-batches", type=int, default=20)
    parser.add_argument("--tune-workers", default="1,2,4,8")
    parser.add_argument("--tune-windows", default="1,4,8,16")
    parser.add_argument("--timeout-s", type=float, default=86400)
    parser.add_argument("--poll-gpu-s", type=float, default=30)
    parser.add_argument("--gpu-stable-s", type=float, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs != 5 or args.max_batches != 100:
        parser.error("this full experiment is fixed to 5 epochs and 100 batches")
    workers = tuple(int(value) for value in args.tune_workers.split(","))
    windows = tuple(int(value) for value in args.tune_windows.split(","))
    if min(*workers, *windows, args.tune_epochs, args.tune_batches) <= 0:
        parser.error("tuning values must be positive")

    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen"
    if not frozen.exists():
        shutil.copytree(
            Path(__file__).resolve().parent,
            frozen / "relconnector_v2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            root / "baseline",
            frozen / "baseline",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    hashes = {
        str(path.relative_to(frozen)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen.rglob("*.py")
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(frozen),
        "RELCONNECTOR_PROJECT_ROOT": str(root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    python = sys.executable
    cases = tuple(sorted(CASES, key=lambda case: PRIORITY.index(case[0])))
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        save(
            manifest_path,
            {
                "purpose": "ten-dataset tuned component ablation with cold disk text cache",
                "cases": cases,
                "profiles": PROFILE_ORDER,
                "improvements": IMPROVEMENTS,
                "epochs": args.epochs,
                "max_batches": args.max_batches,
                "tuning": {
                    "epochs": args.tune_epochs,
                    "max_batches": args.tune_batches,
                    "workers": workers,
                    "windows": windows,
                    "selection": "minimum training seconds per batch",
                },
                "cache_protocol": (
                    "text_embeddings removed before and after every profile; disk profile "
                    "starts cold and persists only inside its five-epoch child process"
                ),
                "initialization_cache": "warm schema/topology",
                "os_page_cache": "not cleared",
                "source_sha256": hashes,
            },
        )
    status_path = output / "status.json"
    state = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {
            "pid": os.getpid(),
            "status": "starting",
            "started_unix": time.time(),
            "current": None,
            "completed": [],
            "failures": [],
        }
    )
    state["pid"] = os.getpid()
    save(status_path, state)
    if args.dry_run:
        state["status"] = "planned"
        save(status_path, state)
        return 0

    # Prepare/reuse fixed plans and warm initialization artifacts without GPU.
    for dataset, task in cases:
        dataset_dir = output / dataset
        dataset_dir.mkdir(exist_ok=True)
        cache = dataset_dir / "cache"
        full_replay = dataset_dir / "replay-full"
        tune_replay = dataset_dir / "replay-tune"
        source_cache = INITIALIZATION_SOURCES.get(dataset)
        if not cache.exists() and source_cache is not None and (root / source_cache).exists():
            _link_tree(root / source_cache, cache)
        source_replay = REPLAY_SOURCES.get(dataset)
        if not full_replay.exists() and source_replay is not None and (root / source_replay).exists():
            _link_tree(root / source_replay, full_replay)
        if not full_replay.exists():
            state.update(status="preparing_replay", current=f"{dataset}/replay-full")
            save(status_path, state)
            prepare = [
                python,
                "-m",
                "relconnector_v2.benchmark",
                "--dataset",
                dataset,
                "--task",
                task,
                "--database-dir",
                str(root / "data" / "relbench"),
                "--cache-dir",
                str(cache),
                "--device",
                "cpu",
                "--max-batches",
                str(args.max_batches),
                "--batch-size",
                "512",
                "--num-neighbors",
                "128,128",
                "--channels",
                "128",
                "--torch-num-threads",
                "1",
                "--assembly-processes",
                "1",
                "--initial-active-processes",
                "1",
                "--cache-strategy",
                "off",
                "--replay-dir",
                str(full_replay),
                "--prepare-replay-only",
            ]
            record = execute(
                prepare,
                dataset_dir / "prepare-replay",
                root,
                env,
                args.timeout_s,
                min_free_bytes=10 * 1024**3,
            )
            if record["returncode"]:
                state["failures"].append({"case": state["current"], **record})
                save(status_path, state)
                continue
        if not tune_replay.exists():
            _subset_replay(full_replay, tune_replay, args.tune_batches)

        vanilla_prepare_result = dataset_dir / "prepare-vanilla" / "result.json"
        if not vanilla_prepare_result.exists():
            state.update(status="preparing_vanilla", current=f"{dataset}/prepare-vanilla")
            save(status_path, state)
            command = _single_command(
                python,
                "relconnector_v2.fixed_vanilla_benchmark",
                dataset,
                task,
                root,
                full_replay,
                cache,
                vanilla_prepare_result,
                epochs=1,
                max_batches=args.max_batches,
                device="cpu",
                extras=["--executor", "sync", "--feature-read-policy", "hybrid", "--prepare-only"],
            )
            record = execute(
                command,
                dataset_dir / "prepare-vanilla",
                root,
                env,
                args.timeout_s,
                min_free_bytes=10 * 1024**3,
            )
            if record["returncode"]:
                state["failures"].append({"case": state["current"], **record})
                save(status_path, state)

    gpu_uuid = _gpu_uuid(args.gpu_index)
    tuning_rows: list[dict[str, Any]] = []
    tuning_path = output / "tuning.json"
    if tuning_path.exists():
        tuning_rows = json.loads(tuning_path.read_text())

    # Full grid: tuning is per dataset but fixed across all component profiles.
    for dataset, task in cases:
        dataset_dir = output / dataset
        cache = dataset_dir / "cache"
        replay_dir = dataset_dir / "replay-tune"
        if not (replay_dir / "manifest.json").exists():
            continue
        for worker_count in workers:
            for window in windows:
                key = (dataset, worker_count, window)
                if any(
                    (row["dataset"], row["workers"], row["window_batches"]) == key
                    for row in tuning_rows
                ):
                    continue
                state.update(
                    status="tuning",
                    current=f"{dataset}/w{worker_count}-win{window}",
                )
                save(status_path, state)
                _wait_for_idle_gpu(
                    gpu_uuid,
                    state,
                    status_path,
                    poll_s=args.poll_gpu_s,
                    stable_s=args.gpu_stable_s,
                )
                state.update(status="tuning")
                save(status_path, state)
                run_dir = dataset_dir / "tuning" / f"w{worker_count}-win{window}"
                before = clear_text_cache(cache)
                base = _base_args(
                    python,
                    dataset,
                    task,
                    root,
                    replay_dir,
                    cache,
                    epochs=args.tune_epochs,
                    max_batches=args.tune_batches,
                    device=args.device,
                )
                result_path = run_dir / "result.jsonl"
                command = _process_command(
                    base,
                    result_path,
                    workers=worker_count,
                    window=window,
                    order="ready",
                    disk=False,
                )
                execution = execute(
                    command,
                    run_dir,
                    root,
                    env,
                    args.timeout_s,
                    min_free_bytes=10 * 1024**3,
                )
                after = clear_text_cache(cache)
                execution["text_cache_reset"] = {"before": before, "after": after}
                save(run_dir / "execution.json", execution)
                if execution["returncode"] == 0:
                    result = _result(result_path)
                    training_s, _ = _training_metrics(result)
                    row = {
                        "dataset": dataset,
                        "workers": worker_count,
                        "window_batches": window,
                        "completed_batches": result["completed_batches"],
                        "training_s": training_s,
                        "training_s_per_batch": training_s / result["completed_batches"],
                        "peak_tree_pss_mb": execution["memory"]["peak_tree_pss_mb"],
                    }
                    tuning_rows.append(row)
                    save(tuning_path, tuning_rows)
                else:
                    state["failures"].append({"case": state["current"], **execution})
                    save(status_path, state)

    selected: dict[str, dict[str, Any]] = {}
    for dataset, _ in cases:
        candidates = [row for row in tuning_rows if row["dataset"] == dataset]
        if candidates:
            selected[dataset] = min(candidates, key=lambda row: row["training_s_per_batch"])
    save(output / "selected_tuning.json", selected)

    rows: list[dict[str, Any]] = []
    if (output / "summary.json").exists():
        rows = json.loads((output / "summary.json").read_text())
    for dataset, task in cases:
        if dataset not in selected:
            continue
        dataset_dir = output / dataset
        cache = dataset_dir / "cache"
        replay_dir = dataset_dir / "replay-full"
        worker_count = int(selected[dataset]["workers"])
        window = int(selected[dataset]["window_batches"])
        for profile in PROFILE_ORDER:
            if any(row["dataset"] == dataset and row["profile"] == profile for row in rows):
                continue
            state.update(status="running_ablation", current=f"{dataset}/{profile}")
            save(status_path, state)
            _wait_for_idle_gpu(
                gpu_uuid,
                state,
                status_path,
                poll_s=args.poll_gpu_s,
                stable_s=args.gpu_stable_s,
            )
            state.update(status="running_ablation")
            save(status_path, state)
            run_dir = dataset_dir / "ablation" / profile
            suffix = "result.jsonl" if profile.startswith("v2_process") else "result.json"
            result_path = run_dir / suffix
            before = clear_text_cache(cache)
            if profile.startswith("vanilla_"):
                policy = "row" if profile.startswith("vanilla_row") else "hybrid"
                command = _single_command(
                    python,
                    "relconnector_v2.fixed_vanilla_benchmark",
                    dataset,
                    task,
                    root,
                    replay_dir,
                    cache,
                    result_path,
                    epochs=args.epochs,
                    max_batches=args.max_batches,
                    device=args.device,
                    extras=["--executor", "sync", "--feature-read-policy", policy],
                )
                profile_workers, profile_window = 0, 1
            elif profile.startswith("v2_process"):
                base = _base_args(
                    python,
                    dataset,
                    task,
                    root,
                    replay_dir,
                    cache,
                    epochs=args.epochs,
                    max_batches=args.max_batches,
                    device=args.device,
                )
                disk = profile.endswith("_disk")
                order = "strict" if profile.endswith("_strict") else "ready"
                command = _process_command(
                    base,
                    result_path,
                    workers=worker_count,
                    window=window,
                    order=order,
                    disk=disk,
                )
                profile_workers, profile_window = worker_count, window
            else:
                executor = "sync" if profile.startswith("v2_sync") else "async"
                text_execution = "official" if profile.endswith("official") else "direct"
                single_window = (
                    window if profile.endswith("tuned_window") else 1
                )
                command = _single_command(
                    python,
                    "relconnector_v2.fixed_v2_benchmark",
                    dataset,
                    task,
                    root,
                    replay_dir,
                    cache,
                    result_path,
                    epochs=args.epochs,
                    max_batches=args.max_batches,
                    device=args.device,
                    extras=[
                        "--executor",
                        executor,
                        "--text-execution",
                        text_execution,
                        "--window-batches",
                        str(single_window),
                    ],
                )
                profile_workers, profile_window = 0, single_window
            execution = execute(
                command,
                run_dir,
                root,
                env,
                args.timeout_s,
                min_free_bytes=10 * 1024**3,
            )
            if execution["returncode"] == 0:
                result = _result(result_path)
                expected = len(
                    json.loads((replay_dir / "manifest.json").read_text())["batches"]
                ) * args.epochs
                if result["completed_batches"] != expected:
                    raise RuntimeError(
                        f"{dataset}/{profile}: expected {expected} batches, "
                        f"got {result['completed_batches']}"
                    )
                row = _row(
                    dataset,
                    profile,
                    result,
                    execution,
                    workers=profile_workers,
                    window=profile_window,
                )
                rows.append(row)
                state["completed"].append(f"{dataset}/{profile}")
            else:
                state["failures"].append({"case": state["current"], **execution})
            after = clear_text_cache(cache)
            execution["text_cache_reset"] = {"before": before, "after": after}
            save(run_dir / "execution.json", execution)
            _write_reports(output, rows)
            save(status_path, state)

    expected_runs = len(cases) * len(PROFILE_ORDER)
    state.update(
        status="completed" if len(rows) == expected_runs and not state["failures"] else "completed_with_failures",
        current=None,
        finished_unix=time.time(),
        completed_runs=len(rows),
        expected_runs=expected_runs,
    )
    save(status_path, state)
    _write_reports(output, rows)
    return int(state["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
