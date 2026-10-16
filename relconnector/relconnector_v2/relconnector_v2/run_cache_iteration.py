"""Fixed-input cache screening and natural-sampling regression checks.

Each run starts with an empty text cache and warm initialization artifacts.
Fixed plans and source hashes are retained; measurements never share a trainer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .run_validation_matrix import CASES

PROFILES = {
    "off": ["--cache-strategy", "off"],
    "legacy": ["--cache-strategy", "legacy"],
    "cost_ram": ["--cache-strategy", "cost", "--disk-text-cache-mode", "off"],
    "cost_disk_only": ["--cache-strategy", "cost", "--text-cache-mb", "0"],
    "cost_disk": ["--cache-strategy", "cost"],
    "roi_disk": ["--cache-strategy", "roi"],
}


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def process_group_pids(pgid: int) -> list[int]:
    """Return live members of a process group without invoking ps."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid:
                members.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return members


def process_memory_kib(pid: int, *, include_pss: bool) -> tuple[int, int | None]:
    """Read current RSS and, when requested, proportional set size."""
    rss = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
                break
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return 0, None
    if not include_pss:
        return rss, None
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return rss, int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return rss, None


def sample_process_group(pgid: int, *, include_pss: bool) -> dict[str, Any]:
    pids = process_group_pids(pgid)
    rss_kib = 0
    pss_kib = 0
    pss_complete = include_pss
    main_rss_kib = 0
    for pid in pids:
        rss, pss = process_memory_kib(pid, include_pss=include_pss)
        rss_kib += rss
        if pid == pgid:
            main_rss_kib = rss
        if include_pss:
            if pss is None:
                pss_complete = False
            else:
                pss_kib += pss
    return {
        "processes": len(pids),
        "tree_rss_mb": rss_kib / 1024,
        "tree_pss_mb": pss_kib / 1024 if pss_complete else None,
        "main_rss_mb": main_rss_kib / 1024,
    }


def clear_text_cache(cache_dir: Path) -> dict[str, object]:
    """Remove only the per-run text cache while retaining warm initialization."""
    target = cache_dir / "text_embeddings"
    if target.parent != cache_dir:
        raise ValueError("Refusing to clean a text cache outside the dataset cache")
    existed = target.exists()
    if existed:
        shutil.rmtree(target)
    return {
        "path": str(target),
        "existed": existed,
        "verified_absent": not target.exists(),
    }


def measure(path: Path, expected: int, fingerprint: str | None) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected exactly one result, refusing mixed run data")
    result = json.loads(lines[0])
    pipeline = result["pipeline"]
    if result["status"] != "ok" or result["completed_batches"] != expected:
        raise ValueError("Failed or incomplete batch coverage")
    actual_fingerprint = pipeline.get("replay", {}).get("fingerprint")
    if fingerprint != actual_fingerprint:
        raise ValueError("Fixed input fingerprint differs between runs")
    if result["categorical_input_validation"]["batches"] != expected:
        raise ValueError("Categorical validation coverage is incomplete")
    epochs = pipeline["epochs"]
    if len(epochs) != result["config"]["epochs"]:
        raise ValueError("Incomplete epoch records")
    if any(record["steps"] * len(epochs) != expected for record in epochs):
        raise ValueError("Batch coverage differs between epochs")
    times = [record["wall_duration_s"] for record in epochs]
    counters: dict[str, float] = {}
    for worker in pipeline["workers"]:
        if worker["stage"] != "assembly":
            continue
        disk = worker["diagnostics"]["text_embedding"]["disk"]
        if disk.get("strategy") in {"cost", "roi"}:
            for key, value in disk["counters"].items():
                counters[key] = counters.get(key, 0) + value
    return {
        "steps": result["completed_batches"],
        "examples": result["completed_examples"],
        "pipeline_s": pipeline["duration_s"],
        "end_to_end_s": result["duration_s"],
        "training_wall_s": sum(times),
        "cold_epoch_s": times[0],
        "warm_epoch_mean_s": sum(times[1:]) / max(len(times) - 1, 1),
        "epoch_wall_s": times,
        "epoch_steps": [record["steps"] for record in epochs],
        "fingerprint": actual_fingerprint,
        "mean_loss": result["mean_loss"],
        "reported_main_peak_rss_mb": result["peak_rss_mb"],
        "cost_counters": counters,
    }


def report(output: Path, runs: list[dict[str, Any]]) -> None:
    rows = [dict(run) for run in runs]
    for row in rows:
        baseline = next(
            (
                r
                for r in rows
                if r["dataset"] == row["dataset"]
                and r["stage"] == row["stage"]
                and r["repeat"] == row["repeat"]
                and r["profile"] == "off"
            ),
            None,
        )
        row["total_change_pct"] = None
        row["warm_change_pct"] = None
        row["break_even_epoch"] = None
        if baseline is not None:
            row["total_change_pct"] = 100 * (
                row["training_wall_s"] / baseline["training_wall_s"] - 1
            )
            row["warm_change_pct"] = 100 * (
                row["warm_epoch_mean_s"] / baseline["warm_epoch_mean_s"] - 1
            )
            deltas = [
                a - b
                for a, b in zip(
                    row["epoch_wall_s"], baseline["epoch_wall_s"], strict=True
                )
            ]
            cumulative = [sum(deltas[: i + 1]) for i in range(len(deltas))]
            # Only call it a measured break-even if it stays ahead thereafter.
            row["break_even_epoch"] = next(
                (
                    i + 1
                    for i in range(len(cumulative))
                    if all(v <= 0 for v in cumulative[i:])
                ),
                None,
            )
    save(output / "summary.json", rows)
    if rows:
        flat = [
            {
                key: value
                for key, value in row.items()
                if key not in {"cost_counters", "epoch_wall_s", "epoch_steps"}
            }
            for row in rows
        ]
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    text = [
        "# 缓存策略迭代：实时筛选结果",
        "",
        "负百分比表示快于同一轮全关缓存。单次结果只能用于筛选，不能证明统计非劣效。",
        "固定回放保持种子、邻居、标签、负样本一致；natural 使用正常跨 epoch 采样。",
        "冷轮包含进程启动；总训练时间包含首批等待，不含初始化与最后诊断回收。",
        "OS 文件页缓存未清空。各策略的应用文本缓存均冷启动，初始化缓存已预热。",
        "",
        "| 数据集 | 输入 | 策略 | 轮次 | 总训练秒 | 后续轮均值秒 | 总时间变化 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        change = row["total_change_pct"]
        text.append(
            f"| {row['dataset']} | {row['stage']} | {row['profile']} | {row['repeat']} | "
            f"{row['training_wall_s']:.2f} | {row['warm_epoch_mean_s']:.2f} | "
            + ("待基线" if change is None else f"{change:+.1f}%")
            + " |"
        )
    (output / "REPORT.md").write_text("\n".join(text) + "\n")


def execute(
    command: list[str],
    directory: Path,
    root: Path,
    environment: dict[str, str],
    timeout: float,
    min_free_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reason = None
    memory: dict[str, Any] = {
        "sampling_interval_s": 0.5,
        "pss_sampling_interval_s": 1.0,
        "samples": 0,
        "pss_samples": 0,
        "peak_processes": 0,
        "peak_main_rss_mb": 0.0,
        "peak_tree_rss_mb": 0.0,
        "peak_tree_pss_mb": None,
        "peak_main_rss_elapsed_s": None,
        "peak_tree_rss_elapsed_s": None,
        "peak_tree_pss_elapsed_s": None,
    }
    with tempfile.TemporaryDirectory(prefix=".cc-", dir=root.parent) as scratch:
        env = {**environment, "TMPDIR": scratch, "SQLITE_TMPDIR": scratch}
        with (directory / "run.log").open("w") as handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_pss = started
            while process.poll() is None:
                now = time.time()
                include_pss = now >= next_pss
                sample = sample_process_group(process.pid, include_pss=include_pss)
                elapsed = now - started
                memory["samples"] += 1
                memory["peak_processes"] = max(
                    memory["peak_processes"], sample["processes"]
                )
                if sample["main_rss_mb"] > memory["peak_main_rss_mb"]:
                    memory["peak_main_rss_mb"] = sample["main_rss_mb"]
                    memory["peak_main_rss_elapsed_s"] = elapsed
                if sample["tree_rss_mb"] > memory["peak_tree_rss_mb"]:
                    memory["peak_tree_rss_mb"] = sample["tree_rss_mb"]
                    memory["peak_tree_rss_elapsed_s"] = elapsed
                if include_pss:
                    next_pss = now + memory["pss_sampling_interval_s"]
                    if sample["tree_pss_mb"] is not None:
                        memory["pss_samples"] += 1
                        if (
                            memory["peak_tree_pss_mb"] is None
                            or sample["tree_pss_mb"] > memory["peak_tree_pss_mb"]
                        ):
                            memory["peak_tree_pss_mb"] = sample["tree_pss_mb"]
                            memory["peak_tree_pss_elapsed_s"] = elapsed
                if elapsed > timeout:
                    reason = "timeout"
                elif shutil.disk_usage(directory).free < min_free_bytes:
                    reason = "disk_space_guard"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(memory["sampling_interval_s"])
            # A clean comparison requires every descendant to be gone before the
            # next profile starts. Give normal multiprocessing teardown a grace
            # period, then terminate any orphaned group members and fail the run.
            teardown_deadline = time.time() + 5
            lingering = process_group_pids(process.pid)
            while lingering and time.time() < teardown_deadline:
                time.sleep(0.1)
                lingering = process_group_pids(process.pid)
            lingering_detected = list(lingering)
            if lingering:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                termination_deadline = time.time() + 2
                while lingering and time.time() < termination_deadline:
                    time.sleep(0.1)
                    lingering = process_group_pids(process.pid)
                if lingering:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    kill_deadline = time.time() + 2
                    while lingering and time.time() < kill_deadline:
                        time.sleep(0.1)
                        lingering = process_group_pids(process.pid)
    record = {
        "command": command,
        "returncode": process.returncode,
        "reason": reason,
        "elapsed_s": time.time() - started,
        "memory": memory,
        "clean_process_exit": not lingering_detected,
        "lingering_processes": lingering_detected,
        "remaining_processes": lingering,
    }
    save(directory / "execution.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=[d for d, _ in CASES])
    parser.add_argument(
        "--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES)
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--natural-check", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--timeout-s", type=float, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.epochs,
            args.max_batches,
            args.repeats,
            args.batch_size,
            args.channels,
            args.timeout_s,
        )
        <= 0
    ):
        parser.error("counts and timeout must be positive")
    if args.epochs < 2:
        parser.error("cross-epoch analysis requires at least two epochs")
    root = Path(
        os.environ.get("RELCONNECTOR_PROJECT_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    priority = [
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
    ]
    cases = sorted(
        [c for c in CASES if args.datasets is None or c[0] in args.datasets],
        key=lambda c: priority.index(c[0]),
    )
    source = output / "frozen" / "relconnector_v2"
    shutil.copytree(
        Path(__file__).resolve().parent,
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*.py")
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(source.parent),
        "RELCONNECTOR_PROJECT_ROOT": str(root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    save(
        output / "manifest.json",
        {
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "cases": cases,
            "source_sha256": hashes,
            "profiles": PROFILES,
            "fixed_order": "strict",
            "fixed_cross_epoch_prefetch": False,
            "natural_order": "ready",
            "natural_cross_epoch_prefetch": True,
            "cache_protocol": (
                "Text cache removed and verified before and after every sequential run; "
                "warm schema/topology; OS page cache not cleared"
            ),
            "memory_scope": (
                "RUSAGE_SELF main-process high-water mark plus sampled process-group "
                "RSS sum and PSS sum; RSS sum double-counts shared pages"
            ),
        },
    )
    state: dict[str, Any] = {
        "pid": os.getpid(),
        "status": "planned",
        "started_unix": time.time(),
        "current": None,
        "completed": [],
        "failures": [],
    }
    save(output / "status.json", state)
    if args.dry_run:
        return 0
    if args.wait_for_pid is not None:
        identity = process_identity(args.wait_for_pid)
        state.update(status="waiting", wait_for_pid=args.wait_for_pid)
        save(output / "status.json", state)
        while identity is not None and process_identity(args.wait_for_pid) == identity:
            time.sleep(15)
    runs: list[dict[str, Any]] = []
    rng = random.Random(1729)
    for dataset, task in cases:
        directory = output / dataset
        directory.mkdir()
        cache_dir, replay_dir = directory / "cache", directory / "replay"
        common = [
            sys.executable,
            "-m",
            "relconnector_v2.benchmark",
            "--dataset",
            dataset,
            "--task",
            task,
            "--database-dir",
            str(root / "data" / "relbench"),
            "--batch-size",
            str(args.batch_size),
            "--num-neighbors",
            args.num_neighbors,
            "--channels",
            str(args.channels),
            "--seed",
            "42",
            "--torch-num-threads",
            "1",
            "--assembly-processes",
            "4",
            "--initial-active-processes",
            "4",
            "--no-adaptive-workers",
            "--training-order",
            "strict",
            "--no-cross-epoch-prefetch",
            "--feature-cache-mb",
            "4096",
            "--encoded-cache-mb",
            "2048",
            "--encoded-cache-scope",
            "shared",
            "--text-cache-mb",
            "512",
            "--text-cache-scope",
            "adaptive",
            "--disk-text-cache-mode",
            "adaptive",
            "--cost-disk-cache-mb",
            "2048",
            "--cache-dir",
            str(cache_dir),
        ]
        prepare = [
            *common,
            "--device",
            "cpu",
            "--max-batches",
            str(args.max_batches),
            "--replay-dir",
            str(replay_dir),
            "--prepare-replay-only",
            "--cache-strategy",
            "off",
        ]
        state.update(status="running", current=f"{dataset}/prepare")
        save(output / "status.json", state)
        record = execute(prepare, directory / "prepare", root, env, args.timeout_s)
        if record["remaining_processes"]:
            state["failures"].append(
                {
                    "case": state["current"],
                    "error": "Live processes remain after forced cleanup",
                    **record,
                }
            )
            state["status"] = "aborted_unclean_process"
            save(output / "status.json", state)
            report(output, runs)
            return 1
        if record["returncode"] or not record["clean_process_exit"]:
            state["failures"].append({"case": state["current"], **record})
            save(output / "status.json", state)
            continue
        manifest = json.loads((replay_dir / "manifest.json").read_text())
        fixed_count = len(manifest["batches"])
        stages = ["fixed", "natural"] if args.natural_check else ["fixed"]
        for stage in stages:
            epochs = args.epochs if stage == "fixed" else 2
            max_batches = (
                args.max_batches if stage == "fixed" else min(50, args.max_batches)
            )
            for repeat in range(1, args.repeats + 1):
                profiles = (
                    list(args.profiles) if stage == "fixed" else ["off", "cost_disk"]
                )
                rng.shuffle(profiles)
                for profile in profiles:
                    run_dir = directory / f"{stage}-r{repeat}-{profile}"
                    state["current"] = f"{dataset}/{run_dir.name}"
                    save(output / "status.json", state)
                    before_reset = clear_text_cache(cache_dir)
                    if not before_reset["verified_absent"]:
                        raise RuntimeError("Text cache cleanup failed before run")
                    command = [
                        *common,
                        "--device",
                        args.device,
                        "--epochs",
                        str(epochs),
                        "--max-batches",
                        str(max_batches),
                        "--validate-categorical-inputs",
                        "--output",
                        str(run_dir / "result.jsonl"),
                        *PROFILES[profile],
                    ]
                    if stage == "fixed":
                        command += ["--replay-dir", str(replay_dir)]
                    else:
                        command.remove("--no-cross-epoch-prefetch")
                        command += ["--training-order", "ready"]
                    record = execute(command, run_dir, root, env, args.timeout_s)
                    record["text_cache_reset"] = {"before_run": before_reset}
                    if record["remaining_processes"]:
                        record["text_cache_reset"]["after_run"] = clear_text_cache(
                            cache_dir
                        )
                        save(run_dir / "execution.json", record)
                        state["failures"].append(
                            {
                                "case": state["current"],
                                "error": "Live processes remain after forced cleanup",
                                **record,
                            }
                        )
                        state["status"] = "aborted_unclean_process"
                        save(output / "status.json", state)
                        report(output, runs)
                        return 1
                    try:
                        if record["returncode"] or not record["clean_process_exit"]:
                            raise RuntimeError(
                                "Child did not exit cleanly: "
                                f"returncode={record['returncode']}, "
                                f"reason={record['reason']}, "
                                f"lingering={record['lingering_processes']}"
                            )
                        expected = (
                            fixed_count * epochs
                            if stage == "fixed"
                            else min(max_batches, fixed_count) * epochs
                        )
                        row = {
                            "dataset": dataset,
                            "task": task,
                            "stage": stage,
                            "profile": profile,
                            "repeat": repeat,
                            "monitored_memory": record["memory"],
                            **measure(
                                run_dir / "result.jsonl",
                                expected,
                                manifest["fingerprint"] if stage == "fixed" else None,
                            ),
                        }
                        runs.append(row)
                        state["completed"].append(state["current"])
                        report(output, runs)
                        print(
                            f"{state['current']}: {row['training_wall_s']:.2f}s",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001 -- retain failures and finish the matrix
                        state["failures"].append(
                            {"case": state["current"], "error": str(exc), **record}
                        )
                    finally:
                        record["text_cache_reset"]["after_run"] = clear_text_cache(
                            cache_dir
                        )
                        save(run_dir / "execution.json", record)
                    save(output / "status.json", state)
        # Keep the fixed inputs, manifests and all diagnostics; release rebuildable topology.
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    state.update(
        status="completed_with_failures" if state["failures"] else "completed",
        current=None,
        finished_unix=time.time(),
    )
    save(output / "status.json", state)
    report(output, runs)
    return int(bool(state["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
