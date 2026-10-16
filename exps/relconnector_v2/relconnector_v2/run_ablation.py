"""Controlled v2 ablations; fresh cache namespace per run, no production changes."""

from __future__ import annotations

import argparse
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

VARIANTS = {
    "full": [],
    "one_worker": ["--assembly-processes", "1", "--initial-active-processes", "1"],
    "no_feature_cache": ["--feature-cache-mb", "0", "--encoded-cache-mb", "0"],
    "no_text_cache": ["--text-cache-mb", "0", "--disk-text-cache-mode", "off"],
    "no_reuse_cache": [
        "--feature-cache-mb",
        "0",
        "--encoded-cache-mb",
        "0",
        "--text-cache-mb",
        "0",
        "--disk-text-cache-mode",
        "off",
    ],
    "no_raw_cache": ["--feature-cache-mb", "0"],
    "no_encoded_cache": ["--encoded-cache-mb", "0"],
    "no_local_text_index": ["--text-cache-scope", "shared"],
    "no_disk_text_cache": ["--disk-text-cache-mode", "off"],
    "single_batch_window": ["--window-batches", "1"],
}
CORE = list(VARIANTS)[:5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--database-dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=CORE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order-seed", type=int, default=1729)
    parser.add_argument(
        "--training-order", choices=["strict", "ready"], default="ready"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--discard-run-cache", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.repeats,
            args.epochs,
            args.max_batches,
            args.workers,
            args.batch_size,
            args.channels,
            args.timeout_s,
        )
        <= 0
    ):
        parser.error("counts and timeout must be positive")
    root = Path(
        os.environ.get(
            "RELCONNECTOR_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
        )
    )
    output = args.output_dir.resolve()
    # Never resume by dataset/task alone: every variant needs a separate identity.
    output.mkdir(parents=True, exist_ok=False)
    database = (args.database_dir or root / "data" / "relbench").resolve()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": environment.get("PYTHONPATH", str(root / "relconnector_v2")),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
    )
    runs = []
    rng = random.Random(args.order_seed)
    for repeat in range(args.repeats):
        variants = list(args.variants)
        rng.shuffle(variants)
        for variant in variants:
            directory = output / f"r{repeat + 1}-{variant}"
            command = [
                sys.executable,
                "-m",
                "relconnector_v2.benchmark",
                "--dataset",
                args.dataset,
                "--task",
                args.task,
                "--database-dir",
                str(database),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--max-batches",
                str(args.max_batches),
                "--batch-size",
                str(args.batch_size),
                "--channels",
                str(args.channels),
                "--num-neighbors",
                args.num_neighbors,
                "--seed",
                str(args.seed),
                "--torch-num-threads",
                "1",
                "--assembly-processes",
                str(args.workers),
                "--initial-active-processes",
                str(args.workers),
                "--no-adaptive-workers",
                "--training-order",
                args.training_order,
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
                "--cache-dir",
                str(directory / "cache"),
                "--output",
                str(directory / "result.jsonl"),
                *VARIANTS[variant],
            ]
            runs.append(
                {
                    "variant": variant,
                    "repeat": repeat + 1,
                    "directory": str(directory),
                    "command": command,
                }
            )
    manifest = {
        "revision": revision,
        "git_status": dirty,
        "runs": runs,
        "training_seed": args.seed,
        "order_seed": args.order_seed,
        "cache_protocol": "fresh application caches for every run; OS cache uncontrolled",
        "memory_scope": "benchmark peak_rss_mb is parent-only, not total memory",
        "mode": "dry_run" if args.dry_run else "execute",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(f"planned {len(runs)} runs: {output / 'manifest.json'}")
        return 0
    failures = 0
    for run in runs:
        directory = Path(run["directory"])
        directory.mkdir()
        # multiprocessing manager sockets have a short AF_UNIX path limit.
        # Keep temporary files on the project filesystem without deep run paths.
        scratch = Path(tempfile.mkdtemp(prefix=".rc-", dir=root.parent))
        run_environment = dict(environment)
        run_environment.update({"TMPDIR": str(scratch), "SQLITE_TMPDIR": str(scratch)})
        print(f"start {directory.name}", flush=True)
        started = time.monotonic()
        with (directory / "run.log").open("w") as log:
            child = subprocess.Popen(
                run["command"],
                cwd=root,
                env=run_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = child.wait(timeout=args.timeout_s)
                status = "ok" if returncode == 0 else "failed"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                if isinstance(exc, KeyboardInterrupt):
                    raise
                returncode, status = child.returncode, "timeout"
        result = {
            "variant": run["variant"],
            "repeat": run["repeat"],
            "status": status,
            "returncode": returncode,
            "subprocess_wall_s": time.monotonic() - started,
            "scratch_dir": str(scratch),
        }
        if args.discard_run_cache:
            cache = directory / "cache"
            result["discarded_cache_bytes"] = (
                sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
                if cache.exists()
                else 0
            )
            # These are fresh, run-owned artifacts, never the user's existing cache.
            if cache.exists():
                shutil.rmtree(cache)
        shutil.rmtree(scratch)
        with (output / "execution.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        if status != "ok":
            failures += 1
            # Fail early: a broken environment must not consume an entire matrix.
            if not args.continue_on_error:
                break
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
