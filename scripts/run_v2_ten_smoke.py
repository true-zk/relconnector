"""Run current relconnector_v2 source with the RelX ten-task smoke settings."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from prepare_relx_ten_smoke import CASES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/relx_ten_smoke"))
    parser.add_argument("--dataset", action="append", choices=[name for name, _ in CASES])
    parser.add_argument("--timeout", type=int, default=1_800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "exps" / "relconnector_v2")
    environment["TMPDIR"] = str(root / "build")
    failures = 0
    for dataset, task in CASES:
        if args.dataset and dataset not in args.dataset:
            continue
        output = output_dir / f"v2-{dataset.removeprefix('rel-')}.jsonl"
        if args.resume and output.is_file() and output.stat().st_size:
            print(f"{dataset}: existing {output}", flush=True)
            continue
        command = [
            sys.executable, "-m", "relconnector_v2.benchmark",
            "--dataset", dataset, "--task", task,
            "--database-dir", str(root / "exps" / "data" / "relbench"),
            "--output", str(output), "--epochs", "1", "--max-batches", "1",
            "--batch-size", "8", "--num-neighbors", "4,4",
            "--channels", "16", "--device", "cpu", "--torch-num-threads", "1",
            "--assembly-processes", "1", "--initial-active-processes", "1",
            "--cache-strategy", "off", "--text-cache-mb", "0",
            "--encoded-cache-mb", "0",
        ]
        print(f"{dataset}: running current v2 source", flush=True)
        started = perf_counter()
        try:
            completed = subprocess.run(
                command, cwd=root, env=environment, text=True,
                capture_output=True, timeout=args.timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            failures += 1
            (output_dir / f"v2-{dataset}.log").write_text(
                f"stdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}",
                encoding="utf-8",
            )
            print(f"{dataset}: timeout after {args.timeout}s", flush=True)
            continue
        (output_dir / f"v2-{dataset}.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        failures += completed.returncode != 0
        status = "ok" if completed.returncode == 0 else "failed"
        print(f"{dataset}: {status}, {perf_counter() - started:.1f}s", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
