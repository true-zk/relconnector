"""Run the three-way, two-epoch validation matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CASES = (
    ("rel-amazon", "item-churn"),
    ("rel-arxiv", "paper-citation"),
    ("rel-avito", "searchstream-click"),
    ("rel-event", "user-attendance"),
    ("rel-f1", "driver-circuit-compete"),
    ("rel-hm", "user-churn"),
    ("rel-ratebeer", "beer_ratings-total_score"),
    ("rel-salt", "item-incoterms"),
    ("rel-stack", "user-badge"),
    ("rel-trial", "eligibilities-adult"),
)


def _common(output: Path, *, device: str) -> list[str]:
    command = [
        "--epochs",
        "2",
        "--max-batches",
        "10000",
        "--batch-size",
        "512",
        "--num-neighbors",
        "128,128",
        "--channels",
        "128",
        "--torch-num-threads",
        "1",
        "--device",
        device,
        "--output",
        str(output),
        "--resume",
    ]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation-2epoch-10000"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; refusing to run this multi-day validation "
                "matrix on CPU"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        (
            "batch",
            [
                sys.executable,
                "-m",
                "benchmark.runner",
                "--executor",
                "sync",
            ],
            args.output_dir / "batch.jsonl",
        ),
        (
            "vanilla",
            [
                sys.executable,
                "-m",
                "benchmark.online_runner",
                "--implementation",
                "vanilla",
                "--executor",
                "sync",
                "--feature-cache-mb",
                "0",
            ],
            args.output_dir / "vanilla.jsonl",
        ),
        (
            "latest",
            [
                sys.executable,
                "-m",
                "benchmark.online_runner",
                "--implementation",
                "latest",
                "--executor",
                "async",
            ],
            args.output_dir / "latest.jsonl",
        ),
    ]
    if not args.resume:
        for _, _, output in runs:
            output.unlink(missing_ok=True)
    for name, prefix, output in runs:
        for index, (dataset, task) in enumerate(CASES, start=1):
            command = [
                *prefix,
                "--dataset",
                dataset,
                "--task",
                task,
                *_common(output, device=args.device),
            ]
            print(
                f"[{name} {index}/{len(CASES)}] {dataset}/{task}",
                flush=True,
            )
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                print(
                    f"{name} {dataset}/{task} returned "
                    f"{completed.returncode}; continuing",
                    file=sys.stderr,
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
