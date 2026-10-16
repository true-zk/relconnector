"""Sequential ten-dataset, five-variant ablation with durable status and summaries."""

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

from .run_ablation import CORE
from .run_validation_matrix import CASES


def save(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def aggregate(output: Path, cases: list[tuple[str, str]]) -> None:
    rows = []
    problems = []
    for dataset, task in cases:
        path = output / dataset / "summary.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        problems.extend(f"{dataset}: {item}" for item in result["problems"])
        rows.extend({"dataset": dataset, "task": task, **row} for row in result["rows"])
    save(
        output / "summary.json",
        {
            "completed": len(rows),
            "expected": len(cases) * len(CORE),
            "problems": problems,
            "rows": rows,
        },
    )
    if rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    priority = [
        "rel-f1",
        "rel-avito",
        "rel-hm",
        "rel-trial",
        "rel-arxiv",
        "rel-salt",
        "rel-stack",
        "rel-amazon",
        "rel-ratebeer",
        "rel-event",
    ]
    cases = sorted(CASES, key=lambda item: priority.index(item[0]))
    commands = []
    for dataset, task in cases:
        commands.append(
            [
                sys.executable,
                "-m",
                "relconnector_v2.run_ablation",
                "--dataset",
                dataset,
                "--task",
                task,
                "--output-dir",
                str(output / dataset),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--max-batches",
                str(args.max_batches),
                "--repeats",
                "1",
                "--timeout-s",
                str(args.timeout_s),
                "--continue-on-error",
                "--discard-run-cache",
            ]
        )
    package = Path(__file__).resolve().parent
    for command in commands:
        command.extend(["--database-dir", str(root / "data" / "relbench")])
    source = output / "source"
    shutil.copytree(
        package, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*.py")
    }
    save(
        output / "manifest.json",
        {
            "pid": os.getpid(),
            "cases": cases,
            "variants": CORE,
            "epochs": args.epochs,
            "max_batches_per_epoch": args.max_batches,
            "repeats": 1,
            "commands": commands,
            "source_sha256": hashes,
            "cache_protocol": "Fresh per-run application caches. Generated caches discarded after each run; diagnostics retained. OS page cache uncontrolled.",
            "memory_scope": "parent-only RSS, not total memory",
            "mode": "dry_run" if args.dry_run else "execute",
        },
    )
    environment = dict(os.environ)
    # Execute the frozen source even if the working checkout is later edited.
    frozen_parent = output / "frozen"
    frozen_parent.mkdir()
    (frozen_parent / "relconnector_v2").symlink_to(source, target_is_directory=True)
    environment["PYTHONPATH"] = str(frozen_parent)
    environment["RELCONNECTOR_PROJECT_ROOT"] = str(root)
    state = {
        "pid": os.getpid(),
        "started_unix": time.time(),
        "status": "planned" if args.dry_run else "running",
        "completed_datasets": [],
        "current_dataset": None,
    }
    save(output / "status.json", state)
    if args.dry_run:
        return 0
    failures = 0
    for (dataset, task), command in zip(cases, commands, strict=True):
        if shutil.disk_usage(output).free < 10 * 1024**3:
            state.update(status="disk_space_blocked", current_dataset=dataset)
            save(output / "status.json", state)
            return 2
        state["current_dataset"] = dataset
        save(output / "status.json", state)
        print(f"start {dataset}/{task}", flush=True)
        with (output / f"{dataset}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        with (output / f"{dataset}-summary.log").open("w") as log:
            check = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "relconnector_v2.summarize_ablation",
                    str(output / dataset),
                ],
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        failed = result.returncode != 0 or check.returncode != 0
        failures += int(failed)
        state["completed_datasets"].append(
            {
                "dataset": dataset,
                "returncode": result.returncode,
                "validation_returncode": check.returncode,
            }
        )
        aggregate(output, cases)
        save(output / "status.json", state)
        print(f"finished {dataset}: {'failed' if failed else 'ok'}", flush=True)
    state.update(
        status="completed_with_failures" if failures else "completed",
        current_dataset=None,
        finished_unix=time.time(),
    )
    save(output / "status.json", state)
    aggregate(output, cases)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
