"""Run the ten-task v2 matrix after the original matrix has completed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation-2epoch-10000"),
    )
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assembly-processes", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.assembly_processes <= 0:
        raise ValueError("poll interval and process count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_for_pid is not None:
        print(
            f"waiting for original matrix pid {args.wait_for_pid}",
            flush=True,
        )
        while _pid_exists(args.wait_for_pid):
            time.sleep(args.poll_seconds)

    if not _predecessors_complete(args.output_dir):
        print(
            "original matrix is incomplete; refusing to start v2",
            file=sys.stderr,
            flush=True,
        )
        return 2

    output = args.output_dir / "v2.jsonl"
    if not args.resume:
        output.unlink(missing_ok=True)
    completed = _completed(output)
    project_root = Path(__file__).resolve().parents[2]
    python_path = str(Path(__file__).resolve().parents[1])
    environment = dict(os.environ)
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_path if not inherited else f"{python_path}:{inherited}"
    )

    failures = 0
    for index, (dataset, task) in enumerate(CASES, start=1):
        if (dataset, task) in completed:
            print(f"[v2 {index}/10] skip {dataset}/{task}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "relconnector_v2.benchmark",
            "--dataset",
            dataset,
            "--task",
            task,
            "--database-dir",
            str(project_root / "data" / "relbench"),
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
            args.device,
            "--assembly-processes",
            str(args.assembly_processes),
            "--initial-active-processes",
            "1",
            "--ready-queue-mb",
            "16384",
            "--training-order",
            "ready",
            "--output",
            str(output),
        ]
        print(f"[v2 {index}/10] {dataset}/{task}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        completed_process = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
        )
        if completed_process.returncode:
            failures += 1
            _append_error(
                output,
                dataset,
                task,
                completed_process.returncode,
            )
            print(
                f"v2 {dataset}/{task} returned "
                f"{completed_process.returncode}; continuing",
                file=sys.stderr,
                flush=True,
            )
    return int(failures > 0)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _completed(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "ok":
            completed.add((str(payload["dataset"]), str(payload["task"])))
    return completed


def _predecessors_complete(output_dir: Path) -> bool:
    expected = set(CASES)
    return all(
        _completed(output_dir / f"{name}.jsonl") >= expected
        for name in ("batch", "vanilla", "latest")
    )


def _append_error(
    output: Path,
    dataset: str,
    task: str,
    returncode: int,
) -> None:
    payload = {
        "status": "error",
        "dataset": dataset,
        "task": task,
        "returncode": returncode,
    }
    with output.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
