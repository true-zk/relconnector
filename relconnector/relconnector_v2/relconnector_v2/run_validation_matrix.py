"""Run the ten-task v2 matrix, optionally checking for predecessor results."""

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

ENCODED_CACHE_PROFILE_BY_DATASET = {"rel-hm": ("local", 4096)}
# Kept for running experiment supervisors that import this name. Text policy
# is now workload-driven and identical for every dataset.
TEXT_CACHE_PROFILE_BY_DATASET: dict[str, tuple[str, int]] = {}


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
    parser.add_argument("--max-batches", type=int, default=10000)
    parser.add_argument("--validate-categorical-inputs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--revalidate-text-cache", nargs="*", default=[],
        choices=[dataset for dataset, _ in CASES],
        help="On resume, require adaptive-text results for these tasks.",
    )
    parser.add_argument(
        "--independent",
        action="store_true",
        help="Run v2 without requiring predecessor result files in the output directory.",
    )
    parser.add_argument(
        "--short-first",
        action="store_true",
        help="Run shorter tasks first while keeping the same benchmark settings.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.assembly_processes <= 0 or args.max_batches <= 0:
        raise ValueError(
            "poll interval, process count, and max batches must be positive"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_for_pid is not None:
        print(
            f"waiting for original matrix pid {args.wait_for_pid}",
            flush=True,
        )
        while _pid_exists(args.wait_for_pid):
            time.sleep(args.poll_seconds)

    if not args.independent and not _predecessors_complete(args.output_dir):
        print(
            "original matrix is incomplete; refusing to start v2",
            file=sys.stderr,
            flush=True,
        )
        return 2

    output = args.output_dir / "v2.jsonl"
    if not args.resume:
        output.unlink(missing_ok=True)
    completed = _completed(
        output,
        encoded_cache_profile_by_dataset=ENCODED_CACHE_PROFILE_BY_DATASET,
        text_cache_profile_by_dataset={
            dataset: ("adaptive", 512) for dataset in args.revalidate_text_cache
        },
    )
    project_root = Path(__file__).resolve().parents[2]
    python_path = str(Path(__file__).resolve().parents[1])
    environment = dict(os.environ)
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_path if not inherited else f"{python_path}:{inherited}"
    )
    scratch_dir = args.output_dir.resolve() / "tmp"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(scratch_dir)
    environment["SQLITE_TMPDIR"] = str(scratch_dir)

    failures = 0
    cases = CASES
    if args.short_first:
        priority = (
            "rel-f1",
            "rel-trial",
            "rel-event",
            "rel-arxiv",
            "rel-hm",
            "rel-avito",
            "rel-salt",
            "rel-stack",
            "rel-amazon",
            "rel-ratebeer",
        )
        cases = tuple(sorted(CASES, key=lambda case: priority.index(case[0])))
    for index, (dataset, task) in enumerate(cases, start=1):
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
            str(args.max_batches),
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
        scope, cache_mb = ENCODED_CACHE_PROFILE_BY_DATASET.get(
            dataset, ("shared", 2048)
        )
        command.extend(
            ["--encoded-cache-scope", scope, "--encoded-cache-mb", str(cache_mb)]
        )
        command.extend(
            ["--text-cache-scope", "adaptive", "--text-cache-mb", "512"]
        )
        if args.validate_categorical_inputs:
            command.append("--validate-categorical-inputs")
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


def _completed(
    path: Path,
    *,
    encoded_cache_profile_by_dataset: dict[str, tuple[str, int]] | None = None,
    text_cache_profile_by_dataset: dict[str, tuple[str, int]] | None = None,
) -> set[tuple[str, str]]:
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
            dataset = str(payload["dataset"])
            expected_scope, expected_mb = (
                encoded_cache_profile_by_dataset or {}
            ).get(dataset, ("shared", 2048))
            config = payload.get("config") or {}
            actual_scope = config.get("encoded_cache_scope", "shared")
            actual_mb = config.get(
                "encoded_feature_cache_bytes", 2048 * 1024 * 1024
            ) // (1024 * 1024)
            if actual_scope != expected_scope or actual_mb != expected_mb:
                continue
            expected_text = (text_cache_profile_by_dataset or {}).get(dataset)
            if expected_text is not None:
                actual_text = (
                    config.get("text_cache_scope", "shared"),
                    config.get("text_embedding_cache_bytes", 512 * 1024 * 1024)
                    // (1024 * 1024),
                )
                if actual_text != expected_text:
                    continue
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
