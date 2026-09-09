"""Run isolated, measured baseline training across local RelBench tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

from ..connector import create_reader
from .config import TrainingConfig
from .facade import RelBenchModel
from .result import TrainingTelemetryReport

DEFAULT_DATABASE_DIR = Path(__file__).resolve().parents[2] / "data" / "relbench"


class BenchmarkPayloadBase(TypedDict):
    dataset: str
    task: str
    model: str
    reader: str


class SuccessPayload(BenchmarkPayloadBase):
    status: Literal["ok"]
    task_type: str
    device: str
    epochs: int
    completed_batches: int
    completed_examples: int
    losses: list[float]
    telemetry: TrainingTelemetryReport


class ErrorPayload(BenchmarkPayloadBase):
    status: Literal["error"]
    duration_s: float
    error_type: str
    error: str
    traceback: str
    worker_returncode: int | None
    worker_stdout: str
    worker_stderr: str


BenchmarkPayload: TypeAlias = SuccessPayload | ErrorPayload


def discover_cases(
    database_dir: Path,
    *,
    datasets: set[str],
    tasks: set[str],
) -> list[tuple[str, str]]:
    """Return sorted dataset/task pairs from local SQLite catalogs."""
    cases: list[tuple[str, str]] = []
    for path in sorted(database_dir.glob("*.sqlite")):
        dataset = path.stem
        if datasets and dataset not in datasets:
            continue
        reader = create_reader("pandas", path)
        for task in sorted(reader.task_metadata()):
            if not tasks or task in tasks:
                cases.append((dataset, task))
    return cases


def run_worker(args: argparse.Namespace) -> int:
    """Run one task and atomically publish its result to the parent."""
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_neighbors=_parse_neighbors(args.num_neighbors),
        channels=args.channels,
        epochs=args.epochs,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        text_batch_size=args.text_batch_size,
        telemetry_interval_s=args.telemetry_interval,
        cache_materialization=args.cache_materialization,
    )
    started = time.perf_counter()
    payload: BenchmarkPayload
    try:
        result = RelBenchModel(
            dataset=args.worker_dataset,
            task=args.worker_task,
            model=args.model,
            reader=args.reader,
            sqlite_dir=args.database_dir,
            text_embedder=None if args.no_text else "glove",
            config=config,
        ).train()
        payload = {"status": "ok", **result.to_dict()}
    except Exception as exc:  # noqa: BLE001 - persist task failures.
        payload = {
            "status": "error",
            "dataset": args.worker_dataset,
            "task": args.worker_task,
            "model": args.model,
            "reader": args.reader,
            "duration_s": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "worker_returncode": None,
            "worker_stdout": "",
            "worker_stderr": "",
        }
    _write_json(args.worker_output, payload)
    return 0 if payload["status"] == "ok" else 1


def run_all(args: argparse.Namespace) -> int:
    """Run every selected task in a fresh Python process."""
    cases = discover_cases(
        args.database_dir,
        datasets=set(args.dataset),
        tasks=set(args.task),
    )
    if not cases:
        raise RuntimeError("No matching local RelBench tasks found")

    output = args.output or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _prepare_output(output, resume=args.resume, overwrite=args.overwrite)
    pending = [case for case in cases if case not in completed]
    print(
        f"Benchmarking {len(pending)} task(s), "
        f"skipping {len(cases) - len(pending)} completed; results: {output}",
        flush=True,
    )

    failures = 0
    with tempfile.TemporaryDirectory(
        prefix=".relconnector-benchmark-",
        dir=output.parent,
    ) as temporary_directory:
        result_dir = Path(temporary_directory)
        for index, (dataset, task) in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {dataset}/{task}", flush=True)
            result_path = result_dir / f"result-{index}.json"
            payload = _run_isolated_case(
                args,
                dataset=dataset,
                task=task,
                result_path=result_path,
            )
            _append_jsonl(output, payload)
            if payload["status"] == "ok":
                print(
                    f"  ok: {payload['completed_batches']} batch(es), "
                    f"peak RSS={_peak_rss(payload):.1f} MiB",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"  error: {payload['error_type']}: {payload['error']}",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        f"Completed {len(pending)} task(s) with {failures} failure(s).",
        flush=True,
    )
    return 1 if failures else 0


def _run_isolated_case(
    args: argparse.Namespace,
    *,
    dataset: str,
    task: str,
    result_path: Path,
) -> BenchmarkPayload:
    command = _worker_command(
        args,
        dataset=dataset,
        task=task,
        result_path=result_path,
    )
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.task_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _parent_error(
            args,
            dataset,
            task,
            "TaskTimeout",
            f"worker exceeded {args.task_timeout} seconds",
            started,
            stdout=_stream_text(exc.stdout),
            stderr=_stream_text(exc.stderr),
        )

    if result_path.exists():
        try:
            return cast(BenchmarkPayload, json.loads(result_path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            return _parent_error(
                args,
                dataset,
                task,
                "WorkerProtocolError",
                f"cannot read worker result: {exc}",
                started,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )

    return _parent_error(
        args,
        dataset,
        task,
        "WorkerExit",
        f"worker exited with code {process.returncode} without a result",
        started,
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def _worker_command(
    args: argparse.Namespace,
    *,
    dataset: str,
    task: str,
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "relconnector.pipeline.benchmark",
        "--worker",
        "--worker-dataset",
        dataset,
        "--worker-task",
        task,
        "--worker-output",
        str(result_path),
        "--database-dir",
        str(args.database_dir),
        "--reader",
        args.reader,
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-neighbors",
        args.num_neighbors,
        "--channels",
        str(args.channels),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--text-batch-size",
        str(args.text_batch_size),
        "--telemetry-interval",
        str(args.telemetry_interval),
    ]
    if args.max_batches is not None:
        command.extend(["--max-batches", str(args.max_batches)])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.no_text:
        command.append("--no-text")
    if args.cache_materialization:
        command.append("--cache-materialization")
    return command


def _parent_error(
    args: argparse.Namespace,
    dataset: str,
    task: str,
    error_type: str,
    error: str,
    started: float,
    *,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> ErrorPayload:
    return {
        "status": "error",
        "dataset": dataset,
        "task": task,
        "model": args.model,
        "reader": args.reader,
        "duration_s": time.perf_counter() - started,
        "error_type": error_type,
        "error": error,
        "traceback": "",
        "worker_returncode": returncode,
        "worker_stdout": stdout[-8_000:],
        "worker_stderr": stderr[-8_000:],
    }


def _prepare_output(
    output: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> set[tuple[str, str]]:
    if overwrite:
        output.unlink(missing_ok=True)
        return set()
    if not output.exists():
        return set()
    if not resume:
        raise FileExistsError(f"{output} already exists; pass --resume or --overwrite")

    completed: set[tuple[str, str]] = set()
    for line_number, line in enumerate(output.read_text().splitlines(), start=1):
        try:
            payload = cast(BenchmarkPayload, json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {output}:{line_number}: {exc}") from exc
        if payload.get("status") == "ok":
            completed.add((str(payload["dataset"]), str(payload["task"])))
    return completed


def _write_json(path: Path, payload: BenchmarkPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: BenchmarkPayload) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _peak_rss(payload: SuccessPayload) -> float:
    return float(payload["telemetry"]["overall"]["peak"]["rss_mb"])


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _parse_neighbors(value: str) -> list[int]:
    try:
        neighbors = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--num-neighbors must be comma-separated integers"
        ) from exc
    if not neighbors or any(item <= 0 for item in neighbors):
        raise argparse.ArgumentTypeError(
            "--num-neighbors values must be positive integers"
        )
    return neighbors


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmarks") / f"baseline-{timestamp}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local SQL read, graph materialization, pyg-lib sampling, "
            "and PyG training. Each task runs in a fresh process."
        )
    )
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reader", choices=("pandas", "connector-x"), default="pandas")
    parser.add_argument("--model", default="graphsage")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--telemetry-interval", type=float, default=0.05)
    parser.add_argument("--task-timeout", type=float)
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--cache-materialization", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset", help=argparse.SUPPRESS)
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.telemetry_interval <= 0:
        raise ValueError("--telemetry-interval must be greater than zero")
    _parse_neighbors(args.num_neighbors)

    if args.worker:
        if (
            not args.worker_dataset
            or not args.worker_task
            or args.worker_output is None
        ):
            raise ValueError("worker mode requires dataset, task, and output")
        return run_worker(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
