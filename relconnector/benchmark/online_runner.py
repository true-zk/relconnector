"""Run isolated online-training benchmarks across local RelBench tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal, NotRequired, TypeAlias, TypedDict, cast

from baseline.cache_baseline import OnlineTrainingConfig as CacheTrainingConfig
from baseline.vanilla_baseline import (
    OnlineTrainingConfig as VanillaTrainingConfig,
)
from relconnector import OnlineTrainingConfig

from .cases import DEFAULT_DATABASE_DIR, discover_cases, prepare_output
from .process import ProcessMeasurement, run_measured_process
from .result import OnlineTrainingRunData


class ProcessEnvelope(TypedDict, total=False):
    process: ProcessMeasurement


class OnlineSuccess(OnlineTrainingRunData, ProcessEnvelope):
    status: Literal["ok"]


class OnlineError(ProcessEnvelope):
    status: Literal["error"]
    dataset: str
    task: str
    duration_s: float
    error_type: str
    error: str
    traceback: str
    progress: NotRequired[dict[str, object]]


OnlinePayload: TypeAlias = OnlineSuccess | OnlineError


def run_worker(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    try:
        if args.implementation == "latest":
            from relconnector import OnlineRelBenchModel

            from .online_training import OnlineTrainingBenchmark

            model = OnlineRelBenchModel(
                dataset=args.worker_dataset,
                task=args.worker_task,
                reader=args.reader,
                sqlite_dir=args.database_dir,
                config=_latest_config(args),
            )
            result = OnlineTrainingBenchmark(
                model, progress_path=args.worker_progress
            ).run()
        elif args.implementation == "cache":
            from baseline.cache_baseline import OnlineRelBenchModel as CacheModel

            from .cache_online_training import CacheOnlineTrainingBenchmark

            cache_model = CacheModel(
                dataset=args.worker_dataset,
                task=args.worker_task,
                reader=args.reader,
                sqlite_dir=args.database_dir,
                config=CacheTrainingConfig(
                    **{
                        field.name: asdict(_latest_config(args))[field.name]
                        for field in fields(CacheTrainingConfig)
                    }
                ),
            )
            result = CacheOnlineTrainingBenchmark(
                cache_model, progress_path=args.worker_progress
            ).run()
        else:
            from baseline.vanilla_baseline import OnlineRelBenchModel

            from .vanilla_online_training import VanillaOnlineTrainingBenchmark

            vanilla_model = OnlineRelBenchModel(
                dataset=args.worker_dataset,
                task=args.worker_task,
                reader=args.reader,
                sqlite_dir=args.database_dir,
                config=_vanilla_config(args),
            )
            result = VanillaOnlineTrainingBenchmark(vanilla_model).run()
        payload: OnlinePayload = {"status": "ok", **result.to_dict()}
    except Exception as exc:  # noqa: BLE001 - persist task failure details.
        payload = {
            "status": "error",
            "dataset": args.worker_dataset,
            "task": args.worker_task,
            "duration_s": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    _write_json(args.worker_output, payload)
    return 0 if payload["status"] == "ok" else 1


def run_all(args: argparse.Namespace) -> int:
    cases = discover_cases(
        args.database_dir,
        datasets=set(args.dataset),
        tasks=set(args.task),
    )
    if not cases:
        raise RuntimeError("No matching local RelBench tasks found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = prepare_output(
        args.output, resume=args.resume, overwrite=args.overwrite
    )
    cases = [case for case in cases if case not in completed]

    failures = 0
    with tempfile.TemporaryDirectory(
        prefix=".online-benchmark-",
        dir=args.output.parent,
    ) as directory:
        for index, (dataset, task) in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {dataset}/{task}", flush=True)
            result_path = Path(directory) / f"result-{index}.json"
            progress_path = Path(directory) / f"progress-{index}.json"
            command = _worker_command(args, dataset, task, result_path, progress_path)
            process = run_measured_process(command, timeout_s=args.task_timeout)
            try:
                if process.timed_out:
                    payload = _error(
                        dataset,
                        task,
                        "TaskTimeout",
                        f"worker exceeded {args.task_timeout} seconds",
                    )
                    if progress_path.exists():
                        payload["progress"] = cast(
                            dict[str, object],
                            json.loads(progress_path.read_text()),
                        )
                elif result_path.exists():
                    payload = cast(
                        OnlinePayload,
                        json.loads(result_path.read_text()),
                    )
                    if payload["status"] == "ok" and process.returncode != 0:
                        raise ValueError(
                            f"worker published success but exited {process.returncode}"
                        )
                else:
                    payload = _error(
                        dataset,
                        task,
                        "WorkerExit",
                        f"worker exited with code {process.returncode}: {process.stderr}",
                    )
            except (ValueError, OSError) as exc:
                payload = _error(
                    dataset,
                    task,
                    "WorkerProtocolError",
                    str(exc),
                )
            payload["process"] = process.measurement()
            if payload["status"] == "error":
                payload["duration_s"] = process.duration_s
            with args.output.open("a", encoding="utf-8") as output:
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            if payload["status"] == "ok":
                print(
                    f"  ok: {payload['completed_batches']} batches, "
                    f"graph={payload['graph_bytes'] / 2**20:.1f} MiB",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"  error: {payload['error_type']}: {payload['error']}",
                    file=sys.stderr,
                    flush=True,
                )
    return 1 if failures else 0


def _latest_config(args: argparse.Namespace) -> OnlineTrainingConfig:
    return OnlineTrainingConfig(
        batch_size=args.batch_size,
        num_neighbors=tuple(int(value) for value in args.num_neighbors.split(",")),
        channels=args.channels,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        graph_scan_batch_size=args.graph_scan_batch_size,
        seed_shuffle_block_size=args.seed_shuffle_block_size,
        feature_block_size=args.feature_block_size,
        feature_cache_bytes=args.feature_cache_mb * 1024 * 1024,
        executor=args.executor,
        seed_queue_bytes=args.seed_queue_mb * 1024 * 1024,
        plan_queue_bytes=args.plan_queue_mb * 1024 * 1024,
        ready_queue_bytes=args.ready_queue_mb * 1024 * 1024,
        feature_window_batches=args.feature_window_batches,
        feature_window_max_batches=args.feature_window_max_batches,
        feature_window_bytes=args.feature_window_mb * 1024 * 1024,
        encode_workers=args.encode_workers,
        fetched_queue_bytes=args.fetched_queue_mb * 1024 * 1024,
        feature_policy=args.feature_policy,
        text_batch_size=args.text_batch_size,
        text_embedding_cache_bytes=args.text_embedding_cache_mb * 1024 * 1024,
        text_cache_admission=args.text_cache_admission,
        text_execution=args.text_execution,
        encoded_feature_cache_bytes=args.encoded_feature_cache_mb * 1024 * 1024,
        encoded_cache_admission=args.encoded_cache_admission,
        text_model_path=args.text_model_path,
        out_channels=args.out_channels,
        operation_telemetry=not args.no_operation_telemetry,
        initialization_cache=not args.no_initialization_cache,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
    )


def _vanilla_config(args: argparse.Namespace) -> VanillaTrainingConfig:
    return VanillaTrainingConfig(
        batch_size=args.batch_size,
        num_neighbors=tuple(int(value) for value in args.num_neighbors.split(",")),
        channels=args.channels,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        graph_scan_batch_size=args.graph_scan_batch_size,
        seed_shuffle_block_size=args.seed_shuffle_block_size,
        feature_block_size=args.feature_block_size,
        feature_cache_bytes=args.feature_cache_mb * 1024 * 1024,
        executor=args.executor,
        seed_queue_bytes=args.seed_queue_mb * 1024 * 1024,
        plan_queue_bytes=args.plan_queue_mb * 1024 * 1024,
        ready_queue_bytes=args.ready_queue_mb * 1024 * 1024,
        text_batch_size=args.text_batch_size,
        text_model_path=args.text_model_path,
        out_channels=args.out_channels,
    )


def _worker_command(
    args: argparse.Namespace,
    dataset: str,
    task: str,
    output: Path,
    progress: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmark.online_runner",
        "--worker",
        "--worker-dataset",
        dataset,
        "--worker-task",
        task,
        "--worker-output",
        str(output),
        "--worker-progress",
        str(progress),
        "--database-dir",
        str(args.database_dir),
        "--reader",
        args.reader,
        "--implementation",
        args.implementation,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-neighbors",
        args.num_neighbors,
        "--channels",
        str(args.channels),
        "--seed",
        str(args.seed),
        "--torch-num-threads",
        str(args.torch_num_threads),
        "--graph-scan-batch-size",
        str(args.graph_scan_batch_size),
        "--seed-shuffle-block-size",
        str(args.seed_shuffle_block_size),
        "--feature-block-size",
        str(args.feature_block_size),
        "--feature-cache-mb",
        str(args.feature_cache_mb),
        "--executor",
        args.executor,
    ]
    if args.max_batches is not None:
        command.extend(["--max-batches", str(args.max_batches)])
    if args.device is not None:
        command.extend(["--device", args.device])
    for name in (
        "seed_queue_mb",
        "plan_queue_mb",
        "ready_queue_mb",
        "feature_window_batches",
        "feature_window_max_batches",
        "feature_window_mb",
        "encode_workers",
        "fetched_queue_mb",
        "text_batch_size",
        "text_embedding_cache_mb",
        "encoded_feature_cache_mb",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.text_model_path is not None:
        command.extend(["--text-model-path", args.text_model_path])
    command.extend(["--text-cache-admission", args.text_cache_admission])
    command.extend(["--text-execution", args.text_execution])
    command.extend(["--encoded-cache-admission", args.encoded_cache_admission])
    command.extend(["--feature-policy", args.feature_policy])
    if args.no_operation_telemetry:
        command.append("--no-operation-telemetry")
    if args.no_initialization_cache:
        command.append("--no-initialization-cache")
    if args.cache_dir is not None:
        command.extend(["--cache-dir", str(args.cache_dir)])
    if args.out_channels is not None:
        command.extend(["--out-channels", str(args.out_channels)])
    return command


def _write_json(path: Path, payload: OnlinePayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _error(dataset: str, task: str, kind: str, message: str) -> OnlineError:
    return {
        "status": "error",
        "dataset": dataset,
        "task": task,
        "duration_s": 0.0,
        "error_type": kind,
        "error": message,
        "traceback": "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/online-all.jsonl")
    )
    parser.add_argument("--reader", choices=("pandas", "connector-x"), default="pandas")
    parser.add_argument(
        "--implementation", choices=("latest", "vanilla", "cache"), default="latest"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--graph-scan-batch-size", type=int, default=1_000_000)
    parser.add_argument("--seed-shuffle-block-size", type=int, default=65_536)
    parser.add_argument("--feature-block-size", type=int, default=4096)
    parser.add_argument("--feature-cache-mb", type=int, default=4096)
    parser.add_argument("--executor", choices=("sync", "async"), default="async")
    parser.add_argument("--seed-queue-mb", type=int, default=64)
    parser.add_argument("--plan-queue-mb", type=int, default=2048)
    parser.add_argument("--ready-queue-mb", type=int, default=8192)
    parser.add_argument("--feature-window-batches", type=int, default=4)
    parser.add_argument("--feature-window-max-batches", type=int, default=32)
    parser.add_argument("--feature-window-mb", type=int, default=512)
    parser.add_argument("--encode-workers", type=int, default=1)
    parser.add_argument("--fetched-queue-mb", type=int, default=2048)
    parser.add_argument(
        "--feature-policy", choices=("static", "adaptive"), default="adaptive"
    )
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--text-embedding-cache-mb", type=int, default=512)
    parser.add_argument(
        "--text-cache-admission",
        choices=("always", "second", "adaptive"),
        default="adaptive",
    )
    parser.add_argument(
        "--text-execution", choices=("official", "direct"), default="direct"
    )
    parser.add_argument("--encoded-feature-cache-mb", type=int, default=2048)
    parser.add_argument(
        "--encoded-cache-admission",
        choices=("always", "second", "adaptive"),
        default="adaptive",
    )
    parser.add_argument("--text-model-path")
    parser.add_argument("--no-operation-telemetry", action="store_true")
    parser.add_argument("--no-initialization-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--task-timeout", type=float)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset", help=argparse.SUPPRESS)
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-progress", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.implementation in {"latest", "cache"}:
        _latest_config(args)
    else:
        _vanilla_config(args)
    if args.task_timeout is not None and args.task_timeout <= 0:
        raise ValueError("task timeout must be positive")
    if args.worker:
        if (
            args.worker_dataset is None
            or args.worker_task is None
            or args.worker_output is None
        ):
            raise ValueError("worker dataset and task are required")
        return run_worker(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
