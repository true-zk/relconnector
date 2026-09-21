"""Standalone benchmark CLI for the independent v2 process runtime."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

from .config import OnlineTrainingConfig
from .process_api import MultiprocessRelBenchModel


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.resume
        and args.output is not None
        and _already_completed(args.output, args.dataset, args.task)
    ):
        print(f"skip: {args.dataset}/{args.task}", flush=True)
        return 0
    config = OnlineTrainingConfig(
        batch_size=args.batch_size,
        num_neighbors=tuple(int(value) for value in args.num_neighbors.split(",")),
        channels=args.channels,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        feature_cache_bytes=args.feature_cache_mb * 1024 * 1024,
        ready_queue_bytes=args.ready_queue_mb * 1024 * 1024,
        feature_window_batches=args.window_batches,
        feature_window_bytes=args.window_mb * 1024 * 1024,
        text_embedding_cache_bytes=args.text_cache_mb * 1024 * 1024,
        encoded_feature_cache_bytes=args.encoded_cache_mb * 1024 * 1024,
        assembly_processes=args.assembly_processes,
        process_initial_active_workers=(
            args.initial_active_processes
            if args.initial_active_processes is not None
            else args.assembly_processes
        ),
        process_adaptive_workers=not args.no_adaptive_workers,
        process_training_order=args.training_order,
        cross_epoch_prefetch=not args.no_cross_epoch_prefetch,
        process_max_inflight_windows=args.max_inflight_windows,
        executor="process",
    )
    model = MultiprocessRelBenchModel(
        dataset=args.dataset,
        task=args.task,
        reader=args.reader,
        sqlite_dir=args.database_dir,
        config=config,
    )
    started = time.perf_counter()
    result = model.train()
    duration = time.perf_counter() - started
    payload = {
        "status": "ok",
        "dataset": args.dataset,
        "task": args.task,
        "duration_s": duration,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "completed_batches": result.steps,
        "completed_examples": result.examples,
        "mean_loss": result.mean_loss,
        "config": asdict(config),
        "pipeline": result.pipeline,
    }
    encoded = json.dumps(payload, ensure_ascii=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as handle:
            handle.write(encoded + "\n")
        print(
            f"ok: {result.steps} batches in {duration:.2f}s; "
            f"results: {args.output}",
            flush=True,
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "relbench",
    )
    parser.add_argument("--reader", choices=("pandas", "connector-x"), default="pandas")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--assembly-processes", type=int, default=4)
    parser.add_argument("--initial-active-processes", type=int)
    parser.add_argument("--no-adaptive-workers", action="store_true")
    parser.add_argument("--training-order", choices=("strict", "ready"), default="ready")
    parser.add_argument("--window-batches", type=int, default=4)
    parser.add_argument("--window-mb", type=int, default=512)
    parser.add_argument("--max-inflight-windows", type=int, default=8)
    parser.add_argument("--feature-cache-mb", type=int, default=4096)
    parser.add_argument("--ready-queue-mb", type=int, default=16384)
    parser.add_argument("--text-cache-mb", type=int, default=512)
    parser.add_argument("--encoded-cache-mb", type=int, default=2048)
    parser.add_argument("--no-cross-epoch-prefetch", action="store_true")
    return parser


def _already_completed(path: Path, dataset: str, task: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            payload.get("status") == "ok"
            and payload.get("dataset") == dataset
            and payload.get("task") == task
        ):
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
