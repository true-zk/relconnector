"""Standalone benchmark CLI for the independent v2 process runtime."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

from . import replay
from .config import OnlineTrainingConfig
from .feature_read_profiles import tuned_feature_read_profile
from .process_api import MultiprocessRelBenchModel
from .training.validation import ValidatedTrainer


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
    read_profile = (
        tuned_feature_read_profile(args.dataset)
        if args.read_profile == "tuned"
        else None
    )
    config = OnlineTrainingConfig(
        batch_size=args.batch_size,
        num_neighbors=tuple(int(value) for value in args.num_neighbors.split(",")),
        channels=args.channels,
        model_kind=args.model,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        feature_cache_bytes=args.feature_cache_mb * 1024 * 1024,
        ready_queue_bytes=args.ready_queue_mb * 1024 * 1024,
        feature_window_batches=(
            args.window_batches
            if args.window_batches is not None
            else read_profile.window_batches
            if read_profile is not None
            else 4
        ),
        feature_cold_window_batches=args.cold_window_batches,
        feature_block_density_threshold=(
            args.density_threshold
            if args.density_threshold is not None
            else read_profile.density_threshold
            if read_profile is not None
            else 0.25
        ),
        feature_policy="static" if read_profile is not None else "adaptive",
        feature_window_bytes=args.window_mb * 1024 * 1024,
        text_embedding_cache_bytes=args.text_cache_mb * 1024 * 1024,
        text_cache_scope=args.text_cache_scope,
        text_batch_size=args.text_batch_size,
        text_execution=args.text_execution,
        disk_text_cache_mode=args.disk_text_cache_mode,
        disk_text_cache_layout=args.disk_text_cache_layout,
        disk_text_block_rows=args.disk_text_block_rows,
        disk_text_block_max_bytes=args.disk_text_block_max_mb * 1024**2,
        node_block_warm_index=args.node_block_warm_index,
        node_block_max_mapped_shards=args.node_block_max_mapped_shards,
        disk_text_cache_min_database_bytes=(
            args.disk_text_cache_min_database_mb * 1024 * 1024
        ),
        disk_text_cache_min_model_inputs=(args.disk_text_cache_min_model_inputs),
        disk_text_cache_max_memory_hit_rate=(args.disk_text_cache_max_memory_hit_rate),
        disk_text_cache_min_model_time_s=(args.disk_text_cache_min_model_time_s),
        encoded_feature_cache_bytes=args.encoded_cache_mb * 1024 * 1024,
        encoded_cache_scope=args.encoded_cache_scope,
        assembly_processes=args.assembly_processes,
        process_initial_active_workers=(
            args.initial_active_processes
            if args.initial_active_processes is not None
            else args.assembly_processes
        ),
        process_adaptive_workers=not args.no_adaptive_workers,
        process_progress_interval_s=args.progress_interval_s,
        process_training_order=args.training_order,
        cross_epoch_prefetch=not args.no_cross_epoch_prefetch,
        process_max_inflight_windows=args.max_inflight_windows,
        cache_dir=(None if args.cache_dir is None else str(args.cache_dir)),
        executor="process",
        cache_strategy=args.cache_strategy,
        cost_disk_cache_bytes=args.cost_disk_cache_mb * 1024**2,
        replay_dir=(
            None
            if args.prepare_replay_only or args.replay_dir is None
            else str(args.replay_dir)
        ),
    )
    model = MultiprocessRelBenchModel(
        dataset=args.dataset,
        task=args.task,
        reader=args.reader,
        sqlite_dir=args.database_dir,
        config=config,
    )
    started = time.perf_counter()
    if args.prepare_only:
        if args.prepare_replay_only:
            parser.error("--prepare-only and --prepare-replay-only are exclusive")
        session = model.prepare()
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "duration_s": time.perf_counter() - started,
                    "initialization_cache": session.initialization_cache,
                }
            )
        )
        return 0
    if args.prepare_replay_only:
        if args.replay_dir is None:
            parser.error("--prepare-replay-only requires --replay-dir")
        session = model.prepare()
        manifest = replay.materialize(
            args.replay_dir,
            session.components.seeds,
            session.components.sampler,
            model.replay_identity(),
        )
        print(
            json.dumps(
                {
                    "status": "replay_prepared",
                    "fingerprint": manifest["fingerprint"],
                    "batches": len(manifest["batches"]),
                    "duration_s": time.perf_counter() - started,
                }
            )
        )
        return 0
    validator = None
    if args.validate_categorical_inputs:
        session = model.prepare()
        validator = ValidatedTrainer(
            session.components.trainer,
            session.feature_schema,
            (args.output.parent if args.output is not None else Path("benchmarks"))
            / "failed-inputs",
        )
        model.session = replace(
            session, components=replace(session.components, trainer=validator)
        )
        print(
            "[v2 validation] checking categorical inputs; failed batches: "
            f"{validator.diagnostic_dir.resolve()}",
            flush=True,
        )
    result = model.train()
    duration = time.perf_counter() - started
    payload = {
        "status": "ok",
        "dataset": args.dataset,
        "task": args.task,
        "read_profile": args.read_profile,
        "duration_s": duration,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "completed_batches": result.steps,
        "completed_examples": result.examples,
        "mean_loss": result.mean_loss,
        "config": asdict(config),
        "effective_config": asdict(model.config),
        "initialization_cache": (
            None if model.session is None else model.session.initialization_cache
        ),
        "pipeline": result.pipeline,
    }
    if validator is not None:
        payload["categorical_input_validation"] = {
            "batches": validator.batches,
            "duration_s": validator.validation_s,
        }
    encoded = json.dumps(payload, ensure_ascii=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as handle:
            handle.write(encoded + "\n")
        print(
            f"ok: {result.steps} batches in {duration:.2f}s; results: {args.output}",
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
    parser.add_argument(
        "--cache-strategy",
        choices=("legacy", "off", "cost", "roi"),
        default="legacy",
    )
    parser.add_argument("--cost-disk-cache-mb", type=int, default=2048)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-replay-only", action="store_true")
    parser.add_argument(
        "--validate-categorical-inputs",
        action="store_true",
        help="Check CPU category indices and save inputs on training failure.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--model", choices=("graphsage", "relgnn"), default="graphsage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--assembly-processes", type=int, default=4)
    parser.add_argument("--initial-active-processes", type=int)
    parser.add_argument("--no-adaptive-workers", action="store_true")
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    parser.add_argument(
        "--training-order", choices=("strict", "ready"), default="ready"
    )
    parser.add_argument("--read-profile", choices=("tuned", "manual"), default="tuned")
    parser.add_argument("--window-batches", type=int)
    parser.add_argument("--cold-window-batches", type=int)
    parser.add_argument("--density-threshold", type=float)
    parser.add_argument("--window-mb", type=int, default=512)
    parser.add_argument("--max-inflight-windows", type=int, default=8)
    parser.add_argument("--feature-cache-mb", type=int, default=4096)
    parser.add_argument("--ready-queue-mb", type=int, default=16384)
    parser.add_argument("--text-cache-mb", type=int, default=512)
    parser.add_argument(
        "--text-cache-scope",
        choices=("adaptive", "shared", "local"),
        default="adaptive",
    )
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument(
        "--text-execution", choices=("official", "direct"), default="direct"
    )
    parser.add_argument(
        "--disk-text-cache-mode",
        choices=("off", "on", "adaptive"),
        default="adaptive",
    )
    parser.add_argument(
        "--disk-text-cache-layout",
        choices=("content", "node_block"),
        default="content",
    )
    parser.add_argument("--disk-text-block-rows", type=int, default=256)
    parser.add_argument("--disk-text-block-max-mb", type=int, default=8192)
    parser.add_argument("--node-block-warm-index", action="store_true")
    parser.add_argument("--node-block-max-mapped-shards", type=int, default=64)
    parser.add_argument("--disk-text-cache-min-database-mb", type=int, default=256)
    parser.add_argument("--disk-text-cache-min-model-inputs", type=int, default=4096)
    parser.add_argument(
        "--disk-text-cache-max-memory-hit-rate", type=float, default=0.10
    )
    parser.add_argument("--disk-text-cache-min-model-time-s", type=float, default=1.0)
    parser.add_argument("--encoded-cache-mb", type=int, default=2048)
    parser.add_argument(
        "--encoded-cache-scope", choices=("shared", "local"), default="shared"
    )
    parser.add_argument("--cache-dir", type=Path)
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
