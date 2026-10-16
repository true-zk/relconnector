"""Measure cache-free feature supply on a fixed replay without GPU training."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from baseline.vanilla_baseline import (
    OnlineRelBenchModel as VanillaModel,
)
from baseline.vanilla_baseline import (
    OnlineTrainingConfig as VanillaConfig,
)
from baseline.vanilla_baseline.features import PreparedBatch as VanillaPreparedBatch
from baseline.vanilla_baseline.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
)
from baseline.vanilla_baseline.runtime import (
    RuntimeComponents as VanillaComponents,
)
from baseline.vanilla_baseline.runtime.contracts import (
    TrainStepResult as VanillaTrainStepResult,
)

from . import replay
from .config import OnlineTrainingConfig
from .features import PreparedBatch
from .fixed_vanilla_benchmark import VanillaReplaySampler, VanillaReplaySeeds
from .process_api import MultiprocessRelBenchModel
from .runtime.contracts import TrainStepResult


class _V2SinkTrainer:
    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        del batch
        return TrainStepResult(loss=0.0, examples=1)


class _VanillaSinkTrainer:
    def train_step(self, batch: VanillaPreparedBatch) -> VanillaTrainStepResult:
        del batch
        return VanillaTrainStepResult(loss=0.0, examples=1)


def _database_version(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def _identity(args: argparse.Namespace, database: Path) -> dict[str, object]:
    import torch

    return {
        "dataset": args.dataset,
        "task": args.task,
        "database_version": _database_version(database),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_batches": args.max_batches,
        "num_neighbors": [int(value) for value in args.num_neighbors.split(",")],
        "torch_version": str(torch.__version__),
        "sampling_contract": 1,
    }


def _run_v2(args: argparse.Namespace, manifest: dict[str, object]):
    config = OnlineTrainingConfig(
        batch_size=args.batch_size,
        num_neighbors=tuple(int(v) for v in args.num_neighbors.split(",")),
        channels=16,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
        device="cpu",
        torch_num_threads=1,
        feature_cache_bytes=0,
        text_embedding_cache_bytes=args.text_cache_mb * 1024**2,
        text_cache_scope="local",
        disk_text_cache_mode=args.disk_text_cache_mode,
        encoded_feature_cache_bytes=0,
        assembly_processes=args.workers,
        process_training_order=args.training_order,
        cross_epoch_prefetch=args.cross_epoch_prefetch,
        process_initial_active_workers=args.workers,
        process_adaptive_workers=False,
        cache_strategy=args.cache_strategy,
        cost_disk_cache_bytes=args.cost_disk_cache_mb * 1024**2,
        replay_dir=str(args.replay_dir),
        cache_dir=str(args.cache_dir),
    )
    model = MultiprocessRelBenchModel(
        dataset=args.dataset,
        task=args.task,
        sqlite_dir=args.database_dir,
        config=config,
    )
    started = time.perf_counter()
    session = model.prepare()
    prepared = time.perf_counter()
    model.session = replace(
        session,
        components=replace(session.components, trainer=_V2SinkTrainer()),
    )
    result = model.train()
    finished = time.perf_counter()
    return result, prepared - started, finished - prepared, result.pipeline


def _run_vanilla(args: argparse.Namespace, manifest: dict[str, object]):
    config = VanillaConfig(
        batch_size=args.batch_size,
        num_neighbors=tuple(int(v) for v in args.num_neighbors.split(",")),
        channels=16,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
        device="cpu",
        torch_num_threads=1,
        feature_cache_bytes=0,
        executor="async",
    )
    model = VanillaModel(
        dataset=args.dataset,
        task=args.task,
        sqlite_dir=args.database_dir,
        config=config,
    )
    started = time.perf_counter()
    session = model.prepare()
    prepared = time.perf_counter()
    components = VanillaComponents(
        seeds=VanillaReplaySeeds(
            args.replay_dir,
            cast(list[dict[str, Any]], manifest["batches"]),
        ),
        sampler=VanillaReplaySampler(args.replay_dir),
        fetcher=session.components.fetcher,
        assembler=session.components.assembler,
        trainer=_VanillaSinkTrainer(),
    )
    result = AsyncPipelineExecutor(
        AsyncRuntimeConfig(
            seed_queue_bytes=config.seed_queue_bytes,
            plan_queue_bytes=config.plan_queue_bytes,
            ready_queue_bytes=config.ready_queue_bytes,
        )
    ).run(components, epochs=config.epochs)
    finished = time.perf_counter()
    return result, prepared - started, finished - prepared, {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("v2", "vanilla"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-order", choices=("strict", "ready"), default="ready"
    )
    parser.add_argument(
        "--cross-epoch-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-strategy", choices=("off", "roi"), default="off"
    )
    parser.add_argument(
        "--disk-text-cache-mode",
        choices=("off", "adaptive", "on"),
        default="off",
    )
    parser.add_argument("--text-cache-mb", type=int, default=0)
    parser.add_argument("--cost-disk-cache-mb", type=int, default=2048)
    args = parser.parse_args(argv)
    if min(args.workers, args.epochs, args.max_batches, args.batch_size) <= 0:
        parser.error("counts must be positive")
    if min(args.text_cache_mb, args.cost_disk_cache_mb) < 0:
        parser.error("cache sizes cannot be negative")
    if args.cache_strategy == "roi" and args.disk_text_cache_mode == "off":
        parser.error("ROI requires a non-off disk text cache mode")
    database = args.database_dir / f"{args.dataset}.sqlite"
    manifest = replay.validate(args.replay_dir, _identity(args, database))
    run = _run_v2 if args.implementation == "v2" else _run_vanilla
    result, prepare_s, supply_s, pipeline = run(args, manifest)
    expected = len(manifest["batches"]) * args.epochs
    if result.steps != expected:
        raise RuntimeError(f"expected {expected} batches, completed {result.steps}")
    payload = {
        "status": "ok",
        "implementation": args.implementation,
        "dataset": args.dataset,
        "task": args.task,
        "workers": args.workers if args.implementation == "v2" else 1,
        "training_order": (
            args.training_order if args.implementation == "v2" else "ready"
        ),
        "cross_epoch_prefetch": (
            args.cross_epoch_prefetch if args.implementation == "v2" else True
        ),
        "cache_strategy": (
            args.cache_strategy if args.implementation == "v2" else "off"
        ),
        "fingerprint": manifest["fingerprint"],
        "epochs": args.epochs,
        "completed_batches": result.steps,
        "prepare_s": prepare_s,
        "feature_supply_s": supply_s,
        "feature_supply_s_per_batch": supply_s / result.steps,
        "batches_per_s": result.steps / supply_s,
        "pipeline": pipeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
