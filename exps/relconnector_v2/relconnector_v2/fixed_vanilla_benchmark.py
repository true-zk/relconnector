"""Run the frozen vanilla online baseline against a v2 fixed replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from baseline.vanilla_baseline import OnlineRelBenchModel, OnlineTrainingConfig
from baseline.vanilla_baseline.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
    RuntimeComponents,
    SyncExecutor,
)
from baseline.vanilla_baseline.sampling import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
)
from baseline.vanilla_baseline.task import (
    BatchKey,
    EntitySeedBatch,
    RecommendationSeedBatch,
)

from . import replay


class VanillaReplaySeeds:
    def __init__(self, directory: Path, records: list[dict[str, Any]]) -> None:
        self.directory = directory
        self.records = records

    def iter_epoch(self, epoch: int):
        for item in self.records:
            payload = torch.load(
                self.directory / item["seed"],
                map_location="cpu",
                weights_only=True,
            )
            kind = payload.pop("kind")
            cls = EntitySeedBatch if kind == "entity" else RecommendationSeedBatch
            yield cls(key=BatchKey(epoch, item["batch"]), **payload)


class VanillaReplaySampler:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def sample(self, seeds):
        payload = torch.load(
            self.directory / f"batch-{seeds.key.batch:06d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if payload["kind"] == "entity":
            return EntitySamplePlan(
                seeds.key,
                SampledSubgraph(**payload["subgraph"]),
                payload["target"],
            )
        return RecommendationSamplePlan(
            seeds.key,
            *[
                SampledSubgraph(**payload[name])
                for name in ("source", "positive", "negative")
            ],
        )


def _database_version(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--executor", choices=("sync", "async"), default="sync"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--model", choices=("graphsage", "relgnn"), default="graphsage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument(
        "--feature-read-policy", choices=("hybrid", "row", "exact"), default="hybrid"
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    neighbors = tuple(int(value) for value in args.num_neighbors.split(","))
    if min(args.epochs, args.max_batches, args.batch_size, args.channels) <= 0:
        parser.error("counts must be positive")

    database = args.database_dir / f"{args.dataset}.sqlite"
    identity = {
        "dataset": args.dataset,
        "task": args.task,
        "database_version": _database_version(database),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_batches": args.max_batches,
        "num_neighbors": list(neighbors),
        "torch_version": str(torch.__version__),
        "sampling_contract": 1,
    }
    manifest = replay.validate(args.replay_dir, identity)
    config = OnlineTrainingConfig(
        batch_size=args.batch_size,
        num_neighbors=neighbors,
        channels=args.channels,
        model_kind=args.model,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
        device=args.device,
        torch_num_threads=args.torch_num_threads,
        feature_cache_bytes=0,
        feature_read_policy=args.feature_read_policy,
        executor=args.executor,
        text_batch_size=args.text_batch_size,
        cache_dir=str(args.cache_dir),
    )
    model = OnlineRelBenchModel(
        dataset=args.dataset,
        task=args.task,
        sqlite_dir=args.database_dir,
        config=config,
    )
    started = time.perf_counter()
    session = model.prepare()
    prepared = time.perf_counter()
    if args.prepare_only:
        payload = {
            "status": "ok",
            "implementation": "vanilla-prepare-only",
            "dataset": args.dataset,
            "task": args.task,
            "fingerprint": manifest["fingerprint"],
            "feature_read_policy": args.feature_read_policy,
            "prepare_s": prepared - started,
            "initialization_cache": session.initialization_cache,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 0
    components = RuntimeComponents(
        seeds=VanillaReplaySeeds(args.replay_dir, manifest["batches"]),
        sampler=VanillaReplaySampler(args.replay_dir),
        fetcher=session.components.fetcher,
        assembler=session.components.assembler,
        trainer=session.components.trainer,
    )
    if args.executor == "sync":
        executor = SyncExecutor()
    else:
        # Keep the same bounded queues as the frozen vanilla async implementation.
        executor = AsyncPipelineExecutor(
            AsyncRuntimeConfig(
                seed_queue_bytes=config.seed_queue_bytes,
                plan_queue_bytes=config.plan_queue_bytes,
                ready_queue_bytes=config.ready_queue_bytes,
            )
        )
    result = executor.run(components, epochs=config.epochs)
    finished = time.perf_counter()
    expected = len(manifest["batches"]) * config.epochs
    if result.steps != expected:
        raise RuntimeError(f"expected {expected} batches, completed {result.steps}")
    payload = {
        "status": "ok",
        "implementation": f"vanilla-fixed-replay-no-cache-{args.executor}",
        "model": args.model,
        "executor": args.executor,
        "feature_read_policy": args.feature_read_policy,
        "dataset": args.dataset,
        "task": args.task,
        "fingerprint": manifest["fingerprint"],
        "epochs": config.epochs,
        "batches_per_epoch": len(manifest["batches"]),
        "completed_batches": result.steps,
        "completed_examples": result.examples,
        "mean_loss": result.mean_loss,
        "prepare_s": prepared - started,
        "training_s": finished - prepared,
        "training_s_per_batch": (finished - prepared) / result.steps,
        "end_to_end_s": finished - started,
        "initialization_cache": session.initialization_cache,
        "operation_metrics": session.metrics.report(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
