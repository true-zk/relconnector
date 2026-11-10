"""Run single-process v2 training against a validated fixed replay."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import replay
from .api import OnlineRelBenchModel
from .config import OnlineTrainingConfig


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
    parser.add_argument("--executor", choices=("sync", "async"), required=True)
    parser.add_argument(
        "--text-execution", choices=("official", "direct"), required=True
    )
    parser.add_argument("--window-batches", type=int, default=1)
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
    args = parser.parse_args(argv)
    neighbors = tuple(int(value) for value in args.num_neighbors.split(","))
    if min(
        args.window_batches,
        args.epochs,
        args.max_batches,
        args.batch_size,
        args.channels,
    ) <= 0:
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
        executor=args.executor,
        feature_window_batches=args.window_batches,
        feature_window_max_batches=args.window_batches,
        feature_policy="static",
        encode_workers=1,
        text_batch_size=args.text_batch_size,
        text_embedding_cache_bytes=0,
        text_cache_scope="local",
        text_execution=args.text_execution,
        disk_text_cache_mode="off",
        encoded_feature_cache_bytes=0,
        cache_strategy="off",
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
    model.session = replace(
        session,
        components=replace(
            session.components,
            seeds=replay.ReplaySeeds(args.replay_dir, manifest["batches"]),
            sampler=replay.ReplaySampler(args.replay_dir),
        ),
    )
    result = model.train()
    finished = time.perf_counter()
    expected = len(manifest["batches"]) * config.epochs
    if result.steps != expected:
        raise RuntimeError(f"expected {expected} batches, completed {result.steps}")
    payload = {
        "status": "ok",
        "implementation": "v2-fixed-replay-no-cache-single-process",
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "fingerprint": manifest["fingerprint"],
        "executor": args.executor,
        "text_execution": args.text_execution,
        "window_batches": args.window_batches,
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
        "pipeline": result.pipeline,
        "config": asdict(config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
