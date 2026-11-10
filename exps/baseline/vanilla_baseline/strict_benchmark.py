"""Strict, serial, uncached online RelBench benchmark.

Each epoch streams seeds; each batch completes sampling, exact SQL feature
reads and assembly, and a synchronized training step before the next batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from .api import OnlineRelBenchModel
from .config import OnlineTrainingConfig


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--model", choices=("graphsage", "relgnn"), default="graphsage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=16)
    args = parser.parse_args(argv)
    neighbors = tuple(int(item) for item in args.num_neighbors.split(","))
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
        text_batch_size=args.text_batch_size,
        feature_cache_bytes=0,
        feature_read_policy="exact",
        executor="sync",
        initialization_cache=False,
        operation_telemetry=True,
    )
    database = args.database_dir / f"{args.dataset}.sqlite"
    stat = database.stat()
    payload = {
        "status": "running",
        "implementation": "strict-vanilla-serial-exact-sql-v1",
        "dataset": args.dataset,
        "task": args.task,
        "database": {"path": str(database.resolve()), "bytes": stat.st_size,
                     "mtime_ns": stat.st_mtime_ns},
        "config": asdict(config),
        "timing_method": "time.perf_counter around each sequential stage; torch.cuda.synchronize(device) before sampling, after assembly, and after train_step; train_step also transfers loss to CPU",
        "memory_method": "resource.RUSAGE_SELF.ru_maxrss for process peak; torch.cuda.max_memory_allocated/reserved for PyTorch allocator peak on selected device",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix": time.time(),
        "epochs": [],
    }
    _save(args.output, payload)
    started = time.perf_counter()
    try:
        if torch.device(args.device).type == "cuda":
            torch.cuda.set_device(torch.device(args.device))
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        model = OnlineRelBenchModel(
            dataset=args.dataset, task=args.task,
            sqlite_dir=args.database_dir, config=config,
        )
        session = model.prepare()
        prepared = time.perf_counter()
        payload["prepare_s"] = prepared - started
        payload["prepare_breakdown_s"] = session.prepare_timings
        payload["initialization_cache"] = session.initialization_cache
        _save(args.output, payload)
        components = session.components
        total_examples = 0
        weighted_loss = 0.0
        for epoch in range(config.epochs):
            _synchronize(args.device)
            epoch_started = time.perf_counter()
            row = {"epoch": epoch, "batches": 0, "examples": 0,
                   "sampling_s": 0.0, "feature_read_assemble_s": 0.0,
                   "training_s": 0.0, "training_loop_s": 0.0}
            for seeds in components.seeds.iter_epoch(epoch):
                _synchronize(args.device)
                t0 = time.perf_counter()
                plan = components.sampler.sample(seeds)
                _synchronize(args.device)
                t1 = time.perf_counter()
                features = components.fetcher.fetch(plan)
                batch = components.assembler.assemble(features)
                _synchronize(args.device)
                t2 = time.perf_counter()
                result = components.trainer.train_step(batch)
                _synchronize(args.device)
                t3 = time.perf_counter()
                row["sampling_s"] += t1 - t0
                row["feature_read_assemble_s"] += t2 - t1
                row["training_s"] += t3 - t2
                row["batches"] += 1
                row["examples"] += result.examples
                total_examples += result.examples
                weighted_loss += result.loss * result.examples
                del seeds, plan, features, batch
                if row["batches"] % args.progress_every == 0:
                    print(f"{args.dataset} epoch {epoch + 1}/{config.epochs} batch {row['batches']}/{config.max_batches}", flush=True)
            _synchronize(args.device)
            row["training_loop_s"] = time.perf_counter() - epoch_started
            payload["epochs"].append(row)
            _save(args.output, payload)
            print(f"{args.dataset} epoch {epoch + 1}/{config.epochs} done: {row['batches']} batches in {row['training_loop_s']:.1f}s", flush=True)
        payload["status"] = "ok"
        payload["completed_batches"] = sum(row["batches"] for row in payload["epochs"])
        payload["completed_examples"] = total_examples
        payload["mean_loss"] = weighted_loss / max(total_examples, 1)
        payload["operation_metrics"] = session.metrics.report()
    except Exception as error:  # noqa: BLE001 - persist the exact failure for the matrix
        payload["status"] = "failed"
        payload["error"] = {"type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc()}
        print(payload["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        _synchronize(args.device)
        payload["end_to_end_s"] = time.perf_counter() - started
        payload["peak_host_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if torch.device(args.device).type == "cuda" and torch.cuda.is_available():
            device = torch.device(args.device)
            payload["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
            payload["peak_gpu_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
        payload["finished_unix"] = time.time()
        _save(args.output, payload)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
