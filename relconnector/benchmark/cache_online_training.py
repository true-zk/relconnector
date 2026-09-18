"""End-to-end benchmark wrapper for the online training implementation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from baseline.cache_baseline import OnlineRelBenchModel
from baseline.cache_baseline.features import (
    PreparedBatch,
    SqlFeatureFetcher,
    TensorFrameBatchAssembler,
)
from baseline.cache_baseline.features.text import GloveTextEmbedder
from baseline.cache_baseline.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
    RuntimeComponents,
    SyncExecutor,
    TrainStepResult,
)
from baseline.cache_baseline.training import OnlineTrainer

from .cache_wrappers import (
    MeasuredBatchAssembler,
    MeasuredFeatureFetcher,
    MeasuredSampler,
    MeasuredSeedSource,
    MeasuredTrainer,
)
from .metadata import experiment_metadata
from .result import OnlineTrainingRunResult
from .telemetry import TelemetryRecorder


class CacheOnlineTrainingBenchmark:
    def __init__(
        self,
        model: OnlineRelBenchModel,
        *,
        telemetry_interval_s: float = 0.05,
        progress_path: Path | None = None,
        progress_interval_s: float = 30.0,
    ) -> None:
        if progress_interval_s <= 0:
            raise ValueError("progress_interval_s must be positive")
        self.model = model
        self.telemetry_interval_s = telemetry_interval_s
        self.progress_path = progress_path
        self.progress_interval_s = progress_interval_s

    def run(self) -> OnlineTrainingRunResult:
        recorder = TelemetryRecorder(sample_interval_s=self.telemetry_interval_s)
        with recorder.activate():
            with recorder.phase("prepare"):
                session = self.model.prepare()
            if self.progress_path is not None:
                _write_json_atomic(
                    self.progress_path,
                    {
                        "stage": "prepared",
                        "graph_nodes": session.graph.node_count,
                        "graph_edges": session.graph.edge_count,
                        "graph_bytes": session.graph.allocated_bytes,
                    },
                )
            original = session.components
            progress_started = time.monotonic()
            last_progress = 0.0
            completed_steps = 0

            def report_progress(batch: PreparedBatch, result: TrainStepResult) -> None:
                nonlocal last_progress, completed_steps
                completed_steps += 1
                now = time.monotonic()
                if (
                    self.progress_path is None
                    or now - last_progress < self.progress_interval_s
                ):
                    return
                last_progress = now
                _write_progress(
                    self.progress_path,
                    original,
                    batch,
                    result,
                    completed_steps,
                    now - progress_started,
                )

            measured = RuntimeComponents(
                seeds=MeasuredSeedSource(original.seeds, recorder),
                sampler=MeasuredSampler(original.sampler, recorder),
                fetcher=MeasuredFeatureFetcher(original.fetcher, recorder),
                assembler=MeasuredBatchAssembler(original.assembler, recorder),
                trainer=MeasuredTrainer(original.trainer, recorder, report_progress),
            )
            with recorder.phase("train"):
                if self.model.config.executor == "sync":
                    runtime_result = SyncExecutor().run(
                        measured,
                        epochs=self.model.config.epochs,
                    )
                else:
                    runtime_result = AsyncPipelineExecutor(
                        AsyncRuntimeConfig(
                            seed_queue_bytes=self.model.config.seed_queue_bytes,
                            plan_queue_bytes=self.model.config.plan_queue_bytes,
                            ready_queue_bytes=self.model.config.ready_queue_bytes,
                            feature_window_batches=(
                                self.model.config.feature_window_batches
                            ),
                            feature_window_bytes=self.model.config.feature_window_bytes,
                        )
                    ).run(measured, epochs=self.model.config.epochs)
        assert isinstance(original.trainer, OnlineTrainer)
        return OnlineTrainingRunResult(
            dataset=self.model.dataset,
            task=self.model.task_name,
            completed_batches=runtime_result.steps,
            completed_examples=runtime_result.examples,
            loss=runtime_result.mean_loss,
            graph_nodes=session.graph.node_count,
            graph_edges=session.graph.edge_count,
            graph_bytes=session.graph.allocated_bytes,
            experiment=experiment_metadata(
                implementation="cache_baseline",
                reader=self.model.reader_kind,
                device=str(original.trainer.device),
                executor=self.model.config.executor,
                feature_encoder="tensorframe-shared-v2",
                feature_schema=session.feature_schema.fingerprint,
                model="heteroencoder-graphsage",
                sampling_policy="shared-seed-pyg-lib-uniform-v4",
                database_path=self.model.sqlite_dir / f"{self.model.dataset}.sqlite",
                config=asdict(self.model.config),
            ),
            telemetry=recorder.report(),
            diagnostics=_diagnostics(original, runtime_result.pipeline),
        )


def _diagnostics(
    components: RuntimeComponents, pipeline: dict[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {"pipeline": pipeline}
    if isinstance(components.fetcher, SqlFeatureFetcher):
        stats = components.fetcher.stats
        result["feature_fetch"] = {
            **asdict(stats),
            "duplicate_factor": stats.duplicate_factor,
            "sql_amplification": stats.sql_amplification,
            "row_cache_coverage": stats.row_cache_coverage,
        }
        cache = components.fetcher.cache
        result["feature_block_cache"] = {
            "hits": cache.hits,
            "misses": cache.misses,
            "hit_rate": cache.hit_rate,
            "insertions": cache.insertions,
            "evictions": cache.evictions,
            "oversized_rejections": cache.oversized_rejections,
            "current_bytes": cache.current_bytes,
            "peak_bytes": cache.peak_bytes,
        }
    if isinstance(components.assembler, TensorFrameBatchAssembler):
        embedder = components.assembler.encoder.text_embedder
        if isinstance(embedder, GloveTextEmbedder):
            result["text_embedding"] = asdict(embedder.stats)
    return result


def _write_progress(
    path: Path,
    components: RuntimeComponents,
    batch: PreparedBatch,
    result: TrainStepResult,
    completed_steps: int,
    elapsed_s: float,
) -> None:
    payload: dict[str, object] = {
        "stage": "train",
        "completed_batches": completed_steps,
        "last_batch": {"epoch": batch.key.epoch, "batch": batch.key.batch},
        "last_loss": result.loss,
        "elapsed_s": elapsed_s,
    }
    if isinstance(components.fetcher, SqlFeatureFetcher):
        stats = components.fetcher.stats
        payload["feature_fetch"] = {
            **asdict(stats),
            "duplicate_factor": stats.duplicate_factor,
            "sql_amplification": stats.sql_amplification,
            "row_cache_coverage": stats.row_cache_coverage,
        }
    _write_json_atomic(path, payload)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
