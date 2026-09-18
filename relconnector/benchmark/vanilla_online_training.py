"""Benchmark wrapper for the frozen vanilla online implementation."""

from __future__ import annotations

from dataclasses import asdict

from baseline.vanilla_baseline import OnlineRelBenchModel
from baseline.vanilla_baseline.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
    RuntimeComponents,
    SyncExecutor,
)
from baseline.vanilla_baseline.training import OnlineTrainer

from .metadata import experiment_metadata
from .result import OnlineTrainingRunResult
from .telemetry import TelemetryRecorder
from .vanilla_wrappers import (
    MeasuredBatchAssembler,
    MeasuredFeatureFetcher,
    MeasuredSampler,
    MeasuredSeedSource,
    MeasuredTrainer,
)


class VanillaOnlineTrainingBenchmark:
    def __init__(
        self,
        model: OnlineRelBenchModel,
        *,
        telemetry_interval_s: float = 0.05,
    ) -> None:
        self.model = model
        self.telemetry_interval_s = telemetry_interval_s

    def run(self) -> OnlineTrainingRunResult:
        recorder = TelemetryRecorder(sample_interval_s=self.telemetry_interval_s)
        with recorder.activate():
            with recorder.phase("prepare"):
                session = self.model.prepare()
            original = session.components
            measured = RuntimeComponents(
                seeds=MeasuredSeedSource(original.seeds, recorder),
                sampler=MeasuredSampler(original.sampler, recorder),
                fetcher=MeasuredFeatureFetcher(original.fetcher, recorder),
                assembler=MeasuredBatchAssembler(original.assembler, recorder),
                trainer=MeasuredTrainer(original.trainer, recorder),
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
                implementation="vanilla_baseline",
                reader=self.model.reader_kind,
                device=str(original.trainer.device),
                executor=self.model.config.executor,
                feature_encoder="tensorframe-shared-v2",
                feature_schema=session.feature_schema.fingerprint,
                model="heteroencoder-graphsage",
                sampling_policy="shared-seed-pyg-lib-uniform-v4",
                database_path=(self.model.sqlite_dir / f"{self.model.dataset}.sqlite"),
                config=asdict(self.model.config),
            ),
            telemetry=recorder.report(),
            diagnostics={},
        )
