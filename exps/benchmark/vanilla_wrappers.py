"""Measurement adapters for the frozen vanilla online contracts."""

from __future__ import annotations

from collections.abc import Iterable

from baseline.vanilla_baseline.features import (
    BatchAssembler,
    FeatureBatch,
    FeatureFetcher,
    PreparedBatch,
)
from baseline.vanilla_baseline.runtime.contracts import (
    Sampler,
    SeedSource,
    Trainer,
    TrainStepResult,
)
from baseline.vanilla_baseline.sampling import SamplePlan
from baseline.vanilla_baseline.task import SeedBatch

from .telemetry import TelemetryRecorder


class MeasuredSeedSource:
    def __init__(self, inner: SeedSource, recorder: TelemetryRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def iter_epoch(self, epoch: int) -> Iterable[SeedBatch]:
        iterator = iter(self.inner.iter_epoch(epoch))
        while True:
            try:
                with self.recorder.operation("seed_read"):
                    batch = next(iterator)
            except StopIteration:
                return
            yield batch


class MeasuredSampler:
    def __init__(self, inner: Sampler, recorder: TelemetryRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def sample(self, seeds: SeedBatch) -> SamplePlan:
        with self.recorder.operation("sampling"):
            return self.inner.sample(seeds)


class MeasuredFeatureFetcher:
    def __init__(self, inner: FeatureFetcher, recorder: TelemetryRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def fetch(self, plan: SamplePlan) -> FeatureBatch:
        with self.recorder.operation("feature_fetch"):
            return self.inner.fetch(plan)


class MeasuredBatchAssembler:
    def __init__(self, inner: BatchAssembler, recorder: TelemetryRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def assemble(self, features: FeatureBatch) -> PreparedBatch:
        with self.recorder.operation("batch_assemble"):
            return self.inner.assemble(features)


class MeasuredTrainer:
    def __init__(self, inner: Trainer, recorder: TelemetryRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        with self.recorder.operation("train_step", synchronize_cuda=True):
            return self.inner.train_step(batch)
