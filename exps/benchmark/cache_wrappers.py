"""Measurement decorators for online training components."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

from baseline.cache_baseline.features import (
    BatchAssembler,
    FeatureBatch,
    FeatureFetcher,
    PreparedBatch,
)
from baseline.cache_baseline.runtime.contracts import (
    Sampler,
    SeedSource,
    Trainer,
    TrainStepResult,
)
from baseline.cache_baseline.sampling import SamplePlan
from baseline.cache_baseline.task import SeedBatch

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

    def fetch_many(self, plans: list[SamplePlan]) -> list[FeatureBatch]:
        with self.recorder.operation("feature_fetch"):
            method = getattr(self.inner, "fetch_many", None)
            if callable(method):
                return cast(list[FeatureBatch], method(plans))
            return [self.inner.fetch(plan) for plan in plans]


class MeasuredBatchAssembler:
    def __init__(self, inner: BatchAssembler, recorder: TelemetryRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def assemble(self, features: FeatureBatch) -> PreparedBatch:
        with self.recorder.operation("batch_assemble"):
            return self.inner.assemble(features)

    def assemble_many(self, features: list[FeatureBatch]) -> list[PreparedBatch]:
        with self.recorder.operation("batch_assemble"):
            method = getattr(self.inner, "assemble_many", None)
            if callable(method):
                return cast(list[PreparedBatch], method(features))
            return [self.inner.assemble(item) for item in features]


class MeasuredTrainer:
    def __init__(
        self,
        inner: Trainer,
        recorder: TelemetryRecorder,
        on_step: Callable[[PreparedBatch, TrainStepResult], None] | None = None,
    ) -> None:
        self.inner = inner
        self.recorder = recorder
        self.on_step = on_step

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        with self.recorder.operation("train_step", synchronize_cuda=True):
            result = self.inner.train_step(batch)
        if self.on_step is not None:
            self.on_step(batch, result)
        return result
