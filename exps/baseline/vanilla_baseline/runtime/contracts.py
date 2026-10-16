"""Execution contracts shared by synchronous and asynchronous runtimes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from baseline.vanilla_baseline.features import (
    BatchAssembler,
    FeatureFetcher,
    PreparedBatch,
)
from baseline.vanilla_baseline.sampling import SamplePlan
from baseline.vanilla_baseline.task import SeedBatch


class SeedSource(Protocol):
    def iter_epoch(self, epoch: int) -> Iterable[SeedBatch]: ...


class Sampler(Protocol):
    def sample(self, seeds: SeedBatch) -> SamplePlan: ...


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    examples: int


class Trainer(Protocol):
    def train_step(self, batch: PreparedBatch) -> TrainStepResult: ...


@dataclass(frozen=True)
class RuntimeComponents:
    seeds: SeedSource
    sampler: Sampler
    fetcher: FeatureFetcher
    assembler: BatchAssembler
    trainer: Trainer


@dataclass(frozen=True)
class RuntimeResult:
    steps: int
    examples: int
    mean_loss: float
