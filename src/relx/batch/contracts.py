"""Interfaces between seeds, sampling, feature fetching, and model adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from relx.core import TableRef
from relx.preprocess.features import FeatureBlock
from relx.sampling import SamplePlan
from relx.task import BatchKey, SeedBatch


@dataclass(frozen=True)
class PreparedBatch:
    """Model-neutral batch; an adapter can convert it to TensorFrame or PyG."""

    key: BatchKey
    sample: SamplePlan
    features: Mapping[tuple[TableRef, str], FeatureBlock]

    def __post_init__(self) -> None:
        if self.key != self.sample.key:
            raise ValueError("Prepared batch key does not match its sample plan")


@dataclass(frozen=True)
class PreparedRecommendationBatch:
    """Three independently sampled feature batches for pairwise ranking."""

    source: PreparedBatch
    positive: PreparedBatch
    negative: PreparedBatch


class SeedSource(Protocol):
    def iter_epoch(self, epoch: int) -> Iterable[SeedBatch]: ...


class GraphSampler(Protocol):
    def sample(self, seeds: SeedBatch) -> SamplePlan: ...


class FeatureFetcher(Protocol):
    def fetch(self, plan: SamplePlan) -> PreparedBatch: ...


class BatchIterator(Protocol):
    def start_epoch(self, epoch: int) -> None: ...

    def next_batch(self) -> PreparedBatch | PreparedRecommendationBatch | None:
        """Return None when the current epoch is exhausted."""
