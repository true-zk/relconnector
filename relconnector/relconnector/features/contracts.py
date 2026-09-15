"""Contracts for online feature retrieval and batch preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import pandas as pd
import torch
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData

from relconnector.sampling import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
)
from relconnector.task import BatchKey

FeatureFrame: TypeAlias = pd.DataFrame | TensorFrame


@dataclass(frozen=True)
class FetchedSubgraph:
    sample: SampledSubgraph
    frames: dict[str, FeatureFrame]


@dataclass(frozen=True)
class EntityFeatureBatch:
    key: BatchKey
    subgraph: FetchedSubgraph
    target: torch.Tensor


@dataclass(frozen=True)
class RecommendationFeatureBatch:
    key: BatchKey
    source: FetchedSubgraph
    positive: FetchedSubgraph
    negative: FetchedSubgraph


FeatureBatch: TypeAlias = EntityFeatureBatch | RecommendationFeatureBatch
SamplePlan: TypeAlias = EntitySamplePlan | RecommendationSamplePlan


@dataclass(frozen=True)
class PreparedBatch:
    key: BatchKey
    data: HeteroData | tuple[HeteroData, HeteroData, HeteroData]
    allocated_bytes: int


class FeatureFetcher(Protocol):
    def fetch(self, plan: SamplePlan) -> FeatureBatch: ...


class BatchAssembler(Protocol):
    def assemble(self, features: FeatureBatch) -> PreparedBatch: ...
