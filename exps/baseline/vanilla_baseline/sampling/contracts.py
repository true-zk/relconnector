"""Data contracts between seed scheduling, sampling, and feature fetching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch_geometric.typing import EdgeType, NodeType

from baseline.vanilla_baseline.task import (
    BatchKey,
    EntitySeedBatch,
    RecommendationSeedBatch,
)


@dataclass(frozen=True)
class SampledSubgraph:
    node_ids: dict[NodeType, torch.Tensor]
    edge_index: dict[EdgeType, torch.Tensor]
    batch: dict[NodeType, torch.Tensor] | None
    node_time: dict[NodeType, torch.Tensor]
    num_sampled_nodes: dict[NodeType, list[int]]
    num_sampled_edges: dict[EdgeType, list[int]]
    seed_node_type: NodeType
    seed_count: int
    seed_time: torch.Tensor | None

    @property
    def allocated_bytes(self) -> int:
        tensors = list(self.node_ids.values()) + list(self.edge_index.values())
        tensors.extend(self.node_time.values())
        if self.batch is not None:
            tensors.extend(self.batch.values())
        if self.seed_time is not None:
            tensors.append(self.seed_time)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


@dataclass(frozen=True)
class EntitySamplePlan:
    key: BatchKey
    subgraph: SampledSubgraph
    target: torch.Tensor

    @property
    def allocated_bytes(self) -> int:
        return (
            self.subgraph.allocated_bytes
            + self.target.numel() * self.target.element_size()
        )


@dataclass(frozen=True)
class RecommendationSamplePlan:
    key: BatchKey
    source: SampledSubgraph
    positive: SampledSubgraph
    negative: SampledSubgraph

    @property
    def allocated_bytes(self) -> int:
        return (
            self.source.allocated_bytes
            + self.positive.allocated_bytes
            + self.negative.allocated_bytes
        )


SamplePlan: TypeAlias = EntitySamplePlan | RecommendationSamplePlan
SeedBatch: TypeAlias = EntitySeedBatch | RecommendationSeedBatch
