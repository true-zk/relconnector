"""Database- and model-independent outputs for future graph samplers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from relx.core import TableRef
from relx.task import BatchKey


@dataclass(frozen=True, order=True)
class RelationRef:
    source: TableRef
    name: str
    target: TableRef


@dataclass(frozen=True)
class SampledSubgraph:
    """Local nodes belong to one seed; shared global nodes may be duplicated."""

    node_ids: Mapping[TableRef, np.ndarray]
    node_seed_index: Mapping[TableRef, np.ndarray]
    edge_index: Mapping[RelationRef, np.ndarray]
    seed_table: TableRef
    seed_local_ids: np.ndarray
    seed_time_us: np.ndarray | None
    sampled_nodes_per_hop: Mapping[TableRef, tuple[int, ...]]
    sampled_edges_per_hop: Mapping[RelationRef, tuple[int, ...]]
    random_seed: int

    def __post_init__(self) -> None:
        """Check array shapes cheaply; call validate() for full content checks."""
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        if self.seed_table not in self.node_ids:
            raise ValueError("seed_table is absent from node_ids")
        if set(self.node_seed_index) != set(self.node_ids):
            raise ValueError("node_seed_index must cover all node types")
        for ref, ids in self.node_ids.items():
            _int64_vector(ids, f"node_ids[{ref}]")
            owners = self.node_seed_index[ref]
            _int64_vector(owners, f"node_seed_index[{ref}]")
            if len(owners) != len(ids):
                raise ValueError(f"node_seed_index[{ref}] must match node_ids")
        _int64_vector(self.seed_local_ids, "seed_local_ids")
        if self.seed_time_us is not None:
            _int64_vector(self.seed_time_us, "seed_time_us")
            if len(self.seed_time_us) != len(self.seed_local_ids):
                raise ValueError("seed_time_us must match the number of seeds")
        for relation, edges in self.edge_index.items():
            if relation.source not in self.node_ids or relation.target not in self.node_ids:
                raise ValueError(f"Relation {relation.name} has missing node type")
            if edges.dtype != np.int64 or edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"edge_index[{relation.name}] must be int64 [2, E]")
        if set(self.sampled_nodes_per_hop) != set(self.node_ids):
            raise ValueError("sampled_nodes_per_hop must cover all node types")
        if set(self.sampled_edges_per_hop) != set(self.edge_index):
            raise ValueError("sampled_edges_per_hop must cover all relations")

    def validate(self) -> None:
        """Check ID ranges and seed ownership when debugging a sampler."""
        for ref, ids in self.node_ids.items():
            if np.any(ids < 0):
                raise ValueError(f"node_ids[{ref}] contains negative IDs")
        if np.any(self.seed_local_ids < 0) or np.any(
            self.seed_local_ids >= len(self.node_ids[self.seed_table])
        ):
            raise ValueError("seed_local_ids is out of range")
        seed_count = len(self.seed_local_ids)
        for ref, owners in self.node_seed_index.items():
            if np.any(owners < 0) or np.any(owners >= seed_count):
                raise ValueError(f"node_seed_index[{ref}] is out of range")
        if not np.array_equal(
            self.node_seed_index[self.seed_table][self.seed_local_ids],
            np.arange(seed_count, dtype=np.int64),
        ):
            raise ValueError("Each seed node must belong to its own seed")
        for relation, edges in self.edge_index.items():
            if np.any(edges[0] < 0) or np.any(
                edges[0] >= len(self.node_ids[relation.source])
            ):
                raise ValueError(f"edge_index[{relation.name}] has invalid source IDs")
            if np.any(edges[1] < 0) or np.any(
                edges[1] >= len(self.node_ids[relation.target])
            ):
                raise ValueError(f"edge_index[{relation.name}] has invalid target IDs")
            if not np.array_equal(
                self.node_seed_index[relation.source][edges[0]],
                self.node_seed_index[relation.target][edges[1]],
            ):
                raise ValueError(f"edge_index[{relation.name}] crosses seed graphs")


@dataclass(frozen=True)
class EntitySamplePlan:
    key: BatchKey
    subgraph: SampledSubgraph
    target: np.ndarray
    sample_keys: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class RecommendationSamplePlan:
    key: BatchKey
    source: SampledSubgraph
    positive: SampledSubgraph
    negative: SampledSubgraph


SamplePlan: TypeAlias = EntitySamplePlan | RecommendationSamplePlan


def _int64_vector(values: np.ndarray, name: str) -> None:
    if values.dtype != np.int64 or values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 array")
