"""Multi-hop temporal sampling through pyg-lib's native PyTorch operator."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from relx.core import DatabaseSchema
from relx.task import EntitySeedBatch, RecommendationSeedBatch, SeedBatch

from .contracts import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    RelationRef,
    SampledSubgraph,
    SamplePlan,
)

if TYPE_CHECKING:
    from relx._relx_native import GraphIndex

_RNG_LOCK = Lock()


class PygLibMultiHopSampler:
    """Sample disjoint subgraphs with inclusive edge-time cutoffs."""

    def __init__(
        self,
        graph: GraphIndex,
        schema: DatabaseSchema,
        *,
        fanouts: tuple[int, ...],
        random_seed: int = 42,
    ) -> None:
        import pyg_lib  # noqa: F401  # registers torch.ops.pyg
        import torch

        if not fanouts or any(value < -1 for value in fanouts) or random_seed < 0:
            raise ValueError("fanouts must be nonempty and >= -1; seed >= 0")
        self.graph = graph
        self.fanouts = fanouts
        self.random_seed = random_seed
        self.refs = {str(ref): ref for ref in schema.tables}
        self.node_types = sorted(graph.node_counts)
        self.relations: tuple[RelationRef, ...] = tuple(
            RelationRef(self.refs[source], name, self.refs[target])
            for name in graph.relation_names()
            for source, target in (graph.relation_endpoints(name),)
        )
        self.edge_types = [
            (str(ref.source), ref.name, str(ref.target)) for ref in self.relations
        ]
        self.names = {
            ref: "__".join((str(ref.source), ref.name, str(ref.target)))
            for ref in self.relations
        }
        raw = {ref.name: graph.relation(ref.name) for ref in self.relations}
        self.colptr = {
            self.names[ref]: torch.from_numpy(raw[ref.name]["colptr"])
            for ref in self.relations
        }
        self.row = {
            self.names[ref]: torch.from_numpy(raw[ref.name]["row"])
            for ref in self.relations
        }
        self.edge_time = {
            self.names[ref]: torch.from_numpy(raw[ref.name]["edge_time"])
            for ref in self.relations
        }
        self.neighbors = {self.names[ref]: list(fanouts) for ref in self.relations}

    def sample(self, seeds: SeedBatch) -> SamplePlan:
        import torch

        seed = self.random_seed + seeds.key.epoch * 1_000_003 + seeds.key.batch
        if isinstance(seeds, RecommendationSeedBatch):
            count = self.graph.node_counts[seeds.dst_node_type]
            generator = torch.Generator().manual_seed(seed + 3)
            negatives = torch.randint(
                count, (len(seeds.src_node_ids),), generator=generator
            ).numpy()
            return RecommendationSamplePlan(
                seeds.key,
                self._sample_nodes(
                    seeds.src_node_type, seeds.src_node_ids,
                    seeds.seed_time_us, seed,
                ),
                self._sample_nodes(
                    seeds.dst_node_type, seeds.positive_dst_ids,
                    seeds.seed_time_us, seed + 1,
                ),
                self._sample_nodes(
                    seeds.dst_node_type, negatives,
                    seeds.seed_time_us, seed + 2,
                ),
            )
        if not isinstance(seeds, EntitySeedBatch):
            raise TypeError("Unsupported seed batch")
        subgraph = self._sample_nodes(
            seeds.node_type, seeds.node_ids, seeds.seed_time_us, seed
        )
        return EntitySamplePlan(seeds.key, subgraph, seeds.target, seeds.sample_keys)

    def _sample_nodes(
        self, node_type: str, node_ids: np.ndarray,
        seed_time_us: np.ndarray | None, seed: int,
    ) -> SampledSubgraph:
        import torch

        cutoff = (
            seed_time_us
            if seed_time_us is not None
            else np.full(len(node_ids), np.iinfo(np.int64).max,
                         dtype=np.int64)
        )
        with _RNG_LOCK, torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            row, col, nodes, _, node_counts, edge_counts = (
                torch.ops.pyg.hetero_neighbor_sample(  # pyright: ignore[reportCallIssue]
                    self.node_types, self.edge_types, self.colptr, self.row,
                    {node_type: torch.from_numpy(node_ids)},
                    self.neighbors, None, self.edge_time,
                    {node_type: torch.from_numpy(cutoff)},
                    None, True, False, True, True, "uniform", False,
                )
            )
        sampled_ids = {}
        owners = {}
        per_node_hop = {}
        for name in self.node_types:
            values = nodes[name].numpy()
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("pyg-lib disjoint output needs owner and node ID")
            ref = self.refs[name]
            owners[ref] = np.asarray(values[:, 0], dtype=np.int64).copy()
            sampled_ids[ref] = np.asarray(values[:, 1], dtype=np.int64).copy()
            per_node_hop[ref] = tuple(int(value) for value in node_counts[name])
        edges = {}
        per_edge_hop = {}
        for relation in self.relations:
            name = self.names[relation]
            edges[relation] = np.stack(
                (row[name].numpy(), col[name].numpy())
            ).astype(np.int64, copy=False)
            per_edge_hop[relation] = tuple(int(value) for value in edge_counts[name])
        ref = self.refs[node_type]
        return SampledSubgraph(
            node_ids=sampled_ids, node_seed_index=owners, edge_index=edges,
            seed_table=ref,
            seed_local_ids=np.arange(len(node_ids), dtype=np.int64),
            seed_time_us=seed_time_us,
            sampled_nodes_per_hop=per_node_hop,
            sampled_edges_per_hop=per_edge_hop,
            random_seed=seed,
        )
