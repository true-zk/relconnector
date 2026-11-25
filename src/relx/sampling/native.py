"""One-hop temporal sampling backed by the threaded native graph index."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from relx.core import DatabaseSchema
from relx.task import EntitySeedBatch, SeedBatch

from .contracts import EntitySamplePlan, RelationRef, SampledSubgraph, SamplePlan

if TYPE_CHECKING:
    from relx._relx_native import GraphIndex


class NativeOneHopSampler:
    """Sample each seed at its own cutoff; native workers are C++ threads."""

    def __init__(
        self,
        graph: GraphIndex,
        schema: DatabaseSchema,
        *,
        fanout: int,
        workers: int = 1,
        random_seed: int = 42,
    ) -> None:
        if fanout < -1 or workers < 1 or random_seed < 0:
            raise ValueError("fanout >= -1, workers >= 1, random_seed >= 0 required")
        self.graph = graph
        self.fanout = fanout
        self.workers = workers
        self.random_seed = random_seed
        refs = {str(ref): ref for ref in schema.tables}
        self.relations = tuple(
            RelationRef(refs[source], name, refs[target])
            for name in graph.relation_names()
            for source, target in (graph.relation_endpoints(name),)
        )
        self.relations_by_name = {relation.name: relation for relation in self.relations}
        self.refs = refs

    def sample(self, seeds: SeedBatch) -> SamplePlan:
        if not isinstance(seeds, EntitySeedBatch):
            raise TypeError("One-hop sampler currently accepts entity seed batches")
        seed_count = len(seeds.node_ids)
        cutoffs = (
            seeds.seed_time_us
            if seeds.seed_time_us is not None
            else np.full(seed_count, np.iinfo(np.int64).max, dtype=np.int64)
        )
        sample_seed = self.seed_for(seeds)
        sampled = self.graph.sample_one_hop(
            seeds.node_type,
            seeds.node_ids.tolist(),
            cutoffs.tolist(),
            self.fanout,
            sample_seed,
            self.workers,
        )
        return self.decode_sampled(seeds, sampled)

    def seed_for(self, seeds: EntitySeedBatch) -> int:
        return self.random_seed + seeds.key.epoch * 1_000_003 + seeds.key.batch

    def decode_sampled(
        self, seeds: EntitySeedBatch, sampled: dict[str, Any]
    ) -> EntitySamplePlan:
        seed_table = self.refs[seeds.node_type]
        seed_count = len(seeds.node_ids)
        sample_seed = self.seed_for(seeds)
        node_ids = {self.refs[name]: ids for name, ids in sampled["node_ids"].items()}
        owners = {
            self.refs[name]: ids for name, ids in sampled["node_seed_index"].items()
        }
        edges = {
            self.relations_by_name[name]: ids
            for name, ids in sampled["edge_index"].items()
        }
        initial = {ref: seed_count if ref == seed_table else 0 for ref in node_ids}
        graph = SampledSubgraph(
            node_ids=node_ids,
            node_seed_index=owners,
            edge_index=edges,
            seed_table=seed_table,
            seed_local_ids=np.arange(seed_count, dtype=np.int64),
            seed_time_us=seeds.seed_time_us,
            sampled_nodes_per_hop={
                ref: (initial[ref], len(ids) - initial[ref])
                for ref, ids in node_ids.items()
            },
            sampled_edges_per_hop={ref: (ids.shape[1],) for ref, ids in edges.items()},
            random_seed=sample_seed,
        )
        return EntitySamplePlan(seeds.key, graph, seeds.target, seeds.sample_keys)


class NativeMultiHopSampler(NativeOneHopSampler):
    """Disjoint temporal neighbor sampling with one fanout per hop."""

    def __init__(
        self,
        graph: GraphIndex,
        schema: DatabaseSchema,
        *,
        fanouts: tuple[int, ...],
        workers: int = 1,
        random_seed: int = 42,
    ) -> None:
        if not fanouts or any(fanout < -1 for fanout in fanouts):
            raise ValueError("fanouts must be nonempty and contain values >= -1")
        super().__init__(
            graph, schema, fanout=fanouts[0], workers=workers,
            random_seed=random_seed,
        )
        self.fanouts = fanouts

    def sample(self, seeds: SeedBatch) -> SamplePlan:
        if not isinstance(seeds, EntitySeedBatch):
            raise TypeError("Multi-hop sampler currently accepts entity seed batches")
        seed_count = len(seeds.node_ids)
        cutoffs = (
            seeds.seed_time_us
            if seeds.seed_time_us is not None
            else np.full(seed_count, np.iinfo(np.int64).max, dtype=np.int64)
        )
        sample_seed = self.seed_for(seeds)
        sampled = self.graph.sample_multi_hop(
            seeds.node_type, seeds.node_ids.tolist(), cutoffs.tolist(),
            list(self.fanouts), sample_seed, self.workers,
        )
        return self.decode_sampled(seeds, sampled)

    def decode_sampled(
        self, seeds: EntitySeedBatch, sampled: dict[str, Any]
    ) -> EntitySamplePlan:
        seed_table = self.refs[seeds.node_type]
        seed_count = len(seeds.node_ids)
        sample_seed = self.seed_for(seeds)
        subgraph = SampledSubgraph(
            node_ids={
                self.refs[name]: ids for name, ids in sampled["node_ids"].items()
            },
            node_seed_index={
                self.refs[name]: ids
                for name, ids in sampled["node_seed_index"].items()
            },
            edge_index={
                self.relations_by_name[name]: ids
                for name, ids in sampled["edge_index"].items()
            },
            seed_table=seed_table,
            seed_local_ids=np.arange(seed_count, dtype=np.int64),
            seed_time_us=seeds.seed_time_us,
            sampled_nodes_per_hop={
                self.refs[name]: tuple(int(value) for value in counts)
                for name, counts in sampled["nodes_per_hop"].items()
            },
            sampled_edges_per_hop={
                self.relations_by_name[name]: tuple(int(value) for value in counts)
                for name, counts in sampled["edges_per_hop"].items()
            },
            random_seed=sample_seed,
        )
        return EntitySamplePlan(seeds.key, subgraph, seeds.target, seeds.sample_keys)
