"""Feature-independent neighbor sampling directly through pyg-lib CSC ops."""

from __future__ import annotations

import torch
from torch_geometric.typing import EdgeType

from baseline.cache_baseline.graph import GraphIndex
from baseline.cache_baseline.randomness import TORCH_RNG_LOCK
from baseline.cache_baseline.task import EntitySeedBatch

from .contracts import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
    SamplePlan,
    SeedBatch,
)


class PygLibNeighborSampler:
    """Sample topology only; feature access is deliberately deferred."""

    def __init__(
        self,
        graph: GraphIndex,
        *,
        num_neighbors: list[int],
        base_seed: int = 42,
    ) -> None:
        if not num_neighbors or any(value <= 0 for value in num_neighbors):
            raise ValueError("num_neighbors must contain positive fanouts")
        self.graph = graph
        self.num_neighbors = {
            _relation_name(edge_type): list(num_neighbors)
            for edge_type in graph.edge_types
        }
        self.base_seed = base_seed
        self._edge_type_by_name = {
            _relation_name(edge_type): edge_type for edge_type in graph.edge_types
        }

    def sample(self, seeds: SeedBatch) -> SamplePlan:
        if isinstance(seeds, EntitySeedBatch):
            subgraph = self._sample_nodes(
                seeds.node_type,
                seeds.node_ids,
                seeds.seed_time,
                random_seed=_derive_seed(
                    self.base_seed, seeds.key.epoch, seeds.key.batch
                ),
            )
            return EntitySamplePlan(seeds.key, subgraph, seeds.target)

        destination_count = int(self.graph.data[seeds.dst_node_type].num_nodes)
        generator = torch.Generator().manual_seed(
            _derive_seed(self.base_seed, seeds.key.epoch, seeds.key.batch, 3)
        )
        negative_dst_ids = torch.randint(
            destination_count,
            (len(seeds.src_node_ids),),
            generator=generator,
        )
        source = self._sample_nodes(
            seeds.src_node_type,
            seeds.src_node_ids,
            seeds.seed_time,
            random_seed=_derive_seed(
                self.base_seed, seeds.key.epoch, seeds.key.batch, 0
            ),
        )
        positive = self._sample_nodes(
            seeds.dst_node_type,
            seeds.positive_dst_ids,
            seeds.seed_time,
            random_seed=_derive_seed(
                self.base_seed, seeds.key.epoch, seeds.key.batch, 1
            ),
        )
        negative = self._sample_nodes(
            seeds.dst_node_type,
            negative_dst_ids,
            seeds.seed_time,
            random_seed=_derive_seed(
                self.base_seed, seeds.key.epoch, seeds.key.batch, 2
            ),
        )
        return RecommendationSamplePlan(seeds.key, source, positive, negative)

    def _sample_nodes(
        self,
        node_type: str,
        node_ids: torch.Tensor,
        seed_time: torch.Tensor | None,
        *,
        random_seed: int,
    ) -> SampledSubgraph:
        seed_time_dict = None if seed_time is None else {node_type: seed_time.cpu()}
        node_time_dict = (
            None
            if seed_time is None
            else {
                sampled_type: store.time
                for sampled_type, store in self.graph.data.node_items()
                if "time" in store
            }
        )
        with TORCH_RNG_LOCK, torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(
                torch.Generator().manual_seed(random_seed).get_state()
            )
            raw = torch.ops.pyg.hetero_neighbor_sample(  # pyright: ignore[reportCallIssue]
                self.graph.node_types,
                list(self.graph.edge_types),
                self.graph.colptr_dict,
                self.graph.row_dict,
                {node_type: node_ids.cpu()},
                self.num_neighbors,
                node_time_dict,
                None,
                seed_time_dict,
                None,
                True,
                False,
                True,
                seed_time is not None,
                "uniform",
                True,
            )
        row, col, sampled_nodes, _, num_nodes, num_edges = raw
        batch = None
        if seed_time is not None:
            unpacked: dict[str, torch.Tensor] = {}
            batch = {}
            for sampled_type, values in sampled_nodes.items():
                values = values.t().contiguous()
                batch[sampled_type] = values[0]
                unpacked[sampled_type] = values[1]
            sampled_nodes = unpacked

        edge_index = {
            self._edge_type_by_name[relation]: torch.stack(
                (row[relation], col[relation])
            )
            for relation in row
        }
        return SampledSubgraph(
            node_ids=sampled_nodes,
            edge_index=edge_index,
            batch=batch,
            node_time={
                sampled_type: self.graph.data[sampled_type].time[sampled_ids]
                for sampled_type, sampled_ids in sampled_nodes.items()
                if "time" in self.graph.data[sampled_type]
            },
            num_sampled_nodes=num_nodes,
            num_sampled_edges={
                self._edge_type_by_name[relation]: values
                for relation, values in num_edges.items()
            },
            seed_node_type=node_type,
            seed_count=len(node_ids),
            seed_time=seed_time,
        )


def _relation_name(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def _derive_seed(*parts: int) -> int:
    value = 0x517CC1B727220A95
    for part in parts:
        value ^= int(part) + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2)
    return value & ((1 << 63) - 1)
