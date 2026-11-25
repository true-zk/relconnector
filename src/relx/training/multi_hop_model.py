"""Small heterogeneous message-passing baseline for native multi-hop batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import torch
from torch import nn

from relx.batch import OnlineFeatureStore, PreparedBatch
from relx.core import TableRef
from relx.sampling import EntitySamplePlan, RelationRef

from .model import TableEncoder


class MultiHopEntityModel(nn.Module):
    """Aggregate from the outer sampled hop toward each seed."""

    def __init__(
        self,
        store: OnlineFeatureStore,
        seed_table: TableRef,
        relations: tuple[RelationRef, ...],
        *,
        hops: int,
        channels: int,
        out_channels: int,
        text_buckets: int = 4_096,
    ) -> None:
        super().__init__()
        if min(hops, channels, out_channels, text_buckets) <= 0:
            raise ValueError("Model dimensions and hop count must be positive")
        self.seed_table = seed_table
        self.relations = relations
        self.hops = hops
        self.table_keys = {
            ref: str(index) for index, ref in enumerate(store.prepared.schema.tables)
        }
        self.encoders = nn.ModuleDict({
            self.table_keys[ref]: TableEncoder(
                ref, store.prepared.features.for_table(ref), store,
                channels, text_buckets,
            )
            for ref in store.prepared.schema.tables
            if str(ref) in store.prepared.graph.node_counts
        })
        self.projections = nn.ModuleList(
            nn.ModuleList(nn.Linear(channels, channels) for _ in relations)
            for _ in range(hops)
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(),
            nn.Linear(channels, out_channels),
        )

    def forward(self, batch: PreparedBatch) -> torch.Tensor:
        plan = batch.sample
        if not isinstance(plan, EntitySamplePlan):
            raise TypeError("MultiHopEntityModel requires entity samples")
        graph = plan.subgraph
        hidden: dict[TableRef, torch.Tensor] = {
            ref: self.encoders[self.table_keys[ref]](batch.features, len(ids))
            for ref, ids in graph.node_ids.items()
        }
        for hop in reversed(range(self.hops)):
            hidden = self._message_step(hidden, graph.edge_index,
                                        graph.sampled_edges_per_hop, hop)
        device = next(self.parameters()).device
        seed_ids = torch.as_tensor(graph.seed_local_ids, device=device)
        return self.head(hidden[self.seed_table][seed_ids])

    def _message_step(
        self,
        hidden: dict[TableRef, torch.Tensor],
        edges: Mapping[RelationRef, np.ndarray],
        counts: Mapping[RelationRef, tuple[int, ...]],
        hop: int,
    ) -> dict[TableRef, torch.Tensor]:
        pooled = {ref: torch.zeros_like(values) for ref, values in hidden.items()}
        degree = {
            ref: values.new_zeros((len(values), 1))
            for ref, values in hidden.items()
        }
        for index, relation in enumerate(self.relations):
            per_hop = counts[relation]
            if len(per_hop) != self.hops:
                raise ValueError(f"Relation {relation.name} has wrong hop counts")
            start = sum(per_hop[:hop])
            stop = start + per_hop[hop]
            if start == stop:
                continue
            edge_index = edges[relation]
            source = torch.as_tensor(
                edge_index[0, start:stop], dtype=torch.long,
                device=hidden[relation.source].device,
            )
            target = torch.as_tensor(
                edge_index[1, start:stop], dtype=torch.long,
                device=hidden[relation.target].device,
            )
            projection = cast(nn.ModuleList, self.projections[hop])[index]
            messages = projection(hidden[relation.source][source])
            pooled[relation.target].index_add_(0, target, messages)
            degree[relation.target].index_add_(
                0, target,
                hidden[relation.target].new_ones((len(target), 1)),
            )
        return {
            ref: torch.relu(values + pooled[ref] / degree[ref].clamp_min(1))
            for ref, values in hidden.items()
        }
