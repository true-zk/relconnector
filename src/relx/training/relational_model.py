"""RDL GraphSAGE and RelGNN backbones over online relational batches."""

from __future__ import annotations

import torch
from relbench.modeling.nn import HeteroGraphSAGE
from torch import nn
from torch_geometric.typing import EdgeType

from relx.batch import OnlineFeatureStore, PreparedBatch
from relx.core import TableRef
from relx.sampling import EntitySamplePlan, RelationRef

from .model import TableEncoder
from .relgnn_backbone import RelGNNBackbone


class RelationalEntityModel(nn.Module):
    """Share RelX feature encoding across two established GNN backbones."""

    def __init__(
        self,
        store: OnlineFeatureStore,
        seed_table: TableRef,
        relations: tuple[RelationRef, ...],
        *,
        kind: str,
        hops: int,
        channels: int,
        out_channels: int,
        text_buckets: int = 4_096,
    ) -> None:
        super().__init__()
        if kind not in {"rdl", "relgnn"}:
            raise ValueError(f"Unknown relational backbone {kind!r}")
        if min(hops, channels, out_channels, text_buckets) <= 0:
            raise ValueError("Model dimensions and hop count must be positive")
        self.seed_table = seed_table
        self.kind = kind
        refs = tuple(
            ref for ref in store.prepared.schema.tables
            if str(ref) in store.prepared.graph.node_counts
        )
        self.table_keys = {ref: str(index) for index, ref in enumerate(refs)}
        self.node_aliases = {str(ref): f"t{index}" for index, ref in enumerate(refs)}
        self.encoders = nn.ModuleDict({
            self.table_keys[ref]: TableEncoder(
                ref, store.prepared.features.for_table(ref), store,
                channels, text_buckets,
            )
            for ref in refs
        })
        bases = [relation for relation in relations
                 if not relation.name.startswith("rev:")]
        aliases = {
            relation.name: f"f2p_{index}"
            for index, relation in enumerate(bases)
        }
        aliases.update({
            f"rev:{relation.name}": f"rev_f2p_{index}"
            for index, relation in enumerate(bases)
        })
        self.aliases = aliases
        self.edge_types: tuple[EdgeType, ...] = tuple(
            (
                self.node_aliases[str(relation.source)],
                aliases[relation.name],
                self.node_aliases[str(relation.target)],
            )
            for relation in relations
        )
        node_types = list(self.node_aliases.values())
        if kind == "rdl":
            self.backbone: nn.Module = HeteroGraphSAGE(
                node_types, list(self.edge_types), channels,
                aggr="mean", num_layers=hops,
            )
        else:
            self.backbone = RelGNNBackbone(
                node_types, list(self.edge_types), channels,
                num_layers=hops, aggr="mean",
            )
        self.head = nn.Linear(channels, out_channels)

    def forward(self, batch: PreparedBatch) -> torch.Tensor:
        return self.head(self.seed_embeddings(batch))

    def seed_embeddings(self, batch: PreparedBatch) -> torch.Tensor:
        """Return embeddings for the sampled seed nodes before the task head."""
        plan = batch.sample
        if not isinstance(plan, EntitySamplePlan):
            raise TypeError("RelationalEntityModel requires entity samples")
        graph = plan.subgraph
        encoded = {
            self.node_aliases[str(ref)]: self.encoders[self.table_keys[ref]](
                batch.features, len(ids)
            )
            for ref, ids in graph.node_ids.items()
            if len(ids)
        }
        device = next(self.parameters()).device
        edges = {
            (
                self.node_aliases[str(relation.source)],
                self.aliases[relation.name],
                self.node_aliases[str(relation.target)],
            ):
            torch.as_tensor(index, dtype=torch.long, device=device)
            for relation, index in graph.edge_index.items()
            if self.node_aliases[str(relation.source)] in encoded
            and self.node_aliases[str(relation.target)] in encoded
        }
        if edges:
            updated = self.backbone(encoded, edges)
            encoded.update(updated)
        seeds = torch.as_tensor(graph.seed_local_ids, dtype=torch.long, device=device)
        return encoded[self.node_aliases[str(graph.seed_table)]][seeds]


class RelationalLinkModel(nn.Module):
    """Score positive and negative destination seeds with shared graph weights."""

    def __init__(
        self, store: OnlineFeatureStore, source_table: TableRef,
        relations: tuple[RelationRef, ...], *, kind: str, hops: int,
        channels: int, text_buckets: int,
    ) -> None:
        super().__init__()
        self.encoder = RelationalEntityModel(
            store, source_table, relations, kind=kind, hops=hops,
            channels=channels, out_channels=1, text_buckets=text_buckets,
        )

    def forward(
        self, source: PreparedBatch, positive: PreparedBatch,
        negative: PreparedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self.encoder.seed_embeddings(source)
        positive_score = (anchor * self.encoder.seed_embeddings(positive)).sum(dim=1)
        negative_score = (anchor * self.encoder.seed_embeddings(negative)).sum(dim=1)
        return positive_score, negative_score
