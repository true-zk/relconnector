"""Small model-neutral adapter for a one-hop relational baseline."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from relx.batch import OnlineFeatureStore, PreparedBatch
from relx.core import TableRef
from relx.preprocess.features import FeatureBlock, FeatureSpec
from relx.sampling import EntitySamplePlan, RelationRef


class TableEncoder(nn.Module):
    def __init__(
        self,
        ref: TableRef,
        specs: tuple[FeatureSpec, ...],
        store: OnlineFeatureStore,
        channels: int,
        text_buckets: int,
    ) -> None:
        super().__init__()
        self.ref = ref
        self.specs = specs
        self.channels = channels
        self.text_buckets = text_buckets
        self.text_mode = store.text_mode
        self.layers = nn.ModuleList()
        self.bias = nn.Parameter(torch.zeros(channels))
        for spec in specs:
            kind = spec.semantic_type
            if kind in {"categorical", "multicategorical"}:
                state = store.states[(ref, spec.column)]
                self.layers.append(
                    nn.Embedding(len(state.values["vocab"]) + 1, channels)
                    if kind == "categorical"
                    else nn.EmbeddingBag(
                        len(state.values["vocab"]) + 1,
                        channels,
                        mode="mean",
                        include_last_offset=True,
                    )
                )
            elif kind == "text":
                self.layers.append(
                    nn.Linear(store.text_embedding_dim, channels)
                    if self.text_mode == "glove"
                    else nn.Embedding(text_buckets, channels)
                )
            else:
                width = spec.output.width
                if width is None:
                    raise ValueError(f"Feature {ref}.{spec.column} needs a fixed width")
                self.layers.append(nn.Linear(width, channels))

    def forward(
        self,
        blocks: Mapping[tuple[TableRef, str], FeatureBlock],
        count: int,
    ) -> torch.Tensor:
        device = self.bias.device
        result = self.bias.expand(count, -1)
        if count == 0:
            return result
        for spec, layer in zip(self.specs, self.layers, strict=True):
            block = blocks[(self.ref, spec.column)]
            if spec.semantic_type == "categorical":
                values = torch.as_tensor(block.values, dtype=torch.long, device=device)
                encoded = layer(values)
            elif spec.semantic_type == "multicategorical":
                assert block.offsets is not None
                values = torch.as_tensor(block.values, dtype=torch.long, device=device)
                offsets = torch.as_tensor(block.offsets, dtype=torch.long, device=device)
                encoded = layer(values, offsets)
            elif spec.semantic_type == "text":
                if self.text_mode == "glove":
                    values = torch.as_tensor(
                        block.values, dtype=torch.float32, device=device
                    )
                    encoded = layer(values)
                else:
                    hashed = np.asarray(
                        [_stable_bucket(value, self.text_buckets)
                         for value in block.values], dtype=np.int64,
                    )
                    values = torch.as_tensor(hashed, dtype=torch.long, device=device)
                    encoded = layer(values)
            else:
                values = torch.as_tensor(block.values, dtype=torch.float32, device=device)
                encoded = layer(values.reshape(count, -1))
            result = result + encoded
        return torch.relu(result / max(1, len(self.specs)))


class OneHopEntityModel(nn.Module):
    def __init__(
        self,
        store: OnlineFeatureStore,
        seed_table: TableRef,
        relations: tuple[RelationRef, ...],
        *,
        channels: int,
        out_channels: int,
        text_buckets: int = 4_096,
    ) -> None:
        super().__init__()
        if channels <= 0 or out_channels <= 0 or text_buckets <= 0:
            raise ValueError("Model dimensions must be positive")
        self.seed_table = seed_table
        self.relations = tuple(
            relation for relation in relations if relation.target == seed_table
        )
        self.table_keys = {
            ref: str(index) for index, ref in enumerate(store.prepared.schema.tables)
        }
        self.encoders = nn.ModuleDict(
            {
                self.table_keys[ref]: TableEncoder(
                    ref,
                    store.prepared.features.for_table(ref),
                    store,
                    channels,
                    text_buckets,
                )
                for ref in store.prepared.schema.tables
                if str(ref) in store.prepared.graph.node_counts
            }
        )
        self.relation_layers = nn.ModuleList(
            nn.Linear(channels, channels) for _ in self.relations
        )
        self.head = nn.Sequential(
            nn.Linear(channels * (1 + len(self.relations)), channels),
            nn.ReLU(),
            nn.Linear(channels, out_channels),
        )

    def forward(self, batch: PreparedBatch) -> torch.Tensor:
        plan = batch.sample
        if not isinstance(plan, EntitySamplePlan):
            raise TypeError("OneHopEntityModel requires entity samples")
        sampled = plan.subgraph
        encoded = {
            ref: self.encoders[self.table_keys[ref]](batch.features, len(ids))
            for ref, ids in sampled.node_ids.items()
        }
        device = next(self.parameters()).device
        seeds = torch.as_tensor(
            sampled.seed_local_ids, dtype=torch.long, device=device
        )
        parts = [encoded[self.seed_table][seeds]]
        for relation, projection in zip(
            self.relations, self.relation_layers, strict=True
        ):
            edges = sampled.edge_index[relation]
            pooled = torch.zeros(
                (len(seeds), parts[0].shape[1]), device=device
            )
            counts = torch.zeros((len(seeds), 1), device=device)
            if edges.shape[1]:
                sources = torch.as_tensor(edges[0], dtype=torch.long, device=device)
                owners = torch.as_tensor(
                    sampled.node_seed_index[relation.source][edges[0]],
                    dtype=torch.long,
                    device=device,
                )
                messages = projection(encoded[relation.source][sources])
                pooled.index_add_(0, owners, messages)
                counts.index_add_(
                    0, owners, torch.ones((len(owners), 1), device=device)
                )
            parts.append(pooled / counts.clamp_min(1))
        return self.head(torch.cat(parts, dim=1))


def _stable_bucket(value: object, buckets: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets
