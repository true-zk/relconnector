"""Convert fetched rows and sampled topology into TensorFrame PyG batches."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import cast

import pandas as pd
import torch
from torch_frame import TensorFrame, cat
from torch_frame.data.multi_tensor import _MultiTensor
from torch_geometric.data import HeteroData

from .cache import EncodedFeatureCache
from .contracts import (
    EntityFeatureBatch,
    FeatureBatch,
    FetchedSubgraph,
    PreparedBatch,
    RecommendationFeatureBatch,
)
from .schema import TensorFrameEncoder


class TensorFrameBatchAssembler:
    """Apply the same TensorFrame conversion used by the eager baseline."""

    def __init__(
        self,
        encoder: TensorFrameEncoder,
        *,
        encoded_cache_bytes: int = 0,
        encoded_cache_admission: str = "second",
        encoded_cache: EncodedFeatureCache | None = None,
    ) -> None:
        self.encoder = encoder
        self.encoded_cache = encoded_cache or EncodedFeatureCache(
            encoded_cache_bytes,
            admission=encoded_cache_admission,
        )

    def assemble(self, features: FeatureBatch) -> PreparedBatch:
        return self.assemble_many([features])[0]

    def assemble_many(self, features: list[FeatureBatch]) -> list[PreparedBatch]:
        encoded: dict[tuple[str, int], TensorFrame] = {}
        return [self._assemble(item, encoded) for item in features]

    def _assemble(
        self, features: FeatureBatch, encoded: dict[tuple[str, int], TensorFrame]
    ) -> PreparedBatch:
        if isinstance(features, EntityFeatureBatch):
            data = self._assemble_subgraph(
                features.subgraph, encoded, features.key.epoch
            )
            data[features.subgraph.sample.seed_node_type].y = features.target
            return PreparedBatch(
                key=features.key,
                data=data,
                allocated_bytes=_tensor_bytes(data),
            )
        if isinstance(features, RecommendationFeatureBatch):
            batches = (
                self._assemble_subgraph(features.source, encoded, features.key.epoch),
                self._assemble_subgraph(features.positive, encoded, features.key.epoch),
                self._assemble_subgraph(features.negative, encoded, features.key.epoch),
            )
            return PreparedBatch(
                key=features.key,
                data=batches,
                allocated_bytes=sum(_tensor_bytes(batch) for batch in batches),
            )
        raise TypeError(f"Unsupported feature batch: {type(features).__name__}")

    def _assemble_subgraph(
        self,
        fetched: FetchedSubgraph,
        encoded: dict[tuple[str, int], TensorFrame],
        epoch: int,
    ) -> HeteroData:
        sample = fetched.sample
        data = HeteroData()
        for node_type, node_ids in sample.node_ids.items():
            features = fetched.frames[node_type]
            frame = features.frame
            cache_key = (node_type, id(frame))
            tensor_frame = encoded.get(cache_key)
            if tensor_frame is None:
                if isinstance(frame, TensorFrame):
                    tensor_frame = frame
                elif features.unique_ids is None:
                    tensor_frame = self._encode_frame(node_type, frame, epoch)
                else:
                    tensor_frame = self._encode_cached(
                        node_type, features.unique_ids, frame, epoch
                    )
                encoded[cache_key] = tensor_frame
            with self.encoder.metrics.timer("frame_gather", table=node_type):
                data[node_type].tf = tensor_frame[features.inverse]
            data[node_type].n_id = node_ids
            data[node_type].num_nodes = len(node_ids)
            sampled_nodes = sample.num_sampled_nodes.get(node_type)
            if sampled_nodes is not None:
                data[node_type].num_sampled_nodes = sampled_nodes
            if node_type in sample.node_time:
                data[node_type].time = sample.node_time[node_type]
            if sample.batch is not None and node_type in sample.batch:
                data[node_type].batch = sample.batch[node_type]

        for edge_type, edge_index in sample.edge_index.items():
            data[edge_type].edge_index = edge_index
            sampled_edges = sample.num_sampled_edges.get(edge_type)
            if sampled_edges is not None:
                data[edge_type].num_sampled_edges = sampled_edges

        seed_store = data[sample.seed_node_type]
        seed_store.batch_size = sample.seed_count
        if sample.seed_time is not None:
            seed_store.seed_time = sample.seed_time
        return data

    def _encode_cached(
        self,
        table: str,
        node_ids: torch.Tensor,
        frame: pd.DataFrame,
        epoch: int,
    ) -> TensorFrame:
        groups, misses = self.encoded_cache.lookup_many(table, node_ids, epoch=epoch)
        self.encoder.metrics.add(
            "encoded_cache_hits",
            len(node_ids) - len(misses),
            table=table,
        )
        self.encoder.metrics.add("encoded_cache_misses", len(misses), table=table)
        if not groups:
            encoded = self._encode_frame(table, frame, epoch)
            self.encoded_cache.put(table, node_ids, encoded, epoch=epoch)
            return encoded

        pieces: list[TensorFrame] = []
        positions: list[int] = []
        for entry, locations in groups:
            positions.extend(position for position, _ in locations)
            rows = torch.tensor([row for _, row in locations], dtype=torch.long)
            pieces.append(entry.frame[rows])
        if misses:
            missing_frame = frame.iloc[misses].reset_index(drop=True)
            missing = self._encode_frame(table, missing_frame, epoch)
            pieces.append(missing)
            positions.extend(misses)
            miss_index = torch.tensor(misses, dtype=torch.long)
            self.encoded_cache.put(table, node_ids[miss_index], missing, epoch=epoch)
        combined = pieces[0] if len(pieces) == 1 else cat(pieces, dim=0)
        if positions == list(range(len(positions))):
            return combined
        restore = torch.argsort(torch.tensor(positions, dtype=torch.long))
        return combined[restore]

    def _encode_frame(self, table: str, frame: pd.DataFrame, epoch: int) -> TensorFrame:
        text_embedder = getattr(self.encoder, "text_embedder", None)
        context = getattr(text_embedder, "epoch_context", None)
        if not callable(context):
            return self.encoder.encode(table, frame)
        with cast(AbstractContextManager[None], context(epoch)):
            return self.encoder.encode(table, frame)


def _tensor_bytes(data: HeteroData) -> int:
    return sum(_value_bytes(value) for store in data.stores for value in store.values())


def _value_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, TensorFrame):
        return sum(_value_bytes(feature) for feature in value.feat_dict.values())
    if isinstance(value, _MultiTensor):
        return _value_bytes(value.values) + _value_bytes(value.offset)
    if isinstance(value, dict):
        return sum(_value_bytes(item) for item in value.values())
    return 0
