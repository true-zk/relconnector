"""Convert fetched rows and sampled topology into TensorFrame PyG batches."""

from __future__ import annotations

import torch
from torch_frame import TensorFrame
from torch_frame.data.multi_tensor import _MultiTensor
from torch_geometric.data import HeteroData

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

    def __init__(self, encoder: TensorFrameEncoder) -> None:
        self.encoder = encoder

    def assemble(self, features: FeatureBatch) -> PreparedBatch:
        return self.assemble_many([features])[0]

    def assemble_many(self, features: list[FeatureBatch]) -> list[PreparedBatch]:
        encoded: dict[tuple[str, int], TensorFrame] = {}
        return [self._assemble(item, encoded) for item in features]

    def _assemble(
        self, features: FeatureBatch, encoded: dict[tuple[str, int], TensorFrame]
    ) -> PreparedBatch:
        if isinstance(features, EntityFeatureBatch):
            data = self._assemble_subgraph(features.subgraph, encoded)
            data[features.subgraph.sample.seed_node_type].y = features.target
            return PreparedBatch(
                key=features.key,
                data=data,
                allocated_bytes=_tensor_bytes(data),
            )
        if isinstance(features, RecommendationFeatureBatch):
            batches = (
                self._assemble_subgraph(features.source, encoded),
                self._assemble_subgraph(features.positive, encoded),
                self._assemble_subgraph(features.negative, encoded),
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
    ) -> HeteroData:
        sample = fetched.sample
        data = HeteroData()
        for node_type, node_ids in sample.node_ids.items():
            features = fetched.frames[node_type]
            frame = features.frame
            cache_key = (node_type, id(frame))
            tensor_frame = encoded.get(cache_key)
            if tensor_frame is None:
                tensor_frame = (
                    frame
                    if isinstance(frame, TensorFrame)
                    else self.encoder.encode(node_type, frame)
                )
                encoded[cache_key] = tensor_frame
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
