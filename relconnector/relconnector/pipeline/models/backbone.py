"""Shared heterogeneous tabular and graph encoder."""

from __future__ import annotations

import torch
from relbench.modeling.nn import (
    HeteroEncoder,
    HeteroGraphSAGE,
    HeteroTemporalEncoder,
)
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from ..types import ColStats


class GraphSAGEBackbone(torch.nn.Module):
    def __init__(
        self,
        *,
        data: HeteroData,
        col_stats_dict: ColStats,
        channels: int,
        num_layers: int,
        aggr: str,
    ) -> None:
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=data.node_types,
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, batch: HeteroData, seed_node_type: NodeType) -> torch.Tensor:
        x_dict = self.encoder(batch.tf_dict)
        seed_storage = batch[seed_node_type]
        seed_time = getattr(seed_storage, "seed_time", None)
        if seed_time is not None:
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return x_dict[seed_node_type][: int(seed_storage.batch_size)]
