"""TensorFrame GraphSAGE models shared semantically with the eager baseline."""

from __future__ import annotations

import torch
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType

ColumnStats = dict[StatType, object]


class OnlineGraphSAGEBackbone(torch.nn.Module):
    def __init__(
        self,
        *,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        col_names_dict: dict[str, dict[stype, list[str]]],
        col_stats_dict: dict[str, dict[str, ColumnStats]],
        channels: int,
        num_layers: int,
        aggr: str,
    ) -> None:
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict=col_names_dict,
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(node_types, channels)
        self.gnn = HeteroGraphSAGE(
            node_types=node_types,
            edge_types=edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, batch: HeteroData, seed_type: NodeType) -> torch.Tensor:
        x_dict = self.encoder(batch.tf_dict)
        seed_store = batch[seed_type]
        seed_time = getattr(seed_store, "seed_time", None)
        if seed_time is not None:
            relative = self.temporal_encoder(
                seed_time,
                batch.time_dict,
                batch.batch_dict,
            )
            for node_type, encoded_time in relative.items():
                x_dict[node_type] = x_dict[node_type] + encoded_time
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return x_dict[seed_type][: int(seed_store.batch_size)]


class OnlineEntityModel(torch.nn.Module):
    def __init__(
        self,
        *,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        col_names_dict: dict[str, dict[stype, list[str]]],
        col_stats_dict: dict[str, dict[str, ColumnStats]],
        out_channels: int,
        channels: int,
        num_layers: int,
        aggr: str,
        norm: str = "batch_norm",
    ) -> None:
        super().__init__()
        self.backbone = OnlineGraphSAGEBackbone(
            node_types=node_types,
            edge_types=edge_types,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
            channels=channels,
            num_layers=num_layers,
            aggr=aggr,
        )
        self.head = MLP(
            in_channels=channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.backbone.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData, seed_type: NodeType) -> torch.Tensor:
        return self.head(self.backbone(batch, seed_type))


class OnlineLinkModel(torch.nn.Module):
    def __init__(
        self,
        *,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        col_names_dict: dict[str, dict[stype, list[str]]],
        col_stats_dict: dict[str, dict[str, ColumnStats]],
        channels: int,
        num_layers: int,
        aggr: str,
    ) -> None:
        super().__init__()
        self.backbone = OnlineGraphSAGEBackbone(
            node_types=node_types,
            edge_types=edge_types,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
            channels=channels,
            num_layers=num_layers,
            aggr=aggr,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.backbone.reset_parameters()

    def forward(
        self,
        source: HeteroData,
        positive: HeteroData,
        negative: HeteroData,
        src_type: NodeType,
        dst_type: NodeType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.backbone(source, src_type)
        pos = self.backbone(positive, dst_type)
        neg = self.backbone(negative, dst_type)
        return (src * pos).sum(dim=-1), (src * neg).sum(dim=-1)
