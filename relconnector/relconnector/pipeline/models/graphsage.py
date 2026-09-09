"""GraphSAGE heads for entity and recommendation tasks."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from ..types import ColStats
from .backbone import GraphSAGEBackbone
from .base import BaseLinkRelBenchModel, BaseRelBenchModel


class GraphSAGEModel(BaseRelBenchModel):
    def __init__(
        self,
        *,
        data: HeteroData,
        col_stats_dict: ColStats,
        out_channels: int,
        channels: int = 128,
        num_layers: int = 2,
        aggr: str = "sum",
        norm: str = "batch_norm",
    ) -> None:
        super().__init__()
        self.backbone = GraphSAGEBackbone(
            data=data,
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

    def forward(self, batch: HeteroData, entity_table: NodeType) -> torch.Tensor:
        return self.head(self.backbone(batch, entity_table))


class LinkGraphSAGEModel(BaseLinkRelBenchModel):
    def __init__(
        self,
        *,
        data: HeteroData,
        col_stats_dict: ColStats,
        channels: int = 128,
        num_layers: int = 2,
        aggr: str = "sum",
        **_: object,
    ) -> None:
        super().__init__()
        self.backbone = GraphSAGEBackbone(
            data=data,
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
        src_batch: HeteroData,
        pos_dst_batch: HeteroData,
        neg_dst_batch: HeteroData,
        src_type: NodeType,
        dst_type: NodeType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.backbone(src_batch, src_type)
        pos_dst = self.backbone(pos_dst_batch, dst_type)
        neg_dst = self.backbone(neg_dst_batch, dst_type)
        return (src * pos_dst).sum(dim=-1), (src * neg_dst).sum(dim=-1)
