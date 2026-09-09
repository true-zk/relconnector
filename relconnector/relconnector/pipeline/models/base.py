"""Interface every heterogeneous entity-task model must implement."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


class BaseRelBenchModel(torch.nn.Module):
    def forward(self, batch: HeteroData, entity_table: NodeType) -> torch.Tensor:
        raise NotImplementedError


class BaseLinkRelBenchModel(torch.nn.Module):
    def forward(
        self,
        src_batch: HeteroData,
        pos_dst_batch: HeteroData,
        neg_dst_batch: HeteroData,
        src_type: NodeType,
        dst_type: NodeType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
