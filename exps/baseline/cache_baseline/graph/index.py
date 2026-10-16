"""Immutable topology-only graph index."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType


@dataclass(frozen=True)
class GraphIndex:
    """Feature-free heterogeneous CSC topology held in host memory."""

    data: HeteroData
    edge_types: tuple[EdgeType, ...]
    colptr_dict: dict[str, torch.Tensor]
    row_dict: dict[str, torch.Tensor]
    node_count: int
    edge_count: int
    allocated_bytes: int

    @property
    def node_types(self) -> list[str]:
        return self.data.node_types
