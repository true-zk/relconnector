"""Batch identities and seed payloads shared by task and sampler stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, order=True)
class BatchKey:
    epoch: int
    batch: int


@dataclass(frozen=True)
class EntitySeedBatch:
    key: BatchKey
    node_type: str
    node_ids: torch.Tensor
    seed_time: torch.Tensor | None
    target: torch.Tensor


@dataclass(frozen=True)
class RecommendationSeedBatch:
    key: BatchKey
    src_node_type: str
    dst_node_type: str
    src_node_ids: torch.Tensor
    positive_dst_ids: torch.Tensor
    seed_time: torch.Tensor | None


SeedBatch: TypeAlias = EntitySeedBatch | RecommendationSeedBatch
