"""Batch identities and seed payloads shared by task and sampler stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(frozen=True, order=True)
class BatchKey:
    epoch: int
    batch: int


@dataclass(frozen=True)
class EntitySeedBatch:
    key: BatchKey
    node_type: str
    node_ids: np.ndarray
    seed_time_us: np.ndarray | None
    target: np.ndarray
    sample_keys: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class RecommendationSeedBatch:
    key: BatchKey
    src_node_type: str
    dst_node_type: str
    src_node_ids: np.ndarray
    positive_dst_ids: np.ndarray
    seed_time_us: np.ndarray | None
    sample_keys: tuple[tuple[object, ...], ...]


SeedBatch: TypeAlias = EntitySeedBatch | RecommendationSeedBatch
