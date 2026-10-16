"""Training / sampling hyper-parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainingConfig:
    batch_size: int = 512
    num_neighbors: Sequence[int] = field(default_factory=lambda: [128, 128])
    channels: int = 128
    aggr: str = "sum"
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    max_batches: int | None = None
    num_workers: int = 0
    seed: int = 42
    device: str | None = None
    torch_num_threads: int = 1
    text_batch_size: int = 256
    telemetry_interval_s: float = 0.05
    cache_materialization: bool = True
    graph_scan_batch_size: int = 1_000_000
    seed_shuffle_block_size: int = 65_536
    executor: Literal["sync", "async"] = "sync"
    seed_queue_bytes: int = 64 * 1024 * 1024
    plan_queue_bytes: int = 2 * 1024 * 1024 * 1024
    ready_queue_bytes: int = 8 * 1024 * 1024 * 1024

    @property
    def num_layers(self) -> int:
        return len(self.num_neighbors)
