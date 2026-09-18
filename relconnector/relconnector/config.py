"""Configuration for the online relational training implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OnlineTrainingConfig:
    batch_size: int = 512
    num_neighbors: tuple[int, ...] = (128, 128)
    channels: int = 128
    aggr: str = "sum"
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 1
    max_batches: int | None = None
    seed: int = 42
    device: str | None = None
    torch_num_threads: int = 1
    graph_scan_batch_size: int = 1_000_000
    seed_shuffle_block_size: int = 65_536
    feature_block_size: int = 4096
    feature_cache_bytes: int = 4 * 1024 * 1024 * 1024
    executor: Literal["sync", "async"] = "async"
    seed_queue_bytes: int = 64 * 1024 * 1024
    plan_queue_bytes: int = 2 * 1024 * 1024 * 1024
    ready_queue_bytes: int = 8 * 1024 * 1024 * 1024
    feature_window_batches: int = 4
    feature_window_max_batches: int = 32
    feature_window_bytes: int = 512 * 1024 * 1024
    encode_workers: int = 1
    fetched_queue_bytes: int = 2 * 1024 * 1024 * 1024
    feature_policy: Literal["static", "adaptive"] = "adaptive"
    out_channels: int | None = None
    text_batch_size: int = 256
    text_embedding_cache_bytes: int = 512 * 1024 * 1024
    text_cache_admission: Literal["always", "second", "adaptive"] = "adaptive"
    text_execution: Literal["official", "direct"] = "direct"
    encoded_feature_cache_bytes: int = 2 * 1024 * 1024 * 1024
    encoded_cache_admission: Literal["always", "second", "adaptive"] = "adaptive"
    text_model_path: str | None = None
    operation_telemetry: bool = True
    initialization_cache: bool = True
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.channels,
            self.torch_num_threads,
            self.epochs,
            self.graph_scan_batch_size,
            self.seed_shuffle_block_size,
            self.feature_block_size,
            self.seed_queue_bytes,
            self.plan_queue_bytes,
            self.ready_queue_bytes,
            self.feature_window_batches,
            self.feature_window_max_batches,
            self.feature_window_bytes,
            self.encode_workers,
            self.fetched_queue_bytes,
            self.text_batch_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("batch, model, epoch and queue sizes must be positive")
        if not self.num_neighbors or any(value <= 0 for value in self.num_neighbors):
            raise ValueError("num_neighbors must contain positive fanouts")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive")
        if (
            min(
                self.feature_cache_bytes,
                self.text_embedding_cache_bytes,
                self.encoded_feature_cache_bytes,
            )
            < 0
        ):
            raise ValueError("feature cache sizes must be nonnegative")
        if self.out_channels is not None and self.out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if self.executor not in {"sync", "async"}:
            raise ValueError("executor must be sync or async")
        if self.feature_policy not in {"static", "adaptive"}:
            raise ValueError("feature_policy must be static or adaptive")
        if self.feature_window_batches > self.feature_window_max_batches:
            raise ValueError(
                "feature_window_batches cannot exceed feature_window_max_batches"
            )
        if self.text_cache_admission not in {"always", "second", "adaptive"}:
            raise ValueError("invalid text cache admission")
        if self.text_execution not in {"official", "direct"}:
            raise ValueError("invalid text execution")
        if self.encoded_cache_admission not in {"always", "second", "adaptive"}:
            raise ValueError("invalid encoded cache admission")

    @property
    def num_layers(self) -> int:
        return len(self.num_neighbors)
