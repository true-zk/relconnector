"""Settings consumed independently by connection, preprocessing, and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from relx.conn.factory import ReaderKind
from relx.core import TableRef
from relx.preprocess import SchemaOverrides
from relx.task.spec import TaskSpec


@dataclass(frozen=True)
class DatabaseConfig:
    location: str = field(repr=False)
    reader: ReaderKind = "sqlalchemy"
    schema_name: str | None = None


@dataclass(frozen=True)
class FeatureColumnConfig:
    table: TableRef
    column: str
    semantic_type: str | None = None
    processor: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeaturesConfig:
    columns: tuple[FeatureColumnConfig, ...] = ()
    sample_limit: int = 1_000
    sample_seed: int | None = None


@dataclass(frozen=True)
class GraphConfig:
    """Table selection and the availability time of graph rows."""

    exclude_tables: tuple[TableRef, ...] = ()
    time_columns: tuple[tuple[TableRef, str], ...] = ()
    static_tables: tuple[TableRef, ...] = ()
    timezone: str = "UTC"
    time_boundary: Literal["strict_before", "at_or_before"] = "at_or_before"


@dataclass(frozen=True)
class ModelConfig:
    name: str = "default"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 1024
    epochs: int = 20
    max_batches: int | None = None
    lr: float = 0.001
    seed: int = 42
    feature_workers: int = 2
    feature_native_workers: int = 1
    prefetch_batches: int = 4
    prefetch_bytes: int = 256 * 1024 * 1024
    feature_cache_bytes: int = 64 * 1024 * 1024


@dataclass(frozen=True)
class ReportConfig:
    directory: Path


@dataclass(frozen=True)
class RunConfig:
    database: DatabaseConfig
    task: TaskSpec
    schema: SchemaOverrides
    graph: GraphConfig
    hidden_columns: tuple[tuple[TableRef, str], ...]
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    report: ReportConfig
