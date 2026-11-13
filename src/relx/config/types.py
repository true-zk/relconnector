"""Settings consumed independently by connection, preprocessing, and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class ModelConfig:
    name: str = "default"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 1024
    epochs: int = 20
    lr: float = 0.001
    seed: int = 42


@dataclass(frozen=True)
class ReportConfig:
    directory: Path


@dataclass(frozen=True)
class RunConfig:
    database: DatabaseConfig
    task: TaskSpec
    schema: SchemaOverrides
    hidden_columns: tuple[tuple[TableRef, str], ...]
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    report: ReportConfig
