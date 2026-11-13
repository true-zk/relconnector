"""Core contracts for the RelX relational learning pipeline."""

from .config import RunConfig, load_config
from .conn import connect
from .core import ColumnSchema, DatabaseSchema, ForeignKey, TableRef, TableSchema
from .preprocess import SchemaOverrides, discover_schema
from .task import (
    BatchKey,
    EntitySeedBatch,
    RecommendationSeedBatch,
    SeedBatch,
    TaskSpec,
    TaskType,
    ValidatedTask,
    validate_task,
)

__all__ = [
    "BatchKey",
    "ColumnSchema",
    "DatabaseSchema",
    "EntitySeedBatch",
    "ForeignKey",
    "RecommendationSeedBatch",
    "RunConfig",
    "SchemaOverrides",
    "SeedBatch",
    "TableRef",
    "TableSchema",
    "TaskSpec",
    "TaskType",
    "ValidatedTask",
    "connect",
    "discover_schema",
    "load_config",
    "validate_task",
]
