"""Core contracts for the RelX relational learning pipeline."""

from .core import ColumnSchema, ForeignKey, TableSchema
from .task import (
    BatchKey,
    EntitySeedBatch,
    RecommendationSeedBatch,
    SeedBatch,
    TaskSpec,
    TaskType,
    load_task,
)

__all__ = [
    "BatchKey",
    "ColumnSchema",
    "EntitySeedBatch",
    "ForeignKey",
    "RecommendationSeedBatch",
    "SeedBatch",
    "TableSchema",
    "TaskSpec",
    "TaskType",
    "load_task",
]
