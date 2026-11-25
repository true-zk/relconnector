"""Core contracts for the RelX relational learning pipeline."""

from .api import PreparationReport, PreparedDataset, prepare, run
from .config import RunConfig, load_config
from .conn import connect
from .core import ColumnSchema, DatabaseSchema, ForeignKey, TableRef, TableSchema
from .preprocess import SchemaOverrides, discover_schema
from .preprocess.features import FeaturePlan, FeatureProcessor
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
    "FeaturePlan",
    "FeatureProcessor",
    "ForeignKey",
    "PreparationReport",
    "PreparedDataset",
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
    "prepare",
    "run",
    "validate_task",
]
