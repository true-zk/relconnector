"""Task specifications, validation, and seed batch contracts."""

from .seeds import BatchKey, EntitySeedBatch, RecommendationSeedBatch, SeedBatch
from .source import SqlEntitySeedSource, SqlRecommendationSeedSource
from .spec import ColumnSplit, DefaultSplit, SqlQuerySplit, TaskSpec, TaskType
from .validation import TaskDataReport, ValidatedTask, validate_task

__all__ = [
    "BatchKey",
    "ColumnSplit",
    "DefaultSplit",
    "EntitySeedBatch",
    "RecommendationSeedBatch",
    "SeedBatch",
    "SqlEntitySeedSource",
    "SqlQuerySplit",
    "SqlRecommendationSeedSource",
    "TaskDataReport",
    "TaskSpec",
    "TaskType",
    "ValidatedTask",
    "validate_task",
]
