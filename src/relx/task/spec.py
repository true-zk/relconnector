"""Typed learning-task and sample-split definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from relx.core import TableRef


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    RECOMMENDATION = "recommendation"


@dataclass(frozen=True)
class ColumnSplit:
    column: str
    train: str | int
    validation: str | int
    test: str | int


@dataclass(frozen=True)
class SqlQuerySplit:
    test: str
    train: str | None = None
    validation: str | None = None
    validation_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class DefaultSplit:
    validation_fraction: float = 0.2
    seed: int = 42


SplitSpec = ColumnSplit | SqlQuerySplit | DefaultSplit


@dataclass(frozen=True)
class TaskSpec:
    name: str
    table: TableRef
    task_type: TaskType
    entity_column: str
    label_column: str
    time_column: str | None = None
    split: SplitSpec = DefaultSplit()
    destination_table: TableRef | None = None
    destination_key: str | None = None
    entity_table: TableRef | None = None

    @property
    def is_recommendation(self) -> bool:
        return self.task_type is TaskType.RECOMMENDATION
