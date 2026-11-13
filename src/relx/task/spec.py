"""Task definitions independent of database storage metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Supported relational table learning tasks."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    RECOMMENDATION = "recommendation"


@dataclass(frozen=True)
class TaskSpec:
    """Specification of a relational table learning task."""

    name: str
    table_name: str
    task_type: TaskType
    entity_table: str
    entity_column: str
    target_column: str | None
    time_column: str | None
    dst_entity_table: str | None
    dst_entity_column: str | None
    hidden_columns: tuple[tuple[str, str], ...]

    @property
    def is_recommendation(self) -> bool:
        return self.task_type is TaskType.RECOMMENDATION
