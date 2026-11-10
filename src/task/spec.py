"""
Specification for relational table learning tasks.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """The type of the task.

    Attributes:
        REGRESSION: Regression task.
        BINARY_CLASSIFICATION: Binary classification task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"



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

