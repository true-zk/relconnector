"""Load one relational learning task from a YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .spec import TaskSpec, TaskType

_TASK_FIELDS = frozenset(
    {
        "name",
        "table_name",
        "task_type",
        "entity_table",
        "entity_column",
        "target_column",
        "time_column",
        "dst_entity_table",
        "dst_entity_column",
        "hidden_columns",
    }
)


def load_task(path: str | Path) -> TaskSpec:
    """Load a task definition; hidden columns must be declared explicitly."""
    source = Path(path)
    values: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Task file {source} must contain a YAML mapping")
    unknown = values.keys() - _TASK_FIELDS
    if unknown:
        raise ValueError(f"Unknown task fields: {', '.join(sorted(map(str, unknown)))}")

    task_type = TaskType(_required_text(values, "task_type"))
    target_column = _optional_text(values, "target_column")
    dst_entity_table = _optional_text(values, "dst_entity_table")
    dst_entity_column = _optional_text(values, "dst_entity_column")
    if task_type is TaskType.RECOMMENDATION:
        if dst_entity_table is None or dst_entity_column is None:
            raise ValueError("Recommendation tasks require both destination fields")
    elif target_column is None:
        raise ValueError("Entity tasks require target_column")
    if "hidden_columns" not in values:
        raise ValueError("Task file must declare hidden_columns, even when empty")

    return TaskSpec(
        name=_required_text(values, "name"),
        table_name=_required_text(values, "table_name"),
        task_type=task_type,
        entity_table=_required_text(values, "entity_table"),
        entity_column=_required_text(values, "entity_column"),
        target_column=target_column,
        time_column=_optional_text(values, "time_column"),
        dst_entity_table=dst_entity_table,
        dst_entity_column=dst_entity_column,
        hidden_columns=_hidden_columns(values["hidden_columns"]),
    )


def _required_text(values: dict[str, Any], field: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Task field {field!r} must be a nonempty string")
    return value


def _optional_text(values: dict[str, Any], field: str) -> str | None:
    value = values.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Task field {field!r} must be a nonempty string or null")
    return value


def _hidden_columns(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise TypeError("hidden_columns must be a list of [table, column] pairs")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(name, str) and name for name in item)
        ):
            raise ValueError("hidden_columns must contain [table, column] pairs")
        pairs.append((item[0], item[1]))
    return tuple(dict.fromkeys(pairs))
