"""Task metadata used by the online training path."""

from __future__ import annotations

from dataclasses import dataclass

from relbench.base import TaskType

from relbench_compat.tasks import hidden_columns
from relconnector.connector.catalog import TaskMetadata


@dataclass(frozen=True)
class TaskSpec:
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
        return self.task_type == TaskType.RECOMMENDATION

    @classmethod
    def from_metadata(cls, metadata: TaskMetadata) -> TaskSpec:
        task_type = TaskType(_required(metadata.task_type, "task_type"))
        destination_table = _optional_string(metadata.extra.get("dst_entity_table"))
        destination_column = _optional_string(metadata.extra.get("dst_entity_column"))
        return cls(
            name=metadata.name,
            table_name=metadata.table_name,
            task_type=task_type,
            entity_table=_required(metadata.entity_table, "entity_table"),
            entity_column=_required(metadata.entity_column, "entity_column"),
            target_column=_optional_string(metadata.target_column),
            time_column=_optional_string(metadata.time_column),
            dst_entity_table=destination_table,
            dst_entity_column=destination_column,
            hidden_columns=hidden_columns(
                metadata.extra,
                name=metadata.name,
                entity_table=metadata.entity_table,
                target_column=metadata.target_column,
            ),
        )


def _required(value: object, field: str) -> str:
    if value is None or value == "":
        raise ValueError(f"Task catalog is missing required field {field!r}")
    return str(value)


def _optional_string(value: object) -> str | None:
    return None if value is None or value == "" else str(value)
