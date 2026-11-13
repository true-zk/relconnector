"""Resolve task references against a discovered relational schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from relx.core import DatabaseSchema, TableRef, TableSchema

from .spec import ColumnSplit, TaskSpec, TaskType

if TYPE_CHECKING:
    from relx.conn import BaseDatabaseReader


@dataclass(frozen=True)
class TaskDataReport:
    train_rows: int
    validation_rows: int
    test_rows: int
    test_labels_available: bool


@dataclass(frozen=True)
class ValidatedTask:
    spec: TaskSpec
    entity_table: TableRef
    entity_key: tuple[str, ...]
    sample_key: tuple[str, ...]
    destination_table: TableRef | None = None
    destination_key: tuple[str, ...] | None = None
    hidden_columns: tuple[tuple[TableRef, str], ...] = ()
    data_report: TaskDataReport | None = None
    split_by_key: dict[tuple[object, ...], str] | None = None


def validate_task(
    task: TaskSpec,
    schema: DatabaseSchema,
    *,
    hidden_columns: tuple[tuple[TableRef, str], ...] = (),
    reader: BaseDatabaseReader | None = None,
    batch_size: int = 10_000,
) -> ValidatedTask:
    """Check structure and, with a reader, sample values and split membership."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sample_table = _table(schema, task.table)
    required = [task.entity_column, task.label_column]
    if task.time_column is not None:
        required.append(task.time_column)
    if isinstance(task.split, ColumnSplit):
        required.append(task.split.column)
    if len(set(required)) != len(required):
        raise ValueError("Task entity, label, time, and split columns must be distinct")
    _columns(sample_table, required)
    sample_key = sample_table.identity_key
    if sample_key is None:
        raise ValueError(
            f"Task table {task.table} needs a primary or schema.identity_key"
        )
    _columns(sample_table, sample_key)

    matches = [
        key
        for key in sample_table.foreign_keys
        if key.source_columns == (task.entity_column,)
        and (task.entity_table is None or key.target_table == task.entity_table)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{task.table}.{task.entity_column} must have exactly one foreign key; "
            "add or disambiguate it in schema.foreign_keys"
        )
    entity_fk = matches[0]
    entity_table = _table(schema, entity_fk.target_table)
    if entity_fk.target_columns != entity_table.identity_key:
        raise ValueError(
            f"{task.table}.{task.entity_column} must reference the identity key "
            f"of {entity_table.ref}"
        )

    destination = None
    destination_key = None
    if task.task_type is TaskType.RECOMMENDATION:
        if task.destination_table is None or task.destination_key is None:
            raise ValueError("Recommendation task requires destination table and key")
        destination = _table(schema, task.destination_table)
        destination_key = (task.destination_key,)
        if destination.identity_key != destination_key:
            raise ValueError(
                f"task.destination.key must be the identity key of {destination.ref}"
            )

    for ref, column in hidden_columns:
        _columns(_table(schema, ref), [column])

    automatic_hidden = [(task.table, task.label_column)]
    if isinstance(task.split, ColumnSplit):
        automatic_hidden.append((task.table, task.split.column))
    if task.time_column is not None:
        automatic_hidden.append((task.table, task.time_column))

    validated = ValidatedTask(
        spec=task,
        entity_table=entity_table.ref,
        entity_key=entity_fk.target_columns,
        sample_key=sample_key,
        destination_table=destination.ref if destination else None,
        destination_key=destination_key,
        hidden_columns=tuple(dict.fromkeys((*automatic_hidden, *hidden_columns))),
    )
    if reader is None:
        return validated
    from .data_checks import check_task_data

    return check_task_data(validated, reader, batch_size=batch_size)


def _table(schema: DatabaseSchema, ref: TableRef) -> TableSchema:
    try:
        return schema.tables[ref]
    except KeyError as exc:
        raise ValueError(f"Task references unknown table {ref}") from exc


def _columns(table: TableSchema, names: list[str] | tuple[str, ...]) -> None:
    known = {column.name for column in table.columns}
    missing = set(names) - known
    if missing:
        raise ValueError(
            f"Unknown columns on {table.ref}: {', '.join(sorted(missing))}"
        )
