"""Check task row values after structural validation."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from numbers import Real
from typing import TYPE_CHECKING, cast

import pandas as pd
from sqlalchemy import MetaData, Table, func, select

from .spec import ColumnSplit, SqlQuerySplit, TaskType
from .splits import (
    _column_assignments,
    _default_assignments,
    _is_null,
    _query_assignments,
    _Sample,
)
from .validation import TaskDataReport, ValidatedTask

if TYPE_CHECKING:
    from relx.conn import BaseDatabaseReader


def check_task_data(
    task: ValidatedTask, reader: BaseDatabaseReader, *, batch_size: int
) -> ValidatedTask:
    """Read only task identity/label columns; retain IDs for split membership."""
    spec = task.spec
    table = Table(
        spec.table.table_name,
        MetaData(),
        schema=spec.table.schema_name,
        autoload_with=reader.engine,
    )
    names = list(
        dict.fromkeys(
            (*task.sample_key, spec.entity_column, spec.label_column)
            + ((spec.time_column,) if spec.time_column else ())
            + ((spec.split.column,) if isinstance(spec.split, ColumnSplit) else ())
        )
    )
    samples: dict[tuple[object, ...], _Sample] = {}
    for frame in reader.iter_query(
        select(*(table.c[name] for name in names)), batch_size=batch_size
    ):
        for row in frame.to_dict("records"):
            key = tuple(row[name] for name in task.sample_key)
            if any(_is_null(part) for part in key):
                raise ValueError(f"Task table {spec.table} contains a null sample key")
            if key in samples:
                raise ValueError(
                    f"Task table {spec.table} contains duplicate sample key {key}"
                )
            entity = row[spec.entity_column]
            if _is_null(entity):
                raise ValueError(
                    f"{spec.table}.{spec.entity_column} contains a null entity key"
                )
            sample_time = (
                _parse_time(row[spec.time_column], f"{spec.table}.{spec.time_column}")
                if spec.time_column
                else None
            )
            split_value = (
                row[spec.split.column] if isinstance(spec.split, ColumnSplit) else None
            )
            samples[key] = _Sample(row[spec.label_column], sample_time, split_value)
    if not samples:
        raise ValueError(f"Task table {spec.table} contains no samples")

    _check_entities(task, reader, table)
    if isinstance(spec.split, ColumnSplit):
        assignments = _column_assignments(samples, spec.split)
    elif isinstance(spec.split, SqlQuerySplit):
        assignments = _query_assignments(task, reader, samples, batch_size)
    else:
        assignments = _default_assignments(samples, spec.split)

    counts = {name: 0 for name in ("train", "validation", "test")}
    labeled_test_rows = 0
    train_labels: list[object] = []
    validation_labels: list[object] = []
    for key, split_name in assignments.items():
        counts[split_name] += 1
        label = samples[key].label
        if split_name == "test":
            labeled_test_rows += not _is_null(label)
        else:
            if _is_null(label):
                raise ValueError(f"{split_name} sample {key} has no label")
            if split_name == "train":
                train_labels.append(label)
            else:
                validation_labels.append(label)
    if not counts["train"] or not counts["validation"]:
        raise ValueError("Task requires nonempty train and validation splits")
    if 0 < labeled_test_rows < counts["test"]:
        raise ValueError("Test split mixes labeled and unlabeled samples")
    destination_ids = _check_labels(task, train_labels, validation_labels)
    if destination_ids is not None:
        _check_destinations(task, reader, destination_ids, batch_size)
    return replace(
        task,
        data_report=TaskDataReport(
            counts["train"],
            counts["validation"],
            counts["test"],
            test_labels_available=bool(labeled_test_rows),
        ),
        split_by_key=assignments,
    )


def _check_entities(
    task: ValidatedTask, reader: BaseDatabaseReader, table: Table
) -> None:
    entity = Table(
        task.entity_table.table_name,
        MetaData(),
        schema=task.entity_table.schema_name,
        autoload_with=reader.engine,
    )
    source = table.c[task.spec.entity_column]
    target = entity.c[task.entity_key[0]]
    if not reader.read_query(select(target).where(target.is_(None)).limit(1)).empty:
        raise ValueError(f"{task.entity_table} has null identity keys")
    if not reader.read_query(
        select(target).group_by(target).having(func.count() > 1).limit(1)
    ).empty:
        raise ValueError(f"{task.entity_table} has duplicate identity keys")
    missing = reader.read_query(
        select(func.count())
        .select_from(table.outerjoin(entity, source == target))
        .where(target.is_(None))
    )
    if int(missing.iloc[0, 0]):
        raise ValueError(
            f"{task.spec.table}.{task.spec.entity_column} contains orphan entity keys"
        )


def _check_labels(
    task: ValidatedTask, train: list[object], validation: list[object]
) -> set[object] | None:
    kind = task.spec.task_type
    labels = [*train, *validation]
    if kind is TaskType.REGRESSION:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (Real, Decimal))
            or not math.isfinite(float(value))
            for value in labels
        ):
            raise ValueError("Regression train labels must be finite numbers")
    elif kind in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        try:
            distinct = set(train)
            unknown_validation = set(validation) - distinct
        except TypeError as exc:
            raise ValueError("Classification labels must be scalar values") from exc
        if len(distinct) < 2 or (
            kind is TaskType.BINARY_CLASSIFICATION and len(distinct) != 2
        ):
            raise ValueError(f"Train labels are incompatible with {kind.value}")
        if unknown_validation:
            raise ValueError("Validation contains classes absent from train")
    elif kind in (TaskType.MULTILABEL_CLASSIFICATION, TaskType.RECOMMENDATION):
        destinations: set[object] = set()
        for value in labels:
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError as exc:
                raise ValueError("List labels must contain JSON arrays") from exc
            if not isinstance(parsed, list):
                raise TypeError("List labels must be arrays")
            if kind is TaskType.RECOMMENDATION:
                try:
                    destinations.update(parsed)
                except TypeError as exc:
                    raise ValueError(
                        "Recommendation IDs must be scalar values"
                    ) from exc
        if kind is TaskType.RECOMMENDATION:
            return destinations
    return None


def _check_destinations(
    task: ValidatedTask,
    reader: BaseDatabaseReader,
    ids: set[object],
    batch_size: int,
) -> None:
    assert task.destination_table is not None and task.destination_key is not None
    table = Table(
        task.destination_table.table_name,
        MetaData(),
        schema=task.destination_table.schema_name,
        autoload_with=reader.engine,
    )
    known: set[object] = set()
    for frame in reader.iter_query(
        select(table.c[task.destination_key[0]]), batch_size=batch_size
    ):
        known.update(frame.iloc[:, 0].tolist())
    if ids - known:
        raise ValueError("Recommendation labels contain unknown destination IDs")


def _parse_time(value: object, path: str) -> pd.Timestamp:
    if _is_null(value) or not isinstance(value, (str, date, datetime, pd.Timestamp)):
        raise ValueError(f"{path} contains a null or invalid prediction time")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} contains an invalid prediction time {value!r}"
        ) from exc
    if pd.isna(stamp):
        raise ValueError(f"{path} contains a null prediction time")
    return cast(
        pd.Timestamp,
        stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC"),
    )
