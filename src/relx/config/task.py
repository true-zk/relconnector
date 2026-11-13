"""Parse task semantics and split rules from the global run configuration."""

from __future__ import annotations

from typing import Any

from relx.task.spec import (
    ColumnSplit,
    DefaultSplit,
    SqlQuerySplit,
    TaskSpec,
    TaskType,
)

from .common import (
    fields,
    fraction,
    mapping,
    nonnegative_int,
    optional_text,
    table_ref,
    text,
)


def parse_task(value: object, default_schema: str | None) -> TaskSpec:
    data = mapping(value, "task")
    fields(
        data,
        {
            "name",
            "type",
            "table",
            "entity_column",
            "entity_table",
            "label_column",
            "time_column",
            "split",
            "destination",
        },
        "task",
    )
    raw_type = text(data.get("type"), "task.type")
    try:
        task_type = TaskType(raw_type)
    except ValueError as exc:
        raise ValueError(f"Unknown task.type: {raw_type}") from exc

    destination_table = None
    destination_key = None
    if "destination" in data:
        destination = mapping(data["destination"], "task.destination")
        fields(destination, {"table", "key"}, "task.destination")
        destination_table = table_ref(
            destination.get("table"), "task.destination.table", default_schema
        )
        destination_key = text(destination.get("key"), "task.destination.key")
    if task_type is TaskType.RECOMMENDATION and destination_table is None:
        raise ValueError("task.destination is required for recommendation tasks")
    if task_type is not TaskType.RECOMMENDATION and destination_table is not None:
        raise ValueError("task.destination is only valid for recommendation tasks")

    return TaskSpec(
        name=text(data.get("name"), "task.name"),
        table=table_ref(data.get("table"), "task.table", default_schema),
        task_type=task_type,
        entity_column=text(data.get("entity_column"), "task.entity_column"),
        label_column=text(data.get("label_column"), "task.label_column"),
        time_column=optional_text(data.get("time_column"), "task.time_column"),
        split=parse_split(data.get("split")),
        entity_table=(
            table_ref(data["entity_table"], "task.entity_table", default_schema)
            if "entity_table" in data
            else None
        ),
        destination_table=destination_table,
        destination_key=destination_key,
    )


def parse_split(value: object) -> ColumnSplit | SqlQuerySplit | DefaultSplit:
    if value is None:
        return DefaultSplit()
    data: dict[str, Any] = mapping(value, "task.split")
    kind = text(data.get("type"), "task.split.type")
    if kind == "column":
        fields(data, {"type", "column", "values"}, "task.split")
        values = mapping(data.get("values"), "task.split.values")
        fields(values, {"train", "validation", "test"}, "task.split.values")
        if set(values) != {"train", "validation", "test"}:
            raise ValueError("task.split.values requires train, validation, and test")
        parsed = [
            _split_value(values[name], name) for name in ("train", "validation", "test")
        ]
        if len({(type(item), item) for item in parsed}) != 3:
            raise ValueError("task.split.values must be distinct")
        return ColumnSplit(text(data.get("column"), "task.split.column"), *parsed)
    if kind == "sql_query":
        fields(
            data,
            {"type", "train", "validation", "test", "validation_fraction", "seed"},
            "task.split",
        )
        train = optional_text(data.get("train"), "task.split.train")
        validation = optional_text(data.get("validation"), "task.split.validation")
        if (train is None) != (validation is None):
            raise ValueError(
                "task.split.sql_query requires both train and validation queries"
            )
        return SqlQuerySplit(
            test=text(data.get("test"), "task.split.test"),
            train=train,
            validation=validation,
            validation_fraction=fraction(
                data.get("validation_fraction", 0.2), "task.split.validation_fraction"
            ),
            seed=nonnegative_int(data.get("seed", 42), "task.split.seed"),
        )
    if kind == "default":
        fields(data, {"type", "validation_fraction", "seed"}, "task.split")
        return DefaultSplit(
            validation_fraction=fraction(
                data.get("validation_fraction", 0.2), "task.split.validation_fraction"
            ),
            seed=nonnegative_int(data.get("seed", 42), "task.split.seed"),
        )
    raise ValueError(f"Unknown task.split.type: {kind}")


def _split_value(value: object, name: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"task.split.values.{name} must be a string or integer")
    if isinstance(value, str) and not value:
        raise ValueError(f"task.split.values.{name} cannot be empty")
    return value
