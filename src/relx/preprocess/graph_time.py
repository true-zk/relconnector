"""Validate graph time declarations and normalize SQL time values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyarrow as pa
import pyarrow.compute as pc

from relx.config.types import GraphConfig
from relx.core import DatabaseSchema, TableRef
from relx.task import ValidatedTask

_NATIVE_TIME_COLUMN = "__relx_time_us"


@dataclass(frozen=True)
class GraphTimePlan:
    columns: Mapping[TableRef, str | None]
    timezone: str
    inclusive: bool


def resolve_graph_time(
    schema: DatabaseSchema,
    task: ValidatedTask,
    config: GraphConfig,
    selected: tuple[TableRef, ...],
) -> GraphTimePlan:
    """Require an explicit availability rule for every temporal graph table."""
    timed = [ref for ref, _ in config.time_columns]
    if len(timed) != len(set(timed)):
        raise ValueError("graph.time_columns contains duplicate tables")
    if set(timed) & set(config.static_tables):
        raise ValueError("graph.static_tables and graph.time_columns overlap")
    try:
        ZoneInfo(config.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown graph.timezone {config.timezone!r}") from exc
    if config.time_boundary not in {"strict_before", "at_or_before"}:
        raise ValueError("graph.time_boundary must be strict_before or at_or_before")
    included = set(selected)
    configured = {ref for ref, _ in config.time_columns} | set(config.static_tables)
    invalid = configured - included
    if invalid:
        raise ValueError(
            "graph time rules reference excluded or unknown tables: "
            + ", ".join(sorted(map(str, invalid)))
        )
    columns: dict[TableRef, str | None] = dict(config.time_columns)
    columns.update(dict.fromkeys(config.static_tables))
    for ref, name in columns.items():
        if name is not None and name not in {
            column.name for column in schema.tables[ref].columns
        }:
            raise ValueError(f"Unknown graph time column {ref}.{name}")
    missing = included - set(columns)
    if task.spec.time_column is not None and missing:
        raise ValueError(
            "Temporal task needs graph.time_columns or graph.static_tables for: "
            + ", ".join(sorted(map(str, missing)))
        )
    columns.update(dict.fromkeys(missing))
    return GraphTimePlan(
        columns=columns,
        timezone=config.timezone,
        inclusive=config.time_boundary == "at_or_before",
    )


def normalize_time_batch(
    batch: pa.RecordBatch, column: str, timezone: str
) -> pa.RecordBatch:
    """Append UTC microseconds; missing times use the earliest graph timestamp."""
    if _NATIVE_TIME_COLUMN in batch.schema.names:
        raise ValueError(f"Reserved graph column {_NATIVE_TIME_COLUMN} already exists")
    values = batch.column(column)
    try:
        if pa.types.is_timestamp(values.type):
            stamp = values
        elif (
            pa.types.is_date(values.type)
            or pa.types.is_string(values.type)
            or pa.types.is_large_string(values.type)
        ):
            try:
                stamp = pc.cast(values, pa.timestamp("us"))
            except pa.ArrowInvalid:
                stamp = pc.cast(values, pa.timestamp("us", tz="UTC"))
        else:
            raise TypeError(f"Unsupported graph time type {values.type}")
        if stamp.type.tz is None:
            stamp = pc.assume_timezone(stamp, timezone)  # pyright: ignore[reportAttributeAccessIssue]
        stamp = pc.cast(stamp, pa.timestamp("us", tz="UTC"))
        micros = pc.cast(stamp, pa.int64())
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise ValueError(f"Cannot parse graph time column {column!r}") from exc
    if micros.null_count:
        # Match the existing RelBench graph convention: a missing event time is
        # older than every real timestamp, so it remains visible to later seeds.
        micros = pc.fill_null(micros, -9_223_372_037_000_000)
    return batch.append_column(_NATIVE_TIME_COLUMN, micros)
