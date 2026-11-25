"""Parse graph table selection from the global run configuration."""

from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from relx.core import TableRef

from .common import fields, mapping, table_ref, text
from .types import GraphConfig


def parse_graph(value: object, default_schema: str | None) -> GraphConfig:
    data = mapping(value, "graph")
    fields(
        data,
        {"exclude_tables", "time_columns", "static_tables", "timezone", "time_boundary"},
        "graph",
    )
    excluded = _table_list(data.get("exclude_tables", []), "graph.exclude_tables", default_schema)
    static = _table_list(data.get("static_tables", []), "graph.static_tables", default_schema)
    raw_times = mapping(data.get("time_columns", {}), "graph.time_columns")
    times = tuple(
        (
            table_ref(name, f"graph.time_columns.{name}", default_schema),
            text(column, f"graph.time_columns.{name}"),
        )
        for name, column in raw_times.items()
    )
    if len(times) != len({ref for ref, _ in times}):
        raise ValueError("graph.time_columns contains duplicate tables")
    overlap = set(static) & {ref for ref, _ in times}
    if overlap:
        raise ValueError("graph.static_tables and graph.time_columns overlap")
    timezone = text(data.get("timezone", "UTC"), "graph.timezone")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown graph.timezone {timezone!r}") from exc
    boundary = data.get("time_boundary", "at_or_before")
    if boundary not in {"strict_before", "at_or_before"}:
        raise ValueError("graph.time_boundary must be strict_before or at_or_before")
    return GraphConfig(
        exclude_tables=excluded,
        time_columns=times,
        static_tables=static,
        timezone=timezone,
        time_boundary=boundary,
    )


def _table_list(
    raw_tables: object, path: str, default_schema: str | None
) -> tuple[TableRef, ...]:
    if not isinstance(raw_tables, list):
        raise TypeError(f"{path} must be a list of table names")
    refs = tuple(
        table_ref(name, f"{path}[{index}]", default_schema)
        for index, name in enumerate(raw_tables)
    )
    if len(refs) != len(set(refs)):
        raise ValueError(f"{path} contains duplicate tables")
    return refs
