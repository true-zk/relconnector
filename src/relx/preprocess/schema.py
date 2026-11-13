"""Reflect relational structure through SQLAlchemy's database inspector."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import inspect
from sqlalchemy.engine.reflection import Inspector

from relx.conn import BaseDatabaseReader
from relx.core import ColumnSchema, DatabaseSchema, ForeignKey, TableRef, TableSchema

from .schema_overrides import SchemaOverrides, apply_schema_overrides


def discover_schema(
    reader: BaseDatabaseReader,
    *,
    table_names: Iterable[str | TableRef] | None = None,
    schema_name: str | None = None,
    overrides: SchemaOverrides | None = None,
) -> DatabaseSchema:
    """Discover selected tables in the default or an explicit namespace."""
    with reader.engine.connect() as connection:
        inspector = inspect(connection)
        refs = _select_tables(inspector, table_names, schema_name)
        tables = {ref: _reflect_table(inspector, ref) for ref in refs}
        if overrides is not None:
            tables = apply_schema_overrides(tables, inspector, overrides)
    return DatabaseSchema(tables)


def _select_tables(
    inspector: Inspector,
    table_names: Iterable[str | TableRef] | None,
    schema_name: str | None,
) -> tuple[TableRef, ...]:
    if table_names is None:
        return tuple(
            TableRef(name, schema_name)
            for name in sorted(inspector.get_table_names(schema=schema_name))
        )

    refs = tuple(
        dict.fromkeys(
            item if isinstance(item, TableRef) else TableRef(item, schema_name)
            for item in table_names
        )
    )
    available: dict[str | None, set[str]] = {}
    missing: list[str] = []
    for ref in refs:
        if ref.schema_name not in available:
            available[ref.schema_name] = set(
                inspector.get_table_names(schema=ref.schema_name)
            )
        if ref.table_name not in available[ref.schema_name]:
            missing.append(str(ref))
    if missing:
        raise ValueError(f"Unknown tables: {', '.join(missing)}")
    return refs


def _reflect_table(inspector: Inspector, ref: TableRef) -> TableSchema:
    key = tuple(
        inspector.get_pk_constraint(ref.table_name, schema=ref.schema_name).get(
            "constrained_columns"
        )
        or ()
    )
    primary_key = key or None
    columns = tuple(
        _column(inspector, index, raw, primary_key)
        for index, raw in enumerate(
            inspector.get_columns(ref.table_name, schema=ref.schema_name)
        )
    )
    foreign_keys = tuple(
        _foreign_key(inspector, ref, raw)
        for raw in inspector.get_foreign_keys(ref.table_name, schema=ref.schema_name)
    )
    return TableSchema(
        ref=ref,
        primary_key=primary_key,
        foreign_keys=foreign_keys,
        columns=columns,
        identity_key=primary_key,
    )


def _column(
    inspector: Inspector,
    ordinal: int,
    raw: Mapping[str, Any],
    primary_key: tuple[str, ...] | None,
) -> ColumnSchema:
    name = str(raw["name"])
    sql_type = str(raw["type"])
    nullable = bool(raw["nullable"])
    if (
        inspector.dialect.name == "sqlite"
        and primary_key == (name,)
        and sql_type.upper() == "INTEGER"
    ):
        nullable = False
    return ColumnSchema(name, ordinal, sql_type, nullable)


def _foreign_key(
    inspector: Inspector, source: TableRef, raw: Mapping[str, Any]
) -> ForeignKey:
    source_columns = tuple(str(name) for name in raw.get("constrained_columns") or ())
    target = TableRef(
        str(raw["referred_table"]),
        raw.get("referred_schema") or source.schema_name,
    )
    target_columns = tuple(str(name) for name in raw.get("referred_columns") or ())
    if not target_columns:
        target_columns = tuple(
            inspector.get_pk_constraint(
                target.table_name, schema=target.schema_name
            ).get("constrained_columns")
            or ()
        )
    if not source_columns or len(source_columns) != len(target_columns):
        raise ValueError(f"Cannot resolve foreign key on {source}")
    return ForeignKey(source_columns, target, target_columns)
