"""Discover relational structure from SQLite's own schema tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from relx.conn import BaseDatabaseReader
from relx.core import ColumnSchema, ForeignKey, TableSchema


def discover_sqlite_schema(
    reader: BaseDatabaseReader,
    *,
    table_names: Iterable[str] | None = None,
) -> dict[str, TableSchema]:
    """Inspect selected tables without relying on a custom catalog.

    Pass ``table_names`` to select data tables in a database that also contains
    task or application tables. Composite keys need a separate graph policy and
    are rejected here rather than represented incorrectly.
    """
    available = {
        str(name)
        for (name,) in reader.read_query(
            "SELECT name FROM sqlite_schema "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).itertuples(index=False, name=None)
    }
    selected = (
        sorted(available) if table_names is None else list(dict.fromkeys(table_names))
    )
    missing = set(selected) - available
    if missing:
        raise ValueError(f"Unknown tables: {', '.join(sorted(missing))}")

    result: dict[str, TableSchema] = {}
    for name in selected:
        columns, primary_key = _columns(reader, name)
        foreign_keys = _foreign_keys(reader, name)
        result[name] = TableSchema(
            name=name,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
            columns=columns,
        )
    return result


def _columns(
    reader: BaseDatabaseReader, table: str
) -> tuple[tuple[ColumnSchema, ...], str | None]:
    query = (
        'SELECT cid, name, type, "notnull", pk '
        f"FROM pragma_table_info({_sql_literal(table)}) ORDER BY cid"
    )
    columns: list[ColumnSchema] = []
    primary_keys: list[tuple[int, str]] = []
    for ordinal, name, declared_type, not_null, pk_position in reader.read_query(
        query
    ).itertuples(index=False, name=None):
        position = int(pk_position)
        name = str(name)
        sql_type = str(declared_type)
        columns.append(
            ColumnSchema(
                name=name,
                ordinal=int(ordinal),
                declared_type=sql_type,
                nullable=not bool(
                    not_null or (position and sql_type.upper() == "INTEGER")
                ),
            )
        )
        if position:
            primary_keys.append((position, name))
    if len(primary_keys) > 1:
        raise ValueError(f"Table {table!r} has a composite primary key")
    primary_key = primary_keys[0][1] if primary_keys else None
    return tuple(columns), primary_key


def _foreign_keys(reader: BaseDatabaseReader, table: str) -> tuple[ForeignKey, ...]:
    query = (
        'SELECT id, seq, "table", "from", COALESCE("to", \'\') '
        f"FROM pragma_foreign_key_list({_sql_literal(table)}) ORDER BY id, seq"
    )
    groups: dict[int, list[tuple[str, str, object]]] = defaultdict(list)
    for (
        key_id,
        _position,
        reference_table,
        column,
        reference_column,
    ) in reader.read_query(query).itertuples(index=False, name=None):
        groups[int(key_id)].append(
            (str(column), str(reference_table), reference_column)
        )

    result: list[ForeignKey] = []
    for key_id, parts in groups.items():
        if len(parts) != 1:
            raise ValueError(f"Table {table!r} has composite foreign key {key_id}")
        column, reference_table, reference_column = parts[0]
        if not reference_column:
            _, resolved = _columns(reader, reference_table)
            if resolved is None:
                raise ValueError(
                    f"Foreign key {table}.{column} references a table without a primary key"
                )
            reference_column = resolved
        result.append(ForeignKey(column, reference_table, str(reference_column)))
    return tuple(result)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
