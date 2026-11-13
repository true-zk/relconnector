"""Database structure independent of a particular SQL dialect."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TableRef:
    table_name: str
    schema_name: str | None = None

    def __str__(self) -> str:
        return (
            self.table_name
            if self.schema_name is None
            else f"{self.schema_name}.{self.table_name}"
        )


@dataclass(frozen=True)
class ForeignKey:
    source_columns: tuple[str, ...]
    target_table: TableRef
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    ordinal: int
    raw_sql_type: str
    nullable: bool


@dataclass(frozen=True)
class TableSchema:
    ref: TableRef
    primary_key: tuple[str, ...] | None
    foreign_keys: tuple[ForeignKey, ...] = ()
    columns: tuple[ColumnSchema, ...] = ()
    identity_key: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DatabaseSchema:
    tables: Mapping[TableRef, TableSchema]
