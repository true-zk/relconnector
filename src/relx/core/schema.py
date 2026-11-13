"""Database structure discovered before graph construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForeignKey:
    column: str
    reference_table: str
    reference_column: str


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    ordinal: int
    declared_type: str
    nullable: bool


@dataclass(frozen=True)
class TableSchema:
    name: str
    primary_key: str | None
    foreign_keys: tuple[ForeignKey, ...] = ()
    columns: tuple[ColumnSchema, ...] = ()
