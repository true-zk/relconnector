"""
Schema classes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping


@dataclass(frozen=True)
class ForeignKey:
    column: str
    reference_table: str
    reference_column: str


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    ordinal: int
    pandas_dtype: str
    encoding: str = "scalar"
    encoding_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TableSchema:
    name: str
    primary_key: str | None
    foreign_keys: tuple[ForeignKey, ...] = ()
    time_column: str | None = None
    columns: tuple[ColumnSchema, ...] = ()
    kind: str = "data"
    task_name: str | None = None
    split_column: str | None = None


@dataclass(frozen=True)
class TaskMetadata:
    name: str
    table_name: str
    task_type: str | None = None
    entity_table: str | None = None
    entity_column: str | None = None
    target_column: str | None = None
    time_column: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)
