"""Data structures passed from a dataset source to a database writer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

METADATA_TABLE = "_relconnector_metadata"
TABLES_TABLE = "_relconnector_tables"
COLUMNS_TABLE = "_relconnector_columns"
FOREIGN_KEYS_TABLE = "_relconnector_foreign_keys"
TASKS_TABLE = "_relconnector_tasks"
CATALOG_VERSION = "1"
INTERNAL_TABLES = frozenset(
    {
        METADATA_TABLE,
        TABLES_TABLE,
        COLUMNS_TABLE,
        FOREIGN_KEYS_TABLE,
        TASKS_TABLE,
    }
)

TARGET_ROW_ID_COLUMN = "__target_row_id__"
TARGET_SPLIT_COLUMN = "split"
NODE_ID_COLUMN = "__relconnector_node_id__"


@dataclass(frozen=True)
class ForeignKeySchema:
    column: str
    reference_table: str
    reference_column: str


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    ordinal: int
    pandas_dtype: str
    encoding: str = "scalar"
    encoding_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TableSchema:
    name: str
    primary_key: str | None
    foreign_keys: tuple[ForeignKeySchema, ...] = ()
    time_column: str | None = None
    columns: tuple[ColumnSchema, ...] = ()
    kind: str = "data"
    task_name: str | None = None
    split_column: str | None = None


@dataclass(frozen=True)
class MaterializedTable:
    frame: pd.DataFrame
    schema: TableSchema


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


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    tables: Mapping[str, MaterializedTable]
    tasks: Mapping[str, TaskMetadata] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = set(self.tables).intersection(INTERNAL_TABLES)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"Dataset uses reserved table name(s): {names}")
        for name, table in self.tables.items():
            if name != table.schema.name:
                raise ValueError(
                    f"Table mapping key {name!r} does not match schema name "
                    f"{table.schema.name!r}"
                )
            expected_columns = [column.name for column in table.schema.columns]
            if list(table.frame.columns) != expected_columns:
                raise ValueError(f"Column schema does not match dataframe for {name!r}")
