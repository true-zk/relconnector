"""Base API shared by all local database readers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import (
    CATALOG_VERSION,
    COLUMNS_TABLE,
    FOREIGN_KEYS_TABLE,
    METADATA_TABLE,
    TABLES_TABLE,
    TASKS_TABLE,
    ColumnSchema,
    ForeignKey,
    TableSchema,
    TaskMetadata,
)
from .decoding import decode_frame


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class BaseDatabaseReader(ABC):
    """Read complete query results and reconstruct RelBench metadata."""

    supports_batched_reads = False

    def __init__(self, location: str | Path) -> None:
        self.url = database_url(location)
        self._schema_cache: dict[str, TableSchema] | None = None

    @abstractmethod
    def read_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return its complete result in memory."""

    def iter_query(
        self, query: str, *, batch_size: int | None = None
    ) -> Iterator[pd.DataFrame]:
        """Future batch API; this baseline emits exactly one full-memory result."""
        if batch_size is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement bounded batch reads yet"
            )
        return iter((self.read_query(query),))

    def metadata(self) -> dict[str, str]:
        frame = self.read_query(
            f"SELECT key, value FROM {quote_identifier(METADATA_TABLE)}"
        )
        return dict(zip(frame["key"], frame["value"], strict=True))

    def validate_catalog(self) -> None:
        """Reject databases using an incompatible catalog contract."""
        version = self.metadata().get("catalog_version")
        if version is not None and version != CATALOG_VERSION:
            raise RuntimeError(
                f"Unsupported catalog version {version!r}; expected {CATALOG_VERSION!r}"
            )

    def schemas(self) -> dict[str, TableSchema]:
        if self._schema_cache is None:
            self.validate_catalog()
            self._schema_cache = self._read_schemas()
        return self._schema_cache

    def task_metadata(self) -> dict[str, TaskMetadata]:
        frame = self.read_query(
            f"SELECT * FROM {quote_identifier(TASKS_TABLE)} ORDER BY task_name"
        )
        return {
            str(row.task_name): TaskMetadata(
                name=str(row.task_name),
                table_name=str(row.table_name),
                task_type=_optional_string(row.task_type),
                entity_table=_optional_string(row.entity_table),
                entity_column=_optional_string(row.entity_column),
                target_column=_optional_string(row.target_column),
                time_column=_optional_string(row.time_column),
                extra=json.loads(str(row.extra)),
            )
            for row in frame.itertuples(index=False)
        }

    def table_names(self, *, include_tasks: bool = True) -> list[str]:
        return [
            schema.name
            for schema in self.schemas().values()
            if include_tasks or schema.kind == "data"
        ]

    def read_table(self, table_name: str) -> pd.DataFrame:
        try:
            schema = self.schemas()[table_name]
        except KeyError as exc:
            raise KeyError(f"Unknown materialized table: {table_name!r}") from exc
        raw = self.read_query(f"SELECT * FROM {quote_identifier(table_name)}")
        return decode_frame(raw, schema.columns)

    def read_all_tables(self, *, include_tasks: bool = True) -> dict[str, pd.DataFrame]:
        return {
            name: self.read_table(name)
            for name in self.table_names(include_tasks=include_tasks)
        }

    def read_task_splits(self, task_name: str) -> dict[str, pd.DataFrame]:
        try:
            task = self.task_metadata()[task_name]
            schema = self.schemas()[task.table_name]
        except KeyError as exc:
            raise KeyError(f"Unknown materialized task: {task_name!r}") from exc
        if schema.split_column is None:
            raise ValueError(f"Task table {task.table_name!r} has no split column")

        frame = self.read_table(task.table_name)
        drop_columns = [schema.split_column]
        if schema.primary_key is not None:
            drop_columns.append(schema.primary_key)
        return {
            str(split): rows.drop(columns=drop_columns).reset_index(drop=True)
            for split, rows in frame.groupby(schema.split_column, sort=False)
        }

    def read_relbench_database(self) -> Any:
        """Reconstruct a ``relbench.base.Database`` from data tables."""
        try:
            from relbench.base import Database, Table
        except ImportError as exc:
            raise RuntimeError(
                "Install relbench to reconstruct a RelBench Database"
            ) from exc

        table_dict = {}
        for schema in self.schemas().values():
            if schema.kind != "data":
                continue
            table_dict[schema.name] = Table(
                df=self.read_table(schema.name),
                fkey_col_to_pkey_table={
                    foreign_key.column: foreign_key.reference_table
                    for foreign_key in schema.foreign_keys
                },
                pkey_col=schema.primary_key,
                time_col=schema.time_column,
            )
        return Database(table_dict)

    def _read_schemas(self) -> dict[str, TableSchema]:
        tables = self.read_query(
            f"SELECT * FROM {quote_identifier(TABLES_TABLE)} ORDER BY table_name"
        )
        columns = self.read_query(
            f"SELECT * FROM {quote_identifier(COLUMNS_TABLE)} "
            "ORDER BY table_name, ordinal_position"
        )
        foreign_keys = self.read_query(
            f"SELECT * FROM {quote_identifier(FOREIGN_KEYS_TABLE)} "
            "ORDER BY table_name, column_name"
        )

        result: dict[str, TableSchema] = {}
        for row in tables.itertuples(index=False):
            name = str(row.table_name)
            table_columns = columns[columns["table_name"] == row.table_name]
            table_foreign_keys = foreign_keys[
                foreign_keys["table_name"] == row.table_name
            ]
            result[name] = TableSchema(
                name=name,
                primary_key=_optional_string(row.primary_key),
                foreign_keys=tuple(
                    ForeignKey(
                        column=str(fkey.column_name),
                        reference_table=str(fkey.reference_table),
                        reference_column=str(fkey.reference_column),
                    )
                    for fkey in table_foreign_keys.itertuples(index=False)
                ),
                time_column=_optional_string(row.time_column),
                columns=tuple(
                    ColumnSchema(
                        name=str(column.column_name),
                        ordinal=int(str(column.ordinal_position)),
                        pandas_dtype=str(column.pandas_dtype),
                        encoding=str(column.encoding),
                        encoding_metadata=json.loads(str(column.encoding_metadata)),
                    )
                    for column in table_columns.itertuples(index=False)
                ),
                kind=str(row.table_kind),
                task_name=_optional_string(row.task_name),
                split_column=_optional_string(row.split_column),
            )
        return result


def database_url(location: str | Path) -> str:
    value = str(location)
    if "://" in value:
        return value
    return f"sqlite://{Path(value).expanduser().resolve()}"


def sqlite_path(url: str) -> str:
    prefix = "sqlite://"
    if not url.startswith(prefix):
        raise ValueError(f"Expected a SQLite URL, got {url!r}")
    path = url[len(prefix) :]
    if not path:
        raise ValueError(f"Invalid SQLite URL: {url!r}")
    return path


def _optional_string(value: Any) -> str | None:
    try:
        return None if pd.isna(value) else str(value)
    except (TypeError, ValueError):
        return str(value)
