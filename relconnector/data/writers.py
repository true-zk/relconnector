"""Database writer abstractions and the SQLite implementation."""

from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, cast

from .encoding import encode_frame, sqlite_type
from .models import (
    CATALOG_VERSION,
    COLUMNS_TABLE,
    FOREIGN_KEYS_TABLE,
    METADATA_TABLE,
    TABLES_TABLE,
    TASKS_TABLE,
    DatasetBundle,
    TableSchema,
)


def quote_identifier(identifier: str) -> str:
    """Quote an SQL identifier using the ANSI/SQLite double-quote form."""
    return '"' + identifier.replace('"', '""') + '"'


class BaseDatabaseWriter(ABC):
    """Persistence backend for a complete :class:`DatasetBundle`."""

    @property
    @abstractmethod
    def url(self) -> str:
        """Connection URL consumed by database readers."""

    @abstractmethod
    def write(self, bundle: DatasetBundle, *, overwrite: bool = False) -> str:
        """Persist a bundle and return its database URL."""


class SQLiteDatabaseWriter(BaseDatabaseWriter):
    """Atomically materialize a complete bundle into one SQLite file."""

    def __init__(self, path: str | Path, *, insertion_chunk_size: int = 10_000) -> None:
        if insertion_chunk_size <= 0:
            raise ValueError("insertion_chunk_size must be greater than zero")
        self.path = Path(path).expanduser().resolve()
        self.insertion_chunk_size = insertion_chunk_size

    @property
    def url(self) -> str:
        return f"sqlite://{self.path}"

    def write(self, bundle: DatasetBundle, *, overwrite: bool = False) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; enable overwrite to replace it")

        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            self._write_file(temporary_path, bundle)
            os.replace(temporary_path, self.path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return self.url

    def _write_file(self, path: Path, bundle: DatasetBundle) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            for materialized in bundle.tables.values():
                _create_data_table(connection, materialized.schema)

            for name, materialized in bundle.tables.items():
                encoded = encode_frame(materialized.frame, materialized.schema.columns)
                encoded.to_sql(
                    name,
                    connection,
                    if_exists="append",
                    index=False,
                    chunksize=self.insertion_chunk_size,
                )

            _write_catalog(connection, bundle)
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(
                    f"Found {len(violations)} foreign-key violations; "
                    f"first entries: {violations[:10]}"
                )
            connection.commit()
        finally:
            connection.close()


def _create_data_table(connection: sqlite3.Connection, schema: TableSchema) -> None:
    definitions: list[str] = []
    for column in schema.columns:
        definition = f"{quote_identifier(column.name)} {sqlite_type(column)}"
        if column.name == schema.primary_key:
            definition += " PRIMARY KEY"
        definitions.append(definition)

    for foreign_key in schema.foreign_keys:
        definitions.append(
            f"FOREIGN KEY ({quote_identifier(foreign_key.column)}) REFERENCES "
            f"{quote_identifier(foreign_key.reference_table)}"
            f"({quote_identifier(foreign_key.reference_column)})"
        )
    if not definitions:
        raise ValueError(f"Cannot store table {schema.name!r} with no columns")
    connection.execute(
        f"CREATE TABLE {quote_identifier(schema.name)} ({', '.join(definitions)})"
    )


def _write_catalog(connection: sqlite3.Connection, bundle: DatasetBundle) -> None:
    connection.execute(
        f"CREATE TABLE {quote_identifier(METADATA_TABLE)} "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        f"CREATE TABLE {quote_identifier(TABLES_TABLE)} ("
        "table_name TEXT PRIMARY KEY, "
        "table_kind TEXT NOT NULL, "
        "primary_key TEXT, "
        "time_column TEXT, "
        "task_name TEXT, "
        "split_column TEXT)"
    )
    connection.execute(
        f"CREATE TABLE {quote_identifier(COLUMNS_TABLE)} ("
        "table_name TEXT NOT NULL, "
        "ordinal_position INTEGER NOT NULL, "
        "column_name TEXT NOT NULL, "
        "pandas_dtype TEXT NOT NULL, "
        "encoding TEXT NOT NULL, "
        "encoding_metadata TEXT NOT NULL, "
        "PRIMARY KEY (table_name, ordinal_position))"
    )
    connection.execute(
        f"CREATE TABLE {quote_identifier(FOREIGN_KEYS_TABLE)} ("
        "table_name TEXT NOT NULL, "
        "column_name TEXT NOT NULL, "
        "reference_table TEXT NOT NULL, "
        "reference_column TEXT NOT NULL, "
        "PRIMARY KEY (table_name, column_name))"
    )
    connection.execute(
        f"CREATE TABLE {quote_identifier(TASKS_TABLE)} ("
        "task_name TEXT PRIMARY KEY, "
        "table_name TEXT NOT NULL, "
        "task_type TEXT, "
        "entity_table TEXT, "
        "entity_column TEXT, "
        "target_column TEXT, "
        "time_column TEXT, "
        "extra TEXT NOT NULL)"
    )

    metadata = {
        "catalog_version": CATALOG_VERSION,
        "dataset_name": bundle.dataset_name,
        **bundle.metadata,
    }
    connection.executemany(
        f"INSERT INTO {quote_identifier(METADATA_TABLE)} (key, value) VALUES (?, ?)",
        [(str(key), _metadata_value(value)) for key, value in metadata.items()],
    )

    for schema in (table.schema for table in bundle.tables.values()):
        connection.execute(
            f"INSERT INTO {quote_identifier(TABLES_TABLE)} VALUES (?, ?, ?, ?, ?, ?)",
            (
                schema.name,
                schema.kind,
                schema.primary_key,
                schema.time_column,
                schema.task_name,
                schema.split_column,
            ),
        )
        connection.executemany(
            f"INSERT INTO {quote_identifier(COLUMNS_TABLE)} VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    schema.name,
                    column.ordinal,
                    column.name,
                    column.pandas_dtype,
                    column.encoding,
                    json.dumps(
                        column.encoding_metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
                for column in schema.columns
            ],
        )
        connection.executemany(
            f"INSERT INTO {quote_identifier(FOREIGN_KEYS_TABLE)} VALUES (?, ?, ?, ?)",
            [
                (
                    schema.name,
                    foreign_key.column,
                    foreign_key.reference_table,
                    foreign_key.reference_column,
                )
                for foreign_key in schema.foreign_keys
            ],
        )

    connection.executemany(
        f"INSERT INTO {quote_identifier(TASKS_TABLE)} VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                task.name,
                task.table_name,
                task.task_type,
                task.entity_table,
                task.entity_column,
                task.target_column,
                task.time_column,
                json.dumps(task.extra, default=str, ensure_ascii=False),
            )
            for task in bundle.tasks.values()
        ],
    )


def _metadata_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


class WriterFactory(Protocol):
    def __call__(
        self, path: str | Path, *, insertion_chunk_size: int
    ) -> BaseDatabaseWriter: ...


_WRITERS: dict[str, WriterFactory] = {
    "sqlite": SQLiteDatabaseWriter,
}


def register_writer(name: str, writer: type[BaseDatabaseWriter]) -> None:
    """Register another writer implementation for ``create_writer``."""
    key = name.strip().lower().replace("_", "-")
    if not issubclass(writer, BaseDatabaseWriter):
        raise TypeError("writer must inherit BaseDatabaseWriter")
    _WRITERS[key] = cast(WriterFactory, writer)


def create_writer(
    backend: str,
    output_path: str | Path,
    *,
    insertion_chunk_size: int = 10_000,
) -> BaseDatabaseWriter:
    """Create a configured writer by backend name."""
    key = backend.strip().lower().replace("_", "-")
    try:
        writer = _WRITERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_WRITERS))
        raise ValueError(
            f"Unknown database backend {backend!r}; available: {available}"
        ) from exc
    return writer(
        output_path,
        insertion_chunk_size=insertion_chunk_size,
    )
