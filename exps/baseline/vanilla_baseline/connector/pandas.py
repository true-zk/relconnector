"""Pandas implementation of the database reader contract."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pandas as pd

from .base import BaseDatabaseReader, sqlite_path


class PandasDatabaseReader(BaseDatabaseReader):
    """Read a local SQLite database through ``pandas.read_sql_query``."""

    supports_batched_reads = True

    def read_query(self, query: str) -> pd.DataFrame:
        with sqlite3.connect(sqlite_path(self.url)) as connection:
            return pd.read_sql_query(
                query,
                connection,
                dtype_backend="numpy_nullable",
            )

    def read_row_queries(self, queries: list[str]) -> pd.DataFrame:
        """Execute one-row SELECTs on one connection and combine their rows.

        This intentionally does not combine predicates into an ``IN`` query.
        It exists only for the naive row-query experiment, while avoiding a
        fresh SQLite connection and Pandas frame for every individual row.
        """
        rows: list[tuple[object, ...]] = []
        columns: list[str] = []
        with sqlite3.connect(sqlite_path(self.url)) as connection:
            cursor = connection.cursor()
            for query in queries:
                cursor.execute(query)
                if not columns and cursor.description is not None:
                    columns = [item[0] for item in cursor.description]
                rows.extend(cursor.fetchall())
        return pd.DataFrame.from_records(rows, columns=columns).convert_dtypes(
            dtype_backend="numpy_nullable"
        )

    def iter_query(
        self,
        query: str,
        *,
        batch_size: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        if batch_size is None:
            yield self.read_query(query)
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        with sqlite3.connect(sqlite_path(self.url)) as connection:
            yield from pd.read_sql_query(
                query,
                connection,
                chunksize=batch_size,
                dtype_backend="numpy_nullable",
            )
