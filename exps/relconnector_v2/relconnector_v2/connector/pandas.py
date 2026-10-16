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
