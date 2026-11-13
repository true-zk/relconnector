"""Pandas reader for materialized SQLite databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pandas as pd

from .base import BaseDatabaseReader


class PandasDatabaseReader(BaseDatabaseReader):
    """Execute SQLite queries without loading all rows during iteration."""

    def __init__(self, location: str | Path) -> None:
        super().__init__(location)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def read_query(self, query: str) -> pd.DataFrame:
        with closing(sqlite3.connect(self.path)) as connection:
            return pd.read_sql_query(query, connection, dtype_backend="numpy_nullable")

    def iter_query(self, query: str, *, batch_size: int) -> Iterator[pd.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        with closing(sqlite3.connect(self.path)) as connection:
            yield from pd.read_sql_query(
                query,
                connection,
                chunksize=batch_size,
                dtype_backend="numpy_nullable",
            )
