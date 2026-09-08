"""Pandas implementation of the database reader contract."""

from __future__ import annotations

import sqlite3

import pandas as pd

from .base import BaseDatabaseReader, sqlite_path


class PandasDatabaseReader(BaseDatabaseReader):
    """Read a local SQLite database through ``pandas.read_sql_query``."""

    def read_query(self, query: str) -> pd.DataFrame:
        with sqlite3.connect(sqlite_path(self.url)) as connection:
            return pd.read_sql_query(
                query,
                connection,
                dtype_backend="numpy_nullable",
            )
