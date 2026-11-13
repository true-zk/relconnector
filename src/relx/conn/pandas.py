"""Pandas query reader backed by SQLAlchemy connections."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from sqlalchemy import text

from .base import BaseDatabaseReader, QueryParams, SqlQuery


class PandasDatabaseReader(BaseDatabaseReader):
    """Read any database supported by an installed SQLAlchemy dialect."""

    def read_query(
        self, query: SqlQuery, *, params: QueryParams = None
    ) -> pd.DataFrame:
        statement = text(query) if isinstance(query, str) else query
        with self.engine.connect() as connection:
            return pd.read_sql_query(
                statement,
                connection,
                params=params,
                dtype_backend="numpy_nullable",
            )

    def iter_query(
        self, query: SqlQuery, *, batch_size: int, params: QueryParams = None
    ) -> Iterator[pd.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        statement = text(query) if isinstance(query, str) else query
        with self.engine.connect().execution_options(
            yield_per=batch_size
        ) as connection:
            yield from pd.read_sql_query(
                statement,
                connection,
                params=params,
                chunksize=batch_size,
                dtype_backend="numpy_nullable",
            )
