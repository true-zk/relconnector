"""ConnectorX reader for databases supported by its source drivers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast
from urllib.parse import quote

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.sql.elements import TextClause

from .base import BaseDatabaseReader, QueryParams, SqlQuery

_SOURCE_SCHEMES = {
    "postgresql": "postgresql",
    "mysql": "mysql",
    "mariadb": "mysql",
    "mssql": "mssql",
    "oracle": "oracle",
    "clickhouse": "clickhouse",
}


class ConnectorXDatabaseReader(BaseDatabaseReader):
    """Read complete results or stream bounded Arrow record batches."""

    def __init__(self, location: str | Path | URL) -> None:
        super().__init__(location)
        if self.database_url.get_backend_name() == "sqlite" and self.path is None:
            raise ValueError("ConnectorX requires a file-backed SQLite database")

    @property
    def connectorx_url(self) -> str:
        """Drop SQLAlchemy driver names that ConnectorX does not recognize."""
        backend = self.database_url.get_backend_name()
        if backend == "sqlite":
            assert self.path is not None
            return "sqlite://" + quote(str(self.path), safe="/")
        scheme = _SOURCE_SCHEMES.get(backend)
        if scheme is None:
            raise ValueError(f"ConnectorX does not support database type {backend!r}")
        url = self.database_url.set(drivername=scheme)
        if backend == "mssql":
            url = url.difference_update_query(["driver"])
        return url.render_as_string(hide_password=False)

    def _sql(self, query: SqlQuery, params: QueryParams) -> str:
        if isinstance(query, str) and not params:
            return query
        if isinstance(query, (str, TextClause)):
            statement = text(query) if isinstance(query, str) else query
            if params:
                statement = statement.bindparams(**params)
        else:
            statement = query.params(**params) if params else query
        dialect = self.database_url.get_dialect()()
        return str(
            statement.compile(
                dialect=dialect,
                compile_kwargs={"literal_binds": True},
            )
        )

    def read_query(
        self, query: SqlQuery, *, params: QueryParams = None
    ) -> pd.DataFrame:
        self._ensure_open()
        try:
            import connectorx as cx
        except ImportError as exc:
            raise RuntimeError("ConnectorX reader requires connectorx") from exc
        return cast(
            pd.DataFrame,
            cx.read_sql(
                self.connectorx_url, self._sql(query, params), return_type="pandas"
            ),
        )

    def iter_query(
        self, query: SqlQuery, *, batch_size: int, params: QueryParams = None
    ) -> Iterator[pd.DataFrame]:
        self._ensure_open()
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        try:
            import connectorx as cx
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError(
                "ConnectorX streaming requires connectorx and pyarrow"
            ) from exc

        stream = cast(
            pa.RecordBatchReader,
            cx.read_sql(
                self.connectorx_url,
                self._sql(query, params),
                return_type="arrow_stream",
                batch_size=batch_size,
            ),
        )
        try:
            for batch in stream:
                if batch.num_rows:
                    yield batch.to_pandas(types_mapper=pd.ArrowDtype)
        finally:
            stream.close()
