"""Shared connection and query interface for relational databases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.selectable import SelectBase
from typing_extensions import Self

SqlQuery: TypeAlias = str | TextClause | SelectBase
# DB drivers accept additional scalar types such as UUID and Decimal.
QueryParams: TypeAlias = Mapping[str, Any] | None


def normalize_location(location: str | Path | URL) -> tuple[URL, Path | None]:
    """Accept a database URL or a path to an existing SQLite database."""
    if isinstance(location, URL):
        url = location
    elif isinstance(location, Path) or "://" not in location:
        url = URL.create("sqlite+pysqlite", database=str(Path(location).expanduser()))
    else:
        url = make_url(location)

    if url.get_backend_name() != "sqlite" or url.database in (None, ":memory:"):
        return url, None

    path = Path(url.database).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return url.set(database=str(path)), path


class BaseDatabaseReader(ABC):
    """Execute SQL without interpreting tasks or database structure."""

    def __init__(self, location: str | Path | URL) -> None:
        self.database_url, self.path = normalize_location(location)
        self._engine: Engine | None = None
        self._closed = False

    @property
    def url(self) -> str:
        """Connection URL for drivers; avoid including it in logs."""
        return self.database_url.render_as_string(hide_password=False)

    @property
    def engine(self) -> Engine:
        """Shared SQLAlchemy engine for reading and later schema inspection."""
        self._ensure_open()
        if self._engine is None:
            self._engine = create_engine(self.database_url)
        return self._engine

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Database reader is closed")

    @abstractmethod
    def read_query(
        self, query: SqlQuery, *, params: QueryParams = None
    ) -> pd.DataFrame:
        """Read a query result into one DataFrame."""

    @abstractmethod
    def iter_query(
        self, query: SqlQuery, *, batch_size: int, params: QueryParams = None
    ) -> Iterator[pd.DataFrame]:
        """Stream a query result in bounded DataFrame chunks."""

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._engine is not None:
                self._engine.dispose()
                self._engine = None

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
