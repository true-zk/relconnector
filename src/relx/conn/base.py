"""Minimal query interface shared by SQLite readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import pandas as pd


class BaseDatabaseReader(ABC):
    """Execute SQL without interpreting tasks or database structure."""

    def __init__(self, location: str | Path) -> None:
        value = str(location)
        if value.startswith("sqlite://"):
            value = value[len("sqlite://") :]
        elif "://" in value:
            raise ValueError(f"Unsupported database URL: {value!r}")
        if not value:
            raise ValueError("Database path must not be empty")
        self.path = Path(value).expanduser().resolve()
        self.url = f"sqlite://{self.path}"

    @abstractmethod
    def read_query(self, query: str) -> pd.DataFrame:
        """Read a query result into one DataFrame."""

    @abstractmethod
    def iter_query(self, query: str, *, batch_size: int) -> Iterator[pd.DataFrame]:
        """Stream a query result in bounded DataFrame chunks."""
