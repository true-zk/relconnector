"""Select a database reader at the public connection entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from sqlalchemy.engine import URL

from .base import BaseDatabaseReader
from .connectorx import ConnectorXDatabaseReader
from .pandas import PandasDatabaseReader

ReaderKind = Literal["sqlalchemy", "pandas", "connectorx"]


def connect(
    location: str | Path | URL, *, reader: ReaderKind = "sqlalchemy"
) -> BaseDatabaseReader:
    """Open a relational database using a Pandas or ConnectorX reader."""
    if reader in ("sqlalchemy", "pandas"):
        return PandasDatabaseReader(location)
    if reader == "connectorx":
        return ConnectorXDatabaseReader(location)
    raise ValueError(f"Unknown reader {reader!r}")
