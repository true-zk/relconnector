"""Reader registration and construction."""

from __future__ import annotations

from pathlib import Path

from .base import BaseDatabaseReader
from .connectorx import ConnectorXDatabaseReader
from .pandas import PandasDatabaseReader

_READERS: dict[str, type[BaseDatabaseReader]] = {
    "pandas": PandasDatabaseReader,
    "connector-x": ConnectorXDatabaseReader,
    "connectorx": ConnectorXDatabaseReader,
}


def register_reader(name: str, reader: type[BaseDatabaseReader]) -> None:
    key = _normalize_name(name)
    if not issubclass(reader, BaseDatabaseReader):
        raise TypeError("reader must inherit BaseDatabaseReader")
    _READERS[key] = reader


def create_reader(backend: str, location: str | Path) -> BaseDatabaseReader:
    key = _normalize_name(backend)
    try:
        reader = _READERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_READERS))
        raise ValueError(
            f"Unknown reader backend {backend!r}; available: {available}"
        ) from exc
    return reader(location)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")
