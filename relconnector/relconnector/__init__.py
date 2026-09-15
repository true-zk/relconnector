"""Online relational graph training backed by SQL feature access."""

from typing import TYPE_CHECKING

from .config import OnlineTrainingConfig
from .connector import (
    BaseDatabaseReader,
    ConnectorXDatabaseReader,
    PandasDatabaseReader,
    create_reader,
    register_reader,
)

if TYPE_CHECKING:
    from .api import OnlineRelBenchModel, OnlineTrainingSession


def __getattr__(name: str) -> object:
    if name in {"OnlineRelBenchModel", "OnlineTrainingSession"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDatabaseReader",
    "ConnectorXDatabaseReader",
    "OnlineRelBenchModel",
    "OnlineTrainingConfig",
    "OnlineTrainingSession",
    "PandasDatabaseReader",
    "create_reader",
    "register_reader",
]
