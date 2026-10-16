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
    from .process_api import MultiprocessRelBenchModel


def __getattr__(name: str) -> object:
    if name in {"OnlineRelBenchModel", "OnlineTrainingSession"}:
        from . import api

        return getattr(api, name)
    if name == "MultiprocessRelBenchModel":
        from .process_api import MultiprocessRelBenchModel

        return MultiprocessRelBenchModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDatabaseReader",
    "ConnectorXDatabaseReader",
    "MultiprocessRelBenchModel",
    "OnlineRelBenchModel",
    "OnlineTrainingConfig",
    "OnlineTrainingSession",
    "PandasDatabaseReader",
    "create_reader",
    "register_reader",
]
