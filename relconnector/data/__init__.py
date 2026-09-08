"""Standalone RelBench download and SQL materialization utilities."""

from typing import TYPE_CHECKING, Any

from .models import (
    ColumnSchema,
    DatasetBundle,
    ForeignKeySchema,
    MaterializedTable,
    TableSchema,
    TaskMetadata,
)
from .source import BaseDatasetSource, RelBenchDatasetSource
from .writers import (
    BaseDatabaseWriter,
    SQLiteDatabaseWriter,
    create_writer,
    register_writer,
)

if TYPE_CHECKING:
    from .download import RelBenchMaterializer, download_relbench_data

__all__ = [
    "BaseDatabaseWriter",
    "BaseDatasetSource",
    "ColumnSchema",
    "DatasetBundle",
    "ForeignKeySchema",
    "MaterializedTable",
    "RelBenchDatasetSource",
    "RelBenchMaterializer",
    "SQLiteDatabaseWriter",
    "TableSchema",
    "TaskMetadata",
    "create_writer",
    "download_relbench_data",
    "register_writer",
]


def __getattr__(name: str) -> Any:
    if name in {"RelBenchMaterializer", "download_relbench_data"}:
        from .download import RelBenchMaterializer, download_relbench_data

        return {
            "RelBenchMaterializer": RelBenchMaterializer,
            "download_relbench_data": download_relbench_data,
        }[name]
    raise AttributeError(name)
