"""Dataset source abstractions and the RelBench implementation."""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .encoding import infer_columns
from .models import (
    TARGET_ROW_ID_COLUMN,
    TARGET_SPLIT_COLUMN,
    DatasetBundle,
    ForeignKeySchema,
    MaterializedTable,
    TableSchema,
    TaskMetadata,
)


class BaseDatasetSource(ABC):
    """Source of complete relational datasets ready for persistence."""

    @abstractmethod
    def load(
        self,
        dataset_name: str,
        task_names: Sequence[str] = (),
        *,
        all_tasks: bool = False,
    ) -> DatasetBundle:
        """Load a complete dataset and any requested task tables."""


class RelBenchDatasetSource(BaseDatasetSource):
    """Download and materialize datasets through the installed RelBench API."""

    def __init__(
        self,
        *,
        revision: str | None = None,
        dataset_loader: Callable[[str], Any] | None = None,
        legacy_task_loader: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.revision = revision
        self._dataset_loader = dataset_loader
        self._legacy_task_loader = legacy_task_loader

    def load(
        self,
        dataset_name: str,
        task_names: Sequence[str] = (),
        *,
        all_tasks: bool = False,
    ) -> DatasetBundle:
        dataset = self._load_dataset(dataset_name)
        selected_tasks = list(dict.fromkeys(task_names))
        if all_tasks:
            get_task_names = getattr(dataset, "get_task_names", None)
            if get_task_names is None:
                raise NotImplementedError(
                    "This RelBench version cannot enumerate available tasks"
                )
            selected_tasks = list(get_task_names())

        database = _load_complete_database(dataset)
        tables = _database_tables(database)
        tasks: dict[str, TaskMetadata] = {}

        for task_name in selected_tasks:
            task = self._load_task(dataset, dataset_name, task_name)
            table_name = _target_table_name(task_name, len(selected_tasks))
            materialized, task_metadata = _task_table(
                task_name, table_name, task, database
            )
            if table_name in tables:
                raise ValueError(
                    f"Generated task table {table_name!r} collides with a data table"
                )
            tables[table_name] = materialized
            tasks[task_name] = task_metadata

        try:
            import relbench

            relbench_version = getattr(relbench, "__version__", "unknown")
        except ImportError:
            relbench_version = "unknown"

        manifest = getattr(dataset, "manifest", None)
        logical_name = getattr(manifest, "name", dataset_name)
        return DatasetBundle(
            dataset_name=str(logical_name),
            tables=tables,
            tasks=tasks,
            metadata={
                "source": "relbench",
                "source_spec": dataset_name,
                "relbench_version": relbench_version,
                "source_revision": self.revision or "",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "val_timestamp": _optional_iso(getattr(dataset, "val_timestamp", None)),
                "test_timestamp": _optional_iso(
                    getattr(dataset, "test_timestamp", None)
                ),
            },
        )

    def _load_dataset(self, dataset_name: str) -> Any:
        if self._dataset_loader is not None:
            return self._dataset_loader(dataset_name)

        import relbench

        loader = getattr(relbench, "load_dataset", None)
        if loader is not None:
            if self.revision is not None:
                return loader(dataset_name, revision=self.revision)
            return loader(dataset_name)
        datasets_module = importlib.import_module("relbench.datasets")
        return datasets_module.get_dataset(dataset_name)

    def _load_task(self, dataset: Any, dataset_name: str, task_name: str) -> Any:
        load_task = getattr(dataset, "load_task", None)
        if load_task is not None:
            return load_task(task_name)
        if self._legacy_task_loader is not None:
            return self._legacy_task_loader(dataset_name, task_name)
        tasks_module = importlib.import_module("relbench.tasks")
        return tasks_module.get_task(dataset_name, task_name)


def _load_complete_database(dataset: Any) -> Any:
    """Load rows after the test timestamp as well as the training history."""
    try:
        return dataset.get_db(upto_test_timestamp=False)
    except TypeError:
        try:
            return dataset.get_db(False)
        except TypeError:
            return dataset.get_db()


def _database_tables(database: Any) -> dict[str, MaterializedTable]:
    result: dict[str, MaterializedTable] = {}
    for name, table in database.table_dict.items():
        frame = table.df.copy()
        foreign_keys = tuple(
            ForeignKeySchema(
                column=column,
                reference_table=reference_table,
                reference_column=_primary_key(database, reference_table),
            )
            for column, reference_table in table.fkey_col_to_pkey_table.items()
        )
        result[name] = MaterializedTable(
            frame=frame,
            schema=TableSchema(
                name=name,
                primary_key=table.pkey_col,
                foreign_keys=foreign_keys,
                time_column=table.time_col,
                columns=infer_columns(frame),
            ),
        )
    return result


def _task_table(
    task_name: str, table_name: str, task: Any, database: Any
) -> tuple[MaterializedTable, TaskMetadata]:
    frames: list[pd.DataFrame] = []
    relation_tables: list[Any] = []
    for split in ("train", "val", "test"):
        try:
            table = task.get_table(split, mask_input_cols=False)
        except TypeError:
            table = task.get_table(split)
        except (KeyError, ValueError):
            continue
        if table is None:
            continue
        if TARGET_SPLIT_COLUMN in table.df or TARGET_ROW_ID_COLUMN in table.df:
            raise ValueError(f"Task {task_name!r} uses a reserved target-table column")
        frame = table.df.copy()
        frame.insert(0, TARGET_SPLIT_COLUMN, split)
        frames.append(frame)
        relation_tables.append(table)

    if not frames:
        raise RuntimeError(f"Task {task_name!r} has no train/val/test data")

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame.insert(0, TARGET_ROW_ID_COLUMN, np.arange(len(frame), dtype=np.int64))
    relation_table = relation_tables[0]
    columns = infer_columns(frame)
    encodings = {column.name: column.encoding for column in columns}
    foreign_keys = tuple(
        ForeignKeySchema(
            column=column,
            reference_table=reference_table,
            reference_column=_primary_key(database, reference_table),
        )
        for column, reference_table in relation_table.fkey_col_to_pkey_table.items()
        if encodings.get(column) != "json"
    )
    time_column = getattr(relation_table, "time_col", None)
    schema = TableSchema(
        name=table_name,
        primary_key=TARGET_ROW_ID_COLUMN,
        foreign_keys=foreign_keys,
        time_column=time_column,
        columns=columns,
        kind="task",
        task_name=task_name,
        split_column=TARGET_SPLIT_COLUMN,
    )

    task_type = getattr(task, "task_type", None)
    task_type = getattr(task_type, "value", task_type)
    metadata = TaskMetadata(
        name=task_name,
        table_name=table_name,
        task_type=str(task_type) if task_type is not None else None,
        entity_table=getattr(
            task, "entity_table", getattr(task, "src_entity_table", None)
        ),
        entity_column=getattr(
            task, "entity_col", getattr(task, "src_entity_col", None)
        ),
        target_column=getattr(
            task, "target_col", getattr(task, "dst_entity_col", None)
        ),
        time_column=getattr(task, "time_col", time_column),
        extra={
            "kind": getattr(task, "kind", None),
            "dst_entity_table": getattr(task, "dst_entity_table", None),
            "dst_entity_column": getattr(task, "dst_entity_col", None),
        },
    )
    return MaterializedTable(frame=frame, schema=schema), metadata


def _primary_key(database: Any, table_name: str) -> str:
    primary_key = database.table_dict[table_name].pkey_col
    if primary_key is None:
        raise ValueError(f"Foreign-key target table {table_name!r} has no primary key")
    return primary_key


def _target_table_name(task_name: str, number_of_tasks: int) -> str:
    if number_of_tasks == 1:
        return "target_table"
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", task_name).strip("_").lower()
    if not slug:
        raise ValueError(f"Cannot derive a table name from task {task_name!r}")
    return f"target_{slug}"


def _optional_iso(value: Any) -> str:
    return "" if value is None else pd.Timestamp(value).isoformat()
