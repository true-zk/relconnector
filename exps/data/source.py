"""Dataset source abstractions and the RelBench implementation."""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import date, datetime, timezone
from typing import cast

import numpy as np
import pandas as pd
from relbench.base import Database, Table, TaskType

from relbench_compat.tasks import hidden_columns as task_hidden_columns

from .contracts import (
    RelBenchDataset,
    RelBenchEntityTask,
    RelBenchRecommendationTask,
    RelBenchTask,
)
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
        dataset_loader: Callable[[str], RelBenchDataset] | None = None,
        legacy_task_loader: Callable[[str, str], RelBenchTask] | None = None,
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
            selected_tasks = dataset.get_task_names()

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

    def _load_dataset(self, dataset_name: str) -> RelBenchDataset:
        if self._dataset_loader is not None:
            return self._dataset_loader(dataset_name)

        import relbench

        loader = getattr(relbench, "load_dataset", None)
        if loader is not None:
            if self.revision is not None:
                return cast(
                    RelBenchDataset, loader(dataset_name, revision=self.revision)
                )
            return cast(RelBenchDataset, loader(dataset_name))
        datasets_module = importlib.import_module("relbench.datasets")
        return cast(RelBenchDataset, datasets_module.get_dataset(dataset_name))

    def _load_task(
        self, dataset: RelBenchDataset, dataset_name: str, task_name: str
    ) -> RelBenchTask:
        if self._legacy_task_loader is not None:
            return self._legacy_task_loader(dataset_name, task_name)
        return dataset.load_task(task_name)


def _load_complete_database(dataset: RelBenchDataset) -> Database:
    """Load rows after the test timestamp as well as the training history."""
    try:
        return dataset.get_db(upto_test_timestamp=False)
    except TypeError:
        try:
            return dataset.get_db(False)
        except TypeError:
            return dataset.get_db()


def _database_tables(database: Database) -> dict[str, MaterializedTable]:
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
    task_name: str, table_name: str, task: RelBenchTask, database: Database
) -> tuple[MaterializedTable, TaskMetadata]:
    frames: list[pd.DataFrame] = []
    relation_tables: list[Table] = []
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
    foreign_keys = []
    for column, reference_table in relation_table.fkey_col_to_pkey_table.items():
        reference_column = database.table_dict[reference_table].pkey_col
        if encodings.get(column) == "json" or reference_column is None:
            continue
        foreign_keys.append(ForeignKeySchema(column, reference_table, reference_column))
    time_column = relation_table.time_col
    schema = TableSchema(
        name=table_name,
        primary_key=TARGET_ROW_ID_COLUMN,
        foreign_keys=tuple(foreign_keys),
        time_column=time_column,
        columns=columns,
        kind="task",
        task_name=task_name,
        split_column=TARGET_SPLIT_COLUMN,
    )

    if task.task_type == TaskType.RECOMMENDATION:
        recommendation_task = cast(RelBenchRecommendationTask, task)
        entity_table = recommendation_task.src_entity_table
        entity_column = recommendation_task.src_entity_col
        target_column = recommendation_task.dst_entity_col
        dst_entity_table = recommendation_task.dst_entity_table
        dst_entity_column = recommendation_task.dst_entity_col
        eval_k = recommendation_task.eval_k
    else:
        entity_task = cast(RelBenchEntityTask, task)
        entity_table = entity_task.entity_table
        entity_column = entity_task.entity_col
        target_column = entity_task.target_col
        dst_entity_table = None
        dst_entity_column = None
        eval_k = None

    metadata = TaskMetadata(
        name=task_name,
        table_name=table_name,
        task_type=task.task_type.value,
        entity_table=entity_table,
        entity_column=entity_column,
        target_column=target_column,
        time_column=task.time_col,
        extra={
            "kind": task.kind,
            "dst_entity_table": dst_entity_table,
            "dst_entity_column": dst_entity_column,
            "hidden_columns": [
                list(pair)
                for pair in task_hidden_columns(
                    {"kind": task.kind, "hidden_columns": _hidden_columns(task)},
                    name=task_name,
                    entity_table=entity_table,
                    target_column=target_column,
                )
            ],
            "timedelta": str(task.timedelta),
            "num_eval_timestamps": task.num_eval_timestamps,
            "eval_k": eval_k,
        },
    )
    return MaterializedTable(frame=frame, schema=schema), metadata


def _primary_key(database: Database, table_name: str) -> str:
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


def _optional_iso(value: pd.Timestamp | datetime | date | str | None) -> str:
    return "" if value is None else pd.Timestamp(value).isoformat()


def _hidden_columns(task: RelBenchTask) -> list[list[str]]:
    return [[str(table), str(column)] for table, column in task.hidden_columns()]


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)
