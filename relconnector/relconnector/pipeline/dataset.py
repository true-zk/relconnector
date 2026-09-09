"""Reconstruct RelBench databases and tasks exclusively from local SQL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.base import Database, Table, TaskType

from ..connector import BaseDatabaseReader, create_reader
from ..connector.catalog import TaskMetadata


@dataclass
class LocalEntityTask:
    name: str
    task_type: TaskType
    entity_table: str
    entity_col: str
    target_col: str
    time_col: str | None
    kind: str | None
    remove_columns: list[tuple[str, str]]
    _splits: dict[str, pd.DataFrame]

    def get_table(self, split: str, mask_input_cols: bool | None = None) -> Table:
        del mask_input_cols
        try:
            frame = self._splits[split]
        except KeyError as exc:
            raise KeyError(f"Task {self.name!r} has no {split!r} split") from exc
        return Table(
            df=frame,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

    def hidden_columns(self) -> list[tuple[str, str]]:
        return list(self.remove_columns)


@dataclass
class LocalRecommendationTask:
    name: str
    task_type: TaskType
    src_entity_table: str
    src_entity_col: str
    dst_entity_table: str
    dst_entity_col: str
    time_col: str | None
    kind: str | None
    remove_columns: list[tuple[str, str]]
    eval_k: int | None
    _splits: dict[str, pd.DataFrame]

    def get_table(self, split: str, mask_input_cols: bool | None = None) -> Table:
        del mask_input_cols
        try:
            frame = self._splits[split]
        except KeyError as exc:
            raise KeyError(f"Task {self.name!r} has no {split!r} split") from exc
        return Table(
            df=frame,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

    def hidden_columns(self) -> list[tuple[str, str]]:
        return list(self.remove_columns)


LocalTask = LocalEntityTask | LocalRecommendationTask


@dataclass
class LocalRelBenchDataset:
    name: str
    path: Path
    reader: BaseDatabaseReader
    val_timestamp: pd.Timestamp | None
    test_timestamp: pd.Timestamp | None

    def get_db(self, *, upto_test_timestamp: bool = True) -> Database:
        database = self.reader.read_relbench_database()
        if upto_test_timestamp and self.test_timestamp is not None:
            database = database.upto(self.test_timestamp)
        return _validate_and_correct_database(database)

    def task_names(self) -> list[str]:
        return sorted(self.reader.task_metadata())

    def load_task(self, task_name: str) -> LocalTask:
        try:
            metadata = self.reader.task_metadata()[task_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown task {task_name!r}; available: {self.task_names()}"
            ) from exc
        splits = self.reader.read_task_splits(task_name)
        return _build_task(metadata, splits)


def _validate_and_correct_database(database: Database) -> Database:
    """Apply the same post-truncation key checks as RelBench Dataset.get_db()."""
    for table_name, table in database.table_dict.items():
        if table.pkey_col is None:
            continue
        primary_keys = table.df[table.pkey_col].to_numpy()
        if not np.array_equal(primary_keys, np.arange(len(table))):
            raise RuntimeError(
                f"Primary key {table.pkey_col!r} in table {table_name!r} "
                "is not consecutively indexed from zero"
            )

    for table in database.table_dict.values():
        for foreign_key, primary_table in table.fkey_col_to_pkey_table.items():
            values = table.df[foreign_key]
            dangling = values.notna() & (
                values >= len(database.table_dict[primary_table])
            )
            if dangling.any():
                table.df.loc[dangling, foreign_key] = None
    return database


def default_sqlite_path(dataset_name: str, root: Path | None = None) -> Path:
    if root is None:
        root = Path(__file__).resolve().parents[2] / "data" / "relbench"
    return root / f"{dataset_name}.sqlite"


def open_dataset(
    name: str,
    *,
    reader_kind: str = "pandas",
    sqlite_path: Path | None = None,
) -> LocalRelBenchDataset:
    path = sqlite_path or default_sqlite_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Local SQLite mirror for {name!r} not found at {path}. "
            "Run `python -m data.download_all --dataset <name>` first."
        )
    reader = create_reader(reader_kind, path)
    reader.validate_catalog()
    metadata = reader.metadata()
    return LocalRelBenchDataset(
        name=name,
        path=path,
        reader=reader,
        val_timestamp=_timestamp(metadata.get("val_timestamp")),
        test_timestamp=_timestamp(metadata.get("test_timestamp")),
    )


def _build_task(metadata: TaskMetadata, splits: dict[str, pd.DataFrame]) -> LocalTask:
    task_type = TaskType(_required(metadata.task_type, "task_type"))
    hidden_columns = _parse_hidden_columns(metadata.extra.get("hidden_columns"))
    time_col = _optional_string(metadata.time_column)
    kind = _optional_string(metadata.extra.get("kind"))
    if task_type == TaskType.RECOMMENDATION:
        return LocalRecommendationTask(
            name=metadata.name,
            task_type=task_type,
            src_entity_table=_required(metadata.entity_table, "entity_table"),
            src_entity_col=_required(metadata.entity_column, "entity_column"),
            dst_entity_table=_required(
                metadata.extra.get("dst_entity_table"), "dst_entity_table"
            ),
            dst_entity_col=_required(
                metadata.extra.get("dst_entity_column"), "dst_entity_column"
            ),
            eval_k=_optional_int(metadata.extra.get("eval_k")),
            time_col=time_col,
            kind=kind,
            remove_columns=hidden_columns,
            _splits=splits,
        )
    return LocalEntityTask(
        name=metadata.name,
        task_type=task_type,
        entity_table=_required(metadata.entity_table, "entity_table"),
        entity_col=_required(metadata.entity_column, "entity_column"),
        target_col=_required(metadata.target_column, "target_column"),
        time_col=time_col,
        kind=kind,
        remove_columns=hidden_columns,
        _splits=splits,
    )


def _timestamp(value: str | None) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    return timestamp if isinstance(timestamp, pd.Timestamp) else None


def _required(value: object, field: str) -> str:
    if value is None or value == "":
        raise ValueError(f"Task catalog is missing required field {field!r}")
    return str(value)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, str)):
        raise TypeError(
            f"Expected an integer-compatible value, got {type(value).__name__}"
        )
    return int(value)


def _parse_hidden_columns(value: object) -> list[tuple[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("hidden_columns must be a list of [table, column] pairs")

    result: list[tuple[str, str]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError("hidden_columns must contain [table, column] pairs")
        result.append((str(pair[0]), str(pair[1])))
    return result
