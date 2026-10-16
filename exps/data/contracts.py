"""Typed interfaces consumed from RelBench dataset and task implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd
from relbench.base import Database, Table, TaskType


class RelBenchDataset(Protocol):
    """Dataset surface required by the SQL materializer."""

    @property
    def val_timestamp(self) -> pd.Timestamp: ...

    @property
    def test_timestamp(self) -> pd.Timestamp: ...

    def get_db(self, upto_test_timestamp: bool = True) -> Database: ...

    def get_task_names(self) -> list[str]: ...

    def load_task(self, task_name: str) -> RelBenchTask: ...


class RelBenchTask(Protocol):
    """Common task surface required while materializing split tables."""

    @property
    def task_type(self) -> TaskType: ...

    @property
    def time_col(self) -> str | None: ...

    @property
    def timedelta(self) -> pd.Timedelta: ...

    @property
    def num_eval_timestamps(self) -> int: ...

    @property
    def kind(self) -> str | None: ...

    def get_table(
        self,
        split: str,
        mask_input_cols: bool | None = None,
    ) -> Table: ...

    def hidden_columns(self) -> Sequence[tuple[str, str]]: ...


class RelBenchEntityTask(RelBenchTask, Protocol):
    @property
    def entity_table(self) -> str: ...

    @property
    def entity_col(self) -> str: ...

    @property
    def target_col(self) -> str: ...


class RelBenchRecommendationTask(RelBenchTask, Protocol):
    @property
    def src_entity_table(self) -> str: ...

    @property
    def src_entity_col(self) -> str: ...

    @property
    def dst_entity_table(self) -> str: ...

    @property
    def dst_entity_col(self) -> str: ...

    @property
    def eval_k(self) -> int: ...
