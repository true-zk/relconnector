"""Bounded-memory task seed scheduling from local SQL tables."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import pandas as pd
import torch
from relbench.base import TaskType
from relbench.modeling.utils import to_unix_time

from baseline.cache_baseline.connector import BaseDatabaseReader
from baseline.cache_baseline.connector.base import quote_identifier
from baseline.cache_baseline.connector.decoding import decode_frame

from .spec import TaskSpec


@dataclass(frozen=True, order=True)
class BatchKey:
    epoch: int
    batch: int


@dataclass(frozen=True)
class EntitySeedBatch:
    key: BatchKey
    node_type: str
    node_ids: torch.Tensor
    seed_time: torch.Tensor | None
    target: torch.Tensor


@dataclass(frozen=True)
class RecommendationSeedBatch:
    key: BatchKey
    src_node_type: str
    dst_node_type: str
    src_node_ids: torch.Tensor
    positive_dst_ids: torch.Tensor
    seed_time: torch.Tensor | None


SeedBatch: TypeAlias = EntitySeedBatch | RecommendationSeedBatch


class SqlSeedReader:
    """Stream each train seed exactly once per epoch using shuffled row blocks."""

    def __init__(
        self,
        reader: BaseDatabaseReader,
        task: TaskSpec,
        *,
        batch_size: int,
        shuffle_block_size: int = 65_536,
        seed: int = 42,
        max_batches: int | None = None,
    ) -> None:
        if batch_size <= 0 or shuffle_block_size <= 0:
            raise ValueError("batch sizes must be greater than zero")
        self.reader = reader
        self.task = task
        self.batch_size = batch_size
        self.shuffle_block_size = max(batch_size, shuffle_block_size)
        self.seed = seed
        self.max_batches = max_batches
        self._schema = reader.schemas()[task.table_name]
        if self._schema.primary_key is None:
            raise ValueError(f"Task table {task.table_name!r} requires a primary key")
        if self._schema.split_column is None:
            raise ValueError(f"Task table {task.table_name!r} requires a split column")

    def iter_epoch(self, epoch: int) -> Iterator[SeedBatch]:
        row_ids = self._train_row_id_bounds()
        if row_ids is None:
            return
        minimum, maximum = row_ids
        blocks = list(range(minimum, maximum + 1, self.shuffle_block_size))
        rng = random.Random(_derive_seed(self.seed, epoch))
        rng.shuffle(blocks)

        batch_id = 0
        for start in blocks:
            stop = min(start + self.shuffle_block_size - 1, maximum)
            frame = self._read_block(start, stop)
            order = np.arange(len(frame))
            np.random.default_rng(_derive_seed(self.seed, epoch, start)).shuffle(order)
            frame = frame.iloc[order].reset_index(drop=True)
            for offset in range(0, len(frame), self.batch_size):
                rows = frame.iloc[offset : offset + self.batch_size]
                if len(rows) == 0:
                    continue
                key = BatchKey(epoch=epoch, batch=batch_id)
                batch_id += 1
                yield self._to_seed_batch(key, rows, rng)
                if self.max_batches is not None and batch_id >= self.max_batches:
                    return

    def _train_row_id_bounds(self) -> tuple[int, int] | None:
        pkey = quote_identifier(self._schema.primary_key or "")
        split = quote_identifier(self._schema.split_column or "")
        table = quote_identifier(self.task.table_name)
        frame = self.reader.read_query(
            f"SELECT MIN({pkey}) AS min_id, MAX({pkey}) AS max_id "
            f"FROM {table} WHERE {split} = 'train'"
        )
        minimum, maximum = frame.iloc[0, 0], frame.iloc[0, 1]
        if pd.isna(minimum) or pd.isna(maximum):
            return None
        return int(minimum), int(maximum)

    def _read_block(self, start: int, stop: int) -> pd.DataFrame:
        pkey = quote_identifier(self._schema.primary_key or "")
        split = quote_identifier(self._schema.split_column or "")
        table = quote_identifier(self.task.table_name)
        columns = list(self._required_columns())
        projection = ", ".join(quote_identifier(column) for column in columns)
        raw = self.reader.read_query(
            f"SELECT {projection} FROM {table} "
            f"WHERE {split} = 'train' AND {pkey} BETWEEN {start} AND {stop} "
            f"ORDER BY {pkey}"
        )
        schemas = [column for column in self._schema.columns if column.name in columns]
        return decode_frame(raw, schemas)

    def _required_columns(self) -> tuple[str, ...]:
        columns = [self.task.entity_column]
        if self.task.time_column is not None:
            columns.append(self.task.time_column)
        target = (
            self.task.dst_entity_column
            if self.task.is_recommendation
            else self.task.target_column
        )
        if target is not None:
            columns.append(target)
        return tuple(dict.fromkeys(columns))

    def _to_seed_batch(
        self,
        key: BatchKey,
        rows: pd.DataFrame,
        rng: random.Random,
    ) -> SeedBatch:
        seed_time = None
        if self.task.time_column is not None:
            seed_time = torch.from_numpy(
                to_unix_time(
                    cast(pd.Series, rows[self.task.time_column]).dt.as_unit("ns")
                )
            )
        source = torch.from_numpy(
            rows[self.task.entity_column].to_numpy(dtype=np.int64).copy()
        )
        if self.task.is_recommendation:
            if (
                self.task.dst_entity_table is None
                or self.task.dst_entity_column is None
            ):
                raise ValueError(
                    f"Recommendation task {self.task.name!r} is incomplete"
                )
            destinations = cast(pd.Series, rows[self.task.dst_entity_column])
            positive = torch.tensor(
                [rng.choice(list(value)) for value in destinations],
                dtype=torch.long,
            )
            return RecommendationSeedBatch(
                key=key,
                src_node_type=self.task.entity_table,
                dst_node_type=self.task.dst_entity_table,
                src_node_ids=source,
                positive_dst_ids=positive,
                seed_time=seed_time,
            )

        if self.task.target_column is None:
            raise ValueError(f"Entity task {self.task.name!r} has no target column")
        target_series = cast(pd.Series, rows[self.task.target_column])
        target = _target_tensor(target_series, self.task.task_type)
        return EntitySeedBatch(
            key=key,
            node_type=self.task.entity_table,
            node_ids=source,
            seed_time=seed_time,
            target=target,
        )


def _target_tensor(series: pd.Series, task_type: TaskType) -> torch.Tensor:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        values = series.map(_binary_value).to_numpy(dtype=np.float32)
        return torch.from_numpy(values)
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return torch.from_numpy(series.to_numpy(dtype=np.int64))
    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return torch.from_numpy(np.stack(series.tolist()))
    return torch.from_numpy(series.to_numpy(dtype=np.float32))


def _binary_value(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"t", "true", "1", "yes"}:
            return 1.0
        if normalized in {"f", "false", "0", "no"}:
            return 0.0
        raise ValueError(f"Unsupported binary label {value!r}")
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"Unsupported binary label type: {type(value).__name__}")


def _derive_seed(*parts: int) -> int:
    value = 0x9E3779B97F4A7C15
    for part in parts:
        value ^= int(part) + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2)
    return value & ((1 << 63) - 1)
