"""Read validated task rows and map business keys to graph seed IDs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
from sqlalchemy import column, select, table

from relx.conn import BaseDatabaseReader

from .seeds import BatchKey, EntitySeedBatch, RecommendationSeedBatch
from .validation import ValidatedTask

if TYPE_CHECKING:
    from relx._relx_native import GraphIndex

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class _SeedRow:
    node_id: int
    time_us: int | None
    target: object
    sample_key: tuple[object, ...]


class SqlEntitySeedSource:
    """Materialize task metadata only; feature rows remain in their own tables."""

    def __init__(
        self,
        reader: BaseDatabaseReader,
        task: ValidatedTask,
        graph: GraphIndex,
        *,
        split: SplitName,
        timezone: str,
        batch_size: int,
        shuffle_seed: int,
    ) -> None:
        if task.spec.is_recommendation and type(self) is SqlEntitySeedSource:
            raise ValueError("Use SqlRecommendationSeedSource for recommendation tasks")
        if task.split_by_key is None:
            raise ValueError("Seed source requires data-level task validation")
        if batch_size <= 0 or shuffle_seed < 0:
            raise ValueError("batch_size must be positive and shuffle_seed nonnegative")
        self.split = split
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.node_type = str(task.entity_table)
        self.graph = graph
        self.task = task
        self._rows = self._read_rows(reader, task, graph, timezone)

    @property
    def max_time_us(self) -> int | None:
        times = [row.time_us for row in self._rows if row.time_us is not None]
        return max(times) if times else None

    def labels(self) -> Iterator[object]:
        """Iterate labels already read for this split without reshuffling rows."""
        return (row.target for row in self._rows)

    def _read_rows(
        self,
        reader: BaseDatabaseReader,
        task: ValidatedTask,
        graph: GraphIndex,
        timezone: str,
    ) -> tuple[_SeedRow, ...]:
        spec = task.spec
        names = tuple(
            dict.fromkeys(
                (*task.sample_key, spec.entity_column, spec.label_column)
                + ((spec.time_column,) if spec.time_column else ())
            )
        )
        source = table(
            spec.table.table_name,
            *(column(name) for name in names),
            schema=spec.table.schema_name,
        )
        query = select(*(source.c[name] for name in names)).order_by(
            *(source.c[name] for name in task.sample_key)
        )
        rows: list[_SeedRow] = []
        from relx.preprocess.graph_time import normalize_time_batch

        assignments = task.split_by_key
        assert assignments is not None
        for frame in reader.iter_query(query, batch_size=10_000):
            values = frame[spec.entity_column].tolist()
            lookup = pa.record_batch(
                [pa.array(values)], names=[task.entity_key[0]]
            )
            ids = graph.lookup_nodes(self.node_type, list(task.entity_key), lookup)
            if np.any(ids < 0):
                raise ValueError("Task entity key is absent from the graph")
            times: list[int] | None = None
            if spec.time_column:
                time_batch = pa.RecordBatch.from_pandas(
                    frame[[spec.time_column]], preserve_index=False
                )
                normalized = normalize_time_batch(time_batch, spec.time_column, timezone)
                times = normalized.column("__relx_time_us").to_pylist()
            for position, record in enumerate(frame.to_dict("records")):
                key = tuple(record[name] for name in task.sample_key)
                if assignments[key] != self.split:
                    continue
                rows.append(
                    _SeedRow(
                        int(ids[position]),
                        times[position] if times is not None else None,
                        record[spec.label_column],
                        key,
                    )
                )
        return tuple(rows)

    def iter_epoch(self, epoch: int) -> Iterator[EntitySeedBatch]:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        order = np.arange(len(self._rows))
        if self.split == "train":
            np.random.default_rng(self.shuffle_seed + epoch).shuffle(order)
        for batch_index, start in enumerate(range(0, len(order), self.batch_size)):
            rows = [self._rows[int(index)] for index in order[start:start+self.batch_size]]
            times = (
                np.asarray([row.time_us for row in rows], dtype=np.int64)
                if rows and rows[0].time_us is not None
                else None
            )
            yield EntitySeedBatch(
                BatchKey(epoch, batch_index),
                self.node_type,
                np.asarray([row.node_id for row in rows], dtype=np.int64),
                times,
                np.asarray([row.target for row in rows], dtype=object),
                tuple(row.sample_key for row in rows),
            )


class SqlRecommendationSeedSource(SqlEntitySeedSource):
    """Select one positive destination per task row for pairwise training."""

    def iter_epoch(self, epoch: int) -> Iterator[RecommendationSeedBatch]:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        destination = self.task.destination_table
        key = self.task.destination_key
        if destination is None or key is None:
            raise ValueError("Recommendation destination is missing")
        order = np.arange(len(self._rows))
        rng = np.random.default_rng(self.shuffle_seed + epoch)
        if self.split == "train":
            rng.shuffle(order)
        for batch_index, start in enumerate(range(0, len(order), self.batch_size)):
            rows = [self._rows[int(index)] for index in order[start:start+self.batch_size]]
            chosen = []
            for row in rows:
                values = json.loads(row.target) if isinstance(row.target, str) else row.target
                if not isinstance(values, list) or not values:
                    raise ValueError("Recommendation sample needs positive destinations")
                chosen.append(values[int(rng.integers(len(values)))])
            lookup = pa.record_batch([pa.array(chosen)], names=[key[0]])
            positive_ids = self.graph.lookup_nodes(str(destination), list(key), lookup)
            if np.any(positive_ids < 0):
                raise ValueError("Recommendation destination is absent from graph")
            times = (
                np.asarray([row.time_us for row in rows], dtype=np.int64)
                if rows and rows[0].time_us is not None else None
            )
            yield RecommendationSeedBatch(
                BatchKey(epoch, batch_index), self.node_type, str(destination),
                np.asarray([row.node_id for row in rows], dtype=np.int64),
                positive_ids.astype(np.int64, copy=False), times,
                tuple(row.sample_key for row in rows),
            )
