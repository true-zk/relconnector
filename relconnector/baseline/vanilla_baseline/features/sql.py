"""Fetch only sampled node features from SQL with a bounded hybrid cache."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pandas as pd
import torch

from baseline.vanilla_baseline.connector import (
    BaseDatabaseReader,
    ColumnSchema,
    TableSchema,
)
from baseline.vanilla_baseline.connector.base import quote_identifier
from baseline.vanilla_baseline.connector.decoding import decode_frame
from baseline.vanilla_baseline.sampling import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
)

from .cache import FeatureBlockCache, FeatureBlockKey
from .contracts import (
    EntityFeatureBatch,
    FeatureBatch,
    FeatureFrame,
    FetchedSubgraph,
    RecommendationFeatureBatch,
)

_NODE_ID_COLUMN = "__node_id__"


@dataclass(frozen=True)
class FeatureFetchStats:
    requested_rows: int
    queried_rows: int
    cache_hits: int
    cache_misses: int


class SqlFeatureFetcher:
    """Fetch sampled rows by dense node ID and preserve sampler-local ordering.

    Dense requests load and cache a complete row block. Sparse random requests are
    merged into batched ``IN`` queries to avoid thousands of tiny range scans and
    excessive block over-fetch.
    """

    def __init__(
        self,
        reader: BaseDatabaseReader,
        *,
        hidden_columns: Iterable[tuple[str, str]] = (),
        feature_columns: dict[str, tuple[ColumnSchema, ...]] | None = None,
        block_size: int = 4096,
        cache_bytes: int = 0,
        database_version: str = "local",
        block_density_threshold: float = 0.25,
        max_ids_per_query: int = 10_000,
    ) -> None:
        if block_size <= 0 or max_ids_per_query <= 0:
            raise ValueError("batch sizes must be greater than zero")
        if not 0 < block_density_threshold <= 1:
            raise ValueError("block_density_threshold must be in (0, 1]")
        self.reader = reader
        self.block_size = block_size
        self.database_version = database_version
        self.hidden_columns = set(hidden_columns)
        self._feature_columns = feature_columns
        self.block_density_threshold = block_density_threshold
        self.max_ids_per_query = max_ids_per_query
        self.cache = FeatureBlockCache(cache_bytes)
        self.last_stats = FeatureFetchStats(0, 0, 0, 0)

    def fetch(self, plan: EntitySamplePlan | RecommendationSamplePlan) -> FeatureBatch:
        before_hits = self.cache.hits
        before_misses = self.cache.misses
        queried_rows = 0

        def fetch_subgraph(sample: SampledSubgraph) -> FetchedSubgraph:
            nonlocal queried_rows
            frames: dict[str, FeatureFrame] = {}
            for node_type, node_ids in sample.node_ids.items():
                frame, rows = self._fetch_table(node_type, node_ids)
                frames[node_type] = frame
                queried_rows += rows
            return FetchedSubgraph(sample=sample, frames=frames)

        if isinstance(plan, EntitySamplePlan):
            result: FeatureBatch = EntityFeatureBatch(
                key=plan.key,
                subgraph=fetch_subgraph(plan.subgraph),
                target=plan.target,
            )
            requested = sum(len(ids) for ids in plan.subgraph.node_ids.values())
        else:
            result = RecommendationFeatureBatch(
                key=plan.key,
                source=fetch_subgraph(plan.source),
                positive=fetch_subgraph(plan.positive),
                negative=fetch_subgraph(plan.negative),
            )
            requested = sum(
                len(ids)
                for sample in (plan.source, plan.positive, plan.negative)
                for ids in sample.node_ids.values()
            )
        self.last_stats = FeatureFetchStats(
            requested_rows=requested,
            queried_rows=queried_rows,
            cache_hits=self.cache.hits - before_hits,
            cache_misses=self.cache.misses - before_misses,
        )
        return result

    def _fetch_table(
        self,
        table_name: str,
        node_ids: torch.Tensor,
    ) -> tuple[pd.DataFrame, int]:
        schema = self.reader.schemas()[table_name]
        columns = self.feature_columns(schema)
        ids = [int(value) for value in node_ids.tolist()]
        if not columns:
            return pd.DataFrame({"__const__": [1.0] * len(ids)}), 0
        if not ids:
            return pd.DataFrame(
                {column.name: pd.Series(dtype="object") for column in columns}
            ), 0

        ids_by_block: dict[int, list[int]] = defaultdict(list)
        for node_id in set(ids):
            ids_by_block[node_id // self.block_size].append(node_id)

        frames: list[pd.DataFrame] = []
        sparse_ids: list[int] = []
        queried_rows = 0
        column_names = tuple(column.name for column in columns)
        for block, block_ids in ids_by_block.items():
            key = FeatureBlockKey(
                self.database_version,
                table_name,
                column_names,
                block,
            )
            frame = self.cache.get(key)
            if frame is not None:
                frames.append(frame)
                continue
            density = len(block_ids) / self.block_size
            if density < self.block_density_threshold:
                sparse_ids.extend(block_ids)
                continue
            frame = self._read_block(schema, columns, block)
            queried_rows += len(frame)
            self.cache.put(key, frame)
            frames.append(frame)

        for offset in range(0, len(sparse_ids), self.max_ids_per_query):
            frame = self._read_ids(
                schema,
                columns,
                sparse_ids[offset : offset + self.max_ids_per_query],
            )
            queried_rows += len(frame)
            frames.append(frame)

        available = pd.concat(frames, axis=0) if frames else pd.DataFrame()
        missing = sorted(set(ids).difference(available.index))
        if missing:
            raise KeyError(f"Missing node IDs in {table_name!r}: {missing[:10]}")
        return available.reindex(ids).reset_index(drop=True), queried_rows

    def feature_columns(
        self,
        schema: TableSchema,
    ) -> tuple[ColumnSchema, ...]:
        if self._feature_columns is not None:
            return self._feature_columns[schema.name]
        excluded = {foreign_key.column for foreign_key in schema.foreign_keys}
        if schema.primary_key is not None:
            excluded.add(schema.primary_key)
        excluded.update(
            column for table, column in self.hidden_columns if table == schema.name
        )
        return tuple(column for column in schema.columns if column.name not in excluded)

    def _read_block(
        self,
        schema: TableSchema,
        columns: tuple[ColumnSchema, ...],
        block: int,
    ) -> pd.DataFrame:
        start = block * self.block_size
        stop = start + self.block_size - 1
        node_id = _lookup_id_expression(schema)
        start = _to_lookup_id(schema, start)
        stop = _to_lookup_id(schema, stop)
        return self._read_where(
            schema,
            columns,
            f"{node_id} BETWEEN {start} AND {stop}",
        )

    def _read_ids(
        self,
        schema: TableSchema,
        columns: tuple[ColumnSchema, ...],
        node_ids: list[int],
    ) -> pd.DataFrame:
        if not node_ids:
            return pd.DataFrame()
        values = ",".join(str(_to_lookup_id(schema, node_id)) for node_id in node_ids)
        node_id = _lookup_id_expression(schema)
        return self._read_where(schema, columns, f"{node_id} IN ({values})")

    def _read_where(
        self,
        schema: TableSchema,
        columns: tuple[ColumnSchema, ...],
        predicate: str,
    ) -> pd.DataFrame:
        node_id = _node_id_expression(schema)
        projection = ", ".join(
            [f"{node_id} AS {_NODE_ID_COLUMN}"]
            + [quote_identifier(column.name) for column in columns]
        )
        raw = self.reader.read_query(
            f"SELECT {projection} FROM {quote_identifier(schema.name)} "
            f"WHERE {predicate} ORDER BY {_lookup_id_expression(schema)}"
        )
        ids = cast(pd.Series, raw.pop(_NODE_ID_COLUMN)).astype("int64")
        decoded = decode_frame(raw, columns)
        decoded.index = ids.to_numpy()
        return decoded


def _node_id_expression(schema: TableSchema) -> str:
    return (
        quote_identifier(schema.primary_key)
        if schema.primary_key is not None
        else "rowid - 1"
    )


def _lookup_id_expression(schema: TableSchema) -> str:
    return quote_identifier(schema.primary_key) if schema.primary_key else "rowid"


def _to_lookup_id(schema: TableSchema, node_id: int) -> int:
    return node_id if schema.primary_key else node_id + 1
