"""Fetch only sampled node features from SQL with a bounded hybrid cache."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pandas as pd
import torch

from baseline.cache_baseline.connector import (
    BaseDatabaseReader,
    ColumnSchema,
    TableSchema,
)
from baseline.cache_baseline.connector.base import quote_identifier
from baseline.cache_baseline.connector.decoding import decode_frame
from baseline.cache_baseline.sampling import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
)

from .cache import FeatureBlockCache, FeatureBlockKey
from .contracts import (
    EntityFeatureBatch,
    FeatureBatch,
    FeatureFrame,
    FetchedNodeFeatures,
    FetchedSubgraph,
    RecommendationFeatureBatch,
)

_NODE_ID_COLUMN = "__node_id__"


@dataclass(frozen=True)
class FeatureFetchStats:
    requested_rows: int = 0
    unique_rows: int = 0
    featureless_rows: int = 0
    queried_rows: int = 0
    cache_served_rows: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    dense_queries: int = 0
    sparse_queries: int = 0
    windows: int = 0
    batches: int = 0

    @property
    def duplicate_factor(self) -> float:
        return self.requested_rows / self.unique_rows if self.unique_rows else 0.0

    @property
    def sql_amplification(self) -> float:
        sql_required_rows = (
            self.unique_rows - self.cache_served_rows - self.featureless_rows
        )
        return self.queried_rows / sql_required_rows if sql_required_rows else 0.0

    @property
    def row_cache_coverage(self) -> float:
        return self.cache_served_rows / self.unique_rows if self.unique_rows else 0.0

    def __add__(self, other: FeatureFetchStats) -> FeatureFetchStats:
        return FeatureFetchStats(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class _TableReadStats:
    featureless_rows: int = 0
    queried_rows: int = 0
    cache_served_rows: int = 0
    dense_queries: int = 0
    sparse_queries: int = 0


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
        self.last_stats = FeatureFetchStats()
        self.stats = FeatureFetchStats()

    def fetch(self, plan: EntitySamplePlan | RecommendationSamplePlan) -> FeatureBatch:
        return self.fetch_many([plan])[0]

    def fetch_many(
        self, plans: list[EntitySamplePlan | RecommendationSamplePlan]
    ) -> list[FeatureBatch]:
        if not plans:
            self.last_stats = FeatureFetchStats()
            return []
        before_hits = self.cache.hits
        before_misses = self.cache.misses
        grouped: dict[str, list[tuple[tuple[int, str, str], torch.Tensor]]] = (
            defaultdict(list)
        )
        samples: dict[tuple[int, str], SampledSubgraph] = {}
        requested = 0
        for plan_index, plan in enumerate(plans):
            for branch, sample in _plan_subgraphs(plan):
                samples[(plan_index, branch)] = sample
                for node_type, node_ids in sample.node_ids.items():
                    grouped[node_type].append(
                        ((plan_index, branch, node_type), node_ids)
                    )
                    requested += len(node_ids)

        shared_frames: dict[str, FeatureFrame] = {}
        inverses: dict[tuple[int, str, str], torch.Tensor] = {}
        unique_rows = 0
        read_stats = _TableReadStats()
        for node_type, references in grouped.items():
            combined = torch.cat([node_ids for _, node_ids in references])
            unique_ids, inverse = torch.unique(
                combined, sorted=True, return_inverse=True
            )
            frame, table_stats = self._fetch_unique_table(node_type, unique_ids)
            shared_frames[node_type] = frame
            unique_rows += len(unique_ids)
            read_stats = _add_read_stats(read_stats, table_stats)
            offset = 0
            for reference, node_ids in references:
                stop = offset + len(node_ids)
                inverses[reference] = inverse[offset:stop]
                offset = stop

        def fetched_subgraph(plan_index: int, branch: str) -> FetchedSubgraph:
            sample = samples[(plan_index, branch)]
            return FetchedSubgraph(
                sample=sample,
                frames={
                    node_type: FetchedNodeFeatures(
                        shared_frames[node_type],
                        inverses[(plan_index, branch, node_type)],
                    )
                    for node_type in sample.node_ids
                },
            )

        results: list[FeatureBatch] = []
        for plan_index, plan in enumerate(plans):
            if isinstance(plan, EntitySamplePlan):
                results.append(
                    EntityFeatureBatch(
                        key=plan.key,
                        subgraph=fetched_subgraph(plan_index, "entity"),
                        target=plan.target,
                    )
                )
            else:
                results.append(
                    RecommendationFeatureBatch(
                        key=plan.key,
                        source=fetched_subgraph(plan_index, "source"),
                        positive=fetched_subgraph(plan_index, "positive"),
                        negative=fetched_subgraph(plan_index, "negative"),
                    )
                )
        self.last_stats = FeatureFetchStats(
            requested_rows=requested,
            unique_rows=unique_rows,
            featureless_rows=read_stats.featureless_rows,
            queried_rows=read_stats.queried_rows,
            cache_served_rows=read_stats.cache_served_rows,
            cache_hits=self.cache.hits - before_hits,
            cache_misses=self.cache.misses - before_misses,
            dense_queries=read_stats.dense_queries,
            sparse_queries=read_stats.sparse_queries,
            windows=1,
            batches=len(plans),
        )
        self.stats += self.last_stats
        return results

    def _fetch_table(
        self,
        table_name: str,
        node_ids: torch.Tensor,
    ) -> tuple[pd.DataFrame, int]:
        unique_ids, inverse = torch.unique(node_ids, sorted=True, return_inverse=True)
        frame, stats = self._fetch_unique_table(table_name, unique_ids)
        return frame.iloc[inverse.tolist()].reset_index(drop=True), stats.queried_rows

    def _fetch_unique_table(
        self,
        table_name: str,
        node_ids: torch.Tensor,
    ) -> tuple[pd.DataFrame, _TableReadStats]:
        schema = self.reader.schemas()[table_name]
        columns = self.feature_columns(schema)
        ids = [int(value) for value in node_ids.tolist()]
        if not columns:
            return pd.DataFrame({"__const__": [1.0] * len(ids)}), _TableReadStats(
                featureless_rows=len(ids)
            )
        if not ids:
            return (
                pd.DataFrame(
                    {column.name: pd.Series(dtype="object") for column in columns}
                ),
                _TableReadStats(),
            )

        ids_by_block: dict[int, list[int]] = defaultdict(list)
        for node_id in ids:
            ids_by_block[node_id // self.block_size].append(node_id)

        frames: list[pd.DataFrame] = []
        sparse_ids: list[int] = []
        queried_rows = 0
        cache_served_rows = 0
        dense_queries = 0
        sparse_queries = 0
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
                cache_served_rows += len(block_ids)
                continue
            density = len(block_ids) / self.block_size
            if density < self.block_density_threshold:
                sparse_ids.extend(block_ids)
                continue
            frame = self._read_block(schema, columns, block)
            queried_rows += len(frame)
            dense_queries += 1
            self.cache.put(key, frame)
            frames.append(frame)

        for offset in range(0, len(sparse_ids), self.max_ids_per_query):
            frame = self._read_ids(
                schema,
                columns,
                sparse_ids[offset : offset + self.max_ids_per_query],
            )
            queried_rows += len(frame)
            sparse_queries += 1
            frames.append(frame)

        available = pd.concat(frames, axis=0) if frames else pd.DataFrame()
        missing = sorted(set(ids).difference(available.index))
        if missing:
            raise KeyError(f"Missing node IDs in {table_name!r}: {missing[:10]}")
        return (
            available.reindex(ids).reset_index(drop=True),
            _TableReadStats(
                queried_rows=queried_rows,
                cache_served_rows=cache_served_rows,
                dense_queries=dense_queries,
                sparse_queries=sparse_queries,
            ),
        )

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
        lookup_id = _lookup_id_expression(schema)
        lookup_start = _to_lookup_id(schema, start)
        lookup_stop = _to_lookup_id(schema, stop)
        return self._read_where(
            schema,
            columns,
            f"{lookup_id} BETWEEN {lookup_start} AND {lookup_stop}",
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
        lookup_id = _lookup_id_expression(schema)
        return self._read_where(schema, columns, f"{lookup_id} IN ({values})")

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
    return (
        quote_identifier(schema.primary_key)
        if schema.primary_key is not None
        else "rowid"
    )


def _plan_subgraphs(
    plan: EntitySamplePlan | RecommendationSamplePlan,
) -> tuple[tuple[str, SampledSubgraph], ...]:
    if isinstance(plan, EntitySamplePlan):
        return (("entity", plan.subgraph),)
    return (
        ("source", plan.source),
        ("positive", plan.positive),
        ("negative", plan.negative),
    )


def _add_read_stats(left: _TableReadStats, right: _TableReadStats) -> _TableReadStats:
    return _TableReadStats(
        featureless_rows=left.featureless_rows + right.featureless_rows,
        queried_rows=left.queried_rows + right.queried_rows,
        cache_served_rows=left.cache_served_rows + right.cache_served_rows,
        dense_queries=left.dense_queries + right.dense_queries,
        sparse_queries=left.sparse_queries + right.sparse_queries,
    )


def _to_lookup_id(schema: TableSchema, node_id: int) -> int:
    return node_id if schema.primary_key is not None else node_id + 1
