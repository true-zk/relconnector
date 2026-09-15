"""Shared TensorFrame schema and bounded SQL statistics preparation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch_frame import TensorFrame, stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data.dataset import DataFrameToTensorFrameConverter
from torch_frame.data.mapper import MultiCategoricalTensorMapper, TimestampTensorMapper
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.stats import StatType
from torch_frame.utils import infer_df_stype

from relconnector.connector import BaseDatabaseReader, ColumnSchema, TableSchema
from relconnector.connector.base import quote_identifier
from relconnector.connector.decoding import decode_frame

from .text import TextEmbedder, normalize_text

ColumnStats = dict[StatType, object]
TableStats = dict[str, ColumnStats]


@dataclass(frozen=True)
class TableFeatureSchema:
    columns: tuple[ColumnSchema, ...]
    col_to_stype: Mapping[str, stype]
    col_stats: Mapping[str, ColumnStats]


@dataclass(frozen=True)
class TensorFrameFeatureSchema:
    tables: Mapping[str, TableFeatureSchema]
    fingerprint: str

    @property
    def col_stats_dict(self) -> dict[str, TableStats]:
        return {
            table: {column: dict(stats) for column, stats in spec.col_stats.items()}
            for table, spec in self.tables.items()
        }

    @property
    def col_names_dict(self) -> dict[str, dict[stype, list[str]]]:
        return {
            table: _merged_col_names(spec.col_to_stype)
            for table, spec in self.tables.items()
        }

    @property
    def feature_columns(self) -> dict[str, tuple[ColumnSchema, ...]]:
        return {table: spec.columns for table, spec in self.tables.items()}


class SqlTensorFrameSchemaBuilder:
    """Infer TensorFrame types and statistics before the training loop."""

    def __init__(
        self,
        reader: BaseDatabaseReader,
        *,
        hidden_columns: Iterable[tuple[str, str]] = (),
        scan_batch_size: int = 1_000_000,
        inference_rows: int = 1_000,
        text_embedding_dim: int = 300,
        encode_text: bool = True,
        cutoff: pd.Timestamp | None = None,
    ) -> None:
        if min(scan_batch_size, inference_rows, text_embedding_dim) <= 0:
            raise ValueError("schema preparation sizes must be positive")
        self.reader = reader
        self.hidden_columns = set(hidden_columns)
        self.scan_batch_size = scan_batch_size
        self.inference_rows = inference_rows
        self.text_embedding_dim = text_embedding_dim
        self.encode_text = encode_text
        self.cutoff = cutoff

    def build(self) -> TensorFrameFeatureSchema:
        tables: dict[str, TableFeatureSchema] = {}
        for schema in self.reader.schemas().values():
            if schema.kind != "data":
                continue
            candidates = self._candidate_columns(schema)
            inferred = self._infer_stypes(schema, candidates)
            columns = tuple(column for column in candidates if column.name in inferred)
            if not columns:
                inferred = {"__const__": stype.numerical}
                stats: TableStats = {"__const__": _numerical_stats([1.0])}
            else:
                stats = {
                    column.name: self._column_stats(
                        schema, column, inferred[column.name]
                    )
                    for column in columns
                }
            tables[schema.name] = TableFeatureSchema(columns, inferred, stats)
        return TensorFrameFeatureSchema(tables, _fingerprint(tables))

    def _candidate_columns(self, schema: TableSchema) -> tuple[ColumnSchema, ...]:
        excluded = {foreign_key.column for foreign_key in schema.foreign_keys}
        if schema.primary_key is not None:
            excluded.add(schema.primary_key)
        excluded.update(
            column for table, column in self.hidden_columns if table == schema.name
        )
        return tuple(column for column in schema.columns if column.name not in excluded)

    def _infer_stypes(
        self,
        schema: TableSchema,
        columns: tuple[ColumnSchema, ...],
    ) -> dict[str, stype]:
        if not columns:
            return {}
        projection = ", ".join(quote_identifier(column.name) for column in columns)
        node_id = (
            quote_identifier(schema.primary_key)
            if schema.primary_key is not None
            else "rowid - 1"
        )
        where_clause = self._time_filter(schema)
        raw = self.reader.read_query(
            f"SELECT {projection} FROM {quote_identifier(schema.name)}"
            f"{where_clause} ORDER BY {node_id} LIMIT {self.inference_rows}"
        )
        inferred = infer_df_stype(decode_frame(raw, columns))
        for column, semantic_type in tuple(inferred.items()):
            if semantic_type == stype.embedding:
                inferred[column] = stype.multicategorical
            elif semantic_type == stype.text_embedded and not self.encode_text:
                inferred[column] = stype.categorical
        return inferred

    def _column_stats(
        self,
        table: TableSchema,
        column: ColumnSchema,
        semantic_type: stype,
    ) -> ColumnStats:
        if semantic_type == stype.text_embedded:
            return {StatType.EMB_DIM: self.text_embedding_dim}
        if semantic_type == stype.numerical:
            return self._numerical_stats(table, column)
        if semantic_type == stype.categorical:
            return self._categorical_stats(table, column)
        if semantic_type == stype.multicategorical:
            return self._multicategorical_stats(table, column)
        if semantic_type == stype.timestamp:
            return self._timestamp_stats(table, column)
        raise NotImplementedError(
            f"Unsupported TensorFrame stype {semantic_type.value!r} for "
            f"{table.name}.{column.name}"
        )

    def _series_chunks(
        self,
        table: TableSchema,
        column: ColumnSchema,
        *,
        order_by: str | None = None,
    ) -> Iterable[pd.Series]:
        identifier = quote_identifier(column.name)
        where_clause = self._time_filter(table)
        conjunction = " AND" if where_clause else " WHERE"
        query = (
            f"SELECT {identifier} FROM {quote_identifier(table.name)}"
            f"{where_clause}{conjunction} {identifier} IS NOT NULL"
        )
        if order_by is not None:
            query += f" ORDER BY {order_by}"
        for raw in self.reader.iter_query(query, batch_size=self.scan_batch_size):
            yield cast(pd.Series, decode_frame(raw, (column,))[column.name])

    def _numerical_stats(
        self,
        table: TableSchema,
        column: ColumnSchema,
    ) -> ColumnStats:
        count = 0
        mean = 0.0
        squared_delta = 0.0
        for series in self._series_chunks(table, column):
            values = cast(pd.Series, pd.to_numeric(series, errors="coerce")).to_numpy(
                dtype="float64", copy=False
            )
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            chunk_count = len(values)
            chunk_mean = float(values.mean())
            chunk_m2 = float(np.square(values - chunk_mean).sum())
            total = count + chunk_count
            delta = chunk_mean - mean
            squared_delta += chunk_m2 + delta * delta * count * chunk_count / total
            mean += delta * chunk_count / total
            count = total
        if count == 0:
            return _numerical_stats([])
        return {
            StatType.MEAN: mean,
            StatType.STD: math.sqrt(squared_delta / count),
            StatType.QUANTILES: self._numerical_quantiles(table, column, count),
        }

    def _numerical_quantiles(
        self,
        table: TableSchema,
        column: ColumnSchema,
        count: int,
    ) -> list[float]:
        positions = [(count - 1) * q for q in (0, 0.25, 0.5, 0.75, 1)]
        required = {math.floor(position) for position in positions}
        required.update(math.ceil(position) for position in positions)
        selected: dict[int, float] = {}
        index = 0
        identifier = quote_identifier(column.name)
        for series in self._series_chunks(
            table, column, order_by=f"CAST({identifier} AS REAL)"
        ):
            values = cast(pd.Series, pd.to_numeric(series, errors="coerce")).to_numpy(
                dtype="float64", copy=False
            )
            for value in values:
                if not np.isfinite(value):
                    continue
                if index in required:
                    selected[index] = float(value)
                index += 1
        output = []
        for position in positions:
            lower, upper = math.floor(position), math.ceil(position)
            fraction = position - lower
            output.append(selected[lower] * (1 - fraction) + selected[upper] * fraction)
        return output

    def _categorical_stats(
        self,
        table: TableSchema,
        column: ColumnSchema,
    ) -> ColumnStats:
        counts: Counter[object] = Counter()
        for series in self._series_chunks(table, column):
            counts.update(value for value in series if not _is_null(value))
        pairs = counts.most_common()
        return {
            StatType.COUNT: (
                [value for value, _ in pairs],
                [count for _, count in pairs],
            )
        }

    def _multicategorical_stats(
        self,
        table: TableSchema,
        column: ColumnSchema,
    ) -> ColumnStats:
        counts: Counter[object] = Counter()
        for series in self._series_chunks(table, column):
            for value in series:
                categories = MultiCategoricalTensorMapper.split_by_sep(
                    None if _is_null(value) else value, None
                )
                counts.update(sorted(categories, key=repr))
        pairs = counts.most_common()
        return {
            StatType.MULTI_COUNT: (
                [value for value, _ in pairs],
                [count for _, count in pairs],
            )
        }

    def _timestamp_stats(
        self,
        table: TableSchema,
        column: ColumnSchema,
    ) -> ColumnStats:
        identifier = quote_identifier(column.name)
        count_frame = self.reader.read_query(
            f"SELECT COUNT({identifier}) AS value_count "
            f"FROM {quote_identifier(table.name)}{self._time_filter(table)}"
        )
        count = int(count_frame.iloc[0, 0])
        if count == 0:
            missing = torch.full((7,), -1, dtype=torch.long)
            return {
                StatType.YEAR_RANGE: [-1, -1],
                StatType.NEWEST_TIME: missing.clone(),
                StatType.OLDEST_TIME: missing.clone(),
                StatType.MEDIAN_TIME: missing.clone(),
            }

        selected: dict[int, pd.Timestamp] = {}
        required = {0, count // 2, count - 1}
        index = 0
        for series in self._series_chunks(
            table, column, order_by=f"julianday({identifier}), {identifier}"
        ):
            parsed = cast(
                pd.Series,
                pd.to_datetime(series, errors="coerce", format="mixed"),
            )
            for value in parsed:
                if pd.isna(value):
                    continue
                if index in required:
                    selected[index] = cast(pd.Timestamp, value)
                index += 1
        if index != count:
            raise ValueError(f"Invalid timestamps in {table.name}.{column.name}")

        def encode(value: pd.Timestamp) -> torch.Tensor:
            return TimestampTensorMapper.to_tensor(pd.Series([value])).squeeze(0)

        oldest, median, newest = selected[0], selected[count // 2], selected[count - 1]
        return {
            StatType.YEAR_RANGE: [oldest.year, newest.year],
            StatType.NEWEST_TIME: encode(newest),
            StatType.OLDEST_TIME: encode(oldest),
            StatType.MEDIAN_TIME: encode(median),
        }

    def _time_filter(self, table: TableSchema) -> str:
        if self.cutoff is None or table.time_column is None:
            return ""
        timestamp = self.cutoff.isoformat().replace("'", "''")
        return (
            f" WHERE julianday({quote_identifier(table.time_column)}) "
            f"<= julianday('{timestamp}')"
        )


class TensorFrameEncoder:
    """Convert full or sampled frames using one immutable feature schema."""

    def __init__(
        self,
        schema: TensorFrameFeatureSchema,
        text_embedder: TextEmbedder | None,
        *,
        text_batch_size: int,
    ) -> None:
        if text_batch_size <= 0:
            raise ValueError("text_batch_size must be positive")
        self.schema = schema
        has_text = any(
            kind == stype.text_embedded
            for spec in schema.tables.values()
            for kind in spec.col_to_stype.values()
        )
        if has_text and text_embedder is None:
            raise ValueError("Text features require a text embedder")
        text_config = (
            None
            if text_embedder is None
            else TextEmbedderConfig(text_embedder, text_batch_size)
        )
        self._converters = {
            table: DataFrameToTensorFrameConverter(
                col_to_stype=dict(spec.col_to_stype),
                col_stats={
                    column: dict(stats) for column, stats in spec.col_stats.items()
                },
                col_to_sep={
                    column: None
                    for column, kind in spec.col_to_stype.items()
                    if kind == stype.multicategorical
                },
                col_to_text_embedder_cfg={
                    column: text_config
                    for column, kind in spec.col_to_stype.items()
                    if kind == stype.text_embedded and text_config is not None
                },
                col_to_text_tokenizer_cfg={},
                col_to_image_embedder_cfg={},
                col_to_time_format={
                    column: None
                    for column, kind in spec.col_to_stype.items()
                    if kind == stype.timestamp
                },
            )
            for table, spec in schema.tables.items()
        }
        self._empty: dict[str, TensorFrame] = {}

    def encode(self, table: str, frame: pd.DataFrame) -> TensorFrame:
        spec = self.schema.tables[table]
        prepared = _prepare_frame(frame, spec)
        if len(prepared) > 0:
            return self._converters[table](prepared)
        if table not in self._empty:
            empty = self._converters[table](_sentinel_frame(spec))[:0]
            embedding = empty.feat_dict.get(stype.embedding)
            if (
                isinstance(embedding, MultiEmbeddingTensor)
                and embedding.values.ndim == 1
            ):
                empty.feat_dict[stype.embedding] = MultiEmbeddingTensor(
                    num_rows=0,
                    num_cols=embedding.num_cols,
                    values=torch.empty((0, int(embedding.offset[-1]))),
                    offset=embedding.offset,
                )
            self._empty[table] = empty
        return self._empty[table]


def _prepare_frame(frame: pd.DataFrame, spec: TableFeatureSchema) -> pd.DataFrame:
    if not spec.columns:
        return pd.DataFrame({"__const__": np.ones(len(frame), dtype=np.float32)})
    output = frame.reindex(columns=[column.name for column in spec.columns]).copy()
    for column, kind in spec.col_to_stype.items():
        if kind == stype.text_embedded:
            output[column] = cast(pd.Series, output[column]).map(normalize_text)
    return output


def _sentinel_frame(spec: TableFeatureSchema) -> pd.DataFrame:
    if not spec.columns:
        return pd.DataFrame({"__const__": [1.0]})
    values: dict[str, list[object]] = {}
    for column, kind in spec.col_to_stype.items():
        if kind == stype.numerical:
            value: object = 0.0
        elif kind == stype.categorical:
            categories = cast(
                tuple[list[object], list[int]],
                spec.col_stats[column][StatType.COUNT],
            )[0]
            value = categories[0] if categories else None
        elif kind == stype.multicategorical:
            value = []
        elif kind == stype.timestamp:
            value = pd.Timestamp("1970-01-01")
        elif kind == stype.text_embedded:
            value = ""
        else:
            raise NotImplementedError(kind.value)
        values[column] = [value]
    return pd.DataFrame(values)


def _merged_col_names(col_to_stype: Mapping[str, stype]) -> dict[stype, list[str]]:
    output: dict[stype, list[str]] = {}
    for column, kind in col_to_stype.items():
        output.setdefault(kind.parent, []).append(column)
    for names in output.values():
        names.sort()
    return output


def _numerical_stats(values: list[float]) -> ColumnStats:
    if not values:
        return {
            StatType.MEAN: np.nan,
            StatType.STD: np.nan,
            StatType.QUANTILES: [np.nan] * 5,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        StatType.MEAN: float(array.mean()),
        StatType.STD: float(array.std()),
        StatType.QUANTILES: np.quantile(array, q=[0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


def _fingerprint(tables: Mapping[str, TableFeatureSchema]) -> str:
    def serializable(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return str(value)
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): serializable(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    payload = {
        table: {
            "columns": [column.name for column in spec.columns],
            "stypes": {
                column: kind.value for column, kind in sorted(spec.col_to_stype.items())
            },
            "stats": {
                column: {
                    stat.value: serializable(value)
                    for stat, value in sorted(
                        stats.items(), key=lambda pair: pair[0].value
                    )
                }
                for column, stats in sorted(spec.col_stats.items())
            },
        }
        for table, spec in sorted(tables.items())
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_null(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
