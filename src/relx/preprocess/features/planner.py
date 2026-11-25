"""Select graph feature columns and infer their semantic types in bounded samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from sqlalchemy import column, select, table
from sqlalchemy.sql.selectable import Select

from relx.config.types import FeaturesConfig, GraphConfig
from relx.conn import BaseDatabaseReader
from relx.core import DatabaseSchema, TableRef
from relx.task import ValidatedTask

from .contracts import FeaturePlan, FeatureProcessor, FeatureSpec, SemanticType
from .defaults import DEFAULT_PROCESSORS, builtin_output

_SEMANTIC_TYPES = frozenset(DEFAULT_PROCESSORS)


def build_feature_plan(
    reader: BaseDatabaseReader,
    schema: DatabaseSchema,
    task: ValidatedTask,
    config: FeaturesConfig,
    graph_config: GraphConfig,
    selected: tuple[TableRef, ...],
    *,
    processors: Mapping[str, FeatureProcessor] | None = None,
    column_processors: Mapping[tuple[TableRef, str], FeatureProcessor] | None = None,
    sample_limit: int = 1_000,
    sample_seed: int = 42,
) -> FeaturePlan:
    """Build a semantic plan without fitting statistics or loading full tables."""
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    if sample_seed < 0:
        raise ValueError("sample_seed must be nonnegative")
    named = processors or {}
    overrides = column_processors or {}
    configured = {(item.table, item.column): item for item in config.columns}
    if len(configured) != len(config.columns):
        raise ValueError("features.columns contains duplicate columns")
    candidates = _candidate_columns(schema, task, selected)
    allowed = {(ref, name) for ref, names in candidates.items() for name in names}
    invalid = (set(configured) | set(overrides)) - allowed
    if invalid:
        names = ", ".join(sorted(f"{ref}.{name}" for ref, name in invalid))
        raise ValueError(f"Features reference excluded, key, or hidden columns: {names}")

    temporal = dict(graph_config.time_columns)
    result: list[FeatureSpec] = []
    for ref in selected:
        names = candidates[ref]
        unresolved = [
            name
            for name in names
            if (ref, name) not in overrides
            and temporal.get(ref) != name
            and (
                (ref, name) not in configured
                or (
                    configured[(ref, name)].semantic_type is None
                    and configured[(ref, name)].processor not in named
                )
            )
        ]
        inferred = _infer_table(
            reader, ref, schema, unresolved, sample_limit, sample_seed
        )
        for name in names:
            item = configured.get((ref, name))
            override = overrides.get((ref, name))
            raw_type = item.semantic_type if item else None
            named_processor = named.get(item.processor) if item and item.processor else None
            semantic = _semantic_type(
                raw_type
                or (override.semantic_type if override is not None else None)
                or (named_processor.semantic_type if named_processor is not None else None)
                or ("timestamp" if temporal.get(ref) == name else inferred.get(name)),
                ref,
                name,
            )
            if override is not None and override.semantic_type != semantic:
                raise ValueError(f"Custom processor for {ref}.{name} has wrong semantic type")
            processor_name = (
                f"custom:{ref}.{name}"
                if override is not None
                else (item.processor if item and item.processor else DEFAULT_PROCESSORS[semantic])
            )
            params: Mapping[str, Any] = item.params if item else {}
            if override is not None:
                output = override.output_spec(params)
            elif processor_name in named:
                custom = named[processor_name]
                if custom.semantic_type != semantic:
                    raise ValueError(f"Processor {processor_name!r} has wrong semantic type")
                output = custom.output_spec(params)
            else:
                output = builtin_output(semantic, processor_name, params)
            result.append(FeatureSpec(ref, name, semantic, processor_name, params, output))
    return FeaturePlan(tuple(result), sample_limit, sample_seed)


def _candidate_columns(
    schema: DatabaseSchema, task: ValidatedTask, selected: tuple[TableRef, ...]
) -> dict[TableRef, tuple[str, ...]]:
    hidden = set(task.hidden_columns)
    candidates: dict[TableRef, tuple[str, ...]] = {}
    for ref in selected:
        table_schema = schema.tables[ref]
        keys = set(table_schema.primary_key or ()) | set(table_schema.identity_key or ())
        for foreign_key in table_schema.foreign_keys:
            keys.update(foreign_key.source_columns)
        candidates[ref] = tuple(
            column.name
            for column in table_schema.columns
            if column.name not in keys and (ref, column.name) not in hidden
        )
    return candidates


def _infer_table(
    reader: BaseDatabaseReader,
    ref: TableRef,
    schema: DatabaseSchema,
    names: list[str],
    sample_limit: int,
    sample_seed: int,
) -> dict[str, str]:
    if not names:
        return {}
    try:
        from torch_frame.utils import infer_df_stype
    except ImportError as exc:
        raise RuntimeError(
            "Automatic feature inference requires torch-frame; "
            "configure every feature's semantic_type or install it"
        ) from exc
    identity = schema.tables[ref].identity_key
    if not identity:
        raise ValueError(f"Feature table {ref} needs an identity key")
    projection = tuple(dict.fromkeys((*identity, *names)))
    source = table(
        ref.table_name,
        *(column(name) for name in projection),
        schema=ref.schema_name,
    )
    query = select(*(source.c[name] for name in names)).order_by(
        *(source.c[name] for name in identity)
    )
    sampled = _priority_sample(reader, query, sample_limit, sample_seed)
    if sampled is None:
        raise ValueError(f"Cannot infer features from empty table {ref}")
    proposals = infer_df_stype(sampled)
    inferred = {
        name: "multicategorical" if kind.value == "embedding" else kind.value
        for name, kind in proposals.items()
    }
    # torch-frame omits columns with no observed values. Preserve the column
    # using its declared SQL type so sparse text/numeric fields remain usable.
    declarations = {
        column.name: column.raw_sql_type.upper()
        for column in schema.tables[ref].columns
    }
    for name in names:
        if name in inferred:
            continue
        declared = declarations[name]
        if any(token in declared for token in ("CHAR", "TEXT", "CLOB")):
            inferred[name] = "text"
        elif any(token in declared for token in (
            "INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"
        )):
            inferred[name] = "numerical"
    return inferred


def _priority_sample(
    reader: BaseDatabaseReader, query: Select, limit: int, seed: int
) -> pd.DataFrame | None:
    """Uniform sample over a stream with one batch plus `limit` rows in memory."""
    rng = np.random.default_rng(seed)
    sampled: pd.DataFrame | None = None
    priorities = np.empty(0, dtype=np.float64)
    for batch in reader.iter_arrow_query(query, batch_size=10_000):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        scores = rng.random(len(frame))
        if sampled is not None:
            frame = pd.concat((sampled, frame), ignore_index=True)
            scores = np.concatenate((priorities, scores))
        count = min(limit, len(frame))
        selected = np.argpartition(scores, count - 1)[:count]
        priorities = scores[selected]
        sampled = frame.iloc[selected].reset_index(drop=True)
    return sampled


def _semantic_type(raw: str | None, ref: TableRef, name: str) -> SemanticType:
    if raw in {"text_embedded", "text_tokenized"}:
        raw = "text"
    if raw not in _SEMANTIC_TYPES:
        raise ValueError(
            f"Cannot infer semantic type for {ref}.{name}; "
            "set features.columns[].semantic_type explicitly"
        )
    return cast(SemanticType, raw)
