"""Experimental full-table feature snapshot; not the online training path."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from sqlalchemy import column, select, table

from relx.conn import BaseDatabaseReader
from relx.core import TableRef
from relx.preprocess.features import (
    FeatureBlock,
    FeatureSpec,
    FeatureState,
    FitContext,
)
from relx.sampling import EntitySamplePlan, SamplePlan

from .contracts import PreparedBatch
from .native_columns import register_native_columns
from .transforms import (
    empty_block,
    items,
    numeric_values,
    timestamp_values,
    token,
    transform_builtin,
)

if TYPE_CHECKING:
    from relx.api import PreparedDataset


class FeatureStore:
    """Benchmark prototype that snapshots feature columns before training.

    This violates RelX's online database feature-read contract. Do not use it
    as the default for production training.
    """

    def __init__(
        self,
        prepared: PreparedDataset,
        reader: BaseDatabaseReader,
        *,
        train_cutoff_us: int | None,
        batch_size: int = 10_000,
        native_workers: int = 1,
    ) -> None:
        if batch_size <= 0 or native_workers <= 0:
            raise ValueError("batch_size and native_workers must be positive")
        from relx._relx_native import NativeFeatureStore

        self.prepared = prepared
        self.native = NativeFeatureStore()
        self.native_workers = native_workers
        self._custom_lock = Lock()
        self.refs = {str(ref): ref for ref in prepared.schema.tables}
        self.frames: dict[TableRef, pd.DataFrame] = {}
        self.states: dict[tuple[TableRef, str], FeatureState] = {}
        for ref, count in prepared.graph.node_counts.items():
            table_ref = next(item for item in prepared.schema.tables if str(item) == ref)
            specs = prepared.features.for_table(table_ref)
            if not specs:
                continue
            frame = self._read_table(reader, table_ref, specs, count, batch_size)
            self.frames[table_ref] = frame
            node_times = prepared.graph.node_times(ref)
            if train_cutoff_us is None:
                visible = np.ones(count, dtype=bool)
            elif prepared.config.graph.time_boundary == "strict_before":
                visible = node_times < train_cutoff_us
            else:
                visible = node_times <= train_cutoff_us
            for spec in specs:
                self.states[(table_ref, spec.column)] = self._fit(
                    spec, frame[spec.column].iloc[np.flatnonzero(visible)],
                )
            fallback = register_native_columns(
                self.native, table_ref, frame, specs, self.states,
                prepared.feature_processors, prepared.config.graph.timezone,
            )
            if fallback is None:
                self.frames.pop(table_ref, None)
            else:
                self.frames[table_ref] = fallback

    def _read_table(
        self,
        reader: BaseDatabaseReader,
        ref: TableRef,
        specs: tuple[FeatureSpec, ...],
        count: int,
        batch_size: int,
    ) -> pd.DataFrame:
        identity = self.prepared.schema.tables[ref].identity_key
        if not identity:
            raise ValueError(f"Feature table {ref} needs an identity key")
        names = tuple(dict.fromkeys((*identity, *(spec.column for spec in specs))))
        source = table(
            ref.table_name, *(column(name) for name in names), schema=ref.schema_name
        )
        query = select(*(source.c[name] for name in names)).order_by(
            *(source.c[name] for name in identity)
        )
        chunks: list[pd.DataFrame] = []
        for chunk in reader.iter_query(query, batch_size=batch_size):
            lookup = pa.RecordBatch.from_pandas(chunk[list(identity)], preserve_index=False)
            ids = self.prepared.graph.lookup_nodes(str(ref), list(identity), lookup)
            if np.any(ids < 0):
                raise ValueError(f"Feature row on {ref} is absent from the graph")
            values = cast(
                pd.DataFrame, chunk.loc[:, [spec.column for spec in specs]].copy()
            )
            values.index = ids
            chunks.append(values)
        if not chunks:
            if count:
                raise ValueError(f"Feature table {ref} has no rows")
            return pd.DataFrame(columns=[spec.column for spec in specs])
        frame = pd.concat(chunks).sort_index()
        if len(frame) != count or not frame.index.is_unique or not np.array_equal(
            frame.index.to_numpy(), np.arange(count)
        ):
            raise ValueError(f"Feature snapshot for {ref} does not match graph nodes")
        return frame

    def _fit(self, spec: FeatureSpec, values: pd.Series) -> FeatureState:
        custom = self.prepared.feature_processors.get(spec.processor)
        if custom is not None:
            return custom.fit(
                (values.iloc[start:start+10_000] for start in range(0, len(values), 10_000)),
                FitContext(spec.table, spec.column, self.prepared.config.graph.timezone),
            )
        if spec.semantic_type == "numerical":
            numbers = numeric_values(values)
            valid = numbers[np.isfinite(numbers)]
            mean = float(valid.mean()) if len(valid) else 0.0
            std = float(valid.std()) if len(valid) and valid.std() > 0 else 1.0
            return FeatureState({"mean": mean, "std": std})
        if spec.semantic_type in {"categorical", "multicategorical"}:
            if spec.semantic_type == "categorical":
                tokens = {token(value) for value in values if not pd.isna(value)}
            else:
                tokens = {
                    token(item)
                    for value in values
                    for item in items(value)
                }
            vocab = {token: index + 1 for index, token in enumerate(sorted(tokens))}
            return FeatureState({"vocab": vocab})
        if spec.semantic_type == "timestamp":
            times = timestamp_values(
                values, self.prepared.config.graph.timezone
            )
            times = times[times != np.iinfo(np.int64).min]
            if len(times):
                mean = float(times.mean())
                std = float(times.astype(np.float64).std())
                return FeatureState({"mean": mean, "std": std or 1.0})
        return FeatureState()

    def fetch(self, plan: SamplePlan) -> PreparedBatch:
        if not isinstance(plan, EntitySamplePlan):
            raise TypeError("FeatureStore currently accepts entity sample plans")
        native = self.native.gather(
            {str(ref): ids for ref, ids in plan.subgraph.node_ids.items()},
            self.native_workers,
        )
        return self.assemble(plan, native)

    def assemble(
        self, plan: EntitySamplePlan, native: dict[str, Any]
    ) -> PreparedBatch:
        """Convert a completed native batch and run custom processor fallbacks."""
        blocks: dict[tuple[TableRef, str], FeatureBlock] = {}
        for table_name, columns in native.items():
            ref = self.refs[table_name]
            for name, result in columns.items():
                values = result["values"]
                if values is None:
                    raise RuntimeError(
                        f"Native feature {table_name}.{name} has no values"
                    )
                blocks[(ref, name)] = FeatureBlock(
                    values, result["missing"], result.get("offsets")
                )
        for ref, ids in plan.subgraph.node_ids.items():
            frame = self.frames.get(ref)
            if frame is None:
                continue
            for spec in self.prepared.features.for_table(ref):
                if (ref, spec.column) in blocks:
                    continue
                if len(ids) == 0:
                    blocks[(ref, spec.column)] = empty_block(spec)
                    continue
                values = frame[spec.column].iloc[ids]
                state = self.states[(ref, spec.column)]
                custom = self.prepared.feature_processors.get(spec.processor)
                if custom is not None:
                    with self._custom_lock:
                        blocks[(ref, spec.column)] = custom.transform(values, state)
                else:
                    blocks[(ref, spec.column)] = transform_builtin(
                        spec, values, state, self.prepared.config.graph.timezone
                    )
        return PreparedBatch(plan.key, plan, blocks)
