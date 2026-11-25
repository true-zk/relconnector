"""Read sampled raw feature rows from the database for every batch."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
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
)
from relx.preprocess.features.glove_export import ensure_glove_export
from relx.sampling import EntitySamplePlan, SamplePlan

from .contracts import PreparedBatch
from .online_fit import DefaultFeatureFit, fit_feature
from .transforms import empty_block

if TYPE_CHECKING:
    from relx.api import PreparedDataset


class OnlineFeatureStore:
    """Keep only fit state and identity keys; fetch feature values per batch."""

    def __init__(
        self,
        prepared: PreparedDataset,
        reader: BaseDatabaseReader,
        *,
        train_cutoff_us: int | None,
        scan_batch_size: int = 10_000,
        native_workers: int = 1,
    ) -> None:
        if min(scan_batch_size, native_workers) <= 0:
            raise ValueError("Online feature batch limits must be positive")
        self.prepared = prepared
        self.native_workers = native_workers
        self._custom_lock = Lock()
        self.states: dict[tuple[TableRef, str], FeatureState] = {}
        self.keys: dict[TableRef, tuple[tuple[object, ...], ...]] = {}
        from relx import _relx_native

        backend = reader.database_url.get_backend_name()
        if backend == "sqlite" and reader.path is not None:
            self.native = _relx_native.SqliteFeatureReader(str(reader.path))
        elif backend in {"mysql", "mariadb"}:
            if not hasattr(_relx_native, "MySqlFeatureReader"):
                raise RuntimeError(
                    "Native MySQL reads require rebuilding with mysql_config"
                )
            url = reader.database_url
            options = dict(url.query)
            unknown = set(options) - {"unix_socket", "charset"}
            if unknown:
                raise ValueError(
                    f"Native MySQL reader does not support URL options {sorted(unknown)}"
                )
            if options.get("charset", "utf8mb4") != "utf8mb4":
                raise ValueError("Native MySQL reader requires charset=utf8mb4")
            socket = options.get("unix_socket", "")
            if not isinstance(socket, str):
                raise TypeError("MySQL unix_socket URL option must be a string")
            self.native = _relx_native.MySqlFeatureReader(
                url.host or "", url.port or 3306, url.username or "",
                url.password or "", url.database or "",
                socket,
            )
        else:
            raise NotImplementedError(
                f"Native online feature reads do not support {backend!r}"
            )
        self.native.set_row_cache_bytes(
            prepared.config.training.feature_cache_bytes
        )
        self.text_mode = str(prepared.config.model.params.get("text_mode", "hash"))
        self.text_embedding_dim = 0
        if self.text_mode == "glove" and any(
            spec.semantic_type == "text" for spec in prepared.features.features
        ):
            params = prepared.config.model.params
            export_dir = params.get("glove_export_dir")
            path = ensure_glove_export(
                params.get("glove_model_path"),
                directory=Path(export_dir) if export_dir else None,
            )
            self.text_embedding_dim = int(json.loads(
                (path / "manifest.json").read_text(encoding="utf-8")
            )["width"])
            cache_bytes = params.get("glove_cache_bytes", 64 * 1024 * 1024)
            if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or (
                cache_bytes < 0
            ):
                raise ValueError("model.params.glove_cache_bytes must be nonnegative")
            disk_cache = params.get("glove_disk_cache", False)
            if not isinstance(disk_cache, bool):
                raise TypeError("model.params.glove_disk_cache must be boolean")
            self.native.set_glove_model(
                str(path), cache_bytes,
                str(path / "text-cache.sqlite") if disk_cache else "",
            )
        for name, count in prepared.graph.node_counts.items():
            ref = next(ref for ref in prepared.schema.tables if str(ref) == name)
            specs = prepared.features.for_table(ref)
            if not specs:
                continue
            self.keys[ref] = self._read_keys(reader, ref, count, scan_batch_size)
            self.states.update(self._fit_table(
                reader, ref, specs, train_cutoff_us, scan_batch_size
            ))
            self.native.add_table(
                ref.table_name, ref.schema_name or "", list(self._identity(ref)),
                self.keys[ref],
            )
            for spec in specs:
                state = self.states[(ref, spec.column)].values
                custom = spec.processor in prepared.feature_processors
                if spec.semantic_type == "timestamp" and not custom and (
                    prepared.config.graph.timezone.upper() != "UTC"
                ):
                    raise NotImplementedError(
                        "Native SQL timestamp features currently need UTC"
                    )
                self.native.add_feature(
                    str(ref), spec.column,
                    "raw" if custom else (
                        "text_glove"
                        if spec.semantic_type == "text" and self.text_mode == "glove"
                        else spec.semantic_type
                    ),
                    float(state.get("mean", 0.0)),
                    float(state.get("std", 1.0)), spec.output.width or 1,
                    dict(state.get("vocab", {})),
                )

    def _identity(self, ref: TableRef) -> tuple[str, ...]:
        identity = self.prepared.schema.tables[ref].identity_key
        if not identity:
            raise ValueError(f"Feature table {ref} needs an identity key")
        return identity

    def _source(self, ref: TableRef, names: tuple[str, ...]):
        return table(
            ref.table_name, *(column(name) for name in names),
            schema=ref.schema_name,
        )

    def _node_ids(
        self, ref: TableRef, identity: tuple[str, ...], frame: pd.DataFrame
    ) -> np.ndarray:
        keys = pa.RecordBatch.from_pandas(frame[list(identity)], preserve_index=False)
        ids = self.prepared.graph.lookup_nodes(str(ref), list(identity), keys)
        if np.any(ids < 0):
            raise ValueError(f"Feature row on {ref} is absent from graph")
        return ids

    def _read_keys(
        self, reader: BaseDatabaseReader, ref: TableRef,
        count: int, batch_size: int,
    ) -> tuple[tuple[object, ...], ...]:
        identity = self._identity(ref)
        source = self._source(ref, identity)
        query = select(*(source.c[name] for name in identity)).order_by(
            *(source.c[name] for name in identity)
        )
        keys: list[tuple[object, ...] | None] = [None] * count
        for frame in reader.iter_query(query, batch_size=batch_size):
            ids = self._node_ids(ref, identity, frame)
            for node_id, values in zip(
                ids, frame.itertuples(index=False, name=None), strict=True
            ):
                if keys[node_id] is not None:
                    raise ValueError(f"Duplicate feature identity for {ref}")
                keys[node_id] = tuple(
                    value.item() if isinstance(value, np.generic) else value
                    for value in values
                )
        if any(key is None for key in keys):
            raise ValueError(f"Feature identity map for {ref} is incomplete")
        return tuple(cast(tuple[object, ...], key) for key in keys)

    def _training_values(
        self, reader: BaseDatabaseReader, spec: FeatureSpec,
        cutoff_us: int | None, batch_size: int,
    ) -> Iterator[pd.Series]:
        identity = self._identity(spec.table)
        names = tuple(dict.fromkeys((*identity, spec.column)))
        source = self._source(spec.table, names)
        query = select(*(source.c[name] for name in names)).order_by(
            *(source.c[name] for name in identity)
        )
        node_times = self.prepared.graph.node_times(str(spec.table))
        strict = self.prepared.config.graph.time_boundary == "strict_before"
        for frame in reader.iter_query(query, batch_size=batch_size):
            ids = self._node_ids(spec.table, identity, frame)
            visible = (
                np.ones(len(ids), dtype=bool)
                if cutoff_us is None
                else (node_times[ids] < cutoff_us if strict
                      else node_times[ids] <= cutoff_us)
            )
            yield cast(pd.Series, frame.loc[visible, spec.column])

    def _fit_table(
        self, reader: BaseDatabaseReader, ref: TableRef,
        specs: tuple[FeatureSpec, ...], cutoff_us: int | None,
        batch_size: int,
    ) -> dict[tuple[TableRef, str], FeatureState]:
        timezone = self.prepared.config.graph.timezone
        defaults = {
            spec.column: DefaultFeatureFit(spec, timezone)
            for spec in specs
            if spec.processor not in self.prepared.feature_processors
        }
        required = [name for name, fit in defaults.items() if fit.needs_values]
        if required:
            identity = self._identity(ref)
            names = tuple(dict.fromkeys((*identity, *required)))
            source = self._source(ref, names)
            query = select(*(source.c[name] for name in names)).order_by(
                *(source.c[name] for name in identity)
            )
            node_times = self.prepared.graph.node_times(str(ref))
            strict = self.prepared.config.graph.time_boundary == "strict_before"
            for frame in reader.iter_query(query, batch_size=batch_size):
                ids = self._node_ids(ref, identity, frame)
                visible = (
                    np.ones(len(ids), dtype=bool)
                    if cutoff_us is None
                    else (node_times[ids] < cutoff_us if strict
                          else node_times[ids] <= cutoff_us)
                )
                for name in required:
                    defaults[name].add(cast(pd.Series, frame.loc[visible, name]))
        states = {(ref, name): fit.finish() for name, fit in defaults.items()}
        for spec in specs:
            if spec.processor in self.prepared.feature_processors:
                states[(ref, spec.column)] = fit_feature(
                    spec,
                    self._training_values(reader, spec, cutoff_us, batch_size),
                    self.prepared.feature_processors, timezone,
                )
        return states

    def fetch(self, plan: SamplePlan) -> PreparedBatch:
        """Send sampled IDs to native SQL readers and feature processors."""
        if not isinstance(plan, EntitySamplePlan):
            raise TypeError("Online features currently require entity samples")
        gathered = self.native.fetch({
            str(ref): ids for ref, ids in plan.subgraph.node_ids.items()
        })
        return self.assemble(plan, gathered)

    def assemble(
        self, plan: EntitySamplePlan, gathered: dict[str, dict[str, dict[str, Any]]]
    ) -> PreparedBatch:
        """Wrap completed native features and invoke custom Python processors."""
        blocks: dict[tuple[TableRef, str], FeatureBlock] = {}
        refs = {str(ref): ref for ref in plan.subgraph.node_ids}
        for table_name, columns in gathered.items():
            ref = refs[table_name]
            for name, result in columns.items():
                values = result["values"]
                if values is None:
                    raise RuntimeError(f"Native feature {table_name}.{name} has no values")
                spec = self.prepared.features.get(ref, name)
                processor = self.prepared.feature_processors.get(spec.processor)
                if processor is None:
                    blocks[(ref, name)] = FeatureBlock(
                        values, result["missing"], result.get("offsets")
                    )
                else:
                    with self._custom_lock:
                        blocks[(ref, name)] = processor.transform(
                            pd.Series(values), self.states[(ref, name)]
                        )
        for ref, ids in plan.subgraph.node_ids.items():
            for spec in self.prepared.features.for_table(ref):
                if (ref, spec.column) in blocks:
                    continue
                if not len(ids):
                    if spec.semantic_type == "text" and self.text_mode == "glove":
                        blocks[(ref, spec.column)] = FeatureBlock(
                            np.empty((0, self.text_embedding_dim), dtype=np.float32)
                        )
                    else:
                        blocks[(ref, spec.column)] = empty_block(spec)
                    continue
                raise RuntimeError(f"Native feature {ref}.{spec.column} is missing")
        return PreparedBatch(plan.key, plan, blocks)
