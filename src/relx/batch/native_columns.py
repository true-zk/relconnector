"""Materialize default feature columns once for shared native batch workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from relx.core import TableRef
from relx.preprocess.features import FeatureProcessor, FeatureSpec, FeatureState

from .transforms import embedding, items, numeric_values, timestamp_values, token

if TYPE_CHECKING:
    from relx._relx_native import NativeFeatureStore


def register_native_columns(
    native: NativeFeatureStore,
    ref: TableRef,
    frame: pd.DataFrame,
    specs: tuple[FeatureSpec, ...],
    states: Mapping[tuple[TableRef, str], FeatureState],
    processors: Mapping[str, FeatureProcessor],
    timezone: str,
) -> pd.DataFrame | None:
    """Register builtins and retain only custom processor columns in Pandas."""
    fallback: list[str] = []
    for spec in specs:
        if spec.processor in processors:
            fallback.append(spec.column)
            continue
        state = states[(ref, spec.column)].values
        values = cast(pd.Series, frame[spec.column])
        if spec.semantic_type == "numerical":
            native.add_numeric(
                str(ref), spec.column, numeric_values(values),
                float(state["mean"]), float(state["std"]),
            )
        elif spec.semantic_type == "categorical":
            vocab = state["vocab"]
            encoded = np.fromiter(
                (
                    0 if pd.isna(value) else vocab.get(token(value), 0)
                    for value in values
                ),
                dtype=np.int64,
                count=len(values),
            )
            native.add_categorical(str(ref), spec.column, encoded)
        elif spec.semantic_type == "timestamp":
            native.add_timestamp(
                str(ref), spec.column, timestamp_values(values, timezone),
                float(state.get("mean", 0.0)), float(state.get("std", 1.0)),
            )
        elif spec.semantic_type == "multicategorical":
            vocab = state["vocab"]
            flattened: list[int] = []
            offsets = [0]
            for value in values:
                flattened.extend(vocab.get(token(item), 0) for item in items(value))
                offsets.append(len(flattened))
            native.add_multicategorical(
                str(ref), spec.column,
                np.asarray(flattened, dtype=np.int64),
                np.asarray(offsets, dtype=np.int64),
            )
        elif spec.semantic_type == "embedding":
            width = spec.output.width
            if width is None:
                raise ValueError(f"Embedding {ref}.{spec.column} needs width")
            vectors = [embedding(value, width) for value in values]
            packed = (
                np.stack(vectors).astype(np.float32)
                if vectors else np.empty((0, width), dtype=np.float32)
            )
            native.add_embedding(str(ref), spec.column, packed)
        elif spec.semantic_type == "text":
            strings = ["" if pd.isna(value) else str(value) for value in values]
            native.add_text(str(ref), spec.column, strings)
        else:
            fallback.append(spec.column)
    return frame.loc[:, fallback].copy() if fallback else None
