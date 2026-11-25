"""Default column transforms for the first model-neutral batch path."""

from __future__ import annotations

import json
from datetime import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from relx.preprocess.features import FeatureBlock, FeatureSpec, FeatureState
from relx.preprocess.graph_time import normalize_time_batch

_MISSING_TIME = np.iinfo(np.int64).min


def numeric_values(values: pd.Series) -> np.ndarray:
    converted = pd.Series(pd.to_numeric(values, errors="coerce"))
    return converted.to_numpy(dtype=np.float64, na_value=np.nan)


def timestamp_values(values: pd.Series, timezone: str) -> np.ndarray:
    result = np.full(len(values), _MISSING_TIME, dtype=np.int64)
    present = ~values.isna().to_numpy(dtype=bool)
    if np.any(present):
        present_values = values.iloc[np.flatnonzero(present)].to_list()
        batch = pa.RecordBatch.from_pandas(
            pd.DataFrame({"value": present_values}),
            preserve_index=False,
        )
        try:
            normalized = normalize_time_batch(batch, "value", timezone)
            parsed = normalized.column("__relx_time_us").to_pylist()
        except ValueError as date_error:
            try:
                clock_values = [time.fromisoformat(str(value)) for value in present_values]
            except ValueError:
                raise date_error from None
            parsed = [
                ((clock.hour * 60 + clock.minute) * 60 + clock.second) * 1_000_000
                + clock.microsecond
                for clock in clock_values
            ]
        result[present] = np.asarray(parsed, dtype=np.int64)
    return result


def transform_builtin(
    spec: FeatureSpec, values: pd.Series, state: FeatureState, timezone: str
) -> FeatureBlock:
    if spec.semantic_type == "numerical":
        numbers = numeric_values(values)
        valid = np.isfinite(numbers)
        mean = float(state.values["mean"])
        std = float(state.values["std"])
        normalized = (np.where(valid, numbers, mean) - mean) / std
        return FeatureBlock(normalized.astype(np.float32).reshape(-1, 1), ~valid)
    if spec.semantic_type == "categorical":
        vocab = state.values["vocab"]
        encoded = [
            vocab.get(token(value), 0) if not pd.isna(value) else 0
            for value in values
        ]
        return FeatureBlock(np.asarray(encoded, dtype=np.int64))
    if spec.semantic_type == "multicategorical":
        vocab = state.values["vocab"]
        flattened: list[int] = []
        offsets = [0]
        for value in values:
            flattened.extend(vocab.get(token(item), 0) for item in items(value))
            offsets.append(len(flattened))
        return FeatureBlock(
            np.asarray(flattened, dtype=np.int64),
            offsets=np.asarray(offsets, dtype=np.int64),
        )
    if spec.semantic_type == "timestamp":
        times = timestamp_values(values, timezone)
        valid = times != _MISSING_TIME
        mean = float(state.values.get("mean", 0.0))
        std = float(state.values.get("std", 1.0))
        normalized = (times.astype(np.float64) - mean) / std
        normalized[~valid] = 0.0
        return FeatureBlock(normalized.astype(np.float32).reshape(-1, 1), ~valid)
    if spec.semantic_type == "text":
        strings = ["" if pd.isna(value) else str(value) for value in values]
        return FeatureBlock(np.asarray(strings, dtype=object))
    width = spec.output.width
    assert width is not None
    vectors = [embedding(value, width) for value in values]
    return FeatureBlock(np.stack(vectors).astype(np.float32))


def empty_block(spec: FeatureSpec) -> FeatureBlock:
    """Return a typed empty output without invoking user processors."""
    if spec.output.ragged:
        return FeatureBlock(
            np.empty(0, dtype=np.int64), offsets=np.zeros(1, dtype=np.int64)
        )
    if spec.semantic_type == "text" or spec.output.dtype == "utf8":
        return FeatureBlock(np.empty(0, dtype=object))
    if spec.semantic_type == "categorical" or (
        spec.output.dtype == "int64" and spec.semantic_type != "timestamp"
    ):
        return FeatureBlock(np.empty(0, dtype=np.int64))
    width = spec.output.width or 1
    missing = (
        np.empty(0, dtype=bool)
        if spec.semantic_type in {"numerical", "timestamp"}
        else None
    )
    return FeatureBlock(np.empty((0, width), dtype=np.float32), missing)


def token(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return f"{type(value).__name__}:{value}"


def items(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _is_missing_scalar(value):
        return []
    parsed: Any = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise TypeError("Multicategorical values must be JSON arrays")
    return parsed


def embedding(value: object, width: int) -> np.ndarray:
    if _is_missing_scalar(value):
        return np.zeros(width, dtype=np.float32)
    parsed: Any = json.loads(value) if isinstance(value, str) else value
    vector = np.asarray(parsed, dtype=np.float32)
    if vector.shape != (width,):
        raise ValueError(f"Embedding value must have width {width}")
    return vector


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)
