"""Encode pandas columns for lossless SQLite storage."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import date, datetime
from decimal import Decimal
from typing import cast

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .models import ColumnSchema


def infer_column_schema(name: str, ordinal: int, series: pd.Series) -> ColumnSchema:
    """Infer a reversible SQLite encoding for a pandas series."""
    dtype = str(series.dtype)
    metadata: dict[str, object] = {}

    if isinstance(series.dtype, pd.CategoricalDtype):
        metadata = {
            "categories": [
                _json_value(value) for value in series.cat.categories.tolist()
            ],
            "ordered": series.cat.ordered,
        }
        return ColumnSchema(name, ordinal, dtype, "category", metadata)
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return ColumnSchema(name, ordinal, dtype, "datetime")
    if ptypes.is_timedelta64_dtype(series.dtype):
        return ColumnSchema(name, ordinal, dtype, "timedelta_ns")

    values = [value for value in series.array if not _is_null(value)]
    if values and all(
        isinstance(value, (list, tuple, dict, set, np.ndarray)) for value in values
    ):
        return ColumnSchema(name, ordinal, dtype, "json")
    if values and all(
        isinstance(value, (pd.Timestamp, datetime, date)) for value in values
    ):
        return ColumnSchema(name, ordinal, dtype, "datetime")
    if values and all(isinstance(value, Decimal) for value in values):
        return ColumnSchema(name, ordinal, dtype, "decimal")
    if values and all(
        isinstance(value, (bytes, bytearray, memoryview)) for value in values
    ):
        return ColumnSchema(name, ordinal, dtype, "bytes")
    return ColumnSchema(name, ordinal, dtype)


def infer_columns(frame: pd.DataFrame) -> tuple[ColumnSchema, ...]:
    return tuple(
        infer_column_schema(name, ordinal, cast(pd.Series, frame[name]))
        for ordinal, name in enumerate(frame.columns)
    )


def sqlite_type(column: ColumnSchema) -> str:
    if column.encoding in {"json", "datetime", "decimal", "category"}:
        return "TEXT"
    if column.encoding == "bytes":
        return "BLOB"
    if column.encoding == "timedelta_ns":
        return "INTEGER"

    dtype = pd.api.types.pandas_dtype(column.pandas_dtype)
    if ptypes.is_bool_dtype(dtype) or ptypes.is_integer_dtype(dtype):
        return "INTEGER"
    if ptypes.is_float_dtype(dtype):
        return "REAL"
    return "TEXT"


def encode_frame(frame: pd.DataFrame, columns: Iterable[ColumnSchema]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        series = cast(pd.Series, result[column.name])
        if column.encoding == "json":
            result[column.name] = _map_as_objects(
                series,
                lambda value: (
                    None
                    if _is_null(value)
                    else json.dumps(
                        _json_value(value),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            )
        elif column.encoding == "datetime":
            result[column.name] = _map_as_objects(
                series,
                lambda value: (
                    None
                    if _is_null(value)
                    else pd.Timestamp(
                        cast(str | date | datetime | np.datetime64, value)
                    ).isoformat()
                ),
            )
        elif column.encoding == "timedelta_ns":
            result[column.name] = _map_as_objects(
                series,
                lambda value: (
                    None if _is_null(value) else int(pd.Timedelta(value).value)
                ),
            )
        elif column.encoding == "decimal":
            result[column.name] = _map_as_objects(
                series,
                lambda value: None if _is_null(value) else str(value),
            )
        elif column.encoding == "bytes":
            result[column.name] = _map_as_objects(
                series,
                lambda value: (
                    None
                    if _is_null(value)
                    else bytes(cast(bytes | bytearray | memoryview, value))
                ),
            )
        else:
            result[column.name] = _map_as_objects(series, _sqlite_scalar)
    return result


def _map_as_objects(
    series: pd.Series, function: Callable[[object], object]
) -> pd.Series:
    """Map without coercing nullable 64-bit integers through float64."""
    return pd.Series(
        [function(value) for value in series.array],
        index=series.index,
        dtype=object,
    )


def _is_null(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _sqlite_scalar(value: object) -> object:
    if _is_null(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value
