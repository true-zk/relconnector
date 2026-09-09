"""Decode SQL values using the logical types stored in the catalog."""

from __future__ import annotations

import json
from collections.abc import Iterable
from decimal import Decimal
from typing import cast

import numpy as np
import pandas as pd

from .catalog import ColumnSchema


def decode_frame(frame: pd.DataFrame, columns: Iterable[ColumnSchema]) -> pd.DataFrame:
    ordered_columns = sorted(columns, key=lambda item: item.ordinal)
    result = frame.copy()
    for column in ordered_columns:
        if column.name not in result:
            raise ValueError(f"Query result is missing column {column.name!r}")
        series = cast(pd.Series, result[column.name])
        if column.encoding == "json":
            result[column.name] = series.map(
                lambda value: None if _is_null(value) else json.loads(str(value))
            )
        elif column.encoding == "datetime":
            result[column.name] = pd.to_datetime(series, format="mixed")
        elif column.encoding == "timedelta_ns":
            result[column.name] = pd.to_timedelta(
                cast(pd.Series, pd.to_numeric(series, errors="coerce")), unit="ns"
            )
        elif column.encoding == "decimal":
            result[column.name] = series.map(
                lambda value: None if _is_null(value) else Decimal(str(value))
            )
        elif column.encoding == "bytes":
            result[column.name] = series.map(
                lambda value: None if _is_null(value) else bytes(value)
            )
        elif column.encoding == "category":
            result[column.name] = pd.Categorical(
                series,
                categories=column.encoding_metadata.get("categories", []),
                ordered=bool(column.encoding_metadata.get("ordered", False)),
            )
        else:
            result[column.name] = _restore_dtype(series, column.pandas_dtype)
    return result.reindex(columns=[column.name for column in ordered_columns])


def _restore_dtype(series: pd.Series, dtype_name: str) -> pd.Series:
    if dtype_name == "object":
        return series
    try:
        return series.astype(dtype_name)
    except (TypeError, ValueError):
        return series


def _is_null(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False
