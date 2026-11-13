"""Assign task rows to train, validation, and test sets."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pandas as pd

from .spec import ColumnSplit, DefaultSplit, SqlQuerySplit
from .validation import ValidatedTask

if TYPE_CHECKING:
    from relx.conn import BaseDatabaseReader


@dataclass(frozen=True)
class _Sample:
    label: object
    time: pd.Timestamp | None
    split_value: object


def _column_assignments(
    samples: dict[tuple[object, ...], _Sample], split: ColumnSplit
) -> dict[tuple[object, ...], str]:
    values = {split.train: "train", split.validation: "validation", split.test: "test"}
    result: dict[tuple[object, ...], str] = {}
    for key, sample in samples.items():
        value = sample.split_value
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError(f"Sample {key} has invalid split value {value!r}")
        try:
            result[key] = values[value]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Sample {key} has unknown split value {value!r}") from exc
    return result


def _query_assignments(
    task: ValidatedTask,
    reader: BaseDatabaseReader,
    samples: dict[tuple[object, ...], _Sample],
    batch_size: int,
) -> dict[tuple[object, ...], str]:
    split = task.spec.split
    assert isinstance(split, SqlQuerySplit)
    test = _query_keys(reader, split.test, task.sample_key, batch_size)
    if split.train is None:
        if any(key not in samples for key in test):
            raise ValueError("task.split.test returned unknown sample keys")
        remaining = {key: sample for key, sample in samples.items() if key not in test}
        if any(_is_null(sample.label) for sample in remaining.values()):
            raise ValueError("Unlabeled samples outside task.split.test")
        result = _default_assignments(
            remaining, DefaultSplit(split.validation_fraction, split.seed)
        )
        result.update(dict.fromkeys(test, "test"))
        return result

    assert split.validation is not None
    groups = {
        "train": _query_keys(reader, split.train, task.sample_key, batch_size),
        "validation": _query_keys(
            reader, split.validation, task.sample_key, batch_size
        ),
        "test": test,
    }
    result: dict[tuple[object, ...], str] = {}
    for name, keys in groups.items():
        for key in keys:
            if key not in samples:
                raise ValueError(f"task.split.{name} returned unknown sample key {key}")
            if key in result:
                raise ValueError(f"Sample {key} appears in multiple split queries")
            result[key] = name
    if len(result) != len(samples):
        raise ValueError("Split queries do not cover every task sample")
    return result


def _query_keys(
    reader: BaseDatabaseReader,
    query: str,
    key_names: tuple[str, ...],
    batch_size: int,
) -> set[tuple[object, ...]]:
    if re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE) is None or ";" in query:
        raise ValueError("Split SQL must be a single SELECT query")
    keys: set[tuple[object, ...]] = set()
    for frame in reader.iter_query(query, batch_size=batch_size):
        if tuple(frame.columns) != key_names:
            raise ValueError(
                f"Split SQL must return only sample key columns {key_names}"
            )
        for row in frame.itertuples(index=False, name=None):
            key = tuple(row)
            if any(_is_null(part) for part in key) or key in keys:
                raise ValueError(
                    f"Split SQL returned null or duplicate sample key {key}"
                )
            keys.add(key)
    return keys


def _default_assignments(
    samples: dict[tuple[object, ...], _Sample], split: DefaultSplit
) -> dict[tuple[object, ...], str]:
    test = {key for key, sample in samples.items() if _is_null(sample.label)}
    labeled = [key for key in samples if key not in test]
    if len(labeled) < 2:
        raise ValueError("Default split needs at least two labeled samples")
    count = max(
        1, min(len(labeled) - 1, math.ceil(len(labeled) * split.validation_fraction))
    )
    if all(samples[key].time is not None for key in labeled):
        ordered = sorted(
            labeled, key=lambda key: (cast(pd.Timestamp, samples[key].time), repr(key))
        )
        cutoff = cast(pd.Timestamp, samples[ordered[-count]].time)
        validation = {
            key for key in labeled if cast(pd.Timestamp, samples[key].time) >= cutoff
        }
        if len(validation) == len(labeled):
            raise ValueError("All labeled samples share one prediction time")
    else:
        ordered = sorted(labeled, key=lambda key: _stable_hash(split.seed, key))
        validation = set(ordered[:count])
    return {
        key: "test" if key in test else "validation" if key in validation else "train"
        for key in samples
    }


def _stable_hash(seed: int, key: tuple[object, ...]) -> bytes:
    return hashlib.blake2b(repr((seed, key)).encode(), digest_size=16).digest()


def _is_null(value: object) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))
