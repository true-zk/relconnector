"""Fit feature state from bounded database result chunks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from relx.preprocess.features import (
    FeatureProcessor,
    FeatureSpec,
    FeatureState,
    FitContext,
)

from .transforms import items, numeric_values, timestamp_values, token


class DefaultFeatureFit:
    """Streaming state for one built-in feature; no raw values are retained."""

    def __init__(self, spec: FeatureSpec, timezone: str) -> None:
        self.spec = spec
        self.timezone = timezone
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.tokens: set[str] = set()

    @property
    def needs_values(self) -> bool:
        return self.spec.semantic_type in {
            "numerical", "timestamp", "categorical", "multicategorical"
        }

    def add(self, chunk: pd.Series) -> None:
        kind = self.spec.semantic_type
        if kind in {"numerical", "timestamp"}:
            raw = (
                numeric_values(chunk)
                if kind == "numerical"
                else timestamp_values(chunk, self.timezone).astype(np.float64)
            )
            valid = (
                raw[np.isfinite(raw)]
                if kind == "numerical"
                else raw[raw != float(np.iinfo(np.int64).min)]
            )
            if not len(valid):
                return
            batch_mean = float(valid.mean())
            batch_m2 = float(np.square(valid - batch_mean).sum())
            total = self.count + len(valid)
            delta = batch_mean - self.mean
            self.m2 += batch_m2 + delta * delta * self.count * len(valid) / total
            self.mean += delta * len(valid) / total
            self.count = total
        elif kind == "categorical":
            self.tokens.update(token(value) for value in chunk if not pd.isna(value))
        elif kind == "multicategorical":
            self.tokens.update(
                token(value) for values in chunk for value in items(values)
            )

    def finish(self) -> FeatureState:
        kind = self.spec.semantic_type
        if kind in {"numerical", "timestamp"}:
            std = float(np.sqrt(self.m2 / self.count)) if self.count else 1.0
            return FeatureState({"mean": self.mean, "std": std or 1.0})
        if kind in {"categorical", "multicategorical"}:
            return FeatureState({
                "vocab": {
                    value: index + 1
                    for index, value in enumerate(sorted(self.tokens))
                }
            })
        return FeatureState()


def fit_feature(
    spec: FeatureSpec,
    chunks: Iterable[pd.Series],
    processors: Mapping[str, FeatureProcessor],
    timezone: str,
) -> FeatureState:
    """Consume only training-visible values, retaining statistics or vocabulary."""
    custom = processors.get(spec.processor)
    if custom is not None:
        return custom.fit(chunks, FitContext(spec.table, spec.column, timezone))

    fit = DefaultFeatureFit(spec, timezone)
    for chunk in chunks:
        fit.add(chunk)
    return fit.finish()
