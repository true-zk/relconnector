"""Model-independent feature planning and processor interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from relx.core import TableRef

SemanticType = Literal[
    "numerical", "categorical", "multicategorical", "timestamp", "text", "embedding"
]


@dataclass(frozen=True)
class FeatureOutputSpec:
    dtype: str
    width: int | None = 1
    ragged: bool = False
    requires_fit: bool = False
    requires_adapter: bool = False

    def __post_init__(self) -> None:
        if self.width is not None and self.width <= 0:
            raise ValueError("Feature output width must be positive")
        if self.ragged and self.width is not None:
            raise ValueError("Ragged feature output cannot have a fixed width")


@dataclass(frozen=True)
class FeatureSpec:
    table: TableRef
    column: str
    semantic_type: SemanticType
    processor: str
    params: Mapping[str, Any]
    output: FeatureOutputSpec


@dataclass(frozen=True)
class FeaturePlan:
    features: tuple[FeatureSpec, ...]
    sample_limit: int
    sample_seed: int
    sample_strategy: str = "uniform_priority_stream"
    inference_backend: str = "torch_frame.infer_df_stype"

    def for_table(self, table: TableRef) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.features if feature.table == table)

    def get(self, table: TableRef, column: str) -> FeatureSpec:
        for feature in self.features:
            if feature.table == table and feature.column == column:
                return feature
        raise KeyError((table, column))


@dataclass(frozen=True)
class FeatureState:
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureBlock:
    """One column's transformed values in sampled node order."""

    values: np.ndarray
    missing: np.ndarray | None = None
    offsets: np.ndarray | None = None


@dataclass(frozen=True)
class FitContext:
    table: TableRef
    column: str
    timezone: str


class FeatureProcessor(ABC):
    """A user processor sees whole column batches, never individual graph edges."""

    semantic_type: SemanticType

    def fit(self, values: Iterable[pd.Series], context: FitContext) -> FeatureState:
        """Fit on training-visible data supplied by the caller."""
        return FeatureState()

    @abstractmethod
    def transform(self, values: pd.Series, state: FeatureState) -> FeatureBlock:
        """Transform one batch using only the previously fitted state."""

    @abstractmethod
    def output_spec(self, params: Mapping[str, Any]) -> FeatureOutputSpec:
        """Describe output without reading data or running the model."""
