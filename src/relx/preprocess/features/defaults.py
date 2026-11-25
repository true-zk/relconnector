"""Plan-time declarations for RelBench-style default feature paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import FeatureOutputSpec, SemanticType

DEFAULT_PROCESSORS: Mapping[SemanticType, str] = {
    "numerical": "relbench_numerical",
    "categorical": "relbench_categorical",
    "multicategorical": "relbench_multicategorical",
    "timestamp": "relbench_timestamp",
    "text": "relbench_text",
    "embedding": "relbench_embedding",
}


def builtin_output(
    semantic_type: SemanticType, processor: str, params: Mapping[str, Any]
) -> FeatureOutputSpec:
    """Describe default preprocessing; batch transforms belong to a later phase."""
    expected = DEFAULT_PROCESSORS[semantic_type]
    if processor != expected:
        raise ValueError(
            f"Processor {processor!r} is incompatible with {semantic_type!r}; "
            f"expected {expected!r}"
        )
    if semantic_type == "embedding":
        width = params.get("width")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("Embedding feature requires a positive params.width")
        return FeatureOutputSpec("float32", width=width)
    if semantic_type == "numerical":
        return FeatureOutputSpec("float32", requires_fit=True)
    if semantic_type == "categorical":
        return FeatureOutputSpec("int64", requires_fit=True)
    if semantic_type == "multicategorical":
        return FeatureOutputSpec("int64", width=None, ragged=True, requires_fit=True)
    if semantic_type == "timestamp":
        return FeatureOutputSpec("int64", requires_fit=True)
    return FeatureOutputSpec("utf8", width=None, requires_adapter=True)
