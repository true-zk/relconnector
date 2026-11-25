"""Feature planning contracts and bounded semantic inference."""

from .contracts import (
    FeatureBlock,
    FeatureOutputSpec,
    FeaturePlan,
    FeatureProcessor,
    FeatureSpec,
    FeatureState,
    FitContext,
    SemanticType,
)
from .planner import build_feature_plan

__all__ = [
    "FeatureBlock",
    "FeatureOutputSpec",
    "FeaturePlan",
    "FeatureProcessor",
    "FeatureSpec",
    "FeatureState",
    "FitContext",
    "SemanticType",
    "build_feature_plan",
]
