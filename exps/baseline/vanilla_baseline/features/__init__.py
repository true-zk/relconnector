"""Online feature fetching, caching, and batch contracts."""

from .assembler import TensorFrameBatchAssembler
from .cache import FeatureBlockCache, FeatureBlockKey
from .contracts import (
    BatchAssembler,
    EntityFeatureBatch,
    FeatureBatch,
    FeatureFetcher,
    FeatureFrame,
    FetchedSubgraph,
    PreparedBatch,
    RecommendationFeatureBatch,
)
from .schema import (
    SqlTensorFrameSchemaBuilder,
    TableFeatureSchema,
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
)
from .sql import FeatureFetchStats, SqlFeatureFetcher

__all__ = [
    "BatchAssembler",
    "EntityFeatureBatch",
    "FeatureBatch",
    "FeatureBlockCache",
    "FeatureBlockKey",
    "FeatureFetchStats",
    "FeatureFetcher",
    "FeatureFrame",
    "FetchedSubgraph",
    "PreparedBatch",
    "RecommendationFeatureBatch",
    "SqlFeatureFetcher",
    "SqlTensorFrameSchemaBuilder",
    "TableFeatureSchema",
    "TensorFrameBatchAssembler",
    "TensorFrameEncoder",
    "TensorFrameFeatureSchema",
]
