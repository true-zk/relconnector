"""Online feature fetching, caching, and batch contracts."""

from .assembler import TensorFrameBatchAssembler
from .cache import EncodedFeatureCache, FeatureBlockCache, FeatureBlockKey
from .contracts import (
    BatchAssembler,
    EntityFeatureBatch,
    FeatureBatch,
    FeatureFetcher,
    FeatureFrame,
    FetchedNodeFeatures,
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
from .text import TextEmbeddingStats

__all__ = [
    "BatchAssembler",
    "EncodedFeatureCache",
    "EntityFeatureBatch",
    "FeatureBatch",
    "FeatureBlockCache",
    "FeatureBlockKey",
    "FeatureFetchStats",
    "FeatureFetcher",
    "FeatureFrame",
    "FetchedNodeFeatures",
    "FetchedSubgraph",
    "PreparedBatch",
    "RecommendationFeatureBatch",
    "SqlFeatureFetcher",
    "SqlTensorFrameSchemaBuilder",
    "TableFeatureSchema",
    "TensorFrameBatchAssembler",
    "TensorFrameEncoder",
    "TensorFrameFeatureSchema",
    "TextEmbeddingStats",
]
