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
from .process_worker import SqlAssemblyWorkerFactory
from .schema import (
    SqlTensorFrameSchemaBuilder,
    TableFeatureSchema,
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
)
from .shared_encoded import SharedEncodedFeatureCache, create_shared_encoded_cache
from .shared_text import SharedTextEmbeddingCache, SharedTextLookup
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
    "SharedEncodedFeatureCache",
    "SharedTextEmbeddingCache",
    "SharedTextLookup",
    "SqlAssemblyWorkerFactory",
    "SqlFeatureFetcher",
    "SqlTensorFrameSchemaBuilder",
    "TableFeatureSchema",
    "TensorFrameBatchAssembler",
    "TensorFrameEncoder",
    "TensorFrameFeatureSchema",
    "TextEmbeddingStats",
    "create_shared_encoded_cache",
]
