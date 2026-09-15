"""Shared GloVe embedder used by the eager and online implementations."""

from relconnector.features.text import GloveTextEmbedder
from relconnector.features.text import normalize_text as _normalize_text

__all__ = ["GloveTextEmbedder", "_normalize_text"]
