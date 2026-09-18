"""Shared GloVe embedder used by the eager and online implementations."""

from baseline.vanilla_baseline.features.text import GloveTextEmbedder
from baseline.vanilla_baseline.features.text import normalize_text as _normalize_text

__all__ = ["GloveTextEmbedder", "_normalize_text"]
