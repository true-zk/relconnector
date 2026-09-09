"""Sentence-transformer text embedders used for text-heavy tables."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from sentence_transformers import SentenceTransformer


class GloveTextEmbedder:
    """Average-pooled Glove.6B.300d, matching the redelex baseline."""

    embedding_dim = 300
    _model_name = "sentence-transformers/average_word_embeddings_glove.6B.300d"

    def __init__(self, device: torch.device | str | None = None) -> None:
        self._device = torch.device(device) if device is not None else None
        self._model = SentenceTransformer(
            self._model_name,
            device=None if self._device is None else str(self._device),
        )

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor:
        normalized = [_normalize_text(value) for value in sentences]
        return self._model.encode(normalized, convert_to_tensor=True)


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)
