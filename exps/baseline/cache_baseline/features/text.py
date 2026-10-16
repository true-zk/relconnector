"""Frozen GloVe sentence embeddings for sampled SQL rows."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import torch


class TextEmbedder(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor: ...


@dataclass(frozen=True)
class TextEmbeddingStats:
    occurrences: int
    unique_texts: int
    cache_hits: int
    cache_misses: int
    model_inputs: int
    evictions: int
    current_bytes: int
    peak_bytes: int


class GloveTextEmbedder:
    embedding_dim = 300
    model_name = "sentence-transformers/average_word_embeddings_glove.6B.300d"

    def __init__(self, *, model_path: str | None = None, cache_bytes: int = 0) -> None:
        if cache_bytes < 0:
            raise ValueError("cache_bytes cannot be negative")
        from sentence_transformers import SentenceTransformer

        # Feature workers stay on CPU; only prepared batches go to the trainer.
        self._model = SentenceTransformer(
            model_path or self.model_name,
            device="cpu",
            local_files_only=True,
        )
        self._model.eval()
        self._cache_bytes = cache_bytes
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._current_bytes = 0
        self._peak_bytes = 0
        self._occurrences = 0
        self._unique_texts = 0
        self._hits = 0
        self._misses = 0
        self._model_inputs = 0
        self._evictions = 0

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor:
        if not sentences:
            return torch.empty((0, self.embedding_dim))
        normalized = [normalize_text(value) for value in sentences]
        unique: list[str] = []
        positions: dict[str, int] = {}
        inverse: list[int] = []
        for value in normalized:
            position = positions.get(value)
            if position is None:
                position = len(unique)
                positions[value] = position
                unique.append(value)
            inverse.append(position)

        self._occurrences += len(normalized)
        self._unique_texts += len(unique)
        embeddings: list[torch.Tensor | None] = [None] * len(unique)
        missing: list[str] = []
        missing_positions: list[int] = []
        for position, value in enumerate(unique):
            cached = self._cache.get(value)
            if cached is None:
                self._misses += 1
                missing.append(value)
                missing_positions.append(position)
            else:
                self._hits += 1
                self._cache.move_to_end(value)
                embeddings[position] = cached

        if missing:
            self._model_inputs += len(missing)
            with torch.inference_mode():
                encoded = self._model.encode(
                    missing,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).cpu()
            for position, value, embedding in zip(
                missing_positions, missing, encoded, strict=True
            ):
                embeddings[position] = embedding
                self._put(value, embedding)

        complete = [embedding for embedding in embeddings if embedding is not None]
        if len(complete) != len(unique):
            raise RuntimeError("Text embedding result is incomplete")
        return torch.stack(complete)[torch.tensor(inverse, dtype=torch.long)]

    def _put(self, value: str, embedding: torch.Tensor) -> None:
        if self._cache_bytes == 0:
            return
        size = embedding.numel() * embedding.element_size()
        if size > self._cache_bytes:
            return
        while self._cache and self._current_bytes + size > self._cache_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= evicted.numel() * evicted.element_size()
            self._evictions += 1
        self._cache[value] = embedding.detach().clone()
        self._current_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)

    @property
    def stats(self) -> TextEmbeddingStats:
        return TextEmbeddingStats(
            occurrences=self._occurrences,
            unique_texts=self._unique_texts,
            cache_hits=self._hits,
            cache_misses=self._misses,
            model_inputs=self._model_inputs,
            evictions=self._evictions,
            current_bytes=self._current_bytes,
            peak_bytes=self._peak_bytes,
        )


def normalize_text(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return value if isinstance(value, str) else str(value)
