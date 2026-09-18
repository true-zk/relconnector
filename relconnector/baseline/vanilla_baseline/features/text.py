"""Frozen GloVe sentence embeddings for sampled SQL rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd
import torch


class TextEmbedder(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor: ...


class GloveTextEmbedder:
    embedding_dim = 300
    model_name = "sentence-transformers/average_word_embeddings_glove.6B.300d"

    def __init__(self, *, model_path: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        # Feature workers stay on CPU; only prepared batches go to the trainer.
        self._model = SentenceTransformer(
            model_path or self.model_name,
            device="cpu",
            local_files_only=True,
        )
        self._model.eval()

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor:
        if not sentences:
            return torch.empty((0, self.embedding_dim))
        with torch.inference_mode():
            return self._model.encode(
                [normalize_text(value) for value in sentences],
                convert_to_tensor=True,
                show_progress_bar=False,
            ).cpu()


def normalize_text(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return value if isinstance(value, str) else str(value)
