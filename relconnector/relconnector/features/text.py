"""Frozen GloVe sentence embeddings for sampled SQL rows."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import torch

from relconnector.observability import OperationMetrics


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
    admission_rejections: int


class GloveTextEmbedder:
    embedding_dim = 300
    model_name = "sentence-transformers/average_word_embeddings_glove.6B.300d"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        cache_bytes: int = 0,
        cache_admission: str = "always",
        admission_entries: int = 250_000,
        execution: Literal["official", "direct"] = "direct",
        metrics: OperationMetrics | None = None,
    ) -> None:
        if cache_bytes < 0:
            raise ValueError("cache_bytes cannot be negative")
        if (
            cache_admission not in {"always", "second", "adaptive"}
            or admission_entries <= 0
        ):
            raise ValueError("invalid text cache admission configuration")
        if execution not in {"official", "direct"}:
            raise ValueError("text execution must be official or direct")
        from sentence_transformers import SentenceTransformer

        # Feature workers stay on CPU; only prepared batches go to the trainer.
        self._model = SentenceTransformer(
            model_path or self.model_name,
            device="cpu",
            local_files_only=True,
        )
        self._model.eval()
        self._direct_modules = (
            self._find_direct_modules() if execution == "direct" else None
        )
        self._cache_bytes = cache_bytes
        self._cache_admission = cache_admission
        self._admission_entries = admission_entries
        self._row_bytes = (
            self.embedding_dim * torch.empty((), dtype=torch.float32).element_size()
        )
        self._capacity = cache_bytes // self._row_bytes
        self._slab_rows = min(max(self._capacity, 1), 4096)
        self._slabs: list[torch.Tensor] = []
        self._next_slot = 0
        self._cache: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self._context = threading.local()
        self._epoch_stats: dict[int, dict[str, int]] = {}
        self.metrics = metrics or OperationMetrics(False)
        self._current_bytes = 0
        self._peak_bytes = 0
        self._occurrences = 0
        self._unique_texts = 0
        self._hits = 0
        self._misses = 0
        self._model_inputs = 0
        self._evictions = 0
        self._admission_rejections = 0

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor:
        if not sentences:
            return torch.empty((0, self.embedding_dim))
        epoch = getattr(self._context, "epoch", None)
        with self.metrics.timer("text_normalize"):
            normalized = [normalize_text(value) for value in sentences]
        self.metrics.add("text_occurrences", len(normalized))
        self.metrics.add("text_characters", sum(map(len, normalized)))
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

        with self._lock:
            self._occurrences += len(normalized)
            self._unique_texts += len(unique)
            self._add_epoch_locked(epoch, "occurrences", len(normalized))
            self._add_epoch_locked(epoch, "unique_texts", len(unique))
        embeddings: list[torch.Tensor | None] = [None] * len(unique)
        missing: list[str] = []
        missing_positions: list[int] = []
        memory_hits = 0
        with self.metrics.timer("text_cache_lookup"), self._lock:
            for position, value in enumerate(unique):
                location = self._cache.get(value)
                if location is None:
                    self._misses += 1
                    missing.append(value)
                    missing_positions.append(position)
                else:
                    memory_hits += 1
                    self._hits += 1
                    self._cache.move_to_end(value)
                    slab, row = location
                    embeddings[position] = self._slabs[slab][row].clone()
            self._add_epoch_locked(epoch, "cache_hits", memory_hits)
            self._add_epoch_locked(epoch, "cache_misses", len(missing))

        if missing:
            with self._lock:
                self._model_inputs += len(missing)
                self._add_epoch_locked(epoch, "model_inputs", len(missing))
            encoded = self._encode_missing(missing)
            for position, value, embedding in zip(
                missing_positions, missing, encoded, strict=True
            ):
                embeddings[position] = embedding
                self._put(value, embedding, epoch=epoch)

        complete = [embedding for embedding in embeddings if embedding is not None]
        if len(complete) != len(unique):
            raise RuntimeError("Text embedding result is incomplete")
        with self.metrics.timer("text_expand"):
            return torch.stack(complete)[torch.tensor(inverse, dtype=torch.long)]

    def _find_direct_modules(self) -> tuple[Any, Any] | None:
        try:
            modules = list(self._model)
        except TypeError:
            return None
        if (
            len(modules) != 2
            or not hasattr(modules[0], "preprocess")
            or not hasattr(modules[0], "emb_layer")
            or getattr(modules[1], "pooling_mode", None) != "mean"
        ):
            return None
        return modules[0], modules[1]

    def _encode_missing(self, sentences: list[str]) -> torch.Tensor:
        if self._direct_modules is None:
            with self.metrics.timer("text_model"), torch.inference_mode():
                return self._model.encode(
                    sentences,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).cpu()
        word_embeddings, pooling = self._direct_modules
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for offset in range(0, len(sentences), 64):
                chunk = sentences[offset : offset + 64]
                with self.metrics.timer("text_tokenize"):
                    features = word_embeddings.preprocess(chunk)
                with self.metrics.timer("text_lookup"):
                    features = word_embeddings(features)
                with self.metrics.timer("text_pool"):
                    features = pooling(features)
                outputs.append(features["sentence_embedding"].cpu())
        return torch.cat(outputs)

    def _put(
        self,
        value: str,
        embedding: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> None:
        if self._cache_bytes == 0:
            return
        with self._lock:
            second_access = self._cache_admission == "second" or (
                self._cache_admission == "adaptive"
                and len(self._cache) >= self._capacity
                and self._hits / max(self._hits + self._misses, 1) < 0.20
            )
            if second_access and value not in self._seen:
                self._seen[value] = None
                while len(self._seen) > self._admission_entries:
                    self._seen.popitem(last=False)
                self._admission_rejections += 1
                self._add_epoch_locked(epoch, "admission_rejections", 1)
                return
            self._seen.pop(value, None)
            self._put_locked(value, embedding, epoch)

    def _put_locked(
        self, value: str, embedding: torch.Tensor, epoch: int | None
    ) -> None:
        size = embedding.numel() * embedding.element_size()
        if size != self._row_bytes or self._capacity == 0:
            return
        location = self._cache.pop(value, None)
        if location is None and len(self._cache) >= self._capacity:
            _, location = self._cache.popitem(last=False)
            self._evictions += 1
            self._add_epoch_locked(epoch, "evictions", 1)
        if location is None:
            location = self._allocate_slot()
        slab, row = location
        self._slabs[slab][row].copy_(embedding.detach())
        self._cache[value] = location
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)

    def _allocate_slot(self) -> tuple[int, int]:
        slab_index = self._next_slot // self._slab_rows
        row = self._next_slot % self._slab_rows
        if slab_index == len(self._slabs):
            remaining = self._capacity - self._next_slot
            rows = min(self._slab_rows, remaining)
            slab = torch.empty((rows, self.embedding_dim), dtype=torch.float32)
            self._slabs.append(slab)
            self._current_bytes += slab.untyped_storage().nbytes()
        self._next_slot += 1
        return slab_index, row

    def _add_epoch_locked(self, epoch: int | None, name: str, value: int) -> None:
        if epoch is None or value == 0:
            return
        stats = self._epoch_stats.setdefault(epoch, {})
        stats[name] = stats.get(name, 0) + value

    @contextmanager
    def epoch_context(self, epoch: int):
        previous = getattr(self._context, "epoch", None)
        self._context.epoch = epoch
        try:
            yield
        finally:
            self._context.epoch = previous

    @property
    def stats(self) -> TextEmbeddingStats:
        with self._lock:
            return TextEmbeddingStats(
                occurrences=self._occurrences,
                unique_texts=self._unique_texts,
                cache_hits=self._hits,
                cache_misses=self._misses,
                model_inputs=self._model_inputs,
                evictions=self._evictions,
                current_bytes=self._current_bytes,
                peak_bytes=self._peak_bytes,
                admission_rejections=self._admission_rejections,
            )

    @property
    def epoch_stats(self) -> dict[int, dict[str, int]]:
        with self._lock:
            return {epoch: dict(values) for epoch, values in self._epoch_stats.items()}


def normalize_text(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return value if isinstance(value, str) else str(value)
