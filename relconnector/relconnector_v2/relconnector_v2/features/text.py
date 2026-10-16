"""Frozen GloVe sentence embeddings for sampled SQL rows."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import torch

from relconnector_v2.observability import OperationMetrics

from .cost_cache import CostAwareTextCache
from .disk_text import AdaptiveDiskTextEmbeddingCache
from .shared_text import SharedTextEmbeddingCache


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
    # Long relational text pays a steep padding/memory-bandwidth penalty at
    # larger chunks. Keep the external feature batch large, but encode it in
    # small sorted chunks inside each CPU worker.
    direct_batch_size = 8

    def __init__(
        self,
        *,
        model_path: str | None = None,
        cache_bytes: int = 0,
        cache_admission: str = "always",
        admission_entries: int = 250_000,
        execution: Literal["official", "direct"] = "direct",
        shared_cache: SharedTextEmbeddingCache | None = None,
        disk_cache_path: str | None = None,
        disk_cache_namespace: str = "",
        disk_cache_mode: Literal["off", "on", "adaptive"] = "off",
        database_bytes: int = 0,
        disk_cache_min_database_bytes: int = 256 * 1024 * 1024,
        disk_cache_min_model_inputs: int = 4096,
        disk_cache_max_memory_hit_rate: float = 0.10,
        disk_cache_min_model_time_s: float = 1.0,
        metrics: OperationMetrics | None = None,
        cost_strategy: bool = False,
        cost_disk_policy: Literal["budget", "roi"] = "budget",
        cost_disk_bytes: int = 0,
        training_epochs: int = 1,
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
        self._shared_cache = shared_cache
        self._reuse_disabled = (
            not cost_strategy
            and cache_bytes == 0
            and shared_cache is None
            and (disk_cache_mode == "off" or disk_cache_path is None)
        )
        self._disk_cache = AdaptiveDiskTextEmbeddingCache(
            path=None if cost_strategy else disk_cache_path,
            namespace=disk_cache_namespace,
            embedding_dim=self.embedding_dim,
            mode="off" if cost_strategy else disk_cache_mode,
            database_bytes=database_bytes,
            min_database_bytes=disk_cache_min_database_bytes,
            min_model_inputs=disk_cache_min_model_inputs,
            max_memory_hit_rate=disk_cache_max_memory_hit_rate,
            min_model_time_s=disk_cache_min_model_time_s,
        )
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
        self._cost_cache = (
            CostAwareTextCache(
                max_bytes=cache_bytes,
                dimension=self.embedding_dim,
                disk_dir=(
                    None
                    if disk_cache_path is None or disk_cache_mode == "off"
                    else Path(disk_cache_path + ".blocks-v1")
                ),
                disk_max_bytes=cost_disk_bytes,
                remaining_epochs=training_epochs,
                disk_policy=cost_disk_policy,
            )
            if cost_strategy
            else None
        )

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
        if self._reuse_disabled:
            result, _ = self._encode_and_record(unique, epoch=epoch)
            with self._lock:
                self._misses += len(unique)
                self._add_epoch_locked(epoch, "cache_misses", len(unique))
            return result[torch.tensor(inverse, dtype=torch.long)]
        if self._cost_cache is not None:
            result, hits = self._cost_cache.resolve(
                unique,
                column=getattr(self._context, "column", "unknown"),
                epoch=epoch,
                encode=lambda values: self._encode_and_record(values, epoch=epoch),
            )
            with self._lock:
                self._hits += hits
                self._misses += len(unique) - hits
                self._add_epoch_locked(epoch, "cache_hits", hits)
                self._add_epoch_locked(epoch, "cache_misses", len(unique) - hits)
            return result[torch.tensor(inverse, dtype=torch.long)]
        embeddings: list[torch.Tensor | None] = [None] * len(unique)
        if self._shared_cache is not None:
            with self.metrics.timer("shared_text_cache_lookup"):
                lookup = self._shared_cache.lookup_or_reserve(unique)
            for position, embedding in lookup.hits.items():
                embeddings[position] = embedding
            missing_positions = sorted([*lookup.reservations, *lookup.uncached])
            missing = [unique[position] for position in missing_positions]
            with self._lock:
                self._hits += len(lookup.hits)
                self._misses += len(missing)
                self._add_epoch_locked(epoch, "cache_hits", len(lookup.hits))
                self._add_epoch_locked(epoch, "cache_misses", len(missing))
            if missing:
                try:
                    encoded = self._resolve_missing(missing, epoch=epoch)
                    published: dict[int, torch.Tensor] = {}
                    for local, position in enumerate(missing_positions):
                        embedding = encoded[local]
                        embeddings[position] = embedding
                        if position in lookup.reservations:
                            published[position] = embedding
                    self._shared_cache.publish(lookup.reservations, published)
                except BaseException:
                    self._shared_cache.abort(lookup.reservations)
                    raise
            pending_hits = self._shared_cache.wait(dict(lookup.pending))
            for position, embedding in pending_hits.items():
                embeddings[position] = embedding
            with self._lock:
                self._hits += len(pending_hits)
                self._add_epoch_locked(epoch, "cache_hits", len(pending_hits))
            complete = [embedding for embedding in embeddings if embedding is not None]
            if len(complete) != len(unique):
                raise RuntimeError("Shared text embedding result is incomplete")
            with self.metrics.timer("text_expand"):
                return torch.stack(complete)[torch.tensor(inverse, dtype=torch.long)]

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
            encoded = self._resolve_missing(missing, epoch=epoch)
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
        pooling_mode = (
            getattr(modules[1], "pooling_mode", None)
            if len(modules) == 2
            else None
        )
        get_pooling_mode = (
            getattr(modules[1], "get_pooling_mode_str", None)
            if len(modules) == 2
            else None
        )
        if pooling_mode is None and callable(get_pooling_mode):
            pooling_mode = get_pooling_mode()
        if (
            len(modules) != 2
            or not (
                hasattr(modules[0], "tokenize")
                or hasattr(modules[0], "preprocess")
            )
            or not hasattr(modules[0], "emb_layer")
            or pooling_mode != "mean"
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
        # SentenceTransformer.encode() sorts by length before batching. Preserve
        # that behavior here to avoid padding every short text to an unrelated
        # long text while retaining the caller's original row order.
        order = sorted(
            range(len(sentences)),
            key=lambda position: len(sentences[position]),
            reverse=True,
        )
        ordered = [sentences[position] for position in order]
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for offset in range(0, len(ordered), self.direct_batch_size):
                chunk = ordered[offset : offset + self.direct_batch_size]
                with self.metrics.timer("text_tokenize"):
                    tokenize = getattr(word_embeddings, "tokenize", None)
                    features = (
                        tokenize(chunk)
                        if tokenize is not None
                        else word_embeddings.preprocess(chunk)
                    )
                with self.metrics.timer("text_lookup"):
                    features = word_embeddings(features)
                with self.metrics.timer("text_pool"):
                    features = pooling(features)
                outputs.append(features["sentence_embedding"].cpu())
        encoded = torch.cat(outputs)
        restored = torch.empty_like(encoded)
        restored[torch.tensor(order, dtype=torch.long)] = encoded
        return restored

    def _resolve_missing(
        self,
        values: list[str],
        *,
        epoch: int | None,
        coordinate_disk: bool = True,
    ) -> torch.Tensor:
        with self.metrics.timer("disk_text_cache_lookup"):
            lookup = self._disk_cache.lookup_or_reserve(
                values,
                coordinated=coordinate_disk,
            )
        if lookup is None:
            encoded, model_time_s = self._encode_and_record(values, epoch=epoch)
            with self._lock:
                memory_hit_rate = self._hits / max(self._hits + self._misses, 1)
            with self.metrics.timer("disk_text_cache_observe"):
                self._disk_cache.observe_encoded(
                    values,
                    encoded,
                    model_time_s=model_time_s,
                    memory_hit_rate=memory_hit_rate,
                )
            return encoded

        self.metrics.add("disk_text_cache_hits", len(lookup.hits))
        self.metrics.add(
            "disk_text_cache_misses",
            len(lookup.reservations) + len(lookup.pending) + len(lookup.uncached),
        )
        embeddings: list[torch.Tensor | None] = [None] * len(values)
        for position, embedding in lookup.hits.items():
            embeddings[position] = embedding

        missing_positions = sorted([*lookup.reservations, *lookup.uncached])
        if missing_positions:
            missing = [values[position] for position in missing_positions]
            try:
                encoded, model_time_s = self._encode_and_record(missing, epoch=epoch)
                published: dict[int, torch.Tensor] = {}
                buffered_values: list[str] = []
                buffered_embeddings: list[torch.Tensor] = []
                for local, position in enumerate(missing_positions):
                    embedding = encoded[local]
                    embeddings[position] = embedding
                    if position in lookup.reservations:
                        published[position] = embedding
                    else:
                        buffered_values.append(values[position])
                        buffered_embeddings.append(embedding)
                with self.metrics.timer("disk_text_cache_publish"):
                    self._disk_cache.publish(lookup.reservations, published)
                self.metrics.add("disk_text_cache_writes", len(published))
                if buffered_values:
                    with self._lock:
                        memory_hit_rate = self._hits / max(self._hits + self._misses, 1)
                    with self.metrics.timer("disk_text_cache_buffer"):
                        self._disk_cache.observe_encoded(
                            buffered_values,
                            torch.stack(buffered_embeddings),
                            model_time_s=model_time_s,
                            memory_hit_rate=memory_hit_rate,
                        )
            except BaseException:
                self._disk_cache.abort(lookup.reservations)
                raise

        with self.metrics.timer("disk_text_cache_wait"):
            pending_hits = self._disk_cache.wait(lookup.pending)
        for position, embedding in pending_hits.items():
            embeddings[position] = embedding
        complete = [embedding for embedding in embeddings if embedding is not None]
        if len(complete) != len(values):
            raise RuntimeError("Persistent text embedding result is incomplete")
        return torch.stack(complete)

    def _encode_and_record(
        self,
        values: list[str],
        *,
        epoch: int | None,
    ) -> tuple[torch.Tensor, float]:
        with self._lock:
            self._model_inputs += len(values)
            self._add_epoch_locked(epoch, "model_inputs", len(values))
        started = time.perf_counter()
        encoded = self._encode_missing(values)
        return encoded, time.perf_counter() - started

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
        shared = None if self._shared_cache is None else self._shared_cache.snapshot()
        with self._lock:
            if self._cost_cache is not None:
                self._current_bytes = self._cost_cache.bytes
                self._peak_bytes = self._cost_cache.peak_bytes
            return TextEmbeddingStats(
                occurrences=self._occurrences,
                unique_texts=self._unique_texts,
                cache_hits=self._hits,
                cache_misses=self._misses,
                model_inputs=self._model_inputs,
                evictions=self._evictions,
                current_bytes=(
                    self._current_bytes
                    if shared is None
                    else int(shared["current_bytes"])
                ),
                peak_bytes=(
                    self._peak_bytes if shared is None else int(shared["current_bytes"])
                ),
                admission_rejections=self._admission_rejections,
            )

    @property
    def epoch_stats(self) -> dict[int, dict[str, int]]:
        with self._lock:
            return {epoch: dict(values) for epoch, values in self._epoch_stats.items()}

    @property
    def disk_cache_stats(self) -> dict[str, object]:
        if self._cost_cache is not None:
            return self._cost_cache.snapshot()
        return self._disk_cache.snapshot()

    def for_column(self, table: str, column: str) -> TextEmbedder:
        return _ColumnTextEmbedder(self, f"{table}.{column}")

    def close(self) -> None:
        self._disk_cache.close()


class _ColumnTextEmbedder:
    embedding_dim = GloveTextEmbedder.embedding_dim

    def __init__(self, parent: GloveTextEmbedder, key: str) -> None:
        self.parent, self.key = parent, key

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor:
        previous = getattr(self.parent._context, "column", "unknown")
        self.parent._context.column = self.key
        try:
            return self.parent(sentences)
        finally:
            self.parent._context.column = previous


def normalize_text(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return value if isinstance(value, str) else str(value)
