"""Run-scoped shared-memory cache for fixed-width text embeddings."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SharedTextLookup:
    hits: dict[int, torch.Tensor]
    reservations: dict[int, int]
    pending: dict[int, int]
    uncached: list[int]


class SharedTextEmbeddingCache:
    """Process-safe hash table backed by a shared float32 tensor slab.

    The first process that observes a key reserves its slot. Other processes
    wait for publication instead of encoding the same text concurrently.
    Entries live for the complete training run and therefore cross epoch
    boundaries.
    """

    def __init__(
        self,
        context: Any,
        *,
        max_bytes: int,
        embedding_dim: int,
        wait_timeout_s: float = 300.0,
    ) -> None:
        if max_bytes < 0 or embedding_dim <= 0 or wait_timeout_s <= 0:
            raise ValueError("shared text cache limits must be valid")
        row_bytes = embedding_dim * torch.empty((), dtype=torch.float32).element_size()
        self.capacity = max_bytes // row_bytes
        self.embedding_dim = embedding_dim
        self.max_bytes = self.capacity * row_bytes
        self.wait_timeout_s = wait_timeout_s
        rows = max(self.capacity, 1)
        self._hash_hi = torch.zeros(rows, dtype=torch.int64).share_memory_()
        self._hash_lo = torch.zeros(rows, dtype=torch.int64).share_memory_()
        self._states = torch.zeros(rows, dtype=torch.uint8).share_memory_()
        self._owners = torch.zeros(rows, dtype=torch.int32).share_memory_()
        self._values = torch.empty(
            (rows, embedding_dim), dtype=torch.float32
        ).share_memory_()
        self._lock = context.RLock()
        self._condition = context.Condition(self._lock)
        self._hits = context.Value("q", 0, lock=False)
        self._misses = context.Value("q", 0, lock=False)
        self._reservations = context.Value("q", 0, lock=False)
        self._published = context.Value("q", 0, lock=False)
        self._full_rejections = context.Value("q", 0, lock=False)
        self._waits = context.Value("q", 0, lock=False)

    def lookup_or_reserve(self, values: list[str]) -> SharedTextLookup:
        if self.capacity == 0:
            return SharedTextLookup({}, {}, {}, list(range(len(values))))
        owner = os.getpid()
        hits: dict[int, torch.Tensor] = {}
        reservations: dict[int, int] = {}
        pending: dict[int, int] = {}
        uncached: list[int] = []
        with self._condition:
            for position, value in enumerate(values):
                high, low = _digest(value)
                slot = self._find_slot(high, low)
                if slot is None:
                    self._misses.value += 1
                    self._full_rejections.value += 1
                    uncached.append(position)
                    continue
                state = int(self._states[slot])
                if state == 0:
                    self._hash_hi[slot] = high
                    self._hash_lo[slot] = low
                    self._owners[slot] = owner
                    self._states[slot] = 1
                    self._misses.value += 1
                    self._reservations.value += 1
                    reservations[position] = slot
                elif state == 2:
                    self._hits.value += 1
                    hits[position] = self._values[slot].clone()
                else:
                    self._waits.value += 1
                    pending[position] = slot

        return SharedTextLookup(hits, reservations, pending, uncached)

    def wait(self, pending: dict[int, int]) -> dict[int, torch.Tensor]:
        hits: dict[int, torch.Tensor] = {}
        deadline = time.monotonic() + self.wait_timeout_s
        with self._condition:
            while pending:
                completed = [
                    position
                    for position, slot in pending.items()
                    if int(self._states[slot]) == 2
                ]
                for position in completed:
                    slot = pending.pop(position)
                    self._hits.value += 1
                    hits[position] = self._values[slot].clone()
                if not pending:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    owners = sorted(
                        {int(self._owners[slot]) for slot in pending.values()}
                    )
                    raise TimeoutError(
                        "timed out waiting for shared text reservations "
                        f"owned by processes {owners}"
                    )
                self._condition.wait(min(remaining, 0.1))
        return hits

    def publish(
        self,
        reservations: dict[int, int],
        embeddings: dict[int, torch.Tensor],
    ) -> None:
        if set(reservations) != set(embeddings):
            raise ValueError("published embeddings do not match reservations")
        owner = os.getpid()
        with self._condition:
            for position, slot in reservations.items():
                if (
                    int(self._states[slot]) != 1
                    or int(self._owners[slot]) != owner
                ):
                    raise RuntimeError("shared text reservation ownership changed")
                embedding = embeddings[position].detach().to(
                    dtype=torch.float32, device="cpu"
                )
                if embedding.shape != (self.embedding_dim,):
                    raise ValueError("shared text embedding has invalid shape")
                self._values[slot].copy_(embedding)
                self._states[slot] = 2
                self._owners[slot] = 0
                self._published.value += 1
            self._condition.notify_all()

    def abort(self, reservations: dict[int, int]) -> None:
        owner = os.getpid()
        with self._condition:
            for slot in reservations.values():
                if (
                    int(self._states[slot]) == 1
                    and int(self._owners[slot]) == owner
                ):
                    self._states[slot] = 0
                    self._owners[slot] = 0
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int | float]:
        with self._condition:
            entries = int((self._states == 2).sum())
            requests = self._hits.value + self._misses.value
            return {
                "capacity": self.capacity,
                "entries": entries,
                "current_bytes": entries * self.embedding_dim * 4,
                "max_bytes": self.max_bytes,
                "hits": int(self._hits.value),
                "misses": int(self._misses.value),
                "hit_rate": self._hits.value / max(requests, 1),
                "reservations": int(self._reservations.value),
                "published": int(self._published.value),
                "reservation_waits": int(self._waits.value),
                "full_rejections": int(self._full_rejections.value),
            }

    def _find_slot(self, high: int, low: int) -> int | None:
        start = (high ^ low) % self.capacity
        for offset in range(self.capacity):
            slot = (start + offset) % self.capacity
            state = int(self._states[slot])
            if state == 0:
                return slot
            if (
                int(self._hash_hi[slot]) == high
                and int(self._hash_lo[slot]) == low
            ):
                return slot
        return None


def _digest(value: str) -> tuple[int, int]:
    digest = hashlib.blake2b(value.encode(), digest_size=16).digest()
    return (
        int.from_bytes(digest[:8], "little", signed=True),
        int.from_bytes(digest[8:], "little", signed=True),
    )
