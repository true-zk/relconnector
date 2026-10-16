"""Run-scoped shared-memory cache for fixed-width text embeddings."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .adaptive_text_index import AdaptiveTextIndex


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
        max_probe: int = 32,
        adaptive_local_index: bool = False,
        local_index_max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if max_bytes < 0 or embedding_dim <= 0 or wait_timeout_s <= 0 or max_probe <= 0:
            raise ValueError("shared text cache limits must be valid")
        if local_index_max_bytes < 0:
            raise ValueError("local text index budget must be nonnegative")
        self.adaptive_local_index = adaptive_local_index
        self.local_index_max_bytes = local_index_max_bytes
        row_bytes = embedding_dim * torch.empty((), dtype=torch.float32).element_size()
        self.capacity = max_bytes // row_bytes
        self.embedding_dim = embedding_dim
        self.max_bytes = self.capacity * row_bytes
        self.wait_timeout_s = wait_timeout_s
        self.max_probe = max_probe
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
        if self.adaptive_local_index:
            return self._lookup_adaptive(values)
        return self._lookup_shared(values)

    def _lookup_shared(self, values: list[str]) -> SharedTextLookup:
        ready, reservations, pending, uncached = self._lookup_slots(
            values, list(range(len(values)))
        )
        return SharedTextLookup(self._gather(ready), reservations, pending, uncached)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        # Reservation waits already have a deadline, but entering the process
        # lock used to wait forever if its owner stopped making progress.
        if not self._condition.acquire(timeout=self.wait_timeout_s):
            raise TimeoutError("timed out acquiring shared text cache lock")
        try:
            yield
        finally:
            self._condition.release()

    def _metadata(self) -> tuple[np.ndarray, ...]:
        # Rebuild zero-copy views after spawn; pickling NumPy views would copy
        # metadata and disconnect the worker from the shared tensor storage.
        if getattr(self, "_metadata_pid", None) != os.getpid():
            self._metadata_views = tuple(
                tensor.numpy()
                for tensor in (self._hash_hi, self._hash_lo, self._states, self._owners)
            )
            self._metadata_pid = os.getpid()
        return self._metadata_views

    def _lookup_slots(
        self, values: list[str], positions: list[int], *, buffered_hits: int = 0
    ) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[int]]:
        owner = os.getpid()
        high_keys, low_keys, states, owners = self._metadata()
        # Hash strings before acquiring the process lock. Probe scalar metadata
        # through NumPy views instead of dispatching a tensor operation per probe.
        keys = [_digest(values[position]) for position in positions]
        ready: dict[int, int] = {}
        reservations: dict[int, int] = {}
        pending: dict[int, int] = {}
        uncached: list[int] = []
        with self._locked():
            for position, (high, low) in zip(positions, keys, strict=True):
                slot = self._find_slot(high, low)
                if slot is None:
                    uncached.append(position)
                    continue
                state = states[slot]
                if state == 0:
                    high_keys[slot] = high
                    low_keys[slot] = low
                    owners[slot] = owner
                    states[slot] = 1
                    reservations[position] = slot
                elif state == 2:
                    ready[position] = slot
                else:
                    pending[position] = slot
            self._hits.value += len(ready) + buffered_hits
            self._misses.value += len(reservations) + len(uncached)
            self._full_rejections.value += len(uncached)
            self._reservations.value += len(reservations)
            self._waits.value += len(pending)
        return ready, reservations, pending, uncached

    def _gather(self, ready: dict[int, int]) -> dict[int, torch.Tensor]:
        if not ready:
            return {}
        # Published rows are immutable. Copy them in one operation outside the
        # process lock; callers own the returned storage, never the shared slab.
        gathered = self._values.index_select(
            0, torch.tensor(list(ready.values()), dtype=torch.long)
        )
        return dict(zip(ready, gathered.unbind(), strict=True))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Local locks and addresses belong to this process, not spawned workers.
        state.pop("_local_index", None)
        state.pop("_local_index_pid", None)
        state.pop("_metadata_views", None)
        state.pop("_metadata_pid", None)
        return state

    def _worker_index(self) -> AdaptiveTextIndex:
        if getattr(self, "_local_index_pid", None) != os.getpid():
            self._local_index = AdaptiveTextIndex(self.local_index_max_bytes)
            self._local_index_pid = os.getpid()
        return self._local_index

    def _lookup_adaptive(self, values: list[str]) -> SharedTextLookup:
        index = self._worker_index()
        with index.lock:
            started = time.perf_counter()
            if not index.active:
                # Low-reuse workloads pay only for window counters while off;
                # do not gather addresses or construct an unused local index.
                lookup = self._lookup_shared(values)
                index.observe(
                    requests=len(values),
                    shared_hits=len(lookup.hits),
                    local_hits=0,
                    lookup_s=time.perf_counter() - started,
                )
                return lookup
            ready: dict[int, int] = {}
            unresolved: list[int] = []
            for position, value in enumerate(values):
                slot = index.get(value) if index.active else None
                if slot is None:
                    unresolved.append(position)
                else:
                    ready[position] = slot
            local_hits = len(ready)
            reservations: dict[int, int] = {}
            pending: dict[int, int] = {}
            uncached: list[int] = []
            learned: dict[str, int] = {}
            shared_hits = 0
            if unresolved or index.unreported_hits >= 65536:
                found, reservations, pending, uncached = self._lookup_slots(
                    values, unresolved, buffered_hits=index.unreported_hits
                )
                index.unreported_hits = 0
                shared_hits = len(found)
                ready.update(found)
                learned = {values[position]: slot for position, slot in found.items()}
            hits = self._gather(ready)
            index.observe(
                requests=len(values),
                shared_hits=shared_hits,
                local_hits=local_hits,
                lookup_s=time.perf_counter() - started,
            )
            for value, slot in learned.items():
                index.remember(value, slot)
            return SharedTextLookup(hits, reservations, pending, uncached)

    def wait(self, pending: dict[int, int]) -> dict[int, torch.Tensor]:
        if not pending:
            return {}
        ready: dict[int, int] = {}
        _, _, states, owners = self._metadata()
        deadline = time.monotonic() + self.wait_timeout_s
        with self._locked():
            while pending:
                completed = [
                    position for position, slot in pending.items() if states[slot] == 2
                ]
                for position in completed:
                    ready[position] = pending.pop(position)
                if not pending:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    owner_pids = sorted(
                        {int(owners[slot]) for slot in pending.values()}
                    )
                    raise TimeoutError(
                        "timed out waiting for shared text reservations "
                        f"owned by processes {owner_pids}"
                    )
                self._condition.wait(min(remaining, 0.1))
            self._hits.value += len(ready)
        return self._gather(ready)

    def publish(
        self,
        reservations: dict[int, int],
        embeddings: dict[int, torch.Tensor],
    ) -> None:
        if set(reservations) != set(embeddings):
            raise ValueError("published embeddings do not match reservations")
        if not reservations:
            return
        positions = list(reservations)
        values = [embeddings[position] for position in positions]
        if any(value.shape != (self.embedding_dim,) for value in values):
            raise ValueError("shared text embedding has invalid shape")
        encoded = torch.stack(values).detach().to(dtype=torch.float32, device="cpu")
        slots = list(reservations.values())
        indices = torch.tensor(slots, dtype=torch.long)
        owner = os.getpid()
        _, _, states, owners = self._metadata()
        with self._locked():
            if any(states[slot] != 1 or owners[slot] != owner for slot in slots):
                raise RuntimeError("shared text reservation ownership changed")
            self._values.index_copy_(0, indices, encoded)
            states[slots] = 2
            owners[slots] = 0
            self._published.value += len(slots)
            self._condition.notify_all()

    def abort(self, reservations: dict[int, int]) -> None:
        owner = os.getpid()
        with self._locked():
            for slot in reservations.values():
                if int(self._states[slot]) == 1 and int(self._owners[slot]) == owner:
                    self._states[slot] = 0
                    self._owners[slot] = 0
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        local_stats: dict[str, object] = {"mode": "off", "active": False}
        if self.adaptive_local_index:
            index = self._worker_index()
            with index.lock:
                with self._locked():
                    self._hits.value += index.unreported_hits
                    index.unreported_hits = 0
                local_stats = index.snapshot()
        with self._locked():
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
                "max_probe": self.max_probe,
                "local_index": local_stats,
            }

    def _find_slot(self, high: int, low: int) -> int | None:
        high_keys, low_keys, states, _ = self._metadata()
        start = (high ^ low) % self.capacity
        for offset in range(min(self.capacity, self.max_probe)):
            slot = (start + offset) % self.capacity
            state = states[slot]
            if state == 0:
                return slot
            if high_keys[slot] == high and low_keys[slot] == low:
                return slot
        return None


def _digest(value: str) -> tuple[int, int]:
    digest = hashlib.blake2b(value.encode(), digest_size=16).digest()
    return (
        int.from_bytes(digest[:8], "little", signed=True),
        int.from_bytes(digest[8:], "little", signed=True),
    )
