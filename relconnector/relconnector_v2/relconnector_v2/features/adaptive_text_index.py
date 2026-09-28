"""Bounded worker-local addresses for immutable shared text embeddings."""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict


class AdaptiveTextIndex:
    """Enable local addresses only when measured shared lookup work is reusable.

    Stores addresses, never embedding copies. Entry accounting includes the
    retained string, integer and an allowance for the ordered dictionary node;
    it is a metadata budget rather than an exact process-RSS measurement.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        window_calls: int = 64,
        min_requests: int = 1024,
        min_lookup_s: float = 0.005,
    ) -> None:
        self.lock = threading.RLock()
        self.max_bytes = max_bytes
        self.window_calls = window_calls
        self.min_requests = min_requests
        self.min_lookup_s = min_lookup_s
        self.active = False
        self.entries: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self.current_bytes = 0
        self.fast_hits = 0
        self.unreported_hits = 0
        self.activations = 0
        self.deactivations = 0
        self.evictions = 0
        self.cooldown = 0
        self.grace_windows = 0
        self.low_reuse_windows = 0
        self.last_shared_hit_rate = 0.0
        self.last_local_hit_rate = 0.0
        self.last_lookup_s = 0.0
        self._reset_window()

    def _reset_window(self) -> None:
        self.calls = self.requests = self.shared_hits = self.local_hits = 0
        self.lookup_s = 0.0

    def get(self, value: str) -> int | None:
        entry = self.entries.get(value)
        if entry is None:
            return None
        self.entries.move_to_end(value)
        return entry[0]

    def remember(self, value: str, slot: int) -> None:
        if not self.active:
            return
        if value in self.entries:
            self.entries.move_to_end(value)
            return
        size = sys.getsizeof(value) + sys.getsizeof(slot) + 128
        if size > self.max_bytes:
            return
        while self.current_bytes + size > self.max_bytes:
            _, (_, removed_bytes) = self.entries.popitem(last=False)
            self.current_bytes -= removed_bytes
            self.evictions += 1
        self.entries[value] = (slot, size)
        self.current_bytes += size

    def observe(
        self, *, requests: int, shared_hits: int, local_hits: int, lookup_s: float
    ) -> None:
        self.calls += 1
        self.requests += requests
        self.shared_hits += shared_hits
        self.local_hits += local_hits
        self.lookup_s += lookup_s
        self.fast_hits += local_hits
        self.unreported_hits += local_hits
        if self.calls < self.window_calls or self.requests < self.min_requests:
            return
        self.last_shared_hit_rate = self.shared_hits / max(self.requests, 1)
        self.last_local_hit_rate = self.local_hits / max(self.requests, 1)
        self.last_lookup_s = self.lookup_s
        if self.active:
            if self.grace_windows:
                self.grace_windows -= 1
            elif self.last_local_hit_rate < 0.20:
                self.low_reuse_windows += 1
                if self.low_reuse_windows >= 2:
                    self.active = False
                    self.deactivations += 1
                    self.entries.clear()
                    self.current_bytes = 0
                    self.cooldown = min(4 * self.deactivations, 32)
            else:
                self.low_reuse_windows = 0
        elif self.cooldown:
            self.cooldown -= 1
        elif (
            self.max_bytes > 0
            and self.last_shared_hit_rate >= 0.80
            and self.lookup_s >= self.min_lookup_s
        ):
            self.active = True
            self.activations += 1
            self.grace_windows = 2
            self.low_reuse_windows = 0
        self._reset_window()

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": "adaptive",
            "active": self.active,
            "entries": len(self.entries),
            "estimated_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "fast_hits": self.fast_hits,
            "activations": self.activations,
            "deactivations": self.deactivations,
            "evictions": self.evictions,
            "cooldown_windows": self.cooldown,
            "last_shared_hit_rate": self.last_shared_hit_rate,
            "last_local_hit_rate": self.last_local_hit_rate,
            "last_lookup_s": self.last_lookup_s,
        }
