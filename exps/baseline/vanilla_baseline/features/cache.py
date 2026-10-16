"""Byte-bounded LRU cache for decoded database feature blocks."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureBlockKey:
    database_version: str
    table: str
    columns: tuple[str, ...]
    block: int


class FeatureBlockCache:
    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes cannot be negative")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[FeatureBlockKey, pd.DataFrame] = OrderedDict()
        self._sizes: dict[FeatureBlockKey, int] = {}
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: FeatureBlockKey) -> pd.DataFrame | None:
        frame = self._entries.get(key)
        if frame is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return frame

    def put(self, key: FeatureBlockKey, frame: pd.DataFrame) -> None:
        if self.max_bytes == 0:
            return
        size = int(frame.memory_usage(index=True, deep=True).sum())
        if size > self.max_bytes:
            return
        previous = self._sizes.pop(key, 0)
        if previous:
            self.current_bytes -= previous
            self._entries.pop(key, None)
        while self._entries and self.current_bytes + size > self.max_bytes:
            evicted_key, _ = self._entries.popitem(last=False)
            self.current_bytes -= self._sizes.pop(evicted_key)
        self._entries[key] = frame
        self._sizes[key] = size
        self.current_bytes += size

    @property
    def hit_rate(self) -> float:
        requests = self.hits + self.misses
        return self.hits / requests if requests else 0.0
