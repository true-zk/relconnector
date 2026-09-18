"""Byte-bounded LRU cache for decoded database feature blocks."""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd
import torch
from torch_frame import TensorFrame
from torch_frame.data.multi_tensor import _MultiTensor


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
        self.insertions = 0
        self.evictions = 0
        self.oversized_rejections = 0
        self.peak_bytes = 0

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
            self.oversized_rejections += 1
            return
        previous = self._sizes.pop(key, 0)
        if previous:
            self.current_bytes -= previous
            self._entries.pop(key, None)
        while self._entries and self.current_bytes + size > self.max_bytes:
            evicted_key, _ = self._entries.popitem(last=False)
            self.current_bytes -= self._sizes.pop(evicted_key)
            self.evictions += 1
        self._entries[key] = frame
        self._sizes[key] = size
        self.current_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self.insertions += 1

    @property
    def hit_rate(self) -> float:
        requests = self.hits + self.misses
        return self.hits / requests if requests else 0.0


@dataclass(frozen=True)
class EncodedFeatureEntry:
    key: int
    table: str
    node_ids: torch.Tensor
    frame: TensorFrame
    charged_bytes: int


class EncodedFeatureCache:
    """Byte-bounded encoded chunks with a node-to-chunk lookup index."""

    def __init__(
        self,
        max_bytes: int,
        *,
        admission: str = "second",
        admission_entries: int = 250_000,
    ) -> None:
        if max_bytes < 0 or admission_entries <= 0:
            raise ValueError("encoded cache limits must be valid")
        if admission not in {"always", "second", "adaptive"}:
            raise ValueError(
                "encoded cache admission must be always, second or adaptive"
            )
        self.max_bytes = max_bytes
        self.admission = admission
        self.admission_entries = admission_entries
        self._entries: OrderedDict[int, EncodedFeatureEntry] = OrderedDict()
        self._index: dict[str, dict[int, tuple[int, int]]] = {}
        self._seen: OrderedDict[tuple[str, int], None] = OrderedDict()
        self._next_key = 0
        self._lock = threading.Lock()
        self._table_requests: dict[str, int] = {}
        self._table_hits: dict[str, int] = {}
        self._table_inserted: dict[str, int] = {}
        self._adaptive_warm_rows = 4096
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hit_rows = 0
        self.miss_rows = 0
        self.inserted_rows = 0
        self.evictions = 0
        self.oversized_rejections = 0
        self.admission_rejections = 0
        self._epoch_stats: dict[int, dict[str, int]] = {}

    def lookup_many(
        self, table: str, node_ids: torch.Tensor, *, epoch: int | None = None
    ) -> tuple[list[tuple[EncodedFeatureEntry, list[tuple[int, int]]]], list[int]]:
        """Return entry-grouped ``(requested position, row)`` and misses."""
        groups: dict[int, list[tuple[int, int]]] = {}
        misses: list[int] = []
        hit_rows = 0
        ids = [int(value) for value in node_ids.tolist()]
        with self._lock:
            table_index = self._index.get(table, {})
            self._table_requests[table] = self._table_requests.get(table, 0) + len(ids)
            touched: set[int] = set()
            for position, node_id in enumerate(ids):
                location = table_index.get(node_id)
                if location is None:
                    misses.append(position)
                    self.miss_rows += 1
                    continue
                entry_key, row = location
                groups.setdefault(entry_key, []).append((position, row))
                touched.add(entry_key)
                hit_rows += 1
                self.hit_rows += 1
            self._add_epoch_locked(epoch, "hit_rows", hit_rows)
            self._add_epoch_locked(epoch, "miss_rows", len(misses))
            for entry_key in touched:
                if entry_key in self._entries:
                    self._entries.move_to_end(entry_key)
            self._table_hits[table] = self._table_hits.get(table, 0) + hit_rows
            resolved = [
                (self._entries[key], positions) for key, positions in groups.items()
            ]
        return resolved, misses

    def put(
        self,
        table: str,
        node_ids: torch.Tensor,
        frame: TensorFrame,
        *,
        epoch: int | None = None,
    ) -> EncodedFeatureEntry | None:
        if self.max_bytes == 0 or len(node_ids) == 0:
            return None
        ids = [int(value) for value in node_ids.tolist()]
        with self._lock:
            table_index = self._index.setdefault(table, {})
            candidate_positions = [
                position
                for position, node_id in enumerate(ids)
                if node_id not in table_index
            ]
            candidate_ids = [ids[position] for position in candidate_positions]
            local_positions = self._admitted_positions(table, candidate_ids)
            admitted_positions = [
                candidate_positions[position] for position in local_positions
            ]
            if not admitted_positions:
                self.admission_rejections += len(ids)
                self._add_epoch_locked(epoch, "admission_rejections", len(ids))
                return None
            admitted_ids = node_ids[admitted_positions].clone()
            admitted_frame = frame[admitted_positions]
            charged = (
                _tensor_frame_storage_bytes(admitted_frame)
                + admitted_ids.untyped_storage().nbytes()
                + _INDEX_BYTES_PER_ROW * len(admitted_ids)
            )
            if charged > self.max_bytes:
                self.oversized_rejections += 1
                self._add_epoch_locked(epoch, "oversized_rejections", 1)
                return None
            while self._entries and self.current_bytes + charged > self.max_bytes:
                self._evict_oldest(epoch)
            key = self._next_key
            self._next_key += 1
            entry = EncodedFeatureEntry(
                key, table, admitted_ids, admitted_frame, charged
            )
            self._entries[key] = entry
            for row, node_id in enumerate(admitted_ids.tolist()):
                table_index[int(node_id)] = (key, row)
            self.current_bytes += charged
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)
            self.inserted_rows += len(admitted_ids)
            self._table_inserted[table] = self._table_inserted.get(table, 0) + len(
                admitted_ids
            )
            self._add_epoch_locked(epoch, "inserted_rows", len(admitted_ids))
            return entry

    def _admitted_positions(self, table: str, ids: list[int]) -> list[int]:
        if self.admission == "always":
            return list(range(len(ids)))
        output: list[int] = []
        start = 0
        if self.admission == "adaptive":
            table_index = self._index.get(table, {})
            requests = self._table_requests.get(table, 0)
            hit_rate = self._table_hits.get(table, 0) / max(requests, 1)
            direct_limit = (
                len(ids)
                if hit_rate >= 0.10
                else max(self._adaptive_warm_rows - len(table_index), 0)
            )
            if self.current_bytes < self.max_bytes * 0.9:
                start = min(direct_limit, len(ids))
                output.extend(range(start))
        for position, node_id in enumerate(ids[start:], start=start):
            key = (table, node_id)
            if key in self._seen:
                self._seen.pop(key)
                output.append(position)
            else:
                self._seen[key] = None
        while len(self._seen) > self.admission_entries:
            self._seen.popitem(last=False)
        return output

    def _evict_oldest(self, epoch: int | None = None) -> None:
        _, entry = self._entries.popitem(last=False)
        table_index = self._index[entry.table]
        for node_id in entry.node_ids.tolist():
            table_index.pop(int(node_id), None)
        if not table_index:
            self._index.pop(entry.table)
        self.current_bytes -= entry.charged_bytes
        self.evictions += 1
        self._add_epoch_locked(epoch, "evictions", 1)

    def _add_epoch_locked(self, epoch: int | None, name: str, value: int) -> None:
        if epoch is None or value == 0:
            return
        stats = self._epoch_stats.setdefault(epoch, {})
        stats[name] = stats.get(name, 0) + value

    @property
    def hit_rate(self) -> float:
        requests = self.hit_rows + self.miss_rows
        return self.hit_rows / requests if requests else 0.0

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def epoch_stats(self) -> dict[int, dict[str, int | float]]:
        with self._lock:
            output: dict[int, dict[str, int | float]] = {}
            for epoch, values in self._epoch_stats.items():
                stats: dict[str, int | float] = dict(values)
                requests = values.get("hit_rows", 0) + values.get("miss_rows", 0)
                stats["hit_rate"] = values.get("hit_rows", 0) / max(requests, 1)
                output[epoch] = stats
            return output

    @property
    def table_stats(self) -> dict[str, dict[str, int | float | str]]:
        with self._lock:
            output: dict[str, dict[str, int | float | str]] = {}
            for table, requests in self._table_requests.items():
                hits = self._table_hits.get(table, 0)
                hit_rate = hits / max(requests, 1)
                indexed_rows = len(self._index.get(table, {}))
                if hit_rate >= 0.20 and indexed_rows <= self._adaptive_warm_rows:
                    mode = "dimension-hot"
                elif hit_rate >= 0.05:
                    mode = "mixed-hot"
                elif indexed_rows >= self._adaptive_warm_rows:
                    mode = "streaming"
                else:
                    mode = "warming"
                output[table] = {
                    "mode": mode,
                    "requests": requests,
                    "hits": hits,
                    "hit_rate": hit_rate,
                    "inserted_rows": self._table_inserted.get(table, 0),
                    "indexed_rows": indexed_rows,
                }
            return output


_INDEX_BYTES_PER_ROW = sys.getsizeof(0) + sys.getsizeof((0, 0))


def _tensor_frame_storage_bytes(frame: TensorFrame) -> int:
    storages: dict[tuple[str, int], int] = {}

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            storages[key] = storage.nbytes()
        elif isinstance(value, _MultiTensor):
            visit(value.values)
            visit(value.offset)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(frame.feat_dict)
    if frame.y is not None:
        visit(frame.y)
    return sum(storages.values())
