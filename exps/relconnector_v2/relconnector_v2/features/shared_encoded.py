"""Process-shared owner for encoded TensorFrame chunks."""

from __future__ import annotations

from multiprocessing.managers import BaseManager
from typing import Any, cast

import torch
from torch_frame import TensorFrame, cat

from .cache import EncodedFeatureCache, EncodedFeatureEntry
from .rows import gather_rows

LookupResult = tuple[
    list[tuple[EncodedFeatureEntry, list[tuple[int, int]]]],
    list[int],
]


class _CacheOwner:
    def __init__(self, max_bytes: int, admission: str) -> None:
        self.cache = EncodedFeatureCache(
            max_bytes, admission=admission, share_memory=True
        )

    def lookup_many(
        self, table: str, node_ids: torch.Tensor, epoch: int | None
    ) -> LookupResult:
        groups, misses = self.cache.lookup_many(table, node_ids, epoch=epoch)
        if not groups:
            return [], misses

        # Sending entire cached chunks through the manager makes a sparse hit
        # increasingly expensive as the cache grows. Return only requested rows.
        pieces: list[TensorFrame] = []
        positions: list[int] = []
        for entry, locations in groups:
            rows = torch.tensor([row for _, row in locations], dtype=torch.long)
            pieces.append(gather_rows(entry.frame, rows))
            positions.extend(position for position, _ in locations)
        projected = pieces[0] if len(pieces) == 1 else cat(pieces, dim=0)
        projected_ids = node_ids[torch.tensor(positions, dtype=torch.long)]
        projected_entry = EncodedFeatureEntry(
            -1, table, projected_ids, projected, 0
        )
        return [
            (projected_entry, [(position, row) for row, position in enumerate(positions)])
        ], misses

    def put(
        self,
        table: str,
        node_ids: torch.Tensor,
        frame: TensorFrame,
        epoch: int | None,
    ) -> EncodedFeatureEntry | None:
        return self.cache.put(table, node_ids, frame, epoch=epoch)

    def snapshot(self) -> dict[str, object]:
        return {
            "hit_rows": self.cache.hit_rows,
            "miss_rows": self.cache.miss_rows,
            "hit_rate": self.cache.hit_rate,
            "inserted_rows": self.cache.inserted_rows,
            "evictions": self.cache.evictions,
            "entries": self.cache.entry_count,
            "current_bytes": self.cache.current_bytes,
            "peak_bytes": self.cache.peak_bytes,
            "epoch_stats": self.cache.epoch_stats,
            "table_stats": self.cache.table_stats,
        }


class _CacheManager(BaseManager):
    pass


_CacheManager.register(
    "encoded_cache",
    _CacheOwner,
    exposed=("lookup_many", "put", "snapshot"),
)


class SharedEncodedFeatureCache:
    """Batch-RPC client facade used by assembly workers."""

    def __init__(self, proxy: Any) -> None:
        self._proxy = proxy

    def lookup_many(
        self,
        table: str,
        node_ids: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> LookupResult:
        return cast(LookupResult, self._proxy.lookup_many(table, node_ids, epoch))

    def put(
        self,
        table: str,
        node_ids: torch.Tensor,
        frame: TensorFrame,
        *,
        epoch: int | None = None,
    ) -> EncodedFeatureEntry | None:
        return cast(
            EncodedFeatureEntry | None,
            self._proxy.put(table, node_ids, frame, epoch),
        )

    def snapshot(self) -> dict[str, object]:
        return cast(dict[str, object], self._proxy.snapshot())


class SharedEncodedCacheResource:
    """Parent-only manager lifetime handle."""

    def __init__(self, manager: BaseManager) -> None:
        self._manager = manager

    def close(self) -> None:
        self._manager.shutdown()


def create_shared_encoded_cache(
    context: Any,
    *,
    max_bytes: int,
    admission: str,
) -> tuple[SharedEncodedFeatureCache, SharedEncodedCacheResource]:
    manager = _CacheManager(ctx=context)
    manager.start()
    proxy = cast(Any, manager).encoded_cache(max_bytes, admission)
    return SharedEncodedFeatureCache(proxy), SharedEncodedCacheResource(manager)
