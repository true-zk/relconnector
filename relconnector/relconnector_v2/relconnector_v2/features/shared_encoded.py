"""Process-shared owner for encoded TensorFrame chunks."""

from __future__ import annotations

from multiprocessing.managers import BaseManager
from typing import Any, cast

import torch
from torch_frame import TensorFrame

from .cache import EncodedFeatureCache, EncodedFeatureEntry

LookupResult = tuple[
    list[tuple[EncodedFeatureEntry, list[tuple[int, int]]]],
    list[int],
]


class _CacheOwner:
    def __init__(self, max_bytes: int, admission: str) -> None:
        self.cache = EncodedFeatureCache(max_bytes, admission=admission)

    def lookup_many(
        self, table: str, node_ids: torch.Tensor, epoch: int | None
    ) -> LookupResult:
        return self.cache.lookup_many(table, node_ids, epoch=epoch)

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
