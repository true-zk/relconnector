"""Thread-safe queues bounded by retained payload bytes."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class QueueStats:
    put_wait_s: float
    get_wait_s: float
    put_wait_count: int
    get_wait_count: int
    peak_items: int
    peak_bytes: int


class ByteBoundedQueue(Generic[T]):
    """Bound queued payload bytes and item count; reject oversized payloads."""

    def __init__(
        self, max_bytes: int, size_of: Callable[[T], int], *, max_items: int = 64
    ) -> None:
        if max_bytes <= 0 or max_items <= 0:
            raise ValueError("queue limits must be greater than zero")
        self.max_bytes = max_bytes
        self.max_items = max_items
        self.size_of = size_of
        self._items: deque[tuple[T, int]] = deque()
        self._bytes = 0
        self._closed = False
        self._condition = threading.Condition()
        self._put_wait_s = 0.0
        self._get_wait_s = 0.0
        self._put_wait_count = 0
        self._get_wait_count = 0
        self._peak_items = 0
        self._peak_bytes = 0

    def put(self, item: T) -> None:
        size = self.size_of(item)
        if size < 0:
            raise ValueError("payload size must be nonnegative")
        if size > self.max_bytes:
            raise ValueError(
                f"Payload requires {size} bytes, queue budget is {self.max_bytes}; "
                "reduce batch size/fanout or increase the queue budget"
            )
        with self._condition:
            wait_started: float | None = None
            while (
                len(self._items) >= self.max_items
                or self._bytes + size > self.max_bytes
            ):
                if self._closed:
                    raise RuntimeError("queue is closed")
                if wait_started is None:
                    wait_started = time.perf_counter()
                self._condition.wait()
            if wait_started is not None:
                self._put_wait_s += time.perf_counter() - wait_started
                self._put_wait_count += 1
            if self._closed:
                raise RuntimeError("queue is closed")
            self._items.append((item, size))
            self._bytes += size
            self._condition.notify_all()
            self._peak_items = max(self._peak_items, len(self._items))
            self._peak_bytes = max(self._peak_bytes, self._bytes)

    def get(self) -> T | None:
        with self._condition:
            wait_started: float | None = None
            while not self._items and not self._closed:
                if wait_started is None:
                    wait_started = time.perf_counter()
                self._condition.wait()
            if wait_started is not None:
                self._get_wait_s += time.perf_counter() - wait_started
                self._get_wait_count += 1
            if not self._items:
                return None
            item, size = self._items.popleft()
            self._bytes -= size
            self._condition.notify_all()
            return item

    def drain(
        self,
        max_items: int,
        *,
        max_bytes: int | None = None,
        predicate: Callable[[T], bool] | None = None,
    ) -> list[T]:
        if max_items < 0 or (max_bytes is not None and max_bytes < 0):
            raise ValueError("drain limits cannot be negative")
        drained: list[T] = []
        drained_bytes = 0
        with self._condition:
            while self._items and len(drained) < max_items:
                item, size = self._items[0]
                if predicate is not None and not predicate(item):
                    break
                if max_bytes is not None and drained_bytes + size > max_bytes:
                    break
                self._items.popleft()
                self._bytes -= size
                drained_bytes += size
                drained.append(item)
            if drained:
                self._condition.notify_all()
        return drained

    def close(self, *, discard: bool = False) -> None:
        with self._condition:
            self._closed = True
            if discard:
                self._items.clear()
                self._bytes = 0
            self._condition.notify_all()

    @property
    def queued_bytes(self) -> int:
        with self._condition:
            return self._bytes

    @property
    def stats(self) -> QueueStats:
        with self._condition:
            return QueueStats(
                put_wait_s=self._put_wait_s,
                get_wait_s=self._get_wait_s,
                put_wait_count=self._put_wait_count,
                get_wait_count=self._get_wait_count,
                peak_items=self._peak_items,
                peak_bytes=self._peak_bytes,
            )
