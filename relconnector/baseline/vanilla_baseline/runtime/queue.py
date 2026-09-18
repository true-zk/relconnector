"""Thread-safe queues bounded by retained payload bytes."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


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
            while (
                len(self._items) >= self.max_items
                or self._bytes + size > self.max_bytes
            ):
                if self._closed:
                    raise RuntimeError("queue is closed")
                self._condition.wait()
            if self._closed:
                raise RuntimeError("queue is closed")
            self._items.append((item, size))
            self._bytes += size
            self._condition.notify_all()

    def get(self) -> T | None:
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            if not self._items:
                return None
            item, size = self._items.popleft()
            self._bytes -= size
            self._condition.notify_all()
            return item

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
