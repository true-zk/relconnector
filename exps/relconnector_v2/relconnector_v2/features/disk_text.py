"""Persistent SQLite cache for normalized text embeddings."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO, Literal, TypeVar, cast

import numpy as np
import torch

_SCHEMA_VERSION = 2
_SQLITE_VARIABLE_CHUNK = 500
_SQLITE_WRITE_TRANSACTION_ROWS = 4096
_Method = TypeVar("_Method", bound=Callable[..., Any])


def _locked(method: _Method) -> _Method:
    @wraps(method)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast(_Method, call)


@dataclass(frozen=True)
class DiskTextLookup:
    hits: dict[int, torch.Tensor]
    reservations: dict[int, bytes]
    pending: dict[int, bytes]
    uncached: list[int]


@dataclass
class _WalReadPolicy:
    """Enter read mode only after sustained hits and low shared WAL growth."""

    read_heavy: bool = False
    transitions: int = 0
    _started: float | None = None
    _last_size: int = 0
    _growth: int = 0
    _requests: int = 0
    _hits: int = 0
    _good_windows: int = 0

    def observe(self, now: float, size: int, requests: int, hits: int) -> None:
        if self._started is None:
            self._started = now
            self._last_size = size
            self._requests = requests
            self._hits = hits
            return
        self._growth += max(0, size - self._last_size)
        self._last_size = size
        elapsed = now - self._started
        if elapsed < 30.0:
            return
        count = requests - self._requests
        hit_rate = (hits - self._hits) / max(count, 1)
        growth_rate = self._growth / elapsed
        previous = self.read_heavy
        if count >= 4096 and hit_rate >= 0.99 and growth_rate <= 4 * 1024**2:
            self._good_windows += 1
            if self._good_windows >= 2:
                self.read_heavy = True
        else:
            self._good_windows = 0
            if (count >= 4096 and hit_rate < 0.95) or growth_rate > 8 * 1024**2:
                self.read_heavy = False
        self.transitions += int(previous != self.read_heavy)
        self._started = now
        self._growth = 0
        self._requests = requests
        self._hits = hits


class SqliteTextEmbeddingCache:
    """SQLite hash index over contiguous memory-mapped float32 shards."""

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        embedding_dim: int,
        wait_timeout_s: float = 900.0,
        reservation_ttl_s: float = 900.0,
        max_mapped_shards: int = 1024,
        write_shard_entries: int = 8192,
        wal_checkpoint_bytes: int | None = None,
        block_read_min_fraction: float = 0.0,
    ) -> None:
        if (
            embedding_dim <= 0 or wait_timeout_s <= 0 or reservation_ttl_s <= 0
            or max_mapped_shards <= 0
            or write_shard_entries <= 0
            or (wal_checkpoint_bytes is not None and wal_checkpoint_bytes <= 0)
            or not 0 <= block_read_min_fraction <= 1
        ):
            raise ValueError("disk text cache limits must be positive")
        self.path = Path(path)
        self.namespace = namespace
        self.embedding_dim = embedding_dim
        self.wait_timeout_s = wait_timeout_s
        self.reservation_ttl_s = reservation_ttl_s
        self.max_mapped_shards = max_mapped_shards
        self.write_shard_entries = write_shard_entries
        self.block_read_min_fraction = block_read_min_fraction
        self._block_reads = 0
        self._block_read_bytes = 0
        self.wal_checkpoint_bytes = wal_checkpoint_bytes
        self._wal_target_bytes = wal_checkpoint_bytes or 64 * 1024 * 1024
        self._wal_policy_checked = 0.0
        self._wal_read_policy = _WalReadPolicy()
        self._wal_lookup_checked = 0.0
        self._owner = f"{os.getpid()}:{uuid.uuid4().hex}"
        self._vector_dir = self.path.with_suffix(".vectors")
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._mapped_shards: OrderedDict[str, np.memmap] = OrderedDict()
        self._read_index: dict[bytes, tuple[str, int]] | None = None
        self._read_index_load_s = 0.0
        self._mapping_opens = 0
        self._mapping_evictions = 0
        self._writer: BinaryIO | None = None
        self._writer_shard = ""
        self._writer_rows = 0
        self._checkpoint_attempts = 0
        self._checkpoint_busy = 0
        self._checkpoint_time_s = 0.0
        self._wal_peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._waits = 0

    @_locked
    def prepare_read_index(self) -> None:
        """Keep the completed cold epoch's key-to-block index in this reader."""
        if self._read_index is not None:
            return
        started = time.perf_counter()
        self._read_index = {
            bytes(key): (sys.intern(str(shard)), int(row_index))
            for key, shard, row_index in self._connect().execute(
                "SELECT key, shard, row_index FROM embeddings"
            )
        }
        self._read_index_load_s += time.perf_counter() - started

    def _read_many(
        self, connection: sqlite3.Connection, keys: list[bytes]
    ) -> dict[bytes, tuple[str, int]]:
        if self._read_index is None:
            return self._read_many_sql(connection, keys)
        found = {key: self._read_index[key] for key in keys if key in self._read_index}
        missing = [key for key in keys if key not in found]
        if missing:
            newly_visible = self._read_many_sql(connection, missing)
            self._read_index.update(newly_visible)
            found.update(newly_visible)
        return found

    @staticmethod
    def has_entries(path: str | Path, *, namespace: str) -> bool:
        cache_path = Path(path)
        if not cache_path.is_file():
            return False
        try:
            connection = sqlite3.connect(
                f"file:{cache_path}?mode=ro", uri=True, timeout=5.0
            )
            try:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'namespace'"
                ).fetchone()
                if row is None or row[0] != namespace:
                    return False
                return (
                    connection.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
                    is not None
                )
            finally:
                connection.close()
        except sqlite3.Error:
            return False

    @_locked
    def lookup_or_reserve(self, values: list[str]) -> DiskTextLookup:
        if not values:
            return DiskTextLookup({}, {}, {}, [])
        keys = [_digest(value) for value in values]
        connection = self._connect()
        found = self._load_many(self._read_many(connection, keys))
        hits: dict[int, torch.Tensor] = {}
        missing: list[tuple[int, bytes]] = []
        for position, key in enumerate(keys):
            embedding = found.get(key)
            if embedding is None:
                missing.append((position, key))
            else:
                hits[position] = embedding

        reservations: dict[int, bytes] = {}
        pending: dict[int, bytes] = {}
        if missing:
            now = time.time()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM reservations WHERE expires_at < ?", (now,)
                )
                raced = self._load_many(
                    self._read_many(connection, [key for _, key in missing])
                )
                for position, key in missing:
                    embedding = raced.get(key)
                    if embedding is not None:
                        hits[position] = embedding
                        continue
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO reservations"
                        " (key, owner, expires_at) VALUES (?, ?, ?)",
                        (key, self._owner, now + self.reservation_ttl_s),
                    )
                    if cursor.rowcount == 1:
                        reservations[position] = key
                    else:
                        pending[position] = key
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            self._checkpoint_if_needed(connection)
        self._hits += len(hits)
        self._misses += len(reservations) + len(pending)
        self._waits += len(pending)
        self._checkpoint_if_needed(connection, from_lookup=True)
        return DiskTextLookup(hits, reservations, pending, [])

    @_locked
    def lookup(self, values: list[str]) -> DiskTextLookup:
        if not values:
            return DiskTextLookup({}, {}, {}, [])
        keys = [_digest(value) for value in values]
        found = self._load_many(self._read_many(self._connect(), keys))
        hits: dict[int, torch.Tensor] = {}
        uncached: list[int] = []
        for position, key in enumerate(keys):
            embedding = found.get(key)
            if embedding is None:
                uncached.append(position)
            else:
                hits[position] = embedding
        self._hits += len(hits)
        self._misses += len(uncached)
        self._checkpoint_if_needed(self._connect(), from_lookup=True)
        return DiskTextLookup(hits, {}, {}, uncached)

    @_locked
    def missing_positions(self, values: list[str]) -> list[int]:
        """Recheck keys before a coordinated write without loading vectors."""
        if not values:
            return []
        keys = [_digest(value) for value in values]
        present = self._read_many(self._connect(), keys)
        return [position for position, key in enumerate(keys) if key not in present]

    @_locked
    def publish(
        self,
        reservations: dict[int, bytes],
        embeddings: dict[int, torch.Tensor],
    ) -> None:
        if set(reservations) != set(embeddings):
            raise ValueError("disk cache publications do not match reservations")
        if not reservations:
            return
        ordered = list(reservations.items())
        rows = self._write_shard(
            [key for _, key in ordered],
            torch.stack([embeddings[position] for position, _ in ordered]),
        )
        connection = self._connect()
        before = connection.total_changes
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.executemany(
                "INSERT OR IGNORE INTO embeddings"
                " (key, shard, row_index) VALUES (?, ?, ?)",
                rows,
            )
            connection.executemany(
                "DELETE FROM reservations WHERE key = ? AND owner = ?",
                [(key, self._owner) for key in reservations.values()],
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        self._writes += connection.total_changes - before - len(reservations)
        self._checkpoint_if_needed(connection)

    @_locked
    def put_many(self, values: list[str], embeddings: torch.Tensor) -> None:
        if not values:
            return
        rows = self._write_shard([_digest(value) for value in values], embeddings)
        connection = self._connect()
        # A cold node-block window can publish hundreds of thousands of rows.
        # One transaction holds SQLite's single writer lock long enough for
        # other assembly processes to exhaust their busy timeout.
        deadline = time.monotonic() + 600.0
        for offset in range(0, len(rows), _SQLITE_WRITE_TRANSACTION_ROWS):
            chunk = rows[offset : offset + _SQLITE_WRITE_TRANSACTION_ROWS]
            before = connection.total_changes
            while True:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if (
                        exc.sqlite_errorcode & 0xFF
                        not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                        or time.monotonic() >= deadline
                    ):
                        raise
                    time.sleep(0.05)
            try:
                connection.executemany(
                    "INSERT OR IGNORE INTO embeddings"
                    " (key, shard, row_index) VALUES (?, ?, ?)",
                    chunk,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            self._writes += connection.total_changes - before
            # SQLite does not promise a fair writer queue. Yield after each
            # chunk so waiting assembly processes can acquire the next turn.
            if offset + _SQLITE_WRITE_TRANSACTION_ROWS < len(rows):
                time.sleep(0.01)
        self._checkpoint_if_needed(connection)

    def wait(self, pending: dict[int, bytes]) -> dict[int, torch.Tensor]:
        if not pending:
            return {}
        remaining = dict(pending)
        hits: dict[int, torch.Tensor] = {}
        deadline = time.monotonic() + self.wait_timeout_s
        with self._lock:
            connection = self._connect()
        while remaining:
            with self._lock:
                found = self._load_many(
                    self._read_many(connection, list(remaining.values()))
                )
            complete = [position for position, key in remaining.items() if key in found]
            for position in complete:
                key = remaining.pop(position)
                hits[position] = found[key]
            if not remaining:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for persistent text embeddings")
            time.sleep(0.05)
        with self._lock:
            self._hits += len(hits)
        return hits

    @_locked
    def abort(self, reservations: dict[int, bytes]) -> None:
        if not reservations:
            return
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.executemany(
                "DELETE FROM reservations WHERE key = ? AND owner = ?",
                [(key, self._owner) for key in reservations.values()],
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        self._checkpoint_if_needed(connection)

    @_locked
    def snapshot(self) -> dict[str, object]:
        entries = int(
            self._connect().execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        )
        requests = self._hits + self._misses
        return {
            "path": str(self.path),
            "vector_dir": str(self._vector_dir),
            "entries": entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(requests, 1),
            "writes": self._writes,
            "waits": self._waits,
            "size_bytes": self._size_bytes(),
            "mapped_shards": len(self._mapped_shards),
            "max_mapped_shards": self.max_mapped_shards,
            "mapping_opens": self._mapping_opens,
            "mapping_evictions": self._mapping_evictions,
            "read_index_entries": 0 if self._read_index is None else len(self._read_index),
            "read_index_load_s": self._read_index_load_s,
            "write_shard_entries": self.write_shard_entries,
            "block_reads": self._block_reads,
            "block_read_bytes": self._block_read_bytes,
            "wal_bytes": self._wal_size(),
            "wal_peak_bytes": self._wal_peak_bytes,
            "wal_checkpoint_bytes": self._wal_target_bytes,
            "wal_checkpoint_adaptive": self.wal_checkpoint_bytes is None,
            "wal_read_heavy": self._wal_read_policy.read_heavy,
            "wal_policy_transitions": self._wal_read_policy.transitions,
            "checkpoint_attempts": self._checkpoint_attempts,
            "checkpoint_busy": self._checkpoint_busy,
            "checkpoint_time_s": self._checkpoint_time_s,
        }

    @_locked
    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        for mapping in self._mapped_shards.values():
            mapped_file = getattr(mapping, "_mmap", None)
            if mapped_file is not None:
                mapped_file.close()
        self._mapped_shards.clear()
        self._read_index = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path, timeout=60.0, isolation_level=None, check_same_thread=False
        )
        connection.execute("PRAGMA busy_timeout = 60000")
        deadline = time.monotonic() + 60.0
        try:
            while True:
                try:
                    connection.execute("PRAGMA journal_mode = WAL").fetchone()
                    break
                except sqlite3.OperationalError as exc:
                    # Two first users may concurrently change an empty file's
                    # journal mode. SQLite can return BUSY immediately for this
                    # lock upgrade, without invoking its busy timeout.
                    if (
                        exc.sqlite_errorcode & 0xFF
                        not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                        or time.monotonic() >= deadline
                    ):
                        raise
                    time.sleep(0.05)
        except BaseException:
            connection.close()
            raise
        connection.execute("PRAGMA synchronous = NORMAL")
        # PASSIVE auto-checkpoints can fall behind continuously overlapping
        # writers. At a bounded threshold, TRUNCATE briefly blocks writers and
        # lets existing readers finish before recycling the WAL.
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute(f"PRAGMA journal_size_limit = {self._wal_target_bytes}")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata"
            " (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings"
            " (key BLOB PRIMARY KEY, shard TEXT NOT NULL,"
            " row_index INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS reservations"
            " (key BLOB PRIMARY KEY, owner TEXT NOT NULL,"
            " expires_at REAL NOT NULL) WITHOUT ROWID"
        )
        expected = {
            "schema_version": str(_SCHEMA_VERSION),
            "namespace": self.namespace,
            "embedding_dim": str(self.embedding_dim),
        }
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = dict(connection.execute("SELECT key, value FROM metadata"))
            if existing and existing != expected:
                raise ValueError(
                    f"incompatible disk text cache metadata at {self.path}"
                )
            connection.executemany(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                expected.items(),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            connection.close()
            raise
        self._connection = connection
        self._checkpoint_if_needed(connection)
        return connection

    def _wal_size(self) -> int:
        try:
            return Path(str(self.path) + "-wal").stat().st_size
        except FileNotFoundError:
            return 0

    def _checkpoint_if_needed(
        self, connection: sqlite3.Connection, *, from_lookup: bool = False
    ) -> None:
        now = time.monotonic()
        if from_lookup:
            # Even a fully warm reader must be able to recycle an old WAL.
            # Avoid a filesystem stat for every small embedding request.
            if now - self._wal_lookup_checked < 1.0:
                return
            self._wal_lookup_checked = now
        size = self._wal_size()
        self._wal_peak_bytes = max(self._wal_peak_bytes, size)
        if self.wal_checkpoint_bytes is None and now - self._wal_policy_checked >= 1.0:
            # A tiny WAL against a large hash index repeatedly flushes sparse
            # dirty pages, which is expensive on rotating disks. Accumulate a
            # larger batch as the index grows, while reserving most free space.
            database_bytes = self.path.stat().st_size
            free_bytes = shutil.disk_usage(self.path.parent).free
            self._wal_read_policy.observe(
                now, size, self._hits + self._misses, self._hits
            )
            self._wal_target_bytes = _wal_checkpoint_target(
                database_bytes, free_bytes,
                read_heavy=self._wal_read_policy.read_heavy,
            )
            self._wal_policy_checked = now
        if size < self._wal_target_bytes:
            return
        started = time.perf_counter()
        self._checkpoint_attempts += 1
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        self._checkpoint_time_s += time.perf_counter() - started
        self._checkpoint_busy += int(result[0] != 0)

    @staticmethod
    def _read_many_sql(
        connection: sqlite3.Connection, keys: list[bytes]
    ) -> dict[bytes, tuple[str, int]]:
        found: dict[bytes, tuple[str, int]] = {}
        for offset in range(0, len(keys), _SQLITE_VARIABLE_CHUNK):
            chunk = keys[offset : offset + _SQLITE_VARIABLE_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                "SELECT key, shard, row_index FROM embeddings"
                f" WHERE key IN ({placeholders})",
                chunk,
            )
            found.update(
                (bytes(key), (str(shard), int(row_index)))
                for key, shard, row_index in rows
            )
        return found

    def _load_many(
        self, locations: dict[bytes, tuple[str, int]]
    ) -> dict[bytes, torch.Tensor]:
        grouped: dict[str, list[tuple[bytes, int]]] = {}
        for key, (shard, row_index) in locations.items():
            grouped.setdefault(shard, []).append((key, row_index))
        output: dict[bytes, torch.Tensor] = {}
        for shard, rows in grouped.items():
            mapping = self._mapped_shards.get(shard)
            if mapping is None:
                # A long run can touch more shards than vm.max_map_count. Close
                # the least recently used mapping before opening another one.
                # Returned tensors own copies, so no caller depends on the map.
                if len(self._mapped_shards) >= self.max_mapped_shards:
                    _, evicted = self._mapped_shards.popitem(last=False)
                    mapped_file = getattr(evicted, "_mmap", None)
                    if mapped_file is not None:
                        mapped_file.close()
                    self._mapping_evictions += 1
                mapping = np.memmap(
                    self._vector_dir / shard, dtype=np.float32, mode="r"
                )
                if mapping.size % self.embedding_dim:
                    mapped_file = getattr(mapping, "_mmap", None)
                    if mapped_file is not None:
                        mapped_file.close()
                    raise ValueError(f"corrupt text embedding shard {shard}")
                self._mapped_shards[shard] = mapping
                self._mapping_opens += 1
            else:
                self._mapped_shards.move_to_end(shard)
            row_count = mapping.size // self.embedding_dim
            row_indices = np.fromiter(
                (row_index for _, row_index in rows),
                dtype=np.int64,
                count=len(rows),
            )
            if np.any(row_indices < 0) or np.any(row_indices >= row_count):
                raise ValueError(f"invalid row in text embedding shard {shard}")
            matrix = mapping.reshape(row_count, self.embedding_dim)
            if (
                self.block_read_min_fraction > 0
                and len(row_indices) >= row_count * self.block_read_min_fraction
            ):
                # One sequential read serves every requested row in this shard.
                block = np.array(matrix, dtype=np.float32, copy=True)
                values = torch.from_numpy(block[row_indices].copy())
                self._block_reads += 1
                self._block_read_bytes += int(block.nbytes)
            else:
                values = torch.from_numpy(
                    np.array(matrix[row_indices], dtype=np.float32, copy=True)
                )
            for (key, _), value in zip(rows, values, strict=True):
                output[key] = value
        return output

    def _write_shard(
        self,
        keys: list[bytes],
        embeddings: torch.Tensor,
    ) -> list[tuple[bytes, str, int]]:
        values = embeddings.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if values.shape != (len(keys), self.embedding_dim):
            raise ValueError("disk text embedding batch has invalid shape")
        self._vector_dir.mkdir(parents=True, exist_ok=True)
        rows: list[tuple[bytes, str, int]] = []
        offset = 0
        while offset < len(keys):
            if self._writer is None or self._writer_rows == self.write_shard_entries:
                if self._writer is not None:
                    self._writer.close()
                self._writer_shard = f"{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex}.f32"
                self._writer = (self._vector_dir / self._writer_shard).open("xb")
                # Fixed size lets an existing reader mapping see newly
                # published rows without remapping. Unpublished rows are never
                # exposed by the SQLite index; each writer owns its own shard.
                self._writer.truncate(self.write_shard_entries * self.embedding_dim * 4)
                self._writer_rows = 0
            count = min(len(keys) - offset, self.write_shard_entries - self._writer_rows)
            self._writer.seek(self._writer_rows * self.embedding_dim * 4)
            values[offset : offset + count].numpy().tofile(self._writer)
            self._writer.flush()
            rows.extend(
                (key, self._writer_shard, self._writer_rows + row)
                for row, key in enumerate(keys[offset : offset + count])
            )
            self._writer_rows += count
            offset += count
        return rows

    def _size_bytes(self) -> int:
        total = self.path.stat().st_size if self.path.exists() else 0
        if self._vector_dir.exists():
            total += sum(
                item.stat().st_size
                for item in self._vector_dir.iterdir()
                if item.is_file()
            )
        return total

    def projected_shard_bytes(self, new_rows: int) -> int:
        """Estimate new allocated shard bytes before appending a batch."""
        if new_rows <= 0:
            return 0
        free_rows = (
            self.write_shard_entries - self._writer_rows
            if self._writer is not None
            else 0
        )
        remaining = max(new_rows - free_rows, 0)
        shards = (remaining + self.write_shard_entries - 1) // self.write_shard_entries
        return shards * self.write_shard_entries * self.embedding_dim * 4


class AdaptiveDiskTextEmbeddingCache:
    """Activate persistent caching only when observed encoding justifies it."""

    def __init__(
        self,
        *,
        path: str | Path | None,
        namespace: str,
        embedding_dim: int,
        mode: Literal["off", "on", "adaptive"],
        database_bytes: int,
        min_database_bytes: int,
        min_model_inputs: int,
        max_memory_hit_rate: float,
        min_model_time_s: float,
        write_batch_entries: int = 8192,
    ) -> None:
        if mode not in {"off", "on", "adaptive"}:
            raise ValueError("disk text cache mode must be off, on or adaptive")
        if (
            database_bytes < 0
            or min_database_bytes < 0
            or min_model_inputs <= 0
            or not 0 <= max_memory_hit_rate <= 1
            or min_model_time_s < 0
            or write_batch_entries <= 0
        ):
            raise ValueError("invalid adaptive disk text cache configuration")
        self.path = None if path is None else Path(path)
        self.namespace = namespace
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.database_bytes = database_bytes
        self.min_database_bytes = min_database_bytes
        self.min_model_inputs = min_model_inputs
        self.max_memory_hit_rate = max_memory_hit_rate
        self.min_model_time_s = min_model_time_s
        self.write_batch_entries = write_batch_entries
        self._cache: SqliteTextEmbeddingCache | None = None
        self._lock = threading.RLock()
        self._observed_model_inputs = 0
        self._observed_model_time_s = 0.0
        self._activation_reason: str | None = None
        self._warmup: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._write_buffer: OrderedDict[str, torch.Tensor] = OrderedDict()

        if self.path is not None and mode == "on":
            self._activate("configured")
        elif (
            self.path is not None
            and mode == "adaptive"
            and self.database_bytes >= self.min_database_bytes
            and SqliteTextEmbeddingCache.has_entries(
                self.path, namespace=self.namespace
            )
        ):
            self._activate("existing-cache")

    @property
    def active(self) -> bool:
        return self._cache is not None

    @_locked
    def lookup_or_reserve(
        self,
        values: list[str],
        *,
        coordinated: bool = True,
    ) -> DiskTextLookup | None:
        self._attach_existing()
        if self._cache is None:
            return None
        hits: dict[int, torch.Tensor] = {}
        remaining_positions: list[int] = []
        remaining_values: list[str] = []
        for position, value in enumerate(values):
            buffered = self._write_buffer.get(value)
            if buffered is None:
                remaining_positions.append(position)
                remaining_values.append(value)
            else:
                hits[position] = buffered.clone()
        lookup = (
            self._cache.lookup_or_reserve(remaining_values)
            if coordinated
            else self._cache.lookup(remaining_values)
        )
        for position, embedding in lookup.hits.items():
            hits[remaining_positions[position]] = embedding
        return DiskTextLookup(
            hits=hits,
            reservations={
                remaining_positions[position]: key
                for position, key in lookup.reservations.items()
            },
            pending={
                remaining_positions[position]: key
                for position, key in lookup.pending.items()
            },
            uncached=[remaining_positions[position] for position in lookup.uncached],
        )

    @_locked
    def publish(
        self,
        reservations: dict[int, bytes],
        embeddings: dict[int, torch.Tensor],
    ) -> None:
        if self._cache is None:
            raise RuntimeError("disk text cache is not active")
        self._cache.publish(reservations, embeddings)

    def wait(self, pending: dict[int, bytes]) -> dict[int, torch.Tensor]:
        if self._cache is None:
            raise RuntimeError("disk text cache is not active")
        return self._cache.wait(pending)

    @_locked
    def abort(self, reservations: dict[int, bytes]) -> None:
        if self._cache is not None:
            self._cache.abort(reservations)

    @_locked
    def observe_encoded(
        self,
        values: list[str],
        embeddings: torch.Tensor,
        *,
        model_time_s: float,
        memory_hit_rate: float,
    ) -> None:
        if not values:
            return
        self._attach_existing()
        if self._cache is not None:
            self._buffer_many(values, embeddings)
            return
        if (
            self.mode != "adaptive"
            or self.path is None
            or self.database_bytes < self.min_database_bytes
        ):
            return
        self._observed_model_inputs += len(values)
        self._observed_model_time_s += model_time_s
        if (
            self._observed_model_inputs >= self.min_model_inputs
            and self._observed_model_time_s >= self.min_model_time_s
            and memory_hit_rate <= self.max_memory_hit_rate
        ):
            self._activate("large-low-reuse-encoding-bound")
            if self._warmup:
                self._buffer_many(
                    list(self._warmup), torch.stack(list(self._warmup.values()))
                )
                self._warmup.clear()
            self._buffer_many(values, embeddings)
            return
        for value, embedding in zip(values, embeddings, strict=True):
            self._warmup[value] = embedding.detach().cpu().clone()
            while len(self._warmup) > max(
                self.min_model_inputs, self.write_batch_entries
            ):
                self._warmup.popitem(last=False)

    @_locked
    def snapshot(self) -> dict[str, object]:
        self.flush()
        base: dict[str, object] = {
            "mode": self.mode,
            "active": self.active,
            "activation_reason": self._activation_reason,
            "database_bytes": self.database_bytes,
            "min_database_bytes": self.min_database_bytes,
            "observed_model_inputs": self._observed_model_inputs,
            "observed_model_time_s": self._observed_model_time_s,
            "warmup_entries": len(self._warmup),
            "path": None if self.path is None else str(self.path),
        }
        if self._cache is not None:
            base.update(self._cache.snapshot())
        return base

    @_locked
    def close(self) -> None:
        self.flush()
        if self._cache is not None:
            self._cache.close()

    @_locked
    def flush(self) -> None:
        if self._cache is None or not self._write_buffer:
            return
        values = list(self._write_buffer)
        embeddings = torch.stack(list(self._write_buffer.values()))
        self._cache.put_many(values, embeddings)
        self._write_buffer.clear()

    def _buffer_many(self, values: list[str], embeddings: torch.Tensor) -> None:
        for value, embedding in zip(values, embeddings, strict=True):
            self._write_buffer[value] = embedding.detach().cpu().clone()
        if len(self._write_buffer) >= self.write_batch_entries:
            self.flush()

    def _activate(self, reason: str) -> None:
        if self.path is None or self._cache is not None:
            return
        self._cache = SqliteTextEmbeddingCache(
            self.path,
            namespace=self.namespace,
            embedding_dim=self.embedding_dim,
        )
        self._activation_reason = reason

    def _attach_existing(self) -> None:
        # Another assembly worker may have activated the cache after this
        # worker started. Attach before the next lookup so later windows and
        # epochs use the embeddings that worker already persisted.
        if (
            self._cache is None
            and self.path is not None
            and self.mode == "adaptive"
            and self.database_bytes >= self.min_database_bytes
            and SqliteTextEmbeddingCache.has_entries(
                self.path, namespace=self.namespace
            )
        ):
            self._activate("existing-cache")
            if self._warmup:
                self._buffer_many(
                    list(self._warmup), torch.stack(list(self._warmup.values()))
                )
                self._warmup.clear()


def text_embedding_cache_location(
    cache_dir: str | Path,
    *,
    database_path: str | Path,
    database_version: str,
    model_identity: str,
    execution: str,
    embedding_dim: int,
) -> tuple[Path, str]:
    payload = "\0".join(
        (
            str(_SCHEMA_VERSION),
            database_version,
            model_identity,
            execution,
            str(embedding_dim),
        )
    )
    namespace = hashlib.sha256(payload.encode()).hexdigest()
    stem = Path(database_path).stem
    path = Path(cache_dir) / "text_embeddings" / f"{stem}-{namespace[:16]}.sqlite"
    return path, namespace


def text_model_identity(model_path: str | Path | None, *, default: str) -> str:
    if model_path is None:
        return default
    path = Path(model_path)
    if not path.exists():
        return str(path)
    if path.is_file():
        stat = path.stat()
        return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    newest_mtime = max(
        (item.stat().st_mtime_ns for item in path.rglob("*") if item.is_file()),
        default=path.stat().st_mtime_ns,
    )
    return f"{path.resolve()}:{newest_mtime}"


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _wal_checkpoint_target(
    database_bytes: int, free_bytes: int, *, read_heavy: bool = False
) -> int:
    minimum = 64 * 1024 * 1024
    maximum = (512 if read_heavy else 4096) * 1024 * 1024
    desired = max(minimum, database_bytes * 4)
    return max(1024 * 1024, min(desired, maximum, free_bytes // 16))
