"""Optional node-ID text-vector cache with batched, block-aware reads."""

from __future__ import annotations

import hashlib
import fcntl
from pathlib import Path

import torch

from .disk_text import SqliteTextEmbeddingCache


def node_block_cache_location(
    cache_dir: Path, *, namespace: str, schema_fingerprint: str
) -> tuple[Path, str]:
    identity = f"node-block-v1:{namespace}:{schema_fingerprint}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return cache_dir / "text-node-blocks" / f"{digest}.sqlite", identity


class NodeBlockTextCache:
    """Store encoded vectors by logical table/column/node ID, not text content.

    Immutable vector shards are appended by each worker. The SQLite index makes
    rows visible only after their vectors are written, and the read path groups
    all requested rows by shard before choosing full-block or precise reads.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        embedding_dim: int,
        rows_per_block: int,
        max_bytes: int,
        max_mapped_shards: int = 64,
    ) -> None:
        if rows_per_block <= 0 or max_bytes < 0:
            raise ValueError("invalid node-block text cache limits")
        self.rows_per_block = rows_per_block
        self.max_bytes = max_bytes
        self.store = SqliteTextEmbeddingCache(
            path,
            namespace=namespace,
            embedding_dim=embedding_dim,
            write_shard_entries=rows_per_block,
            block_read_min_fraction=0.25,
            max_mapped_shards=max_mapped_shards,
        )
        self.admission_rejections = 0
        self._writer_lock_path = self.store.path.with_suffix(".writer.lock")

    @staticmethod
    def key(table: str, column: str, node_id: int) -> str:
        # Length prefixes keep arbitrary SQL identifiers unambiguous.
        return f"{len(table)}:{table}{len(column)}:{column}{node_id}"

    def lookup(self, keys: list[str]) -> dict[int, torch.Tensor]:
        return self.store.lookup(keys).hits

    def prepare_read_index(self) -> None:
        self.store.prepare_read_index()

    def put_many(self, keys: list[str], vectors: torch.Tensor) -> None:
        if not keys:
            return
        # Several workers can miss and encode the same node simultaneously.
        # Coordinate the final recheck and write so losing workers do not
        # create vector shards that SQLite later discards as duplicates.
        self._writer_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._writer_lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            missing = self.store.missing_positions(keys)
            if not missing:
                return
            new_keys = [keys[position] for position in missing]
            projected = self.store.projected_shard_bytes(len(new_keys))
            if self.max_bytes == 0 or self.store._size_bytes() + projected > self.max_bytes:
                self.admission_rejections += len(new_keys)
                return
            indices = torch.tensor(missing, dtype=torch.long)
            self.store.put_many(new_keys, vectors.index_select(0, indices))

    def snapshot(self) -> dict[str, object]:
        return {
            "layout": "node_block",
            "rows_per_block": self.rows_per_block,
            "max_bytes": self.max_bytes,
            "admission_rejections": self.admission_rejections,
            **self.store.snapshot(),
        }

    def close(self) -> None:
        self.store.close()
