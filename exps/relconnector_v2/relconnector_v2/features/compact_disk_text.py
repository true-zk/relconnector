"""Build an independent compact snapshot; never rewrite the source cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import torch

from relconnector_v2.features.disk_text import SqliteTextEmbeddingCache


def compact(source: Path, destination: Path, *, rows_per_shard: int = 8192) -> dict:
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    marker = destination / "INCOMPLETE"
    marker.write_text("Compaction in progress; do not use this directory.\n")
    started = time.perf_counter()
    connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    source_cache = output_cache = None
    try:
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema_version") != "2":
            raise ValueError("Only version 2 immutable vector caches are supported")
        namespace = metadata["namespace"]
        dim = int(metadata["embedding_dim"])
        if metadata != {
            "schema_version": "2",
            "namespace": namespace,
            "embedding_dim": str(dim),
        }:
            raise ValueError("Unsupported cache metadata")
        expected = connection.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        source_cache = SqliteTextEmbeddingCache(
            source, namespace=namespace, embedding_dim=dim, max_mapped_shards=8
        )
        temporary = destination / "_building.sqlite"
        output_cache = SqliteTextEmbeddingCache(
            temporary,
            namespace=namespace,
            embedding_dim=dim,
            write_shard_entries=rows_per_shard,
            max_mapped_shards=2,
        )
        output_db = output_cache._connect()
        output_db.execute("PRAGMA cache_size = -65536")
        output_db.execute(
            "CREATE TABLE _compaction_rows (key BLOB, shard TEXT, row_index INTEGER)"
        )
        # Sorting once makes each old small file a sequential compaction input.
        cursor = connection.execute(
            "SELECT key,shard,row_index FROM embeddings ORDER BY shard,row_index"
        )
        copied = 0
        source_shards = set()
        destination_shards = set()
        digest = hashlib.sha256()
        while True:
            rows = cursor.fetchmany(rows_per_shard)
            if not rows:
                break
            locations = {bytes(key): (str(shard), int(row)) for key, shard, row in rows}
            if any(Path(shard).name != shard for shard, _ in locations.values()):
                raise ValueError("Invalid source shard path")
            source_shards.update(shard for shard, _ in locations.values())
            loaded = source_cache._load_many(locations)
            keys = list(locations)
            values = torch.stack([loaded[key] for key in keys])
            entries = output_cache._write_shard(keys, values)
            # Verify every vector from its new physical location before indexing.
            recovered = output_cache._load_many(
                {key: (shard, row) for key, shard, row in entries}
            )
            restored = torch.stack([recovered[key] for key in keys])
            raw = values.numpy().tobytes()
            if raw != restored.numpy().tobytes():
                raise RuntimeError("Vector bytes changed during compaction")
            digest.update(b"".join(keys))
            digest.update(raw)
            output_db.execute("BEGIN IMMEDIATE")
            try:
                output_db.executemany(
                    "INSERT INTO _compaction_rows(key,shard,row_index) VALUES (?,?,?)",
                    entries,
                )
                output_db.commit()
            except BaseException:
                output_db.rollback()
                raise
            output_cache._checkpoint_if_needed(output_db)
            destination_shards.update(shard for _, shard, _ in entries)
            copied += len(entries)
            if copied % (rows_per_shard * 32) == 0:
                print(
                    json.dumps(
                        {
                            "copied": copied,
                            "total": expected,
                            "elapsed_s": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        # Bulk-build the hash index in key order instead of millions of random
        # B-tree updates while copying vectors in source-file order.
        output_db.execute("BEGIN IMMEDIATE")
        output_db.execute(
            "INSERT INTO embeddings SELECT key,shard,row_index FROM _compaction_rows ORDER BY key"
        )
        output_db.execute("DROP TABLE _compaction_rows")
        output_db.commit()
        output_db.execute("VACUUM")
        if (
            copied != expected
            or output_db.execute("SELECT count(*) FROM embeddings").fetchone()[0]
            != expected
        ):
            raise RuntimeError("Compacted entry count mismatch")
        if output_db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Invalid destination index")
        output_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        output_cache.close()
        output_cache = None
        final = destination / source.name
        # An empty cache has no writer and therefore no vector directory yet.
        temporary.with_suffix(".vectors").mkdir(exist_ok=True)
        temporary.with_suffix(".vectors").rename(final.with_suffix(".vectors"))
        temporary.rename(final)
        result = {
            "source": str(source),
            "destination": str(final),
            "entries": copied,
            "source_shards": len(source_shards),
            "destination_shards": len(destination_shards),
            "rows_per_shard": rows_per_shard,
            "all_vectors_bitwise_verified": True,
            "copy_order_sha256": digest.hexdigest(),
            "duration_s": time.perf_counter() - started,
            "metadata": metadata,
        }
        (destination / "compaction.json").write_text(json.dumps(result, indent=2))
        marker.unlink()
        return result
    finally:
        if source_cache is not None:
            source_cache.close()
        if output_cache is not None:
            output_cache.close()
        connection.close()


def inspect_layout(
    source: Path, *, rows_per_shard: int = 8192, mapping_budget: int = 1024
) -> dict:
    """Screen for fragmentation whose compact layout fits the mapping budget.

    This is a structural heuristic, not a promise of training speedup. Measure
    lookup time and the critical path before promoting a new cache layout.
    """
    if rows_per_shard <= 0 or mapping_budget <= 0:
        raise ValueError("layout limits must be positive")
    connection = sqlite3.connect(
        source.resolve(strict=True).as_uri() + "?mode=ro", uri=True
    )
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema_version") != "2":
            raise ValueError("Only version 2 vector caches are supported")
        entries, shards = connection.execute(
            "SELECT count(*),count(DISTINCT shard) FROM embeddings"
        ).fetchone()
    finally:
        connection.close()
    target = (entries + rows_per_shard - 1) // rows_per_shard
    ratio = shards / max(target, 1)
    return {
        "entries": entries,
        "source_shards": shards,
        "target_shards": target,
        "mapping_budget": mapping_budget,
        "shard_reduction_factor": ratio,
        "recommended": shards > mapping_budget
        and target <= mapping_budget
        and ratio >= 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--rows-per-shard", type=int, default=8192)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Only inspect the layout; do not write files",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.plan:
        print(
            json.dumps(
                inspect_layout(args.source, rows_per_shard=args.rows_per_shard),
                indent=2,
            )
        )
        return
    if args.destination is None:
        parser.error("destination is required unless --plan is used")
    print(
        json.dumps(
            compact(args.source, args.destination, rows_per_shard=args.rows_per_shard),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
