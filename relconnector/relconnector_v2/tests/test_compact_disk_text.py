import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from relconnector_v2.features.compact_disk_text import compact, inspect_layout
from relconnector_v2.features.disk_text import SqliteTextEmbeddingCache


class CompactTest(unittest.TestCase):
    def test_preserves_every_key_and_vector_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.sqlite"
            texts = ["x" + str(i) for i in range(35)]
            vectors = torch.arange(140).float().reshape(35, 4)
            old = SqliteTextEmbeddingCache(
                source, namespace="test", embedding_dim=4, write_shard_entries=1
            )
            lookup = old.lookup_or_reserve(texts)
            old.publish(lookup.reservations, {i: v for i, v in enumerate(vectors)})
            old.close()
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertTrue(
                inspect_layout(source, rows_per_shard=8, mapping_budget=8)[
                    "recommended"
                ]
            )
            result = compact(source, root / "new", rows_per_shard=8)
            self.assertFalse(
                inspect_layout(
                    Path(result["destination"]), rows_per_shard=8, mapping_budget=8
                )["recommended"]
            )
            self.assertEqual(result["entries"], 35)
            self.assertEqual(result["source_shards"], 35)
            self.assertEqual(result["destination_shards"], 5)
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), original_hash
            )
            new = SqliteTextEmbeddingCache(
                result["destination"], namespace="test", embedding_dim=4
            )
            try:
                hits = new.lookup(texts).hits
                torch.testing.assert_close(
                    torch.stack([hits[i] for i in range(35)]), vectors, rtol=0, atol=0
                )
                missing = new.lookup(["missing"])
                self.assertEqual(missing.uncached, [0])
            finally:
                new.close()
            with self.assertRaises(FileExistsError):
                compact(source, root / "new")

    def test_rejects_bad_budget_before_creating_output(self):
        with self.assertRaises(ValueError):
            compact(Path("unused"), Path("unused-target"), rows_per_shard=0)

    def test_empty_cache_and_incompatible_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.sqlite"
            cache = SqliteTextEmbeddingCache(source, namespace="test", embedding_dim=4)
            cache.lookup(["missing"])
            cache.close()
            result = compact(source, root / "empty")
            self.assertEqual(result["entries"], 0)
            self.assertEqual(result["destination_shards"], 0)
            self.assertTrue(Path(result["destination"]).with_suffix(".vectors").is_dir())
            self.assertFalse(inspect_layout(source)["recommended"])
            with sqlite3.connect(source) as connection:
                connection.execute("INSERT INTO metadata VALUES ('unknown', 'future')")
            with self.assertRaisesRegex(ValueError, "Unsupported cache metadata"):
                compact(source, root / "incompatible")
            self.assertFalse((root / "incompatible" / source.name).exists())

    def test_corrupt_source_does_not_publish_a_completed_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.sqlite"
            cache = SqliteTextEmbeddingCache(source, namespace="test", embedding_dim=4)
            request = cache.lookup_or_reserve(["x"])
            cache.publish(request.reservations, {0: torch.ones(4)})
            cache.close()
            with sqlite3.connect(source) as connection:
                connection.execute("UPDATE embeddings SET row_index = 999999")
            with self.assertRaises(ValueError):
                compact(source, root / "new")
            self.assertTrue((root / "new" / "INCOMPLETE").exists())
            self.assertFalse((root / "new" / source.name).exists())


if __name__ == "__main__":
    unittest.main()
