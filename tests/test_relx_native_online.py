"""Native online workers query source rows after pyg-lib sampling."""

from __future__ import annotations

import importlib.util
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native"), "native extension needed"
)
class NativeOnlinePipelineTest(unittest.TestCase):
    def test_batch_query_chunks_and_restores_sample_order(self) -> None:
        from relx._relx_native import SqliteFeatureReader

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, n REAL)")
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    ((index, float(index)) for index in range(1, 1002)),
                )
            reader = SqliteFeatureReader(str(database))
            reader.add_table(
                "items", "", ["id"], ((index,) for index in range(1, 1002))
            )
            reader.add_feature("items", "n", "numerical")
            ids = np.r_[np.arange(1000, -1, -1), [0, 1000]].astype(np.int64)
            fetched = reader.fetch({"items": ids})["items"]["n"]["values"]
            np.testing.assert_array_equal(fetched.ravel(), ids + 1)

    def test_float_category_uses_python_shortest_representation(self) -> None:
        from relx._relx_native import SqliteFeatureReader

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE items (id INTEGER PRIMARY KEY, ratio REAL)"
                )
                connection.execute("INSERT INTO items VALUES (1, 0.1)")
            reader = SqliteFeatureReader(str(database))
            reader.add_table("items", "", ["id"], ((1,),))
            reader.add_feature(
                "items", "ratio", "categorical", vocab={"float:0.1": 1}
            )
            result = reader.fetch({"items": np.array([0], dtype=np.int64)})
            np.testing.assert_array_equal(result["items"]["ratio"]["values"], [1])

    def test_native_glove_encoding_and_disk_cache(self) -> None:
        from relx._relx_native import SqliteFeatureReader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "features.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, note TEXT)")
                connection.execute(
                    "INSERT INTO items VALUES (1, 'Hello, the world')"
                )
            words = ("PADDING_TOKEN", "hello", "world", "the")
            with (root / "vocab.bin").open("wb") as output:
                output.write(struct.pack("<IIIIII", 0x47584C52, 1, 4, 2, 0, 1))
                for token in (*words, "the"):
                    encoded = token.encode()
                    output.write(struct.pack("<I", len(encoded)))
                    output.write(encoded)
            np.asarray(
                [[0, 0], [2, 4], [4, 8], [99, 99]], dtype=np.float32
            ).tofile(root / "weights.f32")
            cache = root / "text-cache.sqlite"
            reader = SqliteFeatureReader(str(database))
            reader.add_table("items", "", ["id"], ((1,),))
            reader.set_glove_model(str(root), 1024, str(cache))
            reader.add_feature("items", "note", "text_glove")
            first = reader.fetch({"items": np.array([0], dtype=np.int64)})
            values = first["items"]["note"]["values"]
            assert values is not None
            np.testing.assert_array_equal(values, [[3, 6]])
            with sqlite3.connect(cache) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM embeddings").fetchone(),
                    (1,),
                )
            second_reader = SqliteFeatureReader(str(database))
            second_reader.add_table("items", "", ["id"], ((1,),))
            second_reader.set_glove_model(str(root), 0, str(cache))
            second_reader.add_feature("items", "note", "text_glove")
            second = second_reader.fetch({"items": np.array([0], dtype=np.int64)})
            np.testing.assert_array_equal(
                second["items"]["note"]["values"], [[3, 6]]
            )

    def test_workers_read_database_and_return_batches_in_order(self) -> None:
        from relx._relx_native import NativeOnlinePipeline, SqliteFeatureReader

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.sqlite"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE items (id INTEGER PRIMARY KEY, amount REAL, note TEXT);
                    INSERT INTO items VALUES (1, 10, 'first'), (2, 20, 'second');
                    """
                )
            reader = SqliteFeatureReader(str(database))
            reader.add_table("items", "", ["id"], ((1,), (2,)))
            reader.add_feature("items", "amount", "numerical", 15, 5)
            reader.add_feature("items", "note", "text")
            pipeline = NativeOnlinePipeline(
                reader, workers=2, max_inflight=2, max_ready_bytes=1
            )
            try:
                pipeline.submit(0, {"items": np.array([1, 0], dtype=np.int64)})
                pipeline.submit(1, {"items": np.array([0], dtype=np.int64)})
                pipeline.finish()
                first = pipeline.next()
                second = pipeline.next()
                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                assert first is not None and second is not None
                self.assertEqual((first["sequence"], second["sequence"]), (0, 1))
                np.testing.assert_array_equal(
                    first["features"]["items"]["amount"]["values"].ravel(),
                    [1, -1],
                )
                np.testing.assert_array_equal(
                    first["features"]["items"]["note"]["values"],
                    ["second", "first"],
                )
                self.assertIsNone(pipeline.next())
            finally:
                pipeline.close()

            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE items SET amount = 30 WHERE id = 1")
            again = reader.fetch({"items": np.array([0], dtype=np.int64)})
            values = again["items"]["amount"]["values"]
            assert values is not None
            np.testing.assert_array_equal(
                values.ravel(), [3]
            )

    def test_worker_reuses_cached_feature_rows(self) -> None:
        from relx._relx_native import NativeOnlinePipeline, SqliteFeatureReader

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, n REAL)")
                connection.execute("INSERT INTO items VALUES (1, 10), (2, 20)")
            reader = SqliteFeatureReader(str(database))
            reader.add_table("items", "", ["id"], ((1,), (2,)))
            reader.add_feature("items", "n", "numerical")
            reader.set_row_cache_bytes(1024)
            pipeline = NativeOnlinePipeline(reader, workers=1, max_inflight=2)
            try:
                pipeline.submit(0, {"items": np.array([0, 1], dtype=np.int64)})
                first = pipeline.next()
                assert first is not None
                np.testing.assert_array_equal(
                    first["features"]["items"]["n"]["values"].ravel(), [10, 20]
                )
                with sqlite3.connect(database) as connection:
                    connection.execute("UPDATE items SET n = 99 WHERE id = 1")
                pipeline.submit(1, {"items": np.array([0], dtype=np.int64)})
                pipeline.finish()
                second = pipeline.next()
                assert second is not None
                np.testing.assert_array_equal(
                    second["features"]["items"]["n"]["values"].ravel(), [10]
                )
            finally:
                pipeline.close()

    def test_worker_reports_missing_source_row(self) -> None:
        from relx._relx_native import NativeOnlinePipeline, SqliteFeatureReader

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, n REAL)")
            reader = SqliteFeatureReader(str(database))
            reader.add_table("items", "", ["id"], ((1,),))
            reader.add_feature("items", "n", "numerical")
            pipeline = NativeOnlinePipeline(reader, workers=1, max_inflight=1)
            try:
                pipeline.submit(0, {"items": np.array([0], dtype=np.int64)})
                pipeline.finish()
                with self.assertRaisesRegex(RuntimeError, "disappeared"):
                    pipeline.next()
            finally:
                pipeline.close()


if __name__ == "__main__":
    unittest.main()
