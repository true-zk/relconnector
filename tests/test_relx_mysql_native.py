"""Native MySQL integration against a dedicated writable test database."""

from __future__ import annotations

import importlib.util
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@unittest.skipUnless(
    os.getenv("RELX_TEST_MYSQL_NATIVE_URL")
    and importlib.util.find_spec("relx._relx_native"),
    "set RELX_TEST_MYSQL_NATIVE_URL to a dedicated MySQL test database",
)
class NativeMySqlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from relx import _relx_native

        if not hasattr(_relx_native, "MySqlFeatureReader"):
            raise unittest.SkipTest("native extension has no MySQL support")
        cls.url = os.environ["RELX_TEST_MYSQL_NATIVE_URL"]
        cls.engine = create_engine(cls.url)
        with cls.engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS relx_native_labels"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_events"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_users"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_bulk"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_items"))
            connection.execute(text("""
                CREATE TABLE relx_native_items (
                    region VARCHAR(20) NOT NULL, id BIGINT NOT NULL,
                    amount DOUBLE, segment TEXT, tags TEXT, vector TEXT,
                    note TEXT, created_at DATETIME(6), raw_value BLOB,
                    PRIMARY KEY (region, id)
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
            """))
            connection.execute(text("""
                INSERT INTO relx_native_items VALUES
                  ('eu', 1, 10, 'A', '["x", "y"]', '[1, 2]',
                   'Hello, world', '2020-01-01 00:00:00.123456', X'0001'),
                  ('us', 1, NULL, NULL, '[]', NULL, NULL, NULL, NULL),
                  ('us', 2, 20, 'B', '["y", "é"]', '[3, 4]',
                   'World!', '2020-01-03 00:00:00', X'FF')
            """))
            connection.execute(
                text("""
                    UPDATE relx_native_items SET tags = :tags
                    WHERE region = 'us' AND id = 2
                """),
                {"tags": '["y", "\\u00e9"]'},
            )
            connection.execute(text("""
                CREATE TABLE relx_native_bulk (
                    id BIGINT PRIMARY KEY, amount DOUBLE, note LONGTEXT,
                    price DECIMAL(24, 6), ratio DOUBLE
                )
            """))
            connection.execute(
                text("INSERT INTO relx_native_bulk (id, amount) VALUES (:id, :amount)"),
                [{"id": index, "amount": float(index)} for index in range(1, 131)],
            )
            connection.execute(
                text("UPDATE relx_native_bulk SET note = :note WHERE id = 130"),
                {"note": "α" * 5000},
            )
            connection.execute(text("""
                UPDATE relx_native_bulk
                SET price = 123456789012345678.123456,
                    ratio = 0.1 WHERE id = 130
            """))
            connection.execute(text("""
                CREATE TABLE relx_native_users (
                    id BIGINT PRIMARY KEY, age DOUBLE
                )
            """))
            connection.execute(text("""
                CREATE TABLE relx_native_events (
                    id BIGINT PRIMARY KEY,
                    user_id BIGINT,
                    amount DOUBLE, event_time DATETIME(6) NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES relx_native_users(id)
                )
            """))
            connection.execute(text("""
                CREATE TABLE relx_native_labels (
                    id BIGINT PRIMARY KEY,
                    event_id BIGINT,
                    target BIGINT, prediction_at DATETIME(6) NOT NULL,
                    split VARCHAR(10) NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES relx_native_events(id)
                )
            """))
            connection.execute(text(
                "INSERT INTO relx_native_users VALUES (1, 20), (2, 30)"
            ))
            connection.execute(text("""
                INSERT INTO relx_native_events VALUES
                    (1, 1, 10, '2020-01-01 00:00:00'),
                    (2, 1, 20, '2020-01-02 00:00:00'),
                    (3, 2, 30, '2020-01-03 00:00:00')
            """))
            connection.execute(text("""
                INSERT INTO relx_native_labels VALUES
                    (1, 1, 0, '2020-02-01 00:00:00', 'train'),
                    (2, 3, 1, '2020-02-01 00:00:00', 'train'),
                    (3, 2, 0, '2020-03-01 00:00:00', 'val'),
                    (4, 3, 1, '2020-03-01 00:00:00', 'val'),
                    (5, 1, NULL, '2020-04-01 00:00:00', 'test')
            """))

    @classmethod
    def tearDownClass(cls) -> None:
        with cls.engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS relx_native_labels"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_events"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_users"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_bulk"))
            connection.execute(text("DROP TABLE IF EXISTS relx_native_items"))
        cls.engine.dispose()

    def _reader(self):
        from relx._relx_native import MySqlFeatureReader

        url = make_url(self.url)
        return MySqlFeatureReader(
            url.host or "", url.port or 3306, url.username or "",
            url.password or "", url.database or "",
            str(url.query.get("unix_socket", "")),
        )

    def test_composite_keys_and_feature_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            words = ("PADDING_TOKEN", "hello", "world")
            with (root / "vocab.bin").open("wb") as output:
                output.write(struct.pack("<IIIIII", 0x47584C52, 1, 3, 2, 0, 0))
                for word in words:
                    encoded = word.encode()
                    output.write(struct.pack("<I", len(encoded)))
                    output.write(encoded)
            np.asarray([[0, 0], [2, 4], [4, 8]], dtype=np.float32).tofile(
                root / "weights.f32"
            )
            reader = self._reader()
            reader.add_table(
                "relx_native_items", "", ["region", "id"],
                (("eu", 1), ("us", 1), ("us", 2)),
            )
            reader.set_glove_model(str(root))
            reader.add_feature("relx_native_items", "amount", "numerical", 15, 5)
            reader.add_feature(
                "relx_native_items", "segment", "categorical",
                vocab={"str:A": 1, "str:B": 2},
            )
            reader.add_feature(
                "relx_native_items", "tags", "multicategorical",
                vocab={"str:x": 1, "str:y": 2, "str:é": 3},
            )
            reader.add_feature("relx_native_items", "vector", "embedding", width=2)
            reader.add_feature("relx_native_items", "note", "text_glove")
            reader.add_feature("relx_native_items", "created_at", "timestamp")
            reader.add_feature("relx_native_items", "raw_value", "raw")
            result = reader.fetch({
                "relx_native_items": np.array([2, 0, 1, 0], dtype=np.int64)
            })["relx_native_items"]
            np.testing.assert_array_equal(
                result["amount"]["values"].ravel(), [1, -1, 0, -1]
            )
            np.testing.assert_array_equal(
                result["segment"]["values"], [2, 1, 0, 1]
            )
            np.testing.assert_array_equal(
                result["tags"]["values"], [2, 3, 1, 2, 1, 2]
            )
            np.testing.assert_array_equal(
                result["tags"]["offsets"], [0, 2, 4, 4, 6]
            )
            np.testing.assert_array_equal(
                result["vector"]["values"], [[3, 4], [1, 2], [0, 0], [1, 2]]
            )
            np.testing.assert_array_equal(
                result["note"]["values"], [[4, 8], [3, 6], [0, 0], [3, 6]]
            )
            np.testing.assert_array_equal(
                result["created_at"]["missing"], [False, False, True, False]
            )
            self.assertEqual(result["raw_value"]["values"].tolist(),
                             [b"\xff", b"\x00\x01", None, b"\x00\x01"])

    def test_chunked_query_restores_order(self) -> None:
        reader = self._reader()
        reader.add_table(
            "relx_native_bulk", "", ["id"],
            tuple((index,) for index in range(1, 131)),
        )
        reader.add_feature("relx_native_bulk", "amount", "numerical")
        reader.add_feature("relx_native_bulk", "note", "text")
        reader.add_feature("relx_native_bulk", "price", "raw")
        reader.add_feature(
            "relx_native_bulk", "ratio", "categorical",
            vocab={"float:0.1": 1},
        )
        ids = np.arange(129, -1, -1, dtype=np.int64)
        result = reader.fetch({"relx_native_bulk": ids})
        np.testing.assert_array_equal(
            result["relx_native_bulk"]["amount"]["values"].ravel(), ids + 1
        )
        self.assertEqual(result["relx_native_bulk"]["note"]["values"][0],
                         "α" * 5000)
        self.assertEqual(result["relx_native_bulk"]["price"]["values"][0],
                         "123456789012345678.123456")
        self.assertEqual(result["relx_native_bulk"]["ratio"]["values"][0], 1)

    def test_missing_source_row_reports_error(self) -> None:
        reader = self._reader()
        reader.add_table("relx_native_bulk", "", ["id"], ((9999,),))
        reader.add_feature("relx_native_bulk", "amount", "numerical")
        with self.assertRaisesRegex(RuntimeError, "disappeared"):
            reader.fetch({"relx_native_bulk": np.array([0], dtype=np.int64)})

    def test_threaded_pipeline_returns_mysql_batches_in_order(self) -> None:
        from relx._relx_native import NativeOnlinePipeline

        reader = self._reader()
        reader.add_table(
            "relx_native_bulk", "", ["id"],
            tuple((index,) for index in range(1, 131)),
        )
        reader.add_feature("relx_native_bulk", "amount", "numerical")
        pipeline = NativeOnlinePipeline(
            reader, workers=2, max_inflight=3, max_ready_bytes=1
        )
        try:
            for sequence, ids in enumerate(([0, 1], [129], [2, 3])):
                pipeline.submit(sequence, {
                    "relx_native_bulk": np.asarray(ids, dtype=np.int64)
                })
            pipeline.finish()
            outputs = [pipeline.next() for _ in range(3)]
            self.assertEqual([item["sequence"] for item in outputs], [0, 1, 2])
            np.testing.assert_array_equal(
                outputs[1]["features"]["relx_native_bulk"]["amount"]["values"],
                [[130]],
            )
            self.assertIsNone(pipeline.next())
        finally:
            pipeline.close()

    def test_yaml_to_training_report_uses_native_mysql(self) -> None:
        import relx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "database": {"url": self.url, "reader": "pandas"},
                "task": {
                    "name": "native-mysql", "type": "binary_classification",
                    "table": "relx_native_labels", "entity_column": "event_id",
                    "label_column": "target", "time_column": "prediction_at",
                    "split": {"type": "column", "column": "split", "values": {
                        "train": "train", "validation": "val", "test": "test",
                    }},
                },
                "graph": {
                    "exclude_tables": ["relx_native_items", "relx_native_bulk"],
                    "time_columns": {"relx_native_events": "event_time"},
                    "static_tables": ["relx_native_users"],
                },
                "features": {"columns": [
                    {"table": "relx_native_users", "column": "age",
                     "semantic_type": "numerical"},
                    {"table": "relx_native_events", "column": "amount",
                     "semantic_type": "numerical"},
                ]},
                "model": {"name": "rdl", "params": {
                    "num_neighbors": [2, 2], "channels": 8, "workers": 2,
                }},
                "training": {
                    "batch_size": 2, "epochs": 1, "feature_workers": 2,
                    "prefetch_batches": 2, "prefetch_bytes": 1024,
                    "feature_cache_bytes": 4096,
                },
                "report": {"directory": str(root / "report")},
            }
            path = root / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            trained = relx.run(path)
            self.assertEqual(trained.report.model_name, "rdl")
            self.assertEqual(len(trained.report.epochs), 1)
            self.assertTrue(trained.report.report_file.is_file())


if __name__ == "__main__":
    unittest.main()
