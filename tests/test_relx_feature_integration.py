"""Default feature processors retain their output through the native cache."""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from relx import prepare
from relx.batch import FeatureStore, OnlineFeatureStore
from relx.batch.transforms import transform_builtin
from relx.conn import connect
from relx.core import TableRef
from relx.sampling import EntitySamplePlan, NativeOneHopSampler
from relx.task import BatchKey, EntitySeedBatch


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native")
    and importlib.util.find_spec("pyarrow"),
    "native graph and pyarrow needed",
)
class NativeFeatureIntegrationTest(unittest.TestCase):
    def test_online_fetch_uses_composite_identity_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "composite.sqlite"
            config_file = Path(directory) / "run.yaml"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE users (
                        region TEXT NOT NULL, user_id INTEGER NOT NULL,
                        score REAL, PRIMARY KEY (region, user_id)
                    );
                    CREATE TABLE events (
                        id INTEGER PRIMARY KEY, region TEXT, user_id INTEGER,
                        FOREIGN KEY (region, user_id) REFERENCES users(region, user_id)
                    );
                    CREATE TABLE labels (
                        id INTEGER PRIMARY KEY, event_id INTEGER REFERENCES events(id),
                        target INTEGER, split TEXT
                    );
                    INSERT INTO users VALUES ('eu', 1, 10), ('us', 1, 20);
                    INSERT INTO events VALUES (1, 'us', 1), (2, 'eu', 1);
                    INSERT INTO labels VALUES
                        (1, 1, 0, 'train'), (2, 2, 1, 'train'),
                        (3, 2, 1, 'val'), (4, 1, NULL, 'test');
                    """
                )
            config = {
                "database": {"url": str(database)},
                "task": {
                    "name": "composite", "type": "binary_classification",
                    "table": "labels", "entity_column": "event_id",
                    "label_column": "target",
                    "split": {"type": "column", "column": "split", "values": {
                        "train": "train", "validation": "val", "test": "test",
                    }},
                },
                "features": {"columns": [{
                    "table": "users", "column": "score", "semantic_type": "numerical",
                }]},
            }
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            prepared = prepare(config_file)
            seeds = EntitySeedBatch(
                BatchKey(0, 0), "events", np.array([0, 1]), None,
                np.array([0, 1]), ((1,), (2,)),
            )
            plan = NativeOneHopSampler(
                prepared.graph, prepared.schema, fanout=-1
            ).sample(seeds)
            assert isinstance(plan, EntitySamplePlan)
            for reader_kind in ("pandas", "connectorx"):
                with self.subTest(reader=reader_kind):
                    with connect(database, reader=reader_kind) as reader:
                        store = OnlineFeatureStore(
                            prepared, reader, train_cutoff_us=None,
                            scan_batch_size=1,
                        )
                    batch = store.fetch(plan)
                    feature = batch.features[(TableRef("users"), "score")]
                    np.testing.assert_array_equal(feature.values.ravel(), [1, -1])

    def test_all_default_feature_types_match_python_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.sqlite"
            config_file = Path(directory) / "run.yaml"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE items (
                        id INTEGER PRIMARY KEY, amount REAL, segment TEXT,
                        tags TEXT, vector TEXT, note TEXT, created_at TEXT
                    );
                    CREATE TABLE labels (
                        id INTEGER PRIMARY KEY,
                        item_id INTEGER REFERENCES items(id),
                        target INTEGER, split TEXT
                    );
                    INSERT INTO items VALUES
                        (1, 10, 'A', '["x", "y"]', '[1, 2]', 'hello', '2020-01-01'),
                        (2, NULL, NULL, '[]', NULL, NULL, NULL),
                        (3, 30, 'B', '["y"]', '[3, 4]', 'world', '2020-01-03');
                    INSERT INTO labels VALUES
                        (1, 1, 0, 'train'), (2, 2, 1, 'train'),
                        (3, 3, 0, 'val'), (4, 3, NULL, 'test');
                    """
                )
            config = {
                "database": {"url": str(database)},
                "task": {
                    "name": "demo", "type": "binary_classification",
                    "table": "labels", "entity_column": "item_id",
                    "label_column": "target",
                    "split": {
                        "type": "column", "column": "split",
                        "values": {
                            "train": "train", "validation": "val", "test": "test"
                        },
                    },
                },
                "features": {
                    "columns": [
                        {"table": "items", "column": name,
                         "semantic_type": semantic, **params}
                        for name, semantic, params in (
                            ("amount", "numerical", {}),
                            ("segment", "categorical", {}),
                            ("tags", "multicategorical", {}),
                            ("vector", "embedding", {"params": {"width": 2}}),
                            ("note", "text", {}),
                            ("created_at", "timestamp", {}),
                        )
                    ]
                },
            }
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            prepared = prepare(config_file)
            with connect(database) as reader:
                store = FeatureStore(
                    prepared, reader, train_cutoff_us=None, native_workers=3
                )
                online = OnlineFeatureStore(
                    prepared, reader, train_cutoff_us=None,
                    scan_batch_size=2, native_workers=3,
                )
            self.assertFalse(store.frames)
            node_ids = np.array([2, 0, 1, 2], dtype=np.int64)
            seeds = EntitySeedBatch(
                BatchKey(0, 0), "items", node_ids, None,
                np.array([0, 1, 0, 0]), ((3,), (1,), (2,), (3,)),
            )
            plan = NativeOneHopSampler(
                prepared.graph, prepared.schema, fanout=-1
            ).sample(seeds)
            assert isinstance(plan, EntitySamplePlan)
            fetched = store.fetch(plan)
            with patch(
                "relx.conn.pandas.PandasDatabaseReader.read_query",
                side_effect=AssertionError("Python must not query batch features"),
            ), patch(
                "relx.conn.connectorx.ConnectorXDatabaseReader.read_query",
                side_effect=AssertionError("Python must not query builtin features"),
            ):
                online_fetched = online.fetch(plan)
            with sqlite3.connect(database) as connection:
                raw = pd.read_sql_query("SELECT * FROM items ORDER BY id", connection)
            ref = TableRef("items")
            for spec in prepared.features.for_table(ref):
                expected = transform_builtin(
                    spec, raw[spec.column].iloc[node_ids],
                    store.states[(ref, spec.column)],
                    prepared.config.graph.timezone,
                )
                actual = fetched.features[(ref, spec.column)]
                online_actual = online_fetched.features[(ref, spec.column)]
                np.testing.assert_array_equal(actual.values, expected.values)
                np.testing.assert_array_equal(online_actual.values, expected.values)
                if expected.missing is None:
                    self.assertIsNone(actual.missing)
                    self.assertIsNone(online_actual.missing)
                else:
                    np.testing.assert_array_equal(actual.missing, expected.missing)
                    np.testing.assert_array_equal(online_actual.missing, expected.missing)
                if expected.offsets is None:
                    self.assertIsNone(actual.offsets)
                    self.assertIsNone(online_actual.offsets)
                else:
                    np.testing.assert_array_equal(actual.offsets, expected.offsets)
                    np.testing.assert_array_equal(online_actual.offsets, expected.offsets)


if __name__ == "__main__":
    unittest.main()
