"""Exercise recommendation sampling and online SQL feature assembly."""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

import yaml

from relx import run


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native")
    and importlib.util.find_spec("pyg_lib"),
    "native extension and pyg-lib needed",
)
class RecommendationSmokeTest(unittest.TestCase):
    def test_one_batch_per_split_reads_sql_and_trains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "recommendation.sqlite"
            with sqlite3.connect(database) as sql:
                sql.executescript("""
                    CREATE TABLE users (id INTEGER PRIMARY KEY, age REAL);
                    CREATE TABLE products (id INTEGER PRIMARY KEY, cost REAL);
                    CREATE TABLE interactions (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        item_id INTEGER REFERENCES products(id)
                    );
                    CREATE TABLE labels (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        items TEXT, split TEXT NOT NULL
                    );
                    INSERT INTO users VALUES (1, 20), (2, 30);
                    INSERT INTO products VALUES (1, 2), (2, 4), (3, 6);
                    INSERT INTO interactions VALUES (1, 1, 1), (2, 1, 2), (3, 2, 3);
                    INSERT INTO labels VALUES
                        (1, 1, '[1, 2]', 'train'),
                        (2, 2, '[3]', 'train'),
                        (3, 1, '[2]', 'val'),
                        (4, 2, '[3]', 'val'),
                        (5, 1, '[1]', 'test'),
                        (6, 2, '[3]', 'test');
                """)
            config = {
                "database": {"url": str(database)},
                "task": {
                    "name": "items", "type": "recommendation", "table": "labels",
                    "entity_column": "user_id", "label_column": "items",
                    "destination": {"table": "products", "key": "id"},
                    "split": {
                        "type": "column", "column": "split",
                        "values": {
                            "train": "train", "validation": "val", "test": "test"
                        },
                    },
                },
                "graph": {"static_tables": ["users", "products", "interactions"]},
                "model": {"name": "rdl", "params": {
                    "channels": 8, "num_neighbors": [2],
                }},
                "training": {"epochs": 1, "batch_size": 1, "max_batches": 1},
                "report": {"directory": str(root / "report")},
            }
            path = root / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            trained = run(path)
            self.assertEqual(trained.report.metric, "pairwise_accuracy")
            self.assertEqual(trained.report.train_rows, 2)
            self.assertTrue(trained.report.report_file.is_file())
            self.assertGreaterEqual(trained.report.epochs[0].train_loss, 0)


if __name__ == "__main__":
    unittest.main()
