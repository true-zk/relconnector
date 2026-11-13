"""Database readers execute SQL without a proprietary catalog."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from importlib.util import find_spec
from pathlib import Path

import pandas as pd

from relx.conn import ConnectorXDatabaseReader, PandasDatabaseReader


class DatabaseReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "plain.sqlite"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE items (id INTEGER, parent_id INTEGER)")
            connection.executemany(
                "INSERT INTO items VALUES (?, ?)",
                [(0, 2**53 + 1), (1, None), (2, 3)],
            )
            connection.commit()

    def test_pandas_reader(self) -> None:
        reader = PandasDatabaseReader(self.path)
        self.assertEqual(
            reader.read_query("SELECT COUNT(*) AS n FROM items")["n"][0], 3
        )
        chunks = list(
            reader.iter_query("SELECT id FROM items ORDER BY id", batch_size=2)
        )
        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])
        with self.assertRaises(ValueError):
            list(reader.iter_query("SELECT id FROM items", batch_size=0))

    @unittest.skipUnless(find_spec("connectorx"), "connectorx is optional")
    def test_connectorx_reader(self) -> None:
        reader = ConnectorXDatabaseReader(self.path)
        self.assertEqual(
            reader.read_query("SELECT COUNT(*) AS n FROM items")["n"][0], 3
        )

    @unittest.skipUnless(
        find_spec("connectorx") and find_spec("pyarrow"),
        "connectorx and pyarrow are optional",
    )
    def test_connectorx_stream_preserves_large_nullable_ids(self) -> None:
        reader = ConnectorXDatabaseReader(self.path)
        chunks = list(
            reader.iter_query("SELECT parent_id FROM items ORDER BY id", batch_size=2)
        )
        self.assertEqual(sum(len(chunk) for chunk in chunks), 3)
        self.assertTrue(all(0 < len(chunk) <= 2 for chunk in chunks))
        values = pd.concat(chunks)["parent_id"].tolist()
        self.assertEqual(values[0], 2**53 + 1)
        self.assertTrue(pd.isna(values[1]))

    def test_missing_database_is_not_created(self) -> None:
        missing = self.path.parent / "missing.sqlite"
        for reader_type in (PandasDatabaseReader, ConnectorXDatabaseReader):
            with self.assertRaises(FileNotFoundError):
                reader_type(missing)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
