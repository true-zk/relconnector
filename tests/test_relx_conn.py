"""Database readers execute SQL without a proprietary catalog."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sqlalchemy import literal, select, text

import relx
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

    def test_public_connection_and_parameters(self) -> None:
        kinds = (
            ("pandas", "connectorx")
            if find_spec("connectorx") and find_spec("pyarrow")
            else ("pandas",)
        )
        for kind in kinds:
            with (
                self.subTest(reader=kind),
                relx.connect(self.path, reader=kind) as reader,
            ):
                actual = reader.read_query(
                    "SELECT id FROM items WHERE parent_id = :value",
                    params={"value": 2**53 + 1},
                )
                self.assertEqual(actual["id"].tolist(), [0])
                chunks = list(
                    reader.iter_query(
                        "SELECT id FROM items WHERE id >= :minimum ORDER BY id",
                        params={"minimum": 1},
                        batch_size=1,
                    )
                )
                self.assertEqual([chunk["id"].tolist() for chunk in chunks], [[1], [2]])
            with self.assertRaisesRegex(RuntimeError, "closed"):
                reader.read_query("SELECT 1")

    def test_sqlalchemy_query_expression(self) -> None:
        kinds = ("pandas", "connectorx") if find_spec("connectorx") else ("pandas",)
        for kind in kinds:
            with (
                self.subTest(reader=kind),
                relx.connect(self.path, reader=kind) as reader,
            ):
                result = reader.read_query(select(literal(7).label("value")))
                self.assertEqual(result["value"].tolist(), [7])

    @unittest.skipUnless(find_spec("connectorx"), "connectorx is optional")
    def test_connectorx_url_and_string_parameter(self) -> None:
        reader = ConnectorXDatabaseReader(
            "postgresql+psycopg://user:secret@localhost/example"
        )
        self.assertEqual(
            reader.connectorx_url,
            "postgresql://user:secret@localhost/example",
        )
        with patch(
            "connectorx.read_sql", return_value=pd.DataFrame({"value": [1]})
        ) as read:
            reader.read_query("SELECT :value AS value", params={"value": "O'Reilly"})
        self.assertEqual(read.call_args.args[1], "SELECT 'O''Reilly' AS value")
        with patch(
            "connectorx.read_sql", return_value=pd.DataFrame({"value": [1]})
        ) as read:
            reader.read_query(text("SELECT :value AS value"), params={"value": 7})
        self.assertEqual(read.call_args.args[1], "SELECT 7 AS value")
        reader.close()

    def test_connectorx_sql_server_url_drops_odbc_driver(self) -> None:
        reader = ConnectorXDatabaseReader(
            "mssql+pyodbc://user:pw@localhost/db?driver=ODBC+Driver+18+for+SQL+Server"
        )
        self.assertEqual(reader.connectorx_url, "mssql://user:pw@localhost/db")
        reader.close()

    def test_database_urls_are_not_sqlite_paths(self) -> None:
        for url, backend in (
            ("postgresql+psycopg://user:pw@localhost/db", "postgresql"),
            ("mysql+pymysql://user:pw@localhost/db", "mysql"),
            ("mssql+pyodbc://user:pw@localhost/db", "mssql"),
            ("oracle+oracledb://user:pw@localhost/db", "oracle"),
        ):
            with self.subTest(backend=backend):
                reader = relx.connect(url)
                self.assertEqual(reader.database_url.get_backend_name(), backend)
                self.assertIsNone(reader.path)
                reader.close()

    def test_unknown_reader(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown reader"):
            relx.connect(self.path, reader="unknown")  # type: ignore[arg-type]

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
