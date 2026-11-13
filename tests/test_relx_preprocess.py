"""Schema discovery uses SQLite metadata rather than custom catalog tables."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from importlib.util import find_spec
from pathlib import Path

from relx.conn import BaseDatabaseReader, ConnectorXDatabaseReader, PandasDatabaseReader
from relx.preprocess import discover_sqlite_schema


class SchemaDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "plain.sqlite"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript(
                """
                CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id)
                );
                CREATE TABLE implicit_fk (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users
                );
                CREATE TABLE text_keys (name TEXT PRIMARY KEY);
                CREATE TABLE metadata (value TEXT);
                """
            )

    def test_discovers_selected_tables_and_foreign_keys(self) -> None:
        reader = PandasDatabaseReader(self.path)
        schemas = discover_sqlite_schema(reader, table_names=["users", "orders"])
        self.assertEqual(list(schemas), ["users", "orders"])
        self.assertEqual(schemas["users"].primary_key, "id")
        self.assertEqual(schemas["users"].columns[0].declared_type, "INTEGER")
        self.assertFalse(schemas["users"].columns[0].nullable)
        self.assertFalse(schemas["users"].columns[1].nullable)
        self.assertEqual(schemas["orders"].foreign_keys[0].reference_table, "users")
        self.assertEqual(schemas["orders"].foreign_keys[0].reference_column, "id")
        self.assertNotIn("metadata", schemas)
        text_key = discover_sqlite_schema(reader, table_names=["text_keys"])[
            "text_keys"
        ]
        self.assertTrue(text_key.columns[0].nullable)

    @unittest.skipUnless(find_spec("connectorx"), "connectorx is optional")
    def test_connectorx_can_discover_the_same_structure(self) -> None:
        schemas = discover_sqlite_schema(
            ConnectorXDatabaseReader(self.path), table_names=["users", "orders"]
        )
        self.assertEqual(schemas["orders"].foreign_keys[0].column, "user_id")

    def test_rejects_unknown_table_and_composite_key(self) -> None:
        reader = PandasDatabaseReader(self.path)
        with self.assertRaisesRegex(ValueError, "Unknown tables"):
            discover_sqlite_schema(reader, table_names=["missing"])
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "CREATE TABLE pairs (a INTEGER, b INTEGER, PRIMARY KEY (a, b))"
            )
        with self.assertRaisesRegex(ValueError, "composite primary key"):
            discover_sqlite_schema(reader, table_names=["pairs"])

    def test_resolves_implicit_foreign_key_target(self) -> None:
        readers: list[BaseDatabaseReader] = [PandasDatabaseReader(self.path)]
        if find_spec("connectorx"):
            readers.append(ConnectorXDatabaseReader(self.path))
        for reader in readers:
            with self.subTest(reader=type(reader).__name__):
                schema = discover_sqlite_schema(reader, table_names=["implicit_fk"])
                self.assertEqual(
                    schema["implicit_fk"].foreign_keys[0].reference_column, "id"
                )


if __name__ == "__main__":
    unittest.main()
