"""Database schema discovery and configured relationship tests."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from relx.conn import ConnectorXDatabaseReader, PandasDatabaseReader
from relx.core import ForeignKey, TableRef
from relx.preprocess import SchemaOverrides, discover_schema


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
                CREATE TABLE events (id INTEGER PRIMARY KEY, user_id INTEGER);
                CREATE TABLE pairs (
                    a INTEGER, b INTEGER, PRIMARY KEY (a, b)
                );
                CREATE TABLE pair_events (
                    a INTEGER, b INTEGER,
                    FOREIGN KEY (a, b) REFERENCES pairs(a, b)
                );
                """
            )

    def test_discovers_selected_tables_and_column_types(self) -> None:
        users = TableRef("users")
        orders = TableRef("orders")
        with PandasDatabaseReader(self.path) as reader:
            schema = discover_schema(reader, table_names=["users", "orders"])

        self.assertEqual(list(schema.tables), [users, orders])
        self.assertEqual(schema.tables[users].primary_key, ("id",))
        self.assertEqual(schema.tables[users].identity_key, ("id",))
        self.assertEqual(schema.tables[users].columns[0].raw_sql_type, "INTEGER")
        self.assertFalse(schema.tables[users].columns[0].nullable)
        self.assertFalse(schema.tables[users].columns[1].nullable)
        self.assertEqual(
            schema.tables[orders].foreign_keys,
            (ForeignKey(("user_id",), users, ("id",)),),
        )
        self.assertNotIn(TableRef("metadata"), schema.tables)

    def test_preserves_composite_keys_and_nullable_text_key(self) -> None:
        pairs = TableRef("pairs")
        events = TableRef("pair_events")
        with PandasDatabaseReader(self.path) as reader:
            schema = discover_schema(
                reader, table_names=["pairs", "pair_events", "text_keys"]
            )
        self.assertEqual(schema.tables[pairs].primary_key, ("a", "b"))
        self.assertEqual(
            schema.tables[events].foreign_keys,
            (ForeignKey(("a", "b"), pairs, ("a", "b")),),
        )
        self.assertTrue(schema.tables[TableRef("text_keys")].columns[0].nullable)

    def test_resolves_implicit_foreign_key_with_either_reader(self) -> None:
        ref = TableRef("implicit_fk")
        users = TableRef("users")
        for reader_type in (PandasDatabaseReader, ConnectorXDatabaseReader):
            with (
                self.subTest(reader=reader_type.__name__),
                reader_type(self.path) as reader,
            ):
                schema = discover_schema(reader, table_names=["implicit_fk"])
                self.assertEqual(
                    schema.tables[ref].foreign_keys,
                    (ForeignKey(("user_id",), users, ("id",)),),
                )

    def test_schema_qualified_tables_do_not_collide(self) -> None:
        main = TableRef("users")
        other = TableRef("users", "aux")
        with PandasDatabaseReader(self.path) as reader:
            with reader.engine.begin() as connection:
                connection.exec_driver_sql("ATTACH DATABASE ':memory:' AS aux")
                connection.exec_driver_sql(
                    "CREATE TABLE aux.users (user_key TEXT PRIMARY KEY)"
                )
            schema = discover_schema(reader, table_names=[main, other])
            aux_only = discover_schema(reader, schema_name="aux")
        self.assertEqual(set(schema.tables), {main, other})
        self.assertEqual(schema.tables[other].primary_key, ("user_key",))
        self.assertEqual(set(aux_only.tables), {other})

    def test_overrides_add_identity_key_and_relationship(self) -> None:
        events = TableRef("events")
        users = TableRef("users")
        metadata = TableRef("metadata")
        overrides = SchemaOverrides(
            identity_keys={metadata: ("value",)},
            foreign_keys={events: (ForeignKey(("user_id",), users, ("id",)),)},
        )
        with PandasDatabaseReader(self.path) as reader:
            schema = discover_schema(
                reader,
                table_names=["events", "metadata"],
                overrides=overrides,
            )
        self.assertIsNone(schema.tables[metadata].primary_key)
        self.assertEqual(schema.tables[metadata].identity_key, ("value",))
        self.assertEqual(
            schema.tables[events].foreign_keys, overrides.foreign_keys[events]
        )

    def test_rejects_invalid_selections_and_overrides(self) -> None:
        events = TableRef("events")
        users = TableRef("users")
        with PandasDatabaseReader(self.path) as reader:
            with self.assertRaisesRegex(ValueError, "Unknown tables"):
                discover_schema(reader, table_names=["missing"])
            with self.assertRaisesRegex(ValueError, "Unknown columns"):
                discover_schema(
                    reader,
                    table_names=["events"],
                    overrides=SchemaOverrides(identity_keys={events: ("missing",)}),
                )

            with self.assertRaisesRegex(ValueError, "Unknown referenced table"):
                discover_schema(
                    reader,
                    table_names=["events"],
                    overrides=SchemaOverrides(
                        foreign_keys={
                            events: (
                                ForeignKey(("user_id",), TableRef("missing"), ("id",)),
                            )
                        }
                    ),
                )
            with self.assertRaisesRegex(ValueError, "unequal column counts"):
                discover_schema(
                    reader,
                    table_names=["events"],
                    overrides=SchemaOverrides(
                        foreign_keys={
                            events: (ForeignKey(("id", "user_id"), users, ("id",)),)
                        }
                    ),
                )


@unittest.skipUnless(
    os.getenv("RELX_TEST_POSTGRES_URL"), "set RELX_TEST_POSTGRES_URL for integration"
)
class PostgresSchemaDiscoveryTest(unittest.TestCase):
    def test_reflects_system_table_without_writing(self) -> None:
        url = os.environ["RELX_TEST_POSTGRES_URL"]
        ref = TableRef("pg_class", "pg_catalog")
        with PandasDatabaseReader(url) as reader:
            schema = discover_schema(reader, table_names=[ref])
        self.assertIn("oid", {column.name for column in schema.tables[ref].columns})


if __name__ == "__main__":
    unittest.main()
