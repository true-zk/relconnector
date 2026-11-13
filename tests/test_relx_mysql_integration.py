"""Read-only MySQL integration check against an imported Rel-F1 database."""

from __future__ import annotations

import os
import unittest

import relx


@unittest.skipUnless(
    os.getenv("RELX_TEST_MYSQL_REL_F1_URL"),
    "set RELX_TEST_MYSQL_REL_F1_URL to an imported Rel-F1 database",
)
class MySQLRelF1SchemaTest(unittest.TestCase):
    def test_reflects_tables_columns_and_relationships(self) -> None:
        url = os.environ["RELX_TEST_MYSQL_REL_F1_URL"]
        for reader_kind in ("pandas", "connectorx"):
            with (
                self.subTest(reader=reader_kind),
                relx.connect(url, reader=reader_kind) as database,
            ):
                schema = relx.discover_schema(
                    database, table_names=["circuits", "races", "results"]
                )
                circuits = schema.tables[relx.TableRef("circuits")]
                races = schema.tables[relx.TableRef("races")]
                results = schema.tables[relx.TableRef("results")]
                self.assertEqual(circuits.primary_key, ("circuitId",))
                self.assertEqual(len(circuits.columns), 8)
                self.assertIn(
                    relx.ForeignKey(
                        ("circuitId",), relx.TableRef("circuits"), ("circuitId",)
                    ),
                    races.foreign_keys,
                )
                self.assertEqual(len(results.foreign_keys), 3)


if __name__ == "__main__":
    unittest.main()
