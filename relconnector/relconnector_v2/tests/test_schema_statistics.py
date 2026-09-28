from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from torch_frame import stype
from torch_frame.data.stats import StatType

from relconnector_v2.connector import ColumnSchema, TableSchema
from relconnector_v2.connector.pandas import PandasDatabaseReader
from relconnector_v2.features.schema import SqlTensorFrameSchemaBuilder


class SchemaStatisticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.sqlite"
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE nodes (id INTEGER, a REAL, b TEXT, empty REAL, date TEXT)"
            )
            conn.executemany(
                "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
                [
                    (0, 1.0, "3", None, "2020-01-01"),
                    (1, None, "bad", None, "2020-01-02"),
                    (2, float("inf"), "7", None, "2020-01-03"),
                    (3, -4.0, None, None, "2020-01-04"),
                    (4, 9.0, "2", None, "2020-01-05"),
                    (5, 1000.0, "1000", None, "2030-01-01"),
                ],
            )
        self.reader = PandasDatabaseReader(str(self.path))
        self.columns = (
            ColumnSchema("a", 1, "float64"),
            ColumnSchema("b", 2, "object"),
            ColumnSchema("empty", 3, "float64"),
        )
        self.table = TableSchema(
            "nodes", "id", time_column="date", columns=self.columns
        )
        self.kinds = {c.name: stype.numerical for c in self.columns}

    def builder(self, budget: int) -> SqlTensorFrameSchemaBuilder:
        return SqlTensorFrameSchemaBuilder(
            self.reader,
            scan_batch_size=2,
            cutoff=pd.Timestamp("2021-01-01"),
            numerical_buffer_bytes=budget,
        )

    def test_grouped_and_streaming_match_exact_statistics(self) -> None:
        expected = {"a": [1.0, -4.0, 9.0], "b": [3.0, 7.0, 2.0], "empty": []}
        for budget in (48, 128, 4096):
            builder = self.builder(budget)
            grouped = builder._table_numerical_stats(
                self.table, self.columns, self.kinds
            )
            self.assertEqual(set(grouped), set(expected))
            for col in self.columns:
                old = builder._numerical_stats(self.table, col)
                for key in (StatType.MEAN, StatType.STD, StatType.QUANTILES):
                    np.testing.assert_allclose(
                        grouped[col.name][key], old[key], equal_nan=True
                    )
                values = expected[col.name]
                if values:
                    np.testing.assert_allclose(
                        grouped[col.name][StatType.QUANTILES],
                        np.quantile(values, [0, 0.25, 0.5, 0.75, 1]),
                    )

    def test_budget_fallback_and_empty_table(self) -> None:
        for budget in (0, 39):
            self.assertEqual(
                self.builder(budget)._table_numerical_stats(
                    self.table, self.columns, self.kinds
                ),
                {},
            )
        builder = self.builder(4096)
        builder.cutoff = pd.Timestamp("1900-01-01")
        stats = builder._table_numerical_stats(self.table, self.columns, self.kinds)
        self.assertTrue(np.isnan(stats["a"][StatType.MEAN]))
        self.assertTrue(np.isnan(stats["a"][StatType.QUANTILES]).all())

    def test_singleton_and_timestamp_chunk_boundaries(self) -> None:
        builder = self.builder(4096)
        builder.cutoff = pd.Timestamp("2020-01-01")
        stats = builder._table_numerical_stats(self.table, self.columns, self.kinds)
        self.assertEqual(stats["a"][StatType.QUANTILES], [1.0] * 5)
        self.assertEqual(stats["a"][StatType.STD], 0.0)
        builder.cutoff = pd.Timestamp("2021-01-01")
        col = ColumnSchema("date", 4, "datetime64[ns]", encoding="datetime")
        actual = builder._timestamp_stats(self.table, col)
        from torch_frame.data.mapper import TimestampTensorMapper

        for key, date in [
            (StatType.OLDEST_TIME, "2020-01-01"),
            (StatType.MEDIAN_TIME, "2020-01-03"),
            (StatType.NEWEST_TIME, "2020-01-05"),
        ]:
            expected = TimestampTensorMapper.to_tensor(
                pd.Series([pd.Timestamp(date)])
            ).squeeze(0)
            self.assertTrue(actual[key].equal(expected))

    def test_build_excludes_hidden_columns_and_roundtrips_cache(self) -> None:
        builder = self.builder(4096)
        builder.hidden_columns = {("nodes", "b")}
        builder.cache_dir = Path(self.temp.name) / "cache"
        with (
            patch.object(self.reader, "schemas", return_value={"nodes": self.table}),
            patch.object(
                SqlTensorFrameSchemaBuilder,
                "_infer_stypes",
                side_effect=lambda table, cols: {c.name: stype.numerical for c in cols},
            ),
        ):
            cold = builder.build()
            self.assertFalse(builder.cache_hit)
            self.assertNotIn("b", cold.tables["nodes"].col_stats)
            warm = builder.build()
            self.assertTrue(builder.cache_hit)
            self.assertEqual(warm.fingerprint, cold.fingerprint)

    def test_large_offset_variance_and_interpolated_quantiles(self) -> None:
        values = 1e9 + np.random.default_rng(42).normal(size=10001)
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM nodes")
            conn.executemany(
                "INSERT INTO nodes VALUES (?, ?, NULL, NULL, '2020-01-01')",
                enumerate(values),
            )
        builder = self.builder(128 * 1024)
        builder.scan_batch_size = 997
        stats = builder._table_numerical_stats(
            self.table, self.columns[:1], self.kinds
        )["a"]
        np.testing.assert_allclose(stats[StatType.STD], values.std(), rtol=1e-12)
        np.testing.assert_allclose(
            stats[StatType.QUANTILES],
            np.quantile(values, [0, 0.25, 0.5, 0.75, 1]),
            rtol=1e-15,
        )


if __name__ == "__main__":
    unittest.main()
