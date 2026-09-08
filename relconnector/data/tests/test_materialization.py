from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from relbench.base import Database, Table

from data.source import RelBenchDatasetSource
from data.writers import SQLiteDatabaseWriter
from relconnector.connector import (
    ConnectorXDatabaseReader,
    PandasDatabaseReader,
)


class FakeTask:
    task_type = "binary_classification"
    entity_table = "users"
    entity_col = "user_id"
    target_col = "label"
    time_col = "timestamp"
    kind = "forecast"

    def __init__(self) -> None:
        self._frames = {
            split: pd.DataFrame(
                {
                    "timestamp": pd.to_datetime([f"2024-0{index + 1}-01"]),
                    "user_id": pd.Series([index], dtype="int64"),
                    "label": pd.Series([index % 2 == 0], dtype="bool"),
                    "recommendations": [[index, index + 1]],
                }
            )
            for index, split in enumerate(("train", "val", "test"))
        }

    def get_table(self, split: str, mask_input_cols: bool = False) -> Table:
        del mask_input_cols
        return Table(
            self._frames[split],
            {"user_id": "users", "recommendations": "users"},
            time_col="timestamp",
        )


class FakeDataset:
    val_timestamp = pd.Timestamp("2024-02-01")
    test_timestamp = pd.Timestamp("2024-03-01")

    def __init__(self) -> None:
        self.task = FakeTask()
        self.database = Database(
            {
                "users": Table(
                    pd.DataFrame(
                        {
                            "user_id": pd.Series([0, 1, 2], dtype="int64"),
                            "external_id": pd.Series(
                                [2**60 + 123, pd.NA, 2**60 + 125],
                                dtype="Int64",
                            ),
                            "joined_at": pd.to_datetime(
                                ["2023-01-01", "2023-02-01", "2023-03-01"]
                            ),
                            "active": pd.Series([True, False, True], dtype="bool"),
                            "segment": pd.Series(["a", "b", "a"], dtype="category"),
                        }
                    ),
                    {},
                    pkey_col="user_id",
                    time_col="joined_at",
                ),
                "orders": Table(
                    pd.DataFrame(
                        {
                            "order_id": pd.Series([0, 1, 2], dtype="int64"),
                            "user_id": pd.Series([0, 1, pd.NA], dtype="Int64"),
                            "amount": [1.25, np.nan, 7.5],
                            "tags": [["new"], [], None],
                        }
                    ),
                    {"user_id": "users"},
                    pkey_col="order_id",
                ),
            }
        )

    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        self.requested_complete_database = not upto_test_timestamp
        return self.database

    def load_task(self, name: str) -> FakeTask:
        if name != "conversion":
            raise KeyError(name)
        return self.task

    def get_task_names(self) -> list[str]:
        return ["conversion"]


class MaterializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "fixture.sqlite"
        self.dataset = FakeDataset()
        source = RelBenchDatasetSource(dataset_loader=lambda _: self.dataset)
        bundle = source.load("fixture", ["conversion"])
        SQLiteDatabaseWriter(self.path, insertion_chunk_size=2).write(bundle)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_pandas_reader_round_trip(self) -> None:
        reader = PandasDatabaseReader(self.path)

        self.assertTrue(self.dataset.requested_complete_database)
        self.assertEqual(reader.table_names(), ["orders", "target_table", "users"])
        self.assertEqual(reader.metadata()["dataset_name"], "fixture")
        self.assertEqual(reader.metadata()["catalog_version"], "1")

        database = reader.read_relbench_database()
        self.assertEqual(
            database.table_dict["orders"].fkey_col_to_pkey_table,
            {"user_id": "users"},
        )
        self.assertEqual(database.table_dict["users"].time_col, "joined_at")
        assert_frame_equal(
            database.table_dict["users"].df,
            self.dataset.database.table_dict["users"].df,
        )
        assert_frame_equal(
            database.table_dict["orders"].df,
            self.dataset.database.table_dict["orders"].df,
        )

        splits = reader.read_task_splits("conversion")
        self.assertEqual(list(splits), ["train", "val", "test"])
        self.assertEqual(
            [key.column for key in reader.schemas()["target_table"].foreign_keys],
            ["user_id"],
        )
        for split, expected in self.dataset.task._frames.items():
            assert_frame_equal(splits[split], expected)

    def test_full_memory_iteration_contract(self) -> None:
        reader = PandasDatabaseReader(self.path)
        batches = list(reader.iter_query('SELECT * FROM "users"'))
        self.assertEqual(len(batches), 1)
        with self.assertRaises(NotImplementedError):
            list(reader.iter_query('SELECT * FROM "users"', batch_size=2))

    @unittest.skipUnless(
        importlib.util.find_spec("connectorx"), "connectorx is not installed"
    )
    def test_connectorx_reader_round_trip(self) -> None:
        pandas_reader = PandasDatabaseReader(self.path)
        connectorx_reader = ConnectorXDatabaseReader(self.path)
        for table_name in pandas_reader.table_names():
            assert_frame_equal(
                connectorx_reader.read_table(table_name),
                pandas_reader.read_table(table_name),
                check_dtype=True,
            )


if __name__ == "__main__":
    unittest.main()
