from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from relbench.base import Database, Table, TaskType

from data.contracts import RelBenchTask
from data.source import RelBenchDatasetSource
from data.writers import SQLiteDatabaseWriter
from relconnector.connector import (
    ConnectorXDatabaseReader,
    PandasDatabaseReader,
)
from relconnector.connector.catalog import ColumnSchema
from relconnector.connector.decoding import decode_frame


class FakeTask:
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_table = "users"
    entity_col = "user_id"
    target_col = "label"
    time_col = "timestamp"
    kind = "forecast"
    timedelta = cast(pd.Timedelta, pd.Timedelta(days=1))
    num_eval_timestamps = 1

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

    def get_table(self, split: str, mask_input_cols: bool | None = False) -> Table:
        del mask_input_cols
        return Table(
            self._frames[split],
            {"user_id": "users", "recommendations": "users"},
            time_col="timestamp",
        )

    def hidden_columns(self) -> list[tuple[str, str]]:
        return []


class FakeDataset:
    val_timestamp: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2024-02-01"))
    test_timestamp: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2024-03-01"))

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

    def load_task(self, task_name: str) -> RelBenchTask:
        if task_name != "conversion":
            raise KeyError(task_name)
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

    def test_mixed_iso_datetime_decoding(self) -> None:
        frame = pd.DataFrame(
            {"timestamp": ["2024-01-01T00:00:00", "2024-01-02T00:00:00.123456"]}
        )
        columns = (ColumnSchema("timestamp", 0, "datetime64[ns]", "datetime"),)

        decoded = decode_frame(frame, columns)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(decoded["timestamp"]))
        self.assertEqual(decoded["timestamp"].iloc[1].microsecond, 123456)

    def test_pandas_iteration_supports_full_and_chunked_reads(self) -> None:
        reader = PandasDatabaseReader(self.path)
        full = list(reader.iter_query('SELECT * FROM "users"'))
        chunks = list(
            reader.iter_query('SELECT * FROM "users" ORDER BY user_id', batch_size=2)
        )

        self.assertEqual(len(full), 1)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])

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
