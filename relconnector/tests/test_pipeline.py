from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from relbench.base import Database, Table, TaskType

from relconnector.connector.catalog import TaskMetadata
from relconnector.pipeline.benchmark import _prepare_output, discover_cases
from relconnector.pipeline.dataset import (
    LocalEntityTask,
    LocalRecommendationTask,
    _build_task,
    _validate_and_correct_database,
)
from relconnector.pipeline.telemetry import TelemetryRecorder, timed
from relconnector.pipeline.text_embedder import _normalize_text


class LocalTaskTest(unittest.TestCase):
    def test_builds_entity_task_from_catalog(self) -> None:
        splits = {
            "train": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-01"]),
                    "user_id": [1],
                    "label": [True],
                }
            )
        }
        metadata = TaskMetadata(
            name="conversion",
            table_name="target_conversion",
            task_type=TaskType.BINARY_CLASSIFICATION.value,
            entity_table="users",
            entity_column="user_id",
            target_column="label",
            time_column="timestamp",
            extra={
                "kind": "forecast",
                "hidden_columns": [["orders", "future_value"]],
            },
        )

        task = _build_task(metadata, splits)

        self.assertIsInstance(task, LocalEntityTask)
        self.assertEqual(task.hidden_columns(), [("orders", "future_value")])
        table = task.get_table("train")
        self.assertEqual(table.fkey_col_to_pkey_table, {"user_id": "users"})
        self.assertEqual(table.time_col, "timestamp")

    def test_builds_recommendation_task_from_catalog(self) -> None:
        splits = {
            "train": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-01"]),
                    "user_id": [1],
                    "item_id": [[2, 3]],
                }
            )
        }
        metadata = TaskMetadata(
            name="recommend",
            table_name="target_recommend",
            task_type=TaskType.RECOMMENDATION.value,
            entity_table="users",
            entity_column="user_id",
            target_column="item_id",
            time_column="timestamp",
            extra={
                "dst_entity_table": "items",
                "dst_entity_column": "item_id",
                "eval_k": 10,
                "hidden_columns": [],
            },
        )

        task = _build_task(metadata, splits)

        self.assertIsInstance(task, LocalRecommendationTask)
        assert isinstance(task, LocalRecommendationTask)
        self.assertEqual(task.eval_k, 10)
        self.assertEqual(
            task.get_table("train").fkey_col_to_pkey_table,
            {"user_id": "users", "item_id": "items"},
        )

    def test_clears_foreign_keys_dangling_after_time_truncation(self) -> None:
        database = Database(
            {
                "users": Table(
                    pd.DataFrame({"user_id": [0, 1]}),
                    {},
                    pkey_col="user_id",
                ),
                "events": Table(
                    pd.DataFrame({"event_id": [0, 1], "user_id": [0, 2]}),
                    {"user_id": "users"},
                    pkey_col="event_id",
                ),
            }
        )

        corrected = _validate_and_correct_database(database)

        self.assertEqual(corrected.table_dict["events"].df["user_id"].iloc[0], 0)
        self.assertTrue(pd.isna(corrected.table_dict["events"].df["user_id"].iloc[1]))


class TextEmbedderTest(unittest.TestCase):
    def test_normalizes_missing_and_non_string_values(self) -> None:
        self.assertEqual(_normalize_text(None), "")
        self.assertEqual(_normalize_text(float("nan")), "")
        self.assertEqual(_normalize_text("hello"), "hello")
        self.assertEqual(_normalize_text(42), "42")


class TelemetryTest(unittest.TestCase):
    def test_records_phases_decorators_operations_and_resources(self) -> None:
        recorder = TelemetryRecorder(sample_interval_s=0.001)

        @timed("decorated")
        def measured() -> int:
            time.sleep(0.002)
            return 7

        with recorder.activate():
            self.assertEqual(measured(), 7)
            with recorder.phase("explicit"), recorder.operation("batch"):
                time.sleep(0.002)

        report = recorder.report()
        self.assertEqual(
            [phase["name"] for phase in report["phases"]],
            ["decorated", "explicit"],
        )
        self.assertEqual(report["operations"]["batch"]["count"], 1)
        self.assertGreater(report["operations"]["batch"]["total_s"], 0)
        self.assertGreater(report["overall"]["duration_s"], 0)
        self.assertGreater(report["overall"]["peak"]["rss_mb"], 0)
        self.assertIn("torch", report["environment"])


class BenchmarkTest(unittest.TestCase):
    def test_discovers_sorted_filtered_cases(self) -> None:
        class Reader:
            def __init__(self, tasks: tuple[str, ...]) -> None:
                self.tasks = tasks

            def task_metadata(self) -> dict[str, object]:
                return {task: object() for task in self.tasks}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rel-b.sqlite").touch()
            (root / "rel-a.sqlite").touch()

            def reader_factory(_: str, path: Path) -> Reader:
                tasks = ("task-z", "task-a") if path.stem == "rel-a" else ("task-b",)
                return Reader(tasks)

            with patch(
                "relconnector.pipeline.benchmark.create_reader",
                side_effect=reader_factory,
            ):
                cases = discover_cases(
                    root,
                    datasets={"rel-a"},
                    tasks={"task-z"},
                )

        self.assertEqual(cases, [("rel-a", "task-z")])

    def test_resume_skips_only_successful_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            rows = [
                {"status": "ok", "dataset": "rel-a", "task": "task-a"},
                {"status": "error", "dataset": "rel-a", "task": "task-b"},
            ]
            output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            completed = _prepare_output(output, resume=True, overwrite=False)

            self.assertEqual(completed, {("rel-a", "task-a")})
            with self.assertRaises(FileExistsError):
                _prepare_output(output, resume=False, overwrite=False)


if __name__ == "__main__":
    unittest.main()
