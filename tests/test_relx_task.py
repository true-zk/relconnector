"""Task configuration is independent of database catalog tables."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relx.task import TaskType, load_task


class TaskConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "task.yaml"

    def test_loads_entity_task_with_explicit_hidden_columns(self) -> None:
        self.path.write_text(
            """name: predict-label
table_name: labels
task_type: binary_classification
entity_table: users
entity_column: user_id
target_column: label
hidden_columns:
  - [users, label]
  - [users, label]
"""
        )
        task = load_task(self.path)
        self.assertEqual(task.task_type, TaskType.BINARY_CLASSIFICATION)
        self.assertEqual(task.hidden_columns, (("users", "label"),))
        self.assertFalse(task.is_recommendation)

    def test_loads_recommendation_task(self) -> None:
        self.path.write_text(
            """name: recommend
table_name: links
task_type: recommendation
entity_table: users
entity_column: user_id
dst_entity_table: items
dst_entity_column: item_id
hidden_columns: []
"""
        )
        task = load_task(self.path)
        self.assertTrue(task.is_recommendation)
        self.assertEqual(task.dst_entity_table, "items")

    def test_rejects_missing_hidden_column_policy(self) -> None:
        self.path.write_text(
            """name: predict-label
table_name: labels
task_type: regression
entity_table: users
entity_column: user_id
target_column: label
"""
        )
        with self.assertRaisesRegex(ValueError, "hidden_columns"):
            load_task(self.path)

    def test_rejects_missing_recommendation_destination(self) -> None:
        self.path.write_text(
            """name: recommend
table_name: links
task_type: recommendation
entity_table: users
entity_column: user_id
hidden_columns: []
"""
        )
        with self.assertRaisesRegex(ValueError, "destination"):
            load_task(self.path)


if __name__ == "__main__":
    unittest.main()
