"""Global YAML parsing and task validation against a real SQLite schema."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import yaml

from relx.config import load_config
from relx.conn import connect
from relx.core import TableRef
from relx.preprocess import discover_schema, select_graph_tables
from relx.task import ColumnSplit, DefaultSplit, SqlQuerySplit, TaskType, validate_task


class TaskConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.database = self.directory / "demo.sqlite"
        self.config_file = self.directory / "run.yaml"
        with sqlite3.connect(self.database) as connection:
            connection.executescript(
                """
                CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
                CREATE TABLE labels (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    target INTEGER,
                    prediction_at TEXT NOT NULL,
                    split TEXT NOT NULL
                );
                CREATE TABLE no_fk_labels (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    target INTEGER,
                    prediction_at TEXT NOT NULL,
                    split TEXT NOT NULL
                );
                CREATE TABLE other_users (id INTEGER PRIMARY KEY);
                CREATE TABLE ambiguous_labels (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    target INTEGER,
                    prediction_at TEXT,
                    split TEXT,
                    FOREIGN KEY (user_id) REFERENCES other_users(id)
                );
                CREATE TABLE items (id INTEGER PRIMARY KEY);
                CREATE TABLE recommend_labels (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    targets TEXT,
                    split TEXT
                );
                INSERT INTO users VALUES (1, 'Ada'), (2, 'Bob');
                INSERT INTO other_users VALUES (1), (2);
                INSERT INTO items VALUES (1), (2);
                INSERT INTO labels VALUES
                    (1, 1, 0, '2020-01-01', 'train'),
                    (2, 2, 1, '2020-02-01', 'train'),
                    (3, 1, 0, '2020-03-01', 'val'),
                    (4, 2, 1, '2020-04-01', 'val'),
                    (5, 1, NULL, '2020-05-01', 'test'),
                    (6, 2, NULL, '2020-06-01', 'test');
                INSERT INTO no_fk_labels SELECT * FROM labels;
                INSERT INTO ambiguous_labels SELECT * FROM labels;
                INSERT INTO recommend_labels VALUES
                    (1, 1, '[1]', 'train'),
                    (2, 2, '[2]', 'train'),
                    (3, 1, '[1, 2]', 'val'),
                    (4, 2, NULL, 'test');
                """
            )

    def write_config(self, *, task: dict | None = None, **parts: object) -> None:
        config = {
            "database": {"url": "demo.sqlite"},
            "task": task
            or {
                "name": "churn",
                "type": "binary_classification",
                "table": "labels",
                "entity_column": "user_id",
                "label_column": "target",
                "time_column": "prediction_at",
                "split": {
                    "type": "column",
                    "column": "split",
                    "values": {"train": "train", "validation": "val", "test": "test"},
                },
            },
            **parts,
        }
        self.config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    def validated(self):
        config = load_config(self.config_file)
        with connect(config.database.location, reader=config.database.reader) as reader:
            schema = discover_schema(reader, overrides=config.schema)
            return validate_task(
                config.task, schema, hidden_columns=config.hidden_columns, reader=reader
            )

    def test_parses_global_sections_and_validates_column_split(self) -> None:
        self.write_config(
            hidden_columns=[["users", "name"]],
            model={"name": "graph_model", "params": {"width": 64}},
            training={"batch_size": 32, "epochs": 5, "lr": 0.01, "seed": 7},
            report={"directory": "reports/churn"},
            features={
                "columns": [
                    {"table": "users", "column": "name", "semantic_type": "text"}
                ]
            },
        )
        config = load_config(self.config_file)
        self.assertEqual(config.task.task_type, TaskType.BINARY_CLASSIFICATION)
        self.assertIsInstance(config.task.split, ColumnSplit)
        self.assertEqual(config.training.epochs, 5)
        self.assertEqual(config.model.params["width"], 64)
        self.assertEqual(config.report.directory, self.directory / "reports/churn")
        self.assertEqual(config.hidden_columns, ((TableRef("users"), "name"),))
        validated = self.validated()
        self.assertEqual(validated.entity_table, TableRef("users"))
        self.assertEqual(validated.entity_key, ("id",))
        self.assertEqual(validated.sample_key, ("id",))
        report = validated.data_report
        assert report is not None
        self.assertEqual(
            (
                report.train_rows,
                report.validation_rows,
                report.test_rows,
            ),
            (2, 2, 2),
        )
        self.assertFalse(report.test_labels_available)

    def test_default_split_uses_unlabeled_test_and_latest_validation(self) -> None:
        self.write_config()
        config = yaml.safe_load(self.config_file.read_text())
        del config["task"]["split"]
        self.config_file.write_text(yaml.safe_dump(config))
        self.assertIsInstance(load_config(self.config_file).task.split, DefaultSplit)
        validated = self.validated()
        split_by_key = validated.split_by_key
        report = validated.data_report
        assert split_by_key is not None
        assert report is not None
        self.assertEqual(split_by_key[(4,)], "validation")
        self.assertEqual(split_by_key[(5,)], "test")
        self.assertEqual(report.train_rows, 3)

    def test_graph_excludes_other_label_tables_and_current_task_table(self) -> None:
        self.write_config(graph={"exclude_tables": ["no_fk_labels", "recommend_labels"]})
        config = load_config(self.config_file)
        with connect(config.database.location) as reader:
            schema = discover_schema(reader)
            task = validate_task(config.task, schema)
        selected = select_graph_tables(schema, task, config.graph)
        self.assertIn(TableRef("users"), selected)
        self.assertNotIn(TableRef("labels"), selected)
        self.assertNotIn(TableRef("no_fk_labels"), selected)
        self.assertNotIn(TableRef("recommend_labels"), selected)

    def test_graph_exclusions_reject_unknown_and_entity_tables(self) -> None:
        self.write_config(graph={"exclude_tables": ["missing"]})
        config = load_config(self.config_file)
        with connect(config.database.location) as reader:
            schema = discover_schema(reader)
            task = validate_task(config.task, schema)
        with self.assertRaisesRegex(ValueError, "unknown tables"):
            select_graph_tables(schema, task, config.graph)

        self.write_config(graph={"exclude_tables": ["users"]})
        config = load_config(self.config_file)
        with self.assertRaisesRegex(ValueError, "cannot remove task entity"):
            select_graph_tables(schema, task, config.graph)

    def test_graph_time_config_parses_and_rejects_conflicts(self) -> None:
        self.write_config(graph={"static_tables": ["users"]})
        self.assertEqual(load_config(self.config_file).graph.time_boundary, "at_or_before")

        self.write_config(graph={
            "exclude_tables": ["recommend_labels"],
            "time_columns": {"no_fk_labels": "prediction_at"},
            "static_tables": ["users"],
            "timezone": "Asia/Shanghai",
            "time_boundary": "at_or_before",
        })
        graph = load_config(self.config_file).graph
        self.assertEqual(graph.time_columns, ((TableRef("no_fk_labels"), "prediction_at"),))
        self.assertEqual(graph.static_tables, (TableRef("users"),))
        self.assertEqual(graph.timezone, "Asia/Shanghai")
        self.assertEqual(graph.time_boundary, "at_or_before")

        self.write_config(graph={
            "time_columns": {"users": "created_at"},
            "static_tables": ["users"],
        })
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_config(self.config_file)

        self.write_config(graph={"timezone": "Invalid/Timezone"})
        with self.assertRaisesRegex(ValueError, "Unknown graph.timezone"):
            load_config(self.config_file)

    def test_sql_query_can_select_only_test_rows(self) -> None:
        self.write_config()
        config = yaml.safe_load(self.config_file.read_text())
        config["task"]["split"] = {
            "type": "sql_query",
            "test": "SELECT\n  id\nFROM labels\nWHERE id >= 5",
        }
        self.config_file.write_text(yaml.safe_dump(config))
        self.assertIsInstance(load_config(self.config_file).task.split, SqlQuerySplit)
        validated = self.validated()
        report = validated.data_report
        assert report is not None
        self.assertEqual(report.test_rows, 2)
        self.assertEqual(report.validation_rows, 1)

    def test_sql_query_rejects_overlap(self) -> None:
        self.write_config()
        config = yaml.safe_load(self.config_file.read_text())
        config["task"]["split"] = {
            "type": "sql_query",
            "train": "SELECT id FROM labels WHERE id <= 2",
            "validation": "SELECT id FROM labels WHERE id BETWEEN 2 AND 4",
            "test": "SELECT id FROM labels WHERE id >= 5",
        }
        self.config_file.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "multiple split queries"):
            self.validated()

    def test_static_default_split_is_reproducible(self) -> None:
        self.write_config()
        config = yaml.safe_load(self.config_file.read_text())
        del config["task"]["split"]
        del config["task"]["time_column"]
        self.config_file.write_text(yaml.safe_dump(config))
        first = self.validated()
        second = self.validated()
        self.assertEqual(first.split_by_key, second.split_by_key)
        report = first.data_report
        assert report is not None
        self.assertEqual(report.test_rows, 2)

    def test_ambiguous_foreign_key_requires_entity_table(self) -> None:
        self.write_config()
        config = yaml.safe_load(self.config_file.read_text())
        config["task"]["table"] = "ambiguous_labels"
        self.config_file.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "exactly one foreign key"):
            self.validated()
        config["task"]["entity_table"] = "users"
        self.config_file.write_text(yaml.safe_dump(config))
        self.assertEqual(self.validated().entity_table, TableRef("users"))

    def test_recommendation_checks_destination_ids(self) -> None:
        self.write_config()
        config = yaml.safe_load(self.config_file.read_text())
        config["task"].update(
            {
                "name": "recommend",
                "type": "recommendation",
                "table": "recommend_labels",
                "label_column": "targets",
                "destination": {"table": "items", "key": "id"},
            }
        )
        del config["task"]["time_column"]
        self.config_file.write_text(yaml.safe_dump(config))
        self.assertEqual(self.validated().destination_table, TableRef("items"))
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE recommend_labels SET targets='[99]' WHERE id=3")
        with self.assertRaisesRegex(ValueError, "unknown destination IDs"):
            self.validated()

    def test_missing_fk_can_be_supplied_in_schema_section(self) -> None:
        self.write_config()
        config = yaml.safe_load(self.config_file.read_text())
        config["task"]["table"] = "no_fk_labels"
        self.config_file.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "exactly one foreign key"):
            self.validated()
        config["schema"] = {
            "foreign_keys": [
                {
                    "source_table": "no_fk_labels",
                    "source_columns": ["user_id"],
                    "target_table": "users",
                    "target_columns": ["id"],
                }
            ]
        }
        self.config_file.write_text(yaml.safe_dump(config))
        self.assertEqual(self.validated().entity_table, TableRef("users"))

    def test_rejects_unknown_config_and_missing_environment(self) -> None:
        self.write_config(training={"batch_size": 0})
        with self.assertRaisesRegex(ValueError, "training.batch_size"):
            load_config(self.config_file)
        self.write_config(unexpected={})
        with self.assertRaisesRegex(ValueError, "Unknown fields"):
            load_config(self.config_file)
        self.write_config(database={"url_env": "RELX_MISSING_TEST_URL"})
        os.environ.pop("RELX_MISSING_TEST_URL", None)
        with self.assertRaisesRegex(ValueError, "RELX_MISSING_TEST_URL"):
            load_config(self.config_file)

    def test_rejects_unknown_split_value_and_orphan_entity(self) -> None:
        self.write_config()
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE labels SET split='other' WHERE id=4")
        with self.assertRaisesRegex(ValueError, "unknown split value"):
            self.validated()
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE labels SET split='val', user_id=999 WHERE id=4")
        with self.assertRaisesRegex(ValueError, "orphan entity keys"):
            self.validated()

    def test_rejects_partially_labeled_test_split(self) -> None:
        self.write_config()
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE labels SET target=1 WHERE id=5")
        with self.assertRaisesRegex(ValueError, "mixes labeled and unlabeled"):
            self.validated()


if __name__ == "__main__":
    unittest.main()
