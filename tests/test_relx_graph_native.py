"""Native graph construction from relational Arrow batches."""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from relx.config import GraphConfig
from relx.conn import connect
from relx.core import TableRef
from relx.preprocess import discover_schema
from relx.preprocess.graph_builder import build_graph
from relx.task import TaskSpec, TaskType, validate_task


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native")
    and importlib.util.find_spec("pyarrow")
    and importlib.util.find_spec("connectorx"),
    "native build, pyarrow, and connectorx needed",
)
class NativeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / "graph.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.executescript(
                """
                CREATE TABLE users (
                    region TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    PRIMARY KEY (region, user_id)
                );
                CREATE TABLE events (
                    event_id TEXT PRIMARY KEY,
                    region TEXT,
                    user_id INTEGER,
                    event_time TEXT NOT NULL,
                    FOREIGN KEY (region, user_id)
                        REFERENCES users(region, user_id)
                );
                CREATE TABLE labels (
                    sample_id INTEGER PRIMARY KEY,
                    event_id TEXT REFERENCES events(event_id),
                    target INTEGER,
                    prediction_at TEXT
                );
                CREATE TABLE other_labels (
                    sample_id INTEGER PRIMARY KEY,
                    event_id TEXT REFERENCES events(event_id),
                    target INTEGER
                );
                INSERT INTO users VALUES ('eu', 1), ('us', 2);
                INSERT INTO events VALUES
                    ('e1', 'eu', 1, '2020-01-01T00:00:00'),
                    ('e2', 'eu', 1, '2020-01-02T00:00:00'),
                    ('e3', NULL, NULL, '2020-01-03T00:00:00'),
                    ('e4', 'xx', 9, '2020-01-04T00:00:00');
                INSERT INTO labels VALUES (1, 'e1', 1, '2020-01-02T00:00:00');
                INSERT INTO other_labels VALUES (2, 'e2', 0);
                """
            )

    def test_composite_foreign_key_and_excluded_labels(self) -> None:
        import pyarrow as pa

        task_spec = TaskSpec(
            "demo", TableRef("labels"), TaskType.BINARY_CLASSIFICATION,
            "event_id", "target",
        )
        for reader_kind in ("connectorx", "pandas"):
            with self.subTest(reader=reader_kind):
                with connect(self.database, reader=reader_kind) as reader:
                    schema = discover_schema(reader)
                    task = validate_task(task_spec, schema)
                    graph = build_graph(
                        reader, schema, task,
                        GraphConfig(exclude_tables=(TableRef("other_labels"),)),
                        batch_size=2,
                    )
                self.assertEqual(graph.node_counts, {"events": 4, "users": 2})
                name = "events:0:region,user_id->users"
                forward = graph.relation(name)
                reverse = graph.relation("rev:" + name)
                np.testing.assert_array_equal(forward["colptr"], [0, 2, 2])
                np.testing.assert_array_equal(forward["row"], [0, 1])
                np.testing.assert_array_equal(reverse["colptr"], [0, 1, 2, 2, 2])
                np.testing.assert_array_equal(reverse["row"], [0, 0])
                self.assertEqual(forward["null_keys"], 1)
                self.assertEqual(forward["dangling_keys"], 1)
                ids = graph.lookup_nodes(
                    "users", ["region", "user_id"],
                    pa.record_batch({"region": ["us", "eu", "xx"], "user_id": [2, 1, 9]}),
                )
                np.testing.assert_array_equal(ids, [1, 0, -1])

    def test_connectorx_exposes_arrow_without_pandas(self) -> None:
        import pyarrow as pa

        with connect(self.database, reader="connectorx") as reader:
            batches = list(
                reader.iter_arrow_query(
                    'SELECT event_id FROM events ORDER BY event_id', batch_size=2
                )
            )
        self.assertEqual([batch.num_rows for batch in batches], [2, 2])
        self.assertTrue(all(isinstance(batch, pa.RecordBatch) for batch in batches))

    def test_temporal_edges_respect_prediction_boundary(self) -> None:
        task_spec = TaskSpec(
            "demo", TableRef("labels"), TaskType.BINARY_CLASSIFICATION,
            "event_id", "target", time_column="prediction_at",
        )
        cutoff = int(datetime(2020, 1, 2, tzinfo=timezone.utc).timestamp() * 1_000_000)
        base = {
            "exclude_tables": (TableRef("other_labels"),),
            "time_columns": ((TableRef("events"), "event_time"),),
            "static_tables": (TableRef("users"),),
        }
        relation = "events:0:region,user_id->users"
        boundaries: tuple[
            tuple[Literal["strict_before", "at_or_before"], list[int]], ...
        ] = (
            ("strict_before", [0]),
            ("at_or_before", [0, 1]),
        )
        for reader_kind in ("connectorx", "pandas"):
            for boundary, expected in boundaries:
                with self.subTest(reader=reader_kind, boundary=boundary):
                    with connect(self.database, reader=reader_kind) as reader:
                        schema = discover_schema(reader)
                        task = validate_task(task_spec, schema)
                        graph = build_graph(
                            reader, schema, task,
                            GraphConfig(**base, time_boundary=boundary),
                            batch_size=2,
                        )
                    self.assertEqual(graph.visible_neighbors(relation, 0, cutoff), expected)
                    self.assertEqual(
                        graph.visible_neighbors("rev:" + relation, 1, cutoff),
                        [] if boundary == "strict_before" else [0],
                    )
                    self.assertEqual(graph.node_times("users")[0], np.iinfo(np.int64).min)
                    self.assertEqual(graph.relation(relation)["edge_time"][1], cutoff)
                    if boundary == "at_or_before":
                        destinations = [0] * 8
                        cutoffs = [cutoff] * 8
                        self.assertEqual(
                            graph.sample_neighbors(
                                relation, destinations, cutoffs, 1, 7, 1
                            ),
                            graph.sample_neighbors(
                                relation, destinations, cutoffs, 1, 7, 4
                            ),
                        )

    def test_default_boundary_and_native_batch_assembly(self) -> None:
        task_spec = TaskSpec(
            "demo", TableRef("labels"), TaskType.BINARY_CLASSIFICATION,
            "event_id", "target", time_column="prediction_at",
        )
        with connect(self.database) as reader:
            schema = discover_schema(reader)
            task = validate_task(task_spec, schema)
            graph = build_graph(
                reader, schema, task,
                GraphConfig(
                    exclude_tables=(TableRef("other_labels"),),
                    time_columns=((TableRef("events"), "event_time"),),
                    static_tables=(TableRef("users"),),
                ),
            )
        cutoff = int(datetime(2020, 1, 2, tzinfo=timezone.utc).timestamp() * 1_000_000)
        relation = "events:0:region,user_id->users"
        self.assertEqual(graph.visible_neighbors(relation, 0, cutoff), [0, 1])
        first = graph.sample_one_hop("users", [0, 1], [cutoff, cutoff], -1, 7, 1)
        threaded = graph.sample_one_hop("users", [0, 1], [cutoff, cutoff], -1, 7, 4)
        for section in ("node_ids", "node_seed_index", "edge_index"):
            for name, values in first[section].items():
                np.testing.assert_array_equal(values, threaded[section][name])
        np.testing.assert_array_equal(first["node_ids"]["events"], [0, 1])
        np.testing.assert_array_equal(first["node_seed_index"]["events"], [0, 0])
        np.testing.assert_array_equal(first["edge_index"][relation], [[0, 1], [0, 0]])

    def test_temporal_tables_need_explicit_time_rules(self) -> None:
        task_spec = TaskSpec(
            "demo", TableRef("labels"), TaskType.BINARY_CLASSIFICATION,
            "event_id", "target", time_column="prediction_at",
        )
        with connect(self.database) as reader:
            schema = discover_schema(reader)
            task = validate_task(task_spec, schema)
            with self.assertRaisesRegex(ValueError, "graph.time_columns or graph.static_tables"):
                build_graph(
                    reader, schema, task,
                    GraphConfig(exclude_tables=(TableRef("other_labels"),)),
                )
            with self.assertRaisesRegex(ValueError, "Unknown graph time column"):
                build_graph(
                    reader, schema, task,
                    GraphConfig(
                        exclude_tables=(TableRef("other_labels"),),
                        time_columns=((TableRef("events"), "missing"),),
                        static_tables=(TableRef("users"),),
                    ),
                )
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_graph(
                    reader, schema, task,
                    GraphConfig(
                        exclude_tables=(TableRef("other_labels"),),
                        time_columns=((TableRef("events"), "event_time"),),
                        static_tables=(TableRef("events"), TableRef("users")),
                    ),
                )

    def test_time_normalization_uses_configured_timezone(self) -> None:
        import pyarrow as pa

        from relx.preprocess.graph_time import normalize_time_batch

        naive = pa.record_batch({"event_time": ["2020-01-01T08:00:00"]})
        aware = pa.record_batch({"event_time": ["2020-01-01T00:00:00+00:00"]})
        local_us = normalize_time_batch(naive, "event_time", "Asia/Shanghai")
        utc_us = normalize_time_batch(aware, "event_time", "UTC")
        self.assertEqual(
            local_us.column("__relx_time_us").to_pylist(),
            utc_us.column("__relx_time_us").to_pylist(),
        )
        missing = normalize_time_batch(
            pa.record_batch({"event_time": [None, "2020-01-01T00:00:00"]}),
            "event_time", "UTC",
        ).column("__relx_time_us").to_pylist()
        self.assertEqual(missing[0], -9_223_372_037_000_000)
        self.assertLess(missing[0], missing[1])

    def test_native_keys_match_across_integer_widths_and_reject_duplicates(self) -> None:
        import pyarrow as pa

        from relx._relx_native import GraphBuilder

        builder = GraphBuilder()
        builder.register_table("parent")
        builder.register_table("child")
        builder.add_nodes(
            "parent", ["id"],
            pa.record_batch({"id": pa.array([5], type=pa.int64())}),
        )
        builder.add_nodes(
            "child", ["id"],
            pa.record_batch({"id": pa.array([10], type=pa.int32())}),
        )
        builder.register_relation("child:parent", "child", "parent")
        builder.add_edges(
            "child:parent", "child", "parent", ["id"], ["parent_id"],
            pa.record_batch({
                "id": pa.array([10], type=pa.int32()),
                "parent_id": pa.array([5.0], type=pa.float64()),
            }),
        )
        np.testing.assert_array_equal(builder.finish().relation("child:parent")["row"], [0])

        duplicate = GraphBuilder()
        duplicate.register_table("bad")
        with self.assertRaisesRegex(ValueError, "duplicate identity key"):
            duplicate.add_nodes("bad", ["id"], pa.record_batch({"id": [1, 1]}))


if __name__ == "__main__":
    unittest.main()
