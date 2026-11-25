"""Compare native temporal neighbors with independent SQL on rel-f1."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml
from sqlalchemy import text

import relx
from relx.batch import FeatureStore
from relx.config import GraphConfig
from relx.conn import connect
from relx.core import DatabaseSchema, TableRef
from relx.preprocess import build_graph, discover_schema
from relx.task import SqlEntitySeedSource, TaskSpec, TaskType, validate_task

_DATABASE = (
    Path(__file__).resolve().parents[1] / "exps/data/relbench/rel-f1.sqlite"
)
_SAMPLES = (0, 200, 9376, 11411, 11977)
_TIME_TABLES = (
    "constructor_results",
    "constructor_standings",
    "qualifying",
    "races",
    "results",
    "standings",
)
_STATIC_TABLES = ("circuits", "constructors", "drivers")
_BOUNDARIES: tuple[
    tuple[Literal["strict_before", "at_or_before"], str], ...
] = (("strict_before", "<"), ("at_or_before", "<="))

if TYPE_CHECKING:
    from relx._relx_native import GraphIndex


@unittest.skipUnless(
    _DATABASE.is_file()
    and importlib.util.find_spec("relx._relx_native")
    and importlib.util.find_spec("connectorx")
    and importlib.util.find_spec("pyarrow"),
    "local rel-f1 SQLite data and native extension needed",
)
class RelF1GraphCompareTest(unittest.TestCase):
    def test_connectorx_and_pandas_build_identical_graphs(self) -> None:
        graphs = []
        for reader_kind in ("connectorx", "pandas"):
            with connect(_DATABASE, reader=reader_kind) as reader:
                schema = discover_schema(reader)
                task = validate_task(
                    TaskSpec(
                        "driver-dnf",
                        TableRef("target_driver_dnf"),
                        TaskType.BINARY_CLASSIFICATION,
                        "driverId",
                        "did_not_finish",
                        time_column="date",
                    ),
                    schema,
                )
                excluded = tuple(
                    ref
                    for ref in schema.tables
                    if ref.table_name.startswith("_relconnector_")
                    or (ref.table_name.startswith("target_") and ref != task.spec.table)
                )
                graph = build_graph(
                    reader,
                    schema,
                    task,
                    GraphConfig(
                        exclude_tables=excluded,
                        time_columns=tuple(
                            (TableRef(name), "date") for name in _TIME_TABLES
                        ),
                        static_tables=tuple(TableRef(name) for name in _STATIC_TABLES),
                    ),
                    batch_size=10_000,
                )
                graphs.append(graph)
        connectorx_graph, pandas_graph = graphs
        self.assertEqual(connectorx_graph.node_counts, pandas_graph.node_counts)
        self.assertEqual(
            connectorx_graph.relation_names(), pandas_graph.relation_names()
        )
        for table_name in connectorx_graph.node_counts:
            np.testing.assert_array_equal(
                connectorx_graph.node_times(table_name),
                pandas_graph.node_times(table_name),
            )
        for relation_name in connectorx_graph.relation_names():
            first = connectorx_graph.relation(relation_name)
            second = pandas_graph.relation(relation_name)
            for field in ("colptr", "row", "edge_time"):
                np.testing.assert_array_equal(first[field], second[field])

    @unittest.skipUnless(
        importlib.util.find_spec("torch_frame"),
        "torch-frame needed for automatic feature inference",
    )
    def test_full_yaml_preparation(self) -> None:
        config = {
            "database": {"url": str(_DATABASE), "reader": "connectorx"},
            "task": {
                "name": "driver-dnf",
                "type": "binary_classification",
                "table": "target_driver_dnf",
                "entity_column": "driverId",
                "label_column": "did_not_finish",
                "time_column": "date",
                "split": {
                    "type": "column",
                    "column": "split",
                    "values": {"train": "train", "validation": "val", "test": "test"},
                },
            },
            "graph": {
                "exclude_tables": [
                    "_relconnector_columns",
                    "_relconnector_foreign_keys",
                    "_relconnector_metadata",
                    "_relconnector_tables",
                    "_relconnector_tasks",
                    "target_driver_circuit_compete",
                    "target_driver_position",
                    "target_driver_top3",
                    "target_qualifying_position",
                    "target_results_position",
                ],
                "time_columns": {name: "date" for name in _TIME_TABLES},
                "static_tables": list(_STATIC_TABLES),
            },
            "features": {"sample_limit": 1_000},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            prepared = relx.prepare(path)
        self.assertEqual(prepared.report.table_count, 9)
        self.assertEqual(prepared.report.relation_count, 26)
        self.assertEqual(prepared.report.feature_count, 45)
        assert prepared.report.task_data is not None
        self.assertEqual(
            (
                prepared.report.task_data.train_rows,
                prepared.report.task_data.validation_rows,
                prepared.report.task_data.test_rows,
            ),
            (11_411, 566, 702),
        )
        with connect(_DATABASE, reader="connectorx") as reader:
            training = SqlEntitySeedSource(
                reader,
                prepared.task,
                prepared.graph,
                split="train",
                timezone=prepared.config.graph.timezone,
                batch_size=128,
                shuffle_seed=prepared.config.training.seed,
            )
            store = FeatureStore(
                prepared, reader, train_cutoff_us=training.max_time_us
            )
        self.assertEqual(len(store.states), 45)

    def test_task_seed_neighbors_match_sql(self) -> None:
        with connect(_DATABASE, reader="connectorx") as reader:
            schema = discover_schema(reader)
            task = validate_task(
                TaskSpec(
                    "driver-dnf",
                    TableRef("target_driver_dnf"),
                    TaskType.BINARY_CLASSIFICATION,
                    "driverId",
                    "did_not_finish",
                    time_column="date",
                ),
                schema,
            )
            excluded = tuple(
                ref
                for ref in schema.tables
                if ref.table_name.startswith("_relconnector_")
                or (
                    ref.table_name.startswith("target_")
                    and ref != task.spec.table
                )
            )
            samples = reader.read_query(
                text(
                    "SELECT __target_row_id__, driverId, date, split "
                    "FROM target_driver_dnf "
                    "WHERE __target_row_id__ IN (0, 200, 9376, 11411, 11977)"
                )
            )
            self.assertEqual(set(samples["__target_row_id__"]), set(_SAMPLES))
            self.assertEqual(
                set(samples["split"]), {"train", "val", "test"}
            )

            for boundary, operator in _BOUNDARIES:
                with self.subTest(boundary=boundary):
                    graph = build_graph(
                        reader,
                        schema,
                        task,
                        GraphConfig(
                            exclude_tables=excluded,
                            time_columns=tuple(
                                (TableRef(name), "date") for name in _TIME_TABLES
                            ),
                            static_tables=tuple(
                                TableRef(name) for name in _STATIC_TABLES
                            ),
                            time_boundary=boundary,
                        ),
                        batch_size=10_000,
                    )
                    self.assertEqual(len(graph.node_counts), 9)
                    self.assertEqual(len(graph.relation_names()), 26)
                    for row in samples.itertuples(index=False):
                        sample_id, driver_id, cutoff, _ = row
                        driver_id = int(driver_id)
                        cutoff_us = _timestamp_us(cutoff)
                        driver_node = _node_id(graph, "drivers", "driverId", driver_id)
                        for source in ("results", "qualifying", "standings"):
                            relation = _relation(schema, source, "drivers")
                            actual = set(
                                graph.visible_neighbors(
                                    relation, driver_node, cutoff_us
                                )
                            )
                            identity = schema.tables[TableRef(source)].identity_key
                            assert identity is not None
                            expected_rows = reader.read_query(
                                text(
                                    f"SELECT {identity[0]} FROM {source} "
                                    f"WHERE driverId = :driver AND date {operator} :cutoff"
                                ),
                                params={"driver": driver_id, "cutoff": cutoff},
                            )
                            expected = {
                                _node_id(graph, source, identity[0], key)
                                for key in expected_rows.iloc[:, 0]
                            }
                            self.assertEqual(
                                actual,
                                expected,
                                (boundary, sample_id, source),
                            )

                    # Qualifying on May 28 is available, but its race on May 29
                    # must follow the selected equality boundary.
                    sample = samples.loc[samples["__target_row_id__"] == 9376].iloc[0]
                    cutoff = sample["date"]
                    qualifying = _node_id(graph, "qualifying", "qualifyId", 80)
                    actual = set(
                        graph.visible_neighbors(
                            "rev:" + _relation(schema, "qualifying", "races"),
                            qualifying,
                            _timestamp_us(cutoff),
                        )
                    )
                    expected_rows = reader.read_query(
                        text(
                            "SELECT races.raceId FROM qualifying "
                            "JOIN races ON races.raceId = qualifying.raceId "
                            "WHERE qualifying.qualifyId = 80 "
                            f"AND qualifying.date {operator} :cutoff "
                            f"AND races.date {operator} :cutoff"
                        ),
                        params={"cutoff": cutoff},
                    )
                    expected = {
                        _node_id(graph, "races", "raceId", key)
                        for key in expected_rows.iloc[:, 0]
                    }
                    self.assertEqual(actual, expected, (boundary, "qualifying->races"))
                    self.assertEqual(len(actual), int(boundary == "at_or_before"))


def _timestamp_us(value: str) -> int:
    stamp = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1_000_000)


def _node_id(graph: GraphIndex, table: str, key: str, value: object) -> int:
    import pyarrow as pa

    batch = pa.record_batch({key: [value]})
    node_id = int(graph.lookup_nodes(table, [key], batch)[0])
    if node_id < 0:
        raise AssertionError(f"Graph has no node for {table}.{key}={value!r}")
    return node_id


def _relation(schema: DatabaseSchema, source: str, target: str) -> str:
    foreign_keys = schema.tables[TableRef(source)].foreign_keys
    for index, foreign_key in enumerate(foreign_keys):
        if foreign_key.target_table == TableRef(target):
            columns = ",".join(foreign_key.source_columns)
            return f"{source}:{index}:{columns}->{target}"
    raise AssertionError(f"No relation from {source} to {target}")


if __name__ == "__main__":
    unittest.main()
