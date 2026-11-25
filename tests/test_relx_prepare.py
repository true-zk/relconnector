"""End-to-end preparation and model-neutral contract checks."""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from relx import prepare, run
from relx.batch import FeatureStore, OnlineFeatureStore, PreparedBatch
from relx.conn import connect
from relx.core import TableRef
from relx.preprocess.features import (
    FeatureBlock,
    FeatureOutputSpec,
    FeatureProcessor,
    FeatureState,
)
from relx.sampling import EntitySamplePlan, NativeOneHopSampler, SampledSubgraph
from relx.task import BatchKey, SqlEntitySeedSource


class NumericPairProcessor(FeatureProcessor):
    semantic_type = "numerical"

    def transform(self, values: pd.Series, state: FeatureState) -> FeatureBlock:
        numbers = np.asarray(values, dtype=np.float32)
        return FeatureBlock(np.stack((numbers, numbers), axis=1))

    def output_spec(self, params: Mapping[str, object]) -> FeatureOutputSpec:
        return FeatureOutputSpec("float32", width=2)


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native")
    and importlib.util.find_spec("pyarrow")
    and importlib.util.find_spec("torch_frame"),
    "native graph and feature inference dependencies needed",
)
class PrepareTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.database = self.directory / "demo.sqlite"
        self.config_file = self.directory / "run.yaml"
        with sqlite3.connect(self.database) as connection:
            connection.executescript(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY, age REAL, segment TEXT, private_note TEXT
                );
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    amount REAL,
                    event_time TEXT NOT NULL
                );
                CREATE TABLE labels (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    target INTEGER,
                    prediction_at TEXT NOT NULL,
                    split TEXT NOT NULL
                );
                CREATE TABLE other_labels (id INTEGER PRIMARY KEY);
                INSERT INTO users VALUES
                    (1, 20, 'A', 'secret A'), (2, 30, 'B', 'secret B');
                INSERT INTO events VALUES
                    (1, 1, 10, '2020-01-01'),
                    (2, 1, 20, '2020-03-01'),
                    (3, 2, 30, '2020-02-01');
                INSERT INTO labels VALUES
                    (1, 1, 0, '2020-02-01', 'train'),
                    (2, 2, 1, '2020-02-01', 'train'),
                    (3, 1, 0, '2020-03-01', 'val'),
                    (4, 2, 1, '2020-03-01', 'val'),
                    (5, 1, NULL, '2020-04-01', 'test');
                """
            )

    def _write_config(self) -> None:
        config = {
            "database": {"url": str(self.database), "reader": "pandas"},
            "task": {
                "name": "demo",
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
            "graph": {
                "exclude_tables": ["other_labels"],
                "time_columns": {"events": "event_time"},
                "static_tables": ["users"],
            },
            "hidden_columns": [["users", "private_note"]],
            "features": {
                "sample_limit": 2,
                "columns": [
                    {
                        "table": "users",
                        "column": "age",
                        "processor": "double_number",
                    },
                    {
                        "table": "users",
                        "column": "segment",
                        "semantic_type": "categorical",
                    },
                ],
            },
            "training": {"seed": 9, "epochs": 2, "batch_size": 2},
        }
        self.config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_prepare_reports_graph_task_and_feature_plan(self) -> None:
        self._write_config()
        processor = NumericPairProcessor()
        prepared = prepare(
            self.config_file,
            feature_processors={
                "double_number": processor,
                ("events", "amount"): processor,
            },
        )
        self.assertEqual(prepared.report.table_count, 2)
        self.assertEqual(prepared.report.node_counts, {"events": 3, "users": 2})
        self.assertEqual(prepared.report.relation_count, 2)
        self.assertEqual(prepared.report.feature_count, 4)
        assert prepared.report.task_data is not None
        self.assertEqual(prepared.report.task_data.train_rows, 2)
        self.assertEqual(prepared.features.sample_limit, 2)
        self.assertEqual(prepared.features.sample_seed, 9)
        self.assertEqual(
            prepared.features.get(TableRef("users"), "age").output.width, 2
        )
        self.assertEqual(
            prepared.features.get(TableRef("events"), "amount").output.width, 2
        )
        self.assertEqual(
            prepared.features.get(TableRef("events"), "event_time").semantic_type,
            "timestamp",
        )
        with self.assertRaises(KeyError):
            prepared.features.get(TableRef("users"), "private_note")
        self.assertEqual(len(prepared.feature_processors), 2)

    def test_hidden_feature_configuration_is_rejected(self) -> None:
        self._write_config()
        config = yaml.safe_load(self.config_file.read_text(encoding="utf-8"))
        config["features"]["columns"].append(
            {"table": "users", "column": "private_note", "semantic_type": "text"}
        )
        self.config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hidden columns"):
            prepare(
                self.config_file,
                feature_processors={"double_number": NumericPairProcessor()},
            )

    def test_online_run_writes_report_and_predicts(self) -> None:
        self._write_config()
        trained = run(
            self.config_file,
            feature_processors={"double_number": NumericPairProcessor()},
        )
        self.assertEqual(len(trained.report.epochs), 2)
        self.assertEqual(trained.report.metric, "accuracy")
        self.assertTrue(trained.report.report_file.is_file())
        self.assertEqual(len(trained.predict_test()), 1)

    def test_online_features_read_database_again_for_each_batch(self) -> None:
        self._write_config()
        prepared = prepare(
            self.config_file,
            feature_processors={"double_number": NumericPairProcessor()},
        )
        with connect(self.database) as reader:
            source = SqlEntitySeedSource(
                reader, prepared.task, prepared.graph, split="train",
                timezone=prepared.config.graph.timezone, batch_size=2,
                shuffle_seed=9,
            )
            store = OnlineFeatureStore(
                prepared, reader, train_cutoff_us=source.max_time_us
            )
        sampler = NativeOneHopSampler(
            prepared.graph, prepared.schema, fanout=-1
        )
        plan = sampler.sample(next(source.iter_epoch(0)))
        assert isinstance(plan, EntitySamplePlan)
        with patch(
            "relx.conn.pandas.PandasDatabaseReader.read_query",
            side_effect=AssertionError("Python must not read custom batch features"),
        ):
            first = store.fetch(plan)
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE events SET amount = 50 WHERE id = 1")
        second = store.fetch(plan)
        key = (TableRef("events"), "amount")
        self.assertFalse(np.array_equal(
            first.features[key].values, second.features[key].values
        ))
        self.assertIn(3.0, second.features[key].values)

    def test_online_multi_hop_training_uses_sampled_outer_hop(self) -> None:
        self._write_config()
        config = yaml.safe_load(self.config_file.read_text(encoding="utf-8"))
        config["model"] = {
            "name": "multi_hop_baseline",
            "params": {
                "num_neighbors": [2, 2], "channels": 8,
                "workers": 2, "text_mode": "hash",
            },
        }
        config["training"].update({
            "epochs": 1, "feature_workers": 2,
            "feature_native_workers": 2, "prefetch_batches": 2,
        })
        self.config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
        trained = run(
            self.config_file,
            feature_processors={"double_number": NumericPairProcessor()},
        )
        self.assertEqual(trained.report.model_name, "multi_hop_baseline")
        self.assertEqual(len(trained.report.epochs), 1)
        self.assertEqual(len(trained.predict_test()), 1)

    def test_online_rdl_and_relgnn_training(self) -> None:
        self._write_config()
        base = yaml.safe_load(self.config_file.read_text(encoding="utf-8"))
        for name in ("rdl", "relgnn"):
            with self.subTest(model=name):
                config = dict(base)
                config["model"] = {
                    "name": name,
                    "params": {
                        "num_neighbors": [2, 2], "channels": 8,
                        "workers": 2, "text_mode": "hash",
                    },
                }
                config["training"] = {
                    **base["training"], "epochs": 1,
                    "prefetch_bytes": 1024,
                }
                self.config_file.write_text(
                    yaml.safe_dump(config), encoding="utf-8"
                )
                trained = run(
                    self.config_file,
                    feature_processors={"double_number": NumericPairProcessor()},
                )
                self.assertEqual(trained.report.model_name, name)
                self.assertEqual(len(trained.report.epochs), 1)

    def test_seed_source_and_threaded_sampler_respect_time(self) -> None:
        self._write_config()
        prepared = prepare(
            self.config_file,
            feature_processors={"double_number": NumericPairProcessor()},
        )
        with connect(self.database) as reader:
            source = SqlEntitySeedSource(
                reader,
                prepared.task,
                prepared.graph,
                split="validation",
                timezone="UTC",
                batch_size=2,
                shuffle_seed=9,
            )
            seeds = next(source.iter_epoch(0))
            training = SqlEntitySeedSource(
                reader,
                prepared.task,
                prepared.graph,
                split="train",
                timezone="UTC",
                batch_size=2,
                shuffle_seed=9,
            )
            store = FeatureStore(
                prepared, reader, train_cutoff_us=training.max_time_us
            )
        self.assertEqual(
            store.states[(TableRef("events"), "amount")].values["mean"], 20.0
        )
        self.assertEqual(seeds.sample_keys, ((3,), (4,)))
        first = NativeOneHopSampler(
            prepared.graph, prepared.schema, fanout=-1, workers=1
        ).sample(seeds)
        threaded = NativeOneHopSampler(
            prepared.graph, prepared.schema, fanout=-1, workers=4
        ).sample(seeds)
        assert isinstance(first, EntitySamplePlan)
        assert isinstance(threaded, EntitySamplePlan)
        first.subgraph.validate()
        threaded.subgraph.validate()
        relation = next(iter(first.subgraph.edge_index))
        self.assertEqual(relation.name, "events:0:user_id->users")
        np.testing.assert_array_equal(
            first.subgraph.edge_index[relation],
            threaded.subgraph.edge_index[relation],
        )
        np.testing.assert_array_equal(
            first.subgraph.node_ids[TableRef("events")], [0, 1, 2]
        )
        fetched = store.fetch(first)
        self.assertEqual(
            fetched.features[(TableRef("users"), "age")].values.shape, (2, 2)
        )
        self.assertEqual(
            fetched.features[(TableRef("events"), "amount")].values.shape,
            (3, 1),
        )


class SamplingContractTest(unittest.TestCase):
    def test_two_seeds_can_reference_the_same_global_node_separately(self) -> None:
        users = TableRef("users")
        subgraph = SampledSubgraph(
            node_ids={users: np.array([7, 7], dtype=np.int64)},
            node_seed_index={users: np.array([0, 1], dtype=np.int64)},
            edge_index={},
            seed_table=users,
            seed_local_ids=np.array([0, 1], dtype=np.int64),
            seed_time_us=np.array([10, 20], dtype=np.int64),
            sampled_nodes_per_hop={users: (2,)},
            sampled_edges_per_hop={},
            random_seed=42,
        )
        plan = EntitySamplePlan(
            BatchKey(0, 0), subgraph, np.array([0, 1]), ((1,), (2,))
        )
        batch = PreparedBatch(plan.key, plan, {})
        assert isinstance(batch.sample, EntitySamplePlan)
        self.assertEqual(len(batch.sample.subgraph.seed_local_ids), 2)
        subgraph.validate()
        with self.assertRaisesRegex(ValueError, "own seed"):
            invalid = SampledSubgraph(
                node_ids=subgraph.node_ids,
                node_seed_index={users: np.array([0, 0], dtype=np.int64)},
                edge_index={},
                seed_table=users,
                seed_local_ids=subgraph.seed_local_ids,
                seed_time_us=subgraph.seed_time_us,
                sampled_nodes_per_hop=subgraph.sampled_nodes_per_hop,
                sampled_edges_per_hop={},
                random_seed=42,
            )
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
