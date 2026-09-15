from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pandas as pd
import test_online
import torch
from relbench.base import Database, Table, TaskType
from test_online import _Sampler, _SeedSource, _Trainer
from torch_geometric.data import HeteroData

from baseline.feature_store import EagerFeatureStore, InMemoryTensorFrameFetcher
from baseline.models.graphsage import GraphSAGEModel
from benchmark.process import run_measured_process
from benchmark.telemetry import TelemetryRecorder
from relconnector.connector import (
    ColumnSchema,
    ConnectorXDatabaseReader,
    ForeignKey,
    PandasDatabaseReader,
    TableSchema,
)
from relconnector.features import (
    EntityFeatureBatch,
    SqlFeatureFetcher,
    SqlTensorFrameSchemaBuilder,
    TensorFrameBatchAssembler,
    TensorFrameEncoder,
)
from relconnector.features.cache import FeatureBlockCache, FeatureBlockKey
from relconnector.graph import InMemoryGraphIndexBuilder
from relconnector.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
    ByteBoundedQueue,
)
from relconnector.sampling import EntitySamplePlan, PygLibNeighborSampler
from relconnector.task import BatchKey, EntitySeedBatch, SqlSeedReader, TaskSpec
from relconnector.training import OnlineTrainer
from relconnector.training.model import OnlineEntityModel


class _TextEncoder:
    embedding_dim = 3

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, sentences: Sequence[object]) -> torch.Tensor:
        self.sizes.append(len(sentences))
        return torch.ones((len(sentences), self.embedding_dim))


class OnlineRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "fixture.sqlite"
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            connection.execute("INSERT INTO users VALUES (0, 'reader')")
            connection.execute(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, user_id INTEGER, "
                "time TEXT, value REAL, text TEXT)"
            )
            connection.executemany(
                "INSERT INTO events VALUES (?, 0, ?, ?, ?)",
                [
                    (0, "2024-01-02T00:00:00", 2.0, "second"),
                    (1, "2024-01-03T00:00:00.000000", 3.0, "third"),
                    (2, "2024-01-01T00:00:00", 1.0, "first"),
                ],
            )
        self.reader = PandasDatabaseReader(self.path)
        self.reader._schema_cache = {
            "users": TableSchema(
                "users",
                "id",
                columns=(
                    ColumnSchema("id", 0, "int64"),
                    ColumnSchema("name", 1, "object"),
                ),
            ),
            "events": TableSchema(
                "events",
                "id",
                foreign_keys=(ForeignKey("user_id", "users", "id"),),
                time_column="time",
                columns=(
                    ColumnSchema("id", 0, "int64"),
                    ColumnSchema("user_id", 1, "int64"),
                    ColumnSchema("time", 2, "datetime64[ns]", "datetime"),
                    ColumnSchema("value", 3, "float64"),
                    ColumnSchema("text", 4, "object"),
                ),
            ),
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_temporal_csc_and_rng_isolation(self) -> None:
        graph = InMemoryGraphIndexBuilder(self.reader, scan_batch_size=1).build()
        self.assertEqual(
            graph.row_dict["events__f2p_user_id__users"].tolist(), [2, 0, 1]
        )
        seeds = EntitySeedBatch(
            BatchKey(0, 0),
            "users",
            torch.tensor([0]),
            torch.tensor([1704153600]),
            torch.tensor([1.0]),
        )
        state = torch.random.get_rng_state().clone()
        plan = PygLibNeighborSampler(graph, num_neighbors=[8]).sample(seeds)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        assert isinstance(plan, EntitySamplePlan)
        self.assertEqual(set(plan.subgraph.node_ids["events"].tolist()), {0, 2})

    def test_cutoff_with_gaps_fails_before_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "dense zero-based"):
            InMemoryGraphIndexBuilder(self.reader).build(
                cutoff=cast(pd.Timestamp, pd.Timestamp("2024-01-02"))
            )

    def test_sparse_fetch_order_schema_and_empty_encoding(self) -> None:
        graph = InMemoryGraphIndexBuilder(self.reader).build()
        seed = EntitySeedBatch(
            BatchKey(0, 0), "events", torch.tensor([2, 0, 2]), None, torch.ones(3)
        )
        plan = PygLibNeighborSampler(graph, num_neighbors=[2]).sample(seed)
        assert isinstance(plan, EntitySamplePlan)
        plan = replace(
            plan,
            subgraph=replace(
                plan.subgraph,
                node_ids={"events": torch.tensor([2, 0, 2])},
                edge_index={},
            ),
        )
        schema = SqlTensorFrameSchemaBuilder(
            self.reader, hidden_columns=[("events", "text")], text_embedding_dim=3
        ).build()
        fetcher = SqlFeatureFetcher(self.reader, feature_columns=schema.feature_columns)
        features = fetcher.fetch(plan)
        assert isinstance(features, EntityFeatureBatch)
        frame = features.subgraph.frames["events"]
        assert isinstance(frame, pd.DataFrame)
        self.assertEqual(frame["value"].tolist(), [1.0, 2.0, 1.0])
        self.assertNotIn("text", frame)
        self.assertEqual(fetcher.last_stats.queried_rows, 2)
        encoder = _TextEncoder()
        assembler = TensorFrameBatchAssembler(
            TensorFrameEncoder(schema, encoder, text_batch_size=2)
        )
        full = assembler.assemble(features)
        assert isinstance(full.data, HeteroData)
        self.assertEqual(full.data["events"].tf.num_rows, 3)
        self.assertEqual(full.data["events"].tf.num_cols, 2)
        empty_plan = replace(
            plan,
            subgraph=replace(
                plan.subgraph, node_ids={"events": torch.empty(0, dtype=torch.long)}
            ),
        )
        empty = assembler.assemble(fetcher.fetch(empty_plan))
        assert isinstance(empty.data, HeteroData)
        self.assertEqual(empty.data["events"].tf.num_rows, 0)
        self.assertEqual(empty.data["events"].tf.num_cols, 2)

    def test_text_batches_are_bounded(self) -> None:
        schema = SqlTensorFrameSchemaBuilder(self.reader, text_embedding_dim=3).build()
        fetcher = SqlFeatureFetcher(self.reader, feature_columns=schema.feature_columns)
        frame, _ = fetcher._fetch_table("events", torch.tensor([2, 0, 1]))
        encoder = _TextEncoder()
        tensor_frame = TensorFrameEncoder(schema, encoder, text_batch_size=2).encode(
            "events", frame
        )
        self.assertEqual(tensor_frame.num_rows, 3)
        self.assertEqual(tensor_frame.num_cols, 3)
        self.assertEqual(encoder.sizes, [2, 1])

    def test_seed_rows_are_covered_once_and_times_are_seconds(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE targets (id INTEGER PRIMARY KEY, node INTEGER, time TEXT, label TEXT, split TEXT)"
            )
            connection.executemany(
                "INSERT INTO targets VALUES (?, ?, '2024-01-02', ?, ?)",
                [
                    (i, i, "t" if i % 2 else "f", "train" if i < 7 else "val")
                    for i in range(8)
                ],
            )
        self.reader.schemas()["targets"] = TableSchema(
            "targets",
            "id",
            kind="task",
            split_column="split",
            columns=(
                ColumnSchema("node", 1, "int64"),
                ColumnSchema("time", 2, "datetime64[us]", "datetime"),
                ColumnSchema("label", 3, "object"),
            ),
        )
        task = TaskSpec(
            "fixture",
            "targets",
            TaskType.BINARY_CLASSIFICATION,
            "events",
            "node",
            "label",
            "time",
            None,
            None,
            (),
        )
        source = SqlSeedReader(self.reader, task, batch_size=2, shuffle_block_size=3)
        for epoch in range(2):
            batches = list(source.iter_epoch(epoch))
            self.assertEqual(
                [batch.key for batch in batches],
                [BatchKey(epoch, i) for i in range(len(batches))],
            )
            ids: list[int] = []
            for batch in batches:
                assert isinstance(batch, EntitySeedBatch)
                ids.extend(batch.node_ids.tolist())
                assert batch.seed_time is not None
                self.assertEqual(set(batch.seed_time.tolist()), {1704153600})
            self.assertEqual(sorted(ids), list(range(7)))

    def test_eager_and_sql_paths_produce_identical_tensorframes(self) -> None:
        schema = SqlTensorFrameSchemaBuilder(self.reader, text_embedding_dim=3).build()
        encoder = TensorFrameEncoder(schema, _TextEncoder(), text_batch_size=2)
        database = Database(
            {
                "users": Table(self.reader.read_table("users"), {}, pkey_col="id"),
                "events": Table(
                    self.reader.read_table("events"),
                    {"user_id": "users"},
                    pkey_col="id",
                    time_col="time",
                ),
            }
        )
        store = EagerFeatureStore.materialize(database, encoder)
        graph = InMemoryGraphIndexBuilder(self.reader).build()
        seeds = EntitySeedBatch(
            BatchKey(0, 0), "events", torch.tensor([2, 0, 2]), None, torch.ones(3)
        )
        plan = PygLibNeighborSampler(graph, num_neighbors=[2]).sample(seeds)
        assert isinstance(plan, EntitySamplePlan)
        online_fetcher = SqlFeatureFetcher(
            self.reader, feature_columns=schema.feature_columns
        )
        online = TensorFrameBatchAssembler(encoder).assemble(online_fetcher.fetch(plan))
        eager = TensorFrameBatchAssembler(encoder).assemble(
            InMemoryTensorFrameFetcher(store).fetch(plan)
        )
        assert isinstance(online.data, HeteroData)
        assert isinstance(eager.data, HeteroData)
        self.assertEqual(online.data.node_types, eager.data.node_types)
        for node_type in online.data.node_types:
            self.assertEqual(online.data[node_type].tf, eager.data[node_type].tf)
        task = TaskSpec(
            "fixture",
            "targets",
            TaskType.BINARY_CLASSIFICATION,
            "events",
            "node",
            "label",
            None,
            None,
            None,
            (),
        )
        losses = []
        for prepared in (eager, online):
            torch.manual_seed(77)
            trainer = OnlineTrainer(
                graph,
                task,
                feature_schema=schema,
                out_channels=1,
                channels=8,
                num_layers=1,
                aggr="sum",
                device="cpu",
                seed=77,
            )
            losses.append(trainer.train_step(prepared).loss)
        self.assertEqual(losses[0], losses[1])

    def test_eager_and_online_models_initialize_identically(self) -> None:
        schema = SqlTensorFrameSchemaBuilder(self.reader, text_embedding_dim=3).build()
        encoder = TensorFrameEncoder(schema, _TextEncoder(), text_batch_size=2)
        database = Database(
            {
                "users": Table(self.reader.read_table("users"), {}, pkey_col="id"),
                "events": Table(
                    self.reader.read_table("events"),
                    {"user_id": "users"},
                    pkey_col="id",
                    time_col="time",
                ),
            }
        )
        store = EagerFeatureStore.materialize(database, encoder)
        graph = InMemoryGraphIndexBuilder(self.reader).build()
        for node_type, frame in store.frames.items():
            graph.data[node_type].tf = frame
        torch.manual_seed(123)
        eager = GraphSAGEModel(
            data=graph.data,
            col_stats_dict=schema.col_stats_dict,
            out_channels=1,
            channels=8,
            num_layers=1,
            aggr="sum",
        )
        torch.manual_seed(123)
        online = OnlineEntityModel(
            node_types=graph.node_types,
            edge_types=list(graph.edge_types),
            col_names_dict=schema.col_names_dict,
            col_stats_dict=schema.col_stats_dict,
            out_channels=1,
            channels=8,
            num_layers=1,
            aggr="sum",
        )
        self.assertEqual(eager.state_dict().keys(), online.state_dict().keys())
        for name, value in eager.state_dict().items():
            self.assertTrue(torch.equal(value, online.state_dict()[name]), name)

    def test_connectorx_arrow_batches(self) -> None:
        chunks = list(
            ConnectorXDatabaseReader(self.path).iter_query(
                "SELECT id FROM events ORDER BY id", batch_size=2
            )
        )
        self.assertTrue(all(len(chunk) <= 2 for chunk in chunks))
        self.assertEqual(pd.concat(chunks)["id"].tolist(), [0, 1, 2])


class RuntimeFailureTest(unittest.TestCase):
    def test_consumer_interrupt_cancels_blocked_producers(self) -> None:
        components = test_online.RuntimeTest()._components(_Trainer())
        with (
            patch.object(
                components.trainer, "train_step", side_effect=KeyboardInterrupt
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            AsyncPipelineExecutor(AsyncRuntimeConfig(16, 16, 4)).run(
                components, epochs=20
            )
        names = {worker.name for worker in threading.enumerate()}
        self.assertFalse(
            names.intersection({"seed-reader", "sampler", "feature-fetcher"})
        )

    def test_each_worker_failure_propagates_and_cancels(self) -> None:
        for component, method in (
            ("seeds", "iter_epoch"),
            ("sampler", "sample"),
            ("fetcher", "fetch"),
            ("assembler", "assemble"),
        ):
            with self.subTest(component=component):
                components = test_online.RuntimeTest()._components(_Trainer())
                with (
                    patch.object(
                        getattr(components, component),
                        method,
                        side_effect=ValueError("fixture failure"),
                    ),
                    self.assertRaises(RuntimeError) as error,
                ):
                    AsyncPipelineExecutor(AsyncRuntimeConfig(16, 16, 4)).run(
                        components, epochs=20
                    )
                self.assertIsInstance(error.exception.__cause__, ValueError)

    def test_queue_rejects_oversize_and_bounds_zero_size(self) -> None:
        queue = ByteBoundedQueue[bytes](4, len, max_items=1)
        with self.assertRaises(ValueError):
            queue.put(b"12345")
        queue.put(b"")
        entered, finished = threading.Event(), threading.Event()

        def producer() -> None:
            entered.set()
            try:
                queue.put(b"")
            except RuntimeError:
                finished.set()

        worker = threading.Thread(target=producer)
        worker.start()
        entered.wait(1)
        self.assertFalse(finished.wait(0.01))
        queue.close(discard=True)
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.is_set())
        self.assertEqual(queue.queued_bytes, 0)

    def test_plan_budget_includes_node_times(self) -> None:
        sample = _Sampler().sample(next(iter(_SeedSource().iter_epoch(0)))).subgraph
        timed_sample = replace(sample, node_time={"users": torch.tensor([1])})
        self.assertEqual(timed_sample.allocated_bytes, sample.allocated_bytes + 8)

    def test_lru_budget_and_eviction(self) -> None:
        frame = pd.DataFrame({"value": [1.0, 2.0]})
        size = int(frame.memory_usage(index=True, deep=True).sum())
        cache = FeatureBlockCache(size)
        first, second = [FeatureBlockKey("v", "users", ("value",), i) for i in range(2)]
        cache.put(first, frame)
        cache.put(second, frame)
        self.assertLessEqual(cache.current_bytes, size)
        self.assertIsNone(cache.get(first))
        self.assertIsNotNone(cache.get(second))


class ObservabilityTest(unittest.TestCase):
    def test_telemetry_retention_keeps_exact_count(self) -> None:
        recorder = TelemetryRecorder(max_operation_samples=8, max_phase_records=2)
        with recorder.activate():
            for _ in range(100):
                with recorder.operation("work"):
                    pass
            for _ in range(3):
                with recorder.phase("stage"):
                    pass
        report = recorder.report()
        self.assertEqual(report["operations"]["work"]["count"], 100)
        self.assertEqual(report["operations"]["work"]["percentile_samples"], 8)
        self.assertEqual(len(report["phases"]), 2)
        self.assertEqual(report["phase_records_dropped"], 1)

    def test_verbose_process_does_not_deadlock_and_timeout_is_measured(self) -> None:
        result = run_measured_process(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('x' * 200000); sys.stderr.write('y' * 200000)",
            ],
            timeout_s=5,
            sample_interval_s=0.01,
        )
        self.assertEqual(result.returncode, 0)
        self.assertLessEqual(len(result.stdout), 8000)
        timed = run_measured_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_s=0.1,
            sample_interval_s=0.01,
        )
        self.assertTrue(timed.timed_out)
        self.assertGreaterEqual(timed.duration_s, 0.1)
        self.assertLess(timed.duration_s, 5)

    def test_layer_dependencies_are_one_way(self) -> None:
        import ast

        root = Path(__file__).resolve().parents[1]
        for package, forbidden in (
            ("data", {"baseline", "benchmark", "relconnector"}),
            ("baseline", {"benchmark", "data"}),
            ("relconnector", {"baseline", "benchmark", "data"}),
        ):
            for path in (root / package).rglob("*.py"):
                if "tests" in path.parts:
                    continue
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    modules: list[str] = []
                    if isinstance(node, ast.Import):
                        modules = [item.name for item in node.names]
                    elif (
                        isinstance(node, ast.ImportFrom)
                        and node.level == 0
                        and node.module
                    ):
                        modules = [node.module]
                    self.assertFalse(
                        {name.split(".")[0] for name in modules} & forbidden, str(path)
                    )

    def test_lightweight_imports_do_not_load_training(self) -> None:
        code = """
import sys
import relconnector.connector
import baseline.config
from benchmark import TelemetryRecorder
assert 'torch' not in sys.modules
assert 'baseline.dataset' not in sys.modules
with TelemetryRecorder().activate() as recorder:
    pass
assert recorder.report()['overall']['duration_s'] >= 0
assert 'torch' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
