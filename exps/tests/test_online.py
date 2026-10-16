from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from torch_frame import TensorFrame, stype
from torch_geometric.data import HeteroData

from relconnector.artifacts import load_graph, save_graph
from relconnector.connector import PandasDatabaseReader
from relconnector.features import (
    EncodedFeatureCache,
    EntityFeatureBatch,
    FeatureBatch,
    FetchedNodeFeatures,
    FetchedSubgraph,
    PreparedBatch,
    TensorFrameBatchAssembler,
)
from relconnector.graph import GraphIndex
from relconnector.observability import OperationMetrics
from relconnector.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
    ByteBoundedQueue,
    RuntimeComponents,
    SyncExecutor,
    TrainStepResult,
)
from relconnector.sampling import (
    EntitySamplePlan,
    PygLibNeighborSampler,
    SampledSubgraph,
    SamplePlan,
)
from relconnector.task import BatchKey, EntitySeedBatch, SeedBatch


class PandasStreamingReaderTest(unittest.TestCase):
    def test_iter_query_is_bounded_by_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE values_table (value INTEGER)")
                connection.executemany(
                    "INSERT INTO values_table VALUES (?)",
                    [(index,) for index in range(7)],
                )
            reader = PandasDatabaseReader(path)

            chunks = list(
                reader.iter_query(
                    "SELECT value FROM values_table ORDER BY value",
                    batch_size=3,
                )
            )

        self.assertEqual([len(chunk) for chunk in chunks], [3, 3, 1])
        self.assertEqual(pd.concat(chunks)["value"].tolist(), list(range(7)))


class PygLibSamplerTest(unittest.TestCase):
    def test_samples_topology_without_node_features(self) -> None:
        data = HeteroData()
        data["users"].num_nodes = 3
        data["events"].num_nodes = 4
        edge_type = ("events", "f2p_user_id", "users")
        reverse_type = ("users", "rev_f2p_user_id", "events")
        data[edge_type].num_edges = 4
        data[reverse_type].num_edges = 4
        graph = GraphIndex(
            data=data,
            edge_types=(edge_type, reverse_type),
            colptr_dict={
                "events__f2p_user_id__users": torch.tensor([0, 2, 3, 4]),
                "users__rev_f2p_user_id__events": torch.tensor([0, 1, 2, 3, 4]),
            },
            row_dict={
                "events__f2p_user_id__users": torch.tensor([0, 1, 2, 3]),
                "users__rev_f2p_user_id__events": torch.tensor([0, 0, 1, 2]),
            },
            node_count=7,
            edge_count=8,
            allocated_bytes=128,
        )
        seeds = EntitySeedBatch(
            key=BatchKey(0, 0),
            node_type="users",
            node_ids=torch.tensor([0, 1]),
            seed_time=None,
            target=torch.tensor([0.0, 1.0]),
        )

        plan = PygLibNeighborSampler(graph, num_neighbors=[2]).sample(seeds)

        self.assertIsInstance(plan, EntitySamplePlan)
        assert isinstance(plan, EntitySamplePlan)
        self.assertEqual(plan.key, BatchKey(0, 0))
        self.assertEqual(plan.subgraph.seed_count, 2)
        self.assertIn("users", plan.subgraph.node_ids)
        self.assertIn(reverse_type, plan.subgraph.edge_index)


class _SeedSource:
    def iter_epoch(self, epoch: int) -> Iterable[SeedBatch]:
        for batch in range(3):
            yield EntitySeedBatch(
                key=BatchKey(epoch, batch),
                node_type="users",
                node_ids=torch.tensor([batch]),
                seed_time=None,
                target=torch.tensor([float(batch)]),
            )


class _Sampler:
    def sample(self, seeds: SeedBatch) -> EntitySamplePlan:
        assert isinstance(seeds, EntitySeedBatch)
        sample = SampledSubgraph(
            node_ids={"users": seeds.node_ids},
            edge_index={},
            batch=None,
            node_time={},
            num_sampled_nodes={"users": [len(seeds.node_ids)]},
            num_sampled_edges={},
            seed_node_type="users",
            seed_count=len(seeds.node_ids),
            seed_time=None,
        )
        return EntitySamplePlan(seeds.key, sample, seeds.target)


class _Fetcher:
    def fetch(self, plan: SamplePlan) -> EntityFeatureBatch:
        assert isinstance(plan, EntitySamplePlan)
        return EntityFeatureBatch(
            key=plan.key,
            subgraph=FetchedSubgraph(
                sample=plan.subgraph,
                frames={
                    "users": FetchedNodeFeatures(
                        pd.DataFrame({"value": [1.0]}), torch.tensor([0])
                    )
                },
            ),
            target=plan.target,
        )

    def fetch_many(self, plans: list[SamplePlan]) -> list[FeatureBatch]:
        return [self.fetch(plan) for plan in plans]


class _Assembler:
    def assemble(self, features: FeatureBatch) -> PreparedBatch:
        assert isinstance(features, EntityFeatureBatch)
        data = HeteroData()
        data["users"].x = torch.ones((1, 1))
        return PreparedBatch(features.key, data, allocated_bytes=4)

    def assemble_many(self, features: list[FeatureBatch]) -> list[PreparedBatch]:
        return [self.assemble(item) for item in features]


class _Trainer:
    def __init__(self) -> None:
        self.keys: list[BatchKey] = []

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        self.keys.append(batch.key)
        return TrainStepResult(loss=float(batch.key.batch), examples=1)


class QueueTest(unittest.TestCase):
    def test_drain_respects_item_and_byte_limits(self) -> None:
        queue = ByteBoundedQueue[int](10, lambda value: value)
        for value in (3, 4, 2):
            queue.put(value)

        self.assertEqual(queue.drain(3, max_bytes=6), [3])
        self.assertEqual(queue.queued_bytes, 6)
        self.assertEqual(queue.stats.peak_items, 3)
        self.assertEqual(queue.stats.peak_bytes, 9)
        self.assertEqual(queue.get(), 4)
        self.assertEqual(queue.get(), 2)


class EncodedFeatureCacheTest(unittest.TestCase):
    def test_second_access_admission_is_bounded(self) -> None:
        cache = EncodedFeatureCache(4096, admission="second")
        node_ids = torch.tensor([10, 20])
        frame = TensorFrame(
            {stype.numerical: torch.tensor([[1.0], [2.0]])},
            {stype.numerical: ["value"]},
        )

        self.assertIsNone(cache.put("items", node_ids, frame))
        self.assertEqual(cache.lookup_many("items", node_ids)[1], [0, 1])
        self.assertIsNotNone(cache.put("items", node_ids, frame))
        groups, misses = cache.lookup_many("items", node_ids)

        self.assertFalse(misses)
        self.assertEqual(len(groups), 1)
        self.assertEqual(cache.hit_rows, 2)
        self.assertLessEqual(cache.current_bytes, cache.max_bytes)

    def test_mixed_hits_and_misses_restore_node_order(self) -> None:
        class Encoder:
            metrics = OperationMetrics(False)

            def encode(self, table: str, frame: pd.DataFrame) -> TensorFrame:
                del table
                return TensorFrame(
                    {
                        stype.numerical: torch.tensor(
                            frame[["value"]].to_numpy(), dtype=torch.float32
                        )
                    },
                    {stype.numerical: ["value"]},
                )

        assembler = TensorFrameBatchAssembler(
            Encoder(),  # type: ignore[arg-type]
            encoded_cache_bytes=4096,
            encoded_cache_admission="always",
        )

        def batch(ids: list[int], values: list[float]) -> EntityFeatureBatch:
            node_ids = torch.tensor(ids)
            sample = SampledSubgraph(
                node_ids={"users": node_ids},
                edge_index={},
                batch=None,
                node_time={},
                num_sampled_nodes={"users": [len(ids)]},
                num_sampled_edges={},
                seed_node_type="users",
                seed_count=len(ids),
                seed_time=None,
            )
            return EntityFeatureBatch(
                BatchKey(0, 0),
                FetchedSubgraph(
                    sample,
                    {
                        "users": FetchedNodeFeatures(
                            pd.DataFrame({"value": values}),
                            torch.arange(len(ids)),
                            unique_ids=node_ids,
                        )
                    },
                ),
                torch.zeros(len(ids)),
            )

        assembler.assemble(batch([10, 20], [1.0, 2.0]))
        mixed = assembler.assemble(batch([20, 30], [2.0, 3.0]))
        assert isinstance(mixed.data, HeteroData)
        values = mixed.data["users"].tf.feat_dict[stype.numerical]

        self.assertEqual(values[:, 0].tolist(), [2.0, 3.0])
        self.assertEqual(assembler.encoded_cache.hit_rows, 1)


class GraphArtifactTest(unittest.TestCase):
    def test_tensor_only_round_trip_and_key_validation(self) -> None:
        data = HeteroData()
        edge_type = ("items", "to", "users")
        data["items"].num_nodes = 2
        data["items"].time = torch.tensor([1, 2])
        data["users"].num_nodes = 1
        data[edge_type].num_edges = 2
        graph = GraphIndex(
            data=data,
            edge_types=(edge_type,),
            colptr_dict={"items__to__users": torch.tensor([0, 2])},
            row_dict={"items__to__users": torch.tensor([0, 1])},
            node_count=3,
            edge_count=2,
            allocated_bytes=40,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.pt"
            save_graph(path, "version", graph)
            restored = load_graph(path, "version")
            self.assertIsNone(load_graph(path, "stale"))
            path.write_bytes(b"truncated")
            self.assertIsNone(load_graph(path, "version"))

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.edge_types, graph.edge_types)
        self.assertTrue(
            torch.equal(restored.row_dict["items__to__users"], torch.tensor([0, 1]))
        )
        self.assertTrue(torch.equal(restored.data["items"].time, torch.tensor([1, 2])))


class RuntimeTest(unittest.TestCase):
    def _components(self, trainer: _Trainer) -> RuntimeComponents:
        return RuntimeComponents(
            seeds=_SeedSource(),
            sampler=_Sampler(),
            fetcher=_Fetcher(),
            assembler=_Assembler(),
            trainer=trainer,
        )

    def test_sync_executor_preserves_epoch_and_batch_order(self) -> None:
        trainer = _Trainer()
        result = SyncExecutor().run(self._components(trainer), epochs=2)

        self.assertEqual(result.steps, 6)
        self.assertEqual(
            trainer.keys,
            [BatchKey(epoch, batch) for epoch in range(2) for batch in range(3)],
        )

    def test_async_executor_preserves_epoch_and_batch_order(self) -> None:
        trainer = _Trainer()
        executor = AsyncPipelineExecutor(
            AsyncRuntimeConfig(
                seed_queue_bytes=32,
                plan_queue_bytes=64,
                ready_queue_bytes=32,
            )
        )
        result = executor.run(self._components(trainer), epochs=2)

        self.assertEqual(result.steps, 6)
        self.assertEqual(
            trainer.keys,
            [BatchKey(epoch, batch) for epoch in range(2) for batch in range(3)],
        )
        self.assertEqual(result.pipeline["feature_window_batches"], 6)
        epoch_stats = result.pipeline["epochs"]
        assert isinstance(epoch_stats, list)
        self.assertEqual([item["steps"] for item in epoch_stats], [3, 3])
        feature_windows = result.pipeline["feature_windows"]
        assert isinstance(feature_windows, int)
        self.assertGreaterEqual(feature_windows, 1)
        ready_stats = result.pipeline["ready_queue"]
        assert isinstance(ready_stats, dict)
        peak_items = ready_stats["peak_items"]
        assert isinstance(peak_items, int)
        self.assertGreaterEqual(peak_items, 1)

    def test_parallel_encode_preserves_order_when_first_window_is_slow(self) -> None:
        class SlowAssembler(_Assembler):
            def assemble_many(
                self, features: list[FeatureBatch]
            ) -> list[PreparedBatch]:
                if features[0].key.batch == 0:
                    time.sleep(0.05)
                return super().assemble_many(features)

        trainer = _Trainer()
        components = self._components(trainer)
        components = RuntimeComponents(
            seeds=components.seeds,
            sampler=components.sampler,
            fetcher=components.fetcher,
            assembler=SlowAssembler(),
            trainer=trainer,
            assembler_factory=SlowAssembler,
        )
        result = AsyncPipelineExecutor(
            AsyncRuntimeConfig(
                seed_queue_bytes=32,
                plan_queue_bytes=64,
                ready_queue_bytes=64,
                feature_window_batches=1,
                feature_window_bytes=64,
                encode_workers=2,
                fetched_queue_bytes=1024,
            )
        ).run(components, epochs=1)

        self.assertEqual(
            trainer.keys,
            [BatchKey(0, batch) for batch in range(3)],
        )
        self.assertEqual(result.pipeline["encode_workers"], 2)


if __name__ == "__main__":
    unittest.main()
