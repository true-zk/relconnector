from __future__ import annotations

import sqlite3
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from relconnector.connector import PandasDatabaseReader
from relconnector.features import (
    EntityFeatureBatch,
    FeatureBatch,
    FetchedSubgraph,
    PreparedBatch,
)
from relconnector.graph import GraphIndex
from relconnector.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
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
                frames={"users": pd.DataFrame({"value": [1.0]})},
            ),
            target=plan.target,
        )


class _Assembler:
    def assemble(self, features: FeatureBatch) -> PreparedBatch:
        assert isinstance(features, EntityFeatureBatch)
        data = HeteroData()
        data["users"].x = torch.ones((1, 1))
        return PreparedBatch(features.key, data, allocated_bytes=4)


class _Trainer:
    def __init__(self) -> None:
        self.keys: list[BatchKey] = []

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        self.keys.append(batch.key)
        return TrainStepResult(loss=float(batch.key.batch), examples=1)


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


if __name__ == "__main__":
    unittest.main()
