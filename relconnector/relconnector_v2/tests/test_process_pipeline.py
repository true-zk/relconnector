from __future__ import annotations

import time
import unittest
from typing import cast

import torch
from torch_geometric.data import HeteroData

from relconnector_v2.features import PreparedBatch
from relconnector_v2.runtime import (
    ProcessPipelineExecutor,
    ProcessRuntimeComponents,
    ProcessRuntimeConfig,
    TrainStepResult,
)
from relconnector_v2.sampling import EntitySamplePlan, SampledSubgraph, SamplePlan
from relconnector_v2.task import BatchKey, EntitySeedBatch, SeedBatch


class _Seeds:
    def __init__(self, batches: int) -> None:
        self.batches = batches

    def iter_epoch(self, epoch: int):
        for batch in range(self.batches):
            values = torch.tensor([batch], dtype=torch.long)
            yield EntitySeedBatch(
                BatchKey(epoch, batch),
                "node",
                values,
                None,
                values.float(),
            )


class _Sampler:
    def sample(self, seeds: SeedBatch) -> EntitySamplePlan:
        if not isinstance(seeds, EntitySeedBatch):
            raise TypeError("synthetic sampler only supports entity seeds")
        sample = SampledSubgraph(
            node_ids={"node": seeds.node_ids},
            edge_index={},
            batch=None,
            node_time={},
            num_sampled_nodes={"node": [len(seeds.node_ids)]},
            num_sampled_edges={},
            seed_node_type="node",
            seed_count=len(seeds.node_ids),
            seed_time=None,
        )
        return EntitySamplePlan(seeds.key, sample, seeds.target)


class _AssemblyFactory:
    def __init__(self, delay_first: bool = False) -> None:
        self.delay_first = delay_first

    def __call__(self, worker_index: int):
        return _AssemblyWorker(worker_index, self.delay_first)


class _AssemblyWorker:
    def __init__(self, worker_index: int, delay_first: bool) -> None:
        self.worker_index = worker_index
        self.delay_first = delay_first
        self.epochs: set[int] = set()

    def assemble_many(self, plans: list[SamplePlan]) -> list[PreparedBatch]:
        output = []
        for raw_plan in plans:
            if not isinstance(raw_plan, EntitySamplePlan):
                raise TypeError("synthetic worker only supports entity plans")
            plan = raw_plan
            if self.delay_first and plan.key.batch == 0:
                time.sleep(0.05)
            self.epochs.add(plan.key.epoch)
            data = HeteroData()
            data["node"].x = plan.subgraph.node_ids["node"].float()
            data["node"].y = plan.target
            data["node"].batch_size = plan.subgraph.seed_count
            output.append(PreparedBatch(plan.key, data, allocated_bytes=16))
        return output

    def diagnostics(self) -> dict[str, object]:
        return {
            "worker": self.worker_index,
            "epochs": sorted(self.epochs),
        }

    def close(self) -> None:
        return


class _Trainer:
    def __init__(self) -> None:
        self.order: list[BatchKey] = []

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        self.order.append(batch.key)
        return TrainStepResult(
            loss=float(batch.key.epoch * 10 + batch.key.batch),
            examples=1,
        )


class ProcessPipelineTest(unittest.TestCase):
    def _run(self, order: str, *, delay_first: bool = False):
        trainer = _Trainer()
        executor = ProcessPipelineExecutor(
            ProcessRuntimeConfig(
                assembly_processes=2,
                seed_queue_bytes=1024 * 1024,
                plan_queue_bytes=1024 * 1024,
                ready_queue_bytes=1024 * 1024,
                window_batches=1,
                window_bytes=1024 * 1024,
                max_inflight_windows=4,
                cross_epoch_prefetch=True,
                training_order=order,  # type: ignore[arg-type]
                queue_items=8,
            )
        )
        result = executor.run(
            ProcessRuntimeComponents(
                seeds=_Seeds(4),
                sampler=_Sampler(),
                assembly_factory=_AssemblyFactory(delay_first),
                trainer=trainer,
            ),
            epochs=2,
        )
        return result, trainer

    def test_strict_mode_preserves_batch_order(self) -> None:
        result, trainer = self._run("strict", delay_first=True)

        self.assertEqual(result.steps, 8)
        self.assertEqual(
            trainer.order,
            [BatchKey(epoch, batch) for epoch in range(2) for batch in range(4)],
        )
        records = cast(list[dict[str, object]], result.pipeline["epochs"])
        self.assertEqual(len(records), 2)

    def test_ready_mode_keeps_epoch_gate(self) -> None:
        result, trainer = self._run("ready", delay_first=True)

        self.assertEqual(result.steps, 8)
        self.assertEqual(set(trainer.order), {
            BatchKey(epoch, batch) for epoch in range(2) for batch in range(4)
        })
        epochs = [key.epoch for key in trainer.order]
        self.assertEqual(epochs, sorted(epochs))
        self.assertEqual(result.pipeline["training_order"], "ready")


if __name__ == "__main__":
    unittest.main()
