from __future__ import annotations

import time
import unittest
from dataclasses import replace
from typing import cast
from unittest.mock import patch

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


class _FailingSeeds(_Seeds):
    def iter_epoch(self, epoch: int):
        yield from super().iter_epoch(epoch)
        raise RuntimeError("synthetic seed read failed")


class _WrongEpochSampler(_Sampler):
    def sample(self, seeds: SeedBatch) -> EntitySamplePlan:
        plan = super().sample(seeds)
        return replace(plan, key=BatchKey(plan.key.epoch + 1, plan.key.batch))


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


class _SlowDiagnosticsFactory:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def __call__(self, worker_index: int):
        return _SlowDiagnosticsWorker(worker_index, self.delay)


class _SlowDiagnosticsWorker(_AssemblyWorker):
    def __init__(self, worker_index: int, delay: float) -> None:
        super().__init__(worker_index, False)
        self.delay = delay

    def diagnostics(self) -> dict[str, object]:
        time.sleep(self.delay)
        return super().diagnostics()


class _ResourceExhaustedFactory:
    def __call__(self, worker_index: int):
        return _ResourceExhaustedWorker(worker_index, False)


class _ResourceExhaustedWorker(_AssemblyWorker):
    def assemble_many(self, plans: list[SamplePlan]) -> list[PreparedBatch]:
        # A mmap limit can also prevent the error queue from starting its first
        # background feeder thread. Keep the failure injected through reporting.
        self.thread_patch = patch(
            "threading.Thread.start", side_effect=RuntimeError("can't start new thread")
        )
        self.thread_patch.start()
        raise OSError("synthetic mmap cannot allocate memory")

    def close(self) -> None:
        self.thread_patch.stop()


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
    def test_training_failure_identifies_actual_seed_batch(self) -> None:
        trainer = _Trainer()
        failure = ValueError("synthetic training failure")
        trainer.train_step = lambda batch: (_ for _ in ()).throw(failure)
        executor = ProcessPipelineExecutor(ProcessRuntimeConfig(assembly_processes=1))
        with self.assertRaisesRegex(
            RuntimeError, "training failed at epoch=0, batch=0, sequence=0"
        ) as caught:
            executor.run(
                ProcessRuntimeComponents(
                    seeds=_Seeds(1), sampler=_Sampler(),
                    assembly_factory=_AssemblyFactory(), trainer=trainer,
                ), epochs=1,
            )
        self.assertIs(caught.exception.__cause__, failure)

    def test_reports_slower_than_old_ten_second_deadline_are_collected(self) -> None:
        executor = ProcessPipelineExecutor(ProcessRuntimeConfig(assembly_processes=1))
        result = executor.run(
            ProcessRuntimeComponents(
                seeds=_Seeds(1), sampler=_Sampler(),
                assembly_factory=_SlowDiagnosticsFactory(11.0), trainer=_Trainer(),
            ), epochs=1,
        )
        reports = [r for r in result.pipeline["workers"] if r["stage"] == "assembly"]
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["diagnostics"]["epochs"], [0])

    def test_missing_diagnostics_raise_instead_of_silent_success(self) -> None:
        executor = ProcessPipelineExecutor(ProcessRuntimeConfig(
            assembly_processes=1, diagnostics_timeout_s=0.1,
        ))
        with self.assertRaisesRegex(RuntimeError, "timed out collecting assembly diagnostics"):
            executor.run(
                ProcessRuntimeComponents(
                    seeds=_Seeds(1), sampler=_Sampler(),
                    assembly_factory=_SlowDiagnosticsFactory(0.5), trainer=_Trainer(),
                ), epochs=1,
            )

    def test_empty_epochs_finish_without_waiting_for_batches(self) -> None:
        result, trainer = self._run("ready", batches=0)
        self.assertEqual(result.steps, 0)
        self.assertEqual(trainer.order, [])

    def test_resource_exhaustion_failure_does_not_need_a_new_thread(self) -> None:
        executor = ProcessPipelineExecutor(ProcessRuntimeConfig(assembly_processes=1))
        with self.assertRaisesRegex(RuntimeError, "assembly.*mmap cannot allocate memory"):
            executor.run(
                ProcessRuntimeComponents(
                    seeds=_Seeds(1), sampler=_Sampler(),
                    assembly_factory=_ResourceExhaustedFactory(), trainer=_Trainer(),
                ),
                epochs=1,
            )

    def test_finished_workers_with_untrainable_batches_raise(self) -> None:
        executor = ProcessPipelineExecutor(
            ProcessRuntimeConfig(assembly_processes=1, window_batches=1)
        )
        with self.assertRaisesRegex(RuntimeError, "all assembly workers finished"):
            executor.run(
                ProcessRuntimeComponents(
                    seeds=_Seeds(1),
                    sampler=_WrongEpochSampler(),
                    assembly_factory=_AssemblyFactory(),
                    trainer=_Trainer(),
                ),
                epochs=1,
            )

    def test_seed_reader_failure_reaches_trainer(self) -> None:
        executor = ProcessPipelineExecutor(
            ProcessRuntimeConfig(assembly_processes=1, window_batches=1)
        )
        with self.assertRaisesRegex(RuntimeError, "seed.*synthetic seed read failed"):
            executor.run(
                ProcessRuntimeComponents(
                    seeds=_FailingSeeds(1),
                    sampler=_Sampler(),
                    assembly_factory=_AssemblyFactory(),
                    trainer=_Trainer(),
                ),
                epochs=1,
            )

    def _run(
        self,
        order: str,
        *,
        delay_first: bool = False,
        batches: int = 4,
        window_batches: int = 1,
    ):
        trainer = _Trainer()
        executor = ProcessPipelineExecutor(
            ProcessRuntimeConfig(
                assembly_processes=2,
                seed_queue_bytes=1024 * 1024,
                plan_queue_bytes=1024 * 1024,
                ready_queue_bytes=1024 * 1024,
                window_batches=window_batches,
                window_bytes=1024 * 1024,
                max_inflight_windows=4,
                cross_epoch_prefetch=True,
                training_order=order,  # type: ignore[arg-type]
                queue_items=8,
            )
        )
        result = executor.run(
            ProcessRuntimeComponents(
                seeds=_Seeds(batches),
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
        self.assertEqual(
            set(trainer.order),
            {BatchKey(epoch, batch) for epoch in range(2) for batch in range(4)},
        )
        epochs = [key.epoch for key in trainer.order]
        self.assertEqual(epochs, sorted(epochs))
        self.assertEqual(result.pipeline["training_order"], "ready")

    def test_cross_epoch_prefetch_keeps_windows_epoch_homogeneous(self) -> None:
        result, trainer = self._run(
            "ready",
            batches=5,
            window_batches=1,
        )

        self.assertEqual(result.steps, 10)
        self.assertEqual(
            set(trainer.order),
            {BatchKey(epoch, batch) for epoch in range(2) for batch in range(5)},
        )
        self.assertEqual(
            [key.epoch for key in trainer.order],
            sorted(key.epoch for key in trainer.order),
        )


if __name__ == "__main__":
    unittest.main()
