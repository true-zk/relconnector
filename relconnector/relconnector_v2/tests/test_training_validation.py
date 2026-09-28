from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from relconnector_v2.features import PreparedBatch
from relconnector_v2.features.schema import (
    TableFeatureSchema,
    TensorFrameFeatureSchema,
)
from relconnector_v2.runtime.contracts import TrainStepResult
from relconnector_v2.task import BatchKey
from relconnector_v2.training.validation import ValidatedTrainer


class _Trainer:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        self.calls += 1
        if self.fail:
            # Like HeteroData.to(cuda), replace stores with device tensors.
            # Meta permits exercising preservation without requiring CUDA.
            batch.data.to("meta")
            raise RuntimeError("synthetic device failure")
        return TrainStepResult(loss=0.5, examples=2)


def _schema() -> TensorFrameFeatureSchema:
    return TensorFrameFeatureSchema({
        "nodes": TableFeatureSchema((), {"kind": stype.categorical}, {
            "kind": {StatType.COUNT: (["a", "b"], [3, 2])},
        }),
    }, "test")


def _batch(values: list[int]) -> PreparedBatch:
    data = HeteroData()
    data["nodes"].tf = TensorFrame(
        {stype.categorical: torch.tensor(values).reshape(-1, 1)},
        {stype.categorical: ["kind"]},
    )
    data["nodes"].n_id = torch.arange(10, 10 + len(values))
    return PreparedBatch(BatchKey(1, 3805), data, 32)


class TrainingValidationTest(unittest.TestCase):
    def test_valid_and_unknown_categories_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = _Trainer()
            validator = ValidatedTrainer(trainer, _schema(), Path(directory))
            batch = _batch([-1, 1])
            result = validator.train_step(batch)
            self.assertEqual(result.examples, 2)
            self.assertEqual(trainer.calls, 1)
            self.assertEqual(validator.batches, 1)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(
                batch.data["nodes"].tf.feat_dict[stype.categorical].tolist(),
                [[-1], [1]],
            )

    def test_invalid_index_fails_before_device_work_and_saves_seed_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = _Trainer()
            validator = ValidatedTrainer(trainer, _schema(), Path(directory))
            with self.assertRaisesRegex(ValueError, r"nodes.kind.*index 2.*node_id=11"):
                validator.train_step(_batch([0, 2]))
            self.assertEqual(trainer.calls, 0)
            path, = Path(directory).glob("*.pt")
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved.key, BatchKey(1, 3805))
            self.assertEqual(
                saved.data["nodes"].tf.feat_dict[stype.categorical].tolist(), [[0], [2]]
            )

    def test_device_failure_preserves_cpu_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            validator = ValidatedTrainer(_Trainer(fail=True), _schema(), Path(directory))
            with self.assertRaisesRegex(RuntimeError, "synthetic device failure"):
                validator.train_step(_batch([0, 1]))
            path, = Path(directory).glob("*.pt")
            saved = torch.load(path, weights_only=False)
            feat = saved.data["nodes"].tf.feat_dict[stype.categorical]
            self.assertEqual(feat.device.type, "cpu")
            self.assertEqual(feat.tolist(), [[0], [1]])


if __name__ == "__main__":
    unittest.main()
