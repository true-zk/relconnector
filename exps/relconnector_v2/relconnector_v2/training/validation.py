"""Opt-in CPU checks and input preservation for failed training steps."""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from relconnector_v2.features import PreparedBatch, TensorFrameFeatureSchema
from relconnector_v2.runtime.contracts import Trainer, TrainStepResult


class ValidatedTrainer:
    """Diagnose categorical errors before a CUDA assertion poisons the context.

    Inputs are never clamped or skipped. Retain CPU stores while the underlying
    trainer replaces its stores with device tensors, so a failed batch can be
    saved even after a CUDA failure.
    """

    def __init__(
        self,
        trainer: Trainer,
        schema: TensorFrameFeatureSchema,
        diagnostic_dir: Path,
    ) -> None:
        self.trainer = trainer
        self.diagnostic_dir = diagnostic_dir
        self.columns = {
            table: names[stype.categorical]
            for table, names in schema.col_names_dict.items()
            if stype.categorical in names
        }
        self.limits = {
            table: torch.tensor([
                len(cast(
                    tuple[list[object], list[int]],
                    schema.tables[table].col_stats[col][StatType.COUNT],
                )[0])
                for col in columns
            ], dtype=torch.long)
            for table, columns in self.columns.items()
        }
        self.batches = 0
        self.validation_s = 0.0

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        started = time.perf_counter()
        inputs = (batch.data,) if isinstance(batch.data, HeteroData) else batch.data
        copies = tuple(copy.copy(data) for data in inputs)
        saved = replace(
            batch, data=copies[0] if isinstance(batch.data, HeteroData) else copies
        )
        try:
            for data in inputs:
                self._validate(data)
            self.validation_s += time.perf_counter() - started
            self.batches += 1
            return self.trainer.train_step(batch)
        except Exception:
            try:
                self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
                path = self.diagnostic_dir / (
                    f"epoch-{batch.key.epoch}-batch-{batch.key.batch}-"
                    f"{time.time_ns()}.pt"
                )
                torch.save(saved, path)
                print(f"[v2 failed input] {path.resolve()}", file=sys.stderr, flush=True)
            except Exception as save_error:  # noqa: BLE001
                # A diagnostic failure must not replace the training exception.
                print(
                    f"[v2 failed input] could not save: {save_error}",
                    file=sys.stderr, flush=True,
                )
            raise

    def _validate(self, data: HeteroData) -> None:
        for table, tf in data.tf_dict.items():
            feat = tf.feat_dict.get(stype.categorical)
            if feat is None:
                continue
            if (not isinstance(feat, torch.Tensor)
                    or feat.device.type != "cpu" or feat.dtype != torch.long):
                raise ValueError(f"{table}: expected CPU int64 categorical inputs")
            if tf.col_names_dict[stype.categorical] != self.columns.get(table):
                raise ValueError(f"{table}: categorical columns differ from model schema")
            invalid = (feat < -1) | (feat >= self.limits[table])
            if bool(invalid.any()):
                row, col = invalid.nonzero()[0].tolist()
                node_ids = getattr(data[table], "n_id", None)
                node = None if node_ids is None else int(node_ids[row])
                raise ValueError(
                    f"{table}.{self.columns[table][col]}: categorical index "
                    f"{int(feat[row, col])} outside [-1, "
                    f"{int(self.limits[table][col]) - 1}], row={row}, node_id={node}"
                )
