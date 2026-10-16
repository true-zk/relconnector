"""Single-batch trainer for online prepared batches."""

from __future__ import annotations

import torch
from relbench.base import TaskType
from torch_geometric.data import HeteroData

from baseline.cache_baseline.features import PreparedBatch, TensorFrameFeatureSchema
from baseline.cache_baseline.graph import GraphIndex
from baseline.cache_baseline.randomness import TORCH_RNG_LOCK
from baseline.cache_baseline.runtime.contracts import TrainStepResult
from baseline.cache_baseline.task import TaskSpec

from .model import OnlineEntityModel, OnlineLinkModel


class OnlineTrainer:
    def __init__(
        self,
        graph: GraphIndex,
        task: TaskSpec,
        *,
        feature_schema: TensorFrameFeatureSchema,
        out_channels: int = 1,
        channels: int = 128,
        num_layers: int = 2,
        aggr: str = "sum",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | torch.device | None = None,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.seed = seed
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if task.is_recommendation:
            self.model: OnlineEntityModel | OnlineLinkModel = OnlineLinkModel(
                node_types=graph.data.node_types,
                col_names_dict=feature_schema.col_names_dict,
                col_stats_dict=feature_schema.col_stats_dict,
                edge_types=graph.data.edge_types,
                channels=channels,
                num_layers=num_layers,
                aggr=aggr,
            ).to(self.device)
        else:
            self.model = OnlineEntityModel(
                node_types=graph.data.node_types,
                col_names_dict=feature_schema.col_names_dict,
                col_stats_dict=feature_schema.col_stats_dict,
                edge_types=graph.data.edge_types,
                out_channels=out_channels,
                channels=channels,
                num_layers=num_layers,
                aggr=aggr,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_step(self, batch: PreparedBatch) -> TrainStepResult:
        devices = (
            [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
            if self.device.type == "cuda"
            else []
        )
        with TORCH_RNG_LOCK, torch.random.fork_rng(devices=devices):
            step_seed = _derive_seed(self.seed, batch.key.epoch, batch.key.batch)
            torch.manual_seed(step_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(step_seed)
            return self._train_step(batch)

    def _train_step(self, batch: PreparedBatch) -> TrainStepResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.task.is_recommendation:
            if not isinstance(self.model, OnlineLinkModel) or not isinstance(
                batch.data, tuple
            ):
                raise TypeError("Recommendation task requires three sampled batches")
            source, positive, negative = (
                sampled_batch.to(self.device) for sampled_batch in batch.data
            )
            assert self.task.dst_entity_table is not None
            positive_score, negative_score = self.model(
                source,
                positive,
                negative,
                self.task.entity_table,
                self.task.dst_entity_table,
            )
            loss = torch.nn.functional.softplus(negative_score - positive_score).mean()
            examples = int(source[self.task.entity_table].batch_size)
        else:
            if not isinstance(self.model, OnlineEntityModel) or not isinstance(
                batch.data, HeteroData
            ):
                raise TypeError("Entity task requires one sampled batch")
            entity_batch = batch.data.to(self.device)
            prediction = self.model(entity_batch, self.task.entity_table)
            target = entity_batch[self.task.entity_table].y
            loss = _entity_loss(prediction, target, self.task.task_type)
            examples = int(entity_batch[self.task.entity_table].batch_size)
        loss.backward()
        self.optimizer.step()
        return TrainStepResult(loss=float(loss.detach().cpu()), examples=examples)


def _derive_seed(*parts: int) -> int:
    value = 0xD1B54A32D192ED03
    for part in parts:
        value ^= int(part) + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2)
    return value & ((1 << 63) - 1)


def _entity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task_type: TaskType,
) -> torch.Tensor:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.view(-1), target.float()
        )
    if task_type == TaskType.REGRESSION:
        return torch.nn.functional.l1_loss(prediction.view(-1), target.float())
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return torch.nn.functional.cross_entropy(prediction, target.long())
    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction,
            target.float(),
        )
    raise NotImplementedError(f"Unsupported entity task type: {task_type}")
