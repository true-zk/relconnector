"""Serializable output of one measured training run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from .metadata import ExperimentMetadata
from .telemetry import TelemetryReport


class TrainingTelemetryReport(TelemetryReport):
    config: dict[str, object]


class TrainingRunData(TypedDict):
    dataset: str
    task: str
    task_type: str
    model: str
    reader: str
    device: str
    epochs: int
    completed_batches: int
    completed_examples: int
    losses: list[float]
    experiment: ExperimentMetadata
    telemetry: TrainingTelemetryReport


def add_training_config(
    report: TelemetryReport,
    config: dict[str, object],
) -> TrainingTelemetryReport:
    return {
        "environment": report["environment"],
        "overall": report["overall"],
        "phases": report["phases"],
        "operations": report["operations"],
        "phase_records_dropped": report["phase_records_dropped"],
        "config": config,
    }


@dataclass
class TrainingRunResult:
    dataset: str
    task: str
    task_type: str
    model: str
    reader: str
    device: str
    epochs: int
    completed_batches: int
    completed_examples: int
    losses: list[float]
    experiment: ExperimentMetadata
    telemetry: TrainingTelemetryReport

    def to_dict(self) -> TrainingRunData:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "task_type": self.task_type,
            "model": self.model,
            "reader": self.reader,
            "device": self.device,
            "epochs": self.epochs,
            "completed_batches": self.completed_batches,
            "completed_examples": self.completed_examples,
            "losses": self.losses,
            "experiment": self.experiment,
            "telemetry": self.telemetry,
        }

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )


class OnlineTrainingRunData(TypedDict):
    dataset: str
    task: str
    completed_batches: int
    completed_examples: int
    loss: float
    graph_nodes: int
    graph_edges: int
    graph_bytes: int
    experiment: ExperimentMetadata
    telemetry: TelemetryReport
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class OnlineTrainingRunResult:
    dataset: str
    task: str
    completed_batches: int
    completed_examples: int
    loss: float
    graph_nodes: int
    graph_edges: int
    graph_bytes: int
    experiment: ExperimentMetadata
    telemetry: TelemetryReport
    diagnostics: dict[str, object]

    def to_dict(self) -> OnlineTrainingRunData:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "completed_batches": self.completed_batches,
            "completed_examples": self.completed_examples,
            "loss": self.loss,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "graph_bytes": self.graph_bytes,
            "experiment": self.experiment,
            "telemetry": self.telemetry,
            "diagnostics": self.diagnostics,
        }
