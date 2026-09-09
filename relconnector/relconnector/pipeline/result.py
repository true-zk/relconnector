"""Serializable output of one measured training run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

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
            "telemetry": self.telemetry,
        }

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )
