"""Compare measured runs without treating different models as speedups."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .metadata import ExperimentMetadata
from .telemetry import TelemetryReport


@dataclass(frozen=True)
class MeasuredRun:
    dataset: str
    task: str
    metadata: ExperimentMetadata
    telemetry: TelemetryReport
    batches: int
    examples: int
    loss: float


def load_runs(path: Path) -> dict[tuple[str, str], MeasuredRun]:
    latest: dict[tuple[str, str], MeasuredRun] = {}
    with path.open() as source:
        for number, line in enumerate(source, 1):
            row = cast(dict[str, object], json.loads(line))
            key = str(row["dataset"]), str(row["task"])
            if row["status"] != "ok":
                latest.pop(key, None)
                continue
            if "experiment" not in row:
                raise ValueError(
                    f"{path}:{number}: legacy run lacks experiment metadata"
                )
            latest[key] = MeasuredRun(
                *key,
                cast(ExperimentMetadata, row["experiment"]),
                cast(TelemetryReport, row["telemetry"]),
                int(str(row["completed_batches"])),
                int(str(row["completed_examples"])),
                _loss(row),
            )
    return latest


def differences(left: MeasuredRun, right: MeasuredRun) -> list[str]:
    fields = (
        "reader",
        "executor",
        "feature_encoder",
        "feature_schema",
        "model",
        "sampling_policy",
        "device",
        "database_path",
        "database_bytes",
        "database_mtime_ns",
    )
    a = cast(dict[str, object], left.metadata)
    b = cast(dict[str, object], right.metadata)
    changed = [field for field in fields if a.get(field) != b.get(field)]
    for key in (
        "epochs",
        "max_batches",
        "batch_size",
        "num_neighbors",
        "channels",
        "aggr",
        "lr",
        "weight_decay",
        "seed",
        "torch_num_threads",
        "text_batch_size",
        "out_channels",
    ):
        if left.metadata["config"].get(key) != right.metadata["config"].get(key):
            changed.append("config." + key)
    if (left.batches, left.examples) != (right.batches, right.examples):
        changed.append("completed_work")
    if left.telemetry["environment"] != right.telemetry["environment"]:
        changed.append("environment")
    if not math.isclose(left.loss, right.loss, rel_tol=1e-7, abs_tol=1e-7):
        changed.append("loss")
    return changed


def render_comparison(left_path: Path, right_path: Path) -> str:
    left, right = load_runs(left_path), load_runs(right_path)
    lines = ["dataset/task\tleft_s\tright_s\tleft_rss_MiB\tright_rss_MiB\tcomparison"]
    for key in sorted(left.keys() | right.keys()):
        name = "/".join(key)
        if key not in left or key not in right:
            lines.append(f"{name}\tmissing successful pair")
            continue
        a, b = left[key], right[key]
        ta, tb = a.telemetry["overall"], b.telemetry["overall"]
        changed = differences(a, b)
        comparison = (
            "not-equivalent:" + ",".join(changed) if changed else "strictly-comparable"
        )
        lines.append(
            f"{name}\t{ta['duration_s']:.3f}\t{tb['duration_s']:.3f}\t"
            f"{ta['peak']['rss_mb']:.1f}\t{tb['peak']['rss_mb']:.1f}\t{comparison}"
        )
    return "\n".join(lines)


def _loss(row: dict[str, object]) -> float:
    if "loss" in row:
        return float(str(row["loss"]))
    losses = cast(list[float], row["losses"])
    return float(losses[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(render_comparison(args.left, args.right))


if __name__ == "__main__":
    main()
