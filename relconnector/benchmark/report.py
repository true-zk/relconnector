"""Summarize benchmark JSONL results by dataset, phase, and failure type."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from .runner import BenchmarkPayload, SuccessPayload


def load_latest(path: Path) -> list[BenchmarkPayload]:
    latest: dict[tuple[str, str], BenchmarkPayload] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            payload = cast(BenchmarkPayload, json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        latest[(payload["dataset"], payload["task"])] = payload
    return list(latest.values())


def render_summary(rows: list[BenchmarkPayload]) -> str:
    successes = [row for row in rows if row["status"] == "ok"]
    failures = [row for row in rows if row["status"] == "error"]
    lines = [
        f"tasks={len(rows)} ok={len(successes)} error={len(failures)}",
        "",
        "dataset\tok\terror",
    ]
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        lines.append(
            f"{dataset}\t"
            f"{sum(row['status'] == 'ok' for row in selected)}\t"
            f"{sum(row['status'] == 'error' for row in selected)}"
        )

    if successes:
        phase_times: dict[str, list[float]] = defaultdict(list)
        peak_rss: list[tuple[float, str]] = []
        for row in cast(list[SuccessPayload], successes):
            telemetry = row["telemetry"]
            peak_rss.append(
                (
                    telemetry["overall"]["peak"]["rss_mb"],
                    f"{row['dataset']}/{row['task']}",
                )
            )
            for phase in telemetry["phases"]:
                phase_times[phase["name"]].append(phase["duration_s"])
        lines.extend(["", "phase\tcount\ttotal_s\tmedian_s"])
        for phase, samples in sorted(phase_times.items()):
            lines.append(
                f"{phase}\t{len(samples)}\t{sum(samples):.3f}\t"
                f"{statistics.median(samples):.3f}"
            )
        lines.extend(["", "largest_peak_rss_mb\ttask"])
        for memory, name in sorted(peak_rss, reverse=True)[:10]:
            lines.append(f"{memory:.1f}\t{name}")

    if failures:
        counts = Counter(row["error_type"] for row in failures)
        lines.extend(["", "error_type\tcount"])
        lines.extend(f"{name}\t{count}" for name, count in counts.most_common())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(render_summary(load_latest(args.input)))


if __name__ == "__main__":
    main()
