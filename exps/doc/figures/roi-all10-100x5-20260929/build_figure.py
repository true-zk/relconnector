"""Build the fixed-100 ROI comparison figure and auditable data table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "benchmarks/roi-all10-100x5-rss-20260929-v1"
OFF_RUN = ROOT / "benchmarks/cache-iteration-100x5-v2/summary.json"

# Rounded values transcribed from the user-provided four-version figure. Event and
# Trial differ from the later replacement runs in final/plotted-data.csv, so the
# attached figure is deliberately kept as a distinct reference.
CHART = {
    "rel-amazon": (1.480, 2.76),
    "rel-arxiv": (0.470, 2.41),
    "rel-avito": (0.186, 3.27),
    "rel-event": (15.900, 3.32),
    "rel-f1": (2.340, 2.03),
    "rel-hm": (0.118, 3.04),
    "rel-ratebeer": (0.772, 3.02),
    "rel-salt": (0.821, 3.12),
    "rel-stack": (0.277, 2.30),
    "rel-trial": (0.557, 2.31),
}
ORDER = [
    "rel-amazon",
    "rel-arxiv",
    "rel-avito",
    "rel-event",
    "rel-f1",
    "rel-hm",
    "rel-ratebeer",
    "rel-salt",
    "rel-stack",
    "rel-trial",
]


def main() -> None:
    measured = {row["dataset"]: row for row in json.loads((RUN / "summary.json").read_text())}
    off = {
        row["dataset"]: row
        for row in json.loads(OFF_RUN.read_text())
        if row["stage"] == "fixed" and row["profile"] == "off"
    }
    rows = []
    for dataset in ORDER:
        row = measured[dataset]
        baseline_time, baseline_rss = CHART[dataset]
        memory = row["monitored_memory"]
        same_input = off[dataset]
        if same_input["fingerprint"] != row["fingerprint"]:
            raise ValueError(f"Replay mismatch for {dataset}")
        training_per_batch = row["training_wall_s"] / row["steps"]
        e2e_per_batch = row["end_to_end_s"] / row["steps"]
        rows.append(
            {
                "dataset": dataset.removeprefix("rel-"),
                "batches_per_epoch": row["epoch_steps"][0],
                "total_batches": row["steps"],
                "attached_chart_v2_s_per_batch": baseline_time,
                "roi_training_s_per_batch": training_per_batch,
                "roi_run_e2e_s_per_batch": e2e_per_batch,
                "training_reduction_vs_chart_pct": 100
                * (1 - training_per_batch / baseline_time),
                "run_e2e_reduction_vs_chart_pct": 100
                * (1 - e2e_per_batch / baseline_time),
                "same_replay_off_training_s": same_input["training_wall_s"],
                "roi_training_s": row["training_wall_s"],
                "roi_reduction_vs_same_replay_off_pct": 100
                * (1 - row["training_wall_s"] / same_input["training_wall_s"]),
                "cold_epoch_s": row["cold_epoch_s"],
                "warm_epoch_mean_s": row["warm_epoch_mean_s"],
                "warm_reduction_vs_cold_pct": 100
                * (1 - row["warm_epoch_mean_s"] / row["cold_epoch_s"]),
                "attached_chart_v2_main_rss_gib": baseline_rss,
                "roi_main_peak_rss_gib": row["reported_main_peak_rss_mb"] / 1024,
                "roi_main_rss_change_vs_chart_pct": 100
                * (row["reported_main_peak_rss_mb"] / 1024 / baseline_rss - 1),
                "roi_tree_peak_pss_gib": memory["peak_tree_pss_mb"] / 1024,
                "roi_tree_peak_rss_sum_gib": memory["peak_tree_rss_mb"] / 1024,
                "disk_write_blocks": row["cost_counters"].get("disk_write_blocks", 0),
                "disk_hit_blocks": row["cost_counters"].get("disk_hit_blocks", 0),
                "disk_roi_rejections": row["cost_counters"].get(
                    "disk_roi_rejections", 0
                ),
            }
        )

    with (OUT / "roi-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "roi-comparison.json").write_text(json.dumps(rows, indent=2) + "\n")

    labels = [row["dataset"].title() for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    chart_time = np.array([row["attached_chart_v2_s_per_batch"] for row in rows])
    roi_time = np.array([row["roi_training_s_per_batch"] for row in rows])
    chart_rss = np.array([row["attached_chart_v2_main_rss_gib"] for row in rows])
    roi_main = np.array([row["roi_main_peak_rss_gib"] for row in rows])
    roi_pss = np.array([row["roi_tree_peak_pss_gib"] for row in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, (time_ax, memory_ax) = plt.subplots(2, 1, figsize=(14, 9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.10, hspace=0.38)
    fig.suptitle("ROI cache on ten datasets: fixed replay, 5 epochs, up to 100 batches/epoch", fontsize=16)

    time_ax.bar(x - width / 2, chart_time, width, label="Attached chart v2", color="#465A8B")
    time_ax.bar(x + width / 2, roi_time, width, label="ROI training", color="#3D9B75")
    time_ax.set_yscale("log")
    time_ax.set_ylabel("Seconds / completed batch (log scale)")
    time_ax.set_xticks(x, labels)
    time_ax.grid(axis="y", alpha=0.22, which="both")
    time_ax.legend(frameon=False, ncol=2)
    for index, row in enumerate(rows):
        reduction = row["training_reduction_vs_chart_pct"]
        time_ax.text(
            index,
            max(chart_time[index], roi_time[index]) * 1.13,
            f"{reduction:+.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#27313A",
        )
    time_ax.set_title("Time; positive annotation means ROI is faster")

    memory_ax.bar(x - width / 2, chart_rss, width, label="Attached v2 main RSS", color="#465A8B")
    memory_ax.bar(x + width / 2, roi_main, width, label="ROI main RSS", color="#3D9B75")
    memory_ax.scatter(x + width / 2, roi_pss, label="ROI process-tree PSS", color="#C44E52", marker="D", s=38, zorder=4)
    memory_ax.set_yscale("log")
    memory_ax.set_ylabel("Peak GiB (log scale)")
    memory_ax.set_xticks(x, labels)
    memory_ax.grid(axis="y", alpha=0.22, which="both")
    memory_ax.legend(frameon=False, ncol=3)
    memory_ax.set_title("Memory; process-tree PSS includes all workers and apportions shared pages")
    fig.text(
        0.08,
        0.043,
        "Attached-chart values are rounded labels and use historical full-run conditions. "
        "ROI time is training wall time on fixed replay; compare directionally.\n"
        "Main RSS is comparable in scope; process-tree PSS has no historical counterpart.",
        fontsize=9,
        color="#4A5560",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"roi-all10-time-rss.{suffix}", dpi=220)


if __name__ == "__main__":
    main()
