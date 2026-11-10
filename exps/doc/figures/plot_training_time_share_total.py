"""Publication figures for pooled five-epoch phase work across ten datasets."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "thesis/figures/training_time_share"
SOURCE = OUTPUT / "phase_times_5epoch.csv"
METHODS = ("vanilla", "relx")
SOURCE_METHODS = {"Vanilla": "vanilla", "V2": "relx"}
STAGES = (
    ("graph_index_s", "离线图索引构建", "#4C70A8"),
    ("sampling_s", "采样", "#39A89B"),
    ("data_prep_s", "数据准备", "#D9675D"),
    ("training_s", "训练", "#D9A737"),
)


def setup() -> None:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({
        "font.family": "Noto Sans CJK SC",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "#263347",
        "axes.labelcolor": "#263347",
        "xtick.color": "#637287",
        "ytick.color": "#263347",
    })


def load_totals() -> dict[str, list[float]]:
    totals = defaultdict(lambda: [0.0] * len(STAGES))
    datasets = defaultdict(set)
    with SOURCE.open(newline="") as file:
        for row in csv.DictReader(file):
            method = SOURCE_METHODS[row["method"]]
            datasets[method].add(row["dataset"])
            for index, (key, _, _) in enumerate(STAGES):
                totals[method][index] += float(row[key]) / 3600
    assert all(len(datasets[method]) == 10 for method in METHODS)
    return dict(totals)


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=.16)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def legend() -> list[Patch]:
    return [Patch(facecolor=color, label=label) for _, label, color in STAGES]


def stacked(totals: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    fig.subplots_adjust(left=.16, right=.90, top=.83, bottom=.25)
    for method, y in (("vanilla", 1), ("relx", 0)):
        left = 0.0
        for value, (key, _, color) in zip(totals[method], STAGES, strict=True):
            ax.barh(y, value, left=left, height=.48, color=color,
                    edgecolor="white", linewidth=1.1)
            if key == "data_prep_s":
                ax.text(left + value / 2, y, f"{value:.1f} h", va="center", ha="center",
                        color="white", fontsize=13, fontweight="bold")
            left += value
        ax.text(left + .35, y, f"{left:.1f} h", va="center", ha="left",
                fontsize=13, fontweight="bold")
    ax.set_yticks([1, 0], METHODS, fontsize=14)
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("累计工作时间（小时）", fontsize=11, labelpad=10)
    ax.tick_params(length=0, labelsize=11, pad=8)
    ax.grid(axis="x", color="#E7EAF0", linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(handles=legend(), loc="upper center", bbox_to_anchor=(.53, .98),
               ncol=4, frameon=False, fontsize=11, columnspacing=1.6)
    save(fig, "total_stacked_5epoch")


def pie(totals: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))
    fig.subplots_adjust(left=.04, right=.96, top=.86, bottom=.19, wspace=.23)
    colors = [color for _, _, color in STAGES]
    reference_total = max(sum(totals[method]) for method in METHODS)
    for ax, method in zip(axes, METHODS, strict=True):
        total = sum(totals[method])
        radius = math.sqrt(total / reference_total)
        ax.pie(
            totals[method],
            colors=colors,
            startangle=90,
            counterclock=False,
            radius=radius,
            wedgeprops={"edgecolor": "white", "linewidth": 1.4},
            autopct=lambda pct, prep=totals[method][2]:
                f"{prep:.1f} h\n{pct:.1f}%" if pct >= 50 else "",
            pctdistance=.60,
            textprops={"color": "white", "fontsize": 15, "fontweight": "bold"},
        )
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_title(f"{method}  {total:.1f} h", fontsize=17, fontweight="bold", pad=18)
        ax.set_aspect("equal")
    fig.legend(handles=legend(), loc="lower center", bbox_to_anchor=(.5, .08),
               ncol=4, frameon=False, fontsize=11, columnspacing=1.7)
    save(fig, "total_proportional_pie_5epoch")


def main() -> None:
    setup()
    totals = load_totals()
    stacked(totals)
    pie(totals)


if __name__ == "__main__":
    main()
