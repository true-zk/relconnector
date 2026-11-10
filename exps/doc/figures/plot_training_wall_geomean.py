"""Paper figures for the ten-dataset geometric-mean wall-time speedup."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "thesis/figures/training_time_share"
SOURCE = OUTPUT / "phase_times_5epoch.csv"
BLUE = "#4C70A8"
RED = "#D9675D"
PALE = "#E5E9EF"
STAGES = (
    ("graph_index_s", "离线图索引构建", "#4C70A8"),
    ("sampling_s", "采样", "#39A89B"),
    ("data_prep_s", "数据准备（含等待）", "#D9675D"),
    ("training_s", "训练", "#D9A737"),
)


def load_speedup() -> float:
    per_dataset: dict[str, dict[str, float]] = {}
    with SOURCE.open(newline="") as file:
        for row in csv.DictReader(file):
            per_dataset.setdefault(row["dataset"], {})[row["method"]] = float(row["training_wall_s"])
    assert len(per_dataset) == 10
    ratios = []
    for pair in per_dataset.values():
        assert set(pair) == {"Vanilla", "V2"}
        ratios.append(pair["Vanilla"] / pair["V2"])
    return math.exp(mean(math.log(value) for value in ratios))


def load_phase_totals() -> dict[str, list[float]]:
    """Partition measured wall time; charge overlap and waits to preparation.

    Sampling uses the single sampler's measured service time, and training uses
    the main training loop's measured service time. Preparation is the remaining
    elapsed wall time, including pipeline waits and uninstrumented overhead.
    """
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(STAGES))
    counts = defaultdict(int)
    with SOURCE.open(newline="") as file:
        for row in csv.DictReader(file):
            assert row["method"] in {"Vanilla", "V2"}
            method = "vanilla" if row["method"] == "Vanilla" else "relx"
            counts[method] += 1
            graph = float(row["graph_index_s"])
            sample = float(row["sampling_s"])
            train = float(row["training_s"])
            wall = float(row["training_wall_s"])
            prep = wall - sample - train
            assert prep >= 0, row["dataset"]
            for index, value in enumerate((graph, sample, prep, train)):
                totals[method][index] += value / 3600
    assert counts == {"vanilla": 10, "relx": 10}
    return dict(totals)


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


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=.16)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def comparison(speedup: float) -> None:
    relative = 100 / speedup
    fig, ax = plt.subplots(figsize=(8.3, 3.9))
    fig.subplots_adjust(left=.18, right=.92, top=.84, bottom=.22)
    ax.barh(1, 100, height=.45, color=BLUE)
    ax.barh(0, relative, height=.45, color=RED)
    ax.text(50, 1, "100%", ha="center", va="center", color="white",
            fontsize=14, fontweight="bold")
    ax.text(relative + 1.2, 0, f"{relative:.1f}%", ha="left", va="center",
            color=RED, fontsize=14, fontweight="bold")
    ax.set_yticks([1, 0], ["vanilla", "relx"], fontsize=14)
    ax.set_xlim(0, 105)
    ax.set_ylim(-.5, 1.5)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.tick_params(length=0, labelsize=11, pad=10)
    ax.grid(axis="x", color=PALE, linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.5, 1.12, f"{speedup:.2f}×", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=24, fontweight="bold", color=RED)
    save(fig, "wall_time_geomean_5epoch")


def phase_stacked(totals: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(9.3, 4.2))
    fig.subplots_adjust(left=.15, right=.96, top=.78, bottom=.27)
    draw_phase_stacked(ax, totals)
    ax.set_xlabel("累计墙钟时间（小时）", fontsize=12, labelpad=10)
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in STAGES],
               loc="upper center", bbox_to_anchor=(.55, .99), ncol=4,
               frameon=False, fontsize=10.5, columnspacing=1.4)
    save(fig, "phase_wall_stacked_5epoch")


def draw_phase_stacked(ax: plt.Axes, totals: dict[str, list[float]]) -> None:
    maximum = max(sum(values) for values in totals.values())
    for method, y in (("vanilla", 1), ("relx", 0)):
        values = totals[method]
        total = sum(values)
        left = 0.0
        for value, (key, _, color) in zip(values, STAGES, strict=True):
            ax.barh(y, value, left=left, height=.46, color=color,
                    edgecolor="white", linewidth=1.1)
            if key == "data_prep_s":
                share = 100 * value / total
                if method == "vanilla":
                    ax.text(left + value / 2, y, f"{share:.1f}%  ·  {value:.1f} h",
                            ha="center", va="center", color="white",
                            fontsize=13, fontweight="bold")
                else:
                    ax.text(left + value / 2, y, f"{value:.1f} h",
                            ha="center", va="center", color="white",
                            fontsize=11, fontweight="bold")
                    ax.annotate(f"{share:.1f}%", xy=(left + value / 2, y + .28),
                                xytext=(left + value / 2, y + .55),
                                ha="center", va="bottom", fontsize=10,
                                arrowprops={"arrowstyle": "-", "color": "#637287", "lw": .7})
            left += value
        ax.text(total + maximum * .014, y, f"{total:.1f} h", ha="left",
                va="center", fontsize=12, fontweight="bold")
    ax.set_yticks([1, 0], ["vanilla", "relx"], fontsize=14)
    ax.set_ylim(-.7, 1.5)
    ax.set_xlim(0, maximum * 1.14)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.tick_params(length=0, labelsize=11, pad=9)
    ax.grid(axis="x", color=PALE, linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def combined(totals: dict[str, list[float]], speedup: float) -> None:
    relative = 100 / speedup
    fig = plt.figure(figsize=(12.5, 4.5))
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.8, 1]})
    fig.subplots_adjust(left=.09, right=.98, top=.77, bottom=.2, wspace=.27)
    left, right = axes
    draw_phase_stacked(left, totals)
    left.set_title("十数据集五轮累计墙钟时间", fontsize=14, pad=18)
    left.set_xlabel("小时", fontsize=12, labelpad=10)
    right.pie([relative, 100 - relative], colors=[RED, PALE],
              startangle=90, counterclock=False,
              wedgeprops={"width": .32, "edgecolor": "white", "linewidth": 1.2})
    right.text(0, .12, f"{speedup:.2f}×", ha="center", va="center",
               fontsize=24, fontweight="bold")
    right.text(0, -.18, "平均加速", ha="center", va="center",
               fontsize=11, color="#637287")
    right.set_title("十数据集几何平均加速", fontsize=14, pad=18)
    right.text(.5, -1.23, f"relx / vanilla = {relative:.1f}%", ha="center",
               va="center", fontsize=11)
    right.set_aspect("equal")
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in STAGES],
               loc="upper center", bbox_to_anchor=(.45, .98), ncol=4,
               frameon=False, fontsize=10.5, columnspacing=1.3)
    save(fig, "training_time_stacked_and_speedup_5epoch")


def savings_pie(speedup: float) -> None:
    relative = 100 / speedup
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    fig.subplots_adjust(left=.06, right=.94, top=.93, bottom=.13)
    ax.pie(
        [relative, 100 - relative],
        colors=[RED, PALE],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": .35, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.text(0, .11, f"{speedup:.2f}×", ha="center", va="center",
            fontsize=25, fontweight="bold", color="#263347")
    ax.text(0, -.16, "平均加速", ha="center", va="center", fontsize=13, color="#637287")
    ax.set_aspect("equal")
    fig.text(.5, .035, f"relx / vanilla = {relative:.1f}%",
             ha="center", fontsize=13, color="#263347")
    save(fig, "wall_time_geomean_donut_5epoch")


def main() -> None:
    setup()
    speedup = load_speedup()
    totals = load_phase_totals()
    comparison(speedup)
    phase_stacked(totals)
    savings_pie(speedup)
    combined(totals, speedup)


if __name__ == "__main__":
    main()
