"""Plot the measured four-stage work share for the paired Arxiv experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "exps/benchmarks/arxiv-two-epoch-validation-20261031"
OUTPUT = ROOT / "thesis/figures/training_time_share"
STAGES = ("离线图索引构建", "采样", "数据准备", "训练")
COLORS = ("#506FAD", "#42A69B", "#DC675D", "#D5A331")


def load_measurements() -> dict[str, dict]:
    vanilla = json.loads((SOURCE / "vanilla.json").read_text())
    v2 = json.loads((SOURCE / "optimized.jsonl").read_text().splitlines()[-1])
    graph_s = vanilla["prepare_breakdown_s"]["graph_build_s"]
    vanilla_epochs = vanilla["epochs"]
    workers = v2["pipeline"]["workers"]
    return {
        "Vanilla": {
            "values": (
                graph_s,
                sum(row["sampling_s"] for row in vanilla_epochs),
                sum(row["feature_read_assemble_s"] for row in vanilla_epochs),
                sum(row["training_s"] for row in vanilla_epochs),
            ),
            "wall": sum(row["training_loop_s"] for row in vanilla_epochs),
        },
        "V2": {
            "values": (
                graph_s,  # Shared one-time build; the V2 run itself hit the topology cache.
                sum(workers[0]["epoch_sample_s"].values()),
                sum(
                    seconds
                    for worker in workers[1:]
                    for epoch in worker["diagnostics"]["epoch_stage"].values()
                    for seconds in epoch.values()
                ),
                sum(row["train_s"] for row in v2["pipeline"]["epochs"]),
            ),
            "wall": sum(row["wall_duration_s"] for row in v2["pipeline"]["epochs"]),
        },
    }


def set_style() -> None:
    font_manager.fontManager.addfont(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK SC",
            "axes.unicode_minus": False,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "text.color": "#263347",
        }
    )


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:,.1f} 秒"


def draw_bar(ax, values: tuple[float, ...], y: float, height: float = 0.40) -> None:
    total = sum(values)
    left = 0.0
    for stage, value, color in zip(STAGES, values, COLORS, strict=True):
        pct = 100 * value / total
        ax.barh(y, pct, left=left, height=height, color=color, edgecolor="white", linewidth=1.5)
        if stage == "数据准备":
            ax.text(left + pct / 2, y, f"数据准备  {pct:.1f}%", ha="center", va="center",
                    color="white", fontsize=15, fontweight="bold")
        left += pct


def save_plot(fig, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def individual(name: str, item: dict, vanilla_data_s: float) -> None:
    values = item["values"]
    total = sum(values)
    fig = plt.figure(figsize=(11.6, 6.5))
    ax = fig.add_axes((0.07, 0.51, 0.86, 0.16))
    draw_bar(ax, values, 0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.38, 0.38)
    ax.set_yticks([])
    ax.set_xticks((0, 25, 50, 75, 100), ("0%", "25%", "50%", "75%", "100%"))
    ax.tick_params(axis="x", length=0, labelsize=10, pad=9, colors="#69778B")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.07, 0.88, f"{name}｜训练阶段累计工作时间占比", fontsize=21, fontweight="bold")
    fig.text(0.07, 0.81, "Arxiv · paper-citation · 2 epoch × 512 batch · 同一采样计划",
             fontsize=11, color="#69778B")
    fig.text(0.07, 0.72, "四阶段耗时结构", fontsize=12, fontweight="bold")
    legend = [Patch(facecolor=color, label=stage) for stage, color in zip(STAGES, COLORS, strict=True)]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.07, 0.39), ncol=4,
               frameon=False, fontsize=10, handlelength=1.2, columnspacing=1.6)
    for i, (stage, value, color) in enumerate(zip(STAGES, values, COLORS, strict=True)):
        x = 0.07 + i * 0.224
        fig.text(x, 0.28, stage, fontsize=10, color="#69778B")
        fig.text(x, 0.22, fmt_seconds(value), fontsize=14, fontweight="bold", color=color)
        fig.text(x, 0.17, f"{100 * value / total:.2f}%", fontsize=10, color="#69778B")
    if name == "V2":
        fig.text(0.07, 0.095,
                 f"数据准备累计工作时间下降 {100 * (1 - values[2] / vanilla_data_s):.1f}%"
                 f"；两轮训练墙钟时间 {fmt_seconds(item['wall'])}。",
                 fontsize=11, fontweight="bold")
        fig.text(0.07, 0.042,
                 "注：组装为 4 个进程的累计工作时间；离线索引成本取共同冷启动构建，V2 实测运行复用该索引。",
                 fontsize=9, color="#69778B")
    else:
        fig.text(0.07, 0.095, f"两轮训练墙钟时间 {fmt_seconds(item['wall'])}。", fontsize=11,
                 fontweight="bold")
        fig.text(0.07, 0.042, "注：阶段计时之和与墙钟时间的细小差异来自种子生成和循环开销。",
                 fontsize=9, color="#69778B")
    save_plot(fig, f"arxiv_{name.lower()}")


def comparison(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.75, bottom=0.34)
    for name, y in (("Vanilla", 1), ("V2", 0)):
        draw_bar(ax, data[name]["values"], y, 0.42)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks((1, 0), ("Vanilla", "V2"))
    ax.set_xticks((0, 25, 50, 75, 100), ("0%", "25%", "50%", "75%", "100%"))
    ax.tick_params(length=0, labelsize=11, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#E6EAF0", zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.12, 0.90, "训练阶段时间结构：Vanilla 与 V2", fontsize=21, fontweight="bold")
    fig.text(0.12, 0.82, "Arxiv · paper-citation · 2 epoch × 512 batch · 同一采样计划",
             fontsize=11, color="#69778B")
    fig.legend(handles=[Patch(facecolor=c, label=s) for s, c in zip(STAGES, COLORS, strict=True)],
               loc="center left", bbox_to_anchor=(0.11, 0.25), ncol=4, frameon=False,
               fontsize=10, handlelength=1.2, columnspacing=1.5)
    reduction = 100 * (1 - data["V2"]["values"][2] / data["Vanilla"]["values"][2])
    fig.text(0.12, 0.135, f"数据准备累计工作时间：22,765 秒 → 2,744 秒（下降 {reduction:.1f}%）",
             fontsize=12, fontweight="bold", color=COLORS[2])
    fig.text(0.12, 0.065,
             "注：V2 组装时间为 4 个进程之和，阶段可重叠；离线索引为共同冷启动成本，V2 实测运行复用该索引。",
             fontsize=9, color="#69778B")
    save_plot(fig, "arxiv_comparison")


def main() -> None:
    set_style()
    data = load_measurements()
    individual("Vanilla", data["Vanilla"], data["Vanilla"]["values"][2])
    individual("V2", data["V2"], data["Vanilla"]["values"][2])
    comparison(data)


if __name__ == "__main__":
    main()
