"""Plot ten-dataset, five-epoch phase shares from the paired GraphSAGE runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
VANILLA_DIR = ROOT / "exps/benchmarks/strict-vanilla-20261027"
V2_NINE_DIR = ROOT / "exps/benchmarks/hyperbatch-node-block-nine-20261030"
V2_SALT = ROOT / "exps/benchmarks/hyperbatch-node-block-rel-salt-20261029/direct.jsonl"
OUTPUT = ROOT / "thesis/figures/training_time_share"
STAGES = ("离线图索引构建", "采样", "数据准备", "训练")
COLORS = ("#506FAD", "#42A69B", "#DC675D", "#D5A331")


def read_last_json(path: Path) -> dict:
    return json.loads(path.read_text().splitlines()[-1])


def load_data() -> dict[str, dict]:
    result = {}
    for vanilla_path in sorted(VANILLA_DIR.glob("rel-*.json")):
        name = vanilla_path.stem
        vanilla = json.loads(vanilla_path.read_text())
        v2_path = V2_SALT if name == "rel-salt" else V2_NINE_DIR / name / "node_block.jsonl"
        v2 = read_last_json(v2_path)
        assert v2["status"] == "ok" and len(v2["pipeline"]["epochs"]) == 5, name
        assert vanilla["epochs"][0]["batches"] == v2["pipeline"]["replay"]["batches_per_epoch"], name
        first = vanilla["epochs"][0]
        graph_s = vanilla["prepare_breakdown_s"]["graph_build_s"]
        workers = v2["pipeline"]["workers"]
        assert workers[0]["stage"] == "sampling", name
        assert all(worker["stage"] == "assembly" for worker in workers[1:]), name
        result[name] = {
            "Vanilla": (
                graph_s,
                5 * first["sampling_s"],
                5 * first["feature_read_assemble_s"],
                5 * first["training_s"],
            ),
            "V2": (
                graph_s,  # Shared one-time cold build; V2 used its cached topology.
                sum(workers[0]["epoch_sample_s"].values()),
                sum(
                    seconds
                    for worker in workers[1:]
                    for epoch in worker["diagnostics"]["epoch_stage"].values()
                    for seconds in epoch.values()
                ),
                sum(epoch["train_s"] for epoch in v2["pipeline"]["epochs"]),
            ),
            "vanilla_wall_s": 5 * first["training_loop_s"],
            "v2_wall_s": sum(epoch["wall_duration_s"] for epoch in v2["pipeline"]["epochs"]),
        }
    assert len(result) == 10
    return result


def shares(values: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(values)
    return tuple(100 * value / total for value in values)


def setup() -> None:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({
        "font.family": "Noto Sans CJK SC",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "#263347",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)


def bar(ax, y: float, percentages: tuple[float, ...], *, height: float = .31) -> None:
    left = 0.0
    for value, color in zip(percentages, COLORS, strict=True):
        ax.barh(y, value, left=left, height=height, color=color,
                edgecolor="white", linewidth=1.1)
        left += value
    data_mid = sum(percentages[:2]) + percentages[2] / 2
    if percentages[2] >= 18:
        ax.text(data_mid, y, f"{percentages[2]:.1f}%", va="center", ha="center",
                color="white", fontsize=10, fontweight="bold")


def clean_axis(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.tick_params(length=0, colors="#69778B", labelsize=9)
    ax.grid(axis="x", color="#E7EBF0")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def wall_bars(ax, v2_percent: float, *, label_size: int = 10) -> None:
    ax.barh(1, 100, height=.32, color="#506FAD")
    ax.barh(0, v2_percent, height=.32, color="#DC675D")
    ax.text(50, 1, "100%", ha="center", va="center", color="white",
            fontsize=label_size, fontweight="bold")
    ax.text(v2_percent + 1.2, 0, f"{v2_percent:.1f}%", ha="left", va="center",
            color="#DC675D", fontsize=label_size, fontweight="bold")
    ax.set_ylim(-.48, 1.48)
    ax.set_yticks([1, 0], ["Vanilla ×5", "V2 实测"])
    clean_axis(ax)


def save(fig, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_average(data: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means = {
        method: tuple(mean(shares(item[method])[i] for item in data.values()) for i in range(4))
        for method in ("Vanilla", "V2")
    }
    ratios = [row["vanilla_wall_s"] / row["v2_wall_s"] for row in data.values()]
    geometric_speedup = math.exp(mean(math.log(x) for x in ratios))
    vanilla_prep_mean = mean(row["Vanilla"][2] for row in data.values())
    v2_prep_mean = mean(row["V2"][2] for row in data.values())
    fig = plt.figure(figsize=(12.5, 9.0))
    ax_wall = fig.add_axes((.13, .59, .82, .18))
    ax_phase = fig.add_axes((.13, .27, .82, .18))
    wall_bars(ax_wall, 100 / geometric_speedup, label_size=12)
    ax_wall.tick_params(axis="y", labelsize=12, colors="#263347", pad=12)
    bar(ax_phase, 1, means["Vanilla"], height=.36)
    bar(ax_phase, 0, means["V2"], height=.36)
    ax_phase.set_ylim(-.5, 1.5)
    ax_phase.set_yticks([1, 0], ["Vanilla ×5", "V2 实测"])
    ax_phase.tick_params(axis="y", labelsize=12, colors="#263347", pad=12)
    clean_axis(ax_phase)
    fig.text(.13, .94, "十数据集平均｜RDL 五轮训练耗时与阶段结构", fontsize=21, fontweight="bold")
    fig.text(.13, .885, "上：墙钟时间相对 Vanilla 的比例（十任务加速倍数的几何平均）",
             fontsize=11, color="#69778B")
    fig.text(.13, .81, f"五轮训练墙钟时间：平均加速 {geometric_speedup:.2f}×",
             fontsize=14, fontweight="bold", color="#DC675D")
    fig.text(.13, .495, "下：四阶段累计工作时间占比（各数据集等权平均）",
             fontsize=11, color="#69778B")
    fig.legend(handles=[Patch(facecolor=c, label=s) for s, c in zip(STAGES, COLORS, strict=True)],
               loc="center left", bbox_to_anchor=(.13, .19), ncol=4, frameon=False,
               fontsize=11, columnspacing=1.7)
    fig.text(.13, .13,
             f"数据准备工作耗时均值：{vanilla_prep_mean:,.0f} → {v2_prep_mean:,.0f} 秒"
             f"（{vanilla_prep_mean / v2_prep_mean:.2f}×）；占比 {means['Vanilla'][2]:.1f}% → {means['V2'][2]:.1f}%",
             fontsize=11.5, color=COLORS[2], fontweight="bold")
    fig.text(.13, .065,
             "注：Vanilla 首轮耗时 ×5；V2 实测五轮。阶段占比来自进程累计工作时间，并非墙钟时间分解。",
             fontsize=9.5, color="#69778B")
    fig.text(.13, .035,
             "离线索引为共同一次性冷启动成本，不计入上方训练墙钟时间；V2 运行复用了该索引。",
             fontsize=9.5, color="#69778B")
    save(fig, "average_10datasets_5epoch")
    return means["Vanilla"], means["V2"]


def plot_each(data: dict) -> None:
    ordered = sorted(data, key=lambda name: data[name]["vanilla_wall_s"], reverse=True)
    fig, axes = plt.subplots(10, 2, figsize=(18.0, 18.0))
    fig.subplots_adjust(left=.10, right=.97, top=.89, bottom=.085, hspace=.49, wspace=.24)
    for (ax_wall, ax_phase), name in zip(axes, ordered, strict=True):
        row = data[name]
        speedup = row["vanilla_wall_s"] / row["v2_wall_s"]
        wall_bars(ax_wall, 100 / speedup, label_size=9)
        ax_wall.tick_params(axis="y", labelsize=8, colors="#263347", pad=7)
        ax_wall.set_title(
            f"{name}  {row['vanilla_wall_s']:,.0f} → {row['v2_wall_s']:,.0f} 秒"
            f"  ｜  {speedup:.1f}× 加速", loc="left", fontsize=11, pad=8, fontweight="bold")
        bar(ax_phase, 1, shares(row["Vanilla"]))
        bar(ax_phase, 0, shares(row["V2"]))
        ax_phase.set_ylim(-.48, 1.48)
        ax_phase.set_yticks([1, 0], ["Vanilla ×5", "V2 实测"])
        ax_phase.tick_params(axis="y", labelsize=8, colors="#263347", pad=7)
        clean_axis(ax_phase)
        ax_phase.set_title(f"四阶段占比｜数据准备 {row['Vanilla'][2]:,.0f} → {row['V2'][2]:,.0f} 秒", loc="left", fontsize=10,
                           pad=8, color="#69778B")
    fig.text(.10, .965, "每个数据集｜RDL 五轮训练耗时与阶段结构", fontsize=23, fontweight="bold")
    fig.text(.10, .925, "左：五轮训练墙钟时间相对比例与加速倍数    右：四阶段累计工作时间占比",
             fontsize=11, color="#69778B")
    fig.legend(handles=[Patch(facecolor=c, label=s) for s, c in zip(STAGES, COLORS, strict=True)],
               loc="center left", bbox_to_anchor=(.10, .052), ncol=4, frameon=False,
               fontsize=11, columnspacing=1.7)
    fig.text(.10, .021,
             "Vanilla 首轮 ×5；V2 实测五轮。V2 组装进程可重叠，右侧占比并非墙钟时间分解；离线索引为一次性成本。",
             fontsize=9.5, color="#69778B")
    save(fig, "each_dataset_5epoch")


def export_csv(data: dict) -> None:
    with (OUTPUT / "phase_times_5epoch.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "method", "training_wall_s", "speedup_x", "graph_index_s", "sampling_s", "data_prep_s",
                         "training_s", "graph_index_pct", "sampling_pct", "data_prep_pct", "training_pct"])
        for name, row in sorted(data.items()):
            speedup = row["vanilla_wall_s"] / row["v2_wall_s"]
            for method in ("Vanilla", "V2"):
                values = row[method]
                wall = row["vanilla_wall_s"] if method == "Vanilla" else row["v2_wall_s"]
                writer.writerow([name, method, f"{wall:.6f}", f"{speedup:.6f}",
                                 *[f"{x:.6f}" for x in values],
                                 *[f"{x:.6f}" for x in shares(values)]])


def main() -> None:
    setup()
    data = load_data()
    plot_average(data)
    plot_each(data)
    export_csv(data)


if __name__ == "__main__":
    main()
