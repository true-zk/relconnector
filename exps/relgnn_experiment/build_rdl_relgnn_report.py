"""Build the ten-dataset RDL/RelGNN timing workbook and figure."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import geometric_mean, mean

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "exps/benchmarks"
OUT = BENCH / "rdl-relgnn-summary-20261103"
RDL_VANILLA = BENCH / "strict-vanilla-20261027"
RDL_NINE = BENCH / "hyperbatch-node-block-nine-20261030"
RDL_SALT = BENCH / "hyperbatch-node-block-rel-salt-20261029"
RDL_SINGLE = BENCH / "rdl-single-process-20261103"
REL = BENCH / "relgnn-three-way-20261101"
DATASETS = (
    "rel-f1",
    "rel-avito",
    "rel-hm",
    "rel-trial",
    "rel-arxiv",
    "rel-salt",
    "rel-stack",
    "rel-amazon",
    "rel-ratebeer",
    "rel-event",
)
SHORT = {
    "rel-f1": "F1",
    "rel-avito": "Avito",
    "rel-hm": "H&M",
    "rel-trial": "Trial",
    "rel-arxiv": "ArXiv",
    "rel-salt": "SALT",
    "rel-stack": "Stack",
    "rel-amazon": "Amazon",
    "rel-ratebeer": "RateBeer",
    "rel-event": "Event",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> dict:
    return json.loads(path.read_text().splitlines()[-1])


def checked_equal(left: object, right: object, context: str) -> None:
    if left != right:
        raise ValueError(f"{context}: {left!r} != {right!r}")


def collect() -> tuple[list[dict], list[dict]]:
    report = read_json(RDL_VANILLA / "requested-five-epoch-report.json")
    reports = {row["dataset"]: row for row in report["rows"]}
    checked_equal(set(reports), set(DATASETS), "RDL report dataset set")
    checked_equal(read_json(REL / "status.json")["status"], "complete", "RelGNN run")
    rows: list[dict] = []
    epoch_rows: list[dict] = []

    for dataset in DATASETS:
        vanilla_path = RDL_VANILLA / f"{dataset}.json"
        rdl_vanilla = read_json(vanilla_path)
        rdl_report = reports[dataset]
        rel_dir = REL / dataset
        rel_summary = read_json(rel_dir / "summary.json")
        rel_vanilla = read_json(rel_dir / "vanilla.json")
        rel_v2_vanilla = read_json(rel_dir / "v2-vanilla.json")
        rel_optimized = read_jsonl(rel_dir / "v2-optimized.jsonl")
        rel_epochs = rel_optimized["pipeline"]["epochs"]
        rdl_single_path = RDL_SINGLE / dataset / "single-process.json"
        rdl_single = read_json(rdl_single_path) if rdl_single_path.exists() else None
        if dataset == "rel-salt":
            rdl_path = RDL_SALT / "direct-summary.json"
            rdl_optimized = read_json(rdl_path)
            rdl_epochs = rdl_optimized["epochs"]
            rdl_epoch_time = [float(epoch["wall_s"]) for epoch in rdl_epochs]
            rdl_total = float(rdl_optimized["training_wall_s"])
            rdl_batches = [int(epoch["completed_batches"]) for epoch in rdl_epochs]
            rdl_examples = [int(epoch["completed_examples"]) for epoch in rdl_epochs]
            rdl_fingerprint = rdl_optimized["replay_fingerprint"]
        else:
            rdl_path = RDL_NINE / dataset / "summary.json"
            rdl_optimized = read_json(rdl_path)
            node = rdl_optimized["node_block"]
            rdl_epochs = node["epochs"]
            rdl_epoch_time = [float(epoch["wall_duration_s"]) for epoch in rdl_epochs]
            rdl_total = float(node["training_wall_s"])
            rdl_batches = [int(epoch["steps"]) for epoch in rdl_epochs]
            rdl_examples = [int(epoch["examples"]) for epoch in rdl_epochs]
            rdl_fingerprint = rdl_optimized["replay_fingerprint"]

        batches = int(rel_summary["batches_per_epoch"])
        examples = int(rel_summary["examples_per_epoch"])
        checked_equal(len(rdl_epochs), 5, f"{dataset} RDL optimized epochs")
        checked_equal(len(rel_epochs), 5, f"{dataset} RelGNN optimized epochs")
        checked_equal(set(rdl_batches), {batches}, f"{dataset} RDL batch count")
        checked_equal(set(rdl_examples), {examples}, f"{dataset} RDL sample count")
        checked_equal(
            {int(epoch["steps"]) for epoch in rel_epochs},
            {batches},
            f"{dataset} RelGNN batch count",
        )
        checked_equal(
            {int(epoch["examples"]) for epoch in rel_epochs},
            {examples},
            f"{dataset} RelGNN sample count",
        )
        checked_equal(
            rdl_fingerprint,
            rel_summary["replay_fingerprint"],
            f"{dataset} optimized replay",
        )
        if rdl_single is not None:
            checked_equal(rdl_single["status"], "ok", f"{dataset} RDL single status")
            checked_equal(
                rdl_single["model"], "graphsage", f"{dataset} RDL single model"
            )
            checked_equal(
                rdl_single["fingerprint"],
                rdl_fingerprint,
                f"{dataset} RDL single replay",
            )
            checked_equal(
                rdl_single["completed_batches"],
                batches,
                f"{dataset} RDL single batches",
            )
            checked_equal(
                rdl_single["completed_examples"],
                examples,
                f"{dataset} RDL single examples",
            )
        checked_equal(
            rdl_vanilla["epochs"][0]["batches"],
            batches,
            f"{dataset} vanilla batch count",
        )
        checked_equal(
            rdl_vanilla["epochs"][0]["examples"],
            examples,
            f"{dataset} vanilla sample count",
        )
        for group, result in (("vanilla", rel_vanilla), ("v2_vanilla", rel_v2_vanilla)):
            checked_equal(result["status"], "ok", f"{dataset} {group} status")
            checked_equal(result["model"], "relgnn", f"{dataset} {group} model")
            checked_equal(
                result["fingerprint"], rdl_fingerprint, f"{dataset} {group} replay"
            )
            checked_equal(
                result["completed_batches"], batches, f"{dataset} {group} batches"
            )
            checked_equal(
                result["completed_examples"], examples, f"{dataset} {group} examples"
            )
        checked_equal(rel_optimized["status"], "ok", f"{dataset} RelGNN optimized")
        checked_equal(
            rel_optimized["completed_batches"],
            batches * 5,
            f"{dataset} RelGNN optimized total batches",
        )
        checked_equal(
            rel_optimized["completed_examples"],
            examples * 5,
            f"{dataset} RelGNN optimized total examples",
        )
        checked_equal(
            rdl_vanilla["epochs"][0]["training_loop_s"],
            rdl_report["first_epoch_training_loop_s"],
            f"{dataset} RDL first epoch",
        )

        rdl_vanilla_epochs = rdl_vanilla["epochs"]
        rdl_vanilla_later_measured = len(rdl_vanilla_epochs) == 5
        rdl_later_reference = (
            mean(float(e["training_loop_s"]) for e in rdl_vanilla_epochs[1:])
            if rdl_vanilla_later_measured
            else float(rdl_vanilla_epochs[0]["training_loop_s"])
        )
        rdl_first = float(rdl_vanilla_epochs[0]["training_loop_s"])
        rel_first = float(rel_vanilla["training_s"])
        rel_v2_first = float(rel_v2_vanilla["training_s"])
        rel_epoch_time = [float(epoch["wall_duration_s"]) for epoch in rel_epochs]
        rel_total = float(rel_optimized["pipeline"]["duration_s"])
        rdl_5 = float(rdl_report["reported_five_epoch_training_loop_s"])
        rel_5_estimate = 5 * rel_first

        rows.append(
            {
                "dataset": dataset,
                "task": rel_summary["task"],
                "batches": batches,
                "examples": examples,
                "rdl_vanilla_first": rdl_first,
                "rdl_single_first": (
                    float(rdl_single["training_s"]) if rdl_single is not None else None
                ),
                "rdl_vanilla_measured_epochs": len(rdl_vanilla_epochs),
                "rdl_vanilla_later_reference": rdl_later_reference,
                "rdl_vanilla_later_basis": (
                    "实测第2–5轮均值" if rdl_vanilla_later_measured else "用首轮代替"
                ),
                "rdl_vanilla_5": rdl_5,
                "rdl_vanilla_5_basis": (
                    "实测5轮" if rdl_vanilla_later_measured else "首轮×5推算"
                ),
                "rdl_opt_first": rdl_epoch_time[0],
                "rdl_opt_warm": mean(rdl_epoch_time[1:]),
                "rdl_opt_5": rdl_total,
                "rel_vanilla_first": rel_first,
                "rel_v2_vanilla_first": rel_v2_first,
                "rel_vanilla_5_estimate": rel_5_estimate,
                "rel_opt_first": rel_epoch_time[0],
                "rel_opt_warm": mean(rel_epoch_time[1:]),
                "rel_opt_5": rel_total,
                "rdl_opt_source": str(rdl_path),
                "rdl_vanilla_source": str(vanilla_path),
                "rel_source": str(rel_dir),
                "replay_fingerprint": rdl_fingerprint,
            }
        )

        for index, epoch in enumerate(rdl_vanilla_epochs):
            epoch_rows.append(
                {
                    "model": "RDL / GraphSAGE",
                    "dataset": dataset,
                    "method": "原始 vanilla",
                    "epoch": index + 1,
                    "wall": float(epoch["training_loop_s"]),
                    "batches": epoch["batches"],
                    "examples": epoch["examples"],
                    "sampling": epoch.get("sampling_s"),
                    "feature_read": epoch.get("feature_read_assemble_s"),
                    "feature_process": None,
                    "train": epoch.get("training_s"),
                    "ready_wait": None,
                    "cache_hit": None,
                    "block_reads": None,
                    "block_bytes": None,
                    "sql_text_bytes": None,
                    "source": str(vanilla_path),
                }
            )
        for index, epoch in enumerate(rdl_epochs):
            epoch_rows.append(
                {
                    "model": "RDL / GraphSAGE",
                    "dataset": dataset,
                    "method": "v2 node_block",
                    "epoch": index + 1,
                    "wall": rdl_epoch_time[index],
                    "batches": rdl_batches[index],
                    "examples": rdl_examples[index],
                    "sampling": epoch.get("sampling_s"),
                    "feature_read": epoch.get("feature_read_s"),
                    "feature_process": epoch.get("feature_process_s"),
                    "train": epoch.get("train_s"),
                    "ready_wait": epoch.get("ready_wait_s"),
                    "cache_hit": epoch.get("text_cache_hit_rate"),
                    "block_reads": epoch.get("block_reads"),
                    "block_bytes": epoch.get("block_read_bytes"),
                    "sql_text_bytes": epoch.get("sql_text_utf8_bytes"),
                    "source": str(rdl_path),
                }
            )
        if rdl_single is not None:
            epoch_rows.append(
                {
                    "model": "RDL / GraphSAGE",
                    "dataset": dataset,
                    "method": "单进程",
                    "epoch": 1,
                    "wall": float(rdl_single["training_s"]),
                    "batches": batches,
                    "examples": examples,
                    "sampling": None,
                    "feature_read": None,
                    "feature_process": None,
                    "train": None,
                    "ready_wait": None,
                    "cache_hit": None,
                    "block_reads": None,
                    "block_bytes": None,
                    "sql_text_bytes": None,
                    "source": str(rdl_single_path),
                }
            )
        for method, result, source in (
            ("原始 vanilla", rel_vanilla, rel_dir / "vanilla.json"),
            ("单进程", rel_v2_vanilla, rel_dir / "v2-vanilla.json"),
        ):
            epoch_rows.append(
                {
                    "model": "RelGNN",
                    "dataset": dataset,
                    "method": method,
                    "epoch": 1,
                    "wall": float(result["training_s"]),
                    "batches": batches,
                    "examples": examples,
                    "sampling": None,
                    "feature_read": None,
                    "feature_process": None,
                    "train": None,
                    "ready_wait": None,
                    "cache_hit": None,
                    "block_reads": None,
                    "block_bytes": None,
                    "sql_text_bytes": None,
                    "source": str(source),
                }
            )
        for index, epoch in enumerate(rel_epochs):
            epoch_rows.append(
                {
                    "model": "RelGNN",
                    "dataset": dataset,
                    "method": "v2 node_block",
                    "epoch": index + 1,
                    "wall": rel_epoch_time[index],
                    "batches": epoch["steps"],
                    "examples": epoch["examples"],
                    "sampling": epoch.get("sampling_s"),
                    "feature_read": epoch.get("feature_read_s"),
                    "feature_process": epoch.get("feature_process_s"),
                    "train": epoch.get("train_s"),
                    "ready_wait": epoch.get("ready_wait_s"),
                    "cache_hit": epoch.get("text_cache_hit_rate"),
                    "block_reads": epoch.get("block_reads"),
                    "block_bytes": epoch.get("block_read_bytes"),
                    "sql_text_bytes": epoch.get("sql_text_utf8_bytes"),
                    "source": str(rel_dir / "v2-optimized.jsonl"),
                }
            )

    return rows, epoch_rows


def number_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def figure(rows: list[dict]) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    specs = (
        ("vanilla", "#3568B0", "..."),
        ("single worker", "#D35B4C", "---"),
        ("relx-e1", "#168A87", "|||"),
        ("relx-e2", "#E19535", "///"),
        ("relx", "#8B5BA6", "xxx"),
    )
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.8), dpi=180)
    fig.suptitle("Training time across ten relational datasets", fontsize=17, y=0.985)
    legend = [
        Patch(
            facecolor="white", edgecolor=color, hatch=hatch, linewidth=1.4, label=label
        )
        for label, color, hatch in specs
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=True,
        fontsize=9.5,
    )
    for ax, row in zip(axes.flat, rows, strict=True):
        values = (
            (
                row["rdl_vanilla_first"],
                row["rdl_single_first"],
                row["rdl_opt_first"],
                row["rdl_opt_warm"],
                (row["rdl_opt_first"] + 4 * row["rdl_opt_warm"]) / 5,
            ),
            (
                row["rel_vanilla_first"],
                row["rel_v2_vanilla_first"],
                row["rel_opt_first"],
                row["rel_opt_warm"],
                (row["rel_opt_first"] + 4 * row["rel_opt_warm"]) / 5,
            ),
        )
        ymax = max(v for group in values for v in group if v is not None) * 1.25
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.55, 1.55)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#D8DDE3", linewidth=0.65)
        ax.set_xticks([0, 1], ["RDL", "RelGNN"])
        ax.set_title(SHORT[row["dataset"]], fontsize=12, pad=8, fontweight="semibold")
        for group_index, group in enumerate(values):
            for method_index, value in enumerate(group):
                x = group_index + (method_index - 2) * 0.15
                if value is None:
                    ax.text(
                        x,
                        ymax * 0.035,
                        "pending",
                        color="#8E9297",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )
                    continue
                _, color, hatch = specs[method_index]
                ax.bar(
                    x,
                    value,
                    width=0.125,
                    facecolor="white",
                    edgecolor=color,
                    hatch=hatch,
                    linewidth=1.25,
                )
                ax.text(
                    x,
                    value + ymax * 0.012,
                    number_label(value),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color=color,
                    fontsize=7.9,
                    fontweight="medium",
                )
        if row["dataset"] in ("rel-f1", "rel-salt"):
            ax.set_ylabel("Time per epoch (s)")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.tick_params(axis="x", labelsize=9.5)
    completion_note = (
        "Pending RDL single-worker results are marked."
        if any(row["rdl_single_first"] is None for row in rows)
        else "All ten datasets have RDL single-worker results."
    )
    fig.text(
        0.5,
        0.012,
        "vanilla / single worker / relx-e1 = epoch 1; "
        "relx-e2 = mean of epochs 2–5; relx = (e1 + 4 × e2) / 5. "
        f"{completion_note} Each dataset has its own linear y-axis.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.subplots_adjust(
        top=0.865, bottom=0.09, left=0.055, right=0.99, wspace=0.38, hspace=0.53
    )
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "training-time-comparison.png"
    fig.savefig(png, dpi=200)
    fig.savefig(OUT / "training-time-comparison.pdf")
    plt.close(fig)
    return png


def style_sheet(ws, header_row: int, widths: dict[str, int] | None = None) -> None:
    ws.freeze_panes = (
        f"E{header_row + 1}" if ws.title == "数据集对比" else f"D{header_row + 1}"
    )
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(ws.max_column)}{ws.max_row}"
    ws.sheet_view.showGridLines = False
    for cell in ws[header_row]:
        cell.fill = PatternFill("solid", fgColor="20375B")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(
            wrap_text=True, vertical="center", horizontal="center"
        )
    ws.row_dimensions[header_row].height = 38
    for row in ws.iter_rows(min_row=header_row + 1):
        for cell in row:
            cell.fill = PatternFill(
                "solid", fgColor="F1F5FA" if cell.row % 2 == 0 else "FFFFFF"
            )
            cell.alignment = Alignment(vertical="center")
            if isinstance(cell.value, float):
                cell.number_format = "#,##0.00"
    for col in range(1, ws.max_column + 1):
        letter = get_column_letter(col)
        ws.column_dimensions[letter].width = (widths or {}).get(letter, 17)


def workbook(rows: list[dict], epoch_rows: list[dict], png: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "数据集对比"
    ws.merge_cells("A1:D1")
    ws.merge_cells("E1:N1")
    ws.merge_cells("O1:X1")
    ws.merge_cells("Y1:Z1")
    for cell, title, color in (
        ("A1", "Dataset / Task", "43536B"),
        ("E1", "RDL / GraphSAGE", "315F9B"),
        ("O1", "RelGNN", "15817D"),
        ("Y1", "Verification", "43536B"),
    ):
        c = ws[cell]
        c.value = title
        c.fill = PatternFill("solid", fgColor=color)
        c.font = Font(color="FFFFFF", bold=True, size=12)
        c.alignment = Alignment(horizontal="center")
    ws.row_dimensions[1].height = 25
    headers = [
        "数据集",
        "任务",
        "每轮 batch",
        "每轮样本",
        "原始 vanilla 首轮 (s)",
        "单进程首轮 (s)",
        "v2 缓存首轮 (s)",
        "v2 缓存后四轮均值 (s)",
        "原始 vanilla 五轮 (s)",
        "v2 缓存五轮实测 (s)",
        "首轮加速比",
        "后四轮加速比*",
        "五轮加速比*",
        "vanilla 五轮口径",
        "原始 vanilla 首轮 (s)",
        "单进程首轮 (s)",
        "v2 缓存首轮 (s)",
        "v2 缓存后四轮均值 (s)",
        "原始 vanilla 首轮×5 (s)",
        "v2 缓存五轮实测 (s)",
        "首轮加速比",
        "后四轮加速比*",
        "五轮加速比*",
        "单进程加速比",
        "优化版回放相同",
        "备注",
    ]
    ws.append(headers)
    for r in rows:
        ws.append(
            [
                r["dataset"],
                r["task"],
                r["batches"],
                r["examples"],
                r["rdl_vanilla_first"],
                r["rdl_single_first"],
                r["rdl_opt_first"],
                r["rdl_opt_warm"],
                r["rdl_vanilla_5"],
                r["rdl_opt_5"],
                r["rdl_vanilla_first"] / r["rdl_opt_first"],
                r["rdl_vanilla_later_reference"] / r["rdl_opt_warm"],
                r["rdl_vanilla_5"] / r["rdl_opt_5"],
                r["rdl_vanilla_5_basis"],
                r["rel_vanilla_first"],
                r["rel_v2_vanilla_first"],
                r["rel_opt_first"],
                r["rel_opt_warm"],
                r["rel_vanilla_5_estimate"],
                r["rel_opt_5"],
                r["rel_vanilla_first"] / r["rel_opt_first"],
                r["rel_vanilla_first"] / r["rel_opt_warm"],
                r["rel_vanilla_5_estimate"] / r["rel_opt_5"],
                r["rel_vanilla_first"] / r["rel_v2_vanilla_first"],
                "是",
                r["rdl_vanilla_later_basis"],
            ]
        )
    style_sheet(ws, 2, {"A": 17, "B": 29, "C": 13, "D": 14, "N": 18, "Y": 17, "Z": 19})
    ws.freeze_panes = "E3"
    ws.auto_filter.ref = f"A2:Z{ws.max_row}"
    for row in range(3, ws.max_row + 1):
        for col in (11, 12, 13, 21, 22, 23, 24):
            ws.cell(row, col).number_format = '0.00"×"'
        for col in (14, 26):
            ws.cell(row, col).alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[row].height = 24

    details = wb.create_sheet("逐轮实测")
    detail_headers = [
        "模型",
        "数据集",
        "方法",
        "轮次",
        "训练墙钟时间 (s)",
        "batch 数",
        "样本数",
        "采样累计 (s)",
        "特征读取累计 (s)",
        "特征处理累计 (s)",
        "模型训练累计 (s)",
        "等待特征累计 (s)",
        "文本缓存命中率",
        "块读取次数",
        "块读取字节",
        "SQL 文本字节",
        "来源文件",
    ]
    details.append(detail_headers)
    for r in epoch_rows:
        details.append(
            [
                r["model"],
                r["dataset"],
                r["method"],
                r["epoch"],
                r["wall"],
                r["batches"],
                r["examples"],
                r["sampling"],
                r["feature_read"],
                r["feature_process"],
                r["train"],
                r["ready_wait"],
                r["cache_hit"],
                r["block_reads"],
                r["block_bytes"],
                r["sql_text_bytes"],
                r["source"],
            ]
        )
    style_sheet(
        details,
        1,
        {
            "A": 20,
            "B": 18,
            "C": 18,
            "D": 10,
            "E": 19,
            "F": 12,
            "G": 14,
            "M": 18,
            "Q": 105,
        },
    )
    for row in range(2, details.max_row + 1):
        details.cell(row, 13).number_format = "0.00%"

    notes = wb.create_sheet("口径与来源")
    notes.append(["项目", "说明"])
    notes_data = [
        ("单位", "秒；图中以每轮时间为口径，不含离线建图/预热。"),
        (
            "图例",
            "vanilla、single worker、relx-e1 均为首轮时间；relx-e2 为第 2–5 轮平均时间；relx 为优化版五轮每轮平均时间，按 (relx-e1 + 4 × relx-e2) ÷ 5 计算。Excel 主表另保留五轮训练总时间。relx 对应 v2 node_block 磁盘文本缓存及多进程实现。",
        ),
        (
            "RDL 模型",
            "此前 GraphSAGE/RDL 实验。原始 vanilla 为逐 batch 精确 SQL 同步实现；新增单进程组使用 v2 同步实现、official 文本编码、无磁盘文本缓存，实测 1 轮；优化版为 v2 node_block 磁盘文本缓存、4 个特征进程。",
        ),
        (
            "RelGNN 模型",
            "RelGNN 复合路径模型；三组为原始 vanilla、单进程、v2 node_block 优化版。前两组各实测 1 轮，优化版实测 5 轮。",
        ),
        ("首轮加速比", "原始 vanilla 实测首轮时间 ÷ v2 node_block 实测首轮时间。"),
        (
            "后四轮加速比*",
            "RDL 前四个数据集使用原始 vanilla 实测第 2–5 轮均值；RDL 其余六个以及全部 RelGNN 使用原始 vanilla 首轮作为代理。",
        ),
        (
            "五轮加速比*",
            "RDL 前四个数据集的分子为实测五轮；其余六个数据集的分子为首轮×5。RelGNN 全部数据集的分子为首轮×5。分母均为优化版实测五轮。",
        ),
        ("首轮与后四轮", "优化版首轮为冷文本缓存；后四轮为同一运行的第 2–5 轮均值。"),
        (
            "单进程",
            "两模型均使用 v2 同步实现、official 文本编码、每窗口 1 batch，不启用磁盘文本缓存或多进程。RDL 补跑结果尚未完成时，图中标 pending，Excel 对应单元格留空。",
        ),
        (
            "样本与回放",
            "全部十个数据集两模型的优化版 replay 指纹一致；首轮的 batch 数与样本数也与原始 vanilla 一致。",
        ),
        (
            "计时限制",
            "训练阶段的采样、特征读取、特征处理、模型训练属于并行流水线累计工作时间，不能直接相加得到墙钟时间。RelGNN 缺少逐轮特征阶段统计的单元格留空。",
        ),
        (
            "跨模型比较",
            "两模型在不同日期运行；同数据集的采样 replay 一致，但跨模型绝对耗时仍受架构与运行环境影响。RDL 原始 vanilla 与新增单进程组均在 CUDA0 上运行，RelGNN 单进程组在 CUDA1 上运行，两卡均为 RTX 4090。优化版与原始 vanilla 还包含执行方式和文本编码的差异，不能把全部加速归因于缓存。",
        ),
        (
            "RDL 单进程比较限制",
            "RDL 原始 vanilla 与新增单进程组在不同日期运行，初始化路径不同，未配平操作系统页缓存与磁盘负载。后者在 Amazon、Trial、Event 等数据集上的 SQL 查询耗时明显高于相同查询次数的 RelGNN 单进程组；因此 0.85× 是这两次运行条件下的实测几何平均，不应归因于单进程代码本身。",
        ),
        (
            "RDL 原始 vanilla 来源",
            str(RDL_VANILLA / "requested-five-epoch-report.json"),
        ),
        ("RDL 优化版 9 数据集来源", str(RDL_NINE)),
        ("RDL 优化版 SALT 来源", str(RDL_SALT / "direct-summary.json")),
        ("RDL 单进程来源", str(RDL_SINGLE)),
        ("RelGNN 来源", str(REL)),
    ]
    for item in notes_data:
        notes.append(item)
    style_sheet(notes, 1, {"A": 27, "B": 125})
    notes.auto_filter.ref = f"A1:B{notes.max_row}"
    for row in range(2, notes.max_row + 1):
        notes.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="center")
        notes.row_dimensions[row].height = (
            60 if len(str(notes.cell(row, 2).value)) > 110 else 34
        )

    speedups = wb.create_sheet("平均加速比")
    speedups.append(
        ["模型", "single worker / vanilla", "有效数据集数", "relx / vanilla", "有效数据集数"]
    )
    all_single: list[float] = []
    all_relx: list[float] = []
    for model, prefix, single_key in (
        ("RDL", "rdl", "rdl_single_first"),
        ("RelGNN", "rel", "rel_v2_vanilla_first"),
    ):
        single_ratios = [
            row[f"{prefix}_vanilla_first"] / row[single_key]
            for row in rows
            if row[single_key] is not None
        ]
        relx_ratios = [
            row[f"{prefix}_vanilla_first"]
            / ((row[f"{prefix}_opt_first"] + 4 * row[f"{prefix}_opt_warm"]) / 5)
            for row in rows
        ]
        all_single.extend(single_ratios)
        all_relx.extend(relx_ratios)
        speedups.append(
            [
                model,
                geometric_mean(single_ratios) if single_ratios else None,
                len(single_ratios),
                geometric_mean(relx_ratios),
                len(relx_ratios),
            ]
        )
    speedups.append(
        ["两模型合计", geometric_mean(all_single), len(all_single), geometric_mean(all_relx), len(all_relx)]
    )
    style_sheet(speedups, 1, {"A": 18, "B": 28, "C": 17, "D": 24, "E": 17})
    for row in range(2, speedups.max_row + 1):
        for col in (2, 4):
            speedups.cell(row, col).number_format = '0.00"×"'
    speedups.append(
        [
            "口径",
            "每个数据集先以 vanilla 首轮时间除以对应方法的单轮时间，再对速度比取几何平均。relx 单轮时间 = (relx-e1 + 4 × relx-e2) ÷ 5；两模型合计按每个模型-数据集组合等权计算。",
        ]
    )
    speedups.merge_cells("B5:E5")
    speedups["B5"].alignment = Alignment(wrap_text=True, vertical="center")
    speedups.row_dimensions[5].height = 60
    speedups.append(
        [
            "比较限制",
            "RDL single worker 与原始 vanilla 在不同日期运行，SQL 页缓存和磁盘负载未配平；RDL 的 0.85× 仅反映这两次运行的实测结果。详见“口径与来源”。",
        ]
    )
    speedups.merge_cells("B6:E6")
    speedups["B6"].alignment = Alignment(wrap_text=True, vertical="center")
    speedups.row_dimensions[6].height = 48

    plot = wb.create_sheet("实验图")
    plot.sheet_view.showGridLines = False
    plot.add_image(ExcelImage(str(png)), "A1")
    plot.column_dimensions["A"].width = 25

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rdl-relgnn-ten-dataset-summary.xlsx"
    wb.save(path)
    return path


def main() -> None:
    rows, epoch_rows = collect()
    png = figure(rows)
    xlsx = workbook(rows, epoch_rows, png)
    print(f"Excel: {xlsx}")
    print(f"Figure: {png}")
    print(f"PDF: {OUT / 'training-time-comparison.pdf'}")
    print(f"Datasets: {len(rows)}, measured epoch rows: {len(epoch_rows)}")


if __name__ == "__main__":
    main()
