"""Plot the by-type and text-length results of a feature profile JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload["status"] != "complete":
        parser.error("Input profile is incomplete")
    datasets = list(payload["datasets"])
    colors = {
        "text_embedded": "#3069a6",
        "categorical": "#dd8b3b",
        "timestamp": "#67a371",
        "numerical": "#9aa3af",
    }
    labels = {
        "text_embedded": "Text",
        "categorical": "Category",
        "timestamp": "Timestamp",
        "numerical": "Numerical",
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.7))
    x = list(range(len(datasets)))
    base = [0.0] * len(datasets)
    for kind in ("numerical", "timestamp", "categorical", "text_embedded"):
        values = []
        for dataset in datasets:
            types = payload["datasets"][dataset]["summary"]["types"]
            value = types.get(kind, {}).get("median_column_s_sum", 0.0)
            if kind == "categorical":
                value += types.get("multicategorical", {}).get(
                    "median_column_s_sum", 0.0
                )
            values.append(value * 1000)
        left.barh(x, values, left=base, color=colors[kind], label=labels[kind])
        base = [old + value for old, value in zip(base, values, strict=True)]
    left.set_yticks(x, [name.removeprefix("rel-") for name in datasets])
    left.invert_yaxis()
    left.set_xlabel("Column-attributed processing time (ms)")
    left.set_title("Feature conversion by dataset")
    left.legend(loc="lower right", fontsize=8)
    left.grid(axis="x", alpha=0.2)
    left.set_axisbelow(True)

    points = []
    for dataset, result in payload["datasets"].items():
        for table, table_result in result["tables"].items():
            for column, item in table_result["columns"].items():
                if item["type"] != "text_embedded" or not item["rows"]:
                    continue
                points.append(
                    (
                        item["total_chars"] / item["rows"] + 1,
                        item["median_total_s"] / item["rows"] * 1e6,
                        dataset,
                        table,
                        column,
                    )
                )
    right.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=20,
        alpha=0.55,
        color=colors["text_embedded"],
        edgecolors="none",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("Mean characters per text value + 1")
    right.set_ylabel("Processing time per value (µs)")
    right.set_title("Text length and per-value cost")
    right.grid(alpha=0.2, which="both")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
