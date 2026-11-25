"""Check RelX smoke reports against the selected v2 SQLite task metadata."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=Path("build/relx_ten_smoke")
    )
    args = parser.parse_args()
    failures = 0
    for config_path in sorted(args.config_dir.glob("rel-*.yaml")):
        dataset = config_path.stem
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        task = config["task"]
        database = Path(config["database"]["url"])
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as sql:
            original = sql.execute(
                "SELECT table_name, task_type, entity_table, entity_column, "
                "target_column, time_column FROM _relconnector_tasks "
                "WHERE task_name = ?", (task["name"],)
            ).fetchone()
            if original is None:
                print(f"{dataset}: missing v2 task metadata")
                failures += 1
                continue
            expected = {
                "table": original[0], "type": original[1],
                "entity_table": original[2], "entity_column": original[3],
                "label_column": original[4], "time_column": original[5],
            }
            task_matches = all(task.get(name) == value for name, value in expected.items())
            report_path = args.config_dir / dataset / "report.json"
            if not report_path.is_file():
                print(f"{dataset}: task={'match' if task_matches else 'DIFF'}, no report")
                failures += 1
                continue
            table = original[0].replace('"', '""')
            counts = dict(sql.execute(
                f'SELECT split, COUNT(*) FROM "{table}" GROUP BY split'
            ).fetchall())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        count_matches = all(
            report[f"{name}_rows"] == counts[split]
            for name, split in (
                ("train", "train"), ("validation", "val"), ("test", "test")
            )
        )
        finite = all(math.isfinite(float(epoch["train_loss"]))
                     for epoch in report["epochs"])
        good = task_matches and count_matches and finite
        failures += not good
        v2_path = args.config_dir / f"v2-{dataset.removeprefix('rel-')}.jsonl"
        comparison = ""
        if v2_path.is_file():
            records = (
                json.loads(line)
                for line in v2_path.read_text(encoding="utf-8").splitlines()
            )
            successful = [row for row in records if row.get("status") == "ok"]
            if successful:
                v2 = successful[-1]
                relx_loss = float(report["epochs"][-1]["train_loss"])
                v2_loss = float(v2["mean_loss"])
                same_loss = math.isclose(relx_loss, v2_loss, rel_tol=1e-5, abs_tol=1e-6)
                failures += not same_loss
                comparison = (
                    f", v2_loss={v2_loss:.6f}, relx_loss={relx_loss:.6f}, "
                    f"numeric={'match' if same_loss else 'DIFF'}"
                )
        print(
            f"{dataset}: task={'match' if task_matches else 'DIFF'}, "
            f"rows={'match' if count_matches else 'DIFF'}, "
            f"loss={'finite' if finite else 'invalid'}, "
            f"metric={report['metric']}{comparison}",
        )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
