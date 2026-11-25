"""Create RelX smoke configs for the ten frozen v2 SQLite tasks.

This benchmark helper reads the experimental SQLite catalogs. RelX itself
continues to use only YAML and database schema introspection.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

CASES = (
    ("rel-amazon", "item-churn"),
    ("rel-arxiv", "paper-citation"),
    ("rel-avito", "searchstream-click"),
    ("rel-event", "user-attendance"),
    ("rel-f1", "driver-circuit-compete"),
    ("rel-hm", "user-churn"),
    ("rel-ratebeer", "beer_ratings-total_score"),
    ("rel-salt", "item-incoterms"),
    ("rel-stack", "user-badge"),
    ("rel-trial", "eligibilities-adult"),
)


def make_config(
    database: Path, task_name: str, report_dir: Path, glove_export_dir: Path
) -> dict:
    """Translate v2 task/scope metadata into a standalone RelX YAML value."""
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as sql:
        sql.row_factory = sqlite3.Row
        task = sql.execute(
            "SELECT * FROM _relconnector_tasks WHERE task_name = ?", (task_name,)
        ).fetchone()
        if task is None:
            raise ValueError(f"{database.name}: missing task {task_name}")
        tables = [dict(row) for row in sql.execute(
            "SELECT * FROM _relconnector_tables ORDER BY table_name"
        )]
        table_by_name = {row["table_name"]: row for row in tables}
        catalog_tables = [row[0] for row in sql.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE '_relconnector_%'"
        )]
        task_foreign_keys = [dict(row) for row in sql.execute(
            f'PRAGMA foreign_key_list("{task["table_name"]}")'
        )]

    extra = json.loads(task["extra"])
    entity = table_by_name[task["entity_table"]]
    config_task = {
        "name": task_name,
        "type": task["task_type"],
        "table": task["table_name"],
        "entity_column": task["entity_column"],
        "entity_table": task["entity_table"],
        "label_column": task["target_column"],
        "split": {
            "type": "column", "column": "split",
            "values": {"train": "train", "validation": "val", "test": "test"},
        },
    }
    if task["time_column"]:
        config_task["time_column"] = task["time_column"]
    destination_name = extra.get("dst_entity_table")
    if destination_name:
        destination = table_by_name[destination_name]
        config_task["destination"] = {
            "table": destination_name, "key": destination["primary_key"]
        }
    if not entity["primary_key"]:
        raise ValueError(f"{database.name}: missing entity identity key")
    schema = {}
    if not any(
        key["from"] == task["entity_column"]
        and key["table"] == task["entity_table"]
        for key in task_foreign_keys
    ):
        schema["foreign_keys"] = [{
            "source_table": task["table_name"],
            "source_columns": [task["entity_column"]],
            "target_table": task["entity_table"],
            "target_columns": [entity["primary_key"]],
        }]

    data_tables = [row for row in tables if row["table_kind"] == "data"]
    hidden = [list(pair) for pair in extra.get("hidden_columns", [])]
    if extra.get("kind") == "autocomplete":
        hidden.append([task["entity_table"], task["target_column"]])
    return {
        "database": {"url": str(database.resolve()), "reader": "connectorx"},
        "task": config_task,
        "schema": schema,
        "graph": {
            "exclude_tables": sorted(set(catalog_tables) | {
                row["table_name"] for row in tables if row["table_kind"] != "data"
            }),
            "time_columns": {
                row["table_name"]: row["time_column"]
                for row in data_tables if row["time_column"]
            },
            "static_tables": [
                row["table_name"] for row in data_tables if not row["time_column"]
            ],
            "time_boundary": "at_or_before",
        },
        "hidden_columns": sorted({tuple(pair) for pair in hidden}),
        "features": {"sample_limit": 1_000},
        "model": {
            "name": "rdl",
            "params": {
                "channels": 16, "num_neighbors": [4, 4],
                "text_mode": "glove",
                "glove_export_dir": str(glove_export_dir.resolve()),
            },
        },
        "training": {
            "batch_size": 8, "epochs": 1, "max_batches": 1,
            "seed": 42, "feature_workers": 2,
            "feature_native_workers": 1, "prefetch_batches": 2,
        },
        "report": {"directory": str(report_dir.resolve())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-dir", type=Path, default=Path("exps/data/relbench")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/relx_ten_smoke")
    )
    parser.add_argument(
        "--glove-export-dir", type=Path, default=Path("build/glove_parity")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, task in CASES:
        config = make_config(
            args.database_dir / f"{dataset}.sqlite", task,
            args.output_dir / dataset, args.glove_export_dir,
        )
        path = args.output_dir / f"{dataset}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
