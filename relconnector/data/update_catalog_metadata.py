"""Refresh task catalog metadata without rewriting materialized data tables."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml
from huggingface_hub.constants import HF_HUB_CACHE

from relbench_compat.tasks import hidden_columns


def find_task_manifest(dataset: str, task: str) -> Path:
    pattern = (
        "datasets--stanford-star--relbench-*/snapshots/"
        f"*/{dataset}/tasks/{task}/manifest.yaml"
    )
    matches = sorted(Path(HF_HUB_CACHE).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No cached task manifest for {dataset}/{task}; rerun data.download_all"
        )
    return matches[-1]


def manifest_extra(manifest: dict[str, object]) -> dict[str, object]:
    extra = {
        "kind": manifest.get("kind"),
        "dst_entity_table": manifest.get("dst_entity_table"),
        "dst_entity_column": manifest.get("dst_entity_col"),
        "hidden_columns": manifest.get("remove_columns", []),
        "timedelta": manifest.get("timedelta", ""),
        "num_eval_timestamps": manifest.get("num_eval_timestamps"),
        "eval_k": manifest.get("eval_k"),
    }

    extra["hidden_columns"] = [
        list(pair)
        for pair in hidden_columns(
            extra,
            name=str(manifest.get("name", "")),
            entity_table=str(manifest.get("entity_table", "")),
            target_column=str(manifest.get("target_col", "")),
        )
    ]
    return extra


def update_database(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT task_name, extra FROM _relconnector_tasks"
        ).fetchall()
        dataset = connection.execute(
            "SELECT value FROM _relconnector_metadata WHERE key = 'dataset_name'"
        ).fetchone()[0]
        for task, encoded_extra in rows:
            manifest = yaml.safe_load(
                find_task_manifest(str(dataset), str(task)).read_text()
            )
            extra = json.loads(encoded_extra)
            extra.update(manifest_extra(manifest))
            connection.execute(
                "UPDATE _relconnector_tasks SET extra = ? WHERE task_name = ?",
                (json.dumps(extra, ensure_ascii=False), task),
            )
        connection.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "relbench",
    )
    args = parser.parse_args()
    for path in sorted(args.database_dir.glob("*.sqlite")):
        count = update_database(path)
        print(f"{path.name}: updated {count} task manifests")


if __name__ == "__main__":
    main()
