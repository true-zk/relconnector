"""Local task discovery and reusable JSONL continuation rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from relconnector.connector import create_reader

DEFAULT_DATABASE_DIR = Path(__file__).resolve().parents[1] / "data" / "relbench"


def discover_cases(
    database_dir: Path, *, datasets: set[str], tasks: set[str]
) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for path in sorted(database_dir.glob("*.sqlite")):
        if datasets and path.stem not in datasets:
            continue
        reader = create_reader("pandas", path)
        for task in sorted(reader.task_metadata()):
            if not tasks or task in tasks:
                cases.append((path.stem, task))
    return cases


def prepare_output(
    output: Path, *, resume: bool, overwrite: bool
) -> set[tuple[str, str]]:
    if overwrite:
        output.unlink(missing_ok=True)
        return set()
    if not output.exists():
        return set()
    if not resume:
        raise FileExistsError(f"{output} already exists; pass --resume or --overwrite")
    completed: set[tuple[str, str]] = set()
    with output.open() as source:
        for number, line in enumerate(source, 1):
            try:
                row = cast(dict[str, object], json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {output}:{number}") from exc
            if row.get("status") == "ok":
                completed.add((str(row["dataset"]), str(row["task"])))
    return completed
