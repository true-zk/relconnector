"""Small, validated stype artifacts; no dependency on any training version."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import cast

ARTIFACT_VERSION = 1
STYPE_SEED = 42


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def database_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dependencies() -> dict[str, str]:
    return {
        name: version(name) for name in ("relbench", "pytorch-frame", "pandas", "numpy")
    }


def artifact_path(database: str | Path) -> Path:
    return Path(database).with_suffix(".stypes.json")


def load_proposal(
    database: str | Path, cutoff: object, *, required: bool = True
) -> dict[str, dict[str, str]] | None:
    path = Path(database)
    source = artifact_path(path)
    if not source.exists():
        if not required:
            return None
        raise FileNotFoundError(
            f"Missing stype artifact: {source}; run "
            f"python -m data.initialize_stypes --database {path}"
        )
    payload = json.loads(source.read_text())
    fingerprint = payload.pop("fingerprint", None)
    if digest(payload) != fingerprint:
        raise ValueError(f"Corrupt stype artifact: {source}")
    # cutoff is the input DB view, not the dataset's raw export boundary.
    expected_cutoff = None if cutoff is None else str(cutoff)
    if (
        payload.get("version") != ARTIFACT_VERSION
        or payload.get("database") != database_identity(path)
        or payload.get("dependencies") != dependencies()
        or payload.get("cutoff") != expected_cutoff
    ):
        raise ValueError(
            f"Stale stype artifact: {source}; regenerate with data.initialize_stypes"
        )
    return cast(dict[str, dict[str, str]], payload["tables"])
