"""Record current official Hub revisions and db Parquet sizes without downloading data."""

import json
from datetime import datetime, timezone
from pathlib import Path

import requests


OUT = Path(__file__).resolve().parent
records = []
for repo in ("stanford-star/relbench-v1", "stanford-star/relbench-v2-extra"):
    url = f"https://huggingface.co/api/datasets/{repo}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    revision = response.json()["sha"]
    tree_url = f"{url}/tree/{revision}?recursive=true&limit=1000"
    files = []
    next_url = tree_url
    while next_url:
        response = requests.get(next_url, timeout=60)
        response.raise_for_status()
        files.extend(item for item in response.json()
                     if item["type"] == "file" and "/db/" in item["path"]
                     and item["path"].endswith(".parquet"))
        next_url = response.links.get("next", {}).get("url")
    records.append(dict(repo=repo, revision=revision, api_url=url,
                        tree_url=tree_url, parquet_files=files))
    print(repo, revision, len(files), "Parquet files", flush=True)
(OUT / "official-hub.json").write_text(json.dumps(
    dict(checked_at=datetime.now(timezone.utc).isoformat(), repositories=records),
    indent=2) + "\n")
