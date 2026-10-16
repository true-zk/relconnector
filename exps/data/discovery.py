"""Discover RelBench datasets hosted on Hugging Face."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_REPOSITORIES = (
    "stanford-star/relbench-v1",
    "stanford-star/relbench-v2-extra",
)


@dataclass(frozen=True)
class RelBenchDatasetReference:
    name: str
    repository: str

    @property
    def spec(self) -> str:
        return f"{self.repository}/{self.name}"


def discover_relbench_datasets(
    repositories: Sequence[str] = DEFAULT_REPOSITORIES,
    *,
    revision: str | None = None,
) -> list[RelBenchDatasetReference]:
    """List top-level dataset manifests from the official repositories."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Dataset discovery requires `pip install huggingface_hub`"
        ) from exc

    api = HfApi()
    discovered: dict[str, RelBenchDatasetReference] = {}
    for repository in repositories:
        files = api.list_repo_files(
            repository,
            repo_type="dataset",
            revision=revision,
        )
        for path in files:
            parts = path.split("/")
            if len(parts) != 2 or parts[1] != "manifest.yaml":
                continue
            name = parts[0]
            discovered.setdefault(
                name,
                RelBenchDatasetReference(name=name, repository=repository),
            )
    return [discovered[name] for name in sorted(discovered)]
