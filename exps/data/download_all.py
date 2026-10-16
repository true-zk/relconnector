"""Download every official RelBench dataset into separate SQLite files."""

from __future__ import annotations

import gc
import os
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path

from .discovery import DEFAULT_REPOSITORIES, discover_relbench_datasets
from .download import RelBenchMaterializer
from .source import RelBenchDatasetSource
from .writers import SQLiteDatabaseWriter

DEFAULT_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "relbench"


def configure_network(proxy: str | None) -> None:
    """Configure proxy variables and accept the common ``HF_TOEKN`` typo."""
    if proxy:
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.setdefault(name, proxy)

    no_proxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
    entries = [entry for entry in no_proxy.split(",") if entry]
    if ".byted.org" not in entries:
        entries.append(".byted.org")
    value = ",".join(entries)
    os.environ["no_proxy"] = value
    os.environ["NO_PROXY"] = value

    if "HF_TOKEN" not in os.environ and "HF_TOEKN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HF_TOEKN"]


def download_all(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    repositories: tuple[str, ...] = DEFAULT_REPOSITORIES,
    dataset_names: tuple[str, ...] = (),
    revision: str | None = None,
    include_tasks: bool = True,
    overwrite: bool = False,
    insertion_chunk_size: int = 10_000,
    fail_fast: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Discover and materialize datasets, returning successes and failures."""
    output_directory = Path(output_directory).expanduser().resolve()
    references = discover_relbench_datasets(repositories, revision=revision)
    if dataset_names:
        requested = set(dataset_names)
        references = [
            reference for reference in references if reference.name in requested
        ]
        missing = requested.difference(reference.name for reference in references)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Unknown RelBench dataset(s): {names}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        references = references[:limit]

    print(f"Discovered {len(references)} RelBench dataset(s)", flush=True)
    if dry_run:
        for reference in references:
            print(f"{reference.name}: {reference.spec}", flush=True)
        return [], {}

    output_directory.mkdir(parents=True, exist_ok=True)
    source = RelBenchDatasetSource(revision=revision)
    succeeded: list[str] = []
    failed: dict[str, str] = {}

    for index, reference in enumerate(references, start=1):
        output_path = output_directory / f"{reference.name}.sqlite"
        if output_path.exists() and not overwrite:
            print(
                f"[{index}/{len(references)}] skip {reference.name}: "
                f"{output_path} exists",
                flush=True,
            )
            succeeded.append(reference.name)
            continue

        print(
            f"[{index}/{len(references)}] download {reference.spec}",
            flush=True,
        )
        try:
            writer = SQLiteDatabaseWriter(
                output_path,
                insertion_chunk_size=insertion_chunk_size,
            )
            RelBenchMaterializer(source, writer).run(
                reference.spec,
                all_tasks=include_tasks,
                overwrite=overwrite,
            )
            succeeded.append(reference.name)
            print(f"[{index}/{len(references)}] wrote {output_path}", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failed[reference.name] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{index}/{len(references)}] failed {reference.name}: "
                f"{failed[reference.name]}",
                file=sys.stderr,
                flush=True,
            )
            if fail_fast:
                raise
            traceback.print_exc()
        finally:
            gc.collect()

    return succeeded, failed


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--repository",
        action="append",
        dest="repositories",
        help="Hugging Face dataset repository; repeat to add repositories",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Only download this dataset; repeat to select multiple",
    )
    parser.add_argument("--revision")
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--without-tasks", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")

    configure_network(args.proxy)
    repositories = tuple(args.repositories or DEFAULT_REPOSITORIES)
    succeeded, failed = download_all(
        args.output_dir,
        repositories=repositories,
        dataset_names=tuple(args.dataset),
        revision=args.revision,
        include_tasks=not args.without_tasks,
        overwrite=args.force,
        insertion_chunk_size=args.chunk_size,
        fail_fast=args.fail_fast,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    print(
        f"Finished: {len(succeeded)} succeeded, {len(failed)} failed",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
