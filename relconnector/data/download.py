"""Download RelBench data and materialize it in a replaceable SQL backend.

Run from the project root, for example:

    python -m data.download \
        --dataset rel-ratebeer \
        --task beer_ratings-total_score \
        --output data/relbench/rel-ratebeer.sqlite
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .source import BaseDatasetSource, RelBenchDatasetSource
from .writers import BaseDatabaseWriter, create_writer

DEFAULT_DATASET = "rel-ratebeer"


class RelBenchMaterializer:
    """Orchestrate a replaceable dataset source and database writer."""

    def __init__(self, source: BaseDatasetSource, writer: BaseDatabaseWriter) -> None:
        self.source = source
        self.writer = writer

    def run(
        self,
        dataset_name: str,
        task_names: Sequence[str] = (),
        *,
        all_tasks: bool = False,
        overwrite: bool = False,
    ) -> str:
        bundle = self.source.load(dataset_name, task_names, all_tasks=all_tasks)
        url = self.writer.write(bundle, overwrite=overwrite)
        if url.startswith("sqlite://"):
            from .initialize_stypes import initialize_database
            initialize_database(url[len("sqlite://"):])
        return url


def download_relbench_data(
    dataset_name: str,
    output_path: str | Path,
    *,
    task_names: Sequence[str] = (),
    all_tasks: bool = False,
    database_backend: str = "sqlite",
    insertion_chunk_size: int = 10_000,
    overwrite: bool = False,
    revision: str | None = None,
    source: BaseDatasetSource | None = None,
) -> str:
    """Download a complete RelBench database and optional task target tables."""
    writer = create_writer(
        database_backend,
        output_path,
        insertion_chunk_size=insertion_chunk_size,
    )
    materializer = RelBenchMaterializer(
        source or RelBenchDatasetSource(revision=revision), writer
    )
    return materializer.run(
        dataset_name,
        task_names,
        all_tasks=all_tasks,
        overwrite=overwrite,
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--revision",
        help="Pinned Hugging Face revision/commit for a reproducible snapshot",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task to materialize; repeat for multiple tasks",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Materialize every task exposed by the dataset",
    )
    parser.add_argument(
        "--database-backend",
        default="sqlite",
        choices=["sqlite"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "relbench" / f"{DEFAULT_DATASET}.sqlite",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")
    if args.all_tasks and args.task:
        parser.error("--all-tasks and --task cannot be used together")
    if args.cache_dir is not None:
        os.environ["HF_HOME"] = str(args.cache_dir.expanduser().resolve())

    url = download_relbench_data(
        args.dataset,
        args.output,
        task_names=args.task,
        all_tasks=args.all_tasks,
        database_backend=args.database_backend,
        insertion_chunk_size=args.chunk_size,
        overwrite=args.force,
        revision=args.revision,
    )
    print(f"Database created: {url}", flush=True)


if __name__ == "__main__":
    main()
