"""Compare the old Python batch assembly with native one-hop assembly.

Each graph uses up to ``--source-rows`` real rows from one FK relation in a
local RelBench SQLite database. Graph construction and SQL reads are excluded
from timed sampling. Both paths use the same C++ neighbor-selection routine;
only batch assembly differs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from relx._relx_native import GraphBuilder, GraphIndex

RELATIONS = {
    "rel-amazon": ("review", "product", "review_time"),
    "rel-arxiv": ("citations", "papers", "Submission_Date"),
    "rel-avito": ("VisitStream", "AdsInfo", "ViewDate"),
    "rel-event": ("event_interest", "events", "timestamp"),
    "rel-f1": ("results", "drivers", "date"),
    "rel-hm": ("transactions", "article", "t_dat"),
    "rel-ratebeer": ("beer_ratings", "users", "created_at"),
    "rel-salt": ("salesdocumentitem", "salesdocument", "CREATIONTIMESTAMP"),
    "rel-stack": ("votes", "posts", "CreationDate"),
    "rel-trial": ("conditions_studies", "studies", "date"),
}


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_graph(
    path: Path, source: str, target: str, time_column: str, source_rows: int
) -> tuple[GraphIndex, np.ndarray, int, str, int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        info = list(connection.execute(f"PRAGMA table_info({_quote(source)})"))
        primary = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
        if len(primary) != 1:
            raise ValueError(f"{path.stem}/{source} needs one primary key")
        fk = next(
            row for row in connection.execute(f"PRAGMA foreign_key_list({_quote(source)})")
            if row[2] == target and row[4] is not None
        )
        source_key, foreign_key, target_key = primary[0], fk[3], fk[4]
        query = (
            f"SELECT {_quote(source_key)}, {_quote(foreign_key)}, "
            f"{_quote(time_column)} FROM {_quote(source)} "
            f"WHERE {_quote(foreign_key)} IS NOT NULL "
            f"AND {_quote(time_column)} IS NOT NULL LIMIT ?"
        )
        rows = connection.execute(query, (source_rows,)).fetchall()

    if not rows:
        raise ValueError(f"{path.stem}/{source} has no usable rows")
    times = pd.to_datetime(
        [row[2] for row in rows], utc=True, errors="coerce", format="mixed"
    )
    valid = ~times.isna()
    selected = [row for row, keep in zip(rows, valid, strict=True) if keep]
    if not selected:
        raise ValueError(f"{path.stem}/{source} has no parsable timestamps")
    time_us = (pd.DatetimeIndex(times[valid]).asi8 // 1_000).astype(np.int64)
    source_ids = [row[0] for row in selected]
    foreign_ids = [row[1] for row in selected]
    target_ids = list(dict.fromkeys(foreign_ids))
    source_batch = pa.record_batch(
        {
            source_key: source_ids,
            foreign_key: foreign_ids,
            "__relx_time_us": time_us,
        }
    )
    target_batch = pa.record_batch({target_key: target_ids})
    builder = GraphBuilder(inclusive=True)
    builder.register_table(source, timed=True)
    builder.register_table(target, timed=False)
    builder.register_relation("fk", source, target)
    builder.add_nodes(source, [source_key], source_batch)
    builder.add_nodes(target, [target_key], target_batch)
    builder.add_edges("fk", source, target, [source_key], [foreign_key], source_batch)
    graph = builder.finish()
    counts = Counter(foreign_ids)
    key_to_id = {key: index for index, key in enumerate(target_ids)}
    ranked_ids = np.asarray(
        [key_to_id[key] for key, _ in counts.most_common(256)], dtype=np.int64
    )
    cutoff_us = int(np.quantile(time_us, 0.75))
    return graph, ranked_ids, cutoff_us, source, len(selected)


def _old_python_sample(
    graph: GraphIndex, table: str, relations: tuple[tuple[str, str], ...],
    seed_ids: list[int], cutoffs_us: list[int], fanout: int,
    random_seed: int, workers: int,
) -> dict[str, dict[str, np.ndarray]]:
    """The per-edge Python assembly used by NativeOneHopSampler previously."""
    node_ids: dict[str, list[int]] = {table: list(seed_ids)}
    owners: dict[str, list[int]] = {table: list(range(len(seed_ids)))}
    edges: dict[str, np.ndarray] = {}
    for name, source in relations:
        sampled = graph.sample_neighbors(
            name, seed_ids, cutoffs_us, fanout, random_seed, workers
        )
        source_ids = node_ids.setdefault(source, [])
        source_owners = owners.setdefault(source, [])
        edge_source: list[int] = []
        edge_target: list[int] = []
        for seed_index, neighbors in enumerate(sampled):
            for neighbor in neighbors:
                edge_source.append(len(source_ids))
                edge_target.append(seed_index)
                source_ids.append(neighbor)
                source_owners.append(seed_index)
        edges[name] = np.asarray([edge_source, edge_target], dtype=np.int64)
    return {
        "node_ids": {name: np.asarray(ids, dtype=np.int64) for name, ids in node_ids.items()},
        "node_seed_index": {
            name: np.asarray(ids, dtype=np.int64) for name, ids in owners.items()
        },
        "edge_index": edges,
    }


def _check_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    for section in ("node_ids", "node_seed_index", "edge_index"):
        if set(left[section]) != set(right[section]):
            raise AssertionError(f"{section} keys differ")
        for name in left[section]:
            np.testing.assert_array_equal(left[section][name], right[section][name])


def _measure(
    graph: GraphIndex, source: str, table: str, ranked_ids: np.ndarray,
    cutoff_us: int, *, batch_size: int, batches: int, repeats: int,
    fanout: int, workers: int, compare_pyg: bool,
) -> dict[str, float | int]:
    rng = np.random.default_rng(31)
    cases = [
        (
            rng.choice(ranked_ids, size=batch_size, replace=True).tolist(),
            [cutoff_us] * batch_size,
            index + 100,
        )
        for index in range(batches)
    ]
    relations = tuple(
        (name, source)
        for name in graph.relation_names()
        for source, target in (graph.relation_endpoints(name),)
        if target == table
    )
    for ids, cutoffs, seed in cases[:1]:
        _check_equal(
            _old_python_sample(
                graph, table, relations, ids, cutoffs, fanout, seed, workers
            ),
            graph.sample_one_hop(table, ids, cutoffs, fanout, seed, workers),
        )
    calls = {
        "python": lambda ids, cutoffs, seed: _old_python_sample(
            graph, table, relations, ids, cutoffs, fanout, seed, workers
        ),
        "cpp": lambda ids, cutoffs, seed: graph.sample_one_hop(
            table, ids, cutoffs, fanout, seed, workers
        ),
    }
    if compare_pyg:
        import pyg_lib  # noqa: F401  # registers the native Torch operator
        import torch

        torch.set_num_threads(workers)
        relation = graph.relation("fk")
        edge_name = f"{source}__fk__{table}"
        rowptr = {edge_name: torch.from_numpy(relation["colptr"])}
        col = {edge_name: torch.from_numpy(relation["row"])}
        edge_time = {edge_name: torch.from_numpy(relation["edge_time"])}
        pyg_cases = [
            (
                [source, table], [(source, "fk", table)], rowptr, col,
                {table: torch.tensor(ids, dtype=torch.int64)},
                {edge_name: [fanout]}, None, edge_time,
                {table: torch.tensor(cutoffs, dtype=torch.int64)}, None,
                True, False, True, True, "uniform", False,
            )
            for ids, cutoffs, _ in cases
        ]
        sample = torch.ops.pyg.hetero_neighbor_sample(  # pyright: ignore[reportCallIssue]
            *pyg_cases[0]
        )
        if sample[5][edge_name][0] != sum(
            values.shape[1]
            for values in calls["cpp"](*cases[0])["edge_index"].values()
        ):
            raise AssertionError("pyg-lib and RelX sampled different edge counts")
    methods = (*calls, *(("pyg_lib",) if compare_pyg else ()))
    measurements: dict[str, list[float]] = {name: [] for name in methods}
    for repeat in range(repeats):
        names = methods if repeat % 2 == 0 else tuple(reversed(methods))
        for name in names:
            start = time.perf_counter_ns()
            if name == "pyg_lib":
                for args in pyg_cases:
                    torch.ops.pyg.hetero_neighbor_sample(  # pyright: ignore[reportCallIssue]
                        *args
                    )
            else:
                for ids, cutoffs, seed in cases:
                    calls[name](ids, cutoffs, seed)
            measurements[name].append((time.perf_counter_ns() - start) / batches / 1e6)
    python_ms = statistics.median(measurements["python"])
    cpp_ms = statistics.median(measurements["cpp"])
    sampled = calls["cpp"](*cases[0])
    result = {
        "python_ms_per_batch": round(python_ms, 3),
        "cpp_ms_per_batch": round(cpp_ms, 3),
        "speedup": round(python_ms / cpp_ms, 2),
        "sampled_edges_first_batch": sum(
            values.shape[1] for values in sampled["edge_index"].values()
        ),
    }
    if compare_pyg:
        result["pyg_lib_ms_per_batch"] = round(
            statistics.median(measurements["pyg_lib"]), 3
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("exps/data/relbench"))
    parser.add_argument("--source-rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--fanout", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compare-pyg", action="store_true")
    parser.add_argument("--dataset", choices=tuple(RELATIONS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.source_rows, args.batch_size, args.batches, args.repeats, args.workers) < 1:
        parser.error("row, batch, repeat, and worker counts must be positive")
    if args.fanout < -1:
        parser.error("fanout must be >= -1")
    results = []
    for name in ((args.dataset,) if args.dataset else RELATIONS):
        source, target, time_column = RELATIONS[name]
        graph, ids, cutoff_us, _, rows = _read_graph(
            args.data_dir / f"{name}.sqlite", source, target, time_column,
            args.source_rows,
        )
        result = {
            "dataset": name,
            "relation": f"{source}->{target}",
            "source_rows": rows,
            "target_nodes": graph.node_counts[target],
            "ranked_seed_nodes": len(ids),
            **_measure(
                graph, source, target, ids, cutoff_us,
                batch_size=args.batch_size, batches=args.batches,
                repeats=args.repeats, fanout=args.fanout, workers=args.workers,
                compare_pyg=args.compare_pyg,
            ),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"settings": vars(args), "results": results}, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
