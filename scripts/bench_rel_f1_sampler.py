"""Compare raw RelX C++ and pyg-lib multi-hop calls on the full rel-f1 graph.

Both receive the same CSC and seed batches, use two hops, inclusive edge-time
filtering, uniform sampling without replacement, and return native outputs.
Their RNGs select different neighbors, so exact sampled node sets may differ.
Feature processing, graph preparation, SQL reads, and model work are excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pyg_lib  # noqa: F401  # registers torch.ops.pyg
import torch
from bench_rel_f1_pipeline import make_inputs, make_prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fanouts", default="16,8")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fanouts = [int(value) for value in args.fanouts.split(",")]
    if min(args.batch_size, args.batches, args.repeats, args.workers) < 1:
        parser.error("batch, repeat, and worker counts must be positive")
    if not fanouts or min(fanouts) < -1:
        parser.error("fanouts must be nonempty and >= -1")

    prepared = make_prepared(args.batch_size)
    source, _ = make_inputs(prepared, args.batch_size, args.batches, 1)
    graph = prepared.graph
    names = graph.relation_names()
    endpoints = {name: graph.relation_endpoints(name) for name in names}
    relations = {name: graph.relation(name) for name in names}
    pyg_names = {
        name: "__".join((endpoints[name][0], name, endpoints[name][1]))
        for name in names
    }
    pyg_prefix = (
        list(graph.node_counts),
        [(endpoints[name][0], name, endpoints[name][1]) for name in names],
        {pyg_names[name]: torch.from_numpy(relations[name]["colptr"])
         for name in names},
        {pyg_names[name]: torch.from_numpy(relations[name]["row"])
         for name in names},
    )
    pyg_fanouts = {pyg_names[name]: fanouts for name in names}
    pyg_edge_time = {
        pyg_names[name]: torch.from_numpy(relations[name]["edge_time"])
        for name in names
    }
    pyg_cases = [
        (
            *pyg_prefix,
            {"drivers": torch.from_numpy(batch.node_ids)},
            pyg_fanouts, None, pyg_edge_time,
            {"drivers": torch.from_numpy(batch.seed_time_us)},
            None, True, False, True, True, "uniform", False,
        )
        for batch in source.batches
    ]
    cpp_cases = [
        (
            "drivers", batch.node_ids.tolist(), batch.seed_time_us.tolist(),
            fanouts, 42 + index, args.workers,
        )
        for index, batch in enumerate(source.batches)
        if batch.seed_time_us is not None
    ]
    if len(cpp_cases) != len(source.batches):
        raise ValueError("The rel-f1 driver task needs time-stamped seeds")
    torch.set_num_threads(args.workers)
    first_pyg = torch.ops.pyg.hetero_neighbor_sample(  # pyright: ignore[reportCallIssue]
        *pyg_cases[0]
    )
    first_cpp = graph.sample_multi_hop(*cpp_cases[0])
    pyg_edges = sum(sum(counts) for counts in first_pyg[5].values())
    cpp_edges = sum(edge.shape[1] for edge in first_cpp["edge_index"].values())
    if pyg_edges != cpp_edges:
        raise AssertionError("pyg-lib and C++ sampled different edge counts")

    runs: dict[str, list[float]] = {"cpp_raw": [], "pyg_lib_raw": []}
    for repeat in range(args.repeats):
        names_to_run = (
            ("cpp_raw", "pyg_lib_raw")
            if repeat % 2 == 0 else ("pyg_lib_raw", "cpp_raw")
        )
        for name in names_to_run:
            start = time.perf_counter()
            if name == "cpp_raw":
                for call in cpp_cases:
                    graph.sample_multi_hop(*call)
            else:
                for call in pyg_cases:
                    torch.ops.pyg.hetero_neighbor_sample(  # pyright: ignore[reportCallIssue]
                        *call
                    )
            runs[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in runs.items()}
    result = {
        "dataset": "rel-f1", "task": "driver-dnf",
        "tables": prepared.report.table_count,
        "relations": prepared.report.relation_count,
        "batches": len(source.batches),
        "settings": {"batch_size": args.batch_size, "fanouts": fanouts,
                     "workers": args.workers, "repeats": args.repeats},
        "first_batch_edges": pyg_edges,
        "first_batch_nodes": {
            "cpp": sum(len(ids) for ids in first_cpp["node_ids"].values()),
            "pyg_lib": sum(len(ids) for ids in first_pyg[2].values()),
        },
        "median_s": medians,
        "cpp_over_pyg_lib_time": medians["cpp_raw"] / medians["pyg_lib_raw"],
        "samples_s": runs,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
