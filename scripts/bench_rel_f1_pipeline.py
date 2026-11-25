"""Benchmark equal RelX sampling/feature work on the full rel-f1 graph.

The Python baseline uses the same native multi-hop sampler, but performs the
default feature transforms on Pandas rows. This isolates feature/pipeline
execution from graph construction, SQL I/O, and model computation. It is not
an end-to-end replay of relconnector_v2's SQL/TensorFrame process pipeline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import Queue
from threading import Semaphore, Thread
from typing import Any

import numpy as np
import yaml

import relx
from relx.batch import FeatureStore, NativeBatchIterator, PreparedBatch
from relx.batch.transforms import empty_block, transform_builtin
from relx.conn import connect
from relx.core import TableRef
from relx.preprocess.features import FeatureBlock
from relx.sampling import EntitySamplePlan, NativeMultiHopSampler
from relx.task import BatchKey, EntitySeedBatch, SqlEntitySeedSource

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "exps/data/relbench/rel-f1.sqlite"
TIMED = (
    "constructor_results", "constructor_standings", "qualifying", "races",
    "results", "standings",
)
STATIC = ("circuits", "constructors", "drivers")
EXCLUDED = (
    "_relconnector_columns", "_relconnector_foreign_keys",
    "_relconnector_metadata", "_relconnector_tables", "_relconnector_tasks",
    "target_driver_circuit_compete", "target_driver_position",
    "target_driver_top3", "target_qualifying_position",
    "target_results_position",
)


class FixedSeeds:
    def __init__(self, batches: list[EntitySeedBatch]) -> None:
        self.batches = batches

    def iter_epoch(self, epoch: int):
        for index, batch in enumerate(self.batches):
            yield replace(batch, key=BatchKey(epoch, index))


class PythonFeatures:
    """Same fitted states and rows, with per-batch Pandas transforms."""

    def __init__(self, prepared: relx.PreparedDataset, store: FeatureStore) -> None:
        self.prepared = prepared
        self.states = store.states
        self.frames = {}
        with connect(DATABASE, reader="connectorx") as reader:
            for name, count in prepared.graph.node_counts.items():
                ref = next(ref for ref in prepared.schema.tables if str(ref) == name)
                specs = prepared.features.for_table(ref)
                if specs:
                    self.frames[ref] = store._read_table(
                        reader, ref, specs, count, 10_000
                    )

    def fetch(self, plan: EntitySamplePlan) -> PreparedBatch:
        blocks: dict[tuple[TableRef, str], FeatureBlock] = {}
        for ref, ids in plan.subgraph.node_ids.items():
            for spec in self.prepared.features.for_table(ref):
                blocks[(ref, spec.column)] = (
                    transform_builtin(
                        spec, self.frames[ref][spec.column].iloc[ids],
                        self.states[(ref, spec.column)],
                        self.prepared.config.graph.timezone,
                    )
                    if len(ids) else empty_block(spec)
                )
        return PreparedBatch(plan.key, plan, blocks)


def make_prepared(batch_size: int):
    config = {
        "database": {"url": str(DATABASE), "reader": "connectorx"},
        "task": {
            "name": "driver-dnf", "type": "binary_classification",
            "table": "target_driver_dnf", "entity_column": "driverId",
            "label_column": "did_not_finish", "time_column": "date",
            "split": {
                "type": "column", "column": "split",
                "values": {"train": "train", "validation": "val", "test": "test"},
            },
        },
        "graph": {
            "exclude_tables": list(EXCLUDED),
            "time_columns": {name: "date" for name in TIMED},
            "static_tables": list(STATIC),
        },
        "features": {"sample_limit": 1_000},
        "training": {"batch_size": batch_size, "seed": 42},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return relx.prepare(path)


def make_inputs(prepared: relx.PreparedDataset, batch_size: int, batches: int,
                native_feature_workers: int):
    with connect(DATABASE, reader="connectorx") as reader:
        source = SqlEntitySeedSource(
            reader, prepared.task, prepared.graph, split="train",
            timezone=prepared.config.graph.timezone, batch_size=batch_size,
            shuffle_seed=42,
        )
        seeds = list(source.iter_epoch(0))[:batches]
        store = FeatureStore(
            prepared, reader, train_cutoff_us=source.max_time_us,
            native_workers=native_feature_workers,
        )
    return FixedSeeds(seeds), store


def check_equal(left: PreparedBatch, right: PreparedBatch) -> None:
    assert isinstance(left.sample, EntitySamplePlan)
    assert isinstance(right.sample, EntitySamplePlan)
    left_graph, right_graph = left.sample.subgraph, right.sample.subgraph
    for ref in left_graph.node_ids:
        np.testing.assert_array_equal(left_graph.node_ids[ref], right_graph.node_ids[ref])
    for relation in left_graph.edge_index:
        np.testing.assert_array_equal(
            left_graph.edge_index[relation], right_graph.edge_index[relation]
        )
    if left.features.keys() != right.features.keys():
        raise AssertionError("Feature columns differ")
    for key in left.features:
        for field in ("values", "missing", "offsets"):
            a, b = getattr(left.features[key], field), getattr(right.features[key], field)
            if a is None or b is None:
                if a is not None or b is not None:
                    raise AssertionError(f"{key}.{field} missingness differs")
            else:
                if np.asarray(a).dtype == object:
                    np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def run_python_serial(source: FixedSeeds, sampler: NativeMultiHopSampler,
                      features: PythonFeatures) -> int:
    rows = 0
    for seed in source.iter_epoch(0):
        plan = sampler.sample(seed)
        assert isinstance(plan, EntitySamplePlan)
        batch = features.fetch(plan)
        assert isinstance(batch.sample, EntitySamplePlan)
        rows += len(batch.sample.target)
    return rows


def run_python_threads(source: FixedSeeds, sampler: NativeMultiHopSampler,
                       features: PythonFeatures | FeatureStore, workers: int,
                       max_inflight: int) -> int:
    queue: Queue[Any] = Queue()
    slots = Semaphore(max_inflight)
    end = object()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def produce() -> None:
            try:
                for seed in source.iter_epoch(0):
                    slots.acquire()
                    plan = sampler.sample(seed)
                    assert isinstance(plan, EntitySamplePlan)
                    queue.put(pool.submit(features.fetch, plan))
            except Exception as exc:  # noqa: BLE001 - forward benchmark errors
                queue.put(exc)
            finally:
                queue.put(end)

        producer = Thread(target=produce)
        producer.start()
        rows = 0
        try:
            while (item := queue.get()) is not end:
                if isinstance(item, Exception):
                    raise item
                batch = item.result()
                slots.release()
                assert isinstance(batch.sample, EntitySamplePlan)
                rows += len(batch.sample.target)
        finally:
            producer.join()
    return rows


def run_cpp(source: FixedSeeds, sampler: NativeMultiHopSampler,
            store: FeatureStore, workers: int, max_inflight: int) -> int:
    pipeline = NativeBatchIterator(
        source, sampler, store, feature_workers=workers,
        max_inflight=max_inflight,
    )
    rows = 0
    try:
        pipeline.start_epoch(0)
        while (batch := pipeline.next_batch()) is not None:
            assert isinstance(batch.sample, EntitySamplePlan)
            rows += len(batch.sample.target)
    finally:
        pipeline.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fanouts", default="16,8")
    parser.add_argument("--sampling-workers", type=int, default=2)
    parser.add_argument("--feature-workers", type=int, default=2)
    parser.add_argument("--feature-native-workers", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fanouts = tuple(int(value) for value in args.fanouts.split(","))
    if min(args.batch_size, args.batches, args.repeats, args.sampling_workers,
           args.feature_workers, args.feature_native_workers,
           args.max_inflight) < 1 or not fanouts or min(fanouts) < -1:
        parser.error("batch, repeat, worker, and inflight counts must be positive")
    started = time.perf_counter()
    prepared = make_prepared(args.batch_size)
    graph_s = time.perf_counter() - started
    started = time.perf_counter()
    source, store = make_inputs(
        prepared, args.batch_size, args.batches, args.feature_native_workers
    )
    python_features = PythonFeatures(prepared, store)
    features_s = time.perf_counter() - started
    sampler = NativeMultiHopSampler(
        prepared.graph, prepared.schema, fanouts=fanouts,
        workers=args.sampling_workers, random_seed=42,
    )
    first = source.batches[0]
    sample = sampler.sample(first)
    assert isinstance(sample, EntitySamplePlan)
    check_equal(python_features.fetch(sample), store.fetch(sample))
    pipeline = NativeBatchIterator(source, sampler, store,
                                   feature_workers=args.feature_workers,
                                   max_inflight=args.max_inflight)
    try:
        pipeline.start_epoch(0)
        native_first = pipeline.next_batch()
        assert native_first is not None
        check_equal(python_features.fetch(sample), native_first)
    finally:
        pipeline.close()

    methods = {
        "python_serial": lambda: run_python_serial(source, sampler, python_features),
        "python_threads_transform": lambda: run_python_threads(
            source, sampler, python_features, args.feature_workers,
            args.max_inflight,
        ),
        "python_threads_native": lambda: run_python_threads(
            source, sampler, store, args.feature_workers, args.max_inflight,
        ),
        "cpp_pipeline": lambda: run_cpp(
            source, sampler, store, args.feature_workers, args.max_inflight
        ),
    }
    expected = sum(len(batch.node_ids) for batch in source.batches)
    for run in methods.values():
        if run() != expected:
            raise AssertionError("Batch row count differs")
    samples: dict[str, list[float]] = {name: [] for name in methods}
    for repeat in range(args.repeats):
        names = list(methods)
        if repeat % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            if methods[name]() != expected:
                raise AssertionError(f"{name} yielded an unexpected row count")
            samples[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    result = {
        "database": str(DATABASE), "dataset": "rel-f1",
        "task": "driver-dnf", "graph_tables": prepared.report.table_count,
        "relations": prepared.report.relation_count,
        "features": prepared.report.feature_count,
        "batches": len(source.batches), "examples": expected,
        "settings": {"batch_size": args.batch_size, "fanouts": fanouts,
                     "sampling_workers": args.sampling_workers,
                     "feature_workers": args.feature_workers,
                     "feature_native_workers": args.feature_native_workers,
                     "max_inflight": args.max_inflight, "repeats": args.repeats},
        "prepare_s": {"graph_and_plan": graph_s, "feature_snapshots": features_s},
        "median_s": medians,
        "speedup_vs_python_serial": medians["python_serial"] / medians["cpp_pipeline"],
        "speedup_vs_python_threads_transform": (
            medians["python_threads_transform"] / medians["cpp_pipeline"]
        ),
        "speedup_vs_python_threads_native": (
            medians["python_threads_native"] / medians["cpp_pipeline"]
        ),
        "samples_s": samples,
    }
    print(json.dumps(result, indent=2, default=str))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
