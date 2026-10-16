"""Compare actual GloVe/TensorFrame inputs on one deterministic F1 SamplePlan."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from torch_frame import TensorFrame
from torch_frame.data.multi_tensor import _MultiTensor
from torch_geometric.data import HeteroData


def compare_inputs() -> list[dict[str, object]]:
    torch.set_num_threads(1)
    outputs: dict[str, tuple[HeteroData, ...]] = {}
    for label, namespace in [
        ("latest", "relconnector"),
        ("vanilla", "baseline.vanilla_baseline"),
        ("cache", "baseline.cache_baseline"),
    ]:
        package = importlib.import_module(namespace)
        model = package.OnlineRelBenchModel(
            dataset="rel-f1",
            task="driver-circuit-compete",
            config=package.OnlineTrainingConfig(device="cpu", max_batches=1),
        )
        session = model.prepare()
        components = session.components
        seed = next(iter(components.seeds.iter_epoch(0)))
        plan = components.sampler.sample(seed)
        outputs[label] = components.assembler.assemble(
            components.fetcher.fetch(plan)
        ).data
        if label == "vanilla":
            from baseline.batch_baseline.feature_store import (
                EagerFeatureStore,
                InMemoryTensorFrameFetcher,
            )

            store = EagerFeatureStore.materialize(
                components.fetcher.reader.read_relbench_database(),
                components.assembler.encoder,
            )
            outputs["batch"] = components.assembler.assemble(
                InMemoryTensorFrameFetcher(store).fetch(plan)
            ).data
    report = []
    for label, batches in outputs.items():
        checks = 0
        for left, right in zip(outputs["latest"], batches, strict=True):
            for table in left.node_types:
                assert torch.equal(left[table].n_id, right[table].n_id)
                a: TensorFrame = left[table].tf
                b: TensorFrame = right[table].tf
                assert a.col_names_dict == b.col_names_dict
                for kind, value in a.feat_dict.items():
                    other = b.feat_dict[kind]
                    _assert_feature_equal(value, other)
                    checks += 1
            for edge in left.edge_types:
                assert torch.equal(left[edge].edge_index, right[edge].edge_index)
        report.append(
            {
                "implementation": label,
                "tensor_groups": checks,
                "equal_values_and_nan_positions": True,
                "rtol": 0,
                "atol": 0,
            }
        )
    return report


def _assert_feature_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, _MultiTensor) and isinstance(right, _MultiTensor):
        _assert_feature_equal(left.offset, right.offset)
        _assert_feature_equal(left.values, right.values)
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_feature_equal(left[key], right[key])
    else:
        raise TypeError(f"Unsupported feature pair: {type(left)}, {type(right)}")


def main() -> None:
    result = compare_inputs()
    target = Path("benchmarks/correctness-real-feature-parity.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
