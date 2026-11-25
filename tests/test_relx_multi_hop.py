"""Native multi-hop sampling against pyg-lib's exhaustive temporal contract."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from relx.core import DatabaseSchema, TableRef, TableSchema
from relx.sampling import EntitySamplePlan, NativeMultiHopSampler
from relx.task import BatchKey, EntitySeedBatch


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native")
    and importlib.util.find_spec("pyarrow"),
    "native graph and pyarrow needed",
)
class MultiHopTest(unittest.TestCase):
    def setUp(self) -> None:
        import pyarrow as pa

        from relx._relx_native import GraphBuilder

        builder = GraphBuilder()
        for table, timed in (("a", True), ("b", True), ("c", False)):
            builder.register_table(table, timed)
        builder.register_relation("ab", "a", "b")
        builder.register_relation("bc", "b", "c")
        a = pa.record_batch({
            "id": [0, 1, 2], "b_id": [0, 0, 1],
            "__relx_time_us": [8, 3, 4],
        })
        b = pa.record_batch({
            "id": [0, 1], "c_id": [0, 0], "__relx_time_us": [5, 2],
        })
        c = pa.record_batch({"id": [0]})
        for table, batch in (("a", a), ("b", b), ("c", c)):
            builder.add_nodes(table, ["id"], batch)
        builder.add_edges("ab", "a", "b", ["id"], ["b_id"], a)
        builder.add_edges("bc", "b", "c", ["id"], ["c_id"], b)
        self.graph = builder.finish()

    def test_exhaustive_temporal_two_hops_and_disjoint_seeds(self) -> None:
        raw = self.graph.sample_multi_hop("c", [0, 0], [6, 10], [-1, -1], 42, 2)
        self.assertEqual(
            {name: counts.tolist() for name, counts in raw["nodes_per_hop"].items()},
            {"a": [0, 0, 5], "b": [0, 4, 0], "c": [2, 0, 0]},
        )
        self.assertEqual(
            {name: counts.tolist() for name, counts in raw["edges_per_hop"].items()},
            {"ab": [0, 5], "bc": [4, 0], "rev:ab": [0, 0], "rev:bc": [0, 4]},
        )
        np.testing.assert_array_equal(raw["node_ids"]["a"], [2, 1, 2, 1, 0])
        np.testing.assert_array_equal(raw["node_seed_index"]["a"], [0, 0, 1, 1, 1])
        self.assertTrue(
            all(
                time <= [6, 10][owner]
                for node, owner in zip(
                    raw["node_ids"]["a"], raw["node_seed_index"]["a"],
                    strict=True,
                )
                for time in [self.graph.node_times("a")[node]]
            )
        )
        schema = DatabaseSchema({
            ref: TableSchema(ref, ("id",))
            for ref in (TableRef("a"), TableRef("b"), TableRef("c"))
        })
        seeds = EntitySeedBatch(
            BatchKey(0, 0), "c", np.array([0, 0], dtype=np.int64),
            np.array([6, 10], dtype=np.int64), np.array([1, 0]), ((1,), (2,)),
        )
        plan = NativeMultiHopSampler(
            self.graph, schema, fanouts=(-1, -1), workers=2,
        ).sample(seeds)
        assert isinstance(plan, EntitySamplePlan)
        plan.subgraph.validate()
        self.assertEqual(plan.subgraph.sampled_nodes_per_hop[TableRef("a")], (0, 0, 5))

    def test_random_sampling_is_worker_count_independent(self) -> None:
        first = self.graph.sample_multi_hop("c", [0, 0], [6, 10], [1, 1], 17, 1)
        parallel = self.graph.sample_multi_hop("c", [0, 0], [6, 10], [1, 1], 17, 4)
        for section in ("node_ids", "node_seed_index", "edge_index",
                        "nodes_per_hop", "edges_per_hop"):
            for name, values in first[section].items():
                np.testing.assert_array_equal(values, parallel[section][name])

    @unittest.skipUnless(
        importlib.util.find_spec("pyg_lib") and importlib.util.find_spec("torch"),
        "pyg-lib and torch needed for parity check",
    )
    def test_exhaustive_output_matches_pyg_lib(self) -> None:
        import pyg_lib  # noqa: F401  # registers the native Torch operator
        import torch

        fanouts = [-1, -1, -1]
        raw = self.graph.sample_multi_hop("c", [0, 0], [6, 10], fanouts, 42, 2)
        names = self.graph.relation_names()
        endpoints = {name: self.graph.relation_endpoints(name) for name in names}
        relations = {name: self.graph.relation(name) for name in names}
        pyg_names = {
            name: "__".join((endpoints[name][0], name, endpoints[name][1]))
            for name in names
        }
        out = torch.ops.pyg.hetero_neighbor_sample(  # pyright: ignore[reportCallIssue]
            ["a", "b", "c"],
            [(endpoints[name][0], name, endpoints[name][1]) for name in names],
            {pyg_names[name]: torch.from_numpy(relations[name]["colptr"]) for name in names},
            {pyg_names[name]: torch.from_numpy(relations[name]["row"]) for name in names},
            {"c": torch.tensor([0, 0])},
            {pyg_names[name]: fanouts for name in names},
            None,
            {pyg_names[name]: torch.from_numpy(relations[name]["edge_time"])
             for name in names},
            {"c": torch.tensor([6, 10])},
            None, True, False, True, True, "uniform", False,
        )
        row, col, nodes, _, node_counts, edge_counts = out
        for table in ("a", "b", "c"):
            np.testing.assert_array_equal(nodes[table][:, 0], raw["node_seed_index"][table])
            np.testing.assert_array_equal(nodes[table][:, 1], raw["node_ids"][table])
            self.assertEqual(node_counts[table], raw["nodes_per_hop"][table].tolist())
        for name in names:
            pyg_name = pyg_names[name]
            np.testing.assert_array_equal(row[pyg_name], raw["edge_index"][name][0])
            np.testing.assert_array_equal(col[pyg_name], raw["edge_index"][name][1])
            self.assertEqual(edge_counts[pyg_name], raw["edges_per_hop"][name].tolist())


if __name__ == "__main__":
    unittest.main()
