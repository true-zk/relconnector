"""Correctness boundaries shared by all baseline revisions."""

from __future__ import annotations

import importlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from relbench.base import Database, Table
from relbench.modeling.utils import get_stype_proposal

from data.decoding import decode_frame
from data.encoding import infer_columns
from data.initialize_stypes import initialize_database
from data.models import DatasetBundle, MaterializedTable, TableSchema
from data.update_catalog_metadata import manifest_extra
from data.writers import SQLiteDatabaseWriter
from relbench_compat.proposal import load_proposal
from relbench_compat.tasks import hidden_columns

VERSIONS = ("relconnector", "baseline.vanilla_baseline", "baseline.cache_baseline")


class CorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture.sqlite"
        # First 1000 rows suggest categorical; random official sample sees rare values.
        self.frame = pd.DataFrame(
            {
                "id": np.arange(2200),
                "url": ["https://same.example"] * 1000
                + [f"https://x{i}.example" for i in range(1200)],
                "text": [f"row {i} text" for i in range(2200)],
                "label": np.arange(2200) % 2,
            }
        )
        schema = TableSchema("entities", "id", columns=infer_columns(self.frame))
        SQLiteDatabaseWriter(self.path).write(
            DatasetBundle(
                "fixture",
                {
                    "entities": MaterializedTable(self.frame, schema),
                },
            )
        )

    def test_official_proposal_and_all_versions_match(self):
        state = np.random.get_state()
        try:
            np.random.seed(42)
            expected = get_stype_proposal(
                Database({"entities": Table(self.frame, {}, pkey_col="id")})
            )
        finally:
            np.random.set_state(state)
        initialize_database(self.path)
        actual = load_proposal(self.path, None)
        self.assertEqual(
            actual, {t: {c: k.value for c, k in v.items()} for t, v in expected.items()}
        )
        fingerprints = []
        for version in VERSIONS:
            reader = importlib.import_module(
                version + ".connector"
            ).PandasDatabaseReader(self.path)
            builder = importlib.import_module(
                version + ".features"
            ).SqlTensorFrameSchemaBuilder
            schema = builder(
                reader, hidden_columns=[("entities", "label")], require_stypes=True
            ).build()
            self.assertNotIn("label", schema.tables["entities"].col_to_stype)
            self.assertEqual(
                schema.tables["entities"].col_to_stype["url"],
                expected["entities"]["url"],
            )
            fingerprints.append(schema.fingerprint)
        self.assertEqual(len(set(fingerprints)), 1)

    def test_missing_corrupt_and_stale_proposal_fail(self):
        with self.assertRaises(FileNotFoundError):
            load_proposal(self.path, None)
        target = initialize_database(self.path)
        payload = json.loads(target.read_text())
        payload["tables"]["entities"]["url"] = "categorical"
        target.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            load_proposal(self.path, None)
        initialize_database(self.path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE entities SET text='changed' WHERE id=1")
        with self.assertRaisesRegex(ValueError, "Stale"):
            load_proposal(self.path, None)

    def test_decoder_restores_object_dtype(self):
        raw = pd.DataFrame({"text": pd.Series(["x", None], dtype="string")})
        columns = infer_columns(
            pd.DataFrame({"text": pd.Series(["x", None], dtype=object)})
        )
        self.assertEqual(decode_frame(raw, columns)["text"].dtype, object)

    def test_official_cutoff_and_synthetic_key_do_not_enter_features(self):
        frame = pd.DataFrame(
            {
                "at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"]),
                "label": [0, 1, 0],
            }
        )
        schema = TableSchema(
            "entities", None, time_column="at", columns=infer_columns(frame)
        )
        SQLiteDatabaseWriter(self.path).write(
            DatasetBundle(
                "fixture",
                {"entities": MaterializedTable(frame, schema)},
                metadata={"test_timestamp": "2024-01-02"},
            ),
            overwrite=True,
        )
        target = initialize_database(self.path)
        payload = json.loads(target.read_text())
        self.assertEqual(set(payload["sample_ids"]["entities"]), {0, 1})
        self.assertNotIn("__relconnector_node_id__", payload["tables"]["entities"])
        self.assertIsNotNone(load_proposal(self.path, pd.Timestamp("2024-01-02")))
        with self.assertRaisesRegex(ValueError, "Stale"):
            load_proposal(self.path, None)

    def test_autocomplete_and_external_label_hidden_forecast_retained(self):
        manifest = {
            "name": "predict",
            "kind": "autocomplete",
            "entity_table": "entities",
            "target_col": "label",
            "remove_columns": [["entities", "proxy"]],
        }
        extra = manifest_extra(manifest)
        self.assertEqual(
            extra["hidden_columns"], [["entities", "proxy"], ["entities", "label"]]
        )
        for version in VERSIONS:
            metadata = importlib.import_module(
                version + ".connector.catalog"
            ).TaskMetadata(
                name="predict",
                table_name="targets",
                task_type="binary_classification",
                entity_table="entities",
                entity_column="id",
                target_column="label",
                extra={
                    "kind": "autocomplete",
                    "hidden_columns": [["entities", "proxy"]],
                },
            )
            spec = importlib.import_module(version + ".task").TaskSpec.from_metadata(
                metadata
            )
            self.assertEqual(
                set(spec.hidden_columns), {("entities", "proxy"), ("entities", "label")}
            )
        from baseline.batch_baseline.dataset import _build_task

        task = _build_task(metadata, {"train": pd.DataFrame()})
        self.assertIn(("entities", "label"), task.hidden_columns())
        self.assertEqual(
            hidden_columns(
                {"kind": "forecast"},
                name="predict",
                entity_table="entities",
                target_column="label",
            ),
            (),
        )
        self.assertEqual(
            hidden_columns(
                {"kind": "external"},
                name="beer_ratings-total_score",
                entity_table="beer_ratings",
                target_column="total_score",
            ),
            (("beer_ratings", "total_score"),),
        )

    def test_all_versions_rowid_sparse_and_dense_use_index(self):
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE notes (text TEXT)")
            c.executemany(
                "INSERT INTO notes VALUES (?)", [("zero",), ("one",), ("two",)]
            )
        for version in VERSIONS:
            connector = importlib.import_module(version + ".connector")
            reader = connector.PandasDatabaseReader(self.path)
            schema = connector.TableSchema(
                "notes", None, columns=(connector.ColumnSchema("text", 0, "object"),)
            )
            reader.schemas()["notes"] = schema
            fetcher_type = importlib.import_module(
                version + ".features"
            ).SqlFeatureFetcher
            for threshold in [1.0, 0.25]:
                fetcher = fetcher_type(
                    reader, block_size=4, block_density_threshold=threshold
                )
                with patch.object(
                    reader, "read_query", wraps=reader.read_query
                ) as read:
                    values, _ = fetcher._fetch_table("notes", torch.tensor([2, 0, 2]))
                self.assertEqual(values["text"].tolist(), ["two", "zero", "two"])
                sql = read.call_args.args[0]
                with sqlite3.connect(self.path) as c:
                    plan = " ".join(
                        row[3] for row in c.execute("EXPLAIN QUERY PLAN " + sql)
                    )
                self.assertIn("SEARCH", plan)
                self.assertNotIn("SCAN notes", plan)

    def test_cache_owns_its_charged_storage(self):
        for version in ("relconnector", "baseline.cache_baseline"):
            embedder_type = importlib.import_module(
                version + ".features.text"
            ).GloveTextEmbedder
            with patch("sentence_transformers.SentenceTransformer"):
                embedder = embedder_type(cache_bytes=1200)
            backing = torch.arange(300 * 128, dtype=torch.float32).reshape(128, 300)
            embedder._put("one", backing[0])
            cached = embedder._cache["one"]
            if isinstance(cached, tuple):
                cached = embedder._slabs[cached[0]][cached[1]]
            self.assertEqual(cached.untyped_storage().nbytes(), 1200)
            backing.zero_()
            self.assertEqual(cached[1].item(), 1.0)

    def test_featureless_rows_are_excluded_from_sql_amplification(self):
        for version in ("relconnector", "baseline.cache_baseline"):
            stats_type = importlib.import_module(
                version + ".features.sql"
            ).FeatureFetchStats
            stats = stats_type(
                unique_rows=100,
                featureless_rows=80,
                cache_served_rows=10,
                queried_rows=10,
            )
            self.assertEqual(stats.sql_amplification, 1.0)

    def test_four_versions_fixed_plan_tensorframe_and_cpu_step(self):
        # Identity-free text fixture: same output irrespective of chunk/window sizes.
        class TextEncoder:
            embedding_dim = 3

            def __call__(self, sentences):
                return torch.tensor([[len(str(v)), 1.0, 2.0] for v in sentences])

        from data.models import ForeignKeySchema

        self.frame["parent_id"] = self.frame["id"]
        schema = TableSchema(
            "entities",
            "id",
            columns=infer_columns(self.frame),
            foreign_keys=(ForeignKeySchema("parent_id", "entities", "id"),),
        )
        SQLiteDatabaseWriter(self.path).write(
            DatasetBundle(
                "fixture", {"entities": MaterializedTable(self.frame, schema)}
            ),
            overwrite=True,
        )
        initialize_database(self.path)
        torch.set_num_threads(1)
        results = []
        for version, eager in [(v, False) for v in VERSIONS] + [
            ("baseline.vanilla_baseline", True)
        ]:
            connector = importlib.import_module(version + ".connector")
            features = importlib.import_module(version + ".features")
            sampling = importlib.import_module(version + ".sampling")
            tasks = importlib.import_module(version + ".task")
            graph_type = importlib.import_module(
                version + ".graph"
            ).InMemoryGraphIndexBuilder
            trainer_type = importlib.import_module(version + ".training").OnlineTrainer
            reader = connector.PandasDatabaseReader(self.path)
            schema = features.SqlTensorFrameSchemaBuilder(
                reader,
                hidden_columns=[("entities", "label")],
                text_embedding_dim=3,
                require_stypes=True,
            ).build()
            graph = graph_type(reader).build()
            sample = sampling.SampledSubgraph(
                node_ids={"entities": torch.tensor([1102, 3, 1102, 1500])},
                edge_index={
                    edge: torch.stack((torch.arange(4), torch.arange(4)))
                    for edge in graph.edge_types
                },
                batch=None,
                node_time={},
                num_sampled_nodes={"entities": [4, 0]},
                num_sampled_edges={edge: [4] for edge in graph.edge_types},
                seed_node_type="entities",
                seed_count=4,
                seed_time=None,
            )
            plan = sampling.EntitySamplePlan(
                tasks.BatchKey(0, 0), sample, torch.tensor([0.0, 1.0, 0.0, 0.0])
            )
            encoder = features.TensorFrameEncoder(
                schema, TextEncoder(), text_batch_size=8
            )
            if eager:
                from baseline.batch_baseline.feature_store import (
                    EagerFeatureStore,
                    InMemoryTensorFrameFetcher,
                )

                store = EagerFeatureStore.materialize(
                    Database({"entities": Table(self.frame, {}, pkey_col="id")}),
                    encoder,
                )
                fetcher = InMemoryTensorFrameFetcher(store)
            else:
                fetcher = features.SqlFeatureFetcher(
                    reader, feature_columns=schema.feature_columns
                )
            batch = features.TensorFrameBatchAssembler(encoder).assemble(
                fetcher.fetch(plan)
            )
            tensor_frame = batch.data["entities"].tf
            task = tasks.TaskSpec.from_metadata(
                connector.catalog.TaskMetadata(
                    name="predict",
                    table_name="targets",
                    task_type="binary_classification",
                    entity_table="entities",
                    entity_column="id",
                    target_column="label",
                    extra={"kind": "autocomplete"},
                )
            )
            torch.manual_seed(72)
            trainer = trainer_type(
                graph,
                task,
                feature_schema=schema,
                channels=8,
                num_layers=1,
                device="cpu",
                seed=72,
            )
            outcome = trainer.train_step(batch)
            results.append(
                (
                    tensor_frame,
                    outcome.loss,
                    {k: v.clone() for k, v in trainer.model.state_dict().items()},
                )
            )
        reference = results[0]
        for actual in results[1:]:
            self.assertEqual(reference[0], actual[0])
            self.assertEqual(reference[1], actual[1])
            for name in reference[2]:
                self.assertTrue(torch.equal(reference[2][name], actual[2][name]), name)

    def test_baseline_imports_and_default_paths(self):
        from baseline.batch_baseline.dataset import default_sqlite_path
        from benchmark.baseline import BaselineExperiment

        self.assertIsNotNone(BaselineExperiment)
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            default_sqlite_path("fixture"), root / "data/relbench/fixture.sqlite"
        )
        for version in VERSIONS:
            model = importlib.import_module(version + ".api").OnlineRelBenchModel(
                dataset="fixture", task="task"
            )
            self.assertEqual(model.sqlite_dir, root / "data/relbench")


if __name__ == "__main__":
    unittest.main()
