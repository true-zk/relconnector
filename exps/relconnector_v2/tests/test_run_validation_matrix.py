from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from relconnector_v2.run_validation_matrix import _completed, main


class ValidationMatrixResumeTest(unittest.TestCase):
    def test_all_tasks_use_same_adaptive_text_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "--independent",
                            "--dry-run",
                            "--output-dir",
                            directory,
                        ]
                    ),
                    0,
                )
            commands = [
                line
                for line in output.getvalue().splitlines()
                if "-m relconnector_v2.benchmark" in line
            ]
            self.assertEqual(len(commands), 10)
            self.assertTrue(
                all(
                    "--text-cache-scope adaptive --text-cache-mb 512" in command
                    for command in commands
                )
            )

    def test_resume_retries_hm_when_cache_scope_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v2.jsonl"
            rows = [
                {
                    "status": "ok",
                    "dataset": "rel-hm",
                    "task": "user-churn",
                    "config": {"encoded_cache_scope": "shared"},
                },
                {
                    "status": "ok",
                    "dataset": "rel-f1",
                    "task": "driver-circuit-compete",
                    "config": {"encoded_cache_scope": "shared"},
                },
            ]
            output.write_text("".join(json.dumps(row) + "\n" for row in rows))
            expected = {"rel-hm": ("local", 4096)}
            self.assertEqual(
                _completed(output, encoded_cache_profile_by_dataset=expected),
                {("rel-f1", "driver-circuit-compete")},
            )
            with output.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            **rows[0],
                            "config": {
                                "encoded_cache_scope": "local",
                                "encoded_feature_cache_bytes": 4096 * 1024 * 1024,
                            },
                        }
                    )
                    + "\n"
                )
            self.assertIn(
                ("rel-hm", "user-churn"),
                _completed(output, encoded_cache_profile_by_dataset=expected),
            )

    def test_resume_can_require_adaptive_text_without_invalidating_other_tasks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v2.jsonl"
            row = {
                "status": "ok",
                "dataset": "rel-salt",
                "task": "item-incoterms",
                "config": {
                    "text_cache_scope": "shared",
                    "text_embedding_cache_bytes": 512 * 1024 * 1024,
                },
            }
            output.write_text(json.dumps(row) + "\n")
            expected = {"rel-salt": ("adaptive", 512)}
            self.assertIn(("rel-salt", "item-incoterms"), _completed(output))
            self.assertNotIn(
                ("rel-salt", "item-incoterms"),
                _completed(output, text_cache_profile_by_dataset=expected),
            )
            row["config"] = {
                "text_cache_scope": "adaptive",
                "text_embedding_cache_bytes": 512 * 1024 * 1024,
            }
            with output.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            self.assertIn(
                ("rel-salt", "item-incoterms"),
                _completed(output, text_cache_profile_by_dataset=expected),
            )


if __name__ == "__main__":
    unittest.main()
