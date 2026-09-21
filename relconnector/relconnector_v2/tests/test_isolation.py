from __future__ import annotations

import ast
import unittest
from pathlib import Path


class PackageIsolationTest(unittest.TestCase):
    def test_v2_has_no_legacy_package_imports(self) -> None:
        package = Path(__file__).resolve().parents[1] / "relconnector_v2"
        violations: list[str] = []
        for source in package.rglob("*.py"):
            tree = ast.parse(source.read_text(), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [] if node.module is None else [node.module]
                else:
                    continue
                for name in names:
                    if name == "relconnector" or name.startswith("relconnector."):
                        violations.append(f"{source}:{node.lineno}: {name}")
                    if name == "relbench_compat" or name.startswith(
                        "relbench_compat."
                    ):
                        violations.append(f"{source}:{node.lineno}: {name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
