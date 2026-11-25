"""Run generated RelX SQLite smoke cases in separate, bounded processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

CASES = (
    "rel-f1", "rel-arxiv", "rel-trial", "rel-event", "rel-hm",
    "rel-avito", "rel-salt", "rel-stack", "rel-amazon", "rel-ratebeer",
)

_RUN = """
import json
import sys
from relx import run
result = run(sys.argv[1])
print(json.dumps({
    "task": result.report.task_name,
    "metric": result.report.metric,
    "node_counts": result.prepared.report.node_counts,
    "relation_count": result.prepared.report.relation_count,
    "feature_count": result.prepared.report.feature_count,
    "train_rows": result.report.train_rows,
    "validation_rows": result.report.validation_rows,
    "test_rows": result.report.test_rows,
    "train_loss": result.report.epochs[-1].train_loss,
    "validation_metric": result.report.epochs[-1].validation_metric,
    "test_metric": result.report.test_metric,
}), flush=True)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=Path("build/relx_ten_smoke")
    )
    parser.add_argument("--dataset", choices=CASES, action="append")
    parser.add_argument("--timeout", type=int, default=1_800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_dir = args.config_dir.resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    env["TMPDIR"] = str(root / "build")
    selected = args.dataset or list(CASES)
    failures = 0
    for dataset in selected:
        report = config_dir / dataset / "report.json"
        if args.resume and report.is_file():
            print(f"{dataset}: existing report {report}", flush=True)
            continue
        config = config_dir / f"{dataset}.yaml"
        log = config_dir / f"{dataset}.log"
        print(f"{dataset}: running {config}", flush=True)
        started = perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _RUN, str(config)],
                cwd=root, env=env, text=True, capture_output=True,
                timeout=args.timeout, check=False,
            )
            log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            status = "ok" if completed.returncode == 0 else "failed"
            if status == "failed":
                failures += 1
            print(
                f"{dataset}: {status}, {perf_counter() - started:.1f}s, "
                f"log={log}", flush=True,
            )
        except subprocess.TimeoutExpired as exc:
            failures += 1
            log.write_text(
                f"stdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}",
                encoding="utf-8",
            )
            print(f"{dataset}: timeout after {args.timeout}s, log={log}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
