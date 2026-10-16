"""Summarize completed ablations and verify cache-disable telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    rows = []
    problems = []
    for run in manifest["runs"]:
        path = Path(run["directory"]) / "result.jsonl"
        if not path.exists():
            problems.append(f"missing result: {path}")
            continue
        results = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(results) != 1 or results[0].get("status") != "ok":
            problems.append(f"expected exactly one successful result: {path}")
            continue
        result = results[0]
        config = result["config"]
        pipeline = result["pipeline"]
        reports = [w for w in pipeline["workers"] if w["stage"] == "assembly"]
        workers = [w["diagnostics"] for w in reports]
        label = f"r{run['repeat']}-{run['variant']}"
        if {w["worker"] for w in reports} != set(range(config["assembly_processes"])):
            problems.append(f"{label}: missing/invalid worker IDs")
        if sum(w["batches"] for w in reports) != result["completed_batches"]:
            problems.append(f"{label}: assembly/trainer batch counts disagree")
        for worker in workers:
            raw = worker["feature_block_cache"]
            encoded = worker["encoded_feature_cache"]
            text = worker["text_embedding"]
            if config["feature_cache_bytes"] == 0 and (
                raw["insertions"] or raw["current_bytes"]
            ):
                problems.append(f"{label}: disabled raw cache retained/wrote data")
            if config["encoded_feature_cache_bytes"] == 0 and (
                encoded["inserted_rows"] or encoded["current_bytes"]
            ):
                problems.append(f"{label}: disabled encoded cache retained/wrote data")
            if config["text_embedding_cache_bytes"] == 0 and (
                text["shared"] is not None or text["current_bytes"]
            ):
                problems.append(f"{label}: disabled RAM text cache retained data")
            if config["disk_text_cache_mode"] == "off" and text["disk"]["active"]:
                problems.append(f"{label}: disabled disk cache active")
        rows.append(
            {
                "variant": run["variant"],
                "repeat": run["repeat"],
                "workers": config["assembly_processes"],
                "batches": result["completed_batches"],
                "examples": result["completed_examples"],
                "benchmark_s": result["duration_s"],
                "pipeline_s": pipeline["duration_s"],
                "ready_wait_s": pipeline["ready_wait_s"],
                "mean_loss": result["mean_loss"],
                "raw_insertions": sum(
                    w["feature_block_cache"]["insertions"] for w in workers
                ),
                "text_model_inputs": sum(
                    w["text_embedding"]["model_inputs"] for w in workers
                ),
                "text_cache_hits": sum(
                    w["text_embedding"]["cache_hits"] for w in workers
                ),
                "disk_active_workers": sum(
                    bool(w["text_embedding"]["disk"]["active"]) for w in workers
                ),
            }
        )
    # Paired workloads must match even when ready order changes SGD trajectories.
    for repeat in {row["repeat"] for row in rows}:
        counts = {
            (row["batches"], row["examples"]) for row in rows if row["repeat"] == repeat
        }
        if len(counts) != 1:
            problems.append(f"repeat {repeat}: workload counts differ")
    if rows:
        with (args.directory / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "completed": len(rows),
        "expected": len(manifest["runs"]),
        "problems": problems,
        "note": "Checks cover telemetry and counts, not tensor equivalence or training quality. Times include startup; no full-process memory measurement.",
        "rows": rows,
    }
    (args.directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
