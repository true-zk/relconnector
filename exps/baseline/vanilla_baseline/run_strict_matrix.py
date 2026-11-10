"""Run the strict vanilla benchmark over the ten established RelBench tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CASES = (
    ("rel-f1", "driver-circuit-compete"),
    ("rel-avito", "searchstream-click"),
    ("rel-hm", "user-churn"),
    ("rel-trial", "eligibilities-adult"),
    ("rel-arxiv", "paper-citation"),
    ("rel-salt", "item-incoterms"),
    ("rel-stack", "user-badge"),
    ("rel-amazon", "item-churn"),
    ("rel-ratebeer", "beer_ratings-total_score"),
    ("rel-event", "user-attendance"),
)


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "relbench")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="128,128")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parents[1]
    hashes = {
        str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package_root.rglob("*.py")
    }
    manifest = {
        "cases": CASES, "epochs": args.epochs,
        "max_batches_per_epoch": args.max_batches,
        "batch_size": args.batch_size, "num_neighbors": args.num_neighbors,
        "channels": args.channels, "seed": args.seed,
        "torch_num_threads": args.torch_num_threads,
        "text_batch_size": args.text_batch_size,
        "device": args.device, "database_dir": str(args.database_dir.resolve()),
        "source_sha256": hashes,
        "python": sys.executable,
        "environment": {"PYTHONPATH": str(root / "exps")},
        "protocol": "Fresh process for each dataset, exactly one active benchmark process at a time. Strict serial batch pipeline, direct exact SQL IN, no application caches or replay.",
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("manifest differs from resumed run")
    else:
        save(manifest_path, manifest)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "exps")
    status_path = output / "status.json"
    status = {"status": "running", "started_unix": time.time(),
              "current_dataset": None, "completed": [], "failed": []}
    if args.resume and status_path.exists():
        status = json.loads(status_path.read_text())
        status["status"] = "running"
    save(status_path, status)
    for dataset, task in CASES:
        result_path = output / f"{dataset}.json"
        if args.resume and result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("status") == "ok":
                if dataset not in status["completed"]:
                    status["completed"].append(dataset)
                save(status_path, status)
                continue
        command = [
            sys.executable, "-m", "baseline.vanilla_baseline.strict_benchmark",
            "--dataset", dataset, "--task", task,
            "--database-dir", str(args.database_dir.resolve()),
            "--output", str(result_path),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--max-batches", str(args.max_batches),
            "--batch-size", str(args.batch_size),
            "--num-neighbors", args.num_neighbors,
            "--channels", str(args.channels),
            "--seed", str(args.seed),
            "--torch-num-threads", str(args.torch_num_threads),
            "--text-batch-size", str(args.text_batch_size),
        ]
        status["current_dataset"] = dataset
        save(status_path, status)
        print(f"start {dataset}/{task}", flush=True)
        with (output / f"{dataset}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0:
            status["completed"].append(dataset)
            print(f"complete {dataset}", flush=True)
        else:
            status["failed"].append({"dataset": dataset, "exit_code": result.returncode,
                                     "result": str(result_path), "log": str(output / f"{dataset}.log")})
            print(f"failed {dataset}: exit {result.returncode}", flush=True)
        save(status_path, status)
    status["current_dataset"] = None
    status["finished_unix"] = time.time()
    status["status"] = "ok" if len(status["completed"]) == len(CASES) else "incomplete"
    save(status_path, status)
    return 0 if status["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
