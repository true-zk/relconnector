"""Run the matched RelGNN vanilla / v2 vanilla / optimized v2 matrix.

Each dataset gets one immutable replay shared by all three groups.  Schema and
topology artifacts are prepared before timing the training groups.  The three
groups run sequentially, with 1, 1, and 5 epochs respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT / ".venv/bin/python"
DATABASE = PROJECT / "exps/data/relbench"
SCRATCH = PROJECT / ".v2tmp"
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


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _source_snapshot(output: Path, name: str = "frozen-source") -> Path:
    source = output / name
    if source.exists():
        return source
    source.mkdir(parents=True)
    for origin, destination in (
        (PROJECT / "exps/baseline/vanilla_baseline", source / "baseline/vanilla_baseline"),
        (PROJECT / "exps/relconnector_v2/relconnector_v2", source / "relconnector_v2"),
        (PROJECT / "exps/relgnn_experiment", source / "relgnn_experiment"),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(origin, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The live umbrella package eagerly imports other baselines; this frozen
    # experiment intentionally ships only vanilla_baseline.
    (source / "baseline/__init__.py").write_text('"""Frozen vanilla baseline."""\n')
    hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*.py")
    }
    hash_name = "source-sha256.json" if name == "frozen-source" else f"{name}-sha256.json"
    _write(output / hash_name, hashes)
    return source


def _run(command: list[str], log: Path, source: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(source),
        TMPDIR=str(SCRATCH),
        SQLITE_TMPDIR=str(SCRATCH),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        RELCONNECTOR_REPLAY_MIN_FREE_BYTES=str(16 * 1024**3),
    )
    with log.open("w") as handle:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {log}")


def _read_jsonl(path: Path) -> dict:
    with path.open() as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"empty result: {path}")
    return json.loads(lines[-1])


def _check_result(
    result: dict,
    *,
    group: str,
    epochs: int,
    batches: int,
    examples: int | None,
    fingerprint: str,
) -> int:
    if result.get("status") != "ok":
        raise ValueError(f"{group}: result status is not ok")
    if group == "v2_optimized":
        if result.get("config", {}).get("model_kind") != "relgnn":
            raise ValueError("optimized v2 used a different model")
        observed_epochs = result.get("pipeline", {}).get("epochs", [])
        if len(observed_epochs) != epochs:
            raise ValueError("optimized v2 epoch count differs")
        if any(row.get("steps") != batches for row in observed_epochs):
            raise ValueError("optimized v2 epoch batch count differs")
        if any(row.get("examples") != observed_epochs[0].get("examples") for row in observed_epochs):
            raise ValueError("optimized v2 epoch sample count differs")
        if result.get("pipeline", {}).get("replay", {}).get("fingerprint") != fingerprint:
            raise ValueError("optimized v2 replay differs")
    elif result.get("model") != "relgnn" or result.get("fingerprint") != fingerprint:
        raise ValueError(f"{group}: model or replay differs")
    if result.get("completed_batches") != epochs * batches:
        raise ValueError(f"{group}: batch count differs")
    observed_examples = result.get("completed_examples")
    if not isinstance(observed_examples, int) or observed_examples <= 0:
        raise ValueError(f"{group}: invalid sample count")
    if examples is not None and observed_examples != examples * epochs:
        raise ValueError(f"{group}: sample count differs")
    if not math.isfinite(float(result.get("mean_loss", float("nan")))):
        raise ValueError(f"{group}: non-finite loss")
    return observed_examples // epochs


def _common(dataset: str, task: str, cache: Path) -> list[str]:
    return [
        "--dataset", dataset, "--task", task,
        "--database-dir", str(DATABASE), "--cache-dir", str(cache),
        "--model", "relgnn", "--max-batches", "512", "--batch-size", "512",
        "--num-neighbors", "128,128", "--channels", "128", "--seed", "42",
        "--device", "cuda:1", "--torch-num-threads", "1",
    ]


def _status(output: Path, stage: str, dataset: str = "", group: str = "", **details: object) -> None:
    complete = [name for name, _ in CASES if (output / name / "summary.json").exists()]
    payload = {
        "status": stage,
        "dataset": dataset,
        "group": group,
        "completed_datasets": complete,
        "total_datasets": len(CASES),
        "updated_unix": time.time(),
        **details,
    }
    _write(output / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _wait_for_gpu(output: Path, dataset: str, group: str) -> None:
    """Keep a training group from starting while another job fills GPU 1."""
    required_free_mib = 20 * 1024
    waiting = False
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        free_mib = int(result.stdout.strip())
        if free_mib >= required_free_mib:
            return
        if not waiting:
            _status(output, "waiting_gpu", dataset, group,
                    gpu_free_mib=free_mib, required_free_mib=required_free_mib)
            waiting = True
        time.sleep(30)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-name", choices=("frozen-source", "frozen-source-v2"),
        default="frozen-source",
    )
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(exist_ok=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 65_536:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65_536, hard), hard))
    source = _source_snapshot(output, args.source_name)
    protocol = {
        "datasets": CASES,
        "model": "RelGNN atomic routes; 2 composite layers, 1 attention head, 128 channels",
        "upstream_model": "https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b",
        "adaptation": "Author atomic routes and composite attention backbone; existing online sampler, feature/temporal encoder, task head and loss; common model settings rather than per-task author tuning",
        "replay": "one immutable sampled plan per dataset, shared by all three groups",
        "groups": {
            "vanilla": "baseline fixed replay, exact SQL, no feature/text cache, sync, 1 epoch",
            "v2_vanilla": "v2 fixed replay, no feature/text cache, official text, 1-batch window, sync, 1 epoch",
            "v2_optimized": "v2 node_block disk text cache, direct text, 4 assembly processes, 5 epochs",
        },
        "batches_per_epoch": 512,
        "batch_size": 512,
        "fanout": [128, 128],
        "source_snapshot": args.source_name,
        "source_sha256": (
            "source-sha256.json" if args.source_name == "frozen-source"
            else f"{args.source_name}-sha256.json"
        ),
    }
    if args.source_name == "frozen-source-v2":
        protocol["amendment"] = (
            "Resume after rel-salt vanilla CUDA OOM; only RelGNN route aggregation "
            "changed from stack-then-sum to sequential sum. Completed results "
            "for rel-f1, rel-avito, rel-hm, rel-trial, rel-arxiv retain v1 source."
        )
    protocol_path = (
        output / "protocol.json" if args.source_name == "frozen-source"
        else output / "protocol-amendment-v2.json"
    )
    _write(protocol_path, protocol)
    current_dataset = ""
    current_group = ""
    try:
        for dataset, task in CASES:
            current_dataset = dataset
            current_group = ""
            directory = output / dataset
            if (directory / "summary.json").exists():
                continue
            directory.mkdir(exist_ok=True)
            if shutil.disk_usage(output).free < 20 * 1024**3:
                raise OSError("less than 20 GiB free on /home before dataset")
            replay = directory / "replay"
            v2_cache = directory / "v2-cache"
            vanilla_cache = directory / "vanilla-cache"
            v2_common = _common(dataset, task, v2_cache)
            vanilla_common = _common(dataset, task, vanilla_cache)
            _status(output, "preparing_initialization", dataset)
            _run(
                [str(PYTHON), "-m", "relconnector_v2.benchmark", *v2_common,
                 "--epochs", "1", "--prepare-only"],
                directory / "v2-prepare.log", source,
            )
            _status(output, "preparing_replay", dataset)
            if replay.exists():
                if (replay / "INCOMPLETE").exists():
                    raise ValueError(f"{dataset}: existing replay is incomplete")
                print(f"[replay] reusing {replay}", flush=True)
            else:
                _run(
                    [str(PYTHON), "-m", "relconnector_v2.benchmark", *v2_common,
                     "--epochs", "1", "--prepare-replay-only", "--replay-dir", str(replay)],
                    directory / "replay.log", source,
                )
            manifest = json.loads((replay / "manifest.json").read_text())
            batches = len(manifest["batches"])
            if not 0 < batches <= 512:
                raise ValueError(f"{dataset}: invalid replay batch count {batches}")
            fingerprint = manifest["fingerprint"]
            prior_failure = output / f"{dataset}-failed-v1/failure-analysis.json"
            if prior_failure.exists():
                expected_fingerprint = json.loads(prior_failure.read_text())[
                    "original_replay_fingerprint"
                ]
                if fingerprint != expected_fingerprint:
                    raise ValueError(f"{dataset}: regenerated replay differs from v1")
            _run(
                [str(PYTHON), "-m", "relconnector_v2.fixed_vanilla_benchmark",
                 *vanilla_common, "--replay-dir", str(replay),
                 "--output", str(directory / "vanilla-prepare.json"),
                 "--executor", "sync", "--epochs", "1",
                 "--feature-read-policy", "exact", "--prepare-only"],
                directory / "vanilla-prepare.log", source,
            )
            _wait_for_gpu(output, dataset, "vanilla")
            _status(output, "training", dataset, "vanilla", batches_per_epoch=batches)
            current_group = "vanilla"
            _run(
                [str(PYTHON), "-m", "relconnector_v2.fixed_vanilla_benchmark",
                 *vanilla_common, "--replay-dir", str(replay),
                 "--output", str(directory / "vanilla.json"),
                 "--executor", "sync", "--epochs", "1",
                 "--feature-read-policy", "exact"],
                directory / "vanilla.log", source,
            )
            vanilla = json.loads((directory / "vanilla.json").read_text())
            one_epoch_examples = _check_result(
                vanilla, group="vanilla", epochs=1, batches=batches,
                examples=None, fingerprint=fingerprint,
            )
            _wait_for_gpu(output, dataset, "v2_vanilla")
            _status(output, "training", dataset, "v2_vanilla", batches_per_epoch=batches)
            current_group = "v2_vanilla"
            _run(
                [str(PYTHON), "-m", "relconnector_v2.fixed_v2_benchmark",
                 *v2_common, "--replay-dir", str(replay),
                 "--output", str(directory / "v2-vanilla.json"),
                 "--executor", "sync", "--text-execution", "official",
                 "--window-batches", "1", "--epochs", "1"],
                directory / "v2-vanilla.log", source,
            )
            v2_vanilla = json.loads((directory / "v2-vanilla.json").read_text())
            _check_result(
                v2_vanilla, group="v2_vanilla", epochs=1, batches=batches,
                examples=one_epoch_examples, fingerprint=fingerprint,
            )
            _wait_for_gpu(output, dataset, "v2_optimized")
            _status(output, "training", dataset, "v2_optimized", batches_per_epoch=batches)
            current_group = "v2_optimized"
            _run(
                [str(PYTHON), "-m", "relconnector_v2.benchmark", *v2_common,
                 "--epochs", "5", "--replay-dir", str(replay),
                 "--output", str(directory / "v2-optimized.jsonl"),
                 "--assembly-processes", "4", "--initial-active-processes", "4",
                 "--no-adaptive-workers", "--training-order", "strict",
                 "--no-cross-epoch-prefetch", "--read-profile", "tuned",
                 "--feature-cache-mb", "0", "--encoded-cache-mb", "0",
                 "--text-cache-mb", "0", "--text-execution", "direct",
                 "--ready-queue-mb", "8192", "--progress-interval-s", "30",
                 "--disk-text-cache-layout", "node_block", "--disk-text-cache-mode", "on",
                 "--window-batches", "8", "--cold-window-batches", "4",
                 "--max-inflight-windows", "4", "--node-block-warm-index",
                 "--node-block-max-mapped-shards", "32768"],
                directory / "v2-optimized.log", source,
            )
            optimized = _read_jsonl(directory / "v2-optimized.jsonl")
            _check_result(
                optimized, group="v2_optimized", epochs=5, batches=batches,
                examples=one_epoch_examples, fingerprint=fingerprint,
            )
            _write(directory / "summary.json", {
                "dataset": dataset,
                "task": task,
                "model": "relgnn",
                "source_snapshot": args.source_name,
                "replay_fingerprint": fingerprint,
                "batches_per_epoch": batches,
                "examples_per_epoch": one_epoch_examples,
                "vanilla_training_s": vanilla["training_s"],
                "v2_vanilla_training_s": v2_vanilla["training_s"],
                "v2_optimized_epoch_s": [row["wall_duration_s"] for row in optimized["pipeline"]["epochs"]],
                "v2_optimized_training_s": optimized["pipeline"]["duration_s"],
                "v2_optimized_cache_hit": optimized["initialization_cache"],
            })
            shutil.rmtree(replay)
            shutil.rmtree(v2_cache)
            shutil.rmtree(vanilla_cache)
            _status(output, "dataset_complete", dataset)
        _status(output, "complete")
        return 0
    except Exception as exc:  # noqa: BLE001 - persist any task failure in status.json
        _status(
            output, "failed", current_dataset, current_group,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
