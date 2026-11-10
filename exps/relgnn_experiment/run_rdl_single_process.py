"""Complete the missing one-epoch RDL single-process comparison on GPU 0."""

from __future__ import annotations

import json
import math
import os
import resource
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BENCH = PROJECT / "exps/benchmarks"
ROOT = BENCH / "rdl-single-process-20261103"
SOURCE = BENCH / "relgnn-three-way-20261101/frozen-source-v2"
RDL_NINE = BENCH / "hyperbatch-node-block-nine-20261030"
RDL_SALT = BENCH / "hyperbatch-node-block-rel-salt-20261029"
REL = BENCH / "relgnn-three-way-20261101"
PYTHON = PROJECT / ".venv/bin/python"
DATABASE = PROJECT / "exps/data/relbench"
SCRATCH = PROJECT / ".v2tmp"
REPORT_BUILDER = PROJECT / "exps/relgnn_experiment/build_rdl_relgnn_report.py"
CASES = tuple(
    tuple(case)
    for case in json.loads(
        (BENCH / "strict-vanilla-20261027/manifest.json").read_text()
    )["cases"]
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def status(stage: str, dataset: str = "", **details: object) -> None:
    complete = [name for name, _ in CASES if (ROOT / name / "summary.json").exists()]
    payload = {
        "status": stage,
        "dataset": dataset,
        "completed_datasets": complete,
        "total_datasets": len(CASES),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": "cuda:0",
        **details,
    }
    write_json(ROOT / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run(command: list[str], log: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(SOURCE),
            "TMPDIR": str(SCRATCH),
            "SQLITE_TMPDIR": str(SCRATCH),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "RELCONNECTOR_REPLAY_MIN_FREE_BYTES": str(20 * 1024**3),
        }
    )
    with log.open("w") as handle:
        process = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(f"exit {process.returncode}; see {log}")


def wait_for_gpu(dataset: str) -> None:
    waiting = False
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        free_mib = int(result.stdout.strip())
        if free_mib >= 20 * 1024:
            return
        if not waiting:
            status(
                "waiting_gpu", dataset, free_mib=free_mib, required_free_mib=20 * 1024
            )
            waiting = True
        time.sleep(30)


def common(dataset: str, task: str, cache: Path) -> list[str]:
    return [
        "--dataset",
        dataset,
        "--task",
        task,
        "--database-dir",
        str(DATABASE),
        "--cache-dir",
        str(cache),
        "--model",
        "graphsage",
        "--max-batches",
        "512",
        "--batch-size",
        "512",
        "--num-neighbors",
        "128,128",
        "--channels",
        "128",
        "--seed",
        "42",
        "--device",
        "cuda:0",
        "--torch-num-threads",
        "1",
        "--epochs",
        "1",
    ]


def initial_cache(dataset: str) -> Path:
    if dataset == "rel-salt":
        return RDL_SALT / "cache"
    return RDL_NINE / dataset / "cache"


def expected(dataset: str) -> tuple[str, int, int]:
    summary = json.loads((REL / dataset / "summary.json").read_text())
    return (
        summary["replay_fingerprint"],
        summary["batches_per_epoch"],
        summary["examples_per_epoch"],
    )


def refresh_report() -> None:
    result = subprocess.run(
        ["python", str(REPORT_BUILDER)],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"report generation failed: {result.stderr[-1500:]}")
    print(result.stdout.strip(), flush=True)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(exist_ok=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 65_536:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65_536, hard), hard))
    status("starting")
    current = ""
    try:
        for dataset, task in CASES:
            current = dataset
            directory = ROOT / dataset
            directory.mkdir(exist_ok=True)
            if (directory / "summary.json").exists():
                continue
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise OSError("less than 20 GiB free before dataset")
            fingerprint, batches, examples = expected(dataset)
            cache = directory / "cache"
            cache.mkdir(exist_ok=True)
            source_cache = initial_cache(dataset)
            for part in ("schema", "topology"):
                source_part = source_cache / part
                if not source_part.exists():
                    raise FileNotFoundError(source_part)
                destination = cache / part
                if not destination.exists():
                    shutil.copytree(source_part, destination, copy_function=os.link)
            args = common(dataset, task, cache)
            status("warming_initialization", dataset)
            wait_for_gpu(dataset)
            run(
                [
                    str(PYTHON),
                    "-m",
                    "relconnector_v2.benchmark",
                    *args,
                    "--prepare-only",
                ],
                directory / "prepare.log",
            )
            replay = (
                RDL_SALT / "replay" if dataset == "rel-salt" else directory / "replay"
            )
            if dataset != "rel-salt" and (replay / "INCOMPLETE").exists():
                shutil.rmtree(replay)
            if not replay.exists():
                status("preparing_replay", dataset)
                run(
                    [
                        str(PYTHON),
                        "-m",
                        "relconnector_v2.benchmark",
                        *args,
                        "--prepare-replay-only",
                        "--replay-dir",
                        str(replay),
                    ],
                    directory / "replay.log",
                )
            manifest = json.loads((replay / "manifest.json").read_text())
            if (replay / "INCOMPLETE").exists():
                raise RuntimeError(f"incomplete replay: {replay}")
            if (
                manifest["fingerprint"] != fingerprint
                or len(manifest["batches"]) != batches
            ):
                raise RuntimeError(f"replay differs from prior experiments: {dataset}")
            status(
                "training",
                dataset,
                expected_batches=batches,
                replay_fingerprint=fingerprint,
            )
            wait_for_gpu(dataset)
            output = directory / "single-process.json"
            output.unlink(missing_ok=True)
            run(
                [
                    str(PYTHON),
                    "-m",
                    "relconnector_v2.fixed_v2_benchmark",
                    *args,
                    "--replay-dir",
                    str(replay),
                    "--output",
                    str(output),
                    "--executor",
                    "sync",
                    "--text-execution",
                    "official",
                    "--window-batches",
                    "1",
                ],
                directory / "single-process.log",
            )
            result = json.loads(output.read_text())
            if (
                result.get("status") != "ok"
                or result.get("model") != "graphsage"
                or result.get("fingerprint") != fingerprint
                or result.get("completed_batches") != batches
                or result.get("completed_examples") != examples
                or result.get("executor") != "sync"
                or result.get("text_execution") != "official"
                or result.get("window_batches") != 1
                or not all(result.get("initialization_cache", {}).values())
                or not math.isfinite(float(result.get("mean_loss", float("nan"))))
            ):
                raise RuntimeError(f"invalid single-process result: {dataset}")
            write_json(
                directory / "summary.json",
                {
                    "status": "ok",
                    "dataset": dataset,
                    "task": task,
                    "model": "graphsage",
                    "method": "single-process",
                    "device": "cuda:0",
                    "epochs": 1,
                    "batches": batches,
                    "examples": examples,
                    "training_s": result["training_s"],
                    "replay_fingerprint": fingerprint,
                    "source_snapshot": str(SOURCE),
                },
            )
            if dataset != "rel-salt":
                shutil.rmtree(replay)
            shutil.rmtree(cache)
            refresh_report()
            status("dataset_complete", dataset)
        refresh_report()
        status("complete", report=str(BENCH / "rdl-relgnn-summary-20261103"))
        return 0
    except Exception as error:  # noqa: BLE001 - persist any benchmark failure
        status(
            "failed",
            current,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
