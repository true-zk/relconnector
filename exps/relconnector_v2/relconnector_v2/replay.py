"""Versioned fixed seed/subgraph artifacts for cache experiments, not training defaults."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from .runtime.contracts import Sampler, SeedSource
from .sampling import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
    SamplePlan,
)
from .task import BatchKey, EntitySeedBatch, RecommendationSeedBatch, SeedBatch


def _pack(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _write_plan(path: Path, plan: SamplePlan) -> str:
    if isinstance(plan, EntitySamplePlan):
        payload = {
            "kind": "entity",
            "subgraph": _pack(plan.subgraph),
            "target": plan.target,
        }
    else:
        payload = {
            "kind": "recommendation",
            "source": _pack(plan.source),
            "positive": _pack(plan.positive),
            "negative": _pack(plan.negative),
        }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_plan(path: Path, key: BatchKey) -> SamplePlan:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["kind"] == "entity":
        return EntitySamplePlan(
            key, SampledSubgraph(**payload["subgraph"]), payload["target"]
        )
    return RecommendationSamplePlan(
        key,
        *[
            SampledSubgraph(**payload[name])
            for name in ("source", "positive", "negative")
        ],
    )


def materialize(
    directory: Path, seeds: SeedSource, sampler: Sampler, identity: dict[str, Any]
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for seed in seeds.iter_epoch(0):
            minimum_free = int(os.environ.get("RELCONNECTOR_REPLAY_MIN_FREE_BYTES", 2 * 1024**3))
            if shutil.disk_usage(directory).free < minimum_free:
                raise OSError(f"Less than {minimum_free} bytes free while constructing fixed replay")
            plan = sampler.sample(seed)
            name = f"batch-{seed.key.batch:06d}.pt"
            digest = _write_plan(directory / name, plan)
            # Seed artifacts preserve labels, time and chosen positives exactly.
            payload = _pack(seed)
            payload.pop("key")
            payload["kind"] = (
                "entity" if isinstance(seed, EntitySeedBatch) else "recommendation"
            )
            seed_name = f"seed-{seed.key.batch:06d}.pt"
            torch.save(payload, directory / seed_name)
            seed_digest = hashlib.sha256(
                (directory / seed_name).read_bytes()
            ).hexdigest()
            records.append(
                {
                    "batch": seed.key.batch,
                    "plan": name,
                    "sha256": digest,
                    "seed": seed_name,
                    "seed_sha256": seed_digest,
                }
            )
            if len(records) % 10 == 0:
                print(f"[replay] materialized {len(records)} batches", flush=True)
        manifest = {"version": 1, "identity": identity, "batches": records}
        manifest["fingerprint"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(directory / "manifest.json")
        return manifest
    except BaseException:
        (directory / "INCOMPLETE").write_text(
            "Replay construction failed; never use this artifact.\n"
        )
        raise


def validate(directory: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if (directory / "INCOMPLETE").exists():
        raise ValueError("Incomplete replay artifact")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["identity"] != identity:
        raise ValueError("Replay identity differs: database/task/seed/batch/fanout")
    signed = dict(manifest)
    fingerprint = signed.pop("fingerprint")
    if (
        hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
        != fingerprint
    ):
        raise ValueError("Replay manifest checksum mismatch")
    for item in manifest["batches"]:
        for name, digest in [("plan", "sha256"), ("seed", "seed_sha256")]:
            if (
                hashlib.sha256((directory / item[name]).read_bytes()).hexdigest()
                != item[digest]
            ):
                raise ValueError(f"Replay artifact checksum mismatch: {item[name]}")
    return manifest


class ReplaySeeds:
    def __init__(self, directory: Path, records: list[dict[str, Any]]) -> None:
        self.directory, self.records = directory, records

    def iter_epoch(self, epoch: int):
        for item in self.records:
            payload = torch.load(
                self.directory / item["seed"], map_location="cpu", weights_only=True
            )
            kind = payload.pop("kind")
            cls = EntitySeedBatch if kind == "entity" else RecommendationSeedBatch
            yield cls(key=BatchKey(epoch, item["batch"]), **payload)


class ReplaySampler:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def sample(self, seeds: SeedBatch) -> SamplePlan:
        return _read_plan(self.directory / f"batch-{seeds.key.batch:06d}.pt", seeds.key)
