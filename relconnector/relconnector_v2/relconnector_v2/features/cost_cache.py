"""Cost-controlled local text reuse and immutable, content-addressed disk blocks.

No database-name rules or centralized RPC. The legacy cost policy spends a small
fraction of observed encoding time; the ROI policy admits blocks only when
measured encoding cost can repay predicted reads and construction in the epochs
that remain. Immutable blocks can be read by other workers and later runs without
a shared SQLite writer.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import sys
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

_HEADER = struct.Struct("<8sIId")
_MAGIC = b"RCTEXT01"


@dataclass
class ReuseGate:
    calls: int = 0
    saved_s: float = 0.0
    overhead_s: float = 0.0
    cooldown: int = 0
    bypasses: int = 0
    transitions: int = 0
    model_per_row_s: float = 0.0

    def probe(self) -> bool:
        if self.cooldown:
            self.cooldown -= 1
            self.bypasses += 1
            return False
        return True

    def observe(self, saved_s: float, overhead_s: float) -> None:
        self.calls += 1
        self.saved_s += saved_s
        self.overhead_s += overhead_s
        if self.calls >= 16:
            if self.saved_s <= 1.25 * self.overhead_s:
                self.cooldown = 64
                self.transitions += 1
            self.calls = 0
            self.saved_s = self.overhead_s = 0.0


class CostAwareTextCache:
    def __init__(
        self,
        *,
        max_bytes: int,
        dimension: int,
        disk_dir: Path | None,
        disk_max_bytes: int,
        remaining_epochs: int,
        disk_policy: Literal["budget", "roi"] = "budget",
    ) -> None:
        if disk_policy not in {"budget", "roi"}:
            raise ValueError("disk policy must be budget or roi")
        self.max_bytes = max_bytes
        self.dimension = dimension
        self.disk_dir = disk_dir
        self.existing_disk_cache = (
            disk_dir is not None
            and disk_dir.is_dir()
            and next(disk_dir.glob("*/*.bin"), None) is not None
        )
        self.disk_max_bytes = disk_max_bytes
        self.remaining_epochs = remaining_epochs
        self.disk_policy = disk_policy
        self.rows: OrderedDict[str, tuple[torch.Tensor, float, int]] = OrderedDict()
        self.gates: dict[str, ReuseGate] = defaultdict(ReuseGate)
        self.disk_gates: dict[str, ReuseGate] = defaultdict(ReuseGate)
        self.bytes = 0
        self.peak_bytes = 0
        self.epoch: int | None = None
        self.stats: dict[str, float] = defaultdict(float)
        self.epochs: dict[int, dict[str, float]] = {}
        self.disk_credit_s = 0.0
        self.write_bytes_per_s = 64 * 1024**2
        self.read_bytes_per_s = 128 * 1024**2
        # A small prior prevents the first tiny file (where mkdir/scheduling
        # dominates) from becoming the effective rate for all later blocks.
        rate_prior_bytes = 4 * 1024**2
        self._read_rate_bytes = rate_prior_bytes
        self._read_rate_s = rate_prior_bytes / self.read_bytes_per_s
        self._write_rate_bytes = rate_prior_bytes
        self._write_rate_s = rate_prior_bytes / self.write_bytes_per_s
        self.disk_read_bytes = 0
        self.disk_read_s = 0.0
        self.disk_write_s = 0.0
        self.disk_written_bytes = 0
        self.disk_cooldown = 0

    def _add(self, name: str, value: float) -> None:
        self.stats[name] += value
        if self.epoch is not None:
            row = self.epochs.setdefault(self.epoch, {})
            row[name] = row.get(name, 0.0) + value

    def resolve(
        self,
        values: list[str],
        *,
        column: str,
        epoch: int | None,
        encode: Callable[[list[str]], tuple[torch.Tensor, float]],
    ) -> tuple[torch.Tensor, int]:
        if not values:
            return torch.empty((0, self.dimension)), 0
        if epoch != self.epoch:
            self.epoch = epoch
            # Workload may change at epoch boundaries: allow fresh bounded probes.
            for gate in self.gates.values():
                gate.cooldown = 0
            for disk_gate in self.disk_gates.values():
                disk_gate.cooldown = 0
        gate = self.gates[column]
        active = bool(self.max_bytes) and gate.probe()
        lookup_start = time.perf_counter()
        hits: dict[int, torch.Tensor] = {}
        saved_s = 0.0
        if active:
            for i, value in enumerate(values):
                entry = self.rows.get(value)
                if entry is not None:
                    self.rows.move_to_end(value)
                    hits[i] = entry[0]
                    saved_s += entry[1]
        lookup_s = time.perf_counter() - lookup_start
        self._add("ram_lookup_s", lookup_s)
        self._add("ram_hits", len(hits))
        if len(hits) == len(values):
            start = time.perf_counter()
            result = torch.stack([hits[i] for i in range(len(values))])
            overhead = lookup_s + time.perf_counter() - start
            self._add("ram_maintenance_s", overhead - lookup_s)
            self._add("ram_estimated_saved_s", saved_s)
            gate.observe(saved_s, overhead)
            return result, len(values)

        # Key the ORIGINAL ordered text block, not cache-dependent RAM misses.
        # Repeated exact workloads then have stable keys across worker routing.
        key: str | None = None
        disk_start = time.perf_counter()
        payload_bytes = len(values) * self.dimension * 4 + _HEADER.size
        predicted_read_s = self._predicted_read_s(payload_bytes)
        estimated_encode_s = gate.model_per_row_s * len(values)
        disk_allowed = self.disk_dir is not None and (
            (
                self.disk_policy == "roi"
                and epoch != 0
                and estimated_encode_s >= 4 * predicted_read_s
            )
            or (
                self.disk_policy == "budget"
                and (
                    estimated_encode_s >= 0.005
                    or (not gate.model_per_row_s and self.existing_disk_cache)
                )
            )
            or (self.existing_disk_cache and not gate.model_per_row_s)
        )
        if self.disk_cooldown:
            self.disk_cooldown -= 1
            disk_allowed = False
        disk_gate = self.disk_gates[column]
        disk_allowed = disk_allowed and disk_gate.probe()
        disk_hit = None
        if disk_allowed:
            key = self._key(values)
            disk_hit = self._read(key, len(values))
        disk_read_s = time.perf_counter() - disk_start
        self._add("disk_lookup_s", disk_read_s)
        if disk_hit is not None:
            result, original_cost = disk_hit
            # Use cumulative throughput instead of a minimum or a per-call
            # sample. Thousands of small immutable files inevitably produce
            # scheduler outliers; allowing one such call to poison admission
            # for the rest of the run rejected profitable RateBeer blocks.
            self.disk_read_bytes += payload_bytes
            self.disk_read_s += disk_read_s
            self._read_rate_bytes += payload_bytes
            self._read_rate_s += disk_read_s
            self.read_bytes_per_s = self._read_rate_bytes / max(
                self._read_rate_s, 1e-6
            )
            gate.model_per_row_s = original_cost / len(values)
            self._add("disk_estimated_saved_s", original_cost)
            disk_gate.observe(
                original_cost * (1 - len(hits) / len(values)), disk_read_s
            )
            if active:
                # RAM did not avoid the whole-block disk read on this call.
                gate.observe(0, lookup_s)
            self._add("disk_hit_blocks", 1)
            self._add("disk_hit_rows", len(values))
            if disk_read_s > original_cost:
                self.disk_cooldown = 64
                self._add("disk_read_backoffs", 1)
            # No row copying on disk hits: avoid turning fast block reads into
            # hundreds of Python objects. Hot RAM is populated on actual encoding.
            return result, len(values)
        if disk_allowed:
            disk_gate.observe(0, disk_read_s)
            self._add("disk_miss_blocks", 1)

        missing = [i for i in range(len(values)) if i not in hits]
        encoded, model_s = encode([values[i] for i in missing])
        self._add("model_s", model_s)
        self._add("model_rows", len(missing))
        unit_cost = model_s / max(len(missing), 1)
        gate.model_per_row_s = (
            unit_cost
            if not gate.model_per_row_s
            else 0.8 * gate.model_per_row_s + 0.2 * unit_cost
        )
        start = time.perf_counter()
        if hits:
            result = torch.empty((len(values), self.dimension), dtype=encoded.dtype)
            result[torch.tensor(missing)] = encoded
            for i, vector in hits.items():
                result[i] = vector
        else:
            result = encoded
        if active:
            for i, vector in zip(missing, encoded, strict=True):
                self._put(values[i], vector, unit_cost)
        ram_overhead = lookup_s + time.perf_counter() - start
        self._add("ram_maintenance_s", ram_overhead - lookup_s)
        self._add("ram_estimated_saved_s", saved_s)
        if active:
            gate.observe(saved_s, ram_overhead)

        # The original policy spends at most 3% of observed encoding time (plus
        # one possible overrun). ROI admission instead uses remaining epochs and
        # measured encode/read/write rates, while retaining a hard byte budget.
        self.disk_credit_s += 0.03 * model_s
        cost = model_s + saved_s
        payload_bytes = result.numel() * result.element_size() + _HEADER.size
        predicted_write_s = max(0.001, payload_bytes / self.write_bytes_per_s)
        common_publish = (
            self.disk_dir is not None
            and self.remaining_epochs > (epoch or 0) + 1
            and self.disk_written_bytes + payload_bytes <= self.disk_max_bytes
        )
        if self.disk_policy == "roi":
            future_uses = max(self.remaining_epochs - (epoch or 0) - 1, 0)
            predicted_read_s = self._predicted_read_s(payload_bytes)
            predicted_savings_s = future_uses * max(cost - predicted_read_s, 0.0)
            can_publish = (
                common_publish
                and cost >= 4 * predicted_read_s
                and predicted_savings_s >= 2 * predicted_write_s
            )
            self._add("disk_roi_candidates", 1)
            self._add("disk_predicted_read_s", predicted_read_s)
            self._add("disk_predicted_write_s", predicted_write_s)
            self._add("disk_predicted_savings_s", predicted_savings_s)
            if not can_publish:
                self._add("disk_roi_rejections", 1)
        else:
            can_publish = (
                common_publish
                and cost >= max(0.01, 8 * predicted_write_s)
                and predicted_write_s <= self.disk_credit_s
            )
        if can_publish:
            start = time.perf_counter()
            published = self._write(key or self._key(values), result, cost)
            elapsed = time.perf_counter() - start
            self.disk_credit_s -= elapsed
            self._add("disk_write_s", elapsed)
            if published:
                self.disk_written_bytes += payload_bytes
                self.disk_write_s += elapsed
                self._write_rate_bytes += payload_bytes
                self._write_rate_s += elapsed
                self.write_bytes_per_s = self._write_rate_bytes / max(
                    self._write_rate_s, 1e-6
                )
                self._add("disk_write_blocks", 1)
                self._add("disk_write_bytes", payload_bytes)
        return result, len(hits)

    def _predicted_read_s(self, payload_bytes: int) -> float:
        return max(0.0001, payload_bytes / self.read_bytes_per_s)

    def _put(self, value: str, vector: torch.Tensor, cost: float) -> None:
        # Clone rows: retaining a tensor view would pin the whole source block.
        size = vector.numel() * vector.element_size() + sys.getsizeof(value) + 256
        if size > self.max_bytes:
            return
        previous = self.rows.pop(value, None)
        if previous is not None:
            self.bytes -= previous[2]
        while self.rows and self.bytes + size > self.max_bytes:
            _, (_, _, charge) = self.rows.popitem(last=False)
            self.bytes -= charge
            self._add("ram_evictions", 1)
        self.rows[value] = vector.clone(), cost, size
        self.bytes += size
        self.peak_bytes = max(self.bytes, self.peak_bytes)

    @staticmethod
    def _key(values: list[str]) -> str:
        digest = hashlib.sha256()
        for value in values:
            data = value.encode("utf-8")
            digest.update(struct.pack("<Q", len(data)))
            digest.update(data)
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        assert self.disk_dir is not None
        return self.disk_dir / key[:2] / f"{key}.bin"

    def _read(self, key: str, rows: int) -> tuple[torch.Tensor, float] | None:
        try:
            with self._path(key).open("rb") as handle:
                header = handle.read(_HEADER.size)
                if len(header) != _HEADER.size:
                    self._add("disk_invalid_blocks", 1)
                    return None
                magic, count, dimension, cost = _HEADER.unpack(header)
                if (magic, count, dimension) != (_MAGIC, rows, self.dimension):
                    self._add("disk_invalid_blocks", 1)
                    return None
                expected = rows * dimension * 4
                payload = handle.read(expected + 1)
                if len(payload) != expected:
                    self._add("disk_invalid_blocks", 1)
                    return None
                array = (
                    np.frombuffer(payload, dtype="<f4").copy().reshape(rows, dimension)
                )
                return torch.from_numpy(array), cost
        except FileNotFoundError:
            return None
        except OSError:
            self._add("disk_io_errors", 1)
            self.disk_cooldown = 64
            return None

    def _write(self, key: str, result: torch.Tensor, cost: float) -> bool:
        path = self._path(key)
        if path.exists():
            return False
        temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(path.parent).free < result.numel() * 4 + 1024**3:
                self._add("disk_space_rejections", 1)
                return False
            with temporary.open("xb") as handle:
                handle.write(_HEADER.pack(_MAGIC, len(result), self.dimension, cost))
                result.detach().cpu().numpy().astype("<f4", copy=False).tofile(handle)
            os.replace(temporary, path)
            return True
        except OSError:
            self._add("disk_io_errors", 1)
            return False
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, object]:
        return {
            "strategy": "roi" if self.disk_policy == "roi" else "cost",
            "ram_bytes": self.bytes,
            "ram_peak_bytes": self.peak_bytes,
            "ram_budget_bytes": self.max_bytes,
            "disk_budget_bytes": self.disk_max_bytes,
            "disk_credit_s": self.disk_credit_s,
            "disk_policy": self.disk_policy,
            "read_bytes_per_s": self.read_bytes_per_s,
            "write_bytes_per_s": self.write_bytes_per_s,
            "counters": dict(self.stats),
            "epochs": dict(self.epochs),
            "columns": {key: vars(value).copy() for key, value in self.gates.items()},
            "disk_columns": {
                key: vars(value).copy() for key, value in self.disk_gates.items()
            },
        }
