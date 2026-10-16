"""Subprocess isolation with timeout and peak RSS measurement."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TypedDict


class ProcessMeasurement(TypedDict):
    returncode: int
    timed_out: bool
    duration_s: float
    peak_rss_mb: float


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    timed_out: bool
    duration_s: float
    peak_rss_mb: float
    stdout: str
    stderr: str

    def measurement(self) -> ProcessMeasurement:
        return {
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_s": self.duration_s,
            "peak_rss_mb": self.peak_rss_mb,
        }


def run_measured_process(
    command: list[str],
    *,
    timeout_s: float | None,
    sample_interval_s: float = 0.1,
) -> ProcessOutcome:
    if sample_interval_s <= 0 or (timeout_s is not None and timeout_s <= 0):
        raise ValueError("timeout and sample interval must be positive")
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout_tail: deque[bytes] = deque(maxlen=2)
    stderr_tail: deque[bytes] = deque(maxlen=2)
    assert process.stdout is not None and process.stderr is not None
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, stdout_tail)),
        threading.Thread(target=_drain, args=(process.stderr, stderr_tail)),
    ]
    for reader in readers:
        reader.start()
    peak_rss_bytes = 0
    timed_out = False
    try:
        while process.poll() is None:
            peak_rss_bytes = max(peak_rss_bytes, _rss_bytes(process.pid))
            if timeout_s is not None and time.perf_counter() - started >= timeout_s:
                timed_out = True
                break
            time.sleep(sample_interval_s)
    finally:
        # Kill the whole worker group, including loader children retaining pipes.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        for reader in readers:
            reader.join()
    return ProcessOutcome(
        returncode=process.returncode,
        timed_out=timed_out,
        duration_s=time.perf_counter() - started,
        peak_rss_mb=peak_rss_bytes / (1024 * 1024),
        stdout=b"".join(stdout_tail).decode(errors="replace")[-8000:],
        stderr=b"".join(stderr_tail).decode(errors="replace")[-8000:],
    )


def _drain(stream: BinaryIO, tail: deque[bytes]) -> None:
    with stream:
        while chunk := stream.read(4096):
            tail.append(chunk)


def _rss_bytes(pid: int) -> int:
    try:
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return 0
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
