"""Background sampler for Ascend NPU utilization and power via ``npu-smi``.

Used by the Task-2 relation tensor benchmark to attach an ``npu_utilization``
energy/efficiency block to its report. Sampling runs in a daemon thread so the
benchmark timing itself is unaffected; ``npu-smi`` is queried at a fixed
interval and the per-field min / avg / max are aggregated when stopped.

All parsing is defensive: if ``npu-smi`` is missing or returns an unexpected
format, the sampler degrades to ``available: False`` rather than raising.
"""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import threading
from typing import Any

# Fields parsed from ``npu-smi info -t usages`` (label -> result key).
_USAGE_FIELDS = {
    "Aicore Usage Rate(%)": "aicore_pct",
    "Aivector Usage Rate(%)": "aivector_pct",
    "Aicube Usage Rate(%)": "aicube_pct",
    "HBM Usage Rate(%)": "hbm_pct",
    "HBM Bandwidth Usage Rate(%)": "hbm_bandwidth_pct",
    "NPU Utilization(%)": "npu_utilization_pct",
}
_POWER_LABEL = "NPU Real-time Power(W)"


def npu_smi_available() -> bool:
    return shutil.which("npu-smi") is not None


def detect_npu_id() -> int | None:
    """Return the first NPU id reported by ``npu-smi info -l`` (else None)."""

    out = _run(["npu-smi", "info", "-l"])
    if out is None:
        # Fall back to the device table in plain ``npu-smi info``.
        out = _run(["npu-smi", "info"])
    if out is None:
        return None
    match = re.search(r"NPU ID\s*:\s*(\d+)", out)
    if match:
        return int(match.group(1))
    # Plain table: first data row "| <id>   <name> ... |".
    for line in out.splitlines():
        m = re.match(r"\|\s*(\d+)\s+\S+", line)
        if m:
            return int(m.group(1))
    return None


class NpuUtilizationSampler:
    """Periodically sample NPU AICore%/utilization%/power in a daemon thread."""

    def __init__(
        self,
        npu_id: int | None = None,
        chip_id: int = 0,
        interval_s: float = 0.5,
    ) -> None:
        self.interval_s = max(0.05, interval_s)
        self.chip_id = chip_id
        self.npu_id = npu_id if npu_id is not None else detect_npu_id()
        self._samples: list[dict[str, float]] = []
        self._power: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = npu_smi_available() and self.npu_id is not None

    @property
    def available(self) -> bool:
        return self._available

    def __enter__(self) -> "NpuUtilizationSampler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if not self._available or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            usage = self._sample_usage()
            if usage:
                self._samples.append(usage)
            power = self._sample_power()
            if power is not None:
                self._power.append(power)
            self._stop.wait(self.interval_s)

    def _sample_usage(self) -> dict[str, float] | None:
        out = _run(
            ["npu-smi", "info", "-t", "usages", "-i", str(self.npu_id), "-c", str(self.chip_id)]
        )
        if out is None:
            return None
        parsed: dict[str, float] = {}
        for label, key in _USAGE_FIELDS.items():
            m = re.search(rf"{re.escape(label)}\s*:\s*([-\d.]+)", out)
            if m:
                try:
                    parsed[key] = float(m.group(1))
                except ValueError:
                    continue
        return parsed or None

    def _sample_power(self) -> float | None:
        out = _run(["npu-smi", "info", "-t", "power", "-i", str(self.npu_id)])
        if out is None:
            return None
        m = re.search(rf"{re.escape(_POWER_LABEL)}\s*:\s*([-\d.]+)", out)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def result(self) -> dict[str, Any]:
        """Aggregate collected samples into a JSON-serializable report block."""

        if not self._available:
            return {
                "available": False,
                "reason": "npu-smi not found" if not npu_smi_available() else "NPU id not detected",
            }
        if not self._samples and not self._power:
            return {
                "available": False,
                "reason": "no samples collected",
                "npu_id": self.npu_id,
                "chip_id": self.chip_id,
            }

        report: dict[str, Any] = {
            "available": True,
            "npu_id": self.npu_id,
            "chip_id": self.chip_id,
            "sample_count": len(self._samples),
            "sample_interval_s": self.interval_s,
            "tool": "npu-smi",
        }
        for key in {k for s in self._samples for k in s}:
            values = [s[key] for s in self._samples if key in s]
            if values:
                report[key] = _stats(values)
        if self._power:
            report["power_w"] = _stats(self._power)
        return report


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "avg": round(statistics.mean(values), 3),
        "max": round(max(values), 3),
    }


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout
