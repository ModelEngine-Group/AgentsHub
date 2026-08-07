from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medgraph_agent.core.models import BackendResult
from medgraph_agent.operators.medical_extraction import build_graph
from medgraph_agent.operators.processing import load_records


@dataclass(frozen=True)
class ExecutionBackend:
    name: str

    def is_available(self) -> bool:
        return True

    def run(self, source: str | Path, repeat: int = 5) -> BackendResult:
        raise NotImplementedError


class CPUBackend(ExecutionBackend):
    def __init__(self) -> None:
        super().__init__("cpu")

    def run(self, source: str | Path, repeat: int = 5) -> BackendResult:
        records = load_records(source)
        payload = records * repeat
        start = time.perf_counter()
        graph = build_graph(payload)
        elapsed = time.perf_counter() - start
        item_count = max(1, len(payload))
        return BackendResult(
            backend=self.name,
            available=True,
            latency_ms=round(elapsed * 1000 / item_count, 4),
            throughput_items_per_second=round(item_count / elapsed, 2) if elapsed else None,
            notes=f"Actual CPU benchmark: {item_count} records, {len(graph.relations)} relations.",
        )


class CudaBackend(ExecutionBackend):
    def __init__(self) -> None:
        super().__init__("cuda")

    def is_available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def run(self, source: str | Path, repeat: int = 5) -> BackendResult:
        available = self.is_available()
        notes = "NVIDIA GPU detected; CUDA-specific extraction kernel is not enabled in this offline submission."
        if not available:
            notes = "nvidia-smi not found; CUDA backend unavailable."
        return BackendResult(self.name, available, None, None, notes)


class AscendNPUBackend(ExecutionBackend):
    def __init__(self) -> None:
        super().__init__("ascend_npu")

    def is_available(self) -> bool:
        return shutil.which("npu-smi") is not None

    def run(self, source: str | Path, repeat: int = 5) -> BackendResult:
        if not self.is_available():
            return BackendResult(
                backend=self.name,
                available=False,
                latency_ms=None,
                throughput_items_per_second=None,
                notes="npu-smi not found. NPU adapter is present but no Ascend/NPU hardware was available for verified numbers.",
            )
        try:
            output = subprocess.check_output(["npu-smi", "info"], text=True, timeout=5)
            notes = "Ascend NPU detected. Hardware information captured; run the same benchmark command on the NPU host for verified throughput."
            if output.strip():
                notes += " npu-smi responded successfully."
        except Exception as exc:
            notes = f"npu-smi exists but failed: {type(exc).__name__}: {exc}"
        return BackendResult(self.name, True, None, None, notes)


def run_benchmarks(source: str | Path, repeat: int = 20) -> dict[str, Any]:
    backends = [CPUBackend(), CudaBackend(), AscendNPUBackend()]
    results = [backend.run(source, repeat=repeat) for backend in backends]
    return {
        "source": str(source),
        "repeat": repeat,
        "results": [result.__dict__ for result in results],
        "integrity_note": "Only CPU values are performance measurements unless a hardware backend reports latency/throughput.",
    }
