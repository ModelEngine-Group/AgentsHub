"""Task 2 KG benchmark helpers with optional NPU runtime detection."""

from __future__ import annotations

import importlib
import platform
import time
from statistics import mean
from typing import Any

from src.operators.kg_ops import (
    build_medical_graph,
    extract_medical_entities,
    extract_relations,
    validate_triples,
)


def benchmark_task2_kg_ops(
    text: str,
    iterations: int = 5,
    npu_probe: bool = True,
    npu_probe_iterations: int = 5,
    npu_probe_size: int = 64,
) -> dict[str, Any]:
    """Benchmark the deterministic task-2 KG operator chain on CPU."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    warmup = _run_kg_ops(text)
    record_count = warmup["extraction"].get("record_count", 0)
    char_count = len(text)
    durations = []
    last_run = warmup
    for _ in range(iterations):
        started = time.perf_counter()
        last_run = _run_kg_ops(text)
        durations.append(time.perf_counter() - started)

    total_duration = sum(durations)
    latency_ms_avg = mean(durations) * 1000
    return {
        "task": "task2_kg_agent",
        "input": {
            "record_count": record_count,
            "char_count": char_count,
            "iterations": iterations,
            "npu_probe": {
                "enabled": npu_probe,
                "iterations": npu_probe_iterations,
                "matrix_size": npu_probe_size,
            },
        },
        "cpu": {
            "status": "completed",
            "backend": "python",
            "iterations": iterations,
            "latency_ms_avg": round(latency_ms_avg, 4),
            "latency_ms_min": round(min(durations) * 1000, 4),
            "latency_ms_max": round(max(durations) * 1000, 4),
            "throughput_records_per_sec": _throughput(record_count * iterations, total_duration),
            "throughput_chars_per_sec": _throughput(char_count * iterations, total_duration),
            "entity_count": sum(last_run["extraction"].get("entity_counts", {}).values()),
            "triple_count": last_run["validation"].get("valid_count", 0),
            "node_count": last_run["graph"]["statistics"].get("node_count", 0),
            "edge_count": last_run["graph"]["statistics"].get("edge_count", 0),
        },
        "npu": detect_npu_runtime(
            probe=npu_probe,
            probe_iterations=npu_probe_iterations,
            probe_size=npu_probe_size,
        ),
        "notes": [
            "CPU numbers are measured locally with deterministic rule-based operators.",
            "NPU runtime probes are reported separately from task-operator CPU timings.",
            "Task-level NPU speedup and energy metrics require operator-specific NPU kernels and profiler support.",
        ],
    }


def detect_npu_runtime(
    probe: bool = True,
    probe_iterations: int = 5,
    probe_size: int = 64,
) -> dict[str, Any]:
    """Detect common Ascend NPU runtime modules without requiring them."""

    errors: list[str] = []

    try:
        importlib.import_module("torch_npu")
    except Exception as exc:
        errors.append(f"torch_npu: {str(exc) or type(exc).__name__}")
    else:
        result = {
            "status": "available",
            "backend": "Ascend PyTorch NPU",
            "module": "torch_npu",
            "platform": platform.platform(),
            "energy_metrics": "requires device-specific profiler",
        }
        if probe:
            result["runtime_probe"] = _probe_torch_npu(
                iterations=probe_iterations,
                matrix_size=probe_size,
            )
        return result

    try:
        importlib.import_module("acl")
    except Exception as exc:
        errors.append(f"acl: {str(exc) or type(exc).__name__}")
    else:
        return {
            "status": "available",
            "backend": "AscendCL",
            "module": "acl",
            "platform": platform.platform(),
            "energy_metrics": "requires device-specific profiler",
        }

    return {
        "status": "unavailable",
        "backend": "none",
        "platform": platform.platform(),
        "reason": f"No supported NPU runtime detected ({'; '.join(errors) if errors else 'not installed'}).",
        "energy_metrics": "not measured because no supported NPU runtime is available",
    }


def _probe_torch_npu(iterations: int, matrix_size: int) -> dict[str, Any]:
    """Run a tiny torch_npu tensor operation to prove the runtime executes work."""

    if iterations < 1:
        raise ValueError("probe_iterations must be >= 1")
    if matrix_size < 1:
        raise ValueError("probe_size must be >= 1")

    try:
        torch = importlib.import_module("torch")
        npu = getattr(torch, "npu", None)
        if npu is None or not callable(getattr(npu, "is_available", None)):
            return {
                "status": "unavailable",
                "reason": "torch.npu is not available in this Python runtime",
            }
        if not npu.is_available():
            return {
                "status": "unavailable",
                "reason": "torch.npu.is_available() returned False",
            }

        x = torch.randn(matrix_size, matrix_size).npu()
        y = torch.randn(matrix_size, matrix_size).npu()
        _synchronize_npu(npu)
        _ = x @ y
        _synchronize_npu(npu)

        durations = []
        last_result = None
        for _ in range(iterations):
            started = time.perf_counter()
            last_result = x @ y
            _synchronize_npu(npu)
            durations.append(time.perf_counter() - started)

        total_duration = sum(durations)
        return {
            "status": "completed",
            "operation": "torch.randn matrix multiplication on npu",
            "iterations": iterations,
            "matrix_shape": [matrix_size, matrix_size],
            "latency_ms_avg": round(mean(durations) * 1000, 4),
            "latency_ms_min": round(min(durations) * 1000, 4),
            "latency_ms_max": round(max(durations) * 1000, 4),
            "throughput_ops_per_sec": _throughput(iterations, total_duration),
            "device_count": _safe_device_count(npu),
            "device": str(getattr(last_result, "device", "npu")),
            "note": "This probe validates the Ascend PyTorch runtime; task operators remain measured on the CPU path.",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc) or type(exc).__name__,
        }


def _synchronize_npu(npu: Any) -> None:
    synchronize = getattr(npu, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _safe_device_count(npu: Any) -> int | None:
    device_count = getattr(npu, "device_count", None)
    if not callable(device_count):
        return None
    try:
        return int(device_count())
    except Exception:
        return None


def _run_kg_ops(text: str) -> dict[str, Any]:
    extraction = extract_medical_entities(text)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])
    return {
        "extraction": extraction,
        "validation": validation,
        "graph": graph,
    }


def _throughput(items: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return round(items / seconds, 4)
