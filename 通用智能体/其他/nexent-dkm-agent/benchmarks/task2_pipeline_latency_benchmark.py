"""End-to-end task-2 pipeline latency benchmark (rule vs tensor CPU backends).

Measures full extract -> relation -> validate -> build latency on real sample
text, not isolated kernel micro-benchmarks. NPU numbers remain in the dedicated
tensor benchmark scripts when Ascend hardware is available.

Usage:
    python benchmarks/task2_pipeline_latency_benchmark.py
    python benchmarks/task2_pipeline_latency_benchmark.py --report benchmarks/reports/task2_pipeline_latency.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.kg_ops import (
    build_medical_graph,
    extract_medical_entities,
    extract_relations,
    extract_relations_tensorized,
    validate_triples,
)

DEFAULT_INPUT = ROOT / "data" / "samples" / "task2_medical_notes.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-2 pipeline latency.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_rule_pipeline(text: str) -> dict:
    extraction = extract_medical_entities(text)
    records = extraction.get("records", [])
    triples = extract_relations(records)
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], records)
    return {
        "record_count": extraction.get("record_count", len(records)),
        "triple_count": validation.get("valid_count", 0),
        "node_count": graph["statistics"].get("node_count", 0),
        "edge_count": graph["statistics"].get("edge_count", 0),
    }


def _run_tensor_pipeline(text: str, backend: str) -> dict:
    extraction = extract_medical_entities(text)
    records = extraction.get("records", [])
    tensor_result = extract_relations_tensorized(records, backend=backend)
    triples = tensor_result.get("triples", [])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], records)
    return {
        "record_count": extraction.get("record_count", len(records)),
        "triple_count": validation.get("valid_count", 0),
        "node_count": graph["statistics"].get("node_count", 0),
        "edge_count": graph["statistics"].get("edge_count", 0),
        "relation_backend": tensor_result.get("backend", backend),
    }


def _benchmark(runner, text: str, iterations: int, warmup: int) -> dict:
    for _ in range(warmup):
        last = runner(text)
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        last = runner(text)
        durations.append(time.perf_counter() - started)
    total = sum(durations)
    return {
        "iterations": iterations,
        "warmup": warmup,
        "latency_ms_avg": round(mean(durations) * 1000, 4),
        "latency_ms_min": round(min(durations) * 1000, 4),
        "latency_ms_max": round(max(durations) * 1000, 4),
        "throughput_records_per_sec": round(
            (last["record_count"] * iterations) / total, 4
        ) if total else 0.0,
        "artifacts": last,
    }


def main() -> int:
    args = parse_args()
    text = _read_text(Path(args.input))
    rule = _benchmark(_run_rule_pipeline, text, args.iterations, args.warmup)
    tensor_cpu = _benchmark(
        lambda payload: _run_tensor_pipeline(payload, "cpu"),
        text,
        args.iterations,
        args.warmup,
    )

    speedup = None
    if tensor_cpu["latency_ms_avg"] > 0:
        speedup = round(rule["latency_ms_avg"] / tensor_cpu["latency_ms_avg"], 4)

    report = {
        "task": "task2_kg_pipeline_latency",
        "input": {
            "path": str(Path(args.input)),
            "char_count": len(text),
        },
        "measurement": "end_to_end_pipeline",
        "rule_backend": rule,
        "tensor_cpu_backend": tensor_cpu,
        "speedup_rule_over_tensor_cpu": speedup,
        "notes": [
            "End-to-end latency includes entity extraction, relation scoring, validation, and graph build.",
            "Tensor CPU path uses extract_relations_tensorized(backend='cpu').",
            "NPU end-to-end numbers require Ascend hardware; see task2_relation_tensor_*.json.",
        ],
        "passed": (
            rule["artifacts"]["triple_count"] > 0
            and tensor_cpu["artifacts"]["triple_count"] > 0
        ),
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
