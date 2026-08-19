"""Tests for task-2 pipeline latency benchmark."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_task2_pipeline_latency_benchmark_runs(monkeypatch):
    from benchmarks.task2_pipeline_latency_benchmark import main

    monkeypatch.setattr(
        "sys.argv",
        ["task2_pipeline_latency_benchmark.py", "--iterations", "1", "--warmup", "0"],
    )
    assert main() == 0
