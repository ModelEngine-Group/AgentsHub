"""算子性能评测的资源/能效指标测试。"""

from __future__ import annotations

import json
from pathlib import Path

from mgi.benchmark import run_benchmark, to_markdown

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus" / "medical_notes.jsonl"


def test_benchmark_reports_all_backends() -> None:
    payload = run_benchmark(CORPUS, repeat=1, scale=2)
    backends = [r["backend"] for r in payload["results"]]
    assert any(b == "cpu-naive" for b in backends)
    assert any(b == "cpu-aho-corasick" for b in backends)
    assert any(b.startswith("npu-tensor") for b in backends)
    assert payload["speedup_vs_naive"]["cpu-naive"] == 1.0


def test_benchmark_resource_metrics() -> None:
    payload = run_benchmark(CORPUS, repeat=1, scale=2)
    ts = payload["tensor_shape"]
    npu = next(r for r in payload["results"] if r["backend"].startswith("npu-tensor"))
    cpu = next(r for r in payload["results"] if r["backend"] == "cpu-aho-corasick")
    # 稠密算子的工作量 = B*N*L*K，且报告峰值张量内存与并行度代理
    assert npu["work_comparisons"] == ts["B"] * ts["N"] * ts["L"] * ts["K"]
    assert npu["peak_tensor_mb"] > 0
    assert npu["comparisons_per_ms"] > 0
    assert npu["compute_kind"] == "dense-parallel"
    # CPU 亚线性算法不计稠密比较次数
    assert cpu["work_comparisons"] == 0


def test_benchmark_markdown_has_resource_section() -> None:
    payload = run_benchmark(CORPUS, repeat=1, scale=2)
    md = to_markdown(payload)
    assert "吞吐 / 延迟" in md
    assert "工作量 / 资源" in md
    assert "峰值张量内存(MB)" in md
    # 后端计数一致（每个后端出现在吞吐表与资源表各一次）
    assert md.count("dense-parallel") == 1


def test_benchmark_entities_consistent_across_cpu_backends() -> None:
    payload = run_benchmark(CORPUS, repeat=1, scale=2)
    ents = {r["backend"]: r["num_entities"] for r in payload["results"]}
    # 朴素与 AC 两种 CPU 后端实体数应完全一致
    assert ents["cpu-naive"] == ents["cpu-aho-corasick"]
    assert ents["cpu-naive"] > 0


def test_corpus_readable() -> None:
    lines = CORPUS.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 40
    assert all("text" in json.loads(ln) for ln in lines)
