"""算子性能评测：CPU 朴素 / CPU Aho-Corasick / NPU(torch/numpy) 三种后端对比。

指标：
- 吞吐量（文档/秒、字符/秒）、平均延迟（ms/文档）；
- 资源/能效代理指标：字符比较次数（工作量）、峰值张量内存、
  归一化能效（吞吐/峰值内存），用于 CPU 与 NPU 算子的对比分析。
输出结构化 JSON + Markdown 表格，供报告与 README 引用。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .operators.entity_extract import EntityExtractOperator
from .operators.npu.ner_npu import EntityExtractNPUOperator, _DEVICE


@dataclass
class BenchResult:
    backend: str
    num_docs: int
    total_chars: int
    elapsed_ms: float
    throughput_docs: float
    throughput_chars: float
    latency_ms_per_doc: float
    num_entities: int
    compute_kind: str = ""          # 算法类型：sublinear / dense-parallel
    work_comparisons: int = 0       # 字符比较次数（工作量代理）
    peak_tensor_mb: float = 0.0     # 峰值中间张量内存（MB，仅张量后端）
    comparisons_per_ms: float = 0.0 # 单位时间处理的比较次数（并行度代理）


def _bench(name: str, run_fn, docs: list[str], repeat: int,
           compute_kind: str = "", work_comparisons: int = 0,
           peak_tensor_mb: float = 0.0) -> BenchResult:
    run_fn(docs)  # 预热
    best = None
    total_ent = 0
    for _ in range(repeat):
        start = time.perf_counter()
        out = run_fn(docs)
        elapsed = (time.perf_counter() - start) * 1000
        total_ent = sum(len(e) for e in out)
        if best is None or elapsed < best:
            best = elapsed
    total_chars = sum(len(d) for d in docs)
    return BenchResult(
        backend=name,
        num_docs=len(docs),
        total_chars=total_chars,
        elapsed_ms=round(best, 3),
        throughput_docs=round(len(docs) / (best / 1000), 1),
        throughput_chars=round(total_chars / (best / 1000), 1),
        latency_ms_per_doc=round(best / len(docs), 4),
        num_entities=total_ent,
        compute_kind=compute_kind,
        work_comparisons=work_comparisons,
        peak_tensor_mb=round(peak_tensor_mb, 2),
        comparisons_per_ms=round(work_comparisons / best, 1) if work_comparisons else 0.0,
    )


def run_benchmark(corpus_path: str | Path, repeat: int = 5, scale: int = 20) -> dict[str, Any]:
    docs = [json.loads(ln)["text"]
            for ln in Path(corpus_path).read_text(encoding="utf-8").strip().splitlines()]
    docs = (docs * scale)  # 放大语料规模，凸显算法差异
    cpu = EntityExtractOperator()
    npu = EntityExtractNPUOperator()

    # 稠密张量后端的工作量与峰值内存：ok ∈ bool[B, N, L]，沿模式长度 K 归约。
    B = len(docs)
    L = max((len(d) for d in docs), default=1)
    N = len(npu.terms)
    K = npu.max_term_len
    dense_comparisons = B * N * L * K
    # bool 张量每元素 1 字节；算子沿批维分块，峰值受单块预算约束
    chunk = max(1, EntityExtractNPUOperator._CHUNK_ELEMS // (N * L))
    peak_tensor_mb = (min(B, chunk) * N * L) / (1024 * 1024)

    results = [
        _bench("cpu-naive", lambda d: cpu.process(d, mode="naive"), docs, repeat,
               compute_kind="sublinear-scan"),
        _bench("cpu-aho-corasick", lambda d: cpu.process(d, mode="ac"), docs, repeat,
               compute_kind="sublinear-automaton"),
        _bench(f"npu-tensor[{_DEVICE}]", lambda d: npu.process(d), docs, repeat,
               compute_kind="dense-parallel", work_comparisons=dense_comparisons,
               peak_tensor_mb=peak_tensor_mb),
    ]
    baseline = results[0].elapsed_ms
    payload = {
        "backend_device": _DEVICE,
        "num_docs": len(docs),
        "repeat": repeat,
        "tensor_shape": {"B": B, "N": N, "L": L, "K": K},
        "results": [asdict(r) for r in results],
        "speedup_vs_naive": {r.backend: round(baseline / r.elapsed_ms, 2) for r in results},
    }
    return payload


def to_markdown(payload: dict[str, Any]) -> str:
    ts = payload.get("tensor_shape", {})
    lines = [
        f"# 算子性能评测（NER，共 {payload['num_docs']} 文档 × {payload['repeat']} 次取最优）",
        "",
        f"张量后端设备：`{payload['backend_device']}`（`npu`=昇腾，`cpu`=torch CPU，`numpy`=NumPy 模拟）",
        "",
        "## 吞吐 / 延迟",
        "",
        "| 后端 | 耗时(ms) | 吞吐(文档/秒) | 吞吐(字符/秒) | 延迟(ms/文档) | 实体数 | 相对朴素加速比 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    speed = payload["speedup_vs_naive"]
    for r in payload["results"]:
        lines.append(
            f"| {r['backend']} | {r['elapsed_ms']} | {r['throughput_docs']} | "
            f"{r['throughput_chars']} | {r['latency_ms_per_doc']} | {r['num_entities']} | "
            f"{speed[r['backend']]}× |"
        )
    lines += [
        "",
        "## 工作量 / 资源（CPU 亚线性算法 vs NPU 稠密并行算子）",
        "",
        f"稠密张量形状 `ok ∈ bool[B={ts.get('B')}, N={ts.get('N')}, L={ts.get('L')}]`，"
        f"沿模式长度 `K={ts.get('K')}` 归约。",
        "",
        "| 后端 | 计算类型 | 字符比较次数 | 比较/ms（并行度代理） | 峰值张量内存(MB) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in payload["results"]:
        comp = f"{r['work_comparisons']:,}" if r["work_comparisons"] else "—(亚线性)"
        cpms = f"{r['comparisons_per_ms']:,.0f}" if r["comparisons_per_ms"] else "—"
        mem = r["peak_tensor_mb"] or "—"
        lines.append(f"| {r['backend']} | {r['compute_kind']} | {comp} | {cpms} | {mem} |")
    return "\n".join(lines) + "\n"
