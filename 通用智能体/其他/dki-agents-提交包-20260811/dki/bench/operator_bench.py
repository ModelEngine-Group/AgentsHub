"""算子吞吐/延迟基准（CPU 实测）。

方法：
- 清洗组算子（吃原始文本）与 KG 组算子（吃前置链产物）分组测试；
- 每个算子独立构建 LocalPipelineRunner 单算子流水线，输入用同一批样本的深拷贝；
- repeat 次取中位吞吐；延迟取全样本 p50/p95；
- 输出 CSV + Markdown（含环境信息），供 metadata.yml 的 metrics 与文档引用。
"""

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dki import dmcompat
from dki.paths import OUTPUT_DIR, RAW_DATA_DIR
from dki.pipeline import LocalPipelineRunner, make_sample, samples_from_dir

# (算子名, 参数, 输入阶段)  阶段: raw=原始文本 / sectioned=分段JSON / entities=实体JSON / relations=关系JSON
DEFAULT_OPS = [
    ("HtmlTagCleaner", {}, "raw"),
    ("FullWidthCharacterCleaner", {}, "raw"),
    ("ExtraSpaceCleaner", {}, "raw"),
    ("AnonymizedPhoneNumber", {}, "raw"),
    ("MedicalTextCleaner", {"maskPii": True}, "raw"),
    ("MedicalSectionSplitter", {}, "cleaned"),
    ("MedicalEntityExtractor", {"backend": "dict"}, "sectioned"),
    ("MedicalRelationExtractor", {}, "entities"),
    ("TripleBuilderValidator", {}, "relations"),
]

_SAMPLE_TEXT = (
    "<p>基本信息：张三，男，58岁。</p>主　诉：胸痛　３天。&nbsp;"
    "现病史：患者胸痛，伴胸闷、气短。现病史：患者胸痛，伴胸闷、气短。"
    "辅助检查：行心电图检查，结果提示异常。诊断：冠心病。合并心律失常。"
    "治疗方案：给予阿司匹林口服治疗。行经皮冠状动脉介入治疗。"
    "医嘱：于心血管内科门诊就诊。联系电话：13812345678。\n"
)


def _prepare_stage_samples(base_samples: List[Dict[str, Any]], runner: LocalPipelineRunner):
    """把原始样本推进到各输入阶段并缓存（KG 组算子的前置产物）。"""
    stages: Dict[str, List[Dict[str, Any]]] = {"raw": base_samples}
    r1 = runner.run([{"MedicalTextCleaner": {"maskPii": True}}],
                    copy.deepcopy(base_samples), name="bench-stage-clean")
    stages["cleaned"] = r1.ok_samples
    r2 = runner.run([{"MedicalSectionSplitter": {}}],
                    copy.deepcopy(stages["cleaned"]), name="bench-stage-section")
    stages["sectioned"] = r2.ok_samples
    r3 = runner.run([{"MedicalEntityExtractor": {"backend": "dict"}}],
                    copy.deepcopy(stages["sectioned"]), name="bench-stage-entity")
    stages["entities"] = r3.ok_samples
    r4 = runner.run([{"MedicalRelationExtractor": {}}],
                    copy.deepcopy(stages["entities"]), name="bench-stage-relation")
    stages["relations"] = r4.ok_samples
    return stages


def run_bench(
    input_dir: Optional[Path] = None,
    ops: Optional[List] = None,
    repeat: int = 3,
    limit: int = 100,
    out_dir: Optional[Path] = None,
) -> Path:
    input_dir = Path(input_dir or RAW_DATA_DIR)
    out_dir = Path(out_dir or (OUTPUT_DIR / "bench"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = ops or DEFAULT_OPS

    # 输入样本：优先真实语料，否则内置样例 ×limit
    if input_dir.exists() and list(input_dir.glob("*.txt")):
        base = samples_from_dir(input_dir, pattern="*.txt")[:limit]
        # 预载文本（基准聚焦算子计算本身，排除磁盘IO差异）
        for s in base:
            s["text"] = Path(s["filePath"]).read_text(encoding="utf-8")
            s["filePath"] = ""
    else:
        base = [make_sample(text=_SAMPLE_TEXT, file_id=f"sample_{i}") for i in range(limit)]

    runner = LocalPipelineRunner()
    stages = _prepare_stage_samples(base, runner)

    results = []
    for op_name, params, stage in ops:
        if op_name not in dmcompat.OPERATORS.modules:
            continue
        stage_samples = stages.get(stage, base)
        if not stage_samples:
            continue
        runs = []
        last_report = None
        for _ in range(max(1, repeat)):
            res = runner.run([{op_name: dict(params)}], copy.deepcopy(stage_samples),
                             name=f"bench-{op_name}")
            m = res.report.op_metrics[0]
            runs.append(m)
            last_report = m
        thr = statistics.median(m.throughput_per_s() for m in runs)
        total_chars = sum(len(s.get("text", "")) for s in stage_samples)
        mb = total_chars * 3 / 1024 / 1024  # 估算 UTF-8 中文 ~3B/字符
        lat = last_report.latency_stats()
        results.append({
            "op": op_name,
            "kind": last_report.op_kind,
            "docs": len(stage_samples),
            "total_chars": total_chars,
            "throughput_docs_s": thr,
            "throughput_mb_s": round(mb / (last_report.wall_s or 1e-9), 3),
            "p50_ms": lat["p50_ms"],
            "p95_ms": lat["p95_ms"],
            "mean_ms": lat["mean_ms"],
            "n_failed": last_report.n_failed,
        })

    # CSV
    csv_path = out_dir / "operator_bench.csv"
    cols = ["op", "kind", "docs", "total_chars", "throughput_docs_s",
            "throughput_mb_s", "p50_ms", "p95_ms", "mean_ms", "n_failed"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # Markdown
    md_path = out_dir / "operator_bench.md"
    env = (
        f"- CPU: {platform.processor() or platform.machine()}\n"
        f"- OS: {platform.system()} {platform.release()}\n"
        f"- Python: {platform.python_version()}\n"
        f"- DataMate runtime: {'可用（真实 base_op/内置算子）' if dmcompat.DATAMATE_AVAILABLE else '不可用（dki.compat 兼容层）'}\n"
        f"- 语料: {results[0]['docs'] if results else 0} 份中文病历，repeat={repeat} 取中位\n"
    )
    lines = [
        "# 算子吞吐/延迟基准（CPU 实测）", "",
        "## 环境", env,
        "## 结果", "",
        "| 算子 | 类型 | 吞吐(docs/s) | 吞吐(MB/s) | p50(ms) | p95(ms) | 均值(ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['op']} | {r['kind']} | {r['throughput_docs_s']} | {r['throughput_mb_s']} "
            f"| {r['p50_ms']} | {r['p95_ms']} | {r['mean_ms']} |"
        )
    lines += [
        "",
        "> 前 4 行为 DataMate 内置算子（对照组），后 5 行为本项目医疗算子。",
        "> KG 组算子（分段/实体/关系/三元组）的输入为前置算子的真实产物。",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "operator_bench.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return md_path


def main():
    ap = argparse.ArgumentParser(description="算子基准测试")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    out = run_bench(repeat=args.repeat, limit=args.limit)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
