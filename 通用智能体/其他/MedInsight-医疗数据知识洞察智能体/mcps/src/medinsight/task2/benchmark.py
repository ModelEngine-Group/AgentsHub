from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
from typing import Any, Callable

from medinsight.common.utils import ensure_dir, write_json
from medinsight.task2.datasets import (
    TASK2_RAW_DIR,
    load_chip_cdn_split,
    load_cmeee_split,
    load_cmeie_split,
    load_cmedcausal_split,
)
from medinsight.task2.operators import (
    MedicalConflictDetector,
    MedicalEntityExtractor,
    MedicalKnowledgeGraphBuilder,
    MedicalRelationExtractor,
    MedicalTripleRepairer,
)
from medinsight.task2.operators.text_kg_extraction import COMMON_DISEASES, COMMON_SYMPTOMS, TEXT_KG_RELATION_SCHEMAS
from medinsight.task2.operators.triple_builder import MedicalTripleBuilder, MedicalTripleValidator
from medinsight.task2.service import parse_medical_text_records


DEFAULT_INPUT = Path("data/task2/samples/medical_text_sample.json")
DEFAULT_OUTPUT = Path("artifacts/task2/benchmark")
DEFAULT_SOURCE = "public-cblue"
DEFAULT_PUBLIC_LIMITS = {
    "cmeie": 400,
    "cmedcausal": 300,
    "cmeee": 200,
    "chip_cdn": 100,
}


def run_task2_operator_benchmark(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    iterations: int = 7,
    warmup: int = 1,
    repeat_records: int = 1,
    source: str = DEFAULT_SOURCE,
    raw_dir: Path = TASK2_RAW_DIR,
    cmeie_limit: int = DEFAULT_PUBLIC_LIMITS["cmeie"],
    cmedcausal_limit: int = DEFAULT_PUBLIC_LIMITS["cmedcausal"],
    cmeee_limit: int = DEFAULT_PUBLIC_LIMITS["cmeee"],
    chip_cdn_limit: int = DEFAULT_PUBLIC_LIMITS["chip_cdn"],
) -> dict[str, Any]:
    ensure_dir(output_dir)
    records, input_meta = load_benchmark_records(
        source=source,
        input_path=input_path,
        repeat_records=repeat_records,
        raw_dir=raw_dir,
        cmeie_limit=cmeie_limit,
        cmedcausal_limit=cmedcausal_limit,
        cmeee_limit=cmeee_limit,
        chip_cdn_limit=chip_cdn_limit,
    )
    benchmark_input_path = output_dir / "benchmark_input_records.json"
    write_json(benchmark_input_path, {"records": records})
    input_meta["preview_path"] = str(benchmark_input_path)
    prepared = prepare_operator_payloads(records)
    operators = build_operator_benchmarks(prepared, len(records))

    operator_results = []
    for item in operators:
        operator_results.append(
            benchmark_callable(
                name=item["name"],
                func=item["func"],
                workload_count=item["workload_count"],
                workload_unit=item["workload_unit"],
                iterations=iterations,
                warmup=warmup,
            )
        )

    end_to_end = benchmark_callable(
        name="task2_text_kg_end_to_end",
        func=lambda: run_end_to_end_pipeline(records),
        workload_count=len(records),
        workload_unit="records",
        iterations=iterations,
        warmup=warmup,
    )
    report = {
        "task": "task2_operator_benchmark",
        "description": "Task2 medical KG operator benchmark for latency, throughput, and available local resource metrics.",
        "environment": build_environment_info(),
        "input": {
            "source": source,
            "input_path": str(input_path) if source == "file" else None,
            "raw_dir": str(raw_dir) if source == "public-cblue" else None,
            "record_count": len(records),
            "repeat_records": repeat_records,
            "iterations": iterations,
            "warmup": warmup,
            "public_dataset_limits": {
                "cmeie": cmeie_limit,
                "cmedcausal": cmedcausal_limit,
                "cmeee": cmeee_limit,
                "chip_cdn": chip_cdn_limit,
            }
            if source == "public-cblue"
            else None,
            "record_sources": input_meta["record_sources"],
            "source_note": input_meta["source_note"],
            "benchmark_input_preview_path": input_meta.get("preview_path"),
            "operator_input_counts": prepared["counts"],
        },
        "resource_note": {
            "energy": "Energy or power is not available from the portable Python runtime on this machine; no energy value is fabricated.",
            "memory": "python_peak_memory_mb is measured by tracemalloc and covers Python allocations during each operator call.",
            "cpu": "cpu_time_ms and cpu_utilization_percent are process-level CPU measurements from Python process_time().",
        },
        "operators": operator_results,
        "end_to_end": end_to_end,
        "summary": build_summary(operator_results, end_to_end),
    }
    json_path = output_dir / "task2_operator_benchmark.json"
    markdown_path = output_dir / "task2_operator_benchmark.md"
    write_json(json_path, report)
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {
        "status": "ok",
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "report": report,
    }


def load_benchmark_records(
    *,
    source: str,
    input_path: Path,
    repeat_records: int,
    raw_dir: Path,
    cmeie_limit: int,
    cmedcausal_limit: int,
    cmeee_limit: int,
    chip_cdn_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_source = source.strip().lower()
    if normalized_source in {"file", "json", "sample"}:
        records = load_records_from_file(input_path, repeat_records)
        return records, {
            "record_sources": {"file": len(records)},
            "source_note": f"Records loaded from {input_path}.",
        }
    if normalized_source not in {"public-cblue", "cblue", "public"}:
        raise ValueError(f"unsupported benchmark source: {source}")
    base_records, source_counts = load_public_cblue_records(
        raw_dir=raw_dir,
        cmeie_limit=cmeie_limit,
        cmedcausal_limit=cmedcausal_limit,
        cmeee_limit=cmeee_limit,
        chip_cdn_limit=chip_cdn_limit,
    )
    records = repeat_records_with_ids(base_records, repeat_records)
    if not records:
        raise ValueError(f"no public CBLUE benchmark records found under {raw_dir}")
    return records, {
        "record_sources": source_counts,
        "source_note": (
            "记录来自本地公开 CBLUE 评测数据副本，混合抽样 CMeIE-V2、CMedCausal、CMeEE-V2 和 CHIP-CDN。"
            "评测使用原始医学文本作为通用医疗输入，不使用预构建知识图谱。"
        ),
    }


def load_records_from_file(input_path: Path, repeat_records: int) -> list[dict[str, Any]]:
    raw = input_path.read_text(encoding="utf-8")
    base_records = parse_medical_text_records(raw, "benchmark-medical-text")
    return repeat_records_with_ids(base_records, repeat_records)


def repeat_records_with_ids(base_records: list[dict[str, Any]], repeat_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    repeat = max(1, repeat_records)
    for repeat_index in range(repeat):
        for record_index, record in enumerate(base_records):
            item = dict(record)
            item["doc_id"] = f"{record.get('doc_id') or 'benchmark-doc'}-r{repeat_index:03d}-{record_index:04d}"
            records.append(item)
    return records


def load_public_cblue_records(
    *,
    raw_dir: Path,
    cmeie_limit: int,
    cmedcausal_limit: int,
    cmeee_limit: int,
    chip_cdn_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    def append_record(dataset: str, split: str, index: int, text: str, source_file: str, extra: dict[str, Any] | None = None) -> None:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return
        record = {
            "doc_id": f"{dataset.lower().replace('-', '_')}-{split}-{index:06d}",
            "text": cleaned,
            "dataset": f"CBLUE/{dataset}",
            "split": split,
            "source_file": source_file,
            "source_index": index,
            "source_refs": [
                {
                    "dataset": f"CBLUE/{dataset}",
                    "split": split,
                    "source_file": source_file,
                    "source_index": index,
                }
            ],
        }
        if extra:
            record.update(extra)
        records.append(record)
        source_counts[f"{dataset}/{split}"] = source_counts.get(f"{dataset}/{split}", 0) + 1

    cmeie_rows = choose_representative_rows(load_cmeie_split(raw_dir, "train", max(cmeie_limit * 4, cmeie_limit)), cmeie_limit)
    for index, row in enumerate(cmeie_rows):
        append_record("CMeIE-V2", "train", index, row.get("text", ""), "CMeIE-V2_train.jsonl")

    cmedcausal_rows = choose_representative_rows(
        load_cmedcausal_split(raw_dir, "train", max(cmedcausal_limit * 4, cmedcausal_limit)),
        cmedcausal_limit,
    )
    for index, row in enumerate(cmedcausal_rows):
        append_record("CMedCausal", "train", index, row.get("text", ""), "CMedCausal_train.json")

    cmeee_rows = choose_representative_rows(load_cmeee_split(raw_dir, "train", max(cmeee_limit * 4, cmeee_limit)), cmeee_limit)
    for index, row in enumerate(cmeee_rows):
        append_record("CMeEE-V2", "train", index, row.get("text", ""), "CMeEE-V2_train.json")

    chip_rows = choose_representative_rows(load_chip_cdn_split(raw_dir, "train", max(chip_cdn_limit * 4, chip_cdn_limit)), chip_cdn_limit)
    for index, row in enumerate(chip_rows):
        normalized = str(row.get("normalized_result") or "")
        text = str(row.get("text") or "")
        if normalized:
            text = f"{text}。标准术语包括：{normalized.replace('##', '、')}。"
        append_record("CHIP-CDN", "train", index, text, "CHIP-CDN_train.json")

    return records, source_counts


def choose_representative_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        scored.append((public_text_score(text), index, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:limit]]


def public_text_score(text: str) -> int:
    score = 0
    score += sum(3 for disease in COMMON_DISEASES if disease in text)
    score += sum(1 for symptom in COMMON_SYMPTOMS if symptom in text)
    for keyword in ("治疗", "检查", "症状", "表现", "病因", "诊断", "药", "可导致", "并发"):
        if keyword in text:
            score += 2
    score += min(len(text) // 80, 8)
    return score


def prepare_operator_payloads(records: list[dict[str, Any]]) -> dict[str, Any]:
    entity_extractor = MedicalEntityExtractor()
    relation_extractor = MedicalRelationExtractor()
    triple_builder = MedicalTripleBuilder()
    triple_validator = MedicalTripleValidator()
    triple_repairer = MedicalTripleRepairer()

    entity_payload = {"medical_text_records": records, "doc_id": "benchmark"}
    relation_payload = {"medical_text_records": records, "doc_id": "benchmark"}
    entity_result = entity_extractor.run(entity_payload)
    relation_result = relation_extractor.run(relation_payload)
    triple_payload = {"relations": relation_result["relations"]}
    triple_result = triple_builder.run(triple_payload)
    validation_payload = {"triples": triple_result["triples"], "schema": TEXT_KG_RELATION_SCHEMAS}
    validation_result = triple_validator.run(validation_payload)
    repair_payload = {
        "invalid_triples": validation_result["invalid_triples"],
        "schema": TEXT_KG_RELATION_SCHEMAS,
    }
    repair_result = triple_repairer.run(repair_payload)
    repaired_validation_payload = {
        "triples": repair_result["repaired_triples"],
        "schema": TEXT_KG_RELATION_SCHEMAS,
    }
    repaired_validation_result = triple_validator.run(repaired_validation_payload)
    valid_triples = validation_result["valid_triples"] + repaired_validation_result["valid_triples"]
    conflict_payload = {"triples": valid_triples}
    graph_payload = {"triples": valid_triples, "schema": TEXT_KG_RELATION_SCHEMAS}
    return {
        "entity_payload": entity_payload,
        "relation_payload": relation_payload,
        "triple_payload": triple_payload,
        "validation_payload": validation_payload,
        "repair_payload": repair_payload,
        "repaired_validation_payload": repaired_validation_payload,
        "conflict_payload": conflict_payload,
        "graph_payload": graph_payload,
        "counts": {
            "records": len(records),
            "entities": len(entity_result["entities"]),
            "relations": len(relation_result["relations"]),
            "triples": len(triple_result["triples"]),
            "invalid_triples": len(validation_result["invalid_triples"]),
            "repaired_triples": len(repair_result["repaired_triples"]),
            "valid_triples": len(valid_triples),
        },
    }


def build_operator_benchmarks(prepared: dict[str, Any], record_count: int) -> list[dict[str, Any]]:
    counts = prepared["counts"]
    entity_extractor = MedicalEntityExtractor()
    relation_extractor = MedicalRelationExtractor()
    triple_builder = MedicalTripleBuilder()
    triple_validator = MedicalTripleValidator()
    triple_repairer = MedicalTripleRepairer()
    conflict_detector = MedicalConflictDetector()
    graph_builder = MedicalKnowledgeGraphBuilder()
    return [
        {
            "name": entity_extractor.spec.name,
            "func": lambda: entity_extractor.run(prepared["entity_payload"]),
            "workload_count": record_count,
            "workload_unit": "records",
        },
        {
            "name": relation_extractor.spec.name,
            "func": lambda: relation_extractor.run(prepared["relation_payload"]),
            "workload_count": record_count,
            "workload_unit": "records",
        },
        {
            "name": triple_builder.spec.name,
            "func": lambda: triple_builder.run(prepared["triple_payload"]),
            "workload_count": counts["relations"],
            "workload_unit": "relations",
        },
        {
            "name": triple_validator.spec.name,
            "func": lambda: triple_validator.run(prepared["validation_payload"]),
            "workload_count": counts["triples"],
            "workload_unit": "triples",
        },
        {
            "name": triple_repairer.spec.name,
            "func": lambda: triple_repairer.run(prepared["repair_payload"]),
            "workload_count": counts["invalid_triples"],
            "workload_unit": "invalid_triples",
        },
        {
            "name": f"{triple_validator.spec.name}_after_repair",
            "func": lambda: triple_validator.run(prepared["repaired_validation_payload"]),
            "workload_count": counts["repaired_triples"],
            "workload_unit": "repaired_triples",
        },
        {
            "name": conflict_detector.spec.name,
            "func": lambda: conflict_detector.run(prepared["conflict_payload"]),
            "workload_count": counts["valid_triples"],
            "workload_unit": "triples",
        },
        {
            "name": graph_builder.spec.name,
            "func": lambda: graph_builder.run(prepared["graph_payload"]),
            "workload_count": counts["valid_triples"],
            "workload_unit": "triples",
        },
    ]


def run_end_to_end_pipeline(records: list[dict[str, Any]]) -> dict[str, Any]:
    entity_extractor = MedicalEntityExtractor()
    relation_extractor = MedicalRelationExtractor()
    triple_builder = MedicalTripleBuilder()
    triple_validator = MedicalTripleValidator()
    triple_repairer = MedicalTripleRepairer()
    conflict_detector = MedicalConflictDetector()
    graph_builder = MedicalKnowledgeGraphBuilder()

    entity_result = entity_extractor.run({"medical_text_records": records, "doc_id": "benchmark"})
    relation_result = relation_extractor.run({"medical_text_records": records, "doc_id": "benchmark"})
    triple_result = triple_builder.run({"relations": relation_result["relations"]})
    validation_result = triple_validator.run({"triples": triple_result["triples"], "schema": TEXT_KG_RELATION_SCHEMAS})
    repair_result = triple_repairer.run(
        {
            "invalid_triples": validation_result["invalid_triples"],
            "schema": TEXT_KG_RELATION_SCHEMAS,
        }
    )
    repaired_validation_result = triple_validator.run(
        {
            "triples": repair_result["repaired_triples"],
            "schema": TEXT_KG_RELATION_SCHEMAS,
        }
    )
    valid_triples = validation_result["valid_triples"] + repaired_validation_result["valid_triples"]
    conflict_result = conflict_detector.run({"triples": valid_triples})
    graph_result = graph_builder.run({"triples": valid_triples, "schema": TEXT_KG_RELATION_SCHEMAS})
    return {
        "entities": len(entity_result["entities"]),
        "relations": len(relation_result["relations"]),
        "triples": len(valid_triples),
        "conflicts": conflict_result["summary"]["conflict_count"],
        "graph_nodes": graph_result["summary"]["node_count"],
        "graph_edges": graph_result["summary"]["edge_count"],
    }


def benchmark_callable(
    *,
    name: str,
    func: Callable[[], Any],
    workload_count: int,
    workload_unit: str,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        func()
    wall_ms: list[float] = []
    cpu_ms: list[float] = []
    memory_mb: list[float] = []
    last_result: Any = None
    for _ in range(max(1, iterations)):
        gc.collect()
        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        last_result = func()
        cpu_elapsed = (time.process_time() - cpu_start) * 1000
        wall_elapsed = (time.perf_counter() - wall_start) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        wall_ms.append(round(wall_elapsed, 6))
        cpu_ms.append(round(cpu_elapsed, 6))
        memory_mb.append(round(peak / (1024 * 1024), 6))

    avg_latency = mean(wall_ms)
    avg_cpu = mean(cpu_ms)
    throughput = None
    if workload_count > 0 and avg_latency > 0:
        throughput = round(workload_count / (avg_latency / 1000), 3)
    return {
        "operator": name,
        "device": "CPU",
        "workload_count": workload_count,
        "workload_unit": workload_unit,
        "iterations": max(1, iterations),
        "latency_ms": {
            "avg": round(avg_latency, 3),
            "median": round(statistics.median(wall_ms), 3),
            "p95": round(percentile(wall_ms, 95), 3),
            "min": round(min(wall_ms), 3),
            "max": round(max(wall_ms), 3),
        },
        "throughput_per_second": throughput,
        "resource_utilization": {
            "avg_cpu_time_ms": round(avg_cpu, 3),
            "avg_cpu_utilization_percent": round((avg_cpu / avg_latency) * 100, 2) if avg_latency else None,
            "peak_python_memory_mb": round(max(memory_mb), 3),
            "energy_joules": None,
            "energy_status": "unavailable",
        },
        "raw_samples": {
            "wall_ms": wall_ms,
            "cpu_ms": cpu_ms,
            "python_peak_memory_mb": memory_mb,
        },
        "result_summary": summarize_result(last_result),
    }


def summarize_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"type": type(result).__name__}
    summary: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, list):
            summary[key] = len(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, dict) and key == "summary":
            summary[key] = value
        elif isinstance(value, dict) and key == "graph":
            summary[key] = {
                "nodes": len(value.get("nodes") or []),
                "edges": len(value.get("edges") or []),
            }
    return summary


def build_environment_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def build_summary(operator_results: list[dict[str, Any]], end_to_end: dict[str, Any]) -> dict[str, Any]:
    slowest = sorted(operator_results, key=lambda item: item["latency_ms"]["avg"], reverse=True)[:3]
    return {
        "slowest_operators": [
            {
                "operator": item["operator"],
                "avg_latency_ms": item["latency_ms"]["avg"],
                "throughput_per_second": item["throughput_per_second"],
            }
            for item in slowest
        ],
        "end_to_end_latency_ms": end_to_end["latency_ms"],
        "end_to_end_throughput_per_second": end_to_end["throughput_per_second"],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Task2 算子性能评测报告",
        "",
        "## 评测设置",
        "",
        f"- 输入来源：`{report['input']['source']}`",
    ]
    if report["input"].get("input_path"):
        lines.append(f"- 输入文件：`{report['input']['input_path']}`")
    if report["input"].get("raw_dir"):
        lines.append(f"- 公开数据目录：`{report['input']['raw_dir']}`")
    lines.extend(
        [
            f"- 记录数：{report['input']['record_count']}",
            f"- 重复次数：{report['input']['iterations']} 次，预热：{report['input']['warmup']} 次",
            "- 运行设备：CPU",
            f"- Python：{report['environment']['python']}",
            f"- 平台：{report['environment']['platform']}",
            f"- 数据说明：{report['input']['source_note']}",
        ]
    )
    if report["input"].get("benchmark_input_preview_path"):
        lines.append(f"- 实际评测输入：`{report['input']['benchmark_input_preview_path']}`")
    lines.extend(["", "## 数据构成", "", "| 数据源 | 记录数 |", "|---|---:|"])
    for name, count in (report["input"].get("record_sources") or {}).items():
        lines.append(f"| {name} | {count} |")
    counts = report["input"].get("operator_input_counts") or {}
    lines.extend(
        [
            "",
            "## 输入规模",
            "",
            f"- 文本记录：{counts.get('records', report['input']['record_count'])}",
            f"- 预抽取实体：{counts.get('entities', 0)}",
            f"- 预抽取关系：{counts.get('relations', 0)}",
            f"- 候选三元组：{counts.get('triples', 0)}",
            f"- 有效三元组：{counts.get('valid_triples', 0)}",
            "",
        ]
    )
    lines.extend(
        [
        "## 指标说明",
        "",
        "- 吞吐量：每秒处理的输入单元数量，单位随算子变化，例如 records/s、relations/s、triples/s。",
        "- 延迟：同一算子多次运行的平均、P50、P95、最小和最大耗时。",
        "- 资源利用率：记录进程 CPU 时间、估算 CPU 使用率和 Python 分配内存峰值。",
        "- 能效：当前本地 Python 运行时无法可靠读取能耗或功率，因此报告为 unavailable，不做虚假估计。",
        "",
        "## 算子结果",
        "",
            "| 算子 | 输入规模 | 平均延迟(ms) | P95(ms) | 吞吐量(/s) | CPU时间(ms) | CPU利用率(%) | 内存峰值(MB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["operators"]:
        lines.append(
            "| {operator} | {count} {unit} | {avg} | {p95} | {throughput} | {cpu} | {cpu_percent} | {mem} |".format(
                operator=item["operator"],
                count=item["workload_count"],
                unit=item["workload_unit"],
                avg=item["latency_ms"]["avg"],
                p95=item["latency_ms"]["p95"],
                throughput=item["throughput_per_second"] if item["throughput_per_second"] is not None else "N/A",
                cpu=item["resource_utilization"]["avg_cpu_time_ms"],
                cpu_percent=item["resource_utilization"]["avg_cpu_utilization_percent"],
                mem=item["resource_utilization"]["peak_python_memory_mb"],
            )
        )
    end = report["end_to_end"]
    lines.extend(
        [
            "",
            "## 端到端结果",
            "",
            "| 流程 | 输入规模 | 平均延迟(ms) | P95(ms) | 吞吐量(records/s) | CPU时间(ms) | 内存峰值(MB) |",
            "|---|---:|---:|---:|---:|---:|---:|",
            "| Task2 文本到知识图谱 | {count} records | {avg} | {p95} | {throughput} | {cpu} | {mem} |".format(
                count=end["workload_count"],
                avg=end["latency_ms"]["avg"],
                p95=end["latency_ms"]["p95"],
                throughput=end["throughput_per_second"],
                cpu=end["resource_utilization"]["avg_cpu_time_ms"],
                mem=end["resource_utilization"]["peak_python_memory_mb"],
            ),
            "",
            "## 结论",
            "",
        ]
    )
    for item in report["summary"]["slowest_operators"]:
        lines.append(
            f"- {item['operator']} 平均延迟 {item['avg_latency_ms']} ms，吞吐量 {item['throughput_per_second']} /s。"
        )
    lines.append("- 能耗/功率指标当前不可获取，报告中不做估算。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Task2 medical KG operators.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Medical text JSON/text input.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output benchmark directory.")
    parser.add_argument(
        "--source",
        choices=["public-cblue", "file", "sample"],
        default=DEFAULT_SOURCE,
        help="Benchmark input source. public-cblue samples local public CBLUE data; file/sample reads --input.",
    )
    parser.add_argument("--raw-dir", default=str(TASK2_RAW_DIR), help="Local public CBLUE raw data directory.")
    parser.add_argument("--cmeie-limit", type=int, default=DEFAULT_PUBLIC_LIMITS["cmeie"])
    parser.add_argument("--cmedcausal-limit", type=int, default=DEFAULT_PUBLIC_LIMITS["cmedcausal"])
    parser.add_argument("--cmeee-limit", type=int, default=DEFAULT_PUBLIC_LIMITS["cmeee"])
    parser.add_argument("--chip-cdn-limit", type=int, default=DEFAULT_PUBLIC_LIMITS["chip_cdn"])
    parser.add_argument("--iterations", type=int, default=7, help="Measured iterations per operator.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per operator.")
    parser.add_argument("--repeat-records", type=int, default=1, help="Repeat input records to scale benchmark workload.")
    args = parser.parse_args()
    result = run_task2_operator_benchmark(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        iterations=args.iterations,
        warmup=args.warmup,
        repeat_records=args.repeat_records,
        source="file" if args.source == "sample" else args.source,
        raw_dir=Path(args.raw_dir),
        cmeie_limit=args.cmeie_limit,
        cmedcausal_limit=args.cmedcausal_limit,
        cmeee_limit=args.cmeee_limit,
        chip_cdn_limit=args.chip_cdn_limit,
    )
    report = result["report"]
    print("Task2 算子性能评测完成。")
    print(f"JSON 报告：{result['json_path']}")
    print(f"Markdown 报告：{result['markdown_path']}")
    print(f"端到端平均延迟：{report['end_to_end']['latency_ms']['avg']} ms")
    print(f"端到端吞吐量：{report['end_to_end']['throughput_per_second']} records/s")


if __name__ == "__main__":
    main()
