# -*- coding: utf-8 -*-
"""
任务二结果报告模块。
"""

from __future__ import annotations

import json
from collections import Counter


def compact(value, limit: int = 1600) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...(截断)"


def format_stage_duration(value: float) -> str:
    if value < 0.01:
        return "<0.01s"
    return f"{value:.2f}s"


def summarize_source_files(
    records: list[dict],
    selected_records: list[dict],
    record_results: list[dict],
) -> list[dict]:
    formats: dict[str, str] = {}
    parsed_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    processed_counts: Counter[str] = Counter()

    for record in records:
        source_file = str(record.get("source_file") or "unknown")
        parsed_counts[source_file] += 1
        formats[source_file] = str(record.get("source_format") or formats.get(source_file) or "unknown")
    for record in selected_records:
        source_file = str(record.get("source_file") or "unknown")
        selected_counts[source_file] += 1
        formats[source_file] = str(record.get("source_format") or formats.get(source_file) or "unknown")
    for item in record_results:
        source_file = str(item.get("source_file") or "unknown")
        processed_counts[source_file] += 1

    return [
        {
            "source_file": source_file,
            "source_format": formats.get(source_file, "unknown"),
            "parsed_records": parsed_counts[source_file],
            "selected_records": selected_counts[source_file],
            "processed_records": processed_counts[source_file],
        }
        for source_file in sorted(parsed_counts)
    ]


def summarize_analytics_refresh(result: dict) -> list[dict]:
    labels = {
        "diseases": "疾病数",
        "disease_facts": "知识事实",
        "disease_symptoms": "疾病-症状关系",
        "disease_drugs": "疾病-药物关系",
        "disease_departments": "疾病-科室关系",
        "disease_complications": "疾病-并发症关系",
        "disease_tests": "疾病-检查关系",
        "disease_procedures": "疾病-治疗关系",
        "disease_causes": "疾病-病因关系",
        "disease_preventions": "疾病-预防关系",
        "disease_populations": "疾病-人群关系",
        "entity_stats": "实体统计项",
        "relation_stats": "关系统计项",
        "qa_examples": "问答样例",
    }
    stats = result.get("stats") or {}
    return [{"metric": labels.get(key, key), "value": value} for key, value in stats.items()]
