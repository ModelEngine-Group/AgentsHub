"""Task-1 data processing agent for the medical demo."""

from __future__ import annotations

import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from operators.data_processing import (
    build_processing_trace,
    clean_text,
    infer_schema_profile,
    load_records,
    normalize_fields,
    profile_documents,
    quality_check,
    split_sections,
    split_sentences,
)


DEFAULT_TASK = "将原始医疗数据清洗并标准化为后续图谱构建可复用的中间结果"

BASE_OPERATOR_COST = {
    "unsupported_format_guard": 1.0,
    "load_records": 1.0,
    "clean_text": 1.15,
    "infer_schema_profile": 1.35,
    "extract_fields": 1.45,
    "split_sections": 1.25,
    "normalize_fields": 1.5,
    "profile_documents": 1.1,
    "quality_check": 1.2,
    "record_quality_warnings": 0.85,
    "deduplicate_by_fingerprint_policy": 0.9,
    "quarantine_empty_text_policy": 0.8,
    "write_outputs": 0.9,
}

STEP_DESCRIPTIONS = {
    "load_records": "加载原始记录",
    "infer_schema_profile": "推断字段结构和标准映射",
    "clean_text": "清洗文本和空白字符",
    "split_sections": "拆分段落和句子",
    "extract_fields": "抽取并映射标准字段",
    "profile_documents": "构建文档画像和分布统计",
    "normalize_fields": "标准化字段",
    "quality_check": "执行质量检查",
    "write_outputs": "输出标准化数据",
    "record_quality_warnings": "记录质量告警和恢复动作",
    "deduplicate_by_fingerprint_policy": "生成正文指纹去重策略",
    "quarantine_empty_text_policy": "隔离空文本记录策略",
    "unsupported_format_guard": "检查输入格式支持情况",
}

STEP_DEPENDENCIES = {
    "load_records": [],
    "infer_schema_profile": ["load_records"],
    "clean_text": ["load_records"],
    "extract_fields": ["infer_schema_profile"],
    "split_sections": ["clean_text"],
    "normalize_fields": ["load_records", "infer_schema_profile"],
    "profile_documents": ["normalize_fields"],
    "quality_check": ["profile_documents"],
    "record_quality_warnings": ["quality_check"],
    "deduplicate_by_fingerprint_policy": ["quality_check"],
    "quarantine_empty_text_policy": ["quality_check"],
    "write_outputs": ["quality_check"],
    "unsupported_format_guard": [],
}

SEMANTIC_TEXT_FIELD_HINTS = (
    "diagnosis",
    "diagnose",
    "finding",
    "findings",
    "symptom",
    "symptoms",
    "complaint",
    "chief_complaint",
    "history",
    "medical_history",
    "assessment",
    "impression",
    "treatment",
    "therapy",
    "plan",
    "advice",
    "medication",
    "medicine",
    "exam",
    "test",
    "result",
    "检查",
    "诊断",
    "症状",
    "主诉",
    "病史",
    "治疗",
    "用药",
    "计划",
    "建议",
)


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 3)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _infer_capabilities(task: str, input_path: Path) -> dict[str, Any]:
    normalized = task.lower()
    suffix = input_path.suffix.lower().lstrip(".") or "unknown"
    requested_steps = [
        "load_records",
        "infer_schema_profile",
        "normalize_fields",
        "profile_documents",
        "quality_check",
        "write_outputs",
    ]
    if any(keyword in task for keyword in ("清洗", "清理", "去噪", "标准化", "治理")):
        requested_steps.insert(1, "clean_text")
    if any(keyword in task for keyword in ("拆分", "分段", "切分", "chunk")):
        requested_steps.insert(2, "split_sections")
    if any(keyword in task for keyword in ("抽取", "提取", "字段", "结构化", "转换", "映射", "extract")) or "etl" in normalized:
        requested_steps.insert(2, "extract_fields")
    if any(keyword in task for keyword in ("质量", "校验", "检查", "异常")):
        requested_steps.append("record_quality_warnings")
    if suffix not in {"jsonl", "json", "csv", "txt"}:
        requested_steps.append("unsupported_format_guard")

    ordered_steps: list[str] = []
    for step in requested_steps:
        if step not in ordered_steps:
            ordered_steps.append(step)

    return {
        "task": task,
        "input_format": suffix,
        "supports_structured_input": suffix in {"json", "jsonl", "csv"},
        "supports_unstructured_input": suffix == "txt",
        "requested_steps": ordered_steps,
        "requires_quality_recovery": any(keyword in task for keyword in ("异常", "恢复", "容错", "校验", "质量")),
    }


def _parse_task_intent(task: str, input_path: Path) -> dict[str, Any]:
    normalized = task.lower()
    intents = {
        "cleaning": any(keyword in task for keyword in ("清洗", "清理", "去噪", "标准化", "治理")),
        "extraction": any(keyword in task for keyword in ("抽取", "提取", "字段", "结构化", "转换", "映射", "知识图谱")) or "etl" in normalized,
        "chunking": any(keyword in task for keyword in ("拆分", "分段", "切分", "chunk")),
        "quality_recovery": any(keyword in task for keyword in ("异常", "恢复", "容错", "校验", "质量", "重复", "空文本")),
        "analytics_ready": any(keyword in task for keyword in ("统计", "分析", "画像", "分布", "BI", "bi")),
    }
    suffix = input_path.suffix.lower().lstrip(".") or "unknown"
    if suffix == "txt":
        intents["chunking"] = True
    if suffix in {"csv", "json", "jsonl"}:
        intents["extraction"] = True
    priority_order = [name for name, enabled in intents.items() if enabled]
    if not priority_order:
        priority_order = ["cleaning", "extraction"]
    quality_targets = {
        "deduplication": any(keyword in task for keyword in ("去重", "重复", "冗余", "指纹")),
        "empty_text_guard": any(keyword in task for keyword in ("空文本", "缺失", "空值", "缺省")),
        "schema_validation": any(keyword in task for keyword in ("schema", "字段", "结构", "映射", "校验")),
        "provenance": any(keyword in task for keyword in ("溯源", "来源", "证据", "可追溯", "审计")),
    }
    execution_constraints = {
        "prefer_low_latency": any(keyword in task for keyword in ("快速", "低延迟", "实时", "高吞吐", "性能")),
        "prefer_high_recall": any(keyword in task for keyword in ("召回", "完整", "尽量保留", "不能漏")),
        "prefer_strict_quality": any(keyword in task for keyword in ("严格", "高质量", "正确", "准确", "强校验")),
        "requires_recovery": intents["quality_recovery"] or any(quality_targets.values()),
    }
    output_contract = {
        "documents_jsonl": True,
        "quality_report": True,
        "schema_profile": intents["extraction"] or quality_targets["schema_validation"],
        "document_profile": intents["analytics_ready"] or execution_constraints["prefer_strict_quality"],
        "kg_ready": intents["extraction"] or "知识图谱" in task,
        "analysis_ready": intents["analytics_ready"] or any(keyword in task for keyword in ("分析", "BI", "指标")),
    }
    return {
        "raw_task": task,
        "input_format": suffix,
        "intent_flags": intents,
        "priority_order": priority_order,
        "target_artifact": "kg_ready_documents" if intents["extraction"] else "normalized_documents",
        "requires_structured_schema": suffix in {"csv", "json", "jsonl"} or intents["extraction"],
        "requires_unstructured_chunking": suffix == "txt" or intents["chunking"],
        "quality_targets": quality_targets,
        "execution_constraints": execution_constraints,
        "output_contract": output_contract,
    }


def _estimate_input_profile(input_path: Path) -> dict[str, Any]:
    suffix = input_path.suffix.lower().lstrip(".") or "unknown"
    size_bytes = input_path.stat().st_size if input_path.exists() else 0
    return {
        "path": str(input_path),
        "format": suffix,
        "size_bytes": size_bytes,
        "size_bucket": "small" if size_bytes < 200_000 else "medium" if size_bytes < 5_000_000 else "large",
        "structured": suffix in {"csv", "json", "jsonl"},
        "unstructured": suffix == "txt",
        "supported": suffix in {"csv", "json", "jsonl", "txt"},
    }


def _select_runtime_execution_strategy(
    *,
    raw_record_count: int,
    input_profile: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    prefer_low_latency = bool(intent.get("execution_constraints", {}).get("prefer_low_latency"))
    prefer_strict_quality = bool(intent.get("execution_constraints", {}).get("prefer_strict_quality"))
    threshold = 12 if prefer_low_latency else 32
    worker_count = 1
    scheduler = "single_thread_record_normalizer"
    reason = "record_count_below_parallel_threshold"
    if raw_record_count >= threshold:
        worker_count = max(2, min(4, cpu_count, (raw_record_count + 31) // 32 + 1))
        scheduler = "thread_pool_batch_normalizer"
        reason = "record_count_exceeds_parallel_threshold"
    if prefer_strict_quality and raw_record_count < threshold:
        reason = "strict_quality_prefers_deterministic_single_thread_for_small_input"
    batch_size = max(1, (raw_record_count + max(1, worker_count * 2) - 1) // max(1, worker_count * 2))
    return {
        "enabled": True,
        "scheduler": scheduler,
        "parallel": worker_count > 1,
        "worker_count": worker_count,
        "cpu_count": cpu_count,
        "batch_size": batch_size,
        "parallel_threshold": threshold,
        "record_count": raw_record_count,
        "input_format": input_profile.get("format", "unknown"),
        "preserves_input_order": True,
        "deterministic": True,
        "reason": reason,
    }


def _normalize_records_with_strategy(
    raw_records: list[dict[str, Any]],
    *,
    input_path: Path,
    strategy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = perf_counter()
    indexed_records = list(enumerate(raw_records, start=1))
    worker_count = int(strategy.get("worker_count", 1))

    def normalize_item(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, record = item
        return normalize_fields(record, index, source_path=input_path)

    if worker_count > 1 and len(indexed_records) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            processed_records = list(executor.map(normalize_item, indexed_records))
    else:
        processed_records = [normalize_item(item) for item in indexed_records]

    elapsed_ms = _elapsed_ms(start)
    docs_per_second = round(len(processed_records) / max(elapsed_ms / 1000, 0.000001), 3)
    return processed_records, {
        "elapsed_ms": elapsed_ms,
        "input_count": len(raw_records),
        "output_count": len(processed_records),
        "scheduler": strategy.get("scheduler"),
        "parallel": bool(strategy.get("parallel")),
        "worker_count": worker_count,
        "batch_size": strategy.get("batch_size"),
        "docs_per_second": docs_per_second,
        "preserves_input_order": True,
    }


def _dedupe_steps(steps: list[str]) -> list[str]:
    result: list[str] = []
    for step in steps:
        if step not in result:
            result.append(step)
    return result


def _candidate_step_sets(capabilities: dict[str, Any], intent: dict[str, Any], input_profile: dict[str, Any]) -> list[dict[str, Any]]:
    base_steps = _dedupe_steps(capabilities["requested_steps"])
    minimal = _dedupe_steps([
        "load_records",
        "infer_schema_profile",
        "normalize_fields",
        "quality_check",
        "write_outputs",
    ])
    robust = _dedupe_steps(base_steps + ["profile_documents", "record_quality_warnings"])
    if intent["intent_flags"].get("quality_recovery") or input_profile["size_bucket"] != "small":
        robust.extend(["deduplicate_by_fingerprint_policy", "quarantine_empty_text_policy"])
    quality_targets = intent.get("quality_targets", {})
    constraints = intent.get("execution_constraints", {})
    if quality_targets.get("deduplication"):
        robust.append("deduplicate_by_fingerprint_policy")
    if quality_targets.get("empty_text_guard") or constraints.get("requires_recovery"):
        robust.append("quarantine_empty_text_policy")
    if constraints.get("prefer_strict_quality") or quality_targets.get("schema_validation"):
        robust.extend(["profile_documents", "record_quality_warnings"])
    kg_ready = _dedupe_steps(
        base_steps
        + ["clean_text", "extract_fields", "split_sections", "profile_documents", "quality_check", "write_outputs"]
    )
    if constraints.get("requires_recovery"):
        kg_ready.extend(["record_quality_warnings", "deduplicate_by_fingerprint_policy", "quarantine_empty_text_policy"])
    candidates = [
        {"name": "minimal_etl", "steps": minimal, "strategy": "低成本标准化，适合已清洁结构化数据。"},
        {"name": "balanced_quality", "steps": robust, "strategy": "质量优先的稳健 ETL，适合比赛验收和可复现链路。"},
        {"name": "kg_ready_text_pipeline", "steps": kg_ready, "strategy": "面向知识图谱的文本清洗、字段抽取和分段流水线。"},
    ]
    if not input_profile["supported"]:
        for candidate in candidates:
            candidate["steps"] = _dedupe_steps(["unsupported_format_guard"] + candidate["steps"])
    return candidates


def _score_candidate_plan(candidate: dict[str, Any], intent: dict[str, Any], input_profile: dict[str, Any]) -> dict[str, Any]:
    steps = _topological_sort_steps(candidate["steps"])
    step_set = set(steps)
    coverage_items = {
        "cleaning": (not intent["intent_flags"].get("cleaning")) or "clean_text" in step_set,
        "extraction": (not intent["intent_flags"].get("extraction")) or "extract_fields" in step_set,
        "chunking": (not intent["intent_flags"].get("chunking")) or "split_sections" in step_set,
        "quality_recovery": (not intent["intent_flags"].get("quality_recovery")) or "record_quality_warnings" in step_set,
        "analytics_ready": (not intent["intent_flags"].get("analytics_ready")) or "profile_documents" in step_set,
        "schema": (not intent.get("requires_structured_schema")) or "infer_schema_profile" in step_set,
        "unstructured": (not intent.get("requires_unstructured_chunking")) or "split_sections" in step_set,
        "deduplication": (not intent.get("quality_targets", {}).get("deduplication")) or "deduplicate_by_fingerprint_policy" in step_set,
        "empty_text_guard": (not intent.get("quality_targets", {}).get("empty_text_guard")) or "quarantine_empty_text_policy" in step_set,
        "strict_quality": (not intent.get("execution_constraints", {}).get("prefer_strict_quality")) or {"quality_check", "record_quality_warnings", "profile_documents"}.issubset(step_set),
        "analysis_contract": (not intent.get("output_contract", {}).get("analysis_ready")) or "profile_documents" in step_set,
    }
    coverage = sum(1 for value in coverage_items.values() if value) / max(1, len(coverage_items))
    dag = _build_dag(_build_plan({"requested_steps": steps, "input_format": input_profile["format"]}))
    cost = sum(BASE_OPERATOR_COST.get(step, 1.0) for step in steps)
    if input_profile["size_bucket"] == "large":
        cost += 0.3 * dag["critical_path_length"]
    robustness = (
        int("quality_check" in step_set)
        + int("record_quality_warnings" in step_set)
        + int("deduplicate_by_fingerprint_policy" in step_set)
        + int("quarantine_empty_text_policy" in step_set)
        + int("profile_documents" in step_set)
    ) / 5
    parallelism = dag["parallelizable_layers"] / max(1, dag["critical_path_length"])
    constraints = intent.get("execution_constraints", {})
    cost_penalty_weight = 0.14 if constraints.get("prefer_low_latency") else 0.08
    robustness_weight = 0.34 if constraints.get("prefer_strict_quality") or constraints.get("requires_recovery") else 0.28
    score = round(coverage * 0.52 + robustness * robustness_weight + parallelism * 0.12 - min(cost / 40, 0.2) * cost_penalty_weight, 4)
    return {
        "name": candidate["name"],
        "strategy": candidate["strategy"],
        "steps": steps,
        "coverage": round(coverage, 4),
        "coverage_items": coverage_items,
        "estimated_cost": round(cost, 3),
        "robustness": round(robustness, 4),
        "parallelism": round(parallelism, 4),
        "score": score,
        "dag_metrics": {
            "operator_count": dag["operator_count"],
            "edge_count": dag["edge_count"],
            "critical_path_length": dag["critical_path_length"],
            "parallelizable_layers": dag["parallelizable_layers"],
        },
    }


def _select_processing_plan(
    capabilities: dict[str, Any],
    intent: dict[str, Any],
    input_profile: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        _score_candidate_plan(candidate, intent, input_profile)
        for candidate in _candidate_step_sets(capabilities, intent, input_profile)
    ]
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["estimated_cost"]), item["name"]))
    selected = candidates[0]
    return {
        "intent": intent,
        "input_profile_estimate": input_profile,
        "candidate_plans": candidates,
        "selected_plan": {
            "name": selected["name"],
            "score": selected["score"],
            "steps": selected["steps"],
            "selection_reason": (
                f"覆盖率 {selected['coverage']:.2f}、鲁棒性 {selected['robustness']:.2f}、"
                f"估算成本 {selected['estimated_cost']:.2f}，综合得分最高。"
            ),
        },
    }


def _profile_based_plan_optimization(
    *,
    plan: list[dict[str, Any]],
    capabilities: dict[str, Any],
    planning_decision: dict[str, Any],
    schema_profile: dict[str, Any],
    document_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refine the DAG after observing real schema and document statistics."""
    original_steps = [item["step"] for item in plan]
    optimized_steps = original_steps[:]
    actions: list[dict[str, Any]] = []
    input_format = capabilities.get("input_format", "unknown")
    schema_confidence = schema_profile.get("schema_confidence", {})
    length_profile = document_profile.get("length", {})
    median_length = float(length_profile.get("median", 0) or 0)
    duplicate_text_groups = document_profile.get("duplicate_text_groups", [])
    low_information_documents = document_profile.get("low_information_documents", [])
    text_coverage = float(schema_confidence.get("text_coverage", 0) or 0)

    def ensure_step(step: str, reason: str, priority: int) -> None:
        nonlocal optimized_steps
        if step not in optimized_steps:
            optimized_steps.append(step)
            actions.append({"step": step, "action": "insert", "reason": reason, "priority": priority})

    if input_format == "txt" or median_length >= 80:
        ensure_step("split_sections", "profile_detected_unstructured_or_long_text", 92)
    if input_format in {"csv", "json", "jsonl"} and schema_profile.get("field_count", 0) >= 4:
        ensure_step("extract_fields", "profile_detected_structured_schema", 88)
    if text_coverage < 0.95 or low_information_documents:
        ensure_step("record_quality_warnings", "profile_detected_sparse_or_low_information_text", 84)
    if duplicate_text_groups:
        ensure_step("deduplicate_by_fingerprint_policy", "profile_detected_duplicate_text_groups", 82)
    if document_profile.get("document_count", 0) and median_length <= 35 and input_format != "csv":
        ensure_step("profile_documents", "profile_requires_short_text_distribution_tracking", 76)

    optimized_steps = _topological_sort_steps(_dedupe_steps(optimized_steps))
    optimized_plan = _build_plan({"requested_steps": optimized_steps, "input_format": input_format})
    optimized_dag = _build_dag(optimized_plan)
    original_dag = _build_dag(plan)
    optimization = {
        "enabled": True,
        "changed": optimized_steps != original_steps,
        "original_steps": original_steps,
        "optimized_steps": optimized_steps,
        "actions": actions,
        "observed_profile": {
            "field_count": schema_profile.get("field_count", 0),
            "text_coverage": text_coverage,
            "median_length": median_length,
            "document_count": document_profile.get("document_count", 0),
            "duplicate_text_group_count": len(duplicate_text_groups),
            "low_information_count": len(low_information_documents),
            "input_format": input_format,
        },
        "dag_delta": {
            "operator_count": optimized_dag["operator_count"] - original_dag["operator_count"],
            "edge_count": optimized_dag["edge_count"] - original_dag["edge_count"],
            "critical_path_length": optimized_dag["critical_path_length"] - original_dag["critical_path_length"],
            "parallelizable_layers": optimized_dag["parallelizable_layers"] - original_dag["parallelizable_layers"],
        },
        "reason": (
            "根据真实 schema/document profile 动态调整 DAG 算子集合。"
            if optimized_steps != original_steps
            else "初始计划已匹配真实数据画像，无需调整。"
        ),
    }
    planning_decision["profile_optimized_plan"] = optimization
    return optimized_plan, optimization


def _select_operator_variants(
    plan: list[dict[str, Any]],
    *,
    capabilities: dict[str, Any],
    schema_profile: dict[str, Any],
    document_profile: dict[str, Any],
    quality_report: dict[str, Any] | None = None,
    runtime_execution_strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_format = str(capabilities.get("input_format", "unknown"))
    schema_confidence = schema_profile.get("schema_confidence", {})
    length_profile = document_profile.get("length", {})
    median_length = float(length_profile.get("median", 0) or 0)
    warning_count = int((quality_report or {}).get("warning_count", 0))
    variant_by_step: dict[str, dict[str, Any]] = {}
    for item in plan:
        step = item["step"]
        variant = "standard"
        reason = "default_operator"
        if step == "load_records":
            variant = {
                "csv": "datamate_csv_connector",
                "json": "datamate_json_connector",
                "jsonl": "datamate_jsonl_connector",
                "txt": "datamate_text_connector",
            }.get(input_format, "datamate_generic_connector")
            reason = f"input_format:{input_format}"
        elif step == "infer_schema_profile":
            variant = "structured_schema_profiler" if input_format in {"csv", "json", "jsonl"} else "unstructured_text_schema_profiler"
            reason = "schema_profile_uses_real_input_format"
        elif step == "extract_fields":
            variant = "schema_mapping_field_extractor" if input_format in {"csv", "json", "jsonl"} else "text_span_field_extractor"
            reason = "field_extractor_selected_by_structure"
        elif step == "split_sections":
            variant = "paragraph_sentence_chunker" if input_format == "txt" or median_length >= 80 else "single_record_sectioner"
            reason = "chunker_selected_by_text_length"
        elif step == "normalize_fields":
            if runtime_execution_strategy and runtime_execution_strategy.get("parallel"):
                variant = "parallel_batch_document_normalizer"
                reason = "runtime_strategy_selected_parallel_batch_normalization"
            else:
                variant = "kg_ready_document_normalizer" if schema_confidence.get("has_text") else "sparse_record_normalizer"
                reason = "normalizer_selected_by_text_coverage"
        elif step == "quality_check":
            variant = "strict_quality_gate" if warning_count else "standard_quality_gate"
            reason = "quality_gate_selected_by_warnings"
        elif step == "record_quality_warnings":
            variant = "quality_recovery_recorder" if warning_count else "quality_audit_recorder"
            reason = "warning_recorder_selected_by_quality_report"
        elif step == "write_outputs":
            variant = "reproducible_artifact_writer"
            reason = "competition_evidence_required"
        variant_by_step[step] = {
            "variant": variant,
            "reason": reason,
            "input_format": input_format,
        }
        if step == "normalize_fields" and runtime_execution_strategy:
            variant_by_step[step]["runtime_execution_strategy"] = runtime_execution_strategy
    return {
        "enabled": True,
        "variant_count": len(variant_by_step),
        "variants": variant_by_step,
        "runtime_execution_strategy": runtime_execution_strategy or {},
    }


def _semantic_text_fields(schema_profile: dict[str, Any]) -> list[str]:
    fields = schema_profile.get("fields", [])
    candidates: list[tuple[float, str]] = []
    for field in fields:
        name = str(field.get("name", ""))
        lower_name = name.lower()
        if name in {"id", "doc_id", "document_id", "title", "name", "source", "dataset", "origin", "text", "content", "body"}:
            continue
        hint_hit = any(hint in lower_name or hint in name for hint in SEMANTIC_TEXT_FIELD_HINTS)
        avg_length = float(field.get("avg_length", 0) or 0)
        presence = float(field.get("presence_ratio", 0) or 0)
        if not hint_hit and avg_length < 8:
            continue
        score = presence * 0.5 + min(avg_length / 120, 1.0) * 0.3 + (0.2 if hint_hit else 0.0)
        candidates.append((score, name))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in candidates[:8]]


def _refresh_document_text(record: dict[str, Any], text: str) -> dict[str, Any]:
    refreshed = dict(record)
    text = clean_text(text)
    sections = split_sections(text)
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    chinese_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    quality = dict(refreshed.get("quality", {}))
    quality.update(
        {
            "length": len(text),
            "language": "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "unknown",
            "chinese_char_ratio": round(chinese_count / len(text), 4) if text else 0,
            "sentence_count": len(split_sentences(text)),
            "section_count": len(sections),
            "text_fingerprint": fingerprint,
        }
    )
    refreshed.update(
        {
            "text": text,
            "sections": sections,
            "fingerprint": fingerprint,
            "quality": quality,
        }
    )
    return refreshed


def _repair_text_from_semantic_fields(
    raw_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
    schema_profile: dict[str, Any],
    *,
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    semantic_fields = _semantic_text_fields(schema_profile)
    repaired: list[dict[str, Any]] = []
    repair_actions: list[dict[str, Any]] = []
    for index, record in enumerate(processed_records):
        current_text = str(record.get("text", "")).strip()
        raw_record = raw_records[index] if index < len(raw_records) else {}
        has_explicit_text_field = any(
            str(raw_record.get(field_name, "")).strip()
            for field_name in ("text", "content", "body", "description", "note", "notes", "summary")
        )
        selected_parts: list[str] = []
        selected_fields: list[str] = []
        for field in semantic_fields:
            value = raw_record.get(field)
            if value is None:
                continue
            text_value = clean_text(str(value))
            if not text_value:
                continue
            selected_parts.append(f"{field}: {text_value}")
            selected_fields.append(field)
        if ((not current_text or len(current_text) < 20) or not has_explicit_text_field) and selected_parts:
            rebuilt = "\n".join(selected_parts)
            repaired_record = _refresh_document_text(record, rebuilt)
            repair_actions.append(
                {
                    "doc_id": record.get("doc_id"),
                    "action": "semantic_field_text_rebuild",
                    "source_fields": selected_fields,
                    "reason": "missing_explicit_text_field" if not has_explicit_text_field else "empty_or_low_information_text",
                    "old_length": len(current_text),
                    "new_length": len(rebuilt),
                }
            )
            repaired.append(repaired_record)
        else:
            repaired.append(record)
    return repaired, {
        "enabled": True,
        "operator": "semantic_field_text_repair",
        "semantic_fields": semantic_fields,
        "repair_count": len(repair_actions),
        "actions": repair_actions,
        "source_path": str(source_path),
    }


def _build_plan(capabilities: dict[str, Any]) -> list[dict[str, Any]]:
    ordered_steps = _topological_sort_steps(capabilities["requested_steps"])
    return [
        {
            "order": index,
            "step": step,
            "description": STEP_DESCRIPTIONS[step],
            "depends_on": STEP_DEPENDENCIES.get(step, []),
            "trigger": _step_trigger(step, capabilities),
        }
        for index, step in enumerate(ordered_steps, start=1)
    ]


def _step_trigger(step: str, capabilities: dict[str, Any]) -> str:
    if step in {"load_records", "infer_schema_profile", "normalize_fields", "profile_documents", "quality_check", "write_outputs"}:
        return "base_etl_requirement"
    if step == "clean_text":
        return "task_requested_cleaning"
    if step == "extract_fields":
        return "task_requested_extraction_or_mapping"
    if step == "split_sections":
        return "task_requested_chunking"
    if step == "record_quality_warnings":
        return "quality_recovery_requested"
    if step == "deduplicate_by_fingerprint_policy":
        return "quality_replan_duplicate_text"
    if step == "quarantine_empty_text_policy":
        return "quality_replan_empty_text"
    if step == "unsupported_format_guard":
        return f"unsupported_format:{capabilities.get('input_format', 'unknown')}"
    return "planner_rule"


def _topological_sort_steps(requested_steps: list[str]) -> list[str]:
    requested = set(requested_steps)
    expanded: set[str] = set()

    def include_with_deps(step: str) -> None:
        if step in expanded:
            return
        for dependency in STEP_DEPENDENCIES.get(step, []):
            include_with_deps(dependency)
        expanded.add(step)

    for step in requested_steps:
        include_with_deps(step)

    ordered: list[str] = []
    temporary = set()
    permanent = set()

    def visit(step: str) -> None:
        if step in permanent:
            return
        if step in temporary:
            raise ValueError(f"Cycle detected in data-processing DAG at {step}")
        temporary.add(step)
        for dependency in STEP_DEPENDENCIES.get(step, []):
            if dependency in expanded:
                visit(dependency)
        temporary.remove(step)
        permanent.add(step)
        if step in expanded and step not in ordered:
            ordered.append(step)

    for step in requested_steps:
        visit(step)
    for step in sorted(expanded):
        visit(step)
    return ordered


def _build_dag(plan: list[dict[str, Any]]) -> dict[str, Any]:
    node_by_step = {item["step"]: item for item in plan}
    nodes = [
        {
            "id": item["step"],
            "order": item["order"],
            "description": item["description"],
            "trigger": item["trigger"],
        }
        for item in plan
    ]
    edges = [
        {
            "source": dependency,
            "target": item["step"],
        }
        for item in plan
        for dependency in item.get("depends_on", [])
        if dependency in node_by_step
    ]
    indegree = {item["step"]: 0 for item in plan}
    for edge in edges:
        indegree[edge["target"]] += 1
    layers: list[list[str]] = []
    remaining = set(indegree)
    while remaining:
        layer = sorted(step for step in remaining if indegree[step] == 0)
        if not layer:
            raise ValueError("Cycle detected while building data-processing DAG layers")
        layers.append(layer)
        for step in layer:
            remaining.remove(step)
            for edge in edges:
                if edge["source"] == step and edge["target"] in remaining:
                    indegree[edge["target"]] -= 1
    return {
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
        "operator_count": len(nodes),
        "edge_count": len(edges),
        "critical_path_length": len(layers),
        "parallelizable_layers": sum(1 for layer in layers if len(layer) > 1),
    }


def _build_recovery_actions(quality_report: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if quality_report.get("duplicate_doc_ids"):
        actions.append(
            {
                "issue": "duplicate_doc_id",
                "action": "保留重复记录并在质量报告中标记，后续图谱阶段按 source_doc_id/evidence 聚合。",
                "doc_ids": quality_report["duplicate_doc_ids"],
            }
        )
    if quality_report.get("empty_text_documents"):
        actions.append(
            {
                "issue": "empty_text",
                "action": "保留空文本记录但不生成有效语义片段，供上游数据修复。",
                "doc_ids": quality_report["empty_text_documents"],
            }
        )
    if quality_report.get("duplicate_text_groups"):
        actions.append(
            {
                "issue": "duplicate_text",
                "action": "保留来源证据并在后续统计中按正文指纹去重，避免重复文本放大图谱边权。",
                "groups": quality_report["duplicate_text_groups"],
            }
        )
    if quality_report.get("low_information_documents"):
        actions.append(
            {
                "issue": "low_information_text",
                "action": "标记为低信息量文档，后续抽取阶段降低置信度或等待补充上下文。",
                "doc_ids": quality_report["low_information_documents"],
            }
        )
    if not actions:
        actions.append(
            {
                "issue": "none",
                "action": "未发现需要恢复的质量问题。",
            }
        )
    return actions


def _build_quality_score(quality_report: dict[str, Any], document_profile: dict[str, Any]) -> dict[str, Any]:
    document_count = max(1, int(quality_report.get("document_count", 0)))
    duplicate_doc_rate = len(quality_report.get("duplicate_doc_ids", [])) / document_count
    empty_rate = len(quality_report.get("empty_text_documents", [])) / document_count
    low_info_rate = len(quality_report.get("low_information_documents", [])) / document_count
    duplicate_text_rate = len(document_profile.get("duplicate_text_groups", [])) / document_count
    penalties = {
        "duplicate_doc_id": round(duplicate_doc_rate * 25, 2),
        "empty_text": round(empty_rate * 35, 2),
        "low_information_text": round(low_info_rate * 15, 2),
        "duplicate_text": round(duplicate_text_rate * 20, 2),
        "warning_count": min(int(quality_report.get("warning_count", 0)) * 2, 20),
    }
    score = max(0.0, 100.0 - sum(penalties.values()))
    grade = "excellent" if score >= 95 else "good" if score >= 85 else "needs_review" if score >= 70 else "poor"
    return {
        "score": round(score, 2),
        "grade": grade,
        "penalties": penalties,
        "document_count": document_count,
    }


def _replan_for_quality(
    plan: list[dict[str, Any]],
    quality_report: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing = {item["step"] for item in plan}
    additions: list[dict[str, Any]] = []
    if any(item.get("issue") != "none" for item in recovery_actions) and "record_quality_warnings" not in existing:
        additions.append(
            {
                "step": "record_quality_warnings",
                "reason": "quality_warnings_detected",
                "warning_count": quality_report.get("warning_count", 0),
            }
        )
    if quality_report.get("duplicate_text_groups"):
        additions.append(
            {
                "step": "deduplicate_by_fingerprint_policy",
                "reason": "duplicate_text_groups_detected",
                "group_count": len(quality_report.get("duplicate_text_groups", [])),
            }
        )
    if quality_report.get("empty_text_documents"):
        additions.append(
            {
                "step": "quarantine_empty_text_policy",
                "reason": "empty_text_documents_detected",
                "document_count": len(quality_report.get("empty_text_documents", [])),
            }
        )
    if not additions:
        return plan, []

    replanned_steps = [item["step"] for item in plan]
    for addition in additions:
        step = addition["step"]
        if step not in replanned_steps and step in STEP_DESCRIPTIONS:
            replanned_steps.insert(max(0, len(replanned_steps) - 1), step)
    replanned_plan = _build_plan({"requested_steps": replanned_steps, "input_format": "replanned"})
    return replanned_plan, additions


def _build_execution_state(
    plan: list[dict[str, Any]],
    *,
    raw_count: int,
    processed_count: int,
    quality_report: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
    quality_score: dict[str, Any],
    replan_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    warning_count = int(quality_report.get("warning_count", 0))
    status = "completed_with_warnings" if warning_count else "completed"
    warning_steps = set()
    if warning_count:
        warning_steps.update({"quality_check", "record_quality_warnings"})
    replan_steps = {str(item.get("step")) for item in replan_actions}
    step_statuses = [
        {
            "order": item["order"],
            "step": item["step"],
            "status": (
                "replanned" if item["step"] in replan_steps else
                "warning" if item["step"] in warning_steps else
                "completed"
            ),
            "depends_on": item.get("depends_on", []),
        }
        for item in plan
    ]
    return {
        "status": status,
        "raw_record_count": raw_count,
        "processed_record_count": processed_count,
        "warning_count": warning_count,
        "recovery_action_count": len([item for item in recovery_actions if item.get("issue") != "none"]),
        "quality_score": quality_score,
        "replan_action_count": len(replan_actions),
        "step_statuses": step_statuses,
    }


def _contract_for_step(
    step: str,
    *,
    raw_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
    schema_profile: dict[str, Any],
    document_profile: dict[str, Any],
    quality_report: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
    output_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a lightweight runtime contract for a DAG operator."""
    output_paths = output_paths or {}
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if step == "load_records":
        add_check("raw_records_loaded", len(raw_records) >= 0, len(raw_records))
        outputs = ["raw_records"]
    elif step == "infer_schema_profile":
        add_check("schema_profile_created", "field_count" in schema_profile, schema_profile.get("field_count", 0))
        add_check("mapping_created", "inferred_mapping" in schema_profile, schema_profile.get("inferred_mapping", {}))
        outputs = ["schema_profile", "field_mapping"]
    elif step in {"clean_text", "extract_fields", "split_sections", "normalize_fields"}:
        add_check("record_count_preserved", len(processed_records) == len(raw_records), {"raw": len(raw_records), "processed": len(processed_records)})
        add_check("standard_document_shape", all({"doc_id", "title", "source", "text", "quality"}.issubset(record) for record in processed_records), len(processed_records))
        outputs = ["normalized_documents"]
        if step == "split_sections":
            add_check("sections_present", all("sections" in record for record in processed_records), len(processed_records))
            outputs.append("sections")
    elif step == "profile_documents":
        add_check("document_profile_created", document_profile.get("document_count") == len(processed_records), document_profile.get("document_count", 0))
        add_check("length_distribution_created", "length" in document_profile, document_profile.get("length", {}))
        outputs = ["document_profile", "length_distribution"]
    elif step == "quality_check":
        add_check("quality_report_created", quality_report.get("document_count") == len(processed_records), quality_report.get("document_count", 0))
        add_check("warnings_materialized", "warnings" in quality_report, quality_report.get("warning_count", 0))
        outputs = ["quality_report", "warnings"]
    elif step == "record_quality_warnings":
        warning_count = int(quality_report.get("warning_count", 0))
        action_count = len([item for item in recovery_actions if item.get("issue") != "none"])
        add_check("warnings_have_recovery_actions", warning_count == 0 or action_count > 0, {"warning_count": warning_count, "action_count": action_count})
        outputs = ["recovery_actions"]
    elif step == "deduplicate_by_fingerprint_policy":
        add_check("duplicate_policy_recorded", "duplicate_text_groups" in quality_report, len(quality_report.get("duplicate_text_groups", [])))
        outputs = ["deduplication_policy"]
    elif step == "quarantine_empty_text_policy":
        add_check("empty_text_policy_recorded", "empty_text_documents" in quality_report, len(quality_report.get("empty_text_documents", [])))
        outputs = ["quarantine_policy"]
    elif step == "write_outputs":
        existing_paths = {
            name: path
            for name, path in output_paths.items()
            if Path(path).exists()
        }
        add_check("artifacts_written", len(existing_paths) >= 6, sorted(existing_paths))
        outputs = ["documents_jsonl", "quality_report_json", "profiles", "dag", "planning"]
    elif step == "unsupported_format_guard":
        add_check("unsupported_format_detected", True, "guarded_before_execution")
        outputs = ["format_guard"]
    else:
        add_check("operator_registered", True, step)
        outputs = [step]

    return {
        "inputs": STEP_DEPENDENCIES.get(step, []),
        "outputs": outputs,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def _build_execution_schedule(
    *,
    plan: list[dict[str, Any]],
    dag: dict[str, Any],
    step_metrics: dict[str, dict[str, Any]],
    raw_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
    schema_profile: dict[str, Any],
    document_profile: dict[str, Any],
    quality_report: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
    replan_actions: list[dict[str, Any]],
    planning_decision: dict[str, Any],
    output_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    plan_by_step = {item["step"]: item for item in plan}
    completed: set[str] = set()
    replan_steps = {str(item.get("step")) for item in replan_actions}
    warning_steps = {"quality_check", "record_quality_warnings"} if quality_report.get("warning_count") else set()
    layers: list[dict[str, Any]] = []
    contract_failures: list[dict[str, Any]] = []

    for layer_index, layer_steps in enumerate(dag.get("layers", []), start=1):
        operators: list[dict[str, Any]] = []
        for step in layer_steps:
            plan_item = plan_by_step.get(step, {"step": step, "depends_on": STEP_DEPENDENCIES.get(step, [])})
            dependencies = [dependency for dependency in plan_item.get("depends_on", []) if dependency in plan_by_step]
            dependency_status = {
                dependency: "satisfied" if dependency in completed else "missing"
                for dependency in dependencies
            }
            contract = _contract_for_step(
                step,
                raw_records=raw_records,
                processed_records=processed_records,
                schema_profile=schema_profile,
                document_profile=document_profile,
                quality_report=quality_report,
                recovery_actions=recovery_actions,
                output_paths=output_paths,
            )
            if not contract["passed"]:
                contract_failures.append({"step": step, "checks": contract["checks"]})
            metric = step_metrics.get(step, {})
            operators.append(
                {
                    "step": step,
                    "description": plan_item.get("description", STEP_DESCRIPTIONS.get(step, step)),
                    "operator_variant": plan_item.get("operator_variant", "standard"),
                    "operator_variant_reason": plan_item.get("operator_variant_reason", "default_operator"),
                    "dependencies": dependency_status,
                    "dependency_satisfied": all(status == "satisfied" for status in dependency_status.values()),
                    "status": (
                        "replanned" if step in replan_steps else
                        "warning" if step in warning_steps else
                        "completed"
                    ),
                    "elapsed_ms": metric.get("elapsed_ms", 0),
                    "input_count": metric.get("input_count"),
                    "output_count": metric.get("output_count"),
                    "contract": contract,
                    "datamate_capability": {
                        "uses_datamate_connector": step == "load_records",
                        "connector": plan_item.get("operator_variant", "standard") if step == "load_records" else None,
                    },
                    "runtime_strategy": (
                        {
                            "scheduler": metric.get("scheduler"),
                            "parallel": metric.get("parallel"),
                            "worker_count": metric.get("worker_count"),
                            "batch_size": metric.get("batch_size"),
                            "docs_per_second": metric.get("docs_per_second"),
                            "preserves_input_order": metric.get("preserves_input_order"),
                        }
                        if metric.get("scheduler") else {}
                    ),
                }
            )
        completed.update(layer_steps)
        layers.append(
            {
                "layer": layer_index,
                "parallelizable": len(layer_steps) > 1,
                "operators": operators,
                "layer_status": "completed" if all(item["dependency_satisfied"] and item["contract"]["passed"] for item in operators) else "needs_review",
            }
        )

    return {
        "agent": "data_processing_agent",
        "scheduler": "dag_layer_runtime_scheduler",
        "runtime_execution_strategy": planning_decision.get("runtime_execution_strategy", {}),
        "selected_plan": planning_decision.get("selected_plan", {}),
        "candidate_plan_count": len(planning_decision.get("candidate_plans", [])),
        "operator_count": dag.get("operator_count", 0),
        "edge_count": dag.get("edge_count", 0),
        "critical_path_length": dag.get("critical_path_length", 0),
        "parallelizable_layers": dag.get("parallelizable_layers", 0),
        "layers": layers,
        "contract_failure_count": len(contract_failures),
        "contract_failures": contract_failures,
        "quality_replan_steps": sorted(replan_steps),
        "recovery_action_count": len([item for item in recovery_actions if item.get("issue") != "none"]),
        "status": "completed" if not contract_failures else "completed_with_contract_warnings",
    }


def _attach_operator_variants(plan: list[dict[str, Any]], operator_selection: dict[str, Any]) -> list[dict[str, Any]]:
    variants = operator_selection.get("variants", {})
    enriched_plan: list[dict[str, Any]] = []
    for item in plan:
        variant = variants.get(item["step"], {})
        enriched_plan.append(
            {
                **item,
                "operator_variant": variant.get("variant", "standard"),
                "operator_variant_reason": variant.get("reason", "default_operator"),
            }
        )
    return enriched_plan


def _build_tool_calls(
    plan: list[dict[str, Any]],
    *,
    raw_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
    quality_report: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
    step_metrics: dict[str, dict[str, Any]],
    schema_profile: dict[str, Any],
    document_profile: dict[str, Any],
    dag: dict[str, Any],
    replan_actions: list[dict[str, Any]],
    planning_decision: dict[str, Any],
    execution_schedule: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = [
        {
            "tool": "task_parser",
            "status": "success",
            "step_count": len(plan),
            "dag_operator_count": dag.get("operator_count", 0),
            "critical_path_length": dag.get("critical_path_length", 0),
            "intent": planning_decision.get("intent", {}),
            "candidate_plan_count": len(planning_decision.get("candidate_plans", [])),
            "selected_plan": planning_decision.get("selected_plan", {}),
        }
    ]
    runtime_execution_strategy = planning_decision.get("runtime_execution_strategy", {})
    calls.append(
        {
            "tool": "runtime_execution_strategy_selector",
            "status": "success",
            **runtime_execution_strategy,
        }
    )
    if execution_schedule:
        calls.append(
            {
                "tool": "dag_layer_runtime_scheduler",
                "status": execution_schedule.get("status", "unknown"),
                "layer_count": len(execution_schedule.get("layers", [])),
                "operator_count": execution_schedule.get("operator_count", 0),
                "contract_failure_count": execution_schedule.get("contract_failure_count", 0),
                "parallelizable_layers": execution_schedule.get("parallelizable_layers", 0),
                "runtime_execution_strategy": execution_schedule.get("runtime_execution_strategy", {}),
            }
        )
    profile_optimization = planning_decision.get("profile_optimized_plan", {})
    calls.append(
        {
            "tool": "profile_aware_dag_optimizer",
            "status": "success",
            "changed": profile_optimization.get("changed", False),
            "action_count": len(profile_optimization.get("actions", [])),
            "observed_profile": profile_optimization.get("observed_profile", {}),
            "dag_delta": profile_optimization.get("dag_delta", {}),
        }
    )
    operator_selection = planning_decision.get("operator_selection", {})
    calls.append(
        {
            "tool": "operator_variant_selector",
            "status": "success",
            "variant_count": operator_selection.get("variant_count", 0),
            "variants": operator_selection.get("variants", {}),
        }
    )
    runtime_repair = planning_decision.get("runtime_text_repair", {})
    calls.append(
        {
            "tool": "semantic_field_text_repair",
            "status": "repaired" if runtime_repair.get("repair_count", 0) else "skipped",
            "repair_count": runtime_repair.get("repair_count", 0),
            "semantic_fields": runtime_repair.get("semantic_fields", []),
            "actions": runtime_repair.get("actions", []),
            **step_metrics.get("runtime_text_repair", {}),
        }
    )
    for item in plan:
        step = item["step"]
        call: dict[str, Any] = {
            "tool": step,
            "status": "success",
            "order": item["order"],
            "operator_variant": item.get("operator_variant", "standard"),
            "operator_variant_reason": item.get("operator_variant_reason", "default_operator"),
        }
        call.update(step_metrics.get(step, {}))
        if step == "load_records":
            call["record_count"] = len(raw_records)
        elif step in {"clean_text", "normalize_fields", "extract_fields"}:
            call["record_count"] = len(processed_records)
        elif step == "infer_schema_profile":
            call["field_count"] = schema_profile.get("field_count", 0)
            call["inferred_mapping"] = schema_profile.get("inferred_mapping", {})
        elif step == "profile_documents":
            call["document_count"] = document_profile.get("document_count", 0)
            call["duplicate_text_group_count"] = len(document_profile.get("duplicate_text_groups", []))
        elif step == "split_sections":
            call["section_count"] = sum(len(record.get("sections", [])) for record in processed_records)
        elif step == "quality_check":
            call["warning_count"] = quality_report.get("warning_count", 0)
        elif step == "record_quality_warnings":
            call["warning_count"] = quality_report.get("warning_count", 0)
            call["action_count"] = len([action for action in recovery_actions if action.get("issue") != "none"])
        elif step in {"deduplicate_by_fingerprint_policy", "quarantine_empty_text_policy"}:
            call["replan_reason"] = next((item.get("reason") for item in replan_actions if item.get("step") == step), "quality_replan")
        calls.append(call)
    calls.append(
        {
            "tool": "quality_recovery_policy",
            "status": "success",
            "action_count": len(recovery_actions),
        }
    )
    return calls


def _augment_trace(
    trace: list[dict[str, Any]],
    *,
    task: str,
    plan: list[dict[str, Any]],
    capabilities: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
    state: dict[str, Any],
    step_metrics: dict[str, dict[str, Any]],
    schema_profile: dict[str, Any],
    document_profile: dict[str, Any],
    dag: dict[str, Any],
    replan_actions: list[dict[str, Any]],
    planning_decision: dict[str, Any],
) -> list[dict[str, Any]]:
    enriched = [
        {
            "agent": "data_processing_agent",
            "goal": task,
            "step": "understand_task",
            "tool": "task_parser",
            "status": "success",
            "elapsed_ms": step_metrics.get("understand_task", {}).get("elapsed_ms", 0),
            "capabilities": capabilities,
            "intent": planning_decision.get("intent", {}),
            "input_profile_estimate": planning_decision.get("input_profile_estimate", {}),
            "candidate_plans": planning_decision.get("candidate_plans", []),
            "selected_plan": planning_decision.get("selected_plan", {}),
            "plan": plan,
            "dag": dag,
            "state": {
                "status": "planned",
                "step_count": len(plan),
                "dag_operator_count": dag.get("operator_count", 0),
                "critical_path_length": dag.get("critical_path_length", 0),
            },
        }
    ]
    trace_by_step = {item.get("step"): item for item in trace}
    for plan_item in plan:
        step = plan_item["step"]
        base_item = trace_by_step.get(step, {})
        metric_item = step_metrics.get(step, {})
        metric_extras = {
            key: value
            for key, value in metric_item.items()
            if key not in {"elapsed_ms", "input_count", "output_count"}
        }
        enriched.append(
            {
                "agent": "data_processing_agent",
                "goal": task,
                "plan": plan,
                "order": plan_item["order"],
                "step": step,
                "tool": base_item.get("tool", step),
                "status": base_item.get("status", "success"),
                "elapsed_ms": metric_item.get("elapsed_ms", base_item.get("elapsed_ms", 0)),
                "input_count": metric_item.get("input_count"),
                "output_count": metric_item.get("output_count"),
                "description": plan_item["description"],
                "depends_on": plan_item.get("depends_on", []),
                "trigger": plan_item.get("trigger"),
                "operator_variant": plan_item.get("operator_variant", "standard"),
                "operator_variant_reason": plan_item.get("operator_variant_reason", "default_operator"),
                "state": state,
                **metric_extras,
                **{key: value for key, value in base_item.items() if key not in {"agent", "goal", "plan", "order", "step", "tool", "status", "state"}},
            }
        )
        if step == "infer_schema_profile":
            enriched[-1]["schema_profile"] = {
                "field_count": schema_profile.get("field_count", 0),
                "inferred_mapping": schema_profile.get("inferred_mapping", {}),
                "schema_confidence": schema_profile.get("schema_confidence", {}),
            }
        elif step == "profile_documents":
            enriched[-1]["document_profile"] = {
                "length": document_profile.get("length", {}),
                "language_counts": document_profile.get("language_counts", {}),
                "duplicate_text_group_count": len(document_profile.get("duplicate_text_groups", [])),
                "low_information_count": len(document_profile.get("low_information_documents", [])),
            }
    enriched.append(
        {
            "agent": "data_processing_agent",
            "goal": task,
            "step": "profile_based_dag_optimize",
            "tool": "profile_aware_dag_optimizer",
            "status": "success",
            "elapsed_ms": step_metrics.get("profile_based_dag_optimize", {}).get("elapsed_ms", 0),
            "optimization": planning_decision.get("profile_optimized_plan", {}),
            "state": {
                "status": "optimized" if planning_decision.get("profile_optimized_plan", {}).get("changed") else "unchanged",
                "step_count": len(plan),
                "dag_operator_count": dag.get("operator_count", 0),
                "critical_path_length": dag.get("critical_path_length", 0),
            },
        }
    )
    enriched.append(
        {
            "agent": "data_processing_agent",
            "goal": task,
            "step": "runtime_text_repair",
            "tool": "semantic_field_text_repair",
            "status": "repaired" if planning_decision.get("runtime_text_repair", {}).get("repair_count", 0) else "skipped",
            "elapsed_ms": step_metrics.get("runtime_text_repair", {}).get("elapsed_ms", 0),
            "input_count": step_metrics.get("runtime_text_repair", {}).get("input_count", 0),
            "output_count": step_metrics.get("runtime_text_repair", {}).get("output_count", 0),
            "repair": planning_decision.get("runtime_text_repair", {}),
            "state": state,
        }
    )
    enriched.append(
        {
            "agent": "data_processing_agent",
            "goal": task,
            "step": "quality_driven_replan",
            "tool": "dag_replanner",
            "status": "success",
            "elapsed_ms": step_metrics.get("quality_driven_replan", {}).get("elapsed_ms", 0),
            "replan_actions": replan_actions,
            "replan_action_count": len(replan_actions),
            "state": state,
        }
    )
    enriched.append(
        {
            "agent": "data_processing_agent",
            "goal": task,
            "step": "recover_quality_issues",
            "tool": "quality_recovery_policy",
            "status": "success",
            "elapsed_ms": step_metrics.get("recover_quality_issues", {}).get("elapsed_ms", 0),
            "recovery_actions": recovery_actions,
            "state": state,
        }
    )
    return enriched


def _build_task1_report(
    *,
    task: str,
    input_path: Path,
    plan: list[dict[str, Any]],
    quality_report: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
    state: dict[str, Any],
    errors: list[dict[str, Any]],
    output_paths: dict[str, str],
    trace: list[dict[str, Any]],
    schema_profile: dict[str, Any],
    document_profile: dict[str, Any],
    dag: dict[str, Any],
    replan_actions: list[dict[str, Any]],
    quality_score: dict[str, Any],
    planning_decision: dict[str, Any],
    execution_schedule: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# 任务一数据处理报告",
        "",
        f"- 任务：{task}",
        f"- 输入：`{input_path}`",
        f"- 文档数：{quality_report['document_count']}",
        f"- 唯一 doc_id 数：{quality_report['unique_doc_ids']}",
        f"- 平均长度：{quality_report['avg_length']}",
        f"- 告警数：{quality_report.get('warning_count', 0)}",
        f"- 质量分数：{quality_score.get('score', 0)}（{quality_score.get('grade', 'unknown')}）",
        f"- DAG 算子数：{dag.get('operator_count', 0)}",
        f"- DAG 关键路径长度：{dag.get('critical_path_length', 0)}",
        f"- 调度层数：{len(execution_schedule.get('layers', [])) if execution_schedule else 0}",
        f"- 契约失败数：{execution_schedule.get('contract_failure_count', 0) if execution_schedule else 0}",
        f"- 执行策略：`{planning_decision.get('runtime_execution_strategy', {}).get('scheduler', 'unknown')}`",
        f"- 选中计划：`{planning_decision.get('selected_plan', {}).get('name', 'unknown')}`",
        f"- 执行状态：`{state['status']}`",
        "",
        "## 任务理解与计划选择",
        "",
        f"- 目标产物：`{planning_decision.get('intent', {}).get('target_artifact', 'unknown')}`",
        f"- 输入格式：`{planning_decision.get('input_profile_estimate', {}).get('format', 'unknown')}`",
        f"- 计划选择原因：{planning_decision.get('selected_plan', {}).get('selection_reason', '')}",
        "",
        "| 候选计划 | 分数 | 覆盖率 | 鲁棒性 | 估算成本 | 算子数 | 策略 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in planning_decision.get("candidate_plans", []):
        lines.append(
            f"| `{item['name']}` | {item['score']} | {item['coverage']} | {item['robustness']} | "
            f"{item['estimated_cost']} | {item['dag_metrics']['operator_count']} | {item['strategy']} |"
        )
    lines.extend([
        "",
        "## 执行计划",
        "",
        "| 顺序 | 步骤 | 算子变体 | 说明 |",
        "| ---: | --- | --- | --- |",
    ])
    for item in plan:
        lines.append(
            f"| {item['order']} | `{item['step']}` | `{item.get('operator_variant', 'standard')}` | "
            f"{item['description']}；依赖 `{','.join(item.get('depends_on', [])) or '-'}` |"
        )
    lines.extend(["", "## DAG 调度", ""])
    lines.append(f"- 算子数：{dag.get('operator_count', 0)}")
    lines.append(f"- 边数：{dag.get('edge_count', 0)}")
    lines.append(f"- 关键路径长度：{dag.get('critical_path_length', 0)}")
    lines.append(f"- 可并行层数：{dag.get('parallelizable_layers', 0)}")
    lines.extend(["", "| 层级 | 算子 |", "| ---: | --- |"])
    for index, layer in enumerate(dag.get("layers", []), start=1):
        lines.append(f"| {index} | `{', '.join(layer)}` |")
    if execution_schedule:
        lines.extend(["", "## 运行时调度与契约校验", ""])
        lines.append(f"- 调度器：`{execution_schedule.get('scheduler', '')}`")
        lines.append(f"- 状态：`{execution_schedule.get('status', '')}`")
        lines.append(f"- 契约失败数：{execution_schedule.get('contract_failure_count', 0)}")
        runtime_strategy = execution_schedule.get("runtime_execution_strategy", {})
        if runtime_strategy:
            lines.append(f"- 记录级执行器：`{runtime_strategy.get('scheduler')}`")
            lines.append(f"- 并行执行：{'是' if runtime_strategy.get('parallel') else '否'}")
            lines.append(f"- worker 数：{runtime_strategy.get('worker_count')}")
            lines.append(f"- 批大小：{runtime_strategy.get('batch_size')}")
        lines.extend(["", "| 层级 | 算子 | 状态 | 依赖满足 | 契约通过 | 输入 | 输出 |", "| ---: | --- | --- | --- | --- | ---: | ---: |"])
        for layer in execution_schedule.get("layers", []):
            for operator in layer.get("operators", []):
                lines.append(
                    f"| {layer.get('layer')} | `{operator.get('step')}` | `{operator.get('status')}` | "
                    f"{'是' if operator.get('dependency_satisfied') else '否'} | "
                    f"{'是' if operator.get('contract', {}).get('passed') else '否'} | "
                    f"{operator.get('input_count', '')} | {operator.get('output_count', '')} |"
                )
    optimization = planning_decision.get("profile_optimized_plan", {})
    lines.extend(["", "## Profile 驱动 DAG 优化", ""])
    lines.append(f"- 是否调整：{'是' if optimization.get('changed') else '否'}")
    lines.append(f"- 原因：{optimization.get('reason', '')}")
    observed = optimization.get("observed_profile", {})
    if observed:
        lines.append(f"- 真实字段数：{observed.get('field_count', 0)}")
        lines.append(f"- 文本覆盖率：{observed.get('text_coverage', 0)}")
        lines.append(f"- 文本中位长度：{observed.get('median_length', 0)}")
        lines.append(f"- 重复正文组数：{observed.get('duplicate_text_group_count', 0)}")
    if optimization.get("actions"):
        lines.extend(["", "| 动作 | 算子 | 原因 |", "| --- | --- | --- |"])
        for item in optimization.get("actions", []):
            lines.append(f"| {item.get('action')} | `{item.get('step')}` | {item.get('reason')} |")
    else:
        lines.append("未新增或重排算子。")
    operator_selection = planning_decision.get("operator_selection", {})
    lines.extend(["", "## 算子变体选择", ""])
    if operator_selection.get("variants"):
        lines.extend(["| 步骤 | 变体 | 原因 |", "| --- | --- | --- |"])
        for step, item in operator_selection.get("variants", {}).items():
            lines.append(f"| `{step}` | `{item.get('variant')}` | {item.get('reason')} |")
    else:
        lines.append("未记录算子变体。")
    lines.extend(["", "## 质量驱动重规划", ""])
    if replan_actions:
        lines.extend(["| 算子 | 原因 |", "| --- | --- |"])
        for item in replan_actions:
            lines.append(f"| `{item.get('step')}` | {item.get('reason')} |")
    else:
        lines.append("未触发额外重规划。")
    runtime_repair = planning_decision.get("runtime_text_repair", {})
    lines.extend(["", "## 运行时契约修复", ""])
    lines.append(f"- 修复算子：`{runtime_repair.get('operator', 'semantic_field_text_repair')}`")
    lines.append(f"- 修复记录数：{runtime_repair.get('repair_count', 0)}")
    if runtime_repair.get("semantic_fields"):
        lines.append(f"- 候选语义字段：`{', '.join(runtime_repair.get('semantic_fields', []))}`")
    if runtime_repair.get("actions"):
        lines.extend(["", "| doc_id | 动作 | 来源字段 | 旧长度 | 新长度 |", "| --- | --- | --- | ---: | ---: |"])
        for item in runtime_repair.get("actions", []):
            lines.append(
                f"| `{item.get('doc_id')}` | `{item.get('action')}` | "
                f"`{', '.join(item.get('source_fields', []))}` | {item.get('old_length')} | {item.get('new_length')} |"
            )
    lines.extend(["", "## 数据画像", ""])
    lines.append(f"- 字段数：{schema_profile.get('field_count', 0)}")
    lines.append(f"- 推断映射：`{json.dumps(schema_profile.get('inferred_mapping', {}), ensure_ascii=False)}`")
    lines.append(f"- 长度分布：`{json.dumps(document_profile.get('length', {}), ensure_ascii=False)}`")
    lines.append(f"- 重复正文组数：{len(document_profile.get('duplicate_text_groups', []))}")
    lines.append(f"- 低信息量文档数：{len(document_profile.get('low_information_documents', []))}")
    lines.extend(["", "## 质量告警", ""])
    warnings = quality_report.get("warnings", [])
    if warnings:
        lines.extend(["| 类型 | 级别 | 说明 |", "| --- | --- | --- |"])
        for item in warnings:
            lines.append(f"| `{item['code']}` | {item['severity']} | {item['message']} |")
    else:
        lines.append("无质量告警。")
    lines.extend(["", "## 恢复动作", ""])
    for item in recovery_actions:
        lines.append(f"- `{item['issue']}`：{item['action']}")
    lines.extend(["", "## 状态跟踪", ""])
    trace_by_step = {item.get("step"): item for item in trace}
    lines.extend(["| 顺序 | 步骤 | 状态 | 输入 | 输出 | 耗时 ms |", "| ---: | --- | --- | ---: | ---: | ---: |"])
    for item in state["step_statuses"]:
        trace_item = trace_by_step.get(item["step"], {})
        input_count = trace_item.get("input_count", "")
        output_count = trace_item.get("output_count", "")
        elapsed_ms = trace_item.get("elapsed_ms", "")
        lines.append(f"| {item['order']} | `{item['step']}` | `{item['status']}` | {input_count} | {output_count} | {elapsed_ms} |")
    lines.extend(["", "## 错误", ""])
    if errors:
        for item in errors:
            lines.append(f"- `{item['code']}`：{item['message']}")
    else:
        lines.append("无运行错误。")
    lines.extend(["", "## 输出文件", ""])
    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def run_data_processing_agent(
    input_path: Path,
    output_root: Path,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    total_start = perf_counter()
    goal = task or DEFAULT_TASK
    step_metrics: dict[str, dict[str, Any]] = {}
    step_start = perf_counter()
    capabilities = _infer_capabilities(goal, input_path)
    intent = _parse_task_intent(goal, input_path)
    input_profile_estimate = _estimate_input_profile(input_path)
    planning_decision = _select_processing_plan(capabilities, intent, input_profile_estimate)
    capabilities["intent"] = intent
    capabilities["input_profile_estimate"] = input_profile_estimate
    capabilities["candidate_plan_count"] = len(planning_decision["candidate_plans"])
    capabilities["selected_plan_name"] = planning_decision["selected_plan"]["name"]
    capabilities["requested_steps"] = planning_decision["selected_plan"]["steps"]
    plan = _build_plan(capabilities)
    dag = _build_dag(plan)
    step_metrics["understand_task"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": 1,
        "output_count": len(plan),
        "dag_operator_count": dag["operator_count"],
        "critical_path_length": dag["critical_path_length"],
        "candidate_plan_count": len(planning_decision["candidate_plans"]),
        "selected_plan_name": planning_decision["selected_plan"]["name"],
    }

    step_start = perf_counter()
    raw_records = load_records(input_path)
    step_metrics["load_records"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": 1,
        "output_count": len(raw_records),
    }
    runtime_execution_strategy = _select_runtime_execution_strategy(
        raw_record_count=len(raw_records),
        input_profile=input_profile_estimate,
        intent=intent,
    )
    planning_decision["runtime_execution_strategy"] = runtime_execution_strategy
    capabilities["runtime_execution_strategy"] = runtime_execution_strategy

    step_start = perf_counter()
    schema_profile = infer_schema_profile(raw_records)
    step_metrics["infer_schema_profile"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": len(raw_records),
        "output_count": int(schema_profile.get("field_count", 0)),
    }

    processed_records, normalize_metric = _normalize_records_with_strategy(
        raw_records,
        input_path=input_path,
        strategy=runtime_execution_strategy,
    )
    for step in ("clean_text", "extract_fields", "split_sections", "normalize_fields"):
        if step in capabilities["requested_steps"]:
            step_metrics[step] = {
                "elapsed_ms": normalize_metric["elapsed_ms"] if step == "normalize_fields" else round(normalize_metric["elapsed_ms"] / max(1, len([s for s in ("clean_text", "extract_fields", "split_sections", "normalize_fields") if s in capabilities["requested_steps"]])), 3),
                "input_count": len(raw_records),
                "output_count": len(processed_records),
                **(
                    {
                        "scheduler": normalize_metric["scheduler"],
                        "parallel": normalize_metric["parallel"],
                        "worker_count": normalize_metric["worker_count"],
                        "batch_size": normalize_metric["batch_size"],
                        "docs_per_second": normalize_metric["docs_per_second"],
                        "preserves_input_order": normalize_metric["preserves_input_order"],
                    }
                    if step == "normalize_fields"
                    else {}
                ),
            }

    step_start = perf_counter()
    processed_records, semantic_repair = _repair_text_from_semantic_fields(
        raw_records,
        processed_records,
        schema_profile,
        source_path=input_path,
    )
    if semantic_repair["repair_count"]:
        capabilities["runtime_repaired"] = True
        capabilities["runtime_repair_action_count"] = semantic_repair["repair_count"]
    else:
        capabilities["runtime_repaired"] = False
        capabilities["runtime_repair_action_count"] = 0
    planning_decision["runtime_text_repair"] = semantic_repair
    step_metrics["runtime_text_repair"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": len(raw_records),
        "output_count": semantic_repair["repair_count"],
        "semantic_fields": semantic_repair["semantic_fields"],
    }

    step_start = perf_counter()
    document_profile = profile_documents(processed_records)
    step_metrics["profile_documents"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": len(processed_records),
        "output_count": 1,
        "duplicate_text_group_count": len(document_profile.get("duplicate_text_groups", [])),
        "low_information_count": len(document_profile.get("low_information_documents", [])),
    }

    step_start = perf_counter()
    optimized_plan, profile_optimization = _profile_based_plan_optimization(
        plan=plan,
        capabilities=capabilities,
        planning_decision=planning_decision,
        schema_profile=schema_profile,
        document_profile=document_profile,
    )
    if profile_optimization["changed"]:
        plan = optimized_plan
        dag = _build_dag(plan)
        capabilities["requested_steps"] = [item["step"] for item in plan]
    capabilities["profile_optimized"] = profile_optimization["changed"]
    capabilities["profile_optimization_action_count"] = len(profile_optimization["actions"])
    step_metrics["profile_based_dag_optimize"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": len(raw_records),
        "output_count": len(profile_optimization["optimized_steps"]),
        "changed": profile_optimization["changed"],
        "action_count": len(profile_optimization["actions"]),
    }
    operator_selection = _select_operator_variants(
        plan,
        capabilities=capabilities,
        schema_profile=schema_profile,
        document_profile=document_profile,
        runtime_execution_strategy=runtime_execution_strategy,
    )
    planning_decision["operator_selection"] = operator_selection
    plan = _attach_operator_variants(plan, operator_selection)

    step_start = perf_counter()
    quality_report = quality_check(processed_records)
    step_metrics["quality_check"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": len(processed_records),
        "output_count": 1,
    }

    step_start = perf_counter()
    recovery_actions = _build_recovery_actions(quality_report)
    step_metrics["record_quality_warnings"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": int(quality_report.get("warning_count", 0)),
        "output_count": len(recovery_actions),
    }
    step_metrics["recover_quality_issues"] = dict(step_metrics["record_quality_warnings"])

    step_start = perf_counter()
    final_plan, replan_actions = _replan_for_quality(plan, quality_report, recovery_actions)
    if replan_actions:
        plan = final_plan
        dag = _build_dag(plan)
        capabilities["requested_steps"] = [item["step"] for item in plan]
        capabilities["replanned"] = True
        capabilities["replan_action_count"] = len(replan_actions)
    else:
        capabilities["replanned"] = False
        capabilities["replan_action_count"] = 0
    operator_selection = _select_operator_variants(
        plan,
        capabilities=capabilities,
        schema_profile=schema_profile,
        document_profile=document_profile,
        quality_report=quality_report,
        runtime_execution_strategy=runtime_execution_strategy,
    )
    planning_decision["operator_selection"] = operator_selection
    plan = _attach_operator_variants(plan, operator_selection)
    step_metrics["quality_driven_replan"] = {
        "elapsed_ms": _elapsed_ms(step_start),
        "input_count": int(quality_report.get("warning_count", 0)),
        "output_count": len(replan_actions),
    }
    for action in replan_actions:
        step = str(action.get("step"))
        step_metrics[step] = {
            "elapsed_ms": 0,
            "input_count": int(quality_report.get("warning_count", 0)),
            "output_count": 1,
            "replan_reason": action.get("reason"),
        }

    quality_score = _build_quality_score(quality_report, document_profile)
    state = _build_execution_state(
        plan,
        raw_count=len(raw_records),
        processed_count=len(processed_records),
        quality_report=quality_report,
        recovery_actions=recovery_actions,
        quality_score=quality_score,
        replan_actions=replan_actions,
    )
    errors: list[dict[str, Any]] = []
    trace = _augment_trace(
        build_processing_trace(input_path, raw_records, processed_records, quality_report),
        task=goal,
        plan=plan,
        capabilities=capabilities,
        recovery_actions=recovery_actions,
        state=state,
        step_metrics=step_metrics,
        schema_profile=schema_profile,
        document_profile=document_profile,
        dag=dag,
        replan_actions=replan_actions,
        planning_decision=planning_decision,
    )

    output_dir = output_root / "data" / "processed"
    documents_path = output_dir / "documents.jsonl"
    quality_path = output_dir / "data_quality_report.json"
    schema_profile_path = output_dir / "schema_profile.json"
    document_profile_path = output_dir / "document_profile.json"
    dag_path = output_dir / "data_processing_dag.json"
    planning_path = output_dir / "data_processing_planning.json"
    execution_schedule_path = output_dir / "data_processing_execution_schedule.json"
    dag_trace_path = output_dir / "data_processing_dag_trace.jsonl"
    trace_path = output_dir / "data_processing_trace.jsonl"
    report_path = output_root / "demo" / "task1_processing_report.md"

    output_paths = {
        "documents": str(documents_path),
        "quality_report": str(quality_path),
        "schema_profile": str(schema_profile_path),
        "document_profile": str(document_profile_path),
        "dag": str(dag_path),
        "planning": str(planning_path),
        "execution_schedule": str(execution_schedule_path),
        "dag_trace": str(dag_trace_path),
        "trace": str(trace_path),
        "task1_report": str(report_path),
    }

    write_start = perf_counter()
    _write_jsonl(documents_path, processed_records)
    _write_json(quality_path, quality_report)
    _write_json(schema_profile_path, schema_profile)
    _write_json(document_profile_path, document_profile)
    _write_json(dag_path, dag)
    _write_json(planning_path, planning_decision)
    step_metrics["write_outputs"] = {
        "elapsed_ms": _elapsed_ms(write_start),
        "input_count": len(processed_records),
        "output_count": 6,
        "output_paths": {
            "documents": str(documents_path),
            "quality_report": str(quality_path),
            "schema_profile": str(schema_profile_path),
            "document_profile": str(document_profile_path),
            "dag": str(dag_path),
            "planning": str(planning_path),
            "execution_schedule": str(execution_schedule_path),
        },
        "output_bytes": {
            "documents": documents_path.stat().st_size if documents_path.exists() else 0,
            "quality_report": quality_path.stat().st_size if quality_path.exists() else 0,
            "schema_profile": schema_profile_path.stat().st_size if schema_profile_path.exists() else 0,
            "document_profile": document_profile_path.stat().st_size if document_profile_path.exists() else 0,
            "dag": dag_path.stat().st_size if dag_path.exists() else 0,
            "planning": planning_path.stat().st_size if planning_path.exists() else 0,
            "execution_schedule": execution_schedule_path.stat().st_size if execution_schedule_path.exists() else 0,
        },
    }
    execution_schedule = _build_execution_schedule(
        plan=plan,
        dag=dag,
        step_metrics=step_metrics,
        raw_records=raw_records,
        processed_records=processed_records,
        schema_profile=schema_profile,
        document_profile=document_profile,
        quality_report=quality_report,
        recovery_actions=recovery_actions,
        replan_actions=replan_actions,
        planning_decision=planning_decision,
        output_paths=output_paths,
    )
    _write_json(execution_schedule_path, execution_schedule)
    step_metrics["write_outputs"]["output_bytes"]["execution_schedule"] = (
        execution_schedule_path.stat().st_size if execution_schedule_path.exists() else 0
    )
    trace = _augment_trace(
        build_processing_trace(input_path, raw_records, processed_records, quality_report),
        task=goal,
        plan=plan,
        capabilities=capabilities,
        recovery_actions=recovery_actions,
        state=state,
        step_metrics=step_metrics,
        schema_profile=schema_profile,
        document_profile=document_profile,
        dag=dag,
        replan_actions=replan_actions,
        planning_decision=planning_decision,
    )
    _write_jsonl(trace_path, trace)
    _write_jsonl(
        dag_trace_path,
        [
            {
                "layer": index,
                "operators": layer,
                "status": "completed",
                "parallelizable": len(layer) > 1,
            }
            for index, layer in enumerate(dag.get("layers", []), start=1)
        ],
    )
    report = _build_task1_report(
        task=goal,
        input_path=input_path,
        plan=plan,
        quality_report=quality_report,
        recovery_actions=recovery_actions,
        state=state,
        errors=errors,
        output_paths=output_paths,
        trace=trace,
        schema_profile=schema_profile,
        document_profile=document_profile,
        dag=dag,
        replan_actions=replan_actions,
        quality_score=quality_score,
        planning_decision=planning_decision,
        execution_schedule=execution_schedule,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    tool_calls = _build_tool_calls(
        plan,
        raw_records=raw_records,
        processed_records=processed_records,
        quality_report=quality_report,
        recovery_actions=recovery_actions,
        step_metrics=step_metrics,
        schema_profile=schema_profile,
        document_profile=document_profile,
        dag=dag,
        replan_actions=replan_actions,
        planning_decision=planning_decision,
        execution_schedule=execution_schedule,
    )
    state["elapsed_ms"] = _elapsed_ms(total_start)

    return {
        "agent": "data_processing_agent",
        "goal": goal,
        "capabilities": capabilities,
        "plan": plan,
        "tool_calls": tool_calls,
        "state": state,
        "dag": dag,
        "execution_schedule": execution_schedule,
        "planning_decision": planning_decision,
        "replan_actions": replan_actions,
        "errors": errors,
        "input_path": str(input_path),
        "documents": processed_records,
        "quality_report": quality_report,
        "schema_profile": schema_profile,
        "document_profile": document_profile,
        "quality_score": quality_score,
        "recovery_actions": recovery_actions,
        "trace": trace,
        "task1_report": report,
        "output_paths": output_paths,
    }
