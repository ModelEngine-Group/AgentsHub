"""Small learned orchestration policy for Task-1 DataMate processing plans."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any
import warnings


CAPABILITY_LABELS = [
    "pii_cleanup",
    "text_normalization",
    "length_filter",
    "quality_filter",
    "deduplication",
    "schema_guard",
    "low_cost",
]

OPERATOR_LABELS = [
    "clean_email_mapper",
    "clean_links_mapper",
    "clean_ip_mapper",
    "remove_extra_spaces_mapper",
    "text_length_filter",
    "alphanumeric_filter",
    "frequency_specified_field_selector",
    "text_pair_similarity_filter",
]

OPERATOR_ALIAS_LABELS = {
    "average_line_length_filter": "text_length_filter",
    "maximum_line_length_filter": "text_length_filter",
    "minimum_line_length_filter": "text_length_filter",
    "clean_email_mapper": "clean_email_mapper",
    "clean_links_mapper": "clean_links_mapper",
    "clean_ip_mapper": "clean_ip_mapper",
    "remove_extra_spaces_mapper": "remove_extra_spaces_mapper",
    "text_length_filter": "text_length_filter",
    "alphanumeric_filter": "alphanumeric_filter",
    "frequency_specified_field_selector": "frequency_specified_field_selector",
    "specified_field_filter": "frequency_specified_field_selector",
    "general_field_filter": "frequency_specified_field_selector",
    "topk_specified_field_selector": "frequency_specified_field_selector",
    "text_pair_similarity_filter": "text_pair_similarity_filter",
}

KEYWORD_LABELS = [
    "clean",
    "filter",
    "quality",
    "length",
    "dedup",
    "duplicate",
    "schema",
    "field",
    "json",
    "csv",
    "text",
    "space",
    "sentence",
    "remove",
    "normalize",
]

WORKFLOW_LABELS = ["structured_etl", "text_processing_pipeline"]

RANKER_OPERATOR_PROTOTYPES = [
    {
        "id": "clean_email_mapper",
        "name": "邮箱清洗映射器",
        "description": "Cleans email addresses from text samples using a regular expression.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "clean_links_mapper",
        "name": "链接清洗映射器",
        "description": "Cleans URLs and HTTP links from text samples.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "clean_ip_mapper",
        "name": "IP清洗映射器",
        "description": "Cleans IP addresses from text samples.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "remove_extra_spaces_mapper",
        "name": "多余空格清洗映射器",
        "description": "Remove extra spaces and normalize whitespace.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "text_length_filter",
        "name": "文本长度过滤器",
        "description": "Filter to keep samples with total text length within a specific range.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "average_line_length_filter",
        "name": "平均行长过滤器",
        "description": "Filter samples based on average line length.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "maximum_line_length_filter",
        "name": "最大行长过滤器",
        "description": "Filter samples based on maximum line length.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "alphanumeric_filter",
        "name": "字母数字质量过滤器",
        "description": "Quality filter based on alphanumeric ratio.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "frequency_specified_field_selector",
        "name": "频率指定字段选择器",
        "description": "Selector to filter samples based on the frequency of a specified field.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "specified_field_filter",
        "name": "指定字段过滤器",
        "description": "Filter based on specified fields and schema values.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "general_field_filter",
        "name": "通用字段过滤器",
        "description": "General field and schema filter.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "text_pair_similarity_filter",
        "name": "文本对相似度过滤器",
        "description": "Similarity filter for duplicate or near-duplicate text pairs.",
        "inputs": "text",
        "outputs": "text",
    },
    {
        "id": "download_file_mapper",
        "name": "文件下载映射器",
        "description": "Download linked files from text references.",
        "inputs": "text",
        "outputs": "text",
    },
]

STRICT_QUALITY_TERMS = ("严格", "准确", "正确", "高质量", "强校验", "完整", "质量门禁", "可复核", "validate", "quality")
LOW_LATENCY_TERMS = ("快速", "低延迟", "实时", "高吞吐", "性能", "成本", "压缩", "optimize", "performance", "latency")
OFFICIAL_CHAIN_TERMS = ("DataMate", "Nexent", "官方", "真实链路", "不模拟", "不fallback", "不替代", "official")
QUALITY_TERMS = ("质量", "校验", "检查", "异常", "缺失", "错误", "空文本", "去重", "重复", "quality", "validate")
STRUCTURED_TERMS = ("结构化", "表", "字段", "schema", "CSV", "JSON", "JSONL", "ETL", "etl")
TEXT_TERMS = ("文本", "非结构化", "病历", "段落", "notes", "txt")


def _signal_count(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered or term in text)


@dataclass
class Task1PolicyPrediction:
    workflow_type: str
    required_capabilities: list[str]
    operator_keywords: list[str]
    recommended_operator_ids: list[str]
    prefer_low_latency: bool
    strict_quality: bool
    requires_recovery: bool
    quality_priority: str
    cost_tier: str
    avoid_operator_ids: list[str]
    failure_memory_matches: list[dict[str, Any]]
    confidence: float
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": "task1_orchestration_policy_v2_multi_objective",
            "workflow_type": self.workflow_type,
            "required_capabilities": self.required_capabilities,
            "operator_keywords": self.operator_keywords,
            "recommended_operator_ids": self.recommended_operator_ids,
            "prefer_low_latency": self.prefer_low_latency,
            "strict_quality": self.strict_quality,
            "requires_recovery": self.requires_recovery,
            "quality_priority": self.quality_priority,
            "cost_tier": self.cost_tier,
            "avoid_operator_ids": self.avoid_operator_ids,
            "failure_memory_matches": self.failure_memory_matches,
            "confidence": self.confidence,
            "source": self.source,
        }


def build_policy_features(task: str, input_profile: dict[str, Any]) -> dict[str, Any]:
    schema_profile = input_profile.get("schema_profile", {}) if isinstance(input_profile.get("schema_profile"), dict) else {}
    document_profile = input_profile.get("document_profile", {}) if isinstance(input_profile.get("document_profile"), dict) else {}
    quality_report = input_profile.get("quality_report", {}) if isinstance(input_profile.get("quality_report"), dict) else {}
    batch_profile = input_profile.get("batch_profile", {}) if isinstance(input_profile.get("batch_profile"), dict) else {}
    return {
        "task": task,
        "format": input_profile.get("format"),
        "structured": bool(input_profile.get("structured")),
        "unstructured": bool(input_profile.get("unstructured")),
        "batch": bool(input_profile.get("batch")),
        "complexity": input_profile.get("complexity"),
        "size_bytes": int(input_profile.get("size_bytes") or 0),
        "record_count_estimate": int(input_profile.get("record_count_estimate") or 0),
        "sampled_record_count": int(input_profile.get("sampled_record_count") or 0),
        "schema_field_count": int(schema_profile.get("field_count") or 0),
        "schema_text_coverage": float(schema_profile.get("schema_confidence", {}).get("text_coverage") or 0),
        "schema_has_id": bool(schema_profile.get("schema_confidence", {}).get("has_id")),
        "schema_has_text": bool(schema_profile.get("schema_confidence", {}).get("has_text")),
        "quality_warning_count": int(quality_report.get("warning_count") or 0),
        "document_count": int(document_profile.get("document_count") or 0),
        "batch_file_count": int(batch_profile.get("file_count") or 1),
        "structured_file_count": int(batch_profile.get("structured_file_count") or 0),
        "unstructured_file_count": int(batch_profile.get("unstructured_file_count") or 0),
        "strict_quality_signal_count": _signal_count(task, STRICT_QUALITY_TERMS),
        "low_latency_signal_count": _signal_count(task, LOW_LATENCY_TERMS),
        "official_chain_signal_count": _signal_count(task, OFFICIAL_CHAIN_TERMS),
        "quality_signal_count": _signal_count(task, QUALITY_TERMS),
        "structured_signal_count": _signal_count(task, STRUCTURED_TERMS),
        "text_signal_count": _signal_count(task, TEXT_TERMS),
    }


def _feature_to_text(features: dict[str, Any]) -> str:
    parts = [str(features.get("task") or "")]
    for key, value in sorted(features.items()):
        if key == "task":
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _numeric_features(features: dict[str, Any]) -> list[float]:
    keys = [
        "structured",
        "unstructured",
        "batch",
        "size_bytes",
        "record_count_estimate",
        "sampled_record_count",
        "schema_field_count",
        "schema_text_coverage",
        "schema_has_id",
        "schema_has_text",
        "quality_warning_count",
        "document_count",
        "batch_file_count",
        "structured_file_count",
        "unstructured_file_count",
        "strict_quality_signal_count",
        "low_latency_signal_count",
        "official_chain_signal_count",
        "quality_signal_count",
        "structured_signal_count",
        "text_signal_count",
    ]
    values: list[float] = []
    for key in keys:
        value = features.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        else:
            try:
                values.append(float(value or 0))
            except (TypeError, ValueError):
                values.append(0.0)
    return values


def _operator_to_ranker_text(operator: dict[str, Any]) -> str:
    values = [
        operator.get("id"),
        operator.get("name"),
        operator.get("description"),
        operator.get("categories"),
        operator.get("inputs"),
        operator.get("outputs"),
    ]
    return " ".join(
        json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "")
        for value in values
    )


def _operator_capability_guess(operator: dict[str, Any]) -> list[str]:
    text = _operator_to_ranker_text(operator).lower()
    capabilities: list[str] = []
    if any(token in text for token in ("link", "url", "http", "email", "ip", "phone", "pii")):
        capabilities.append("pii_cleanup")
    if any(token in text for token in ("space", "whitespace", "normalize", "clean", "mapper", "remove")):
        capabilities.append("text_normalization")
    if any(token in text for token in ("length", "empty", "short", "long", "line")):
        capabilities.append("length_filter")
    if any(token in text for token in ("quality", "filter", "alphanumeric", "similarity", "相似度")):
        capabilities.append("quality_filter")
    if any(token in text for token in ("dedup", "duplicate", "fingerprint", "similarity", "相似度", "frequency", "频率")):
        capabilities.append("deduplication")
    if any(token in text for token in ("schema", "field", "csv", "json", "table", "字段")):
        capabilities.append("schema_guard")
    if any(token in text for token in ("fast", "simple", "mapper", "filter")):
        capabilities.append("low_cost")
    return _multihot_to_labels(_labels_to_multihot(capabilities, CAPABILITY_LABELS), CAPABILITY_LABELS)


def _operator_cost_guess(operator: dict[str, Any]) -> float:
    text = _operator_to_ranker_text(operator).lower()
    cost = 1.0
    if any(token in text for token in ("embedding", "model", "llm", "extract", "parse", "table", "pdf")):
        cost += 2.0
    if any(token in text for token in ("dedup", "fingerprint", "quality", "similarity", "相似度")):
        cost += 0.6
    if any(token in text for token in ("mapper", "filter", "regex", "length")):
        cost -= 0.2
    return round(max(0.5, cost), 3)


def _operator_ranker_features(
    policy_features: dict[str, Any],
    operator: dict[str, Any],
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = labels or {}
    capabilities = _operator_capability_guess(operator)
    required = set(labels.get("required_capabilities") or [])
    recommended = set(labels.get("recommended_operator_ids") or [])
    failed = set(labels.get("failed_operator_ids") or [])
    operator_id = str(operator.get("id") or operator.get("name") or "")
    canonical_id = OPERATOR_ALIAS_LABELS.get(operator_id, operator_id)
    matched = sorted(required & set(capabilities))
    return {
        **policy_features,
        "operator_id": operator_id,
        "operator_canonical_id": canonical_id,
        "operator_text": _operator_to_ranker_text(operator),
        "operator_cost": _operator_cost_guess(operator),
        "operator_capabilities": ",".join(capabilities),
        "operator_required_capability_match_count": len(matched),
        "operator_required_capability_matches": ",".join(matched),
        "operator_recommended_exact": operator_id in recommended,
        "operator_recommended_alias": canonical_id in recommended,
        "operator_failed_exact": operator_id in failed,
        "operator_failed_alias": canonical_id in failed,
        "operator_inputs_text": "text" in str(operator.get("inputs") or "").lower(),
        "operator_outputs_text": "text" in str(operator.get("outputs") or "").lower(),
        "operator_quality_priority": labels.get("quality_priority", "standard"),
        "operator_cost_tier": labels.get("cost_tier", "balanced"),
        "runtime_feedback_score": float(labels.get("runtime_feedback_score", 1.0) or 0.0),
        "runtime_success": bool(labels.get("runtime_success", True)),
    }


def _ranker_feature_to_text(features: dict[str, Any]) -> str:
    parts = [_feature_to_text(features)]
    for key in (
        "operator_id",
        "operator_canonical_id",
        "operator_text",
        "operator_capabilities",
        "operator_required_capability_matches",
        "operator_recommended_exact",
        "operator_recommended_alias",
        "operator_failed_exact",
        "operator_failed_alias",
        "operator_quality_priority",
        "operator_cost_tier",
        "runtime_success",
    ):
        parts.append(f"{key}={features.get(key, '')}")
    return " ".join(parts)


def _ranker_numeric_features(features: dict[str, Any]) -> list[float]:
    values = _numeric_features(features)
    for key in (
        "operator_cost",
        "operator_required_capability_match_count",
        "operator_recommended_exact",
        "operator_recommended_alias",
        "operator_failed_exact",
        "operator_failed_alias",
        "operator_inputs_text",
        "operator_outputs_text",
        "runtime_feedback_score",
        "runtime_success",
    ):
        value = features.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        else:
            try:
                values.append(float(value or 0))
            except (TypeError, ValueError):
                values.append(0.0)
    return values


def _dag_ids(operators: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("id") or item.get("operator_id") or item.get("name") or "") for item in operators if item]


def _dag_capabilities_from_operator_ids(operator_ids: list[str]) -> list[str]:
    capabilities: list[str] = []
    prototypes = {
        str(item.get("id")): item
        for item in RANKER_OPERATOR_PROTOTYPES
        if item.get("id")
    }
    for operator_id in operator_ids:
        prototype = prototypes.get(operator_id)
        if prototype is None:
            prototype = {"id": operator_id, "name": operator_id, "description": operator_id, "inputs": "text", "outputs": "text"}
        capabilities.extend(_operator_capability_guess(prototype))
    return _multihot_to_labels(_labels_to_multihot(capabilities, CAPABILITY_LABELS), CAPABILITY_LABELS)


def _dag_policy_features(
    policy_features: dict[str, Any],
    operator_ids: list[str],
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = labels or {}
    canonical_ids = [OPERATOR_ALIAS_LABELS.get(str(item), str(item)) for item in operator_ids if item]
    capabilities = _dag_capabilities_from_operator_ids(canonical_ids)
    required = set(labels.get("required_capabilities") or [])
    failed = set(labels.get("failed_operator_ids") or [])
    recommended = set(labels.get("recommended_operator_ids") or [])
    covered = sorted(required & set(capabilities))
    failed_overlap = sorted(set(canonical_ids) & failed)
    recommended_overlap = sorted(set(canonical_ids) & recommended)
    total_cost = sum(_operator_cost_guess({"id": item, "name": item, "description": item, "inputs": "text", "outputs": "text"}) for item in canonical_ids)
    return {
        **policy_features,
        "dag_operator_ids": ",".join(canonical_ids),
        "dag_operator_count": len(canonical_ids),
        "dag_capabilities": ",".join(capabilities),
        "dag_required_coverage_count": len(covered),
        "dag_required_count": len(required),
        "dag_required_coverage_ratio": len(covered) / len(required) if required else 1.0,
        "dag_failed_operator_count": len(failed_overlap),
        "dag_failed_operator_ids": ",".join(failed_overlap),
        "dag_recommended_operator_count": len(recommended_overlap),
        "dag_total_cost": total_cost,
        "dag_low_cost_count": sum(1 for capability in capabilities if capability == "low_cost"),
        "dag_quality_priority": labels.get("quality_priority", "standard"),
        "dag_cost_tier": labels.get("cost_tier", "balanced"),
        "runtime_feedback_score": float(labels.get("runtime_feedback_score", 1.0) or 0.0),
        "runtime_success": bool(labels.get("runtime_success", True)),
    }


def _canonical_unique_operator_ids(operator_ids: list[str]) -> list[str]:
    canonical = [OPERATOR_ALIAS_LABELS.get(str(item), str(item)) for item in operator_ids if item]
    return [item for index, item in enumerate(canonical) if item and item not in canonical[:index]]


def _capability_preserving_subset(operator_ids: list[str], required: set[str]) -> list[str]:
    canonical = _canonical_unique_operator_ids(operator_ids)
    if len(canonical) <= 1:
        return canonical
    best_subset: list[str] = canonical
    best_cost = float("inf")
    for size in range(1, len(canonical) + 1):
        for combo in itertools.combinations(canonical, size):
            capabilities = set(_dag_capabilities_from_operator_ids(list(combo)))
            if required and not required.issubset(capabilities):
                continue
            cost = sum(_operator_cost_guess({"id": item, "name": item, "description": item, "inputs": "text", "outputs": "text"}) for item in combo)
            if len(combo) < len(best_subset) or (len(combo) == len(best_subset) and cost < best_cost):
                best_subset = list(combo)
                best_cost = cost
        if len(best_subset) == size:
            break
    return best_subset


def _operator_ids_for_missing_capability_negative(operator_ids: list[str], required: set[str]) -> list[str]:
    canonical = _canonical_unique_operator_ids(operator_ids)
    if len(canonical) <= 1 or not required:
        return []
    for capability in sorted(required):
        reduced = [
            item for item in canonical
            if capability not in _dag_capabilities_from_operator_ids([item])
        ]
        reduced_capabilities = set(_dag_capabilities_from_operator_ids(reduced))
        if reduced and reduced_capabilities & required and not required.issubset(reduced_capabilities):
            return reduced
    for size in range(1, len(canonical)):
        for combo in itertools.combinations(canonical, size):
            combo_capabilities = set(_dag_capabilities_from_operator_ids(list(combo)))
            if combo_capabilities & required and not required.issubset(combo_capabilities):
                return list(combo)
    return []


def _operator_ids_for_low_efficiency_negative(operator_ids: list[str]) -> list[str]:
    canonical = _canonical_unique_operator_ids(operator_ids)
    extras = [
        "download_file_mapper",
        "text_pair_similarity_filter",
        "alphanumeric_filter",
        "clean_links_mapper",
        "clean_ip_mapper",
        "clean_email_mapper",
        "clean_links_mapper",
    ]
    for extra in extras:
        if extra not in canonical:
            canonical.append(extra)
        if len(canonical) >= 6:
            break
    return canonical


def _dag_feature_to_text(features: dict[str, Any]) -> str:
    parts = [_feature_to_text(features)]
    for key in (
        "dag_operator_ids",
        "dag_capabilities",
        "dag_failed_operator_ids",
        "dag_quality_priority",
        "dag_cost_tier",
        "runtime_success",
    ):
        parts.append(f"{key}={features.get(key, '')}")
    return " ".join(parts)


def _dag_numeric_features(features: dict[str, Any]) -> list[float]:
    values = _numeric_features(features)
    for key in (
        "dag_operator_count",
        "dag_required_coverage_count",
        "dag_required_count",
        "dag_required_coverage_ratio",
        "dag_failed_operator_count",
        "dag_recommended_operator_count",
        "dag_total_cost",
        "dag_low_cost_count",
        "runtime_feedback_score",
        "runtime_success",
    ):
        value = features.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        else:
            try:
                values.append(float(value or 0))
            except (TypeError, ValueError):
                values.append(0.0)
    return values


def _dag_objective_weights(quality_priority: str, cost_tier: str) -> dict[str, float]:
    quality_priority = str(quality_priority or "standard")
    cost_tier = str(cost_tier or "balanced")
    weights = {"success": 0.5, "quality": 0.3, "efficiency": 0.2}
    if quality_priority == "critical" or cost_tier == "quality_first":
        weights = {"success": 0.42, "quality": 0.43, "efficiency": 0.15}
    elif quality_priority == "high":
        weights = {"success": 0.47, "quality": 0.36, "efficiency": 0.17}
    if cost_tier == "low":
        weights["efficiency"] += 0.12
        weights["quality"] -= 0.06
        weights["success"] -= 0.06
    total = sum(weights.values())
    normalized = {key: value / total for key, value in weights.items()}
    feasibility_weight = 0.15
    return {
        **{key: round(value * (1.0 - feasibility_weight), 4) for key, value in normalized.items()},
        "feasibility": feasibility_weight,
    }


def _labels_to_multihot(values: list[str], labels: list[str]) -> list[int]:
    value_set = set(values)
    return [1 if label in value_set else 0 for label in labels]


def _multihot_to_labels(values: list[int], labels: list[str]) -> list[str]:
    return [label for label, value in zip(labels, values) if int(value) == 1]


def _keyword_baseline_capability_f1(rows: list[dict[str, Any]], indices: Any, y_true: Any) -> float:
    from sklearn.metrics import f1_score
    import numpy as np

    predictions: list[list[int]] = []
    for index in indices:
        features = rows[int(index)]["features"]
        task = str(features.get("task") or "")
        labels: list[str] = []
        if _signal_count(task, QUALITY_TERMS) or features.get("quality_warning_count"):
            labels.extend(["quality_filter", "length_filter"])
        if _signal_count(task, STRUCTURED_TERMS) or features.get("structured"):
            labels.append("schema_guard")
        if _signal_count(task, TEXT_TERMS) or features.get("unstructured"):
            labels.append("text_normalization")
        if "去重" in task or "重复" in task or "dedup" in task.lower() or "duplicate" in task.lower():
            labels.append("deduplication")
        if _signal_count(task, LOW_LATENCY_TERMS):
            labels.append("low_cost")
        if "清洗" in task or "隐私" in task or "pii" in task.lower():
            labels.append("pii_cleanup")
        predictions.append(_labels_to_multihot(_multihot_to_labels(_labels_to_multihot(labels, CAPABILITY_LABELS), CAPABILITY_LABELS), CAPABILITY_LABELS))
    return round(float(f1_score(y_true[indices], np.array(predictions, dtype=int), average="micro", zero_division=0)), 4)


def _keyword_baseline_operator_f1(rows: list[dict[str, Any]], indices: Any, y_true: Any) -> float:
    from sklearn.metrics import f1_score
    import numpy as np

    predictions: list[list[int]] = []
    for index in indices:
        features = rows[int(index)]["features"]
        task = str(features.get("task") or "")
        labels = ["clean_email_mapper"]
        if _signal_count(task, QUALITY_TERMS) or features.get("quality_warning_count"):
            labels.append("text_length_filter")
        if _signal_count(task, STRICT_QUALITY_TERMS):
            labels.append("alphanumeric_filter")
        if _signal_count(task, STRUCTURED_TERMS) or "去重" in task or "重复" in task:
            labels.append("frequency_specified_field_selector")
        if "相似" in task or "duplicate" in task.lower():
            labels.append("text_pair_similarity_filter")
        predictions.append(_labels_to_multihot(labels[:4], OPERATOR_LABELS))
    return round(float(f1_score(y_true[indices], np.array(predictions, dtype=int), average="micro", zero_division=0)), 4)


def _load_policy_dataset_rows(dataset_path: Path, feedback_dataset_paths: list[Path] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    feedback_rows: list[dict[str, Any]] = []
    for feedback_path in feedback_dataset_paths or []:
        if not feedback_path.exists():
            continue
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("source", "datamate_runtime_feedback")
            labels = row.setdefault("labels", {})
            runtime_feedback = row.get("runtime_feedback", {}) if isinstance(row.get("runtime_feedback"), dict) else {}
            if runtime_feedback:
                failed_operator_ids = runtime_feedback.get("failed_operator_ids") or []
                failed_labels = [
                    OPERATOR_ALIAS_LABELS.get(str(item), str(item))
                    for item in failed_operator_ids
                    if item
                ]
                labels["failed_operator_ids"] = failed_labels
                if failed_labels:
                    failed_set = set(failed_labels)
                    labels["recommended_operator_ids"] = [
                        item
                        for item in labels.get("recommended_operator_ids", [])
                        if OPERATOR_ALIAS_LABELS.get(str(item), str(item)) not in failed_set
                    ]
                labels["runtime_feedback_score"] = float(runtime_feedback.get("feedback_score", 1.0) or 0.0)
                labels["runtime_success"] = (
                    runtime_feedback.get("status") == "completed"
                    and runtime_feedback.get("cleaning_task_status") in {"COMPLETED", "PARTIAL_SUCCESS"}
                )
            feedback_rows.append(row)
    _resolve_runtime_feedback_conflicts(feedback_rows)
    rows.extend(feedback_rows)
    return rows, feedback_rows


def _feedback_context_key(row: dict[str, Any]) -> tuple[Any, ...]:
    features = row.get("features", {}) if isinstance(row.get("features"), dict) else {}
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    return (
        labels.get("workflow_type"),
        features.get("format"),
        bool(features.get("structured")),
        bool(features.get("unstructured")),
        bool(labels.get("strict_quality")),
        bool(labels.get("requires_recovery")),
    )


def _feedback_dag_key(row: dict[str, Any]) -> tuple[str, ...]:
    runtime_feedback = row.get("runtime_feedback", {}) if isinstance(row.get("runtime_feedback"), dict) else {}
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    operator_ids = runtime_feedback.get("operator_ids") or labels.get("recommended_operator_ids") or []
    return tuple(sorted({
        OPERATOR_ALIAS_LABELS.get(str(item), str(item))
        for item in operator_ids
        if item
    }))


def _resolve_runtime_feedback_conflicts(feedback_rows: list[dict[str, Any]]) -> None:
    successful_dags = {
        (_feedback_context_key(row), _feedback_dag_key(row))
        for row in feedback_rows
        if row.get("labels", {}).get("runtime_success") and _feedback_dag_key(row)
    }
    successful_operator_sets_by_context: dict[tuple[Any, ...], set[str]] = {}
    for context, dag in successful_dags:
        successful_operator_sets_by_context.setdefault(context, set()).update(dag)

    for row in feedback_rows:
        labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
        if labels.get("runtime_success"):
            continue
        context = _feedback_context_key(row)
        dag = _feedback_dag_key(row)
        if (context, dag) in successful_dags:
            labels["failed_operator_ids"] = []
            labels["runtime_feedback_conflict_resolution"] = "older_failed_same_context_dag_superseded_by_success"
            continue
        successful_ops = successful_operator_sets_by_context.get(context, set())
        if successful_ops and labels.get("failed_operator_ids"):
            filtered_failed = [
                item
                for item in labels.get("failed_operator_ids", [])
                if OPERATOR_ALIAS_LABELS.get(str(item), str(item)) not in successful_ops
            ]
            if len(filtered_failed) != len(labels.get("failed_operator_ids", [])):
                labels["failed_operator_ids"] = filtered_failed
                labels["runtime_feedback_conflict_resolution"] = "operator_failures_softened_by_later_success_in_same_context"


def _build_failure_memory(feedback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memory: list[dict[str, Any]] = []
    for row in feedback_rows:
        labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
        failed_ids = labels.get("failed_operator_ids") or []
        runtime_feedback = row.get("runtime_feedback", {}) if isinstance(row.get("runtime_feedback"), dict) else {}
        if labels.get("runtime_feedback_conflict_resolution") and not failed_ids:
            continue
        if not failed_ids and runtime_feedback.get("status") != "blocked":
            continue
        features = row.get("features", {}) if isinstance(row.get("features"), dict) else {}
        failed_operator_ids = [
            OPERATOR_ALIAS_LABELS.get(str(item), str(item))
            for item in (failed_ids or runtime_feedback.get("operator_ids") or [])
            if item
        ]
        if not failed_operator_ids:
            continue
        memory.append(
            {
                "id": row.get("id"),
                "workflow_type": labels.get("workflow_type"),
                "format": features.get("format"),
                "structured": bool(features.get("structured")),
                "unstructured": bool(features.get("unstructured")),
                "strict_quality": bool(labels.get("strict_quality")),
                "requires_recovery": bool(labels.get("requires_recovery")),
                "failed_operator_ids": failed_operator_ids,
                "status": runtime_feedback.get("status"),
                "cleaning_task_status": runtime_feedback.get("cleaning_task_status"),
                "feedback_score": float(runtime_feedback.get("feedback_score", 0.0) or 0.0),
            }
        )
    return memory


def _failure_memory_matches(model: Any, features: dict[str, Any], workflow_type: str, *, limit: int = 5) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in model.get("failure_memory", []) if isinstance(model, dict) else []:
        score = 0.0
        if item.get("workflow_type") == workflow_type:
            score += 0.35
        if str(item.get("format") or "") == str(features.get("format") or ""):
            score += 0.25
        if bool(item.get("structured")) == bool(features.get("structured")):
            score += 0.15
        if bool(item.get("unstructured")) == bool(features.get("unstructured")):
            score += 0.1
        if bool(item.get("strict_quality")) and int(features.get("strict_quality_signal_count") or 0) > 0:
            score += 0.1
        if bool(item.get("requires_recovery")) and int(features.get("quality_warning_count") or 0) > 0:
            score += 0.05
        if score < 0.85 or not item.get("failed_operator_ids"):
            continue
        matches.append(
            {
                "memory_id": item.get("id"),
                "similarity": round(score, 4),
                "failed_operator_ids": item.get("failed_operator_ids", []),
                "status": item.get("status"),
                "cleaning_task_status": item.get("cleaning_task_status"),
            }
        )
    matches.sort(key=lambda item: (-float(item["similarity"]), str(item.get("memory_id") or "")))
    return matches[:limit]


def train_policy_model(
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    feedback_dataset_paths: list[Path] | None = None,
) -> dict[str, Any]:
    import joblib
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rows, feedback_rows = _load_policy_dataset_rows(dataset_path, feedback_dataset_paths)
    failure_memory = _build_failure_memory(feedback_rows)
    if len(rows) < 20:
        raise ValueError("Task-1 policy dataset must contain at least 20 rows")
    texts = [_feature_to_text(row["features"]) for row in rows]
    numeric = np.array([_numeric_features(row["features"]) for row in rows], dtype=float)
    workflow_y = np.array([WORKFLOW_LABELS.index(row["labels"]["workflow_type"]) for row in rows], dtype=int)
    capability_y = np.array([
        _labels_to_multihot(row["labels"]["required_capabilities"], CAPABILITY_LABELS)
        for row in rows
    ], dtype=int)
    operator_y = np.array([
        _labels_to_multihot(row["labels"].get("recommended_operator_ids", []), OPERATOR_LABELS)
        for row in rows
    ], dtype=int)
    keyword_y = np.array([
        _labels_to_multihot(row["labels"]["operator_keywords"], KEYWORD_LABELS)
        for row in rows
    ], dtype=int)
    low_latency_y = np.array([1 if row["labels"].get("prefer_low_latency") else 0 for row in rows], dtype=int)
    strict_quality_y = np.array([1 if row["labels"].get("strict_quality") else 0 for row in rows], dtype=int)
    recovery_y = np.array([1 if row["labels"].get("requires_recovery") else 0 for row in rows], dtype=int)
    quality_priority_labels = ["standard", "high", "critical"]
    cost_tier_labels = ["low", "balanced", "quality_first"]
    quality_priority_y = np.array([
        quality_priority_labels.index(row["labels"].get("quality_priority", "standard"))
        for row in rows
    ], dtype=int)
    cost_tier_y = np.array([
        cost_tier_labels.index(row["labels"].get("cost_tier", "balanced"))
        for row in rows
    ], dtype=int)
    sample_weight = np.array([
        (
            2.2
            if row.get("source") == "datamate_runtime_feedback" and row["labels"].get("runtime_success")
            else 1.7
            if row.get("source") == "datamate_runtime_feedback"
            else 1.0
        )
        for row in rows
    ], dtype=float)

    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=workflow_y,
    )
    ood_input_paths = {
        "data/raw/medical_demo_notes.txt",
        "data/raw/medical_demo_invalid.jsonl",
        "data/raw/medical_demo_schema_drift.csv",
        "data/raw/medical_demo_no_text.jsonl",
        "data/raw/medical_demo_long_notes.txt",
        "data/raw/medical_demo_privacy_noise.txt",
    }
    ood_terms = ("不允许", "高吞吐", "OOD", "schema漂移", "无标准text", "隐私噪声", "长篇病历", "旧失败反馈")
    ood_idx = np.array([
        index
        for index, row in enumerate(rows)
        if row.get("input_path") in ood_input_paths
        or any(path in ood_input_paths for path in row.get("input_paths", []))
        or any(term in str(row.get("task", "")) for term in ood_terms)
    ], dtype=int)
    if len(ood_idx) > 0:
        ood_train_overlap = set(int(item) for item in train_idx)
        ood_idx = np.array([int(item) for item in ood_idx if int(item) not in ood_train_overlap], dtype=int)
    vectorizer = HashingVectorizer(n_features=2048, alternate_sign=False, ngram_range=(1, 3), norm="l2")
    scaler = StandardScaler()
    text_x = vectorizer.transform(texts)
    numeric_x = scaler.fit_transform(numeric)
    x = hstack([text_x, numeric_x]).tocsr()

    def fit_classifier(y: Any) -> Any:
        clf = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
        clf.fit(x[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        return clf

    workflow_model = fit_classifier(workflow_y)
    low_latency_model = fit_classifier(low_latency_y)
    strict_quality_model = fit_classifier(strict_quality_y)
    recovery_model = fit_classifier(recovery_y)
    quality_priority_model = fit_classifier(quality_priority_y)
    cost_tier_model = fit_classifier(cost_tier_y)
    def fit_multilabel(y: Any) -> list[dict[str, Any]]:
        heads: list[dict[str, Any]] = []
        for column in range(y.shape[1]):
            values = y[train_idx, column]
            unique = sorted(set(int(item) for item in values))
            if len(unique) == 1:
                heads.append({"constant": unique[0], "model": None})
                continue
            clf = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
            clf.fit(x[train_idx], values, sample_weight=sample_weight[train_idx])
            heads.append({"constant": None, "model": clf})
        return heads

    def predict_multilabel(heads: list[dict[str, Any]], sample_x: Any) -> Any:
        columns: list[Any] = []
        for head in heads:
            if head["model"] is None:
                columns.append(np.full(sample_x.shape[0], int(head["constant"]), dtype=int))
            else:
                columns.append(head["model"].predict(sample_x).astype(int))
        return np.vstack(columns).T

    capability_model = fit_multilabel(capability_y)
    operator_model = fit_multilabel(operator_y)
    keyword_model = fit_multilabel(keyword_y)

    ranker_rows: list[dict[str, Any]] = []
    ranker_y: list[int] = []
    ranker_weight: list[float] = []
    ranker_failed_negative_count = 0
    ranker_feedback_positive_count = 0
    for row in rows:
        labels = row["labels"]
        recommended = set(labels.get("recommended_operator_ids") or [])
        failed = set(labels.get("failed_operator_ids") or [])
        runtime_feedback_score = float(labels.get("runtime_feedback_score", 1.0) or 0.0)
        runtime_success = bool(labels.get("runtime_success", True))
        is_runtime_feedback = row.get("source") == "datamate_runtime_feedback"
        for operator in RANKER_OPERATOR_PROTOTYPES:
            features = _operator_ranker_features(row["features"], operator, labels)
            operator_id = str(operator.get("id") or "")
            canonical_id = OPERATOR_ALIAS_LABELS.get(operator_id, operator_id)
            failed_operator = operator_id in failed or canonical_id in failed
            positive = (operator_id in recommended or canonical_id in recommended) and not failed_operator
            if (
                not positive
                and not failed_operator
                and not (is_runtime_feedback and not runtime_success)
                and features["operator_required_capability_match_count"] >= 2
            ):
                positive = labels.get("quality_priority") == "critical"
            weight = 1.0
            if is_runtime_feedback:
                weight = 1.0 + max(0.0, min(1.0, runtime_feedback_score))
                if runtime_success and positive:
                    weight += 0.9
                    ranker_feedback_positive_count += 1
                if failed_operator:
                    positive = False
                    weight += 6.0
                    ranker_failed_negative_count += 1
            ranker_rows.append(features)
            ranker_y.append(1 if positive else 0)
            ranker_weight.append(weight)
    ranker_texts = [_ranker_feature_to_text(item) for item in ranker_rows]
    ranker_numeric = np.array([_ranker_numeric_features(item) for item in ranker_rows], dtype=float)
    ranker_vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False, ngram_range=(1, 3), norm="l2")
    ranker_scaler = StandardScaler()
    ranker_text_x = ranker_vectorizer.transform(ranker_texts)
    ranker_numeric_x = ranker_scaler.fit_transform(ranker_numeric)
    ranker_x = hstack([ranker_text_x, ranker_numeric_x]).tocsr()
    ranker_y_array = np.array(ranker_y, dtype=int)
    ranker_weight_array = np.array(ranker_weight, dtype=float)
    ranker_train_idx = np.concatenate([
        np.arange(index * len(RANKER_OPERATOR_PROTOTYPES), (index + 1) * len(RANKER_OPERATOR_PROTOTYPES))
        for index in train_idx
    ])
    ranker_test_idx = np.concatenate([
        np.arange(index * len(RANKER_OPERATOR_PROTOTYPES), (index + 1) * len(RANKER_OPERATOR_PROTOTYPES))
        for index in test_idx
    ])
    operator_ranker_model = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
    operator_ranker_model.fit(
        ranker_x[ranker_train_idx],
        ranker_y_array[ranker_train_idx],
        sample_weight=ranker_weight_array[ranker_train_idx],
    )

    dag_rows: list[dict[str, Any]] = []
    dag_y: list[int] = []
    dag_quality_y: list[int] = []
    dag_efficiency_y: list[int] = []
    dag_feasibility_y: list[int] = []
    dag_weight: list[float] = []
    dag_row_owner: list[int] = []
    dag_negative_kind: list[str] = []
    dag_feedback_positive_count = 0
    dag_feedback_negative_count = 0
    dag_hard_negative_count = 0
    dag_efficiency_negative_count = 0
    dag_distilled_minimal_positive_count = 0
    for row_index, row in enumerate(rows):
        labels = row["labels"]
        recommended_ids = [str(item) for item in labels.get("recommended_operator_ids", []) if item]
        failed_ids = [str(item) for item in labels.get("failed_operator_ids", []) if item]
        required_ids = {str(item) for item in labels.get("required_capabilities", []) if item}
        runtime_feedback_score = float(labels.get("runtime_feedback_score", 1.0) or 0.0)
        runtime_success = bool(labels.get("runtime_success", True))
        is_runtime_feedback = row.get("source") == "datamate_runtime_feedback"
        conflict_resolved = bool(labels.get("runtime_feedback_conflict_resolution"))

        def add_dag(
            operator_ids: list[str],
            positive: bool,
            weight: float,
            *,
            force_efficiency_negative: bool = False,
            negative_kind: str = "",
        ) -> None:
            canonical = _canonical_unique_operator_ids(operator_ids)
            if not canonical:
                return
            features = _dag_policy_features(row["features"], canonical, labels)
            feasible = (
                float(features.get("dag_required_coverage_ratio", 0) or 0) >= 0.999
                and int(features.get("dag_failed_operator_count", 0) or 0) == 0
            )
            efficient = (
                float(features.get("dag_total_cost", 99) or 99) <= 2.6
                and int(features.get("dag_operator_count", 99) or 99) <= 4
                and not force_efficiency_negative
            )
            dag_y.append(1 if positive else 0)
            dag_quality_y.append(1 if positive and feasible else 0)
            dag_efficiency_y.append(1 if positive and efficient else 0)
            dag_feasibility_y.append(1 if feasible else 0)
            dag_rows.append(features)
            dag_weight.append(weight)
            dag_row_owner.append(row_index)
            dag_negative_kind.append(negative_kind if negative_kind else ("" if positive else "generic_negative"))

        if recommended_ids:
            add_dag(recommended_ids, True, 1.0 + (1.2 if is_runtime_feedback and runtime_success else 0.0))
            if len(recommended_ids) > 1:
                prefix_features = _dag_policy_features(row["features"], recommended_ids[:1], labels)
                if float(prefix_features.get("dag_required_coverage_ratio", 0) or 0) < 0.999:
                    add_dag(recommended_ids[:1], False, 1.4, negative_kind="prefix_missing_capability")
            minimal_positive = _capability_preserving_subset(recommended_ids, required_ids)
            if minimal_positive and set(minimal_positive) != set(_canonical_unique_operator_ids(recommended_ids)):
                add_dag(minimal_positive, True, 1.15 + (0.8 if is_runtime_feedback and runtime_success else 0.0))
                dag_distilled_minimal_positive_count += 1
            missing_capability_negative = _operator_ids_for_missing_capability_negative(recommended_ids, required_ids)
            if missing_capability_negative and set(missing_capability_negative) != set(_canonical_unique_operator_ids(recommended_ids)):
                add_dag(missing_capability_negative, False, 3.4, negative_kind="hard_missing_capability")
                dag_hard_negative_count += 1
            low_efficiency_negative = _operator_ids_for_low_efficiency_negative(recommended_ids)
            if set(low_efficiency_negative) != set(_canonical_unique_operator_ids(recommended_ids)):
                add_dag(low_efficiency_negative, True, 2.8, force_efficiency_negative=True, negative_kind="efficiency_negative")
                dag_efficiency_negative_count += 1
        if failed_ids:
            add_dag(failed_ids, False, 3.0 + (1.0 - min(1.0, max(0.0, runtime_feedback_score))) * 2.0, negative_kind="runtime_failed_operator")
        add_dag(["download_file_mapper"], False, 1.2, negative_kind="irrelevant_operator")
        if labels.get("quality_priority") == "critical":
            add_dag(["clean_email_mapper"], False, 1.3, negative_kind="critical_quality_undercovered")
        if is_runtime_feedback and runtime_success:
            actual_ids = [
                str(item.get("operator_id") or item.get("id") or "")
                for item in row.get("runtime_feedback", {}).get("operator_dag", [])
                if isinstance(item, dict)
            ]
            actual_ids = actual_ids or [str(item) for item in row.get("runtime_feedback", {}).get("operator_ids", []) if item]
            if actual_ids:
                add_dag(actual_ids, True, 2.5)
                dag_feedback_positive_count += 1
                low_efficiency_negative = _operator_ids_for_low_efficiency_negative(actual_ids)
                if set(low_efficiency_negative) != set(_canonical_unique_operator_ids(actual_ids)):
                    add_dag(low_efficiency_negative, True, 3.2, force_efficiency_negative=True, negative_kind="efficiency_negative")
                    dag_efficiency_negative_count += 1
        if is_runtime_feedback and not runtime_success and not conflict_resolved:
            actual_ids = [str(item) for item in row.get("runtime_feedback", {}).get("operator_ids", []) if item]
            if actual_ids:
                add_dag(actual_ids, False, 4.0, negative_kind="runtime_failed_dag")
                dag_feedback_negative_count += 1

    if len(set(dag_y)) < 2:
        dag_success_model = None
        dag_quality_model = None
        dag_efficiency_model = None
        dag_feasibility_model = None
        dag_vectorizer = None
        dag_scaler = None
        dag_x = None
        dag_y_array = np.array(dag_y, dtype=int)
        dag_quality_y_array = np.array(dag_quality_y, dtype=int)
        dag_efficiency_y_array = np.array(dag_efficiency_y, dtype=int)
        dag_feasibility_y_array = np.array(dag_feasibility_y, dtype=int)
        dag_weight_array = np.array(dag_weight, dtype=float)
    else:
        dag_texts = [_dag_feature_to_text(item) for item in dag_rows]
        dag_numeric = np.array([_dag_numeric_features(item) for item in dag_rows], dtype=float)
        dag_vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False, ngram_range=(1, 3), norm="l2")
        dag_scaler = StandardScaler()
        dag_text_x = dag_vectorizer.transform(dag_texts)
        dag_numeric_x = dag_scaler.fit_transform(dag_numeric)
        dag_x = hstack([dag_text_x, dag_numeric_x]).tocsr()
        dag_y_array = np.array(dag_y, dtype=int)
        dag_quality_y_array = np.array(dag_quality_y, dtype=int)
        dag_efficiency_y_array = np.array(dag_efficiency_y, dtype=int)
        dag_feasibility_y_array = np.array(dag_feasibility_y, dtype=int)
        dag_weight_array = np.array(dag_weight, dtype=float)
        dag_indices = np.arange(len(dag_y_array))
        dag_train_indices, dag_eval_indices = train_test_split(
            dag_indices,
            test_size=0.25,
            random_state=42,
            stratify=dag_y_array,
        )
        dag_success_model = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
        dag_success_model.fit(
            dag_x[dag_train_indices],
            dag_y_array[dag_train_indices],
            sample_weight=dag_weight_array[dag_train_indices],
        )
        dag_quality_model = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
        dag_quality_model.fit(
            dag_x[dag_train_indices],
            dag_quality_y_array[dag_train_indices],
            sample_weight=dag_weight_array[dag_train_indices],
        )
        dag_efficiency_model = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
        dag_efficiency_model.fit(
            dag_x[dag_train_indices],
            dag_efficiency_y_array[dag_train_indices],
            sample_weight=dag_weight_array[dag_train_indices],
        )
        dag_feasibility_model = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
        dag_feasibility_model.fit(
            dag_x[dag_train_indices],
            dag_feasibility_y_array[dag_train_indices],
            sample_weight=dag_weight_array[dag_train_indices],
        )

    workflow_pred = workflow_model.predict(x[test_idx])
    capability_pred = predict_multilabel(capability_model, x[test_idx])
    operator_pred = predict_multilabel(operator_model, x[test_idx])
    keyword_pred = predict_multilabel(keyword_model, x[test_idx])
    low_latency_pred = low_latency_model.predict(x[test_idx])
    strict_quality_pred = strict_quality_model.predict(x[test_idx])
    recovery_pred = recovery_model.predict(x[test_idx])
    quality_priority_pred = quality_priority_model.predict(x[test_idx])
    cost_tier_pred = cost_tier_model.predict(x[test_idx])
    ranker_test_proba = operator_ranker_model.predict_proba(ranker_x[ranker_test_idx])[:, 1]
    ranker_test_pred = (ranker_test_proba >= 0.5).astype(int)

    def ranker_topk_hit(row_indices: Any, *, k: int) -> float:
        hits = 0
        total = 0
        for row_index in row_indices:
            row = rows[int(row_index)]
            labels = row["labels"]
            recommended = set(labels.get("recommended_operator_ids") or [])
            start = int(row_index) * len(RANKER_OPERATOR_PROTOTYPES)
            end = start + len(RANKER_OPERATOR_PROTOTYPES)
            scores = operator_ranker_model.predict_proba(ranker_x[start:end])[:, 1]
            ranked = [
                str(RANKER_OPERATOR_PROTOTYPES[index].get("id"))
                for index in np.argsort(-scores)[:k]
            ]
            aliases = {OPERATOR_ALIAS_LABELS.get(item, item) for item in ranked}
            if recommended and (recommended & set(ranked) or recommended & aliases):
                hits += 1
            total += 1
        return round(hits / total, 4) if total else 0.0

    def failed_operator_negative_score(row_indices: Any) -> dict[str, Any]:
        scores: list[float] = []
        for row_index in row_indices:
            row = rows[int(row_index)]
            labels = row["labels"]
            failed = set(labels.get("failed_operator_ids") or [])
            if not failed:
                continue
            start = int(row_index) * len(RANKER_OPERATOR_PROTOTYPES)
            end = start + len(RANKER_OPERATOR_PROTOTYPES)
            row_scores = operator_ranker_model.predict_proba(ranker_x[start:end])[:, 1]
            for index, operator in enumerate(RANKER_OPERATOR_PROTOTYPES):
                operator_id = str(operator.get("id") or "")
                canonical_id = OPERATOR_ALIAS_LABELS.get(operator_id, operator_id)
                if operator_id in failed or canonical_id in failed:
                    scores.append(float(row_scores[index]))
        return {
            "count": len(scores),
            "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
            "max_score": round(max(scores), 4) if scores else None,
        }

    failed_negative_eval = failed_operator_negative_score(indices)

    dag_metrics: dict[str, Any] = {
        "dag_policy_sample_count": len(dag_y),
        "dag_feedback_positive_count": dag_feedback_positive_count,
        "dag_feedback_negative_count": dag_feedback_negative_count,
        "dag_hard_negative_count": dag_hard_negative_count,
        "dag_efficiency_negative_count": dag_efficiency_negative_count,
        "dag_distilled_minimal_positive_count": dag_distilled_minimal_positive_count,
        "dag_policy_available": dag_success_model is not None,
    }
    if dag_success_model is not None and dag_x is not None:
        if len(dag_eval_indices) and len(set(int(item) for item in dag_y_array[dag_eval_indices])) > 1:
            dag_eval_proba = dag_success_model.predict_proba(dag_x[dag_eval_indices])[:, 1]
            dag_eval_pred = (dag_eval_proba >= 0.5).astype(int)
            dag_metrics.update(
                {
                    "dag_policy_auc": round(float(roc_auc_score(dag_y_array[dag_eval_indices], dag_eval_proba)), 4),
                    "dag_policy_micro_f1": round(float(f1_score(dag_y_array[dag_eval_indices], dag_eval_pred, average="micro", zero_division=0)), 4),
                }
            )
        if len(dag_eval_indices) and len(set(int(item) for item in dag_quality_y_array[dag_eval_indices])) > 1:
            dag_quality_proba = dag_quality_model.predict_proba(dag_x[dag_eval_indices])[:, 1]
            dag_quality_pred = (dag_quality_proba >= 0.5).astype(int)
            dag_metrics.update(
                {
                    "dag_quality_auc": round(float(roc_auc_score(dag_quality_y_array[dag_eval_indices], dag_quality_proba)), 4),
                    "dag_quality_micro_f1": round(float(f1_score(dag_quality_y_array[dag_eval_indices], dag_quality_pred, average="micro", zero_division=0)), 4),
                }
            )
        if len(dag_eval_indices) and len(set(int(item) for item in dag_efficiency_y_array[dag_eval_indices])) > 1:
            dag_efficiency_proba = dag_efficiency_model.predict_proba(dag_x[dag_eval_indices])[:, 1]
            dag_efficiency_pred = (dag_efficiency_proba >= 0.5).astype(int)
            dag_metrics.update(
                {
                    "dag_efficiency_auc": round(float(roc_auc_score(dag_efficiency_y_array[dag_eval_indices], dag_efficiency_proba)), 4),
                    "dag_efficiency_micro_f1": round(float(f1_score(dag_efficiency_y_array[dag_eval_indices], dag_efficiency_pred, average="micro", zero_division=0)), 4),
                }
            )
        if len(dag_eval_indices) and len(set(int(item) for item in dag_feasibility_y_array[dag_eval_indices])) > 1:
            dag_feasibility_proba = dag_feasibility_model.predict_proba(dag_x[dag_eval_indices])[:, 1]
            dag_feasibility_pred = (dag_feasibility_proba >= 0.5).astype(int)
            dag_metrics.update(
                {
                    "dag_feasibility_auc": round(float(roc_auc_score(dag_feasibility_y_array[dag_eval_indices], dag_feasibility_proba)), 4),
                    "dag_feasibility_micro_f1": round(float(f1_score(dag_feasibility_y_array[dag_eval_indices], dag_feasibility_pred, average="micro", zero_division=0)), 4),
                }
            )
        failed_scores: list[float] = []
        success_scores: list[float] = []
        hard_negative_success_scores: list[float] = []
        efficiency_negative_scores: list[float] = []
        negative_scores_by_kind: dict[str, list[float]] = {}
        for index, row in enumerate(dag_rows):
            score = float(dag_success_model.predict_proba(dag_x[index:index + 1])[0][1])
            if dag_y[index]:
                success_scores.append(score)
                if not dag_efficiency_y[index] and dag_negative_kind[index] == "efficiency_negative":
                    efficiency_negative_scores.append(float(dag_efficiency_model.predict_proba(dag_x[index:index + 1])[0][1]))
            else:
                failed_scores.append(score)
                negative_scores_by_kind.setdefault(dag_negative_kind[index] or "generic_negative", []).append(score)
                if dag_negative_kind[index] in {
                    "prefix_missing_capability",
                    "hard_missing_capability",
                    "runtime_failed_operator",
                    "runtime_failed_dag",
                    "irrelevant_operator",
                    "critical_quality_undercovered",
                }:
                    hard_negative_success_scores.append(score)
        dag_metrics.update(
            {
                "dag_policy_positive_avg_score": round(sum(success_scores) / len(success_scores), 4) if success_scores else None,
                "dag_policy_negative_avg_score": round(sum(failed_scores) / len(failed_scores), 4) if failed_scores else None,
                "dag_policy_negative_max_score": round(max(failed_scores), 4) if failed_scores else None,
                "dag_hard_negative_success_avg_score": round(sum(hard_negative_success_scores) / len(hard_negative_success_scores), 4) if hard_negative_success_scores else None,
                "dag_hard_negative_success_max_score": round(max(hard_negative_success_scores), 4) if hard_negative_success_scores else None,
                "dag_efficiency_negative_avg_efficiency_score": round(sum(efficiency_negative_scores) / len(efficiency_negative_scores), 4) if efficiency_negative_scores else None,
                "dag_efficiency_negative_max_efficiency_score": round(max(efficiency_negative_scores), 4) if efficiency_negative_scores else None,
                "dag_negative_score_by_kind": {
                    kind: {
                        "count": len(values),
                        "avg_success_score": round(sum(values) / len(values), 4),
                        "max_success_score": round(max(values), 4),
                    }
                    for kind, values in sorted(negative_scores_by_kind.items())
                    if values
                },
            }
        )

    ood_metrics: dict[str, float] = {}
    if len(ood_idx) >= 20:
        ood_metrics = {
            "ood_sample_count": int(len(ood_idx)),
            "ood_workflow_accuracy": round(float(accuracy_score(workflow_y[ood_idx], workflow_model.predict(x[ood_idx]))), 4),
            "ood_capability_micro_f1": round(float(f1_score(capability_y[ood_idx], predict_multilabel(capability_model, x[ood_idx]), average="micro", zero_division=0)), 4),
            "ood_operator_micro_f1": round(float(f1_score(operator_y[ood_idx], predict_multilabel(operator_model, x[ood_idx]), average="micro", zero_division=0)), 4),
            "ood_keyword_micro_f1": round(float(f1_score(keyword_y[ood_idx], predict_multilabel(keyword_model, x[ood_idx]), average="micro", zero_division=0)), 4),
            "ood_recovery_accuracy": round(float(accuracy_score(recovery_y[ood_idx], recovery_model.predict(x[ood_idx]))), 4),
            "ood_operator_ranker_top3_hit_rate": ranker_topk_hit(ood_idx, k=3),
        }
    metrics = {
        "model_version": "task1_orchestration_policy_v2_multi_objective",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "sample_count": len(rows),
        "distillation_sample_count": len(rows) - len(feedback_rows),
        "runtime_feedback_sample_count": len(feedback_rows),
        "runtime_feedback_dataset_paths": [str(path) for path in feedback_dataset_paths or [] if path.exists()],
        "uses_runtime_feedback": bool(feedback_rows),
        "failure_memory_count": len(failure_memory),
        "runtime_feedback_weighting": "success_weight_2.2_failure_weight_1.7_ranker_failed_operator_negative_weight",
        "ranker_failed_operator_negative_count": ranker_failed_negative_count,
        "ranker_feedback_positive_count": ranker_feedback_positive_count,
        "failed_operator_ranker_eval": failed_negative_eval,
        **dag_metrics,
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "parameter_count_estimate": int(x.shape[1] * (
            len(WORKFLOW_LABELS)
            + len(CAPABILITY_LABELS)
            + len(OPERATOR_LABELS)
            + len(KEYWORD_LABELS)
            + 5
        )),
        "less_than_1b_parameters": True,
        "workflow_accuracy": round(float(accuracy_score(workflow_y[test_idx], workflow_pred)), 4),
        "capability_micro_f1": round(float(f1_score(capability_y[test_idx], capability_pred, average="micro", zero_division=0)), 4),
        "operator_micro_f1": round(float(f1_score(operator_y[test_idx], operator_pred, average="micro", zero_division=0)), 4),
        "keyword_micro_f1": round(float(f1_score(keyword_y[test_idx], keyword_pred, average="micro", zero_division=0)), 4),
        "low_latency_accuracy": round(float(accuracy_score(low_latency_y[test_idx], low_latency_pred)), 4),
        "strict_quality_accuracy": round(float(accuracy_score(strict_quality_y[test_idx], strict_quality_pred)), 4),
        "recovery_accuracy": round(float(accuracy_score(recovery_y[test_idx], recovery_pred)), 4),
        "quality_priority_accuracy": round(float(accuracy_score(quality_priority_y[test_idx], quality_priority_pred)), 4),
        "cost_tier_accuracy": round(float(accuracy_score(cost_tier_y[test_idx], cost_tier_pred)), 4),
        "baseline_keyword_only_capability_micro_f1": _keyword_baseline_capability_f1(rows, test_idx, capability_y),
        "baseline_keyword_only_operator_micro_f1": _keyword_baseline_operator_f1(rows, test_idx, operator_y),
        "operator_ranker_auc": round(float(roc_auc_score(ranker_y_array[ranker_test_idx], ranker_test_proba)), 4),
        "operator_ranker_micro_f1": round(float(f1_score(ranker_y_array[ranker_test_idx], ranker_test_pred, average="micro", zero_division=0)), 4),
        "operator_ranker_top1_hit_rate": ranker_topk_hit(test_idx, k=1),
        "operator_ranker_top3_hit_rate": ranker_topk_hit(test_idx, k=3),
        **ood_metrics,
    }
    metrics["parameter_count_estimate"] += int(ranker_x.shape[1] * 2)
    if dag_x is not None:
        metrics["parameter_count_estimate"] += int(dag_x.shape[1] * 2)
    model = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "workflow_model": workflow_model,
        "capability_model": capability_model,
        "operator_model": operator_model,
        "keyword_model": keyword_model,
        "low_latency_model": low_latency_model,
        "strict_quality_model": strict_quality_model,
        "recovery_model": recovery_model,
        "quality_priority_model": quality_priority_model,
        "cost_tier_model": cost_tier_model,
        "operator_ranker_model": operator_ranker_model,
        "operator_ranker_vectorizer": ranker_vectorizer,
        "operator_ranker_scaler": ranker_scaler,
        "operator_ranker_prototypes": RANKER_OPERATOR_PROTOTYPES,
        "dag_success_model": dag_success_model,
        "dag_quality_model": dag_quality_model,
        "dag_efficiency_model": dag_efficiency_model,
        "dag_feasibility_model": dag_feasibility_model,
        "dag_vectorizer": dag_vectorizer,
        "dag_scaler": dag_scaler,
        "failure_memory": failure_memory,
        "metrics": metrics,
        "labels": {
            "workflow": WORKFLOW_LABELS,
            "capabilities": CAPABILITY_LABELS,
            "operators": OPERATOR_LABELS,
            "keywords": KEYWORD_LABELS,
            "quality_priority": quality_priority_labels,
            "cost_tier": cost_tier_labels,
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_policy_model(model_path: Path) -> Any:
    import joblib

    if not model_path.exists():
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        return joblib.load(model_path)


def predict_policy(model: Any, task: str, input_profile: dict[str, Any]) -> Task1PolicyPrediction | None:
    if model is None:
        return None
    from scipy.sparse import hstack
    import numpy as np

    features = build_policy_features(task, input_profile)
    text_x = model["vectorizer"].transform([_feature_to_text(features)])
    numeric_x = model["scaler"].transform(np.array([_numeric_features(features)], dtype=float))
    x = hstack([text_x, numeric_x]).tocsr()
    def predict_multilabel(heads: list[dict[str, Any]]) -> list[int]:
        values: list[int] = []
        for head in heads:
            if head["model"] is None:
                values.append(int(head["constant"]))
            else:
                values.append(int(head["model"].predict(x)[0]))
        return values

    workflow_index = int(model["workflow_model"].predict(x)[0])
    capability_values = predict_multilabel(model["capability_model"])
    operator_values = predict_multilabel(model["operator_model"]) if "operator_model" in model else []
    keyword_values = predict_multilabel(model["keyword_model"])
    prefer_low_latency = bool(int(model["low_latency_model"].predict(x)[0]))
    strict_quality = bool(int(model["strict_quality_model"].predict(x)[0]))
    requires_recovery = bool(int(model["recovery_model"].predict(x)[0])) if "recovery_model" in model else strict_quality
    quality_priority_labels = model.get("labels", {}).get("quality_priority", ["standard", "high", "critical"])
    cost_tier_labels = model.get("labels", {}).get("cost_tier", ["low", "balanced", "quality_first"])
    quality_priority_index = (
        int(model["quality_priority_model"].predict(x)[0])
        if "quality_priority_model" in model
        else (2 if strict_quality else 1 if requires_recovery else 0)
    )
    cost_tier_index = (
        int(model["cost_tier_model"].predict(x)[0])
        if "cost_tier_model" in model
        else (0 if prefer_low_latency else 2 if strict_quality else 1)
    )

    confidence_parts: list[float] = []
    for clf_name in (
        "workflow_model",
        "low_latency_model",
        "strict_quality_model",
        "recovery_model",
        "quality_priority_model",
        "cost_tier_model",
    ):
        if clf_name not in model:
            continue
        clf = model[clf_name]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x)[0]
            confidence_parts.append(float(max(probs)))
    confidence = round(sum(confidence_parts) / len(confidence_parts), 4) if confidence_parts else 0.5
    operator_labels = model.get("labels", {}).get("operators", OPERATOR_LABELS)
    workflow_type = WORKFLOW_LABELS[workflow_index]
    failure_matches = _failure_memory_matches(model, features, workflow_type)
    avoid_operator_ids = _multihot_to_labels(
        _labels_to_multihot(
            [
                OPERATOR_ALIAS_LABELS.get(str(item), str(item))
                for match in failure_matches
                for item in match.get("failed_operator_ids", [])
            ],
            OPERATOR_LABELS,
        ),
        OPERATOR_LABELS,
    )
    recommended_operator_ids = _multihot_to_labels(operator_values, operator_labels) if operator_values else []
    if avoid_operator_ids:
        avoid_set = set(avoid_operator_ids)
        recommended_operator_ids = [
            item for item in recommended_operator_ids
            if OPERATOR_ALIAS_LABELS.get(str(item), str(item)) not in avoid_set
        ]
    return Task1PolicyPrediction(
        workflow_type=workflow_type,
        required_capabilities=_multihot_to_labels(capability_values, CAPABILITY_LABELS),
        recommended_operator_ids=recommended_operator_ids,
        operator_keywords=_multihot_to_labels(keyword_values, KEYWORD_LABELS),
        prefer_low_latency=prefer_low_latency,
        strict_quality=strict_quality,
        requires_recovery=requires_recovery,
        quality_priority=str(quality_priority_labels[min(quality_priority_index, len(quality_priority_labels) - 1)]),
        cost_tier=str(cost_tier_labels[min(cost_tier_index, len(cost_tier_labels) - 1)]),
        avoid_operator_ids=avoid_operator_ids,
        failure_memory_matches=failure_matches,
        confidence=confidence,
        source="trained_task1_orchestration_policy_model",
    )


def score_operator_with_policy_ranker(
    model: Any,
    task: str,
    input_profile: dict[str, Any],
    operator: dict[str, Any],
    policy_prediction: dict[str, Any] | None = None,
) -> float | None:
    if not model or "operator_ranker_model" not in model:
        return None
    canonical_operator_id = OPERATOR_ALIAS_LABELS.get(str(operator.get("id") or operator.get("name") or ""), str(operator.get("id") or operator.get("name") or ""))
    avoid_operator_ids = {
        OPERATOR_ALIAS_LABELS.get(str(item), str(item))
        for item in (policy_prediction or {}).get("avoid_operator_ids", [])
        if item
    }
    if canonical_operator_id in avoid_operator_ids:
        return 0.0
    from scipy.sparse import hstack
    import numpy as np

    features = build_policy_features(task, input_profile)
    labels = {
        "required_capabilities": (policy_prediction or {}).get("required_capabilities", []),
        "recommended_operator_ids": (policy_prediction or {}).get("recommended_operator_ids", []),
        "quality_priority": (policy_prediction or {}).get("quality_priority", "standard"),
        "cost_tier": (policy_prediction or {}).get("cost_tier", "balanced"),
    }
    ranker_features = _operator_ranker_features(features, operator, labels)
    text_x = model["operator_ranker_vectorizer"].transform([_ranker_feature_to_text(ranker_features)])
    numeric_x = model["operator_ranker_scaler"].transform(np.array([_ranker_numeric_features(ranker_features)], dtype=float))
    x = hstack([text_x, numeric_x]).tocsr()
    ranker = model["operator_ranker_model"]
    if hasattr(ranker, "predict_proba"):
        return round(float(ranker.predict_proba(x)[0][1]), 6)
    return round(float(ranker.decision_function(x)[0]), 6) if hasattr(ranker, "decision_function") else None


def score_dag_with_policy_model(
    model: Any,
    task: str,
    input_profile: dict[str, Any],
    operators: list[dict[str, Any]],
    policy_prediction: dict[str, Any] | None = None,
) -> float | None:
    scores = score_dag_objectives_with_policy_model(model, task, input_profile, operators, policy_prediction)
    return scores.get("success_score") if scores else None


def score_dag_objectives_with_policy_model(
    model: Any,
    task: str,
    input_profile: dict[str, Any],
    operators: list[dict[str, Any]],
    policy_prediction: dict[str, Any] | None = None,
) -> dict[str, float] | None:
    if not model or not model.get("dag_success_model"):
        return None
    from scipy.sparse import hstack
    import numpy as np

    features = build_policy_features(task, input_profile)
    labels = {
        "required_capabilities": (policy_prediction or {}).get("required_capabilities", []),
        "recommended_operator_ids": (policy_prediction or {}).get("recommended_operator_ids", []),
        "quality_priority": (policy_prediction or {}).get("quality_priority", "standard"),
        "cost_tier": (policy_prediction or {}).get("cost_tier", "balanced"),
        "runtime_feedback_score": 1.0,
        "runtime_success": True,
    }
    dag_features = _dag_policy_features(features, _dag_ids(operators), labels)
    text_x = model["dag_vectorizer"].transform([_dag_feature_to_text(dag_features)])
    numeric_x = model["dag_scaler"].transform(np.array([_dag_numeric_features(dag_features)], dtype=float))
    x = hstack([text_x, numeric_x]).tocsr()
    def score_head(name: str) -> float | None:
        head = model.get(name)
        if head is None:
            return None
        if hasattr(head, "predict_proba"):
            return round(float(head.predict_proba(x)[0][1]), 6)
        return round(float(head.decision_function(x)[0]), 6) if hasattr(head, "decision_function") else None

    success = score_head("dag_success_model")
    quality = score_head("dag_quality_model")
    efficiency = score_head("dag_efficiency_model")
    if success is None:
        return None
    quality_value = quality if quality is not None else success
    efficiency_value = efficiency if efficiency is not None else success
    feasibility = score_head("dag_feasibility_model")
    feasibility_value = feasibility if feasibility is not None else quality_value
    weights = _dag_objective_weights(
        str((policy_prediction or {}).get("quality_priority", "standard")),
        str((policy_prediction or {}).get("cost_tier", "balanced")),
    )
    raw_overall = round(
        success * weights["success"]
        + quality_value * weights["quality"]
        + efficiency_value * weights["efficiency"]
        + feasibility_value * weights.get("feasibility", 0.0),
        6,
    )
    coverage_ratio = float(dag_features.get("dag_required_coverage_ratio", 1.0) or 0.0)
    required = {
        str(item)
        for item in (policy_prediction or {}).get("required_capabilities", [])
        if item
    }
    covered = set(str(dag_features.get("dag_capabilities") or "").split(",")) if dag_features.get("dag_capabilities") else set()
    uncovered = sorted(required - covered)
    constraint_penalty = 0.0
    if uncovered:
        constraint_penalty += min(0.65, 0.22 * len(uncovered))
    if coverage_ratio < 0.999:
        constraint_penalty += (1.0 - coverage_ratio) * 0.35
    if feasibility_value < 0.5:
        constraint_penalty += (0.5 - feasibility_value) * 0.3
    constraint_penalty = round(min(0.9, max(0.0, constraint_penalty)), 6)
    constraint_ceiling = round(min(1.0, max(0.0, coverage_ratio * 0.72 + feasibility_value * 0.28)), 6)
    calibrated_overall = max(0.0, min(raw_overall, constraint_ceiling) - constraint_penalty)
    overall = round(calibrated_overall, 6)
    return {
        "success_score": success,
        "quality_score": quality_value,
        "efficiency_score": efficiency_value,
        "feasibility_score": feasibility_value,
        "raw_overall_score": raw_overall,
        "overall_score": overall,
        "constraint_penalty": constraint_penalty,
        "constraint_ceiling": constraint_ceiling,
        "required_coverage_ratio": round(coverage_ratio, 6),
        "uncovered_capabilities": uncovered,
        "utility_weights": weights,
    }
