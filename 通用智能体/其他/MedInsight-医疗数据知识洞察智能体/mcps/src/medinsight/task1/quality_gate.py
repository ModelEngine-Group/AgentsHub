from __future__ import annotations

from typing import Any


def evaluate_quality_gate(
    profile: dict[str, Any],
    quality: dict[str, Any],
    anomalies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate Task1 data quality and return a deterministic gate decision."""
    rules: list[dict[str, Any]] = []
    recommendations: list[str] = []

    def add_rule(
        rule_id: str,
        name: str,
        status: str,
        observed: Any,
        threshold: str,
        action: str,
    ) -> None:
        rules.append(
            {
                "id": rule_id,
                "name": name,
                "status": status,
                "observed": observed,
                "threshold": threshold,
                "action": action,
            }
        )
        if status in {"warning", "failed"}:
            recommendations.append(action)

    detected_mode = str(profile.get("detected_mode") or "unknown")
    text_coverage = float(profile.get("text_coverage") or 0)
    success_rate = float(quality.get("success_rate") or 0)
    input_records = int(quality.get("input_records") or 0)
    documents = int(quality.get("documents") or 0)
    normalized_records = int(quality.get("normalized_records") or 0)
    segments = int(quality.get("segments") or 0)
    error_count = int(quality.get("error_count") or 0)
    warning_count = int(quality.get("warning_count") or 0)

    if detected_mode in {"text", "hybrid"}:
        if text_coverage < 0.5:
            add_rule(
                "text_coverage",
                "文本覆盖率",
                "failed",
                text_coverage,
                ">= 0.85, hard fail < 0.50",
                "补充 text 字段，或先通过结构化字段重建文本后再执行清洗切分。",
            )
        elif text_coverage < 0.85:
            add_rule(
                "text_coverage",
                "文本覆盖率",
                "warning",
                text_coverage,
                ">= 0.85",
                "检查缺失 text 的记录，并启用字段重建策略。",
            )
        else:
            add_rule("text_coverage", "文本覆盖率", "passed", text_coverage, ">= 0.85", "允许进入文本处理链路。")

    if success_rate < 0.8:
        add_rule(
            "success_rate",
            "处理成功率",
            "failed",
            success_rate,
            ">= 0.95, hard fail < 0.80",
            "阻断自动交付，先修复不可恢复记录。",
        )
    elif success_rate < 0.95:
        add_rule(
            "success_rate",
            "处理成功率",
            "warning",
            success_rate,
            ">= 0.95",
            "允许继续，但需要在报告中标记低成功率并复核异常记录。",
        )
    else:
        add_rule("success_rate", "处理成功率", "passed", success_rate, ">= 0.95", "允许进入下游任务。")

    max_error_ratio = round(error_count / input_records, 4) if input_records else 0.0
    if error_count and max_error_ratio > 0.05:
        add_rule(
            "error_ratio",
            "错误记录占比",
            "failed",
            max_error_ratio,
            "<= 0.05",
            "错误比例过高，停止批量交付并输出错误样本。",
        )
    elif error_count:
        add_rule(
            "error_ratio",
            "错误记录占比",
            "warning",
            max_error_ratio,
            "0 preferred, <= 0.05 tolerated",
            "少量错误已隔离，建议人工抽查被丢弃记录。",
        )
    else:
        add_rule("error_ratio", "错误记录占比", "passed", 0.0, "0 preferred", "无错误记录。")

    if normalized_records and documents < normalized_records:
        add_rule(
            "document_yield",
            "文档生成率",
            "failed",
            {"documents": documents, "normalized_records": normalized_records},
            "documents == normalized_records",
            "文档构建数量不足，检查 loader 或表格转文档算子。",
        )
    else:
        add_rule(
            "document_yield",
            "文档生成率",
            "passed",
            {"documents": documents, "normalized_records": normalized_records},
            "documents == normalized_records",
            "标准化记录已全部转为 Document。",
        )

    avg_segments_per_doc = round(segments / documents, 4) if documents else 0.0
    if documents and avg_segments_per_doc < 1:
        add_rule(
            "segment_density",
            "句子片段密度",
            "warning",
            avg_segments_per_doc,
            ">= 1 segment/document",
            "切分结果偏低，建议检查标点、段落边界或文本清洗规则。",
        )
    elif documents:
        add_rule(
            "segment_density",
            "句子片段密度",
            "passed",
            avg_segments_per_doc,
            ">= 1 segment/document",
            "切分密度正常。",
        )

    failed = [rule for rule in rules if rule["status"] == "failed"]
    warned = [rule for rule in rules if rule["status"] == "warning"]
    status = "failed" if failed else "warning" if warned else "passed"
    decision = {
        "passed": status != "failed",
        "status": status,
        "summary": f"{len(failed)} failed, {len(warned)} warning, {len(rules) - len(failed) - len(warned)} passed",
        "policy": {
            "hard_fail_on_success_rate_below": 0.8,
            "warn_on_success_rate_below": 0.95,
            "warn_on_text_coverage_below": 0.85,
            "hard_fail_on_text_coverage_below": 0.5,
            "hard_fail_on_error_ratio_above": 0.05,
        },
        "rules": rules,
        "recommendations": list(dict.fromkeys(recommendations)),
        "anomaly_sample": anomalies[:5],
        "warning_count": warning_count,
        "error_count": error_count,
    }
    return decision
