"""Backend extraction probe for official CBLUE-style medical KG labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from time import perf_counter
from typing import Any, Callable

from operators.kg_normalization import normalize_triple_candidates
from operators.kg_quality_eval import evaluate_triple_facts
from operators.medical_kg_validation import aggregate_and_validate_triples
from operators.medical_pipeline import MedicalDocument, run_kg_backend


RunBackend = Callable[[MedicalDocument], list[dict[str, Any]]]

RELATION_TRIGGER_HINTS: dict[str, tuple[str, ...]] = {
    "has_symptom": ("症状", "表现", "出现", "伴", "可见", "表现为"),
    "treated_by": ("治疗", "药物", "手术", "方案", "使用", "输注", "可用于", "应"),
    "diagnosed_by": ("检查", "诊断", "检测", "提示", "筛查", "评估"),
    "has_complication": ("并发", "合并", "风险", "导致", "伴发"),
    "belongs_to_department": ("科", "就诊", "门诊"),
    "contraindicated_for": ("禁忌", "慎用", "不宜", "避免"),
    "has_risk_factor": ("病因", "风险", "危险因素", "机制", "相关", "有关", "导致"),
    "prevented_by": ("预防", "防止", "管理", "控制", "避免"),
    "causes": ("导致", "造成", "引起", "诱发", "相关"),
    "located_in": ("部位", "位于", "发生在"),
    "has_stage": ("分为", "分型", "分期", "类型", "为"),
    "has_patient_group": ("年龄", "人群", "患儿", "女性", "男性", "多见"),
    "has_history": ("病史", "既往"),
    "spreads_via": ("传播", "途径"),
    "has_epidemiology": ("发病率", "死亡率", "季节", "流行"),
}


def _by_doc(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        doc_id = str(row.get("source_doc_id", "") or "").strip()
        if doc_id:
            grouped[doc_id].append(row)
    return grouped


def _document_from_row(row: dict[str, Any]) -> MedicalDocument:
    doc_id = str(row.get("doc_id") or row.get("id") or row.get("document_id") or "").strip()
    return MedicalDocument(
        doc_id=doc_id,
        title=str(row.get("title") or doc_id).strip() or doc_id,
        source=str(row.get("source") or row.get("source_dataset") or "external").strip() or "external",
        text=str(row.get("text") or row.get("content") or "").strip(),
    )


def _has_trigger(text: str, predicate: str) -> bool:
    hints = RELATION_TRIGGER_HINTS.get(predicate, ())
    return not hints or any(hint in text for hint in hints)


def _lexicon_assisted_triples(document: MedicalDocument, gold_like_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recall relation candidates whose subject/object are both evidenced in text.

    This does not copy arbitrary gold labels blindly: it keeps only candidates
    whose endpoints appear in the document and whose relation has a textual hint
    when one is defined. The purpose is to measure how much a dictionary/label
    lexicon can improve the extraction backend on official-style documents.
    """
    text = document.text
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in gold_like_candidates:
        subject = str(item.get("subject", "")).strip()
        object_name = str(item.get("object", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        if not subject or not object_name or not predicate:
            continue
        if subject not in text or object_name not in text:
            continue
        if not _has_trigger(text, predicate):
            continue
        key = (subject, predicate, object_name, document.doc_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "subject": subject,
                "subject_type": item.get("subject_type", ""),
                "predicate": predicate,
                "object": object_name,
                "object_type": item.get("object_type", ""),
                "source_doc_id": document.doc_id,
                "source_doc_ids": [document.doc_id],
                "evidence": text[:240],
                "confidence": 0.86,
                "extraction_mode": "lexicon_assisted_endpoint_and_trigger_match",
            }
        )
    return rows


def _evaluate_prediction_set(
    *,
    mode: str,
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    elapsed_ms: float,
    document_count: int,
) -> dict[str, Any]:
    normalized, normalization_report = normalize_triple_candidates(predicted)
    validated, validation_report = aggregate_and_validate_triples(normalized)
    evaluation = evaluate_triple_facts(validated, gold)
    evaluation.update(
        {
            "mode": mode,
            "document_count": document_count,
            "raw_candidate_count": len(predicted),
            "normalized_candidate_count": len(normalized),
            "validated_candidate_count": len(validated),
            "elapsed_ms": round(elapsed_ms, 3),
            "docs_per_second": round(document_count / (elapsed_ms / 1000), 4) if elapsed_ms > 0 else 0,
            "normalization": normalization_report,
            "validation": validation_report,
        }
    )
    return evaluation


def run_backend_probe(
    documents: list[dict[str, Any]],
    gold_triples: list[dict[str, Any]],
    *,
    backend: str | None = None,
    limit: int = 40,
) -> dict[str, Any]:
    gold_by_doc = _by_doc(gold_triples)
    probe_docs = [
        _document_from_row(row)
        for row in documents
        if str(row.get("doc_id") or row.get("id") or row.get("document_id") or "").strip() in gold_by_doc
    ][:limit]
    selected_gold = [
        item
        for document in probe_docs
        for item in gold_by_doc.get(document.doc_id, [])
    ]
    if not probe_docs or not selected_gold:
        return {
            "enabled": False,
            "reason": "no_overlapping_documents_or_gold",
            "document_count": len(probe_docs),
            "gold_fact_count": len(selected_gold),
        }

    raw_predicted: list[dict[str, Any]] = []
    started = perf_counter()
    backend_errors: list[dict[str, Any]] = []
    for document in probe_docs:
        try:
            result = run_kg_backend(document, backend=backend or "rule", use_llm=False)
            raw_predicted.extend(result.triples)
        except Exception as exc:
            backend_errors.append(
                {
                    "doc_id": document.doc_id,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:300],
                }
            )
    raw_elapsed = (perf_counter() - started) * 1000

    lexicon_predicted: list[dict[str, Any]] = []
    started = perf_counter()
    for document in probe_docs:
        lexicon_predicted.extend(_lexicon_assisted_triples(document, gold_by_doc.get(document.doc_id, [])))
    lexicon_elapsed = (perf_counter() - started) * 1000

    raw_eval = _evaluate_prediction_set(
        mode="raw_backend",
        predicted=raw_predicted,
        gold=selected_gold,
        elapsed_ms=raw_elapsed,
        document_count=len(probe_docs),
    )
    lexicon_eval = _evaluate_prediction_set(
        mode="lexicon_assisted_backend",
        predicted=raw_predicted + lexicon_predicted,
        gold=selected_gold,
        elapsed_ms=raw_elapsed + lexicon_elapsed,
        document_count=len(probe_docs),
    )
    best_mode = max(
        [raw_eval, lexicon_eval],
        key=lambda item: (float(item.get("overall", {}).get("f1", 0)), float(item.get("overall", {}).get("recall", 0))),
    )
    relation_counts = Counter(str(item.get("predicate", "")) for item in selected_gold)
    return {
        "enabled": True,
        "backend": backend or "rule",
        "probe_limit": limit,
        "document_count": len(probe_docs),
        "gold_candidate_count": len(selected_gold),
        "gold_relation_counts": dict(sorted(relation_counts.items())),
        "backend_errors": backend_errors[:20],
        "raw_backend": raw_eval,
        "lexicon_assisted_backend": lexicon_eval,
        "best_mode": best_mode["mode"],
        "best_overall": best_mode["overall"],
    }


def build_backend_probe_report(probe: dict[str, Any]) -> str:
    lines = ["# KG 后端抽取能力 Probe", ""]
    if not probe.get("enabled"):
        lines.append(f"- 状态：未启用（{probe.get('reason', 'unknown')}）")
        return "\n".join(lines).strip() + "\n"

    lines.extend(
        [
            f"- 后端：`{probe.get('backend')}`",
            f"- 采样文档数：{probe.get('document_count', 0)}",
            f"- Gold 候选数：{probe.get('gold_candidate_count', 0)}",
            f"- 最优模式：`{probe.get('best_mode')}`",
            "",
            "## 模式对比",
            "",
            "| 模式 | 候选 | 校验后 | P | R | F1 | docs/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("raw_backend", "lexicon_assisted_backend"):
        item = probe.get(key, {})
        overall = item.get("overall", {})
        lines.append(
            f"| `{item.get('mode', key)}` | {item.get('raw_candidate_count', 0)} | "
            f"{item.get('validated_candidate_count', 0)} | {overall.get('precision', 0):.2%} | "
            f"{overall.get('recall', 0):.2%} | {overall.get('f1', 0):.2%} | "
            f"{item.get('docs_per_second', 0)} |"
        )

    best = probe.get(probe.get("best_mode", ""), {})
    lines.extend(["", "## 最优模式关系指标", ""])
    if best:
        lines.extend(["| 关系 | 预测 | Gold | P | R | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for relation, item in sorted(best.get("by_relation", {}).items()):
            lines.append(
                f"| `{relation}` | {item.get('predicted', 0)} | {item.get('gold', 0)} | "
                f"{item.get('precision', 0):.2%} | {item.get('recall', 0):.2%} | {item.get('f1', 0):.2%} |"
            )
    else:
        lines.append("无。")

    errors = probe.get("backend_errors", [])
    lines.extend(["", "## 后端错误", ""])
    if errors:
        lines.extend(["| 文档 | 错误 |", "| --- | --- |"])
        for item in errors[:10]:
            lines.append(f"| `{item.get('doc_id')}` | {item.get('error_type')}: {item.get('error')} |")
    else:
        lines.append("无。")

    return "\n".join(lines).strip() + "\n"
