"""单段医疗文本的任务二抽取服务。

该服务把实体、关系和三元组作为同一次抽取的三个视图返回，避免入口层
重复运行抽取链，也保证“没有关系”仍是一个结构完整的成功结果。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any

from core.medical_extraction_service import VALID_BACKENDS, extract_medical_knowledge


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def validate_text_backend(value: str | None) -> str:
    """校验公开入口的后端名称，禁止无提示地退回离线模式。"""

    backend = str(value or "offline").strip().lower()
    if backend not in VALID_BACKENDS:
        allowed = ", ".join(sorted(VALID_BACKENDS))
        raise ValueError(f"unsupported backend {backend!r}; expected one of: {allowed}")
    return backend


def extract_text_knowledge(
    text: str,
    *,
    backend: str,
    kg_db_path: str,
    llm: Any = None,
) -> dict[str, Any]:
    """执行一次抽取并返回可直接交给 Nexent 的稳定对象契约。"""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    selected = validate_text_backend(backend)
    result = extract_medical_knowledge(
        text,
        backend=selected,
        kg_db_path=kg_db_path,
        llm=llm,
    )
    entities = _jsonable(result.entities)
    relations = _jsonable(result.relations)
    triples = _jsonable(result.triples)
    status = "partial_success" if result.llm_error else "success"
    return {
        "status": status,
        "backend": result.backend,
        "source": {
            "kind": "inline_text",
            "character_count": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        "entities": entities,
        "relations": relations,
        "triples": triples,
        "counts": {
            "entity_count": len(entities),
            "relation_count": len(relations),
            "triple_count": len(triples),
        },
        "score_semantics": {
            "confidence": "grouped_validation_precision_or_llm_review_score",
            "note": (
                "dictionary/rule scores are validation-set precision shared by the same "
                "method and type; they are not per-item probabilities"
            ),
        },
        "cascade": {
            "gap_segment_count": result.gap_segment_count,
            "gap_candidate_count": result.gap_candidate_count,
            "reviewed_candidate_count": result.reviewed_candidate_count,
            "auto_accepted_candidate_count": result.auto_accepted_candidate_count,
            "review_skipped_candidate_count": result.review_skipped_candidate_count,
            "offline_filtered_candidate_count": result.offline_filtered_candidate_count,
            "rejected_candidate_count": result.rejected_candidate_count,
            "llm_added_count": result.llm_added_count,
            "llm_added_entity_count": result.llm_added_entity_count,
            "llm_added_relation_count": result.llm_added_relation_count,
            "gap_budget_skipped_count": result.gap_budget_skipped_count,
            "review_budget_skipped_count": result.review_budget_skipped_count,
        },
        "performance": {
            "elapsed_seconds": result.elapsed_seconds,
            "characters_per_second": round(
                len(text) / result.elapsed_seconds, 4
            )
            if result.elapsed_seconds
            else 0.0,
        },
        "extraction_errors": [result.llm_error] if result.llm_error else [],
        "report_markdown": (
            f"单段医疗文本抽取完成：识别 {len(entities)} 个实体、"
            f"{len(relations)} 条关系和 {len(triples)} 条三元组，"
            f"耗时 {result.elapsed_seconds:.4f} 秒。"
            + (" 模型增强失败，已保留离线结果。" if result.llm_error else "")
        ),
    }
