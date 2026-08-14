"""Tensorized Task-1 DataMate operator ranking with CPU/NPU backends."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from operators.npu_tensor_kg_ops import hashed_vector, make_tensor, select_tensor_device, sync_tensor_device


TASK1_RANKER_VECTOR_DIM = 512
TASK1_CAPABILITY_LABELS = [
    "pii_cleanup",
    "text_normalization",
    "length_filter",
    "quality_filter",
    "deduplication",
    "schema_guard",
    "low_cost",
]


def _operator_text(operator: dict[str, Any]) -> str:
    return " ".join(
        str(operator.get(key) or "")
        for key in ("id", "name", "description", "inputs", "outputs")
    ).strip()


def _canonical_operator_id(operator: dict[str, Any]) -> str:
    return str(operator.get("canonical_id") or operator.get("id") or operator.get("name") or "")


def _operator_cost(operator: dict[str, Any]) -> float:
    explicit = operator.get("estimated_cost")
    if explicit is not None:
        try:
            return max(0.5, float(explicit))
        except (TypeError, ValueError):
            pass
    text = _operator_text(operator).lower()
    cost = 1.0
    if any(token in text for token in ("embedding", "model", "llm", "extract", "parse", "table", "pdf")):
        cost += 2.0
    if any(token in text for token in ("dedup", "fingerprint", "quality", "similarity", "相似度")):
        cost += 0.6
    if any(token in text for token in ("mapper", "filter", "regex", "length")):
        cost -= 0.2
    return round(max(0.5, cost), 3)


@lru_cache(maxsize=32768)
def _cached_hashed_vector(text: str, dim: int) -> tuple[float, ...]:
    return tuple(hashed_vector(text, dim=dim))


def _capability_vector(capabilities: list[str] | set[str] | tuple[str, ...]) -> list[float]:
    capability_set = {str(item) for item in capabilities}
    return [1.0 if label in capability_set else 0.0 for label in TASK1_CAPABILITY_LABELS]


def _operator_capabilities(operator: dict[str, Any]) -> list[str]:
    raw = operator.get("capabilities")
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw]
    text = _operator_text(operator).lower()
    operator_id = str(operator.get("id") or "")
    capabilities: set[str] = set()
    if operator_id in {"EmailNumberCleaner", "AnonymizedUrlCleaner", "AnonymizedIpAddress", "AnonymizedPhoneNumber", "PiiDetector"}:
        capabilities.add("pii_cleanup")
    if operator_id in {"ExtraSpaceCleaner", "UnicodeSpaceCleaner", "InvisibleCharactersCleaner", "FullWidthCharacterCleaner", "TraditionalChineseCleaner"}:
        capabilities.add("text_normalization")
    if operator_id in {"FileWithShortOrLongLengthFilter"}:
        capabilities.update({"length_filter", "quality_filter"})
    if operator_id in {"FileWithHighRepeatPhraseRateFilter", "FileWithHighRepeatWordRateFilter", "FileWithHighSpecialCharRateFilter", "FileWithManySensitiveWordsFilter"}:
        capabilities.add("quality_filter")
    if operator_id in {"DuplicateFilesFilter", "DuplicateSentencesFilter"}:
        capabilities.add("deduplication")
    if any(token in text for token in ("link", "url", "http", "email", "ip", "phone", "pii")):
        capabilities.add("pii_cleanup")
    if any(token in text for token in ("space", "whitespace", "normalize", "clean", "mapper", "remove", "空格", "不可见", "全角", "繁体")):
        capabilities.add("text_normalization")
    if any(token in text for token in ("length", "empty", "short", "long", "字数")):
        capabilities.add("length_filter")
    if any(token in text for token in ("quality", "filter", "alphanumeric", "similarity", "相似度", "检查", "过滤")):
        capabilities.add("quality_filter")
    if any(token in text for token in ("dedup", "duplicate", "fingerprint", "similarity", "相似度", "frequency", "频率")):
        capabilities.add("deduplication")
    if any(token in text for token in ("schema", "field", "csv", "json", "table", "字段")):
        capabilities.add("schema_guard")
    if any(token in text for token in ("fast", "simple", "mapper", "filter")):
        capabilities.add("low_cost")
    return sorted(capabilities)


def rank_task1_operator_candidates(
    task_text: str,
    candidates: list[dict[str, Any]],
    *,
    required_capabilities: list[str] | set[str] | tuple[str, ...] | None = None,
    recommended_operator_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    avoid_operator_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    device: str = "auto",
    vector_dim: int = TASK1_RANKER_VECTOR_DIM,
) -> dict[str, Any]:
    """Rank DataMate operator candidates with the same tensor math on CPU or Ascend NPU."""
    if not candidates:
        selected_device = select_tensor_device(device)
        return {
            "selected_device": selected_device,
            "candidate_count": 0,
            "vector_dim": vector_dim,
            "ranked": [],
        }

    import torch

    selected_device = select_tensor_device(device)
    required = {str(item) for item in (required_capabilities or [])}
    recommended = {str(item) for item in (recommended_operator_ids or [])}
    avoid = {str(item) for item in (avoid_operator_ids or [])}

    task_vector = make_tensor([list(_cached_hashed_vector(task_text, vector_dim))], selected_device)
    candidate_vectors = make_tensor([list(_cached_hashed_vector(_operator_text(item), vector_dim)) for item in candidates], selected_device)
    semantic_scores = torch.matmul(candidate_vectors, task_vector.T).squeeze(1)

    capability_matrix = make_tensor([_capability_vector(_operator_capabilities(item)) for item in candidates], selected_device)
    required_vector = make_tensor([_capability_vector(required)], selected_device).T
    capability_hits = torch.matmul(capability_matrix, required_vector).squeeze(1)
    capability_denominator = max(1, len(required))
    capability_scores = capability_hits / float(capability_denominator)

    recommended_values = [
        1.0 if str(item.get("id") or item.get("name") or "") in recommended or _canonical_operator_id(item) in recommended else 0.0
        for item in candidates
    ]
    avoid_values = [
        1.0 if str(item.get("id") or item.get("name") or "") in avoid or _canonical_operator_id(item) in avoid else 0.0
        for item in candidates
    ]
    text_io_values = [
        (0.5 if "text" in str(item.get("inputs") or "").lower() else 0.0)
        + (0.5 if "text" in str(item.get("outputs") or "").lower() else 0.0)
        for item in candidates
    ]
    cost_values = [min(1.0, _operator_cost(item) / 4.0) for item in candidates]

    recommended_tensor = make_tensor([[value] for value in recommended_values], selected_device).squeeze(1)
    avoid_tensor = make_tensor([[value] for value in avoid_values], selected_device).squeeze(1)
    text_io_tensor = make_tensor([[value] for value in text_io_values], selected_device).squeeze(1)
    cost_tensor = make_tensor([[value] for value in cost_values], selected_device).squeeze(1)

    scores = (
        semantic_scores * 0.36
        + capability_scores * 0.42
        + recommended_tensor * 0.16
        + text_io_tensor * 0.08
        - avoid_tensor * 0.35
        - cost_tensor * 0.07
    )
    sync_tensor_device(selected_device)

    score_values = [float(value) for value in scores.detach().to("cpu").tolist()]
    semantic_values = [float(value) for value in semantic_scores.detach().to("cpu").tolist()]
    capability_values = [float(value) for value in capability_scores.detach().to("cpu").tolist()]
    ranked = []
    for index, (operator, score) in enumerate(zip(candidates, score_values)):
        ranked.append(
            {
                "index": index,
                "id": operator.get("id"),
                "name": operator.get("name"),
                "score": round(score, 6),
                "semantic_score": round(semantic_values[index], 6),
                "capability_score": round(capability_values[index], 6),
                "recommended_bonus": recommended_values[index],
                "avoid_penalty": avoid_values[index],
                "text_io_bonus": text_io_values[index],
                "estimated_cost": _operator_cost(operator),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item.get("id") or item.get("name") or "")))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return {
        "selected_device": selected_device,
        "candidate_count": len(candidates),
        "vector_dim": vector_dim,
        "required_capabilities": sorted(required),
        "recommended_operator_ids": sorted(recommended),
        "avoid_operator_ids": sorted(avoid),
        "ranked": ranked,
    }


def build_operator_descriptor() -> dict[str, Any]:
    return {
        "name": "task1_tensorized_operator_planning_ranker",
        "backend": "cpu_or_ascend_npu",
        "operators": ["operator_candidate_semantic_score", "capability_match_score", "planning_rank_fusion"],
        "input": "DataMate operator market candidates + task intent + required capabilities",
        "output": "ranked operator candidates used by Task-1 DataMate DAG planner",
    }
