"""Orchestration layer for the medical knowledge graph demo."""

from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import error, request

from agents.data_processing_agent import run_data_processing_agent
from agents.analysis_agent import run_analysis_agent
from operators.llm_client import chat_completion, get_openai_compat_config
from operators.kg_backend_probe import build_backend_probe_report, run_backend_probe
from operators.medical_pipeline import (
    DISEASE_ALIAS_MAP,
    MedicalDocument,
    answer_question,
    build_charts,
    build_graph_snapshot,
    detect_question_disease,
    run_kg_backend,
    run_tensor_kg_backend_batch,
    select_kg_backend,
)
from operators.kg_normalization import canonicalize_entity, normalize_triple_candidates
from operators.kg_quality_eval import build_quality_eval_report, evaluate_triples_against_gold_file
from operators.medical_kg_validation import aggregate_and_validate_triples


DEFAULT_QUESTIONS = [
    "2型糖尿病有哪些症状和治疗方式？",
    "高血压有哪些并发症？",
    "消化性溃疡需要注意哪些治疗和禁忌？",
]
DEFAULT_DATAMATE_URL = "http://localhost:30000"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _should_use_llm(use_llm: bool | None) -> bool:
    if use_llm is not None:
        return use_llm
    return _env_flag("MEDICAL_DEMO_USE_LLM", default=False)


def _rewrite_answer_with_llm(question: str, rule_answer: str, supporting_triples: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    config = get_openai_compat_config()
    if config is None:
        return rule_answer, {"llm_enabled": False, "llm_used": False}

    triples_text = "\n".join(
        f"- {item['subject']} -{item['predicate']}-> {item['object']}（来源 {item['source_doc_id']}）"
        for item in supporting_triples[:12]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个医疗知识图谱问答助手。"
                "只能基于给定的三元组和参考答案润色，不要新增事实。"
                "输出必须是简洁的中文答案正文，不要列出思考过程。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{question}\n"
                f"参考答案：{rule_answer}\n"
                f"可用三元组：\n{triples_text}\n"
                "请把参考答案润色成更自然的中文，但不要改变事实。"
            ),
        },
    ]
    try:
        refined = chat_completion(
            messages,
            model=config.model,
            temperature=0.2,
            max_tokens=256,
            timeout_seconds=config.timeout_seconds,
        )
    except Exception:
        return rule_answer, {
            "llm_enabled": True,
            "llm_used": False,
            "llm_model": config.model,
        }

    refined = refined.strip()
    if not refined:
        return rule_answer, {
            "llm_enabled": True,
            "llm_used": False,
            "llm_model": config.model,
        }

    return refined, {
        "llm_enabled": True,
        "llm_used": True,
        "llm_model": config.model,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(payload)
    return rows


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _http_json_post(
    *,
    url: str,
    body: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started_at = _now_iso()
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read(200_000).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                payload = None
            return {
                "name": "DataMate live knowledge-base retrieve",
                "method": "POST",
                "url": url,
                "started_at": started_at,
                "status_code": response.status,
                "ok": 200 <= response.status < 300,
                "body_json": payload,
                "body_preview": raw[:2000] if payload is None else "",
                "error": "",
            }
    except error.HTTPError as exc:
        raw = exc.read(200_000).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = None
        return {
            "name": "DataMate live knowledge-base retrieve",
            "method": "POST",
            "url": url,
            "started_at": started_at,
            "status_code": exc.code,
            "ok": False,
            "body_json": payload,
            "body_preview": raw[:2000] if payload is None else "",
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "name": "DataMate live knowledge-base retrieve",
            "method": "POST",
            "url": url,
            "started_at": started_at,
            "status_code": None,
            "ok": False,
            "body_json": None,
            "body_preview": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _keyword_overlap_score(question: str, text: str, disease: str | None = None) -> float:
    text = text or ""
    keywords = [
        token
        for token in (
            disease,
            "高血压" if "高血压" in question else "",
            "糖尿病" if "糖尿病" in question else "",
            "并发症" if "并发" in question or "风险" in question else "",
            "症状" if "症状" in question or "表现" in question else "",
            "治疗" if "治疗" in question or "用药" in question else "",
            "检查" if "检查" in question or "诊断" in question else "",
            "禁忌" if "禁忌" in question or "慎用" in question else "",
            "预防" if "预防" in question or "管理" in question else "",
            "因果" if "原因" in question or "导致" in question or "因果" in question else "",
        )
        if token
    ]
    if not keywords:
        keywords = [item for item in question.replace("？", "").replace("?", "").split() if len(item) >= 2]
    if not keywords:
        return 0.0
    matched = sum(1 for keyword in dict.fromkeys(keywords) if keyword in text)
    return matched / max(len(dict.fromkeys(keywords)), 1)


def _disease_aliases(disease: str | None) -> list[str]:
    if not disease:
        return []
    aliases = list(DISEASE_ALIAS_MAP.get(disease, []))
    aliases.append(disease)
    return [item for item in dict.fromkeys(aliases) if item]


def _is_datamate_item_relevant(item: dict[str, Any], question: str, disease: str | None) -> bool:
    text = _datamate_item_text(item)
    aliases = _disease_aliases(disease)
    if aliases:
        return any(alias in text for alias in aliases)
    return _keyword_overlap_score(question, text, disease) >= 0.5


def _datamate_base_url() -> str:
    return (
        os.getenv("DATAMATE_URL")
        or os.getenv("DATAMATE_API_BASE")
        or DEFAULT_DATAMATE_URL
    ).rstrip("/")


def _extract_kb_ids_from_platform_evidence(output_root: Path) -> list[str]:
    platform = _read_json_file(output_root / "demo" / "integration" / "platform_evidence.json")
    ids: list[str] = []
    for value in platform.get("knowledge_base_ids", []):
        if isinstance(value, str) and value.strip():
            ids.append(value.strip())
    for trace in platform.get("datamate_search_trace", []):
        if not isinstance(trace, dict):
            continue
        for value in trace.get("knowledge_base_ids", []):
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())
    return [item for item in dict.fromkeys(ids) if item]


def _resolve_datamate_kb_ids(output_root: Path, knowledge_base_id: str | None) -> list[str]:
    ids: list[str] = []
    if knowledge_base_id and knowledge_base_id.strip():
        ids.append(knowledge_base_id.strip())
    env_value = os.getenv("DATAMATE_KNOWLEDGE_BASE_ID", "")
    if env_value.strip():
        ids.extend(item.strip() for item in env_value.split(",") if item.strip())
    ids.extend(_extract_kb_ids_from_platform_evidence(output_root))
    return [item for item in dict.fromkeys(ids) if item]


def _datamate_result_count(call: dict[str, Any]) -> int:
    return len(_unwrap_datamate_payload(call.get("body_json")))


def _live_datamate_retrieve(
    *,
    output_root: Path,
    question: str,
    knowledge_base_id: str | None,
    timeout_seconds: float,
    top_k: int,
) -> dict[str, Any]:
    kb_ids = _resolve_datamate_kb_ids(output_root, knowledge_base_id)
    if not kb_ids:
        return {
            "status": "skipped",
            "reason": "knowledge_base_id_missing",
            "query": question,
            "knowledge_base_ids": [],
            "calls": [],
        }

    url = _join_url(_datamate_base_url(), "/api/knowledge-base/v2/retrieve")
    call = _http_json_post(
        url=url,
        body={
            "query": question,
            "topK": top_k,
            "threshold": 0.2,
            "knowledgeBaseIds": kb_ids,
        },
        timeout_seconds=timeout_seconds,
    )
    trace = {
        "agent": "hybrid_medical_qa_agent",
        "tool": "datamate_live_retrieve",
        "status": "success" if call.get("ok") and _datamate_result_count(call) > 0 else "failed",
        "query": question,
        "knowledge_base_ids": kb_ids,
        "tool_input": {
            "query": question,
            "top_k": top_k,
            "threshold": 0.2,
            "knowledge_base_ids": kb_ids,
        },
        "backend": {
            "endpoint": url,
            "method": "POST",
            "source": "DataMate /api/knowledge-base/v2/retrieve",
        },
        "response_status": call.get("status_code"),
        "result_count": _datamate_result_count(call),
        "error": call.get("error", ""),
        "created_at": _now_iso(),
    }
    trace_path = output_root / "demo" / "integration" / "datamate_live_retrieve_trace.jsonl"
    rows = _read_jsonl(trace_path) if trace_path.exists() else []
    rows.append(trace)
    _write_jsonl(trace_path, rows[-200:])
    return {
        "status": trace["status"],
        "query": question,
        "knowledge_base_ids": kb_ids,
        "calls": [call],
        "trace": trace,
    }


def _unwrap_datamate_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            for key in ("content", "items", "records", "list"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _datamate_item_text(item: dict[str, Any]) -> str:
    entity = item.get("entity")
    if isinstance(entity, dict):
        for key in ("text", "content", "pageContent", "chunk"):
            value = entity.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("text", "content", "pageContent", "chunk"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(item, ensure_ascii=False)[:800]


def _datamate_item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    entity = item.get("entity")
    if isinstance(entity, dict):
        raw_metadata = entity.get("metadata")
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    metadata.update(parsed)
            except json.JSONDecodeError:
                metadata["metadata_text"] = raw_metadata
        elif isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
    for key in ("knowledgeBaseId", "knowledgeBaseName", "id"):
        if item.get(key) is not None:
            metadata[key] = item.get(key)
    return metadata


def _score_datamate_item(item: dict[str, Any], question: str, disease: str | None) -> float:
    text = _datamate_item_text(item)
    overlap = _keyword_overlap_score(question, text, disease)
    raw_score = item.get("score")
    retriever_score = 0.5
    if isinstance(raw_score, int | float):
        # DataMate deployments may expose either a distance-like score or a similarity-like score.
        if 0 <= raw_score <= 1:
            retriever_score = max(float(raw_score), 1.0 - float(raw_score))
        else:
            retriever_score = 1.0 / (1.0 + abs(float(raw_score)))
    return round(min(1.0, 0.62 * overlap + 0.38 * retriever_score), 4)


def _build_datamate_evidence_payload(
    *,
    question: str,
    disease: str | None,
    retrieval_items: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    selected_call: dict[str, Any] | None,
    source_mode: str,
    limit: int,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    filtered_irrelevant = 0
    for index, item in enumerate(retrieval_items, start=1):
        text = _datamate_item_text(item)
        if not text or text in seen_text:
            continue
        if not _is_datamate_item_relevant(item, question, disease):
            filtered_irrelevant += 1
            continue
        seen_text.add(text)
        evidence.append(
            {
                "rank": len(evidence) + 1,
                "source": "datamate",
                "source_mode": source_mode,
                "text": text[:1200],
                "score": _score_datamate_item(item, question, disease),
                "raw_score": item.get("score"),
                "metadata": _datamate_item_metadata(item),
                "retrieval_index": index,
            }
        )
        if len(evidence) >= limit:
            break

    best_trace = trace_rows[-1] if trace_rows else {}
    return {
        "status": "success" if evidence else "unavailable",
        "source_mode": source_mode,
        "query": question,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "knowledge_base_ids": best_trace.get("knowledge_base_ids", []),
        "retriever_status": best_trace.get("status") or ("success" if evidence else "missing"),
        "result_count": best_trace.get("result_count", len(retrieval_items)),
        "filtered_irrelevant": filtered_irrelevant,
        "source_call": {
            "name": selected_call.get("name") if isinstance(selected_call, dict) else None,
            "status_code": selected_call.get("status_code") if isinstance(selected_call, dict) else None,
            "ok": selected_call.get("ok") if isinstance(selected_call, dict) else None,
        },
    }


def _load_datamate_retrieval_evidence(
    output_root: Path,
    question: str,
    *,
    knowledge_base_id: str | None = None,
    enable_live: bool = True,
    timeout_seconds: float = 1.5,
    limit: int = 5,
) -> dict[str, Any]:
    integration_dir = output_root / "demo" / "integration"
    platform_path = integration_dir / "platform_evidence.json"
    trace_path = integration_dir / "datamate_search_trace.jsonl"
    disease = detect_question_disease(question)
    calls: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    if enable_live:
        live = _live_datamate_retrieve(
            output_root=output_root,
            question=question,
            knowledge_base_id=knowledge_base_id,
            timeout_seconds=timeout_seconds,
            top_k=limit,
        )
        if live.get("calls"):
            live_call = live["calls"][-1]
            live_items = _unwrap_datamate_payload(live_call.get("body_json"))
            live_trace = live.get("trace")
            live_trace_rows = [live_trace] if isinstance(live_trace, dict) else []
            payload = _build_datamate_evidence_payload(
                question=question,
                disease=disease,
                retrieval_items=live_items,
                trace_rows=live_trace_rows,
                selected_call=live_call,
                source_mode="live",
                limit=limit,
            )
            payload["live_retrieve"] = {
                "attempted": True,
                "status": live.get("status"),
                "error": live_call.get("error", ""),
                "response_status": live_call.get("status_code"),
            }
            if payload["evidence"]:
                return payload
        elif live.get("status") == "skipped":
            live_skip = live
        else:
            live_skip = {}
    else:
        live_skip = {"status": "disabled"}

    if platform_path.exists():
        payload = json.loads(platform_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            calls.extend(item for item in payload.get("datamate_api_calls", []) if isinstance(item, dict))
            calls.extend(item for item in payload.get("datamate_bootstrap_calls", []) if isinstance(item, dict))
            trace_rows.extend(item for item in payload.get("datamate_search_trace", []) if isinstance(item, dict))
    if trace_path.exists():
        trace_rows.extend(_read_jsonl(trace_path))

    retrieval_items: list[dict[str, Any]] = []
    selected_call: dict[str, Any] | None = None
    for call in reversed(calls):
        name = str(call.get("name", ""))
        if "retrieve" not in name.lower():
            continue
        items = _unwrap_datamate_payload(call.get("body_json"))
        if not items:
            continue
        selected_call = call
        retrieval_items = items
        break

    if not retrieval_items:
        for row in reversed(trace_rows):
            embedded = row.get("results") or row.get("documents") or row.get("items")
            if isinstance(embedded, list):
                retrieval_items = [item for item in embedded if isinstance(item, dict)]
                selected_call = row
                break

    payload = _build_datamate_evidence_payload(
        question=question,
        disease=disease,
        retrieval_items=retrieval_items,
        trace_rows=trace_rows,
        selected_call=selected_call,
        source_mode="cached_trace",
        limit=limit,
    )
    payload["live_retrieve"] = {
        "attempted": enable_live,
        "status": live_skip.get("status", "failed") if isinstance(live_skip, dict) else "failed",
        "reason": live_skip.get("reason", "") if isinstance(live_skip, dict) else "",
    }
    return payload


def _score_kg_triple(triple: dict[str, Any], question: str, disease: str | None) -> float:
    evidence_text = " ".join(
        str(triple.get(key, ""))
        for key in ("subject", "predicate", "object", "evidence")
    )
    overlap = _keyword_overlap_score(question, evidence_text, disease)
    try:
        confidence = float(triple.get("confidence", 0.75))
    except (TypeError, ValueError):
        confidence = 0.75
    return round(min(1.0, 0.55 * confidence + 0.45 * overlap), 4)


def _build_hybrid_qa_trace(
    *,
    output_root: Path,
    question: str,
    qa_trace: dict[str, Any],
    knowledge_base_id: str | None = None,
    enable_live_datamate: bool = False,
    datamate_timeout_seconds: float = 1.5,
) -> dict[str, Any]:
    disease = detect_question_disease(question)
    datamate = _load_datamate_retrieval_evidence(
        output_root,
        question,
        knowledge_base_id=knowledge_base_id,
        enable_live=enable_live_datamate,
        timeout_seconds=datamate_timeout_seconds,
    )
    kg_evidence = [
        {
            "rank": index,
            "source": "kg",
            "subject": triple.get("subject"),
            "predicate": triple.get("predicate"),
            "object": triple.get("object"),
            "source_doc_id": triple.get("source_doc_id"),
            "evidence": triple.get("evidence", ""),
            "confidence": triple.get("confidence", 0),
            "score": max(_score_kg_triple(triple, question, disease), float(triple.get("qa_score", 0) or 0)),
            "qa_score": triple.get("qa_score"),
            "qa_score_features": triple.get("qa_score_features", {}),
        }
        for index, triple in enumerate(qa_trace.get("supporting_triples", []), start=1)
    ]
    reasoning_paths = qa_trace.get("reasoning_paths", [])
    datamate_evidence = datamate["evidence"]
    fused = sorted(
        kg_evidence + datamate_evidence,
        key=lambda item: (float(item.get("score", 0)), 1 if item.get("source") == "kg" else 0),
        reverse=True,
    )[:12]
    source_counts = Counter(str(item.get("source", "unknown")) for item in fused)
    fusion_score = round(
        sum(float(item.get("score", 0)) for item in fused) / len(fused),
        4,
    ) if fused else 0.0
    answer = qa_trace["answer"]
    if datamate_evidence and "DataMate" not in answer:
        answer = f"{answer} DataMate 检索证据与图谱结论一致，已用于补充校验。"

    return {
        "agent": "hybrid_medical_qa_agent",
        "goal": question,
        "plan": [
            "理解问题意图",
            "DataMate 知识库检索",
            "知识图谱子图召回",
            "多源证据融合打分",
            "生成带引用答案",
        ],
        "steps": [
            {"step": "understand_question", "status": "success", "focus_disease": disease},
            {
                "step": "datamate_retrieve",
                "status": datamate["status"],
                "result_count": datamate["evidence_count"],
                "knowledge_base_ids": datamate.get("knowledge_base_ids", []),
                "source_mode": datamate.get("source_mode"),
                "live_retrieve": datamate.get("live_retrieve", {}),
            },
            {
                "step": "kg_subgraph_retrieve",
                "status": "success" if kg_evidence else "empty",
                "result_count": len(kg_evidence),
            },
            {
                "step": "kg_path_reason",
                "status": "success" if reasoning_paths else "empty",
                "path_count": len(reasoning_paths),
            },
            {
                "step": "evidence_fusion",
                "status": "success" if fused else "empty",
                "fusion_score": fusion_score,
                "source_balance": dict(source_counts),
            },
            {"step": "answer_generate", "status": "success", "answer_length": len(answer)},
        ],
        "question": question,
        "answer": answer,
        "focus_disease": disease,
        "datamate_retrieval": datamate,
        "kg_evidence": kg_evidence,
        "ranked_supporting_triples": qa_trace.get("ranked_supporting_triples", []),
        "reasoning_paths": reasoning_paths,
        "fused_evidence": fused,
        "fusion_score": fusion_score,
        "source_balance": dict(source_counts),
    }


def _build_analysis_report(
    processed_documents: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    qa_traces: list[dict[str, Any]],
) -> str:
    entity_type_counts = Counter(item["type"] for item in entities)
    relation_counts = Counter(item["predicate"] for item in triples)
    top_entities = Counter(item["name"] for item in entities).most_common(8)

    lines: list[str] = []
    lines.append("# 医疗 Demo 分析报告")
    lines.append("")
    lines.append(f"- 文档数：{len(processed_documents)}")
    lines.append(f"- 实体数：{len(entities)}")
    lines.append(f"- 三元组数：{len(triples)}")
    lines.append(f"- 问答样例数：{len(qa_traces)}")
    lines.append("")
    lines.append("## 实体类型分布")
    lines.append("| 类型 | 数量 |")
    lines.append("| --- | ---: |")
    for name, value in sorted(entity_type_counts.items()):
        lines.append(f"| {name} | {value} |")
    lines.append("")
    lines.append("## 关系类型分布")
    lines.append("| 关系 | 数量 |")
    lines.append("| --- | ---: |")
    for name, value in sorted(relation_counts.items()):
        lines.append(f"| {name} | {value} |")
    lines.append("")
    lines.append("## 高频实体")
    lines.append("| 实体 | 次数 |")
    lines.append("| --- | ---: |")
    for name, value in top_entities:
        lines.append(f"| {name} | {value} |")
    lines.append("")
    lines.append("## 示例问答")
    for trace in qa_traces:
        lines.append(f"### {trace['question']}")
        lines.append(trace["answer"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_kg_quality_report(
    *,
    processed_documents: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    kg_validation_report: dict[str, Any],
    qa_traces: list[dict[str, Any]],
) -> str:
    entity_type_counts = Counter(item["type"] for item in entities)
    relation_counts = Counter(item["predicate"] for item in triples)
    evidence_count = sum(1 for item in triples if str(item.get("evidence", "")).strip())
    source_doc_ids = {
        doc_id
        for item in triples
        for doc_id in item.get("source_doc_ids", [item.get("source_doc_id", "")])
        if str(doc_id).strip()
    }
    low_confidence = [item for item in triples if float(item.get("confidence", 0)) < 0.8]
    qa_supported = sum(1 for item in qa_traces if item.get("supporting_triples"))
    normalization = kg_validation_report.get("normalization", {})

    lines = [
        "# 任务二知识图谱质量报告",
        "",
        f"- 文档数：{len(processed_documents)}",
        f"- 实体数：{len(entities)}",
        f"- 三元组数：{len(triples)}",
        f"- 候选三元组数：{kg_validation_report.get('candidate_triples', 0)}",
        f"- 接受三元组数：{kg_validation_report.get('accepted_triples', 0)}",
        f"- 拒绝三元组数：{kg_validation_report.get('rejected_triples', 0)}",
        f"- 归一化改写三元组数：{normalization.get('changed_triples', 0)}",
        f"- 归一化丢弃三元组数：{normalization.get('dropped_triples', 0)}",
        f"- 冲突过滤三元组数：{kg_validation_report.get('conflict_filtered_triples', 0)}",
        f"- 三元组 evidence 覆盖率：{(evidence_count / len(triples)):.2%}" if triples else "- 三元组 evidence 覆盖率：0.00%",
        f"- 三元组文档覆盖数：{len(source_doc_ids)}",
        f"- 有引用问答数：{qa_supported}/{len(qa_traces)}",
        f"- 低置信度三元组数：{len(low_confidence)}",
        "",
        "## 实体类型覆盖",
        "",
        "| 类型 | 数量 |",
        "| --- | ---: |",
    ]
    for name, value in sorted(entity_type_counts.items()):
        lines.append(f"| {name} | {value} |")

    lines.extend(["", "## 关系类型覆盖", "", "| 关系 | 数量 |", "| --- | ---: |"])
    for name, value in sorted(relation_counts.items()):
        lines.append(f"| {name} | {value} |")

    lines.extend(["", "## 归一化与校准", ""])
    lines.extend(["| 指标 | 数值 |", "| --- | ---: |"])
    lines.append(f"| 输入候选三元组 | {normalization.get('input_triples', 0)} |")
    lines.append(f"| 输出候选三元组 | {normalization.get('output_triples', 0)} |")
    lines.append(f"| 改写三元组 | {normalization.get('changed_triples', 0)} |")
    lines.append(f"| 丢弃三元组 | {normalization.get('dropped_triples', 0)} |")
    for reason, count in sorted(normalization.get("reason_counts", {}).items()):
        lines.append(f"| `{reason}` | {count} |")

    lines.extend(["", "## 问答引用覆盖", ""])
    lines.extend(["| 问题 | 引用三元组数 |", "| --- | ---: |"])
    for trace in qa_traces:
        lines.append(f"| {trace['question']} | {len(trace.get('supporting_triples', []))} |")

    lines.extend(["", "## 低置信度样例", ""])
    if low_confidence:
        lines.extend(["| 主语 | 关系 | 宾语 | 置信度 |", "| --- | --- | --- | ---: |"])
        for item in low_confidence[:20]:
            lines.append(
                f"| {item['subject']} | `{item['predicate']}` | {item['object']} | {item['confidence']} |"
            )
    else:
        lines.append("无低于 0.8 的三元组。")

    return "\n".join(lines).strip() + "\n"


def _normalize_external_document(row: dict[str, Any], index: int) -> dict[str, Any]:
    doc_id = str(row.get("doc_id") or row.get("id") or row.get("document_id") or f"external-doc-{index:06d}").strip()
    text = str(row.get("text") or row.get("content") or "").strip()
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#0",
        "title": str(row.get("title") or doc_id).strip() or doc_id,
        "source": str(row.get("source") or row.get("source_dataset") or "external").strip() or "external",
        "text": text,
        "sections": [text] if text else [],
        "quality": {
            "length": len(text),
            "language": "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "unknown",
            "sentence_count": 1 if text else 0,
            "section_count": 1 if text else 0,
            "source_format": "external_jsonl",
            "field_count": len(row),
        },
        "raw_fields": sorted(row.keys()),
    }


def _merge_entity(
    entity_pool: dict[tuple[str, str], dict[str, Any]],
    *,
    name: str,
    entity_type: str,
    aliases: list[str] | set[str],
    source_doc_ids: list[str] | set[str],
    confidence: float,
) -> None:
    canonical_name, canonical_aliases = canonicalize_entity(name, entity_type)
    alias_values = set(aliases)
    alias_values.update(canonical_aliases)
    key = (canonical_name, entity_type)
    bucket = entity_pool.get(key)
    if bucket is None:
        bucket = {
            "name": canonical_name,
            "type": entity_type,
            "aliases": set(),
            "source_doc_ids": set(),
            "confidence": confidence,
        }
        entity_pool[key] = bucket
    bucket["aliases"].update(alias for alias in alias_values if alias)
    bucket["source_doc_ids"].update(doc_id for doc_id in source_doc_ids if doc_id)
    bucket["confidence"] = max(bucket["confidence"], confidence)


def _build_entity_linking_report(entities: list[dict[str, Any]], triples: list[dict[str, Any]]) -> dict[str, Any]:
    alias_rich_entities = [item for item in entities if len(item.get("aliases", [])) > 1]
    source_link_counts = [len(item.get("source_doc_ids", [])) for item in entities]
    connected_keys = {
        (item.get("subject"), item.get("subject_type"))
        for item in triples
    } | {
        (item.get("object"), item.get("object_type"))
        for item in triples
    }
    entity_keys = {(item.get("name"), item.get("type")) for item in entities}
    isolated_entities = [
        {"name": name, "type": entity_type}
        for name, entity_type in sorted(entity_keys - connected_keys, key=lambda item: (str(item[1]), str(item[0])))
        if name and entity_type
    ]
    alias_type_counts = Counter(str(item.get("type", "unknown")) for item in alias_rich_entities)
    return {
        "entity_count": len(entities),
        "linked_entity_count": len(connected_keys & entity_keys),
        "isolated_entity_count": len(isolated_entities),
        "isolated_entity_samples": isolated_entities[:20],
        "alias_rich_entity_count": len(alias_rich_entities),
        "alias_rich_entity_samples": [
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "aliases": item.get("aliases", [])[:8],
            }
            for item in alias_rich_entities[:20]
        ],
        "alias_type_counts": dict(sorted(alias_type_counts.items())),
        "source_linking": {
            "avg_source_docs_per_entity": round(sum(source_link_counts) / len(source_link_counts), 4)
            if source_link_counts else 0,
            "max_source_docs_per_entity": max(source_link_counts) if source_link_counts else 0,
        },
    }


def _run_kg_backend_batch(
    documents: list[MedicalDocument],
    *,
    kg_backend: str | None,
    use_llm: bool,
    backend_selection: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_count = 1
    selected_backend = getattr(backend_selection, "selected_backend", "rule")
    if selected_backend == "tensor":
        started = perf_counter()
        backend_results = run_tensor_kg_backend_batch(documents)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        avg_elapsed_ms = round(elapsed_ms / len(documents), 3) if documents else 0
        rows = [
            {
                "index": index,
                "document": document,
                "result": backend_result,
                "elapsed_ms": avg_elapsed_ms,
            }
            for index, (document, backend_result) in enumerate(zip(documents, backend_results))
        ]
        dispatch = {
            "scheduler": "tensor_batch_document_runner",
            "parallel": False,
            "worker_count": 1,
            "document_count": len(documents),
            "elapsed_ms": elapsed_ms,
            "docs_per_second": round(len(documents) / (elapsed_ms / 1000), 4) if elapsed_ms > 0 else 0,
            "avg_doc_elapsed_ms": avg_elapsed_ms,
            "max_doc_elapsed_ms": avg_elapsed_ms,
            "preserves_input_order": True,
            "tensor_batch": True,
        }
        if backend_results:
            tensor_operator = backend_results[0].metadata.get("tensor_operator", {})
            dispatch["tensor_device"] = tensor_operator.get("selected_device")
            dispatch["tensor_vector_dim"] = tensor_operator.get("vector_dim")
            dispatch["tensor_entity_candidate_count"] = tensor_operator.get("entity_candidate_count")
            dispatch["tensor_relation_candidate_count"] = tensor_operator.get("relation_candidate_count")
        return rows, dispatch

    if selected_backend == "rule" and len(documents) > 1:
        worker_count = min(4, len(documents), max(2, (os.cpu_count() or 2)))
    scheduler = "thread_pool_document_parallel" if worker_count > 1 else "sequential_document_runner"
    started = perf_counter()

    def run_one(index_and_document: tuple[int, MedicalDocument]) -> dict[str, Any]:
        index, document = index_and_document
        item_started = perf_counter()
        backend_result = run_kg_backend(document, backend=kg_backend, use_llm=use_llm)
        elapsed_ms = round((perf_counter() - item_started) * 1000, 3)
        return {
            "index": index,
            "document": document,
            "result": backend_result,
            "elapsed_ms": elapsed_ms,
        }

    indexed_documents = list(enumerate(documents))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(run_one, indexed_documents))
    else:
        rows = [run_one(item) for item in indexed_documents]
    rows.sort(key=lambda item: int(item["index"]))
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    doc_times = [float(item["elapsed_ms"]) for item in rows]
    dispatch = {
        "scheduler": scheduler,
        "parallel": worker_count > 1,
        "worker_count": worker_count,
        "document_count": len(documents),
        "elapsed_ms": elapsed_ms,
        "docs_per_second": round(len(documents) / (elapsed_ms / 1000), 4) if elapsed_ms > 0 else 0,
        "avg_doc_elapsed_ms": round(sum(doc_times) / len(doc_times), 3) if doc_times else 0,
        "max_doc_elapsed_ms": round(max(doc_times), 3) if doc_times else 0,
        "preserves_input_order": True,
    }
    return rows, dispatch


def _build_kg_operator_profile(
    *,
    processed_documents: list[dict[str, Any]],
    triple_candidates: list[dict[str, Any]],
    normalized_triple_candidates: list[dict[str, Any]],
    triple_rows: list[dict[str, Any]],
    entity_rows: list[dict[str, Any]],
    normalization_report: dict[str, Any],
    kg_validation_report: dict[str, Any],
    backend_selection: Any,
    kg_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    entity_counts = [int(item.get("entity_count", 0)) for item in kg_trace if item.get("step") == "entity_extract"]
    relation_counts = [int(item.get("triple_candidate_count", 0)) for item in kg_trace if item.get("step") == "relation_extract"]
    relation_mix = Counter(str(item.get("predicate", "unknown")) for item in triple_rows)
    evidence_fusion = kg_validation_report.get("evidence_fusion", {})
    extraction_dispatch = next((item for item in kg_trace if item.get("step") == "kg_document_batch_extract"), {})
    return {
        "agent": "medical_kg_agent",
        "backend": {
            "requested": getattr(backend_selection, "requested_backend", None),
            "selected": getattr(backend_selection, "selected_backend", None),
            "fallback_reason": getattr(backend_selection, "fallback_reason", None),
            "llm_configured": getattr(backend_selection, "llm_configured", False),
        },
        "operators": [
            {
                "name": "entity_extract",
                "module": "operators.medical_pipeline.run_kg_backend",
                "input_unit": "document",
                "input_count": len(processed_documents),
                "output_count": sum(entity_counts),
                "avg_output_per_doc": round(sum(entity_counts) / len(entity_counts), 4) if entity_counts else 0,
                "scheduler": extraction_dispatch.get("scheduler", "unknown"),
                "parallel": extraction_dispatch.get("parallel", False),
                "worker_count": extraction_dispatch.get("worker_count", 1),
                "docs_per_second": extraction_dispatch.get("docs_per_second", 0),
                "reusable": True,
            },
            {
                "name": "relation_extract",
                "module": "operators.medical_pipeline.extract_triples",
                "input_unit": "document",
                "input_count": len(processed_documents),
                "output_count": len(triple_candidates),
                "avg_output_per_doc": round(sum(relation_counts) / len(relation_counts), 4) if relation_counts else 0,
                "scheduler": extraction_dispatch.get("scheduler", "unknown"),
                "parallel": extraction_dispatch.get("parallel", False),
                "worker_count": extraction_dispatch.get("worker_count", 1),
                "docs_per_second": extraction_dispatch.get("docs_per_second", 0),
                "reusable": True,
            },
            {
                "name": "triple_normalize",
                "module": "operators.kg_normalization.normalize_triple_candidates",
                "input_unit": "candidate_triple",
                "input_count": len(triple_candidates),
                "output_count": len(normalized_triple_candidates),
                "changed_count": normalization_report.get("changed_triples", 0),
                "dropped_count": normalization_report.get("dropped_triples", 0),
                "reusable": True,
            },
            {
                "name": "triple_validate_arbitrate",
                "module": "operators.medical_kg_validation.aggregate_and_validate_triples",
                "input_unit": "normalized_triple",
                "input_count": len(normalized_triple_candidates),
                "output_count": len(triple_rows),
                "rejected_count": kg_validation_report.get("rejected_triples", 0),
                "duplicate_merge_count": kg_validation_report.get("duplicate_triples", 0),
                "conflict_filtered_count": kg_validation_report.get("conflict_filtered_triples", 0),
                "reusable": True,
            },
            {
                "name": "entity_link",
                "module": "agents.medical_kg_agent._merge_entity",
                "input_unit": "entity_or_triple_endpoint",
                "input_count": len(entity_rows),
                "output_count": len(entity_rows),
                "reusable": True,
            },
        ],
        "quality": {
            "accepted_triples": len(triple_rows),
            "entity_count": len(entity_rows),
            "relation_diversity": len(relation_mix),
            "relation_type_counts": dict(sorted(relation_mix.items())),
            "avg_support_score": evidence_fusion.get("support_score", {}).get("avg", 0),
            "multi_source_fact_count": evidence_fusion.get("multi_source_fact_count", 0),
        },
        "trace_step_count": len(kg_trace),
    }


def _merge_external_artifacts(
    *,
    processed_documents: list[dict[str, Any]],
    triple_candidates: list[dict[str, Any]],
    entity_pool: dict[tuple[str, str], dict[str, Any]],
    kg_trace: list[dict[str, Any]],
    external_documents_path: Path | None,
    external_triples_path: Path | None,
    external_entities_path: Path | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "external_document_count": 0,
        "external_triple_count": 0,
        "external_entity_count": 0,
        "paths": {},
    }

    existing_doc_ids = {str(item.get("doc_id", "")) for item in processed_documents}

    if external_documents_path is not None:
        rows = _read_jsonl(external_documents_path)
        summary["paths"]["documents"] = str(external_documents_path)
        for index, row in enumerate(rows, start=1):
            document = _normalize_external_document(row, index)
            if document["doc_id"] in existing_doc_ids:
                continue
            processed_documents.append(document)
            existing_doc_ids.add(document["doc_id"])
        summary["external_document_count"] = len(rows)

    if external_triples_path is not None:
        rows = _read_jsonl(external_triples_path)
        summary["paths"]["triples"] = str(external_triples_path)
        triple_candidates.extend(rows)
        summary["external_triple_count"] = len(rows)
        for row in rows:
            subject = str(row.get("subject", "")).strip()
            subject_type = str(row.get("subject_type", "")).strip()
            object_name = str(row.get("object", "")).strip()
            object_type = str(row.get("object_type", "")).strip()
            source_doc_ids = row.get("source_doc_ids")
            if not isinstance(source_doc_ids, list):
                source_doc_ids = [str(row.get("source_doc_id", "")).strip()]
            confidence = float(row.get("confidence", 0.84))
            if subject and subject_type:
                _merge_entity(
                    entity_pool,
                    name=subject,
                    entity_type=subject_type,
                    aliases=[subject],
                    source_doc_ids=source_doc_ids,
                    confidence=confidence,
                )
            if object_name and object_type:
                _merge_entity(
                    entity_pool,
                    name=object_name,
                    entity_type=object_type,
                    aliases=[object_name],
                    source_doc_ids=source_doc_ids,
                    confidence=confidence,
                )

    if external_entities_path is not None:
        rows = _read_jsonl(external_entities_path)
        summary["paths"]["entities"] = str(external_entities_path)
        summary["external_entity_count"] = len(rows)
        for row in rows:
            name = str(row.get("name") or row.get("entity") or "").strip()
            entity_type = str(row.get("type") or row.get("entity_type") or "").strip()
            if not name or not entity_type:
                continue
            aliases = row.get("aliases")
            if not isinstance(aliases, list):
                aliases = [name]
            source_doc_ids = row.get("source_doc_ids")
            if not isinstance(source_doc_ids, list):
                source_doc_ids = [str(row.get("source_doc_id", "")).strip()]
            _merge_entity(
                entity_pool,
                name=name,
                entity_type=entity_type,
                aliases=[str(item) for item in aliases],
                source_doc_ids=[str(item) for item in source_doc_ids],
                confidence=float(row.get("confidence", 0.86)),
            )

    if any(summary[key] for key in ("external_document_count", "external_triple_count", "external_entity_count")):
        kg_trace.append(
            {
                "agent": "medical_kg_agent",
                "goal": "从标准化医疗文本构建知识图谱",
                "step": "external_artifact_merge",
                "status": "success",
                **summary,
            }
        )
    return summary


def _build_qa_trace(question: str, triples: list[dict[str, Any]], use_llm: bool | None = None) -> dict[str, Any]:
    result = answer_question(question, triples)
    llm_enabled = _should_use_llm(use_llm)
    llm_meta: dict[str, Any] = {"llm_enabled": llm_enabled, "llm_used": False}
    if llm_enabled:
        refined_answer, llm_meta = _rewrite_answer_with_llm(question, result["answer"], result["supporting_triples"])
        result["answer"] = refined_answer
    return {
        "agent": "medical_kg_agent",
        "goal": question,
        "plan": ["识别疾病", "关系意图识别", "三元组重排序", "多跳路径推理", "构建局部子图", "生成引用答案"],
        "tool_calls": [
            {"tool": "graph_query", "status": "success"},
            {"tool": "kg_reranker", "status": "success"},
            {"tool": "kg_path_reasoner", "status": "success"},
            {"tool": "graph_subgraph", "status": "success"},
            {"tool": "kg_answer", "status": "success"},
        ],
        "question": question,
        "answer": result["answer"],
        "supporting_triples": result["supporting_triples"],
        "ranked_supporting_triples": result.get("ranked_supporting_triples", result["supporting_triples"]),
        "reasoning_paths": result.get("reasoning_paths", []),
        "subgraph": result.get("subgraph", {}),
        "llm": llm_meta,
        "result": {
            "supporting_triple_count": len(result["supporting_triples"]),
            "ranked_supporting_triple_count": len(result.get("ranked_supporting_triples", [])),
            "reasoning_path_count": len(result.get("reasoning_paths", [])),
            "subgraph_node_count": len(result.get("subgraph", {}).get("nodes", [])),
            "subgraph_edge_count": len(result.get("subgraph", {}).get("edges", [])),
        },
    }


def build_medical_demo(
    input_path: Path,
    output_root: Path,
    questions: list[str] | None = None,
    use_llm: bool | None = None,
    kg_backend: str | None = None,
    external_documents_path: Path | None = None,
    external_triples_path: Path | None = None,
    external_entities_path: Path | None = None,
    gold_triples_path: Path | None = None,
) -> dict[str, Any]:
    llm_requested = _should_use_llm(use_llm)
    llm_configured = get_openai_compat_config() is not None
    llm_active = llm_requested and llm_configured

    processing_result = run_data_processing_agent(input_path, output_root)
    processed_documents: list[dict[str, Any]] = processing_result["documents"]
    documents = [
        MedicalDocument(
            doc_id=item["doc_id"],
            title=item["title"],
            source=item["source"],
            text=item["text"],
        )
        for item in processed_documents
    ]

    entity_rows: list[dict[str, Any]] = []
    triple_candidates: list[dict[str, Any]] = []
    kg_trace: list[dict[str, Any]] = []

    entity_pool: dict[tuple[str, str], dict[str, Any]] = {}
    backend_selection = select_kg_backend(backend=kg_backend, use_llm=llm_requested)
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "select_kg_backend",
            "status": "success",
            "requested_backend": backend_selection.requested_backend,
            "selected_backend": backend_selection.selected_backend,
            "fallback_reason": backend_selection.fallback_reason,
            "available_backends": backend_selection.available_backends,
            "llm_configured": backend_selection.llm_configured,
        }
    )

    batch_results, extraction_dispatch = _run_kg_backend_batch(
        documents,
        kg_backend=kg_backend,
        use_llm=llm_requested,
        backend_selection=backend_selection,
    )
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "kg_document_batch_extract",
            "status": "success",
            **extraction_dispatch,
        }
    )

    for batch_item in batch_results:
        document = batch_item["document"]
        backend_result = batch_item["result"]
        entities = backend_result.entities
        triples = backend_result.triples
        main_disease = backend_result.main_disease
        triple_candidates.extend(triples)
        kg_trace.append(
            {
                "agent": "medical_kg_agent",
                "goal": "从标准化医疗文本构建知识图谱",
                "step": "entity_extract",
                "status": "success",
                "backend": backend_result.backend,
                "backend_version": backend_result.version,
                "backend_metadata": backend_result.metadata,
                "doc_id": document.doc_id,
                "entity_count": len(entities),
                "disease": main_disease,
                "elapsed_ms": batch_item["elapsed_ms"],
                "batch_index": batch_item["index"],
            }
        )
        kg_trace.append(
            {
                "agent": "medical_kg_agent",
                "goal": "从标准化医疗文本构建知识图谱",
                "step": "relation_extract",
                "status": "success",
                "backend": backend_result.backend,
                "backend_version": backend_result.version,
                "backend_metadata": backend_result.metadata,
                "doc_id": document.doc_id,
                "triple_candidate_count": len(triples),
                "elapsed_ms": batch_item["elapsed_ms"],
                "batch_index": batch_item["index"],
            }
        )

        for entity in entities:
            _merge_entity(
                entity_pool,
                name=entity["name"],
                entity_type=entity["type"],
                aliases=entity["aliases"],
                source_doc_ids=entity["source_doc_ids"],
                confidence=entity["confidence"],
            )

    external_summary = _merge_external_artifacts(
        processed_documents=processed_documents,
        triple_candidates=triple_candidates,
        entity_pool=entity_pool,
        kg_trace=kg_trace,
        external_documents_path=external_documents_path,
        external_triples_path=external_triples_path,
        external_entities_path=external_entities_path,
    )

    normalized_triple_candidates, normalization_report = normalize_triple_candidates(triple_candidates)
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "triple_normalize",
            "status": "success",
            **normalization_report,
        }
    )

    for triple in normalized_triple_candidates:
        source_doc_ids = triple.get("source_doc_ids")
        if not isinstance(source_doc_ids, list):
            source_doc_ids = [str(triple.get("source_doc_id", "")).strip()]
        confidence = float(triple.get("confidence", 0.84))
        subject_aliases = triple.get("subject_aliases")
        object_aliases = triple.get("object_aliases")
        if not isinstance(subject_aliases, list):
            subject_aliases = [str(triple.get("subject", ""))]
        if not isinstance(object_aliases, list):
            object_aliases = [str(triple.get("object", ""))]
        _merge_entity(
            entity_pool,
            name=str(triple.get("subject", "")),
            entity_type=str(triple.get("subject_type", "")),
            aliases=[str(item) for item in subject_aliases],
            source_doc_ids=[str(item) for item in source_doc_ids],
            confidence=confidence,
        )
        _merge_entity(
            entity_pool,
            name=str(triple.get("object", "")),
            entity_type=str(triple.get("object_type", "")),
            aliases=[str(item) for item in object_aliases],
            source_doc_ids=[str(item) for item in source_doc_ids],
            confidence=confidence,
        )

    triple_rows, kg_validation_report = aggregate_and_validate_triples(normalized_triple_candidates)
    kg_validation_report["external_summary"] = external_summary
    kg_validation_report["normalization"] = normalization_report
    resolved_gold_triples_path = gold_triples_path or external_triples_path
    kg_quality_eval = evaluate_triples_against_gold_file(
        triple_rows,
        resolved_gold_triples_path,
    )
    backend_probe: dict[str, Any] = {"enabled": False, "reason": "gold_or_external_documents_missing"}
    if external_documents_path is not None and resolved_gold_triples_path is not None:
        external_documents_for_probe = _read_jsonl(external_documents_path)
        gold_triples_for_probe = _read_jsonl(resolved_gold_triples_path)
        backend_probe = run_backend_probe(
            external_documents_for_probe,
            gold_triples_for_probe,
            backend=backend_selection.selected_backend,
            limit=40,
        )
    if kg_quality_eval.get("enabled"):
        kg_validation_report["gold_quality_eval"] = {
            "gold_path": kg_quality_eval.get("gold_path"),
            "predicted_fact_count": kg_quality_eval.get("predicted_fact_count", 0),
            "gold_fact_count": kg_quality_eval.get("gold_fact_count", 0),
            "matched_fact_count": kg_quality_eval.get("matched_fact_count", 0),
            "overall": kg_quality_eval.get("overall", {}),
            "relation_coverage": kg_quality_eval.get("relation_coverage", {}),
        }
    if backend_probe.get("enabled"):
        kg_validation_report["backend_probe"] = {
            "backend": backend_probe.get("backend"),
            "document_count": backend_probe.get("document_count", 0),
            "gold_candidate_count": backend_probe.get("gold_candidate_count", 0),
            "best_mode": backend_probe.get("best_mode"),
            "best_overall": backend_probe.get("best_overall", {}),
            "raw_backend": {
                "overall": backend_probe.get("raw_backend", {}).get("overall", {}),
                "raw_candidate_count": backend_probe.get("raw_backend", {}).get("raw_candidate_count", 0),
                "validated_candidate_count": backend_probe.get("raw_backend", {}).get("validated_candidate_count", 0),
                "docs_per_second": backend_probe.get("raw_backend", {}).get("docs_per_second", 0),
            },
            "lexicon_assisted_backend": {
                "overall": backend_probe.get("lexicon_assisted_backend", {}).get("overall", {}),
                "raw_candidate_count": backend_probe.get("lexicon_assisted_backend", {}).get("raw_candidate_count", 0),
                "validated_candidate_count": backend_probe.get("lexicon_assisted_backend", {}).get("validated_candidate_count", 0),
                "docs_per_second": backend_probe.get("lexicon_assisted_backend", {}).get("docs_per_second", 0),
            },
        }

    for index, ((name, entity_type), bucket) in enumerate(
        sorted(entity_pool.items(), key=lambda item: (item[0][1], item[0][0])),
        start=1,
    ):
        entity_rows.append(
            {
                "entity_id": f"E{index:04d}",
                "name": name,
                "type": entity_type,
                "aliases": sorted(bucket["aliases"]),
                "source_doc_ids": sorted(bucket["source_doc_ids"]),
                "confidence": round(bucket["confidence"], 2),
            }
        )

    entity_linking_report = _build_entity_linking_report(entity_rows, triple_rows)
    kg_operator_profile = _build_kg_operator_profile(
        processed_documents=processed_documents,
        triple_candidates=triple_candidates,
        normalized_triple_candidates=normalized_triple_candidates,
        triple_rows=triple_rows,
        entity_rows=entity_rows,
        normalization_report=normalization_report,
        kg_validation_report=kg_validation_report,
        backend_selection=backend_selection,
        kg_trace=kg_trace,
    )
    kg_validation_report["entity_linking"] = entity_linking_report
    kg_validation_report["operator_profile"] = {
        "operator_count": len(kg_operator_profile["operators"]),
        "operators": [
            {
                "name": item["name"],
                "input_count": item["input_count"],
                "output_count": item["output_count"],
                "reusable": item["reusable"],
            }
            for item in kg_operator_profile["operators"]
        ],
    }

    graph_snapshot = build_graph_snapshot(entity_rows, triple_rows)
    charts = build_charts(entity_rows, triple_rows)
    qa_questions = questions or DEFAULT_QUESTIONS
    qa_traces = [_build_qa_trace(question, triple_rows, use_llm=llm_requested) for question in qa_questions]
    hybrid_qa_traces = [
        _build_hybrid_qa_trace(output_root=output_root, question=trace["question"], qa_trace=trace)
        for trace in qa_traces
    ]
    subgraphs = [trace["subgraph"] for trace in qa_traces if trace.get("subgraph")]
    analysis_report = _build_analysis_report(processed_documents, entity_rows, triple_rows, qa_traces)
    kg_quality_report = _build_kg_quality_report(
        processed_documents=processed_documents,
        entities=entity_rows,
        triples=triple_rows,
        kg_validation_report=kg_validation_report,
        qa_traces=qa_traces,
    )
    kg_quality_eval_report = build_quality_eval_report(kg_quality_eval)
    backend_probe_report = build_backend_probe_report(backend_probe)
    quality_report = processing_result["quality_report"]
    documents_path = output_root / "data" / "processed" / "documents.jsonl"
    _write_jsonl(documents_path, processed_documents)
    if any(external_summary[key] for key in ("external_document_count", "external_triple_count", "external_entity_count")):
        kg_trace.append(
            {
                "agent": "medical_kg_agent",
                "goal": "从标准化医疗文本构建知识图谱",
                "step": "write_merged_documents",
                "status": "success",
                "output_path": str(documents_path),
                "document_count": len(processed_documents),
            }
        )

    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "triple_validate",
            "status": "success",
            "candidate_triples": len(normalized_triple_candidates),
            "raw_candidate_triples": len(triple_candidates),
            "accepted_triples": kg_validation_report["accepted_triples"],
            "rejected_triples": kg_validation_report["rejected_triples"],
            "duplicate_triples": kg_validation_report["duplicate_triples"],
            "arbitration": kg_validation_report.get("arbitration", {}),
            "evidence_fusion": kg_validation_report.get("evidence_fusion", {}),
        }
    )
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "entity_link",
            "status": "success",
            "entity_count": entity_linking_report["entity_count"],
            "linked_entity_count": entity_linking_report["linked_entity_count"],
            "alias_rich_entity_count": entity_linking_report["alias_rich_entity_count"],
            "isolated_entity_count": entity_linking_report["isolated_entity_count"],
        }
    )
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "operator_profile",
            "status": "success",
            "operator_count": len(kg_operator_profile["operators"]),
            "avg_support_score": kg_operator_profile["quality"]["avg_support_score"],
            "relation_diversity": kg_operator_profile["quality"]["relation_diversity"],
        }
    )
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "gold_quality_evaluate",
            "status": "success" if kg_quality_eval.get("enabled") else "skipped",
            "gold_path": kg_quality_eval.get("gold_path"),
            "evaluation_mode": kg_quality_eval.get("evaluation_mode"),
            "predicted_fact_count": kg_quality_eval.get("predicted_fact_count", 0),
            "gold_fact_count": kg_quality_eval.get("gold_fact_count", 0),
            "matched_fact_count": kg_quality_eval.get("matched_fact_count", 0),
            "overall": kg_quality_eval.get("overall", {}),
            "reason": kg_quality_eval.get("reason", ""),
        }
    )
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "backend_probe_evaluate",
            "status": "success" if backend_probe.get("enabled") else "skipped",
            "backend": backend_probe.get("backend"),
            "document_count": backend_probe.get("document_count", 0),
            "gold_candidate_count": backend_probe.get("gold_candidate_count", 0),
            "best_mode": backend_probe.get("best_mode"),
            "best_overall": backend_probe.get("best_overall", {}),
            "raw_overall": backend_probe.get("raw_backend", {}).get("overall", {}),
            "lexicon_assisted_overall": backend_probe.get("lexicon_assisted_backend", {}).get("overall", {}),
            "reason": backend_probe.get("reason", ""),
        }
    )
    kg_trace.append(
        {
            "agent": "medical_kg_agent",
            "goal": "从标准化医疗文本构建知识图谱",
            "step": "graph_snapshot",
            "status": "success",
            "node_count": len(graph_snapshot["nodes"]),
            "edge_count": len(graph_snapshot["edges"]),
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "graph" / "entities.jsonl", entity_rows)
    _write_jsonl(output_root / "graph" / "triples.jsonl", triple_rows)
    _write_json(output_root / "graph" / "graph_snapshot.json", graph_snapshot)
    _write_json(output_root / "graph" / "kg_validation_report.json", kg_validation_report)
    _write_json(output_root / "graph" / "kg_quality_eval.json", kg_quality_eval)
    _write_json(output_root / "graph" / "kg_backend_probe.json", backend_probe)
    _write_json(output_root / "graph" / "entity_linking_report.json", entity_linking_report)
    _write_json(output_root / "graph" / "kg_operator_profile.json", kg_operator_profile)
    _write_jsonl(output_root / "graph" / "kg_build_trace.jsonl", kg_trace)
    _write_json(output_root / "demo" / "charts" / "entity_types.json", charts["entity_types"])
    _write_json(output_root / "demo" / "charts" / "relation_types.json", charts["relation_types"])
    _write_jsonl(output_root / "demo" / "subgraphs" / "question_subgraphs.jsonl", subgraphs)
    _write_jsonl(output_root / "demo" / "qa_trace.jsonl", qa_traces)
    _write_jsonl(output_root / "demo" / "hybrid_qa_trace.jsonl", hybrid_qa_traces)
    (output_root / "demo" / "analysis_report.md").write_text(analysis_report, encoding="utf-8")
    (output_root / "demo" / "kg_quality_report.md").write_text(kg_quality_report, encoding="utf-8")
    (output_root / "demo" / "kg_quality_eval_report.md").write_text(kg_quality_eval_report, encoding="utf-8")
    (output_root / "demo" / "kg_backend_probe_report.md").write_text(backend_probe_report, encoding="utf-8")

    analysis_result = run_analysis_agent(
        output_root,
        processed_documents=processed_documents,
        entities=entity_rows,
        triples=triple_rows,
        graph_snapshot=graph_snapshot,
        qa_traces=qa_traces,
        questions=qa_questions,
    )

    return {
        "documents": processed_documents,
        "entities": entity_rows,
        "triples": triple_rows,
        "graph_snapshot": graph_snapshot,
        "kg_validation_report": kg_validation_report,
        "kg_quality_eval": kg_quality_eval,
        "kg_backend_probe": backend_probe,
        "entity_linking_report": entity_linking_report,
        "kg_operator_profile": kg_operator_profile,
        "kg_extraction_dispatch": extraction_dispatch,
        "kg_trace": kg_trace,
        "qa_traces": qa_traces,
        "hybrid_qa_traces": hybrid_qa_traces,
        "subgraphs": subgraphs,
        "analysis_report": analysis_result["analysis_report"],
        "kg_overview_report": analysis_report,
        "kg_quality_report": kg_quality_report,
        "kg_quality_eval_report": kg_quality_eval_report,
        "kg_backend_probe_report": backend_probe_report,
        "analysis_summary": analysis_result["analysis_summary"],
        "analysis_planning": analysis_result.get("analysis_planning", {}),
        "analysis_queries": analysis_result["analysis_queries"],
        "nl2sql_eval": analysis_result["nl2sql_eval"],
        "quality_report": quality_report,
        "charts": analysis_result["charts"],
        "analysis_charts": analysis_result["charts"],
        "graph_insights": analysis_result["graph_insights"],
        "analysis_trace": analysis_result["analysis_trace"],
        "data_processing_trace": processing_result["trace"],
        "external_summary": external_summary,
        "hybrid_qa": {
            "enabled": True,
            "trace_count": len(hybrid_qa_traces),
            "datamate_supported_count": sum(
                1
                for trace in hybrid_qa_traces
                if trace.get("datamate_retrieval", {}).get("evidence_count", 0) > 0
            ),
            "average_fusion_score": round(
                sum(float(trace.get("fusion_score", 0)) for trace in hybrid_qa_traces) / len(hybrid_qa_traces),
                4,
            )
            if hybrid_qa_traces
            else 0.0,
        },
        "llm_requested": llm_requested,
        "llm_configured": llm_configured,
        "llm_active": llm_active,
        "kg_backend": {
            "requested_backend": backend_selection.requested_backend,
            "selected_backend": backend_selection.selected_backend,
            "fallback_reason": backend_selection.fallback_reason,
            "available_backends": backend_selection.available_backends,
            "llm_configured": backend_selection.llm_configured,
            "extraction_dispatch": extraction_dispatch,
        },
    }


def ask_medical_question(
    output_root: Path,
    question: str,
    use_llm: bool | None = None,
    use_hybrid: bool = True,
    knowledge_base_id: str | None = None,
    enable_live_datamate: bool = True,
    datamate_timeout_seconds: float = 1.5,
) -> dict[str, Any]:
    triples_path = output_root / "graph" / "triples.jsonl"
    if not triples_path.exists():
        raise FileNotFoundError(f"Missing graph triples file: {triples_path}")
    triples: list[dict[str, Any]] = []
    with triples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                triples.append(json.loads(line))
    trace = _build_qa_trace(question, triples, use_llm=use_llm)
    if not use_hybrid:
        return trace

    hybrid_trace = _build_hybrid_qa_trace(
        output_root=output_root,
        question=question,
        qa_trace=trace,
        knowledge_base_id=knowledge_base_id,
        enable_live_datamate=enable_live_datamate,
        datamate_timeout_seconds=datamate_timeout_seconds,
    )
    trace["answer"] = hybrid_trace["answer"]
    trace["hybrid_qa"] = {
        "enabled": True,
        "fusion_score": hybrid_trace["fusion_score"],
        "source_balance": hybrid_trace["source_balance"],
        "datamate_evidence_count": hybrid_trace["datamate_retrieval"]["evidence_count"],
        "kg_evidence_count": len(hybrid_trace["kg_evidence"]),
        "reasoning_path_count": len(hybrid_trace.get("reasoning_paths", [])),
        "reasoning_paths": hybrid_trace.get("reasoning_paths", []),
        "ranked_supporting_triples": hybrid_trace.get("ranked_supporting_triples", []),
        "fused_evidence": hybrid_trace["fused_evidence"],
        "steps": hybrid_trace["steps"],
        "datamate_retrieval": hybrid_trace["datamate_retrieval"],
    }
    trace["tool_calls"].extend(
        [
            {
                "tool": "datamate_retrieve",
                "status": hybrid_trace["datamate_retrieval"]["status"],
                "result_count": hybrid_trace["datamate_retrieval"]["evidence_count"],
            },
            {
                "tool": "kg_path_reasoner",
                "status": "success" if hybrid_trace.get("reasoning_paths") else "empty",
                "path_count": len(hybrid_trace.get("reasoning_paths", [])),
            },
            {
                "tool": "evidence_fusion",
                "status": "success" if hybrid_trace["fused_evidence"] else "empty",
                "fusion_score": hybrid_trace["fusion_score"],
            },
        ]
    )
    trace_path = output_root / "demo" / "hybrid_qa_trace.jsonl"
    existing = _read_jsonl(trace_path) if trace_path.exists() else []
    existing.append(hybrid_trace)
    _write_jsonl(trace_path, existing[-100:])
    return trace
