"""LLM-enhanced medical entity and relation extraction for task 2.

Uses an OpenAI-compatible LLM to extract entities and relations beyond
the rule-based dictionary baseline.  Falls back gracefully when the LLM
is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.common.llm_config import openai_extra_kwargs

logger = logging.getLogger(__name__)

_ENTITY_TYPES = ["Disease", "Symptom", "Drug", "Examination", "Treatment"]
_RELATION_TYPES = [
    "has_symptom", "treated_by", "diagnosed_by",
    "recommended_treatment", "complication_of",
]

_EXTRACTION_PROMPT = """\
你是一名医疗信息抽取专家。请从以下中文医疗文本中提取实体和关系。

## 实体类型
- Disease: 疾病名称
- Symptom: 症状、体征
- Drug: 药物名称
- Examination: 检查、检验项目
- Treatment: 治疗方法或建议

## 关系类型
- has_symptom: 疾病的症状
- treated_by: 疾病的治疗药物
- diagnosed_by: 疾病的诊断检查
- recommended_treatment: 疾病的推荐治疗
- complication_of: 疾病的并发症关系

## 医疗文本
{text}

## 输出格式
请严格按照以下 JSON 格式输出，不要添加其他文字：

```json
{{
  "entities": {{
    "Disease": ["疾病1", "疾病2"],
    "Symptom": ["症状1", "症状2"],
    "Drug": ["药物1", "药物2"],
    "Examination": ["检查1", "检查2"],
    "Treatment": ["治疗1", "治疗2"]
  }},
  "relations": [
    {{"subject": "疾病", "predicate": "has_symptom", "object": "症状", "confidence": 0.9}},
    {{"subject": "疾病", "predicate": "treated_by", "object": "药物", "confidence": 0.85}}
  ]
}}
```"""


def extract_entities_with_llm(
    text: str,
    llm_config: dict[str, Any],
    fallback_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract medical entities using an LLM, with rule-based fallback.

    Parameters
    ----------
    text:
        Raw medical text to process.
    llm_config:
        Dict with ``base_url``, ``api_key``, ``model_name``.
    fallback_records:
        Optional pre-computed rule-based records to merge.
    """
    chunks = _split_chunks(text)
    all_entities: dict[str, set[str]] = {t: set() for t in _ENTITY_TYPES}
    # Entities scoped to each chunk so a record never inherits another record's
    # entities (a global union would pollute every record and corrupt
    # downstream relation extraction).
    per_chunk_entities: list[dict[str, list[str]]] = [
        {t: [] for t in _ENTITY_TYPES} for _ in chunks
    ]
    all_relations: list[dict[str, Any]] = []
    chunk_count = 0
    llm_available = True

    for chunk_index, chunk_text in enumerate(chunks):
        if not llm_available:
            break
        try:
            llm_result = _call_llm_extraction(chunk_text, llm_config)
            chunk_count += 1
            for etype, names in llm_result.get("entities", {}).items():
                if etype in _ENTITY_TYPES:
                    all_entities[etype].update(names)
                    per_chunk_entities[chunk_index][etype] = sorted(set(names))
            for rel in llm_result.get("relations", []):
                if _valid_relation(rel):
                    rel["record_id"] = f"chunk_{chunk_count}"
                    rel["evidence"] = _evidence_snippet(
                        chunk_text, rel.get("subject", ""), rel.get("object", "")
                    )
                    rel.setdefault("subject_type", _infer_subject_type(rel["predicate"]))
                    rel.setdefault("object_type", _infer_object_type(rel["predicate"]))
                    rel.setdefault("confidence", 0.7)
                    all_relations.append(rel)
        except Exception:
            logger.warning("LLM extraction failed for chunk; fast-failing remaining chunks.", exc_info=True)
            llm_available = False
            continue

    # Merge per-chunk rule-based fallback entities so each record keeps both its
    # LLM and rule entities, and the aggregate union stays accurate.
    if fallback_records:
        for chunk_index in range(len(chunks)):
            if chunk_index < len(fallback_records):
                record = fallback_records[chunk_index]
                for etype, names in record.get("entities", {}).items():
                    if etype in _ENTITY_TYPES:
                        all_entities[etype].update(names)
                        merged = set(per_chunk_entities[chunk_index][etype]) | set(names)
                        per_chunk_entities[chunk_index][etype] = sorted(merged)
        # Account for any extra fallback records beyond the chunk count.
        for record in fallback_records[len(chunks):]:
            for etype, names in record.get("entities", {}).items():
                if etype in _ENTITY_TYPES:
                    all_entities[etype].update(names)

    entity_counts = {t: len(v) for t, v in all_entities.items()}
    records = []
    for idx, chunk_text in enumerate(chunks, start=1):
        records.append({
            "record_id": f"record_{idx}",
            "text": chunk_text,
            "entities": {t: list(per_chunk_entities[idx - 1][t]) for t in _ENTITY_TYPES},
            "normalization": {"aliases": {}},
            "mentions": {},
        })

    return {
        "status": "completed",
        "record_count": len(records),
        "records": records,
        "entity_counts": entity_counts,
        "normalization": {"aliases": {}},
        "llm_chunks_processed": chunk_count,
        # Cache LLM-extracted relations so the caller can reuse them
        # without calling the LLM a second time.
        "_cached_llm_relations": all_relations,
    }


def extract_relations_with_llm(
    text: str,
    llm_config: dict[str, Any],
    rule_based_triples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract relations using LLM, merging with rule-based results."""
    chunks = _split_chunks(text)
    llm_triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    llm_available = True

    for chunk_text in chunks:
        if not llm_available:
            break
        try:
            llm_result = _call_llm_extraction(chunk_text, llm_config)
            for rel in llm_result.get("relations", []):
                if _valid_relation(rel):
                    key = (rel["subject"], rel["predicate"], rel["object"])
                    if key not in seen:
                        seen.add(key)
                        rel["evidence"] = _evidence_snippet(
                            chunk_text, rel["subject"], rel["object"]
                        )
                        rel.setdefault("subject_type", _infer_subject_type(rel["predicate"]))
                        rel.setdefault("object_type", _infer_object_type(rel["predicate"]))
                        rel.setdefault("confidence", 0.7)
                        rel.setdefault("record_id", "llm")
                        llm_triples.append(rel)
        except Exception:
            logger.warning("LLM relation extraction failed; fast-failing remaining chunks.", exc_info=True)
            llm_available = False

    # Merge with rule-based
    merged = list(llm_triples)
    merge_seen: set[tuple[str, str, str]] = {
        (t["subject"], t["predicate"], t["object"]) for t in merged
    }
    if rule_based_triples:
        for triple in rule_based_triples:
            key = (triple["subject"], triple["predicate"], triple["object"])
            if key not in merge_seen:
                merge_seen.add(key)
                merged.append(triple)

    return merged


# Reasoning models spend part of the token budget on internal reasoning, so a
# small cap can be consumed before any JSON is emitted. Default high; override
# via config["max_tokens"].
_DEFAULT_LLM_MAX_TOKENS = 4096


def _call_llm_extraction(text: str, config: dict[str, Any]) -> dict[str, Any]:
    """Call LLM and parse the JSON extraction result."""
    import openai

    prompt = _EXTRACTION_PROMPT.format(text=text[:2000])
    client = openai.OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )
    response = client.chat.completions.create(
        model=config.get("model_name", "glm-5.1"),
        messages=[
            {"role": "system", "content": "你是医疗信息抽取专家，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=config.get("max_tokens", _DEFAULT_LLM_MAX_TOKENS),
        # Reasoning models can take >100s per chunk; default generously and
        # allow override via config["timeout"].
        timeout=config.get("timeout", 180.0),
        **openai_extra_kwargs(config),
    )
    raw = response.choices[0].message.content or ""
    return _parse_extraction_json(raw)


def _parse_extraction_json(raw: str) -> dict[str, Any]:
    """Parse LLM response, handling fenced code blocks."""
    text = raw.strip()
    # Strip code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find any JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"entities": {}, "relations": []}


def _split_chunks(text: str, max_chars: int = 1500) -> list[str]:
    """Split text by record separators, respecting max length."""
    chunks = [c.strip() for c in re.split(r"\n\s*---\s*\n?", text) if c.strip()]
    if not chunks:
        chunks = [text.strip()] if text.strip() else []
    # Further split oversized chunks
    result = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            sentences = re.split(r"(?<=[。；;])", chunk)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) > max_chars and buf:
                    result.append(buf)
                    buf = sent
                else:
                    buf += sent
            if buf:
                result.append(buf)
    return result


def _valid_relation(rel: dict[str, Any]) -> bool:
    return (
        isinstance(rel, dict)
        and rel.get("subject")
        and rel.get("predicate") in _RELATION_TYPES
        and rel.get("object")
    )


def _infer_subject_type(predicate: str) -> str:
    return "Disease"


def _infer_object_type(predicate: str) -> str:
    mapping = {
        "has_symptom": "Symptom",
        "treated_by": "Drug",
        "diagnosed_by": "Examination",
        "recommended_treatment": "Treatment",
        "complication_of": "Disease",
    }
    return mapping.get(predicate, "Unknown")


def _evidence_snippet(text: str, subject: str, object_: str, window: int = 28) -> str:
    positions = [
        pos for pos, term in ((text.find(subject), subject), (text.find(object_), object_))
        if pos >= 0
    ]
    if not positions:
        return text[:window * 2].strip()
    start = max(min(positions) - window, 0)
    end = min(max(positions) + window, len(text))
    return text[start:end].strip()
