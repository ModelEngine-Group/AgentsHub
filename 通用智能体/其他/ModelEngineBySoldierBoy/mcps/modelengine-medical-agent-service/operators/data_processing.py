"""Generic data-processing operators for the medical demo."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TEXT_FIELD_NAMES = ("text", "content", "body", "description", "note", "notes", "summary")
TITLE_FIELD_NAMES = ("title", "name", "subject")
SOURCE_FIELD_NAMES = ("source", "dataset", "origin")
ID_FIELD_NAMES = ("id", "doc_id", "document_id")


def _detect_source_format(path: Path | None) -> str:
    if path is None:
        return "unknown"
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving the original meaning."""
    return _normalize_whitespace(text.replace("\u200b", "").replace("\ufeff", ""))


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"[。！？!?；;\n]+", text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def split_sections(text: str) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []

    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", normalized) if item.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    sentences = split_sentences(normalized)
    if len(sentences) > 1:
        return sentences

    return [normalized]


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    text = str(value).strip()
    if not text:
        return "empty"
    if re.fullmatch(r"[-+]?\d+", text):
        return "int"
    if re.fullmatch(r"[-+]?\d+\.\d+", text):
        return "float"
    return "text"


def _median(values: list[int]) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2, 1)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _extract_text_field(record: dict[str, Any]) -> str:
    for field_name in TEXT_FIELD_NAMES:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value

    collected: list[str] = []
    skip_keys = set(TEXT_FIELD_NAMES) | set(TITLE_FIELD_NAMES) | set(SOURCE_FIELD_NAMES) | set(ID_FIELD_NAMES)
    for key, value in record.items():
        if key in skip_keys:
            continue
        text = _to_text(value).strip()
        if text:
            collected.append(text)
    return "\n".join(collected)


def _extract_title(record: dict[str, Any], fallback: str) -> str:
    for field_name in TITLE_FIELD_NAMES:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return clean_text(fallback)


def _extract_source(record: dict[str, Any], fallback: str) -> str:
    for field_name in SOURCE_FIELD_NAMES:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return clean_text(fallback)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load raw records from JSONL, CSV, JSON, or TXT files."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        raw_text = path.read_text(encoding="utf-8")
        stripped = raw_text.strip()
        if not stripped:
            return []

        if suffix == ".json":
            payload = json.loads(stripped)
            if isinstance(payload, list):
                return [dict(item) for item in payload]
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                return [dict(item) for item in payload["records"]]
            if isinstance(payload, dict):
                return [dict(payload)]
            raise ValueError(f"Unsupported JSON payload in {path}")

        records: list[dict[str, Any]] = []
        for raw_line in stripped.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL records must be objects: {path}")
            records.append(dict(payload))
        return records

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    if suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8")
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", raw_text) if block.strip()]
        if not blocks:
            blocks = [raw_text.strip()]
        records: list[dict[str, Any]] = []
        for index, block in enumerate(blocks, start=1):
            records.append(
                {
                    "id": f"{path.stem}-{index:03d}",
                    "title": path.stem,
                    "source": path.stem,
                    "text": block,
                }
            )
        return records

    raise ValueError(f"Unsupported input format: {path.suffix or '(no suffix)'}")


def infer_schema_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer source schema, field roles, sparsity, and text-candidate strength."""
    field_names = sorted({key for record in records for key in record.keys()})
    field_profiles: list[dict[str, Any]] = []
    role_candidates = {
        "doc_id": [field for field in field_names if field in ID_FIELD_NAMES],
        "title": [field for field in field_names if field in TITLE_FIELD_NAMES],
        "source": [field for field in field_names if field in SOURCE_FIELD_NAMES],
        "text": [field for field in field_names if field in TEXT_FIELD_NAMES],
    }

    for field_name in field_names:
        values = [record.get(field_name) for record in records]
        non_empty_texts = [str(value).strip() for value in values if str(value or "").strip()]
        type_counts = Counter(_value_type(value) for value in values)
        lengths = [len(text) for text in non_empty_texts]
        distinct_values = len(set(non_empty_texts))
        presence_count = len(non_empty_texts)
        field_profiles.append(
            {
                "name": field_name,
                "presence_count": presence_count,
                "presence_ratio": round(presence_count / len(records), 4) if records else 0,
                "empty_count": len(records) - presence_count,
                "type_counts": dict(sorted(type_counts.items())),
                "distinct_count": distinct_values,
                "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "candidate_role": (
                    "doc_id" if field_name in ID_FIELD_NAMES else
                    "title" if field_name in TITLE_FIELD_NAMES else
                    "source" if field_name in SOURCE_FIELD_NAMES else
                    "text" if field_name in TEXT_FIELD_NAMES else
                    "attribute"
                ),
                "sample_values": non_empty_texts[:3],
            }
        )

    text_candidate_fields = sorted(
        field_profiles,
        key=lambda item: (
            item["candidate_role"] == "text",
            item["presence_ratio"],
            item["avg_length"],
        ),
        reverse=True,
    )
    inferred_text_field = text_candidate_fields[0]["name"] if text_candidate_fields else None
    id_candidate = role_candidates["doc_id"][0] if role_candidates["doc_id"] else None
    return {
        "record_count": len(records),
        "field_count": len(field_names),
        "fields": field_profiles,
        "role_candidates": role_candidates,
        "inferred_mapping": {
            "doc_id": id_candidate,
            "title": role_candidates["title"][0] if role_candidates["title"] else None,
            "source": role_candidates["source"][0] if role_candidates["source"] else None,
            "text": inferred_text_field,
        },
        "schema_confidence": {
            "has_id": bool(id_candidate),
            "has_text": bool(inferred_text_field),
            "text_coverage": next(
                (item["presence_ratio"] for item in field_profiles if item["name"] == inferred_text_field),
                0,
            ),
        },
    }


def normalize_fields(record: dict[str, Any], index: int, source_path: Path | None = None) -> dict[str, Any]:
    """Normalize a raw record into the standard document shape."""
    fallback_id = f"doc-{index:03d}"
    doc_id = clean_text(_to_text(record.get("id") or record.get("doc_id") or record.get("document_id") or fallback_id)) or fallback_id
    title_fallback = source_path.stem if source_path else doc_id
    source_fallback = source_path.stem if source_path else "unknown"
    title = _extract_title(record, title_fallback)
    source = _extract_source(record, source_fallback)
    text = clean_text(_extract_text_field(record))
    sections = split_sections(text)
    sentence_count = len(split_sentences(text))
    source_format = _detect_source_format(source_path)
    text_fingerprint = _fingerprint(text)
    chinese_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#0",
        "title": title or doc_id,
        "source": source,
        "text": text,
        "sections": sections,
        "fingerprint": text_fingerprint,
        "quality": {
            "length": len(text),
            "language": "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "unknown",
            "chinese_char_ratio": round(chinese_count / len(text), 4) if text else 0,
            "sentence_count": sentence_count,
            "section_count": len(sections),
            "source_format": source_format,
            "field_count": len(record),
            "text_fingerprint": text_fingerprint,
            "row_number": index,
        },
        "raw_fields": sorted(record.keys()),
    }


def profile_documents(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build document-level statistics used by later KG and analysis stages."""
    lengths = [int(record.get("quality", {}).get("length", 0)) for record in records]
    sentence_counts = [int(record.get("quality", {}).get("sentence_count", 0)) for record in records]
    section_counts = [int(record.get("quality", {}).get("section_count", 0)) for record in records]
    language_counts = Counter(str(record.get("quality", {}).get("language", "unknown")) for record in records)
    source_counts = Counter(str(record.get("source", "unknown")) for record in records)
    fingerprint_to_docs: dict[str, list[str]] = {}
    for record in records:
        fingerprint = str(record.get("fingerprint") or record.get("quality", {}).get("text_fingerprint") or "")
        if not fingerprint:
            continue
        fingerprint_to_docs.setdefault(fingerprint, []).append(str(record.get("doc_id", "")))
    duplicate_text_groups = [
        {"fingerprint": fingerprint, "doc_ids": sorted(doc_ids)}
        for fingerprint, doc_ids in sorted(fingerprint_to_docs.items())
        if len(set(doc_ids)) > 1
    ]
    low_information_documents = [
        str(record.get("doc_id"))
        for record in records
        if 0 < int(record.get("quality", {}).get("length", 0)) < 20
    ]
    return {
        "document_count": len(records),
        "length": {
            "min": min(lengths) if lengths else 0,
            "p25": _percentile(lengths, 0.25),
            "median": _median(lengths),
            "p75": _percentile(lengths, 0.75),
            "max": max(lengths) if lengths else 0,
            "avg": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
        "sentence_count": {
            "min": min(sentence_counts) if sentence_counts else 0,
            "median": _median(sentence_counts),
            "max": max(sentence_counts) if sentence_counts else 0,
        },
        "section_count": {
            "min": min(section_counts) if section_counts else 0,
            "median": _median(section_counts),
            "max": max(section_counts) if section_counts else 0,
        },
        "language_counts": dict(sorted(language_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "duplicate_text_groups": duplicate_text_groups,
        "low_information_documents": low_information_documents,
    }


def quality_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(record.get("text", "")) for record in records]
    sentence_counts = [int(record.get("quality", {}).get("sentence_count", 0)) for record in records]
    section_counts = [int(record.get("quality", {}).get("section_count", 0)) for record in records]
    doc_ids = [str(record.get("doc_id", "")) for record in records if record.get("doc_id")]
    duplicate_doc_ids = sorted([doc_id for doc_id, count in Counter(doc_ids).items() if count > 1])
    source_format_counts = Counter(
        str(record.get("quality", {}).get("source_format", "unknown")) for record in records
    )
    empty_text_documents = [record["doc_id"] for record in records if not record.get("text", "").strip()]
    duplicate_text_groups = profile_documents(records)["duplicate_text_groups"]
    low_information_documents = [
        record["doc_id"]
        for record in records
        if 0 < int(record.get("quality", {}).get("length", 0)) < 20
    ]
    warnings: list[dict[str, Any]] = []
    if duplicate_doc_ids:
        warnings.append(
            {
                "code": "duplicate_doc_id",
                "severity": "warning",
                "message": "存在重复 doc_id，后续图谱构建会按 doc_id 聚合证据。",
                "doc_ids": duplicate_doc_ids,
            }
        )
    if empty_text_documents:
        warnings.append(
            {
                "code": "empty_text",
                "severity": "warning",
                "message": "存在空文本记录，已保留记录并在质量报告中标记。",
                "doc_ids": empty_text_documents,
            }
        )
    if records and len(set(doc_ids)) < len(records):
        warnings.append(
            {
                "code": "non_unique_documents",
                "severity": "info",
                "message": "文档数与唯一 doc_id 数不一致，建议检查上游数据主键。",
            }
        )
    if duplicate_text_groups:
        warnings.append(
            {
                "code": "duplicate_text",
                "severity": "warning",
                "message": "存在正文完全重复的文档，后续图谱阶段会保留来源但应避免重复计数。",
                "groups": duplicate_text_groups,
            }
        )
    if low_information_documents:
        warnings.append(
            {
                "code": "low_information_text",
                "severity": "info",
                "message": "存在低信息量短文本，建议补充上下文后再进入知识抽取。",
                "doc_ids": low_information_documents,
            }
        )
    return {
        "document_count": len(records),
        "unique_doc_ids": len(set(doc_ids)),
        "duplicate_doc_ids": duplicate_doc_ids,
        "empty_text_documents": empty_text_documents,
        "duplicate_text_groups": duplicate_text_groups,
        "low_information_documents": low_information_documents,
        "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "length_percentiles": {
            "p25": _percentile(lengths, 0.25),
            "p50": int(_median(lengths)),
            "p75": _percentile(lengths, 0.75),
        },
        "avg_sentence_count": round(sum(sentence_counts) / len(sentence_counts), 1) if sentence_counts else 0,
        "avg_section_count": round(sum(section_counts) / len(section_counts), 1) if section_counts else 0,
        "source_format_counts": dict(sorted(source_format_counts.items())),
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def build_processing_trace(
    input_path: Path,
    raw_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
    quality_report: dict[str, Any],
) -> list[dict[str, Any]]:
    source_formats = Counter(
        str(record.get("quality", {}).get("source_format", "unknown")) for record in processed_records
    )
    return [
        {
            "agent": "data_processing_agent",
            "goal": "将原始医疗数据清洗并标准化为后续图谱构建可复用的中间结果",
            "plan": [
                "加载原始记录",
                "清洗文本并拆分段落",
                "标准化字段",
                "执行质量检查",
                "输出标准化数据",
            ],
            "step": "load_records",
            "tool": "load_records",
            "status": "success",
            "input_path": str(input_path),
            "raw_record_count": len(raw_records),
            "source_format": _detect_source_format(input_path),
        },
        {
            "agent": "data_processing_agent",
            "goal": "将原始医疗数据清洗并标准化为后续图谱构建可复用的中间结果",
            "step": "normalize_fields",
            "tool": "normalize_fields",
            "status": "success",
            "processed_count": len(processed_records),
            "document_ids": [record["doc_id"] for record in processed_records[:8]],
            "source_format_counts": dict(sorted(source_formats.items())),
        },
        {
            "agent": "data_processing_agent",
            "goal": "将原始医疗数据清洗并标准化为后续图谱构建可复用的中间结果",
            "step": "quality_check",
            "tool": "quality_check",
            "status": "success",
            "summary": {
                "document_count": quality_report["document_count"],
                "unique_doc_ids": quality_report["unique_doc_ids"],
                "avg_length": quality_report["avg_length"],
                "avg_sentence_count": quality_report["avg_sentence_count"],
                "avg_section_count": quality_report["avg_section_count"],
                "empty_text_documents": quality_report["empty_text_documents"],
            },
        },
    ]
