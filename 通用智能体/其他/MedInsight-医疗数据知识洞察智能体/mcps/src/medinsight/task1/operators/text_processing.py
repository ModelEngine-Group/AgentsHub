from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from medinsight.common.operators.base import OperatorSpec
from medinsight.common.schemas import Document, Segment
from medinsight.common.utils import clean_text, legacy_clean_text, legacy_split_sentences, split_sentences


class RawRecordProfiler:
    spec = OperatorSpec(
        name="raw_record_profiler",
        description="Profile raw structured/unstructured medical records before ETL.",
        input_schema={"records": "list[dict]"},
        output_schema={"profile": "dict"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        records = payload["records"]
        total = len(records)
        fields = Counter(field for row in records for field in row)
        texts = [str(row.get("text") or "").strip() for row in records if str(row.get("text") or "").strip()]
        text_ready = len(texts)
        unique_text_count = len(set(texts))
        structured_ready = sum(
            1
            for row in records
            if isinstance(row.get("original_fields"), dict)
            or any(key not in {"doc_id", "title", "text"} and str(value or "").strip() for key, value in row.items())
        )
        source_chunks = [
            str(chunk).strip()
            for row in records
            if isinstance(row.get("source_texts"), list)
            for chunk in row["source_texts"]
            if str(chunk).strip()
        ]
        unique_source_chunk_count = len(set(source_chunks))
        entity_terms_ready = sum(1 for row in records if isinstance(row.get("entity_terms"), dict))
        source_datasets = sorted({str(row.get("source_dataset")) for row in records if row.get("source_dataset")})
        profile = {
            "record_count": total,
            "field_count": len(fields),
            "top_fields": fields.most_common(12),
            "text_coverage": round(text_ready / total, 4) if total else 0.0,
            "avg_text_length": round(sum(len(text) for text in texts) / len(texts), 2) if texts else 0.0,
            "unique_text_count": unique_text_count,
            "duplicate_text_ratio": round((text_ready - unique_text_count) / text_ready, 4) if text_ready else 0.0,
            "source_chunk_count": len(source_chunks),
            "unique_source_chunk_count": unique_source_chunk_count,
            "duplicate_source_chunk_ratio": round(
                (len(source_chunks) - unique_source_chunk_count) / len(source_chunks), 4
            ) if source_chunks else 0.0,
            "structured_field_coverage": round(structured_ready / total, 4) if total else 0.0,
            "entity_terms_coverage": round(entity_terms_ready / total, 4) if total else 0.0,
            "source_datasets": source_datasets,
            "detected_mode": "hybrid" if text_ready and structured_ready else "text" if text_ready else "structured",
        }
        return {"profile": profile}


class MedicalRecordNormalizer:
    spec = OperatorSpec(
        name="medical_record_normalizer",
        description="Validate and normalize raw medical records into a stable ETL contract.",
        input_schema={"records": "list[dict]"},
        output_schema={"records": "list[dict]", "anomalies": "list[dict]"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        clean_fn = clean_text if payload.get("use_optimized_text_ops", True) else legacy_clean_text
        normalized: List[dict[str, Any]] = []
        anomalies: List[dict[str, Any]] = []
        for index, row in enumerate(payload["records"], start=1):
            raw_doc_id = str(row.get("doc_id") or "").strip()
            raw_title = str(row.get("title") or row.get("name") or "").strip()
            doc_id = raw_doc_id or f"doc-{index}"
            title = raw_title or f"document-{index}"
            if not raw_doc_id:
                anomalies.append(
                    {
                        "doc_id": doc_id,
                        "level": "warning",
                        "action": "filled_default_doc_id",
                        "message": "missing doc_id; generated stable replacement id",
                    }
                )
            if not raw_title:
                anomalies.append(
                    {
                        "doc_id": doc_id,
                        "level": "warning",
                        "action": "filled_default_title",
                        "message": "missing title; generated replacement title",
                    }
                )
            text = clean_fn(str(row.get("text") or ""))
            if not text and isinstance(row.get("original_fields"), dict):
                text = clean_fn("。".join(str(value) for value in row["original_fields"].values() if value))
                anomalies.append(
                    {
                        "doc_id": doc_id,
                        "level": "warning",
                        "action": "rebuilt_text_from_original_fields",
                        "message": "text rebuilt from original_fields",
                    }
                )
            if not text:
                anomalies.append(
                    {
                        "doc_id": doc_id,
                        "level": "error",
                        "action": "dropped_record",
                        "message": "missing text and no structured fields available",
                    }
                )
                continue

            item = dict(row)
            item["doc_id"] = doc_id
            item["title"] = title
            item["text"] = text
            normalized.append(item)
        return {"records": normalized, "anomalies": anomalies}


class MedicalTextLoader:
    spec = OperatorSpec(
        name="medical_text_loader",
        description="Load raw medical JSONL records into Document objects.",
        input_schema={"records": "list[dict]"},
        output_schema={"documents": "list[Document]"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        documents: List[Document] = []
        for index, row in enumerate(payload["records"], start=1):
            documents.append(
                Document(
                    doc_id=str(row.get("doc_id") or f"doc-{index}"),
                    title=str(row.get("title") or f"document-{index}"),
                    text=str(row.get("text") or ""),
                    metadata={k: v for k, v in row.items() if k not in {"doc_id", "title", "text"}},
                )
            )
        return {"documents": documents}


class MedicalTextCleaner:
    spec = OperatorSpec(
        name="medical_text_cleaner",
        description="Normalize whitespace and sentence punctuation in medical text.",
        input_schema={"documents": "list[Document]"},
        output_schema={"documents": "list[Document]"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        documents = payload["documents"]
        parallel_workers = int(payload.get("parallel_workers") or 1)
        enable_cache = bool(payload.get("enable_content_cache"))
        clean_fn = clean_text if payload.get("use_optimized_text_ops", True) else legacy_clean_text

        def clean_document(document: Document) -> Document:
            return Document(
                doc_id=document.doc_id,
                title=document.title,
                text=clean_fn(document.text),
                metadata=document.metadata,
            )

        if enable_cache:
            unique_texts = list(dict.fromkeys(document.text for document in documents))
            if len(unique_texts) == len(documents):
                enable_cache = False
            else:
                if parallel_workers > 1 and len(unique_texts) > 1:
                    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                        cleaned_texts = list(pool.map(clean_text, unique_texts))
                    text_cache = dict(zip(unique_texts, cleaned_texts))
                else:
                    text_cache = {text: clean_fn(text) for text in unique_texts}
                cleaned = [
                    Document(
                        doc_id=document.doc_id,
                        title=document.title,
                        text=text_cache[document.text],
                        metadata=document.metadata,
                    )
                    for document in documents
                ]
                return {
                    "documents": cleaned,
                    "optimization": {
                        "content_cache": True,
                        "cache_scope": "document_text",
                        "unique_texts": len(unique_texts),
                        "cache_hits": max(0, len(documents) - len(unique_texts)),
                        "parallel_workers": parallel_workers if parallel_workers > 1 else 1,
                    },
                }

        if parallel_workers > 1 and len(documents) > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                cleaned = list(pool.map(clean_document, documents))
        else:
            cleaned = [
                clean_document(document)
                for document in documents
            ]
        return {
            "documents": cleaned,
            "optimization": {
                "content_cache": False,
                "unique_texts": len({document.text for document in documents}),
                "cache_hits": 0,
                "parallel_workers": parallel_workers if parallel_workers > 1 else 1,
            },
        }


class MedicalTextSegmenter:
    spec = OperatorSpec(
        name="medical_text_segmenter",
        description="Split cleaned documents into sentence-level segments.",
        input_schema={"documents": "list[Document]"},
        output_schema={"segments": "list[Segment]"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        documents = payload["documents"]
        parallel_workers = int(payload.get("parallel_workers") or 1)
        enable_cache = bool(payload.get("enable_content_cache"))
        split_fn = split_sentences if payload.get("use_optimized_text_ops", True) else legacy_split_sentences

        def segment_document(document: Document, sentences: list[str] | None = None) -> list[Segment]:
            return [
                Segment(
                    segment_id=f"{document.doc_id}-s{order}",
                    doc_id=document.doc_id,
                    text=sentence,
                    order=order,
                    metadata=document.metadata,
                )
                for order, sentence in enumerate(sentences or split_fn(document.text), start=1)
            ]

        segments: List[Segment] = []
        if enable_cache:
            unique_texts = list(dict.fromkeys(document.text for document in documents))
            if parallel_workers > 1 and len(unique_texts) > 1:
                with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                    sentence_lists = list(pool.map(split_fn, unique_texts))
                sentence_cache = dict(zip(unique_texts, sentence_lists))
            else:
                sentence_cache = {text: split_fn(text) for text in unique_texts}
            for document in documents:
                segments.extend(segment_document(document, sentence_cache[document.text]))
            return {
                "segments": segments,
                "optimization": {
                    "content_cache": True,
                    "unique_texts": len(unique_texts),
                    "cache_hits": max(0, len(documents) - len(unique_texts)),
                    "parallel_workers": parallel_workers if parallel_workers > 1 else 1,
                },
            }

        if parallel_workers > 1 and len(documents) > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                batches = list(pool.map(segment_document, documents))
            for batch in batches:
                segments.extend(batch)
        else:
            for document in documents:
                segments.extend(segment_document(document))
        return {
            "segments": segments,
            "optimization": {
                "content_cache": False,
                "unique_texts": len({document.text for document in documents}),
                "cache_hits": 0,
                "parallel_workers": parallel_workers if parallel_workers > 1 else 1,
            },
        }
