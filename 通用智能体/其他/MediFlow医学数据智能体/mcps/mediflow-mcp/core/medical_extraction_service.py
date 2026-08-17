# -*- coding: utf-8 -*-
"""
任务二医学抽取服务。

该模块统一调度实体识别、关系抽取和三元组生成能力，并支持本地规则链与模型增强链按配置切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

from .llm_client import LLMClient
from .medical_ner import extract_entities as extract_entities_llm
from .medical_re import extract_relations as extract_relations_llm
from .medical_extraction_validation import relations_to_triples
from .medical_offline_extraction import (
    extract_entities_offline,
    extract_relations_offline,
    generate_triples_offline,
)
from .schemas import Entity, Relation, Triple
from .task2_cascade import extract_medical_knowledge_cascade
from .task2_cascade import filter_low_reliability_results


VALID_BACKENDS = {"offline", "llm", "hybrid"}


@dataclass
class ExtractionBundle:
    entities: list[Entity]
    relations: list[Relation]
    triples: list[Triple]
    backend: str
    elapsed_seconds: float
    llm_error: str = ""
    gap_segment_count: int = 0
    gap_candidate_count: int = 0
    reviewed_candidate_count: int = 0
    auto_accepted_candidate_count: int = 0
    review_skipped_candidate_count: int = 0
    offline_filtered_candidate_count: int = 0
    rejected_candidate_count: int = 0
    llm_added_count: int = 0
    llm_added_entity_count: int = 0
    llm_added_relation_count: int = 0
    gap_budget_skipped_count: int = 0
    review_budget_skipped_count: int = 0


def normalize_backend(value: str | None) -> str:
    backend = (value or "offline").strip().lower()
    return backend if backend in VALID_BACKENDS else "offline"


def _merge_entities(primary: Iterable[Entity], secondary: Iterable[Entity]) -> list[Entity]:
    merged: list[Entity] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for entity in list(primary) + list(secondary):
        key = (entity.text, entity.type, entity.start_idx, entity.end_idx)
        if key in seen:
            continue
        seen.add(key)
        merged.append(entity)
    return sorted(merged, key=lambda item: (item.start_idx or 0, -len(item.text)))


def _merge_relations(primary: Iterable[Relation], secondary: Iterable[Relation]) -> list[Relation]:
    merged: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in list(primary) + list(secondary):
        key = (relation.subject, relation.predicate, relation.object)
        if key in seen:
            continue
        seen.add(key)
        merged.append(relation)
    return merged


def extract_medical_knowledge(
    text: str,
    *,
    backend: str = "offline",
    kg_db_path: str = "",
    llm: LLMClient | None = None,
    apply_offline_gate: bool = True,
) -> ExtractionBundle:
    """从医学文本中抽取实体、关系和三元组。

    ``offline`` never calls an LLM. ``llm`` uses the existing full-text LLM
    extractors. ``hybrid`` is the production cascade: offline scans the full
    text first, then LLM reviews uncertain offline relations and fills selected
    uncovered sentences. LLM failures are reported without discarding offline
    results. ``apply_offline_gate=False`` is reserved for the batch cascade's
    internal candidate-routing phase; callers still receive a gated result from
    the cascade merge.
    """
    selected = normalize_backend(backend)
    started = perf_counter()

    if selected == "llm":
        if llm is None:
            raise ValueError("llm backend requires an LLMClient")
        entities = extract_entities_llm(text, llm)
        relations = extract_relations_llm(text, llm, entities=entities)
        triples = relations_to_triples(relations, min_confidence=0.0)
        return ExtractionBundle(
            entities=entities,
            relations=relations,
            triples=triples,
            backend="llm",
            elapsed_seconds=round(perf_counter() - started, 4),
        )

    llm_error = ""
    if selected == "hybrid" and llm is not None:
        try:
            cascade = extract_medical_knowledge_cascade(
                text,
                kg_db_path=kg_db_path,
                llm=llm,
            )
            return ExtractionBundle(
                entities=cascade.entities,
                relations=cascade.relations,
                triples=cascade.triples,
                backend="hybrid",
                elapsed_seconds=round(perf_counter() - started, 4),
                gap_segment_count=cascade.gap_segment_count,
                gap_candidate_count=cascade.gap_candidate_count,
                reviewed_candidate_count=cascade.reviewed_candidate_count,
                auto_accepted_candidate_count=cascade.auto_accepted_candidate_count,
                review_skipped_candidate_count=cascade.review_skipped_candidate_count,
                offline_filtered_candidate_count=cascade.offline_filtered_candidate_count,
                rejected_candidate_count=cascade.rejected_candidate_count,
                llm_added_count=cascade.llm_added_count,
                llm_added_entity_count=cascade.llm_added_entity_count,
                llm_added_relation_count=cascade.llm_added_relation_count,
            )
        except Exception as exc:
            llm_error = f"{type(exc).__name__}: {str(exc)[:240]}"

    raw_entities = extract_entities_offline(text, kg_db_path)
    raw_relations = extract_relations_offline(
        text,
        entities=raw_entities,
        db_path=kg_db_path,
    )
    if apply_offline_gate:
        entities, relations, filtered_count = filter_low_reliability_results(
            raw_entities,
            raw_relations,
        )
    else:
        entities, relations, filtered_count = raw_entities, raw_relations, 0
    triples = generate_triples_offline(
        text,
        entities=entities,
        relations=relations,
        db_path=kg_db_path,
    )
    if selected == "hybrid" and llm is None:
        llm_error = "LLM client is not configured; returned offline results only"

    return ExtractionBundle(
        entities=entities,
        relations=relations,
        triples=triples,
        backend=selected,
        elapsed_seconds=round(perf_counter() - started, 4),
        llm_error=llm_error,
        offline_filtered_candidate_count=filtered_count,
    )
