# -*- coding: utf-8 -*-
"""任务二级联抽取使用的内部数据结构。

这些结构只描述级联编排过程，不依赖 DataMate、MCP 或数据库，便于在两个
入口之间复用，也便于用假 LLM 做确定性测试。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CascadeSegment:
    """需要交给 LLM 查缺补漏的一段原文。end_idx 为开区间。"""

    segment_id: str
    start_idx: int
    end_idx: int
    text: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReviewCandidate:
    """交给 LLM 复核的既有事实或新增候选。"""

    candidate_id: str
    kind: str
    source: str
    evidence: str = ""
    reliability_level: str = ""
    confidence: float = 0.0
    entity_text: str = ""
    entity_type: str = ""
    subject: str = ""
    subject_type: str = ""
    predicate: str = ""
    object: str = ""
    object_type: str = ""
    segment_id: str = ""
    extraction_method: str = ""

    def to_prompt_dict(self) -> dict[str, object]:
        values = {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "source": self.source,
            "evidence": self.evidence[:320],
            "reliability_level": self.reliability_level,
            "confidence": round(float(self.confidence), 4),
            "segment_id": self.segment_id,
            "extraction_method": self.extraction_method,
        }
        if self.kind == "entity":
            values.update({"text": self.entity_text, "type": self.entity_type})
        else:
            values.update(
                {
                    "subject": self.subject,
                    "subject_type": self.subject_type,
                    "predicate": self.predicate,
                    "object": self.object,
                    "object_type": self.object_type,
                }
            )
        return values


@dataclass(frozen=True)
class ReviewDecision:
    candidate_id: str
    decision: str
    reason: str = ""
    confidence: float = 0.0


@dataclass
class CascadeOutput:
    """级联抽取结果以及可观测的路由统计。"""

    entities: list
    relations: list
    triples: list
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
