"""Graph perception: load the task-2 knowledge graph and build a profile.

The analysis agent uses the profile to (a) understand the graph schema (entity
types, relation types, top entities) and (b) answer association questions by
graph traversal. If no task-2 graph exists yet, an embedded example graph is
returned so Task 3 runs standalone (placeholder for the real task-2 output).
"""
from __future__ import annotations

from pathlib import Path

from medigraph.graph.local_store import LocalGraphStore

# Embedded example graph (used only when no task-2 graph.json is present). Mirrors
# the seed corpus so the relational layer and graph stay consistent.
_EXAMPLE_TRIPLES = [
    ("高血压", "Disease", "has_symptom", "头痛", "Symptom"),
    ("高血压", "Disease", "has_symptom", "头晕", "Symptom"),
    ("高血压", "Disease", "recommend_drug", "硝苯地平", "Drug"),
    ("高血压", "Disease", "recommend_drug", "氨氯地平", "Drug"),
    ("高血压", "Disease", "need_examination", "心电图", "Examination"),
    ("高血压", "Disease", "complication", "冠心病", "Disease"),
    ("高血压", "Disease", "treated_in_department", "心内科", "Department"),
    ("2型糖尿病", "Disease", "recommend_drug", "二甲双胍", "Drug"),
    ("2型糖尿病", "Disease", "complication", "糖尿病肾病", "Disease"),
    ("2型糖尿病", "Disease", "need_examination", "糖化血红蛋白", "Examination"),
    ("2型糖尿病", "Disease", "treated_in_department", "内分泌科", "Department"),
    ("急性心肌梗死", "Disease", "has_symptom", "胸痛", "Symptom"),
    ("急性心肌梗死", "Disease", "recommend_drug", "阿司匹林", "Drug"),
    ("急性心肌梗死", "Disease", "treated_in_department", "心内科", "Department"),
    ("嗜铬细胞瘤", "Tumor", "positive_marker", "嗜铬粒蛋白", "Biomarker"),
    ("嗜铬细胞瘤", "Tumor", "associated_gene", "RET", "Gene"),
    ("嗜铬细胞瘤", "Tumor", "treated_in_department", "内分泌科", "Department"),
]


def load_graph(graph_json: str | Path | None) -> tuple[LocalGraphStore, bool]:
    """Return (store, used_example). Loads task-2 graph.json if available."""
    if graph_json and Path(graph_json).exists():
        return LocalGraphStore.load_json(graph_json), False
    store = LocalGraphStore()
    for h, ht, r, t, tt in _EXAMPLE_TRIPLES:
        store.upsert_triple(h, ht, r, t, tt, confidence=1.0, source="example_graph")
    return store, True


def build_profile(store: LocalGraphStore) -> dict:
    """Structured profile of the graph for schema injection / understanding."""
    stats = store.stats()
    return {
        "num_entities": stats["num_entities"],
        "num_triples": stats["num_triples"],
        "entity_types": stats["entity_type_counts"],
        "relation_types": stats["relation_counts"],
        "top_entities": stats["top_entities"],
    }


def profile_prompt_block(profile: dict) -> str:
    """Human/LLM-readable graph schema summary."""
    ets = ", ".join(f"{k}({v})" for k, v in profile["entity_types"].items())
    rels = ", ".join(f"{k}({v})" for k, v in profile["relation_types"].items())
    return (
        f"知识图谱概览：实体 {profile['num_entities']} 个，三元组 {profile['num_triples']} 条。\n"
        f"实体类型分布：{ets}\n关系类型分布：{rels}"
    )
