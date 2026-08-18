"""Relation mapping between the full CMeIE schema and the demo clinical ontology.

The trained neural GPLinker is evaluated on the complete CMeIE predicate space
(``cmeie:*``).  The user-facing graph/QA/BI layer intentionally keeps a smaller
clinical ontology such as ``has_symptom`` and ``recommend_drug`` so prompts,
GraphRAG intents and dashboards stay readable.  This module bridges the two
without hiding provenance: mapped triples keep their original CMeIE predicate.
"""
from __future__ import annotations

from medigraph.schema.ontology import check_relation_constraint
from medigraph.schema.normalize import canonical_key


CMEIE_TO_CLINICAL_RELATION: dict[str, str] = {
    "cmeie:clinical_manifestation": "has_symptom",
    "cmeie:related_symptom": "has_symptom",
    "cmeie:post_treatment_symptom": "has_symptom",
    "cmeie:invasion_symptom": "has_symptom",
    "cmeie:drug_treatment": "recommend_drug",
    "cmeie:laboratory_examination": "need_examination",
    "cmeie:imaging_examination": "need_examination",
    "cmeie:auxiliary_examination": "need_examination",
    "cmeie:histological_examination": "need_examination",
    "cmeie:endoscopic_examination": "need_examination",
    "cmeie:screening": "need_examination",
    "cmeie:complication": "complication",
    "cmeie:department": "treated_in_department",
    "cmeie:onset_site": "located_in",
    "cmeie:metastatic_site": "located_in",
    "cmeie:invasion_site": "located_in",
    "cmeie:surgical_treatment": "treated_by_procedure",
}


def entity_type_lookup(*entity_lists: list[dict]) -> dict[str, str]:
    """Build a canonical surface -> type lookup from one or more entity lists."""
    lookup: dict[str, str] = {}
    for entities in entity_lists:
        for item in entities or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "")).strip()
            if name and etype:
                lookup.setdefault(canonical_key(name), etype)
    return lookup


def map_cmeie_triple_to_clinical(triple: dict, type_lookup: dict[str, str] | None = None) -> dict | None:
    """Return a clinical-ontology triple, or ``None`` when no safe mapping exists."""
    relation = str(triple.get("relation", "")).strip()
    mapped = CMEIE_TO_CLINICAL_RELATION.get(relation)
    if not mapped:
        return None

    head = str(triple.get("head", "")).strip()
    tail = str(triple.get("tail", "")).strip()
    if not head or not tail:
        return None
    type_lookup = type_lookup or {}
    head_type = str(triple.get("head_type", "")).strip() or type_lookup.get(canonical_key(head), "")
    tail_type = str(triple.get("tail_type", "")).strip() or type_lookup.get(canonical_key(tail), "")
    if not check_relation_constraint(mapped, head_type, tail_type):
        return None

    out = dict(triple)
    out["relation"] = mapped
    out["head_type"] = head_type
    out["tail_type"] = tail_type
    out["cmeie_relation"] = relation
    if triple.get("predicate"):
        out["cmeie_predicate"] = triple.get("predicate")
    out["evidence_method"] = out.get("evidence_method", "neural_cmeie_mapped")
    return out


def map_cmeie_triples_to_clinical(
    triples: list[dict],
    *entity_lists: list[dict],
) -> list[dict]:
    """Map a list of CMeIE triples and deduplicate by clinical triple key."""
    lookup = entity_type_lookup(*entity_lists)
    merged: dict[tuple[str, str, str], dict] = {}
    for triple in triples or []:
        mapped = map_cmeie_triple_to_clinical(triple, lookup)
        if mapped is None:
            continue
        key = (
            canonical_key(str(mapped.get("head", ""))),
            str(mapped.get("relation", "")),
            canonical_key(str(mapped.get("tail", ""))),
        )
        old = merged.get(key)
        if old is None or float(mapped.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
            merged[key] = mapped
    return sorted(
        merged.values(),
        key=lambda item: (-float(item.get("confidence", 0.0)), item.get("head", ""), item.get("relation", "")),
    )
