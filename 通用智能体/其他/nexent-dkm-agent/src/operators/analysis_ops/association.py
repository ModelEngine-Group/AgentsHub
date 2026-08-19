"""Association analysis operators for task 3."""

from __future__ import annotations

from typing import Any

_PROFILE_BUCKETS = {
    "has_symptom": "symptoms",
    "treated_by": "drugs",
    "diagnosed_by": "examinations",
    "recommended_treatment": "treatments",
    "complication_of": "complications",
}


def generate_association_analysis(graph: dict[str, Any]) -> dict[str, Any]:
    """Build disease-centric association profiles from graph edges."""

    node_lookup = {node["id"]: node for node in graph.get("nodes", [])}
    profiles: dict[str, dict[str, Any]] = {}

    for edge in graph.get("edges", []):
        predicate = edge.get("predicate")
        bucket = _PROFILE_BUCKETS.get(predicate)
        if not bucket:
            continue
        disease_id = edge["target"] if predicate == "complication_of" else edge["source"]
        target_id = edge["source"] if predicate == "complication_of" else edge["target"]
        disease = node_lookup.get(disease_id, {"name": disease_id, "type": "Unknown"})
        target = node_lookup.get(target_id, {"name": target_id, "type": "Unknown"})
        if disease.get("type") != "Disease":
            continue
        profile = profiles.setdefault(
            disease_id,
            {
                "disease_id": disease_id,
                "disease": disease.get("name", disease_id),
                "symptoms": [],
                "drugs": [],
                "examinations": [],
                "treatments": [],
                "complications": [],
                "evidence_count": 0,
            },
        )
        target_name = target.get("name", target_id)
        if target_name not in profile[bucket]:
            profile[bucket].append(target_name)
        profile["evidence_count"] += len(edge.get("evidence", [])) or 1

    disease_profiles = sorted(
        profiles.values(),
        key=lambda item: (-sum(len(item[b]) for b in _PROFILE_BUCKETS.values()), item["disease"]),
    )
    relation_matrix = _build_relation_matrix(disease_profiles)
    return {
        "status": "completed",
        "disease_profiles": disease_profiles,
        "relation_matrix": relation_matrix,
        "top_associations": disease_profiles[:5],
    }


def _build_relation_matrix(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for profile in profiles:
        rows.append(
            {
                "disease": profile["disease"],
                "symptom_count": len(profile["symptoms"]),
                "drug_count": len(profile["drugs"]),
                "examination_count": len(profile["examinations"]),
                "treatment_count": len(profile["treatments"]),
                "complication_count": len(profile["complications"]),
            }
        )
    return rows
