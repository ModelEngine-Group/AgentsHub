"""Visualization spec builders for task 3."""

from __future__ import annotations

from typing import Any


def build_analysis_visualizations(
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
) -> dict[str, Any]:
    """Build serializable chart specs for BI-style rendering."""

    charts = {
        "entity_distribution": _bar_chart(
            title="Entity type distribution",
            x=list(statistics.get("entity_type_counts", {}).keys()),
            y=list(statistics.get("entity_type_counts", {}).values()),
        ),
        "relation_distribution": _bar_chart(
            title="Relation type distribution",
            x=list(statistics.get("relation_type_counts", {}).keys()),
            y=list(statistics.get("relation_type_counts", {}).values()),
        ),
        "record_trend": _record_trend_chart(trends.get("record_trends", [])),
        "disease_network": _network_chart(associations.get("disease_profiles", [])),
    }
    return {"status": "completed", "charts": charts}


def _bar_chart(title: str, x: list[Any], y: list[Any]) -> dict[str, Any]:
    return {
        "type": "bar",
        "title": title,
        "encoding": {"x": "category", "y": "value"},
        "data": [{"category": category, "value": value} for category, value in zip(x, y)],
    }


def _record_trend_chart(record_trends: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "line",
        "title": "Record sequence trend",
        "encoding": {"x": "record_id", "y": "edge_count"},
        "data": [
            {
                "record_id": row["record_id"],
                "edge_count": row["edge_count"],
                "entity_count": row["entity_count"],
            }
            for row in record_trends
        ],
    }


def _network_chart(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges = []
    for profile in profiles:
        disease = profile["disease"]
        nodes[disease] = {"id": disease, "label": disease, "type": "Disease"}
        for bucket, target_type in (
            ("symptoms", "Symptom"),
            ("drugs", "Drug"),
            ("examinations", "Examination"),
            ("treatments", "Treatment"),
            ("complications", "Disease"),
        ):
            for target in profile.get(bucket, []):
                nodes[target] = {"id": target, "label": target, "type": target_type}
                edges.append({"source": disease, "target": target, "relation": bucket})
    return {
        "type": "network",
        "title": "Disease-centered relation network",
        "nodes": list(nodes.values()),
        "edges": edges,
    }
