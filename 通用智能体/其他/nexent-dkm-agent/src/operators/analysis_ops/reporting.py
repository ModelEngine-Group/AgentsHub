"""Task-3 analysis quality reporting."""

from __future__ import annotations

from typing import Any


def build_analysis_report(
    graph: dict[str, Any],
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
    nl2sql: dict[str, Any],
    visualizations: dict[str, Any],
    insight_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize task-3 readiness and reproducibility evidence."""

    charts = visualizations.get("charts", {})
    readiness = {
        "graph_loaded": bool(graph.get("nodes")) and bool(graph.get("edges")),
        "statistics_ready": statistics.get("status") == "completed",
        "associations_ready": associations.get("status") == "completed",
        "trends_ready": trends.get("status") == "completed",
        "nl2sql_answered": nl2sql.get("status") == "completed" and bool(nl2sql.get("rows")),
        "visualizations_ready": visualizations.get("status") == "completed" and len(charts) >= 3,
        "insight_report_exported": (
            insight_report is not None
            and insight_report.get("status") == "completed"
            and bool(insight_report.get("html_path"))
            and bool(insight_report.get("markdown_path"))
        ),
        "dashboard_exported": (
            insight_report is not None
            and insight_report.get("status") == "completed"
            and bool(insight_report.get("dashboard_path"))
        ),
    }
    status = "passed" if all(readiness.values()) else "warning"
    return {
        "status": status,
        "readiness": readiness,
        "metrics": {
            "node_count": graph.get("statistics", {}).get("node_count", len(graph.get("nodes", []))),
            "edge_count": graph.get("statistics", {}).get("edge_count", len(graph.get("edges", []))),
            "chart_count": len(charts),
            "nl2sql_row_count": len(nl2sql.get("rows", [])),
            "disease_profile_count": len(associations.get("disease_profiles", [])),
        },
    }
