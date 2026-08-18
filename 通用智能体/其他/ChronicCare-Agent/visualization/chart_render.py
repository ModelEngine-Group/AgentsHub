from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from visualization.chart_templates import metric_card_html, quality_score_html, table_html


def load_plotly() -> Any | None:
    try:
        import plotly.express as px  # type: ignore

        return px
    except Exception:
        return None


def _plotly_bar(px: Any, title: str, rows: list[dict[str, Any]], x_field: str, y_field: str, config: dict[str, Any]) -> str:
    fig = px.bar(rows, x=x_field, y=y_field, title=title, template=config["chart_defaults"]["template"])
    fig.update_layout(width=config["chart_defaults"]["width"], height=config["chart_defaults"]["height"])
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def _plotly_line(px: Any, title: str, rows: list[dict[str, Any]], x_field: str, y_field: str, config: dict[str, Any]) -> str:
    fig = px.line(rows, x=x_field, y=y_field, title=title, template=config["chart_defaults"]["template"], markers=True)
    fig.update_layout(width=config["chart_defaults"]["width"], height=config["chart_defaults"]["height"])
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def _guess_xy(item: dict[str, Any]) -> Tuple[str, str]:
    columns = item["table"]["columns"]
    if len(columns) < 2:
        column = columns[0] if columns else "value"
        return column, column
    if "disease_group" in columns and "abnormal_rate" in columns:
        return "disease_group", "abnormal_rate"
    if "drug_category" in columns and "patient_count" in columns:
        return "drug_category", "patient_count"
    if "entity_type" in columns and "node_count" in columns:
        return "entity_type", "node_count"
    if "relation_type" in columns and "edge_count" in columns:
        return "relation_type", "edge_count"
    if "month" in columns and len(columns) >= 2:
        return "month", columns[1]
    return columns[0], columns[1]


def render_indicator_chart(item: dict[str, Any], output_path: Path, config: dict[str, Any], safety_note: str) -> Dict[str, Any]:
    chart_type = item["chart_type"]
    question = item["question"]
    px = load_plotly()
    plotly_available = px is not None
    rows = item["table"]["rows"]
    columns = item["table"]["columns"]
    if chart_type == "metric_card":
        html = metric_card_html(question, item.get("metric", {}), item["insight"], safety_note)
    elif chart_type == "table":
        html = table_html(question, columns, rows, item["insight"], safety_note, "Table")
    elif chart_type == "bar":
        if len(columns) < 2 or not rows:
            html = table_html(question, columns, rows, item["insight"], safety_note, "Bar Fallback")
        else:
            x_field, y_field = _guess_xy(item)
            if plotly_available:
                html = _plotly_bar(px, question, rows, x_field, y_field, config)
            else:
                html = table_html(question, columns, rows, item["insight"], safety_note, "Bar Fallback")
    elif chart_type == "line":
        if len(columns) < 2 or not rows:
            html = table_html(question, columns, rows, item["insight"], safety_note, "Line Fallback")
        else:
            x_field, y_field = _guess_xy(item)
            if plotly_available:
                html = _plotly_line(px, question, rows, x_field, y_field, config)
            else:
                html = table_html(question, columns, rows, item["insight"], safety_note, "Line Fallback")
    else:
        html = table_html(question, columns, rows, item["insight"], safety_note, "Fallback")
    output_path.write_text(html, encoding="utf-8")
    return {
        "plotly_available": plotly_available,
        "fallback_used": not plotly_available and chart_type in {"bar", "line"},
    }


def render_graph_summary_chart(title: str, rows: list[dict[str, Any]], x_field: str, y_field: str, output_path: Path, config: dict[str, Any], safety_note: str) -> Dict[str, Any]:
    px = load_plotly()
    if px is not None:
        html = _plotly_bar(px, title, rows, x_field, y_field, config)
        fallback_used = False
    else:
        html = table_html(title, [x_field, y_field], rows, f"{title} 图谱统计概览。", safety_note, "Graph Summary")
        fallback_used = True
    output_path.write_text(html, encoding="utf-8")
    return {"plotly_available": px is not None, "fallback_used": fallback_used}


def render_quality_score(title: str, quality_score: dict[str, Any], output_path: Path, safety_note: str) -> None:
    output_path.write_text(quality_score_html(title, quality_score, safety_note), encoding="utf-8")
