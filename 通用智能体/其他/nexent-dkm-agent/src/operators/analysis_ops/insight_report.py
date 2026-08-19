"""Human-readable insight report export for task 3."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from src.operators.analysis_ops.dashboard import export_analysis_dashboard


def export_insight_report(
    target_dir: str | Path,
    graph: dict[str, Any],
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
    nl2sql: dict[str, Any],
    visualizations: dict[str, Any],
    centrality: dict[str, Any] | None = None,
    graph_analytics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Export task-3 insights as Markdown and static HTML."""

    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = _build_markdown(
        graph, statistics, associations, trends, nl2sql, visualizations,
        centrality, graph_analytics,
    )
    html_text = _markdown_to_html(markdown, visualizations)
    dashboard = export_analysis_dashboard(
        target_dir=output_dir,
        statistics=statistics,
        associations=associations,
        trends=trends,
        nl2sql=nl2sql,
        visualizations=visualizations,
        centrality=centrality,
    )

    markdown_path = output_dir / "task3_insight_report.md"
    html_path = output_dir / "task3_insight_report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return {
        "status": "completed",
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "dashboard_path": dashboard["dashboard_path"],
    }


def _build_markdown(
    graph: dict[str, Any],
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
    nl2sql: dict[str, Any],
    visualizations: dict[str, Any],
    centrality: dict[str, Any] | None = None,
    graph_analytics: dict[str, Any] | None = None,
) -> str:
    graph_stats = graph.get("statistics", {})
    confidence = statistics.get("confidence", {})
    top_nodes = statistics.get("top_degree_nodes", [])[:5]
    top_profiles = associations.get("top_associations", [])[:5]
    peak_record = trends.get("peak_record") or {}
    chart_names = sorted(visualizations.get("charts", {}).keys())

    lines = [
        "# Task 3 Graph Analysis Insight Report",
        "",
        "## Executive Summary",
        "",
        f"- Graph size: {graph_stats.get('node_count', len(graph.get('nodes', [])))} nodes, "
        f"{graph_stats.get('edge_count', len(graph.get('edges', [])))} edges.",
        f"- Relation confidence: average={confidence.get('average', 0)}, "
        f"min={confidence.get('min', 0)}, max={confidence.get('max', 0)}.",
        f"- NL2SQL intent: `{nl2sql.get('intent', 'unknown')}` with {len(nl2sql.get('rows', []))} result rows.",
        f"- Visualization specs: {', '.join(chart_names)}.",
        "",
        "## Top Graph Hubs",
        "",
        "| Rank | Entity | Degree |",
        "| --- | --- | ---: |",
    ]
    for index, node in enumerate(top_nodes, start=1):
        lines.append(f"| {index} | {node.get('name', '')} | {node.get('degree', 0)} |")

    lines.extend([
        "",
        "## Disease Association Highlights",
        "",
        "| Disease | Symptoms | Drugs | Examinations | Complications |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for profile in top_profiles:
        lines.append(
            f"| {profile.get('disease', '')} | {len(profile.get('symptoms', []))} | "
            f"{len(profile.get('drugs', []))} | {len(profile.get('examinations', []))} | "
            f"{len(profile.get('complications', []))} |"
        )

    insight = generate_graph_insights(
        graph, statistics, associations, centrality, graph_analytics
    )
    if insight.get("insights"):
        lines.extend([
            "",
            "## 图谱驱动的自然语言洞察 (Graph-Driven Insights)",
            "",
            "以下结论由知识图谱结构（实体、关系、枢纽、社区）自动生成：",
            "",
        ])
        lines.extend(f"- {sentence}" for sentence in insight["insights"])

    _append_graph_analytics_sections(lines, centrality, graph_analytics)

    lines.extend([
        "",
        "## Trend Signal",
        "",
        f"- Peak record: {peak_record.get('record_id', 'n/a')} with "
        f"{peak_record.get('edge_count', 0)} graph edges.",
        "",
        "## NL2SQL Evidence",
        "",
        "```sql",
        nl2sql.get("sql", ""),
        "```",
        "",
        "Rows:",
        "",
        "```json",
        json.dumps(nl2sql.get("rows", []), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Visualization Spec Index",
        "",
    ])
    for name in chart_names:
        chart = visualizations["charts"][name]
        lines.append(f"- `{name}`: {chart.get('type')} - {chart.get('title')}")
    lines.append("")
    return "\n".join(lines)


_PREDICATE_LABELS_ZH = {
    "has_symptom": "症状",
    "treated_by": "用药",
    "diagnosed_by": "检查",
    "recommended_treatment": "治疗",
    "complication_of": "并发",
}
_TYPE_LABELS_ZH = {
    "Disease": "疾病",
    "Symptom": "症状",
    "Drug": "药物",
    "Examination": "检查",
    "Treatment": "治疗",
}


def generate_graph_insights(
    graph: dict[str, Any],
    statistics: dict[str, Any] | None = None,
    associations: dict[str, Any] | None = None,
    centrality: dict[str, Any] | None = None,
    graph_analytics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive natural-language insights directly from the knowledge graph.

    Produces a deterministic (no-LLM) free-form narrative grounded in the graph
    structure: scale and node-type mix, the dominant hub, the predicate
    distribution, the richest disease profile, multi-hop diagnosis→treatment
    chains, complication links, and community structure. Returns
    ``{"status", "insights": [str, ...], "metrics": {...}}``.
    """

    statistics = statistics or {}
    associations = associations or {}
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []

    if not nodes or not edges:
        return {"status": "skipped", "insights": [], "metrics": {}}

    type_counts: dict[str, int] = {}
    for node in nodes:
        ntype = node.get("type", "Unknown")
        type_counts[ntype] = type_counts.get(ntype, 0) + 1

    predicate_counts: dict[str, int] = {}
    for edge in edges:
        pred = edge.get("predicate", "unknown")
        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

    insights: list[str] = []

    type_mix = "、".join(
        f"{_TYPE_LABELS_ZH.get(t, t)}{c}"
        for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])
    )
    insights.append(
        f"知识图谱由 {len(nodes)} 个实体与 {len(edges)} 条关系组成，"
        f"覆盖 {len(type_counts)} 类节点（{type_mix}）。"
    )

    total_edges = len(edges)
    top_pred = max(predicate_counts.items(), key=lambda kv: kv[1])
    pred_share = round(100 * top_pred[1] / total_edges, 1) if total_edges else 0.0
    pred_breakdown = "、".join(
        f"{_PREDICATE_LABELS_ZH.get(p, p)} {c} 条"
        for p, c in sorted(predicate_counts.items(), key=lambda kv: -kv[1])
    )
    insights.append(
        f"关系以 {_PREDICATE_LABELS_ZH.get(top_pred[0], top_pred[0])}"
        f"（{top_pred[0]}）为主，占比约 {pred_share}%；完整分布：{pred_breakdown}。"
    )

    top_hub = None
    if centrality and centrality.get("status") == "completed":
        hubs = centrality.get("top_hubs", [])
        if hubs:
            top_hub = hubs[0]
            backend = centrality.get("top_hubs_backend", "python")
            insights.append(
                f"最核心的实体是「{top_hub.get('name', top_hub.get('id', ''))}」"
                f"（类型 {_TYPE_LABELS_ZH.get(top_hub.get('type', ''), top_hub.get('type', ''))}），"
                f"直接连接 {top_hub.get('degree', 0)} 个实体，是图谱主要枢纽"
                f"（中心性由 `{backend}` 计算）。"
            )

    top_profiles = associations.get("top_associations", [])
    if top_profiles:
        profile = top_profiles[0]
        disease = profile.get("disease", "")
        symptoms = profile.get("symptoms", [])
        drugs = profile.get("drugs", [])
        exams = profile.get("examinations", [])
        insights.append(
            f"关联最丰富的疾病是「{disease}」：{len(symptoms)} 个症状、"
            f"{len(drugs)} 种药物、{len(exams)} 项检查。"
        )
        if symptoms and drugs:
            insights.append(
                f"可形成诊疗链路：「{disease}」—has_symptom→「{symptoms[0]}」、"
                f"「{disease}」—treated_by→「{drugs[0]}」，"
                f"为「症状识别→对症用药」提供图谱依据。"
            )

    complication_count = predicate_counts.get("complication_of", 0)
    if complication_count:
        example = next(
            (e for e in edges if e.get("predicate") == "complication_of"), None
        )
        example_text = ""
        if example:
            example_text = f"，例如「{example.get('source', '')}」并发于「{example.get('target', '')}」"
        insights.append(
            f"图谱包含 {complication_count} 条并发关系{example_text}，"
            f"提示需关注疾病间的合并风险。"
        )

    if graph_analytics and graph_analytics.get("status") == "completed":
        communities = (graph_analytics.get("communities") or {})
        count = communities.get("community_count", len(communities.get("communities", [])))
        if count:
            insights.append(
                f"社区检测将图谱划分为 {count} 个相对独立的子结构，"
                f"对应不同的疾病-症状-用药聚簇。"
            )

    return {
        "status": "completed",
        "insights": insights,
        "metrics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "type_counts": type_counts,
            "predicate_counts": predicate_counts,
            "top_hub": (top_hub or {}).get("name") if top_hub else None,
        },
    }


def _append_graph_analytics_sections(
    lines: list[str],
    centrality: dict[str, Any] | None,
    graph_analytics: dict[str, Any] | None,
) -> None:
    """Append graph-driven analytics (centrality, communities, reachability).

    These sections turn the computed graph analytics into the human-readable
    narrative required by task 3, instead of leaving them only in the JSON
    artifact.
    """

    if centrality and centrality.get("status") == "completed":
        top_hubs = centrality.get("top_hubs", [])[:5]
        backend = centrality.get("top_hubs_backend", "python")
        lines.extend([
            "",
            "## Graph Centrality",
            "",
            f"Degree centrality identifies the most connected entities "
            f"(compute backend: `{backend}`).",
            "",
            "| Rank | Entity | Type | Degree | Centrality |",
            "| --- | --- | --- | ---: | ---: |",
        ])
        for index, hub in enumerate(top_hubs, start=1):
            lines.append(
                f"| {index} | {hub.get('name', hub.get('id', ''))} | "
                f"{hub.get('type', '')} | {hub.get('degree', 0)} | "
                f"{hub.get('degree_centrality', 0)} |"
            )
        type_centrality = centrality.get("type_centrality", {})
        if type_centrality:
            lines.extend(["", "Per-type hubs:", ""])
            for node_type, info in type_centrality.items():
                lines.append(
                    f"- {node_type}: {info.get('count', 0)} nodes, "
                    f"top hub `{info.get('top_node', 'n/a')}` "
                    f"(avg degree {info.get('avg_degree', 0)})."
                )

    if graph_analytics and graph_analytics.get("status") == "completed":
        communities = (graph_analytics.get("communities") or {})
        community_list = communities.get("communities", [])[:5]
        if community_list:
            lines.extend([
                "",
                "## Community Structure",
                "",
                f"Label propagation detected "
                f"{communities.get('community_count', len(community_list))} communities; "
                f"the largest are summarised below.",
                "",
                "| Community | Size | Dominant types |",
                "| --- | ---: | --- |",
            ])
            for community in community_list:
                dist = community.get("type_distribution", {})
                dominant = ", ".join(
                    f"{ctype}×{count}"
                    for ctype, count in sorted(dist.items(), key=lambda kv: -kv[1])[:3]
                )
                lines.append(
                    f"| {community.get('community_id', '')} | "
                    f"{community.get('size', 0)} | {dominant or 'n/a'} |"
                )

        paths = graph_analytics.get("shortest_paths") or {}
        start_hub = graph_analytics.get("start_hub")
        if paths.get("status") == "reachable":
            lines.extend([
                "",
                "## Reachability From Top Hub",
                "",
                f"From hub `{start_hub}`, {paths.get('reachable_count', 0)} entities "
                f"are reachable within {paths.get('max_depth', 0)} hops.",
            ])
        elif paths.get("status") == "path_found":
            steps = (paths.get("paths") or [{}])[0].get("steps", [])
            chain = " -> ".join(
                [steps[0]["source"]] + [step["target"] for step in steps]
            ) if steps else ""
            lines.extend([
                "",
                "## Reachability From Top Hub",
                "",
                f"Shortest path from `{paths.get('start_entity', start_hub)}` to "
                f"`{paths.get('end_entity', '')}`: {chain}",
            ])


def _markdown_to_html(markdown: str, visualizations: dict[str, Any]) -> str:
    escaped = html.escape(markdown)
    charts_json = html.escape(json.dumps(visualizations.get("charts", {}), ensure_ascii=False, indent=2))
    body = escaped.replace("\n", "<br>\n")
    chart_grid = _chart_grid_html(visualizations.get("charts", {}))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Task 3 Graph Analysis Insight Report</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5f6b7a;
      --line: #d9e2ec;
      --panel: #ffffff;
      --page: #f5f7fb;
      --accent: #246bfe;
      --accent-2: #0f8f70;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 0; line-height: 1.55; color: var(--ink); background: var(--page); }}
    h1, h2 {{ color: #0f172a; }}
    code, pre {{ background: #f3f4f6; border-radius: 4px; padding: 2px 4px; }}
    .report {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 40px; }}
    .markdown-body, .chart-card, .charts {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .markdown-body {{ padding: 20px; }}
    .insight-chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .chart-card {{ padding: 16px; overflow: hidden; }}
    .chart-card h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .chart-meta {{ color: var(--muted); font-size: 13px; margin: 0 0 10px; }}
    .chart-svg {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fbfdff; }}
    .charts {{ margin-top: 18px; padding: 16px; }}
    .charts summary {{ cursor: pointer; font-weight: 700; }}
    @media (max-width: 860px) {{
      .insight-chart-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="report">
    <section class="markdown-body">{body}</section>
    {chart_grid}
    <details class="charts">
      <summary>Raw Visualization Specs</summary>
      <pre>{charts_json}</pre>
    </details>
  </main>
</body>
</html>
"""


def _chart_grid_html(charts: dict[str, Any]) -> str:
    if not charts:
        return '<section class="insight-chart-grid"><article class="chart-card">No chart data.</article></section>'
    cards = [_chart_card_html(name, chart) for name, chart in charts.items()]
    return '<section class="insight-chart-grid">' + "\n".join(cards) + "</section>"


def _chart_card_html(name: str, chart: dict[str, Any]) -> str:
    chart_type = chart.get("type", "unknown")
    if chart_type == "bar":
        rendered = _bar_svg(chart)
    elif chart_type == "line":
        rendered = _line_svg(chart)
    elif chart_type == "network":
        rendered = _network_svg(chart)
    else:
        rendered = f"<pre>{_text(json.dumps(chart, ensure_ascii=False, indent=2))}</pre>"
    return (
        '<article class="chart-card">'
        f"<h2>{_text(chart.get('title', name))}</h2>"
        f'<p class="chart-meta">{_text(name)} / {_text(chart_type)}</p>'
        f"{rendered}</article>"
    )


def _bar_svg(chart: dict[str, Any]) -> str:
    rows = chart.get("data", [])
    if not rows:
        return '<p class="chart-meta">No bar data.</p>'
    width = 720
    row_height = 32
    label_width = 180
    value_width = 56
    height = 36 + row_height * len(rows)
    max_value = max(float(row.get("value", 0) or 0) for row in rows) or 1.0
    bars = []
    for index, row in enumerate(rows):
        y = 24 + index * row_height
        value = float(row.get("value", 0) or 0)
        bar_width = max(2.0, (width - label_width - value_width - 52) * value / max_value)
        bars.append(
            f'<text x="14" y="{y + 17}" font-size="13" fill="#172033">{_text(_short(row.get("category", "")))}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" height="18" rx="4" fill="#246bfe" />'
            f'<text x="{width - value_width}" y="{y + 15}" font-size="13" fill="#172033">{_text(row.get("value", 0))}</text>'
        )
    return f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img">{"".join(bars)}</svg>'


def _line_svg(chart: dict[str, Any]) -> str:
    rows = chart.get("data", [])
    if not rows:
        return '<p class="chart-meta">No line data.</p>'
    y_key = chart.get("encoding", {}).get("y", "value")
    values = [float(row.get(y_key, 0) or 0) for row in rows]
    width = 720
    height = 260
    pad_x = 42
    pad_y = 32
    max_value = max(values) or 1.0
    x_step = (width - pad_x * 2) / max(1, len(rows) - 1)
    points = []
    markers = []
    labels = []
    for index, value in enumerate(values):
        x = pad_x + x_step * index
        y = height - pad_y - (value / max_value) * (height - pad_y * 2)
        points.append(f"{x:.1f},{y:.1f}")
        markers.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#246bfe" />')
        labels.append(f'<text x="{x:.1f}" y="{height - 10}" text-anchor="middle" font-size="10">{index + 1}</text>')
    return (
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img">'
        f'<line x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" stroke="#d9e2ec" />'
        f'<line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{height - pad_y}" stroke="#d9e2ec" />'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#246bfe" stroke-width="3" />'
        f'{"".join(markers)}{"".join(labels)}</svg>'
    )


def _network_svg(chart: dict[str, Any]) -> str:
    nodes = chart.get("nodes", [])[:10]
    edges = chart.get("edges", [])[:18]
    if not nodes:
        return '<p class="chart-meta">No network data.</p>'
    width = 720
    height = 360
    center_x = width / 2
    center_y = height / 2
    positions: dict[str, tuple[float, float]] = {}
    positions[str(nodes[0].get("id", "root"))] = (center_x, center_y)
    outer = nodes[1:]
    for index, node in enumerate(outer):
        angle = 2 * math.pi * index / max(1, len(outer))
        x = center_x + 230 * math.cos(angle)
        y = center_y + 115 * math.sin(angle)
        positions[str(node.get("id", index))] = (x, y)

    edge_svg = []
    for edge in edges:
        source = positions.get(str(edge.get("source")))
        target = positions.get(str(edge.get("target")))
        if source and target:
            edge_svg.append(
                f'<line x1="{source[0]:.1f}" y1="{source[1]:.1f}" x2="{target[0]:.1f}" y2="{target[1]:.1f}" '
                'stroke="#b9c6d4" stroke-width="1.4" />'
            )

    node_svg = []
    for node in nodes:
        node_id = str(node.get("id", ""))
        x, y = positions.get(node_id, (center_x, center_y))
        fill = "#246bfe" if node.get("type") == "Disease" else "#0f8f70"
        node_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="18" fill="{fill}" opacity="0.9" />'
            f'<text x="{x:.1f}" y="{y + 34:.1f}" text-anchor="middle" font-size="11" fill="#172033">'
            f'{_text(_short(node.get("label", node_id), 10))}</text>'
        )
    return f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img">{"".join(edge_svg)}{"".join(node_svg)}</svg>'


def _short(value: Any, limit: int = 18) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)
