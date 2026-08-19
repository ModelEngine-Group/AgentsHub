"""Static HTML dashboard exporter for task 3 analysis results."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def export_analysis_dashboard(
    target_dir: str | Path,
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
    nl2sql: dict[str, Any],
    visualizations: dict[str, Any],
    centrality: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Export a dependency-free HTML dashboard from task-3 chart specs."""

    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "task3_analysis_dashboard.html"
    dashboard_path.write_text(
        _build_dashboard_html(
            statistics, associations, trends, nl2sql, visualizations, centrality,
        ),
        encoding="utf-8",
    )
    return {"status": "completed", "dashboard_path": str(dashboard_path)}


def _build_dashboard_html(
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
    nl2sql: dict[str, Any],
    visualizations: dict[str, Any],
    centrality: dict[str, Any] | None = None,
) -> str:
    charts = visualizations.get("charts", {})
    entity_chart = charts.get("entity_distribution", {})
    relation_chart = charts.get("relation_distribution", {})
    record_chart = charts.get("record_trend", {})
    network_chart = charts.get("disease_network", {})
    confidence = statistics.get("confidence", {})
    top_profiles = associations.get("top_associations", [])[:5]
    peak_record = trends.get("peak_record") or {}
    top_hubs = (centrality or {}).get("top_hubs", [])[:8]
    if not top_hubs:
        top_hubs = _fallback_top_hubs(network_chart)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Task 3 Analysis Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5f6b7a;
      --line: #d9e2ec;
      --panel: #ffffff;
      --page: #f5f7fb;
      --accent: #246bfe;
      --accent-soft: #dfe8ff;
      --good: #0f8f70;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    .analysis-dashboard {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 20px;
    }}
    h1 {{ margin: 0; font-size: 28px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h3 {{ margin: 16px 0 8px; font-size: 15px; }}
    .subtitle {{ margin: 6px 0 0; color: var(--muted); }}
    .status-pill {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--good);
      font-weight: 700;
      white-space: nowrap;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .kpi, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .kpi {{ padding: 14px; min-height: 84px; }}
    .kpi-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .kpi-value {{ margin-top: 6px; font-size: 24px; font-weight: 800; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{ padding: 16px; overflow: hidden; }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(96px, 160px) 1fr minmax(36px, auto);
      gap: 10px;
      align-items: center;
      margin: 8px 0;
      font-size: 13px;
    }}
    .bar-label {{ overflow-wrap: anywhere; }}
    .bar-track {{ height: 12px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--accent); border-radius: 999px; }}
    .trend {{
      height: 190px;
      width: 100%;
      border: 1px solid var(--line);
      background: linear-gradient(#fff, #f8fbff);
      border-radius: 6px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    code {{ background: #eef2f7; border-radius: 4px; padding: 2px 4px; }}
    .network-list {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .network-item {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfdff;
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      header {{ display: block; }}
      .status-pill {{ display: inline-block; margin-top: 12px; }}
      .kpis, .grid, .network-list {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="analysis-dashboard">
    <header>
      <div>
        <h1>Task 3 Graph Analysis Dashboard</h1>
        <p class="subtitle">Static reviewer dashboard generated from graph statistics, NL2SQL results, and visualization specs.</p>
      </div>
      <div class="status-pill">Ready for review</div>
    </header>
    <section class="kpis">
      {_kpi("Entities", sum(statistics.get("entity_type_counts", {}).values()))}
      {_kpi("Relations", sum(statistics.get("relation_type_counts", {}).values()))}
      {_kpi("Avg confidence", confidence.get("average", 0))}
      {_kpi("NL2SQL rows", len(nl2sql.get("rows", [])))}
    </section>
    <section class="grid">
      <article class="panel">
        <h2>{_text(entity_chart.get("title", "Entity type distribution"))}</h2>
        {_bar_rows(entity_chart.get("data", []))}
      </article>
      <article class="panel">
        <h2>{_text(relation_chart.get("title", "Relation type distribution"))}</h2>
        {_bar_rows(relation_chart.get("data", []))}
      </article>
      <article class="panel">
        <h2>{_text(record_chart.get("title", "Record sequence trend"))}</h2>
        {_trend_svg(record_chart.get("data", []))}
        <p class="subtitle">Peak record: {_text(peak_record.get("record_id", "n/a"))} / {_text(peak_record.get("edge_count", 0))} edges</p>
      </article>
      <article class="panel">
        <h2>Top Hub 节点 (度中心性)</h2>
        {_hub_rows(top_hubs)}
      </article>
      <article class="panel">
        <h2>Disease association highlights</h2>
        {_profile_table(top_profiles)}
      </article>
      <article class="panel">
        <h2>NL2SQL evidence</h2>
        <p><code>{_text(nl2sql.get("sql", ""))}</code></p>
        {_sql_table(nl2sql.get("rows", []))}
      </article>
      <article class="panel">
        <h2>{_text(network_chart.get("title", "Disease-centered relation network"))}</h2>
        {_network_summary(network_chart)}
      </article>
    </section>
  </main>
</body>
</html>
"""


def _kpi(label: str, value: Any) -> str:
    return (
        '<article class="kpi">'
        f'<div class="kpi-label">{_text(label)}</div>'
        f'<div class="kpi-value">{_text(value)}</div>'
        "</article>"
    )


def _bar_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="subtitle">No chart data.</p>'
    max_value = max(float(row.get("value", 0) or 0) for row in rows) or 1.0
    fragments = []
    for row in rows:
        value = float(row.get("value", 0) or 0)
        width = max(2.0, min(100.0, value / max_value * 100.0))
        fragments.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{_text(row.get("category", ""))}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width: {width:.1f}%"></span></span>'
            f'<strong>{_text(row.get("value", 0))}</strong>'
            "</div>"
        )
    return "\n".join(fragments)


def _trend_svg(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="subtitle">暂无数据：Record sequence trend</p>'
    edge_values = [float(row.get("edge_count", 0) or 0) for row in rows]
    entity_values = [float(row.get("entity_count", 0) or 0) for row in rows]
    width = 680
    height = 220
    pad = 28
    max_value = max(edge_values + entity_values) or 1.0
    x_step = (width - pad * 2) / max(1, len(edge_values) - 1)

    def _points(values: list[float]) -> list[str]:
        pts = []
        for index, value in enumerate(values):
            x = pad + x_step * index
            y = height - pad - (value / max_value) * (height - pad * 2)
            pts.append(f"{x:.1f},{y:.1f}")
        return pts

    edge_points = _points(edge_values)
    entity_points = _points(entity_values)

    edge_markers = "\n".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="4" fill="#246bfe" />'
        for p in edge_points
    )
    entity_markers = "\n".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="4" fill="#0f8f70" />'
        for p in entity_points
    )
    labels = "\n".join(
        f'<text x="{pad + x_step * i:.0f}" y="{height - 6}" text-anchor="middle" font-size="10" fill="#5f6b7a">{_text(row.get("record_id", f"R{i+1}"))}</text>'
        for i, row in enumerate(rows)
    )
    legend = (
        f'<g font-size="11">'
        f'<rect x="{width - 180}" y="8" width="12" height="12" fill="#246bfe" /><text x="{width - 162}" y="18" fill="#172033">边数</text>'
        f'<rect x="{width - 100}" y="8" width="12" height="12" fill="#0f8f70" /><text x="{width - 82}" y="18" fill="#172033">实体数</text>'
        f'</g>'
    )
    return (
        f'<svg class="trend" viewBox="0 0 {width} {height}" role="img" aria-label="Record trend (edge and entity counts)">'
        f'<polyline points="{" ".join(edge_points)}" fill="none" stroke="#246bfe" stroke-width="3" />'
        f'{edge_markers}'
        f'<polyline points="{" ".join(entity_points)}" fill="none" stroke="#0f8f70" stroke-width="3" stroke-dasharray="5 4" />'
        f'{entity_markers}'
        f'{labels}'
        f'{legend}'
        f'</svg>'
    )


def _hub_rows(hubs: list[dict[str, Any]]) -> str:
    if not hubs:
        return '<p class="subtitle">暂无数据：Top Hub 节点</p>'
    return _bar_rows([
        {"category": hub.get("name", hub.get("id", "")), "value": hub.get("degree", 0)}
        for hub in hubs
    ])


def _fallback_top_hubs(network_chart: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Derive a hub ranking from the disease network when centrality is absent."""

    edges = network_chart.get("edges", []) or []
    nodes = network_chart.get("nodes", []) or []
    if not edges and not nodes:
        return []
    degree: dict[str, int] = {}
    for edge in edges:
        for endpoint in (edge.get("source", ""), edge.get("target", "")):
            if endpoint:
                degree[endpoint] = degree.get(endpoint, 0) + 1
    if not degree:
        for node in nodes:
            node_id = node.get("id") or node.get("label", "")
            if node_id:
                degree[node_id] = 0
    ranked = sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"name": node_id, "degree": count} for node_id, count in ranked]


def _profile_table(profiles: list[dict[str, Any]]) -> str:
    if not profiles:
        return '<p class="subtitle">No association profiles.</p>'
    rows = []
    for profile in profiles:
        rows.append(
            "<tr>"
            f"<td>{_text(profile.get('disease', ''))}</td>"
            f"<td>{len(profile.get('symptoms', []))}</td>"
            f"<td>{len(profile.get('drugs', []))}</td>"
            f"<td>{len(profile.get('examinations', []))}</td>"
            f"<td>{len(profile.get('complications', []))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Disease</th><th>Symptoms</th><th>Drugs</th>"
        "<th>Exams</th><th>Complications</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _sql_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="subtitle">No SQL result rows.</p>'
    columns = list(rows[0].keys())
    head = "".join(f"<th>{_text(column)}</th>" for column in columns)
    body = []
    for row in rows[:8]:
        body.append("<tr>" + "".join(f"<td>{_text(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _network_summary(chart: dict[str, Any]) -> str:
    nodes = chart.get("nodes", [])[:8]
    edges = chart.get("edges", [])
    items = "\n".join(
        f'<div class="network-item"><strong>{_text(node.get("label", ""))}</strong><br>{_text(node.get("type", ""))}</div>'
        for node in nodes
    )
    return (
        f'<p class="subtitle">{len(chart.get("nodes", []))} nodes / {len(edges)} edges in rendered network spec.</p>'
        f'<div class="network-list">{items}</div>'
    )


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)
