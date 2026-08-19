"""ECharts-enhanced interactive dashboard exporter for task 3.

Generates a rich interactive HTML dashboard using Apache ECharts.
Inline SVG charts render immediately and are progressively enhanced after the
pinned CDN asset loads.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def export_echarts_dashboard(
    target_dir: str | Path,
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
    nl2sql: dict[str, Any],
    visualizations: dict[str, Any],
    centrality: dict[str, Any] | None = None,
    api_base: str = "",
    graph_file: str | None = None,
) -> dict[str, str]:
    """Export an ECharts-powered interactive HTML dashboard with SVG fallback."""

    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "task3_interactive_dashboard.html"
    dashboard_path.write_text(
        _build_echarts_html(
            statistics,
            associations,
            trends,
            nl2sql,
            visualizations,
            centrality,
            api_base=api_base,
            graph_file=graph_file,
        ),
        encoding="utf-8",
    )
    return {"status": "completed", "dashboard_path": str(dashboard_path)}


def _build_echarts_html(
    statistics: dict[str, Any],
    associations: dict[str, Any],
    trends: dict[str, Any],
    nl2sql: dict[str, Any],
    visualizations: dict[str, Any],
    centrality: dict[str, Any] | None,
    api_base: str = "",
    graph_file: str | None = None,
) -> str:
    charts = visualizations.get("charts", {})
    entity_data = charts.get("entity_distribution", {}).get("data", [])
    relation_data = charts.get("relation_distribution", {}).get("data", [])
    trend_data = charts.get("record_trend", {}).get("data", [])
    confidence = statistics.get("confidence", {})
    top_profiles = associations.get("top_associations", [])[:8]
    top_hubs = (centrality or {}).get("top_hubs", [])[:10]

    # Prepare ECharts option JSON
    entity_option = _json_for_script({
        "title": {"text": "实体类型分布", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "item"},
        "series": [{
            "type": "pie", "radius": ["35%", "65%"], "center": ["50%", "55%"],
            "data": [{"name": d.get("category", ""), "value": d.get("value", 0)} for d in entity_data],
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"}},
            "label": {"formatter": "{b}: {c} ({d}%)"},
        }],
    })

    relation_option = _json_for_script({
        "title": {"text": "关系类型分布", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "animation": False,
        "grid": {"left": 48, "right": 24, "top": 64, "bottom": 48, "containLabel": True},
        "xAxis": {"type": "category", "data": [d.get("category", "") for d in relation_data], "axisLabel": {"rotate": 20}},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "data": [d.get("value", 0) for d in relation_data],
                     "itemStyle": {"color": "#5470c6"}, "barWidth": "60%"}],
    })

    trend_rows = trends.get("record_trends") or []
    if not trend_rows and trend_data:
        trend_rows = trend_data
    trend_names = [d.get("record_id", f"R{i+1}") for i, d in enumerate(trend_rows)]
    trend_counts = [d.get("edge_count", 0) for d in trend_rows]
    trend_entity_counts = [d.get("entity_count", 0) for d in trend_rows]
    trend_option = _json_for_script({
        "title": {"text": "记录序列趋势", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "animation": False,
        "legend": {"data": ["边数", "实体数"], "top": 30},
        "grid": {"left": 48, "right": 24, "top": 64, "bottom": 32, "containLabel": True},
        "xAxis": {"type": "category", "data": trend_names},
        "yAxis": {"type": "value"},
        "series": [
            {"name": "边数", "type": "line", "data": trend_counts, "smooth": True, "itemStyle": {"color": "#5470c6"}},
            {"name": "实体数", "type": "line", "data": trend_entity_counts, "smooth": True, "itemStyle": {"color": "#91cc75"}},
        ],
    })

    # KG force-directed graph
    network_chart = charts.get("disease_network", {})
    net_nodes = network_chart.get("nodes", [])[:40]
    net_edges = network_chart.get("edges", [])[:60]

    # Hub nodes bar chart
    if not top_hubs:
        top_hubs = _fallback_top_hubs_from_network(net_nodes, net_edges)
    hub_names = [h.get("name", "") for h in top_hubs[:8]]
    hub_degrees = [h.get("degree", 0) for h in top_hubs[:8]]
    hub_option = _json_for_script({
        "title": {"text": "Top Hub 节点 (度中心性)", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "animation": False,
        "grid": {"left": 120, "right": 24, "top": 56, "bottom": 24, "containLabel": True},
        "xAxis": {"type": "value"},
        "yAxis": {"type": "category", "data": list(reversed(hub_names))},
        "series": [{
            "type": "bar",
            "data": list(reversed(hub_degrees)),
            "itemStyle": {"color": "#ee6666"},
            "barWidth": "50%",
        }],
    })

    _TYPE_COLORS = {"Disease": "#ee6666", "Symptom": "#5470c6", "Drug": "#91cc75", "Examination": "#fac858", "Treatment": "#73c0de"}
    kg_graph_option = _json_for_script({
        "title": {"text": "知识图谱结构可视化", "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {
            "trigger": "item",
            "renderMode": "richText",
            "formatter": _KG_TOOLTIP_FORMAT,
        },
        "legend": {"data": ["Disease", "Symptom", "Drug", "Examination", "Treatment"], "top": 30, "textStyle": {"fontSize": 11}},
        "animationDuration": 1500,
        "animationEasingUpdate": "quinticInOut",
        "series": [{
            "type": "graph",
            "layout": "force",
            "data": [
                {
                    "name": n.get("label", n.get("id", "")),
                    "category": ["Disease", "Symptom", "Drug", "Examination", "Treatment"].index(n.get("type", "Disease")) if n.get("type") in _TYPE_COLORS else 0,
                    "symbolSize": 8 if n.get("type") == "Disease" else 5,
                    "itemStyle": {"color": _TYPE_COLORS.get(n.get("type", ""), "#999")},
                    "label": {"show": n.get("type") == "Disease", "fontSize": 10, "color": "#333"},
                }
                for n in net_nodes
            ],
            "links": [
                {
                    "source": e.get("source", ""),
                    "target": e.get("target", ""),
                    "name": e.get("relation", ""),
                    "value": e.get("relation", ""),
                }
                for e in net_edges
            ],
            "categories": [
                {"name": "Disease"}, {"name": "Symptom"}, {"name": "Drug"},
                {"name": "Examination"}, {"name": "Treatment"},
            ],
            "roam": True,
            "draggable": True,
            "force": {"repulsion": 200, "gravity": 0.1, "edgeLength": [50, 120]},
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 4}},
            "edgeLabel": {"show": False},
            "lineStyle": {"opacity": 0.6, "width": 1.5, "curveness": 0.1},
        }],
    })

    # --- Inline SVG fallbacks for offline use ---
    svg_entity = _svg_pie(entity_data, "实体类型分布")
    svg_relation = _svg_bars(relation_data, "关系类型分布")
    svg_trend = _svg_trend(trend_rows, "记录序列趋势")
    svg_hubs = _svg_bars(
        [{"category": h.get("name", ""), "value": h.get("degree", 0)} for h in top_hubs[:8]],
        "Top Hub 节点",
    )

    # Profile table HTML
    profile_rows = ""
    for p in top_profiles:
        profile_rows += (
            f"<tr>"
            f"<td>{_e(p.get('disease', ''))}</td>"
            f"<td>{len(p.get('symptoms', []))}</td>"
            f"<td>{', '.join(_e(s) for s in p.get('symptoms', [])[:5])}</td>"
            f"<td>{len(p.get('drugs', []))}</td>"
            f"<td>{', '.join(_e(d) for d in p.get('drugs', [])[:5])}</td>"
            f"<td>{len(p.get('complications', []))}</td>"
            f"</tr>"
        )

    # SQL table
    sql_rows_html = ""
    for row in nl2sql.get("rows", [])[:10]:
        cols = "".join(f"<td>{_e(row.get(c, ''))}</td>" for c in row.keys())
        sql_rows_html += f"<tr>{cols}</tr>"
    sql_headers = "".join(f"<th>{_e(c)}</th>" for c in (nl2sql.get("rows", [{}])[0].keys() if nl2sql.get("rows") else []))

    # KPI values
    total_entities = sum(statistics.get("entity_type_counts", {}).values())
    total_relations = sum(statistics.get("relation_type_counts", {}).values())
    avg_conf = f"{confidence.get('average', 0):.2f}"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Task 3 Interactive Analysis Dashboard</title>
<style>
  :root {{ --ink: #172033; --muted: #5f6b7a; --line: #e2e8f0; --panel: #fff; --page: #f0f4f8; --accent: #3b82f6; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--page); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; }}
  .dashboard {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px 40px; }}
  header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
  h1 {{ font-size: 24px; font-weight: 700; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .badge {{ background: #dcfce7; color: #166534; padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
  .kpi {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; text-align: center; }}
  .kpi-value {{ font-size: 28px; font-weight: 800; color: var(--accent); }}
  .kpi-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 16px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
  .panel-full {{ grid-column: 1 / -1; }}
  .chart {{ width: 100%; height: 320px; display: none; }}
  .fallback {{ display: block; width: 100%; text-align: center; }}
  .fallback svg {{ max-width: 100%; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
  .nl2sql-info {{ margin-bottom: 12px; padding: 8px 12px; background: #f8fafc; border-radius: 6px; font-size: 13px; }}
  .nl2sql-live {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
  .nl2sql-live input {{ flex: 1 1 320px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; font-size: 14px; }}
  .nl2sql-live button {{ padding: 10px 16px; background: var(--accent); color: #fff; border: 0; border-radius: 6px; cursor: pointer; font-weight: 600; }}
  .nl2sql-live-status {{ font-size: 12px; color: var(--muted); margin-bottom: 8px; min-height: 18px; }}
  @media (max-width: 860px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="dashboard">
  <header>
    <div>
      <h1>Medical KG Analysis Dashboard</h1>
      <p class="subtitle">Interactive analysis powered by Apache ECharts | Task 3</p>
    </div>
    <span class="badge">Analysis Complete</span>
  </header>

  <section class="kpis">
    <div class="kpi"><div class="kpi-value">{total_entities}</div><div class="kpi-label">Entities</div></div>
    <div class="kpi"><div class="kpi-value">{total_relations}</div><div class="kpi-label">Relations</div></div>
    <div class="kpi"><div class="kpi-value">{avg_conf}</div><div class="kpi-label">Avg Confidence</div></div>
    <div class="kpi"><div class="kpi-value">{len(nl2sql.get('rows', []))}</div><div class="kpi-label">NL2SQL Rows</div></div>
  </section>

  <section class="grid">
    <div class="panel">
      <div id="entity-chart" class="chart"></div>
      <div id="entity-fallback" class="fallback">{svg_entity}</div>
    </div>
    <div class="panel">
      <div id="relation-chart" class="chart"></div>
      <div id="relation-fallback" class="fallback">{svg_relation}</div>
    </div>
    <div class="panel">
      <div id="trend-chart" class="chart"></div>
      <div id="trend-fallback" class="fallback">{svg_trend}</div>
    </div>
    <div class="panel">
      <div id="hub-chart" class="chart"></div>
      <div id="hub-fallback" class="fallback">{svg_hubs}</div>
    </div>
  </section>

  <section class="grid">
    <div class="panel panel-full">
      <div id="kg-chart" class="chart" style="height:420px;"></div>
      <div id="kg-fallback" class="fallback">
        <p style="padding:20px;color:#888;">KG force-directed graph requires ECharts. See disease association table below.</p>
      </div>
    </div>
  </section>

  <section class="grid">
    <div class="panel panel-full">
      <h3 style="margin-bottom:12px;">Disease Association Profiles</h3>
      <table>
        <thead><tr><th>Disease</th><th>Symptoms</th><th>Top Symptoms</th><th>Drugs</th><th>Top Drugs</th><th>Complications</th></tr></thead>
        <tbody>{profile_rows}</tbody>
      </table>
    </div>
  </section>

  <section class="grid">
    <div class="panel panel-full">
      <h3 style="margin-bottom:12px;">NL2SQL Evidence</h3>
      <div class="nl2sql-live">
        <input id="nl2sql-question" type="text" placeholder="输入分析问题，例如：哪些疾病关联最多症状？" value="{_e(nl2sql.get('question', '哪些疾病关联最多症状？'))}" />
        <button id="nl2sql-submit" type="button">实时查询</button>
      </div>
      <div id="nl2sql-live-status" class="nl2sql-live-status">连接 Task 3 API 后可在此提交 NL2SQL 查询。</div>
      <div class="nl2sql-info">
        <strong>Intent:</strong> <span id="nl2sql-intent">{_e(nl2sql.get('intent', 'N/A'))}</span> &nbsp;|&nbsp;
        <strong>Translator:</strong> <span id="nl2sql-translator">{_e(nl2sql.get('translator', 'N/A'))}</span> &nbsp;|&nbsp;
        <strong>Rows:</strong> <span id="nl2sql-row-count">{len(nl2sql.get('rows', []))}</span>
      </div>
      <p><code id="nl2sql-sql">{_e(nl2sql.get('sql', ''))}</code></p>
      <table><thead><tr id="nl2sql-head">{sql_headers}</tr></thead><tbody id="nl2sql-body">{sql_rows_html}</tbody></table>
    </div>
  </section>
</div>

<script>
function setDashboardMode(interactive) {{
  var ids = ['entity','relation','trend','hub','kg'];
  for (var i = 0; i < ids.length; i++) {{
    var chart = document.getElementById(ids[i] + '-chart');
    var fallback = document.getElementById(ids[i] + '-fallback');
    if (chart) chart.style.display = interactive ? 'block' : 'none';
    if (fallback) fallback.style.display = interactive ? 'none' : 'block';
  }}
}}

function hideFallbackPanels() {{
  var ids = ['entity','relation','trend','hub','kg'];
  for (var i = 0; i < ids.length; i++) {{
    var fallback = document.getElementById(ids[i] + '-fallback');
    if (fallback) fallback.style.display = 'none';
  }}
}}

function showChartPanels() {{
  var ids = ['entity','relation','trend','hub','kg'];
  for (var i = 0; i < ids.length; i++) {{
    var chart = document.getElementById(ids[i] + '-chart');
    if (chart) chart.style.display = 'block';
  }}
}}

function initializeECharts() {{
  try {{
    if (typeof echarts === 'undefined') throw new Error('ECharts not loaded');
    showChartPanels();
    window.requestAnimationFrame(function() {{
      window.requestAnimationFrame(function() {{
        var c1 = echarts.init(document.getElementById('entity-chart'));
        c1.setOption({entity_option});
        var c2 = echarts.init(document.getElementById('relation-chart'));
        c2.setOption({relation_option});
        var c3 = echarts.init(document.getElementById('trend-chart'));
        c3.setOption({trend_option});
        var c4 = echarts.init(document.getElementById('hub-chart'));
        c4.setOption({hub_option});
        var c5 = echarts.init(document.getElementById('kg-chart'));
        c5.setOption({kg_graph_option});
        var instances = [c1, c2, c3, c4, c5];
        instances.forEach(function(chart) {{ chart.resize(); }});
        hideFallbackPanels();
        window.addEventListener('resize', function() {{
          instances.forEach(function(chart) {{ chart.resize(); }});
        }});
      }});
    }});
  }} catch(e) {{
    setDashboardMode(false);
  }}
}}

setDashboardMode(false);
var echartsScript = document.createElement('script');
echartsScript.src = 'https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js';
echartsScript.async = true;
echartsScript.onload = initializeECharts;
echartsScript.onerror = function() {{ setDashboardMode(false); }};
document.head.appendChild(echartsScript);

var NL2SQL_API_BASE = {_json_for_script(api_base or "")};
var NL2SQL_GRAPH_FILE = {_json_for_script(graph_file or "")};

function renderNl2SqlTable(rows) {{
  var body = document.getElementById('nl2sql-body');
  var head = document.getElementById('nl2sql-head');
  if (!body || !head) return;
  if (!rows || !rows.length) {{
    head.innerHTML = '';
    body.innerHTML = '<tr><td>No rows</td></tr>';
    return;
  }}
  var columns = Object.keys(rows[0]);
  head.innerHTML = columns.map(function(c) {{ return '<th>' + c + '</th>'; }}).join('');
  body.innerHTML = rows.slice(0, 20).map(function(row) {{
    return '<tr>' + columns.map(function(c) {{ return '<td>' + String(row[c] ?? '') + '</td>'; }}).join('') + '</tr>';
  }}).join('');
  var count = document.getElementById('nl2sql-row-count');
  if (count) count.textContent = String(rows.length);
}}

async function submitNl2SqlQuery() {{
  var status = document.getElementById('nl2sql-live-status');
  var questionInput = document.getElementById('nl2sql-question');
  if (!questionInput) return;
  var question = questionInput.value.trim();
  if (!question) {{
    if (status) status.textContent = '请输入问题。';
    return;
  }}
  if (status) status.textContent = '查询中...';
  try {{
    var payload = {{ question: question }};
    if (NL2SQL_GRAPH_FILE) payload.graph_file = NL2SQL_GRAPH_FILE;
    var response = await fetch((NL2SQL_API_BASE || '') + '/api/nl2sql', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload),
    }});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    var data = await response.json();
    var intent = document.getElementById('nl2sql-intent');
    var translator = document.getElementById('nl2sql-translator');
    var sql = document.getElementById('nl2sql-sql');
    if (intent) intent.textContent = data.intent || 'N/A';
    if (translator) translator.textContent = data.translator || 'N/A';
    if (sql) sql.textContent = data.sql || '';
    renderNl2SqlTable(data.rows || []);
    if (status) status.textContent = '查询完成。';
  }} catch (err) {{
    if (status) status.textContent = '实时查询失败：' + err.message + '（请启动 Task 3 API 服务）';
  }}
}}

var nl2sqlButton = document.getElementById('nl2sql-submit');
if (nl2sqlButton) nl2sqlButton.addEventListener('click', submitNl2SqlQuery);
var nl2sqlInput = document.getElementById('nl2sql-question');
if (nl2sqlInput) nl2sqlInput.addEventListener('keydown', function(evt) {{
  if (evt.key === 'Enter') submitNl2SqlQuery();
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Inline SVG fallback chart generators
# ---------------------------------------------------------------------------

_PALETTE = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272", "#fc8452", "#9a60b4"]

_KG_TOOLTIP_FORMAT = "{b}"


def _fallback_top_hubs_from_network(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Derive hub ranking from the disease network when centrality is absent."""

    if not nodes and not edges:
        return []
    degree: dict[str, int] = {}
    for edge in edges:
        for endpoint in (edge.get("source", ""), edge.get("target", "")):
            if endpoint:
                degree[endpoint] = degree.get(endpoint, 0) + 1
    ranked = sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"name": node_id, "degree": count} for node_id, count in ranked]


def _svg_pie(data: list[dict], title: str) -> str:
    """Generate a simple SVG pie chart."""
    if not data:
        return f'<p style="padding:20px;color:#888;">暂无数据：{_e(title)}</p>'
    total = sum(d.get("value", 0) for d in data)
    if total == 0:
        return '<p style="padding:20px;color:#888;">数值均为 0</p>'
    cx, cy, r = 130, 140, 95
    paths = []
    legend = []
    start = 0.0
    for i, d in enumerate(data):
        frac = d.get("value", 0) / total
        if frac <= 0:
            continue
        angle = frac * 360
        x1 = cx + r * _cos(start)
        y1 = cy + r * _sin(start)
        x2 = cx + r * _cos(start + angle)
        y2 = cy + r * _sin(start + angle)
        large = 1 if angle > 180 else 0
        color = _PALETTE[i % len(_PALETTE)]
        paths.append(
            f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{r},{r} 0 {large},1 {x2:.1f},{y2:.1f} Z" fill="{color}" opacity="0.85"/>'
        )
        pct = f"{frac*100:.0f}%"
        legend_y = 48 + len(legend) * 24
        legend.append(
            f'<rect x="260" y="{legend_y - 11}" width="12" height="12" '
            f'rx="2" fill="{color}"/>'
            f'<text x="280" y="{legend_y}" font-size="12" fill="#172033">'
            f'{_e(d.get("category", ""))} ({pct})</text>'
        )
        start += angle
    return (
        f'<svg viewBox="0 0 500 280" style="max-width:100%;" role="img" aria-label="{_e(title)}">'
        f'<text x="250" y="16" text-anchor="middle" font-size="13" fill="#172033">{_e(title)}</text>'
        f'{"".join(paths)}'
        f'{"".join(legend)}'
        f'</svg>'
    )


def _svg_bars(data: list[dict], title: str) -> str:
    """Generate a simple SVG bar chart."""
    if not data:
        return f'<p style="padding:20px;color:#888;">暂无数据：{_e(title)}</p>'
    max_val = max(d.get("value", 0) for d in data) or 1
    w, h, pad, bar_h = 500, 30 * len(data) + 40, 100, 22
    bars = []
    for i, d in enumerate(data):
        y = 30 + i * (bar_h + 8)
        bw = max(2, d.get("value", 0) / max_val * (w - pad - 20))
        color = _PALETTE[i % len(_PALETTE)]
        bars.append(
            f'<text x="{pad-4}" y="{y+bar_h-4}" text-anchor="end" font-size="11" fill="#172033">{_e(d.get("category",""))}</text>'
            f'<rect x="{pad}" y="{y}" width="{bw:.0f}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>'
            f'<text x="{pad+bw+4:.0f}" y="{y+bar_h-4}" font-size="11" fill="#5f6b7a">{d.get("value",0)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" style="max-width:100%;" role="img" aria-label="{_e(title)}">'
        f'<text x="{w//2}" y="16" text-anchor="middle" font-size="13" fill="#172033">{_e(title)}</text>'
        f'{"".join(bars)}'
        f'</svg>'
    )


def _svg_trend(data: list[dict], title: str) -> str:
    """Generate a simple SVG line chart for record trends."""
    if not data:
        return f'<p style="padding:20px;color:#888;">暂无数据：{_e(title)}</p>'
    edge_values = [d.get("edge_count", 0) for d in data]
    entity_values = [d.get("entity_count", 0) for d in data]
    w, h, pad = 500, 220, 30
    max_val = max(edge_values + entity_values) or 1
    x_step = (w - pad * 2) / max(1, len(edge_values) - 1)

    def _points(values: list[int]) -> list[str]:
        pts = []
        for i, v in enumerate(values):
            x = pad + x_step * i
            y = h - pad - (v / max_val) * (h - pad * 2)
            pts.append(f"{x:.1f},{y:.1f}")
        return pts

    edge_points = _points(edge_values)
    entity_points = _points(entity_values)
    edge_markers = "".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="4" fill="#5470c6" />'
        for p in edge_points
    )
    entity_markers = "".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="4" fill="#91cc75" />'
        for p in entity_points
    )
    labels = "".join(
        f'<text x="{pad + x_step * i:.0f}" y="{h - 8}" text-anchor="middle" font-size="10" fill="#5f6b7a">{_e(data[i].get("record_id", f"R{i+1}"))}</text>'
        for i in range(len(edge_values))
    )
    legend = (
        f'<rect x="{w - 170}" y="8" width="12" height="12" fill="#5470c6" />'
        f'<text x="{w - 152}" y="18" font-size="11" fill="#172033">边数</text>'
        f'<rect x="{w - 90}" y="8" width="12" height="12" fill="#91cc75" />'
        f'<text x="{w - 72}" y="18" font-size="11" fill="#172033">实体数</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" style="max-width:100%;" role="img" aria-label="{_e(title)}">'
        f'<text x="{w//2}" y="16" text-anchor="middle" font-size="13" fill="#172033">{_e(title)}</text>'
        f'{legend}'
        f'<polyline points="{" ".join(edge_points)}" fill="none" stroke="#5470c6" stroke-width="2.5" />'
        f'{edge_markers}'
        f'<polyline points="{" ".join(entity_points)}" fill="none" stroke="#91cc75" stroke-width="2.5" stroke-dasharray="5 4" />'
        f'{entity_markers}'
        f'{labels}'
        f'</svg>'
    )


def _cos(deg: float) -> float:
    import math
    return math.cos(math.radians(deg - 90))


def _sin(deg: float) -> float:
    import math
    return math.sin(math.radians(deg - 90))


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_for_script(value: Any) -> str:
    """Serialize JSON safely for inline script contexts."""

    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
