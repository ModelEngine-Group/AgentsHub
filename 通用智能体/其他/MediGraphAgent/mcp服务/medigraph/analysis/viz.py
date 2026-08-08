"""BI + graph visualization: render analysis results to a self-contained HTML.

Uses a vendored ECharts build to draw bar / line / pie charts for tabular analysis
results, and a relationship network for graph-driven association analysis. Each
chart is paired with an LLM-generated natural-language insight (图文结合).

The ECharts runtime is inlined rather than pulled from a CDN so every generated
report opens correctly offline; the CDN URL is kept only as a fallback for the
case where the vendored asset is missing.
"""
from __future__ import annotations

import html as html_lib
import json
from functools import lru_cache
from pathlib import Path

_ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
# Vendored copy (Apache-2.0, see THIRD_PARTY_NOTICES.md) so generated reports are
# self-contained: a reviewer opening the HTML offline -- or behind a network where
# jsdelivr is unreachable -- still sees the charts instead of a blank panel.
_ECHARTS_LOCAL = Path(__file__).resolve().parent / "assets" / "echarts.min.js"


@lru_cache(maxsize=1)
def _echarts_script() -> str:
    """Inline <script> for ECharts, falling back to the CDN if the asset is absent."""
    try:
        source = _ECHARTS_LOCAL.read_text(encoding="utf-8")
    except OSError:
        return f'<script src="{_ECHARTS_CDN}"></script>'
    # ECharts is minified JS with no literal "</script>", but guard anyway so a
    # future asset swap cannot break out of the tag.
    source = source.replace("</script>", "<\\/script>")
    return f"<script>{source}</script>"


def pick_chart_type(question: str, columns: list[str], rows: list) -> str:
    """Heuristic chart selection by question intent + result shape."""
    q = question.lower()
    if not rows or len(columns) < 2:
        return "table"
    if any(k in q for k in ("热力", "矩阵", "交叉")) and len(columns) >= 3:
        return "heatmap"
    if any(k in q for k in ("散点", "相关性", "相关关系")):
        return "scatter"
    if any(k in question for k in ("趋势", "随时间", "每月", "每年", "变化", "trend")) or "month" in " ".join(columns).lower():
        return "line"
    if any(k in question for k in ("比例", "占比", "分布", "构成", "ratio", "proportion")):
        return "pie"
    return "bar"


def _bar_or_line_option(title: str, columns: list[str], rows: list, kind: str) -> dict:
    cats = [str(r[0]) for r in rows]
    vals = [r[1] for r in rows]
    # Long Chinese disease names at the 12px ECharts default are unreadable once
    # the page is captured and rescaled, so set type sizes explicitly.
    return {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 18}},
        "tooltip": {"trigger": "axis"},
        "grid": {"bottom": 130 if len(cats) > 6 else 70, "left": 70, "right": 40, "top": 70},
        "xAxis": {"type": "category", "data": cats, "name": columns[0],
                   "nameTextStyle": {"fontSize": 14},
                   "axisLabel": {"rotate": 30 if len(cats) > 6 else 0, "fontSize": 14,
                                  "interval": 0, "hideOverlap": False}},
        "yAxis": {"type": "value", "name": columns[1],
                   "nameTextStyle": {"fontSize": 14},
                   "axisLabel": {"fontSize": 14}},
        "series": [{"type": kind, "data": vals, "smooth": kind == "line",
                     "label": {"show": kind == "bar", "position": "top", "fontSize": 14}}],
    }


def _pie_option(title: str, columns: list[str], rows: list) -> dict:
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0},
        "series": [{"type": "pie", "radius": "60%",
                     "data": [{"name": str(r[0]), "value": r[1]} for r in rows]}],
    }


def _scatter_option(title: str, columns: list[str], rows: list) -> dict:
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "name": columns[0]},
        "yAxis": {"type": "value", "name": columns[1]},
        "series": [
            {
                "type": "scatter",
                "data": [[row[0], row[1]] for row in rows],
                "symbolSize": 12,
            }
        ],
    }


def _heatmap_option(title: str, columns: list[str], rows: list) -> dict:
    x_values = list(dict.fromkeys(str(row[0]) for row in rows))
    y_values = list(dict.fromkeys(str(row[1]) for row in rows))
    data = [
        [x_values.index(str(row[0])), y_values.index(str(row[1])), row[2]]
        for row in rows
    ]
    numeric = [float(row[2]) for row in rows if isinstance(row[2], (int, float))]
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"position": "top"},
        "grid": {"top": 70, "bottom": 80},
        "xAxis": {"type": "category", "data": x_values, "name": columns[0], "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": y_values, "name": columns[1], "splitArea": {"show": True}},
        "visualMap": {
            "min": min(numeric) if numeric else 0,
            "max": max(numeric) if numeric else 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 10,
        },
        "series": [{"type": "heatmap", "data": data, "label": {"show": True}}],
    }


def _graph_option(title: str, triples: list[dict]) -> dict:
    degree: dict[str, int] = {}
    for t in triples:
        for name in (t["head"], t["tail"]):
            degree[name] = degree.get(name, 0) + 1
    nodes, seen = [], set()
    for t in triples:
        for name in (t["head"], t["tail"]):
            if name not in seen:
                seen.add(name)
                # Hubs read as hubs: size grows with degree instead of every node
                # being an identical dot.
                nodes.append({"name": name, "symbolSize": min(30 + 6 * degree[name], 72),
                              "draggable": True})
    links = [{"source": t["head"], "target": t["tail"],
              "value": t.get("relation_zh", t.get("relation", ""))} for t in triples]
    # Scale type and spacing to the graph size: ECharts' 12px default is
    # unreadable at the zoom level the page opens at.
    n = len(nodes)
    node_font = 20 if n <= 20 else (17 if n <= 45 else 14)
    edge_font = 15 if n <= 20 else (13 if n <= 45 else 11)
    repulsion = 900 if n <= 20 else (600 if n <= 45 else 380)
    edge_length = 190 if n <= 20 else (150 if n <= 45 else 110)
    return {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 18}},
        "tooltip": {},
        "series": [{
            "type": "graph", "layout": "force", "roam": True,
            "label": {
                "show": True, "position": "right", "fontSize": node_font,
                "color": "#1b2a2e", "textBorderColor": "#ffffff", "textBorderWidth": 3,
            },
            "edgeSymbol": ["none", "arrow"], "edgeSymbolSize": 9,
            "lineStyle": {"color": "#8fa6ad", "width": 1.6, "curveness": 0.08, "opacity": 0.9},
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 3}},
            "force": {"repulsion": repulsion, "edgeLength": edge_length, "gravity": 0.08},
            "data": nodes,
            "links": [{
                "source": l["source"], "target": l["target"],
                "label": {
                    "show": True, "formatter": l["value"], "fontSize": edge_font,
                    "color": "#4a5f66", "textBorderColor": "#ffffff", "textBorderWidth": 3,
                },
            } for l in links],
        }],
    }


def render_report(
    path: str | Path,
    question: str,
    chart_type: str,
    columns: list[str],
    rows: list,
    insight: str,
    sql: str = "",
    triples: list[dict] | None = None,
) -> str:
    """Write a self-contained HTML report (chart + table + insight)."""
    chart_rows = rows
    chart_note = ""
    if chart_type in {"bar", "pie"} and len(rows) > 30 and len(columns) >= 2:
        numeric_rows = [r for r in rows if len(r) >= 2 and isinstance(r[1], (int, float))]
        if len(numeric_rows) == len(rows):
            chart_rows = sorted(numeric_rows, key=lambda r: r[1], reverse=True)[:30]
            chart_note = "（图表展示数值最高的 Top 30；下表保留完整结果的前 200 行）"
    if chart_type == "graph" and triples:
        option = _graph_option(question, triples)
    elif chart_type == "heatmap" and len(columns) >= 3:
        option = _heatmap_option(question, columns, chart_rows)
    elif chart_type == "scatter":
        option = _scatter_option(question, columns, chart_rows)
    elif chart_type == "pie":
        option = _pie_option(question, columns, chart_rows)
    elif chart_type in ("bar", "line"):
        option = _bar_or_line_option(question, columns, chart_rows, chart_type)
    else:
        option = None

    # data table
    head = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html_lib.escape(str(c))}</td>" for c in r) + "</tr>"
        for r in rows[:200]
    )
    table_html = f"<table border=1 cellpadding=6 style='border-collapse:collapse'><tr>{head}</tr>{body}</table>" if columns else ""

    chart_div = ""
    chart_js = ""
    if option is not None:
        chart_div = "<div id='chart' style='width:100%;height:480px;'></div>"
        option_json = (
            json.dumps(option, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        chart_js = (
            f"var ch=echarts.init(document.getElementById('chart'));ch.setOption({option_json});"
            "ch.on('click',function(p){var d=document.getElementById('drilldown');"
            "d.textContent='钻取：'+p.name+' / '+JSON.stringify(p.value);});"
            "window.addEventListener('resize',function(){ch.resize();});"
        )

    sql_html = f"<pre style='background:#f5f5f5;padding:10px'>{html_lib.escape(sql)}</pre>" if sql else ""
    safe_question = html_lib.escape(question)
    safe_insight = html_lib.escape(insight).replace("\n", "<br>")
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>分析报告 - {safe_question}</title>
{_echarts_script()}
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px}}
h2{{color:#2c3e50}} .insight{{background:#eef6ff;border-left:4px solid #3498db;padding:12px 16px;margin:16px 0;line-height:1.7}}
.badge{{display:inline-block;background:#e8f5e9;color:#256029;border-radius:12px;padding:4px 10px;margin-right:8px}}</style>
</head><body>
<h2>📊 数据分析报告</h2>
<p><b>问题：</b>{safe_question}</p>
<p><span class="badge">图表：{html_lib.escape(chart_type)}</span><span class="badge">结果行：{len(rows)}</span> {html_lib.escape(chart_note)}</p>
{chart_div}
<div id="drilldown" style="padding:8px 0;color:#555">点击图表元素可查看钻取值</div>
<div class="insight"><b>💡 洞察：</b><br>{safe_insight}</div>
<h3>数据明细</h3>{table_html}
<h3>执行的查询</h3>{sql_html}
<script>{chart_js}</script>
</body></html>"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)
