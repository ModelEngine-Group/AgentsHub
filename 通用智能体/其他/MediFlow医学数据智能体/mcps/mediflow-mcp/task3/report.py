"""任务三分析报告导出。"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import zipfile
from datetime import datetime
from typing import Any


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', "_", str(value or "")).strip(" ._")
    return (cleaned or fallback)[:60]


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def _chart_html(chart: dict[str, Any] | None) -> str:
    if not chart or not chart.get("data"):
        return ""
    data = chart["data"][:20]
    label_key = chart.get("label_key")
    value_key = chart.get("value_key")
    values = [float(row.get(value_key) or 0) for row in data]
    maximum = max(values or [1.0]) or 1.0
    chart_type = str(chart.get("type") or "bar")
    colors = [
        "#2563eb", "#0f9d75", "#f59e0b", "#7c3aed", "#e11d48",
        "#0891b2", "#65a30d", "#ea580c", "#64748b",
    ]
    if chart_type in {"donut", "pie"}:
        total = sum(max(0.0, value) for value in values) or 1.0
        stops: list[str] = []
        legend: list[str] = []
        offset = 0.0
        for index, (row, value) in enumerate(zip(data, values)):
            color = colors[index % len(colors)]
            end = offset + max(0.0, value) / total * 100
            stops.append(f"{color} {offset:.2f}% {end:.2f}%")
            legend.append(
                '<div class="donut-item">'
                f'<i style="background:{color}"></i>'
                f'<span>{html.escape(str(row.get(label_key, "")))}</span>'
                f'<strong>{value / total * 100:.1f}%</strong>'
                "</div>"
            )
            offset = end
        body = (
            '<div class="donut-layout">'
            f'<div class="donut" style="background:conic-gradient({",".join(stops)})">'
            f'<span>合计<br><strong>{total:,.0f}</strong></span></div>'
            f'<div class="donut-legend">{"".join(legend)}</div></div>'
        )
    elif chart_type == "column":
        columns = []
        for row, value in zip(data[:12], values[:12]):
            height = max(2.0, value / maximum * 100)
            columns.append(
                '<div class="column-item">'
                f'<span class="column-value">{value:,.0f}</span>'
                f'<span class="column-track"><i style="height:{height:.2f}%"></i></span>'
                f'<span class="column-label">{html.escape(str(row.get(label_key, "")))}</span>'
                "</div>"
            )
        body = f'<div class="column-chart">{"".join(columns)}</div>'
    elif chart_type == "line":
        points = []
        labels = []
        subset = list(zip(data[:16], values[:16]))
        for index, (row, value) in enumerate(subset):
            x = 40 + index * (880 / max(len(subset) - 1, 1))
            y = 250 - value / maximum * 190
            points.append(f"{x:.1f},{y:.1f}")
            labels.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb">'
                f'<title>{html.escape(str(row.get(label_key, "")))}：{value:,.0f}</title></circle>'
            )
        body = (
            '<svg class="report-line" viewBox="0 0 960 290">'
            '<line x1="40" y1="250" x2="930" y2="250" stroke="#dfe6f1"/>'
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="4"/>'
            f'{"".join(labels)}</svg>'
        )
    else:
        bars = []
        for row, value in zip(data, values):
            width = max(1.0, value / maximum * 100)
            bars.append(
                '<div class="bar-row">'
                f'<span class="bar-label">{html.escape(str(row.get(label_key, "")))}</span>'
                f'<span class="bar-track"><span class="bar-fill" style="width:{width:.2f}%"></span></span>'
                f'<span class="bar-value">{html.escape(str(row.get(value_key, "")))}</span>'
                "</div>"
            )
        body = f'<div class="chart">{"".join(bars)}</div>'
    return (
        f"<h3>{html.escape(str(chart.get('title') or '统计图表'))}</h3>"
        f"<p class='muted'>{html.escape(str(chart.get('subtitle') or ''))}</p>"
        f"{body}"
    )


def _table_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>无匹配记录。</p>"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in rows[:200]
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _answer_html(markdown_text: str) -> str:
    lines = []
    for raw in str(markdown_text or "").splitlines():
        text = html.escape(raw)
        if text.startswith("### "):
            lines.append(f"<h2>{text[4:]}</h2>")
        elif text.startswith("- "):
            lines.append(f"<p class='bullet'>• {text[2:]}</p>")
        elif text:
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
            lines.append(f"<p>{text}</p>")
    return "".join(lines)


def _scope_html(scope: dict[str, Any]) -> str:
    statement = html.escape(str(scope.get("statement") or "暂无数据范围说明"))
    sources = scope.get("sources") or []
    source_items = "".join(
        "<li>"
        f"{html.escape(str(item.get('source_name') or '未命名来源'))}"
        f"（{int(item.get('record_count') or 0):,} 条）"
        "</li>"
        for item in sources
    )
    source_list = f"<ul>{source_items}</ul>" if source_items else ""
    return f"<p>{statement}</p>{source_list}"


def build_report_archive(result: dict[str, Any]) -> tuple[str, bytes]:
    """生成包含 HTML、CSV 与审计 JSON 的可移植报告压缩包。"""

    generated = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"医学数据分析报告_{generated}.zip"
    analyses = result.get("analyses", [])
    provenance = result.get("provenance") or {}
    analysis_scope = result.get("analysis_scope") or {}
    sections = []
    for item in analyses:
        sections.append(
            "<section>"
            f"<h2>{html.escape(str(item.get('title') or '分析项'))}</h2>"
            f"<p>{html.escape(str(item.get('purpose') or ''))}</p>"
            f"<details><summary>查看只读 SQL</summary><pre>{html.escape(str(item.get('sql') or ''))}</pre></details>"
            f"{_chart_html(item.get('chart'))}"
            f"{_table_html(item.get('rows') or [])}"
            "</section>"
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>医学数据分析报告</title>
<style>
body{{font-family:"Microsoft YaHei",Arial,sans-serif;color:#172033;margin:0;background:#f4f7fb}}
main{{max-width:1080px;margin:32px auto;padding:32px;background:white}}
h1,h2,h3{{margin:0 0 14px}} section{{margin-top:30px;padding-top:24px;border-top:1px solid #dfe6f1}}
.meta,.muted{{color:#68748a}} pre{{white-space:pre-wrap;background:#f5f7fb;padding:12px}}
table{{width:100%;border-collapse:collapse;margin-top:14px}} th,td{{padding:9px;border:1px solid #dfe6f1;text-align:left}}
th{{background:#f5f7fb}} .bar-row{{display:grid;grid-template-columns:180px 1fr 90px;gap:10px;align-items:center;margin:9px 0}}
.bar-track{{height:20px;background:#e9eef7}} .bar-fill{{display:block;height:100%;background:#2563eb}}
.column-chart{{display:flex;align-items:flex-end;gap:12px;height:310px;padding:20px 8px 0}}
.column-item{{display:grid;grid-template-rows:24px 220px 52px;flex:1;min-width:42px;text-align:center}}
.column-track{{display:flex;align-items:flex-end;background:#f1f5f9}} .column-track i{{display:block;width:100%;background:#2563eb}}
.column-label{{font-size:12px;padding-top:8px;overflow-wrap:anywhere}} .column-value{{font-size:12px}}
.donut-layout{{display:grid;grid-template-columns:300px 1fr;gap:34px;align-items:center}}
.donut{{width:250px;height:250px;border-radius:50%;display:grid;place-items:center}}
.donut span{{display:grid;place-items:center;width:135px;height:135px;border-radius:50%;background:white;text-align:center}}
.donut-legend{{display:grid;gap:10px}} .donut-item{{display:grid;grid-template-columns:14px 1fr 70px;gap:9px;align-items:center}}
.donut-item i{{width:14px;height:14px;border-radius:3px}} .donut-item strong{{text-align:right}}
.report-line{{width:100%;height:auto;background:#fbfdff}}
.bullet{{margin:5px 0}} @media print{{body{{background:white}}main{{margin:0;max-width:none}}details{{display:none}}}}
</style>
</head>
<body><main>
<h1>医学数据分析报告</h1>
<p class="meta">分析编号：{html.escape(str(result.get('analysis_id') or ''))}<br>
生成时间：{html.escape(str(provenance.get('generated_at') or ''))}</p>
<h2>分析问题</h2><p>{html.escape(str(result.get('question') or ''))}</p>
{_answer_html(result.get('answer') or '')}
{''.join(sections)}
<section><h2>数据口径与追溯</h2>
<p>本报告由只读 SQL 的实际返回结果生成。分析回答、证据表和图表共享同一分析编号。</p>
{_scope_html(analysis_scope)}
<pre>{html.escape(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre>
</section>
</main></body></html>"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("医学数据分析报告.html", page.encode("utf-8"))
        archive.writestr(
            "分析追溯.json",
            json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        for index, item in enumerate(analyses, start=1):
            title = _safe_name(item.get("title"), f"分析{index}")
            archive.writestr(
                f"数据/{index:02d}_{title}.csv",
                _csv_bytes(item.get("rows") or []),
            )
    return filename, buffer.getvalue()
