from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Dict, List

from analysis.open_sql.alias_registry import DISEASE_LABELS
from runtime_common.analysis_context import AnalysisContext
from runtime_common.common import resolve_path
from tool_server.utils import load_server_config, public_artifact_url, service_artifact_url

CHART_DIR = "outputs/charts/open_sql"
SQLITE_DB = "data/sqlite/chroniccare.db"


def _label(key: str) -> str:
    labels = {
        "patient_count": "患者人数",
        "plan_count": "计划数量",
        "lab_count": "检验记录数",
        "denominator": "分母",
        "numerator": "分子",
        "abnormal_rate": "异常率",
        "control_rate": "达标率",
        "avg_value": "平均值",
        "avg_bmi": "平均 BMI",
        "risk_level": "风险等级",
        "month": "月份",
        "drug_category": "药物类别",
        "disease_tags": "疾病标签",
        "abnormal_count": "异常记录数",
        "tested_patient_count": "检测人数",
        "abnormal_patient_count": "异常人数",
    }
    return labels.get(key, key)


def _markdown_table(rows: List[Dict[str, Any]], limit: int = 20) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    rendered_headers = [_label(header) for header in headers]
    lines = [
        "| " + " | ".join(rendered_headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _percent(value: Any) -> str | None:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return None


def _write_chart(question: str, query_spec: Dict[str, Any], rows: List[Dict[str, Any]]) -> str | None:
    if not rows or query_spec.get("aggregation") not in {"trend", "distribution"}:
        return None
    cfg = load_server_config()
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()[:12]
    out_dir = resolve_path(CHART_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"open_sql_{digest}.html"
    json_rows = json.dumps(rows, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Open SQL Chart</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ font-size: 20px; }}
    table {{ border-collapse: collapse; min-width: 520px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
    th {{ background: #f8fafc; }}
    .bar {{ height: 14px; background: #2563eb; display: inline-block; vertical-align: middle; }}
  </style>
</head>
<body>
  <h1>{question}</h1>
  <div id="chart"></div>
  <script>
    const rows = {json_rows};
    const keys = rows.length ? Object.keys(rows[0]) : [];
    const valueKey = keys.find(k => /count|rate|avg|value|数量|人数/.test(k)) || keys[keys.length - 1];
    const labelKey = keys.find(k => k !== valueKey) || keys[0];
    const maxValue = Math.max(...rows.map(r => Number(r[valueKey]) || 0), 1);
    const table = document.createElement('table');
    table.innerHTML = '<thead><tr>' + keys.map(k => `<th>${{k}}</th>`).join('') + '<th>可视化</th></tr></thead>';
    const tbody = document.createElement('tbody');
    rows.forEach(row => {{
      const tr = document.createElement('tr');
      keys.forEach(k => {{
        const td = document.createElement('td');
        td.textContent = row[k];
        tr.appendChild(td);
      }});
      const td = document.createElement('td');
      const bar = document.createElement('span');
      bar.className = 'bar';
      bar.style.width = `${{Math.max(4, (Number(row[valueKey]) || 0) / maxValue * 180)}}px`;
      td.appendChild(bar);
      tr.appendChild(td);
      tbody.appendChild(tr);
    }});
    table.appendChild(tbody);
    document.getElementById('chart').appendChild(table);
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return public_artifact_url(cfg, f"/artifacts/charts/open_sql/{html_path.name}")


def _future_days_window(time_range: Dict[str, Any] | None) -> tuple[int, int] | None:
    if not time_range or time_range.get("type") != "future_days":
        return None
    days = max(1, int(time_range.get("value") or 1))
    return days, days - 1


def _disease_label(diseases: List[str]) -> str:
    if not diseases:
        return "全部患者"
    return "、".join(DISEASE_LABELS.get(str(item), str(item)) for item in diseases)


def _followup_trend_rows(query_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    window = _future_days_window(query_spec.get("time_range"))
    if not window:
        return []
    days, offset_days = window
    diseases = [str(item).lower() for item in (query_spec.get("diseases") or [])]
    risk_level = str(query_spec.get("risk_level") or "").lower()
    context_payload = query_spec.get("analysis_context") or {}
    as_of_date = str(context_payload.get("as_of_date") or AnalysisContext.current().as_of_date)
    where = ["f.status IN ('pending','scheduled')"]
    params: List[Any] = []
    for disease in diseases:
        where.append("lower(p.disease_tags) LIKE ?")
        params.append(f"%{disease}%")
    if risk_level:
        where.append("lower(f.priority) = ?")
        params.append(risk_level)
    sql = (
        "SELECT f.followup_date AS followup_date, COUNT(DISTINCT f.patient_id) AS patient_count "
        "FROM followup_plan f JOIN patient_profile p ON f.patient_id = p.patient_id "
        "WHERE "
        + " AND ".join(where)
        + " AND date(f.followup_date) BETWEEN date(?) AND date(?, '+' || ? || ' day') "
        "GROUP BY f.followup_date ORDER BY f.followup_date"
    )
    params = params + [as_of_date, as_of_date, offset_days]
    with sqlite3.connect(resolve_path(SQLITE_DB)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        bounds = conn.execute(
            "SELECT date(?) AS start_date, date(?, '+' || ? || ' day') AS end_date",
            [as_of_date, as_of_date, offset_days],
        ).fetchone()
    by_date = {str(row["followup_date"]): int(row["patient_count"] or 0) for row in rows}
    start_date = str(bounds["start_date"])
    end_date = str(bounds["end_date"])
    with sqlite3.connect(":memory:") as mem:
        date_rows = mem.execute(
            """
            WITH RECURSIVE dates(day) AS (
              SELECT date(?)
              UNION ALL
              SELECT date(day, '+1 day') FROM dates WHERE day < date(?)
            )
            SELECT day FROM dates
            """,
            [start_date, end_date],
        ).fetchall()
    return [{"followup_date": row[0], "patient_count": by_date.get(row[0], 0)} for row in date_rows[:days]]


def _line_svg(title: str, rows: List[Dict[str, Any]]) -> str:
    width, height = 960, 520
    margin_left, margin_right, margin_top, margin_bottom = 72, 42, 70, 84
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    values = [float(row.get("patient_count", 0) or 0) for row in rows]
    max_value = max(values + [1.0])
    points = []
    for index, value in enumerate(values):
        x = margin_left + (plot_w * index / max(1, len(values) - 1))
        y = margin_top + plot_h - (plot_h * value / max_value)
        points.append((x, y, value))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
    label_step = 1 if len(rows) <= 14 else 5 if len(rows) <= 45 else 10
    circles = "\n".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='#2563eb' />"
        + (
            f"<text x='{x:.1f}' y='{y - 12:.1f}' text-anchor='middle' font-size='16' fill='#1f2937'>{int(value)}</text>"
            if value or index % label_step == 0 or index == len(points) - 1
            else ""
        )
        for index, (x, y, value) in enumerate(points)
    )
    labels = "\n".join(
        f"<text x='{x:.1f}' y='{height - 38}' text-anchor='middle' font-size='13' fill='#475569'>{str(rows[index].get('followup_date', ''))[5:]}</text>"
        for index, (x, _, _) in enumerate(points)
        if index % label_step == 0 or index == len(points) - 1
    )
    grid = "\n".join(
        f"<line x1='{margin_left}' y1='{margin_top + plot_h * i / 4:.1f}' x2='{width - margin_right}' y2='{margin_top + plot_h * i / 4:.1f}' stroke='#e2e8f0' />"
        for i in range(5)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="42" font-size="26" font-weight="700" fill="#111827">{title}</text>
  {grid}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#94a3b8"/>
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#94a3b8"/>
  <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
  {circles}
  {labels}
  <text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" font-size="14" fill="#64748b">日期（月-日）</text>
</svg>"""


def _write_followup_chart(question: str, query_spec: Dict[str, Any]) -> Dict[str, Any]:
    if query_spec.get("intent") != "followup_count":
        return {}
    rows = _followup_trend_rows(query_spec)
    if not rows:
        return {}
    cfg = load_server_config()
    digest = hashlib.md5((question + json.dumps(query_spec, ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()[:12]
    out_dir = resolve_path(CHART_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = (query_spec.get("time_range") or {}).get("value") or len(rows)
    diseases = query_spec.get("diseases") or []
    disease_label = _disease_label(diseases)
    title = f"未来 {days} 天{disease_label}随访人数趋势"
    svg = _line_svg(title, rows)
    svg_path = out_dir / f"followup_trend_{digest}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    route = f"/artifacts/charts/open_sql/{svg_path.name}"
    return {
        "image_url": public_artifact_url(cfg, route),
        "image_service_url": service_artifact_url(cfg, route),
        "charts": [
            {
                "name": title,
                "type": "line",
                "url": public_artifact_url(cfg, route),
                "service_url": service_artifact_url(cfg, route),
                "path": f"{CHART_DIR}/{svg_path.name}",
            }
        ],
        "trend_rows": rows,
    }


def format_result(
    *,
    question: str,
    query_spec: Dict[str, Any],
    template: Dict[str, Any],
    execution: Dict[str, Any],
    allow_chart: bool,
) -> Dict[str, Any]:
    rows = execution.get("rows") or []
    if execution.get("status") != "success":
        summary = f"SQL 执行失败：{execution.get('error')}"
        return {"summary_text": summary, "answer_markdown": summary, "chart_url": None}
    if not rows:
        summary = "当前数据未检索到符合条件的记录。"
        return {"summary_text": summary, "answer_markdown": summary, "chart_url": None}

    table = _markdown_table(rows)
    explanation = template.get("explanation") or "已完成开放式 SQL 查询。"
    summary = explanation
    first = rows[0]
    if "control_rate" in first and len(rows) == 1:
        percent = _percent(first.get("control_rate"))
        rate_text = f"{first.get('control_rate')}" + (f"（{percent}）" if percent else "")
        summary = f"{explanation}：达标率为 {rate_text}，分子 {first.get('numerator')}，分母 {first.get('denominator')}。"
    elif "abnormal_rate" in first and len(rows) == 1:
        percent = _percent(first.get("abnormal_rate"))
        rate_text = f"{first.get('abnormal_rate')}" + (f"（{percent}）" if percent else "")
        summary = f"{explanation}：异常率为 {rate_text}，分子 {first.get('numerator')}，分母 {first.get('denominator')}。"
    elif "avg_value" in first and len(rows) == 1:
        summary = f"{explanation}：平均值为 {first.get('avg_value')}，样本患者 {first.get('patient_count', 'N/A')} 人。"
    elif "avg_bmi" in first and len(rows) == 1:
        summary = f"{explanation}：平均 BMI 为 {first.get('avg_bmi')}，样本患者 {first.get('patient_count', 'N/A')} 人。"
    elif "patient_count" in first and len(rows) == 1:
        summary = f"{explanation}：患者人数 {first.get('patient_count')} 人。"

    chart_url = _write_chart(question, query_spec, rows) if allow_chart else None
    followup_chart = _write_followup_chart(question, query_spec) if allow_chart else {}
    parts = [summary, "", table]
    if followup_chart.get("image_url"):
        parts.extend(["", f"![随访趋势图]({followup_chart['image_url']})"])
    if chart_url:
        parts.extend(["", f"图表入口：{chart_url}"])
    return {
        "summary_text": summary,
        "answer_markdown": "\n".join(parts).strip(),
        "chart_url": chart_url,
        "image_url": followup_chart.get("image_url"),
        "image_service_url": followup_chart.get("image_service_url"),
        "charts": followup_chart.get("charts") or [],
        "trend_rows": followup_chart.get("trend_rows") or [],
    }
