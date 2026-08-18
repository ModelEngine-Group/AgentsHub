from __future__ import annotations

from html import escape
from typing import Any

from visualization.html_utils import html_page, render_table


def metric_card_html(question: str, metric: dict[str, Any], insight: str, safety_note: str) -> str:
    body = f"""
    <div class="card">
      <span class="badge">Metric Card</span>
      <h1>{escape(question)}</h1>
      <div class="muted">{escape(metric.get('name', 'value'))}</div>
      <div class="metric-value">{escape(str(metric.get('value', '-')))}</div>
      <div class="muted">单位：{escape(str(metric.get('unit', '')))}</div>
      <p>{escape(insight)}</p>
      <div class="safety">{escape(safety_note)}</div>
    </div>
    """
    return html_page(question, body)


def table_html(question: str, columns: list[str], rows: list[dict[str, Any]], insight: str, safety_note: str, label: str) -> str:
    body = f"""
    <div class="card">
      <span class="badge">{escape(label)}</span>
      <h1>{escape(question)}</h1>
      <p>{escape(insight)}</p>
      {render_table(columns, rows)}
      <div class="safety">{escape(safety_note)}</div>
    </div>
    """
    return html_page(question, body)


def quality_score_html(title: str, scores: dict[str, Any], safety_note: str) -> str:
    cards = []
    for key, value in scores.items():
        cards.append(
            f"""
            <div class="card">
              <div class="muted">{escape(key)}</div>
              <div class="metric-value" style="font-size:40px">{escape(str(value))}</div>
            </div>
            """
        )
    body = f"""
    <h1>{escape(title)}</h1>
    <div class="grid">{''.join(cards)}</div>
    <div class="safety">{escape(safety_note)}</div>
    """
    return html_page(title, body)
