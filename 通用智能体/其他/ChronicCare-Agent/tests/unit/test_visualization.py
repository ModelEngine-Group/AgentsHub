from pathlib import Path

from visualization import chart_render
from visualization.chart_templates import metric_card_html, table_html

CONFIG = {"chart_defaults": {"template": "plotly_white", "width": 800, "height": 500}}


def test_metric_card_escapes_user_content() -> None:
    html = metric_card_html("<script>", {"name": "count", "value": 3, "unit": "人"}, "insight", "safe")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_table_html_renders_values() -> None:
    html = table_html("疾病", ["name", "count"], [{"name": "高血压", "count": 433}], "说明", "安全", "Table")
    assert "高血压" in html
    assert "433" in html


def test_guess_xy_prefers_known_disease_fields() -> None:
    assert chart_render._guess_xy({"table": {"columns": ["disease_group", "abnormal_rate"]}}) == (
        "disease_group",
        "abnormal_rate",
    )


def test_bar_chart_falls_back_without_plotly(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(chart_render, "load_plotly", lambda: None)
    output = tmp_path / "bar.html"
    result = chart_render.render_indicator_chart(
        {
            "chart_type": "bar",
            "question": "疾病分布",
            "insight": "说明",
            "table": {"columns": ["disease", "count"], "rows": [{"disease": "高血压", "count": 433}]},
        },
        output,
        CONFIG,
        "安全说明",
    )
    assert result == {"plotly_available": False, "fallback_used": True}
    assert output.exists() and "高血压" in output.read_text(encoding="utf-8")


def test_metric_chart_does_not_mark_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(chart_render, "load_plotly", lambda: None)
    output = tmp_path / "metric.html"
    result = chart_render.render_indicator_chart(
        {
            "chart_type": "metric_card",
            "question": "患者数",
            "metric": {"name": "patient_count", "value": 2000, "unit": "人"},
            "insight": "说明",
            "table": {"columns": ["patient_count"], "rows": [{"patient_count": 2000}]},
        },
        output,
        CONFIG,
        "安全说明",
    )
    assert result == {"plotly_available": False, "fallback_used": False}
    assert "2000" in output.read_text(encoding="utf-8")
