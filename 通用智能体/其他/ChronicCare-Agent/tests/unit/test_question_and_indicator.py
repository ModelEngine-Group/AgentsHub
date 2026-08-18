from analysis.indicator_utils import SAFETY_NOTE, build_indicator_item, build_insight
from analysis.question_parser import parse_question_entry, parse_questions


def test_parse_supported_question() -> None:
    parsed, errors = parse_question_entry(
        {
            "id": "Q1",
            "question": "患者数",
            "intent": "count_patients_by_disease_recent",
            "expected_chart_type": "metric_card",
        }
    )
    assert parsed["status"] == "success"
    assert errors == []


def test_parse_unknown_intent_fails_without_sql_template() -> None:
    parsed, errors = parse_question_entry({"id": "Q2", "question": "未知", "intent": "not_supported"})
    assert parsed["status"] == "failed"
    assert errors == ["Unsupported intent: not_supported"]


def test_sql_template_allows_custom_question() -> None:
    parsed, errors = parse_question_entry(
        {"id": "Q3", "question": "自定义", "intent": "custom", "sql_template": "SELECT 1"}
    )
    assert parsed["status"] == "success"
    assert errors == []


def test_parse_questions_collects_errors() -> None:
    parsed, errors = parse_questions(
        [
            {"id": "Q1", "question": "患者数", "intent": "count_patients_by_disease_recent"},
            {"id": "Q2", "question": "未知", "intent": "not_supported"},
        ]
    )
    assert len(parsed) == 2
    assert len(errors) == 1


def test_metric_card_indicator_uses_first_column() -> None:
    item = build_indicator_item(
        {
            "id": "Q1",
            "question": "患者数",
            "intent": "metric",
            "status": "success",
            "expected_chart_type": "metric_card",
            "columns": ["patient_count"],
            "rows": [{"patient_count": 2000}],
        }
    )
    assert item["metric"] == {"name": "patient_count", "value": 2000, "unit": "项"}
    assert SAFETY_NOTE in item["insight"]


def test_line_indicator_assigns_axes() -> None:
    item = build_indicator_item(
        {
            "id": "Q2",
            "question": "趋势",
            "intent": "trend",
            "status": "success",
            "expected_chart_type": "line",
            "columns": ["month", "count"],
            "rows": [{"month": "2026-07", "count": 10}],
        }
    )
    assert item["x_field"] == "month"
    assert item["y_field"] == "count"


def test_empty_metric_has_safe_explanation() -> None:
    assert "未查询到有效结果" in build_insight("患者数", "metric_card", [])
