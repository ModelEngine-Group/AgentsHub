import pytest

from orchestration.intent_router import route_intent
from orchestration.question_classifier import classify_question
from orchestration.question_parser import build_query_plan


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        ("当前数据规模是多少？", "data_summary"),
        ("当前知识图谱有多少节点和边？", "kg_summary"),
        ("现在 ChronicCare 支持哪些 NPU 算子？", "npu_supported_operators"),
        ("请启用 NPU 运行 DataMate 全流程", "datamate_pipeline_run_npu"),
        ("排除糖尿病患者后有多少高血压患者？", "unsupported_negation_query"),
    ],
)
def test_classifier_routes_key_questions(query: str, expected_intent: str) -> None:
    result = classify_question({"query": query})
    assert result["intent"] == expected_intent


def test_route_intent_maps_to_expected_tool() -> None:
    result = route_intent({"query": "当前知识图谱有多少节点和边？"})
    assert result["intent"] == "kg_summary"
    assert result["tool"] == "chroniccare_kg_summary"
    assert result["executor"] == "direct_tool"


def test_indicator_trend_selects_trend_tool() -> None:
    plan = build_query_plan(
        {"intent": "indicator_analysis", "normalized_entities": {"indicators": ["hba1c"]}},
        "最近 6 个月 HbA1c 趋势",
    )
    assert plan["tool"] == "chroniccare_trend_query"
    assert plan["requires_chart"] is True


def test_unknown_intent_falls_back_to_open_analysis() -> None:
    plan = build_query_plan({"intent": "unknown", "normalized_entities": {}}, "未知问题")
    assert plan["tool"] == "chroniccare_open_analysis_query"
    assert plan["executor"] == "legacy_open_analysis"
