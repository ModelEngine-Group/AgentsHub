from __future__ import annotations

import pytest

from orchestration.question_classifier import classify_question


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("现在 ChronicCare 支持哪些 CPU 算子？", "datamate_pipelines"),
        ("执行 DataMate 算子链路", "datamate_pipeline_run"),
        ("DataMate 流程状态如何？", "datamate_pipeline_status"),
        ("系统状态如何？", "system_status"),
        ("查看完整分析报告入口", "report_summary"),
        ("支持哪些分析问题？", "capability_examples"),
        ("某个患者未来有哪些随访计划？", "kg_patient_path_query"),
        ("高盐饮食和血压异常有什么关系？", "kg_relation_query"),
        ("高血压关联哪些检查指标？", "kg_entity_query"),
        ("未来 N 天需要随访的高风险患者有多少？", "future_n_days_high_risk_followup"),
        ("未来 7 天高风险随访患者的疾病类型分布", "cohort_disease_distribution"),
        ("高风险患者中哪些疾病最多？", "cohort_disease_distribution"),
        ("未来 7 天需要随访的高风险患者有多少？", "future_n_days_high_risk_followup"),
        ("未来 7 天随访患者有多少？", "future_n_days_followup"),
        ("高风险患者有多少？", "risk_level_distribution"),
        ("不同风险等级人数分布", "risk_level_distribution"),
        ("不同疾病组合的人数分布是多少？", "disease_combination_distribution"),
        ("高血压患者有多少？", "disease_distribution"),
        ("当前常见病有哪些？", "disease_distribution"),
        ("生成高血压知识图谱子图", "kg_subgraph_render"),
        ("最近 6 个月 HbA1c 异常比例趋势", "indicator_analysis"),
        ("系统并发能力如何？", "performance_query"),
        ("请自由分析这个问题", "open_sql_analysis"),
    ],
)
def test_classifier_extended_routes(query: str, expected: str) -> None:
    assert classify_question({"query": query})["intent"] == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("NPU 支持哪些算子？", "npu_supported_operators"),
        ("启用 NPU 执行全流程", "datamate_pipeline_run_npu"),
        ("测试 NPU 性能加速效果", "npu_operator_benchmark"),
        ("NPU 环境是否就绪？", "npu_readiness_query"),
    ],
)
def test_classifier_npu_routes(query: str, expected: str) -> None:
    assert classify_question({"query": query})["intent"] == expected


def test_classifier_pronoun_context_and_normalized_entities() -> None:
    result = classify_question(
        {
            "query": "未来 7 天他们这些高风险患者需要随访的人数？",
            "last_context": {"cohort_label": "上一轮高风险患者"},
        }
    )
    assert result["intent"] == "cohort_disease_distribution"
    assert result["normalized_entities"]["cohort"] == "上一轮高风险患者"

    result = classify_question(
        {
            "query": "这些患者主要有哪些疾病？",
            "last_context": {"name": "糖尿病队列"},
        }
    )
    assert result["intent"] == "cohort_disease_distribution"
    assert result["normalized_entities"]["cohort"] == "糖尿病队列"
