from __future__ import annotations

import pytest

from dki.analysis.nl2sql_compile_result import CompileDisposition
from dki.analysis.nl2sql_four_table import (
    compile_explicit_four_table_result,
    compile_grounded_four_table_result,
)


@pytest.mark.parametrize(
    ("question", "fragments"),
    [
        (
            "输出地区为「南京」且性别为「女」的患者编号、姓名和年龄。",
            ("FROM patients", "region='南京'", "gender='女'"),
        ),
        (
            "连接患者与就诊信息，针对年龄至少30岁且科室「骨科」的记录，按地区求平均费用。",
            ("JOIN visits", "p.age >= 30", "AVG(v.cost)"),
        ),
        (
            "用公共表表达式先筛出住院天数至少5天的就诊，再按主要检查汇总费用。",
            ("WITH inpatient_visits", "los_days >= 5", "SUM(cost)"),
        ),
        (
            "返回性别「女」但地区不是「北京」的患者编号。",
            ("EXCEPT", "gender='女'", "region='北京'"),
        ),
        (
            "在处方药物「示例药」且费用低于500.25元的记录中，按费用升序编号并输出就诊编号、费用和行号。",
            ("ROW_NUMBER() OVER", "drug_prescribed='示例药'", "cost < 500.25"),
        ),
    ],
)
def test_explicit_four_table_compiler_uses_structural_slots(
    question: str,
    fragments: tuple[str, ...],
) -> None:
    result = compile_explicit_four_table_result(question)
    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert all(fragment in result.sql for fragment in fragments)
    assert result.fallback_fenced
    assert result.contract_self_validated


def test_explicit_four_table_owned_incomplete_surface_fails_closed() -> None:
    result = compile_explicit_four_table_result(
        "连接患者与就诊信息，列出某些可能相关的记录。"
    )
    assert result.disposition is CompileDisposition.BLOCKED
    assert result.sql is None


def test_unrelated_question_remains_not_applicable() -> None:
    result = compile_explicit_four_table_result("患者总数是多少？")
    assert result.disposition is CompileDisposition.NOT_APPLICABLE
    assert result.sql is None


@pytest.mark.parametrize(
    "question",
    ["按科室统计就诊量", "按性别统计就诊量"],
)
def test_compact_historical_group_surface_is_not_stolen(question: str) -> None:
    result = compile_explicit_four_table_result(question)
    assert result.disposition is CompileDisposition.NOT_APPLICABLE
    assert result.sql is None


def test_cross_row_average_comparison_is_left_to_exclusive_owner() -> None:
    result = compile_explicit_four_table_result(
        "找出南京地区年龄至少35岁的女性患者，年龄高于全部患者平均年龄，"
        "返回患者编号、姓名、年龄和地区。"
    )
    assert result.disposition is CompileDisposition.NOT_APPLICABLE
    assert result.sql is None


@pytest.mark.parametrize(
    "question",
    [
        "查询科室为呼吸科且费用至少800元的就诊，返回就诊编号和费用。",
        "返回头实体类型为Disease且中文关系为相关疾病的头实体。",
        "返回每个科室中费用等于该科室最高费用的就诊编号和费用。",
        "按患者汇总费用，列出累计费用最高的前5位患者姓名。",
        "在科室呼吸科中按患者累计费用排名，返回前5名及排名。",
        "呈现症状关系的三元组置信度至少0.8，返回头实体。",
    ],
)
def test_established_semantic_owner_surfaces_are_not_stolen(question: str) -> None:
    result = compile_explicit_four_table_result(question)
    assert result.disposition is CompileDisposition.NOT_APPLICABLE
    assert result.sql is None


@pytest.mark.parametrize(
    ("question", "fragments"),
    [
        (
            "对科室为「儿科」的就诊，按主诊断统计就诊次数。",
            ("department='儿科'", "GROUP BY disease", "COUNT(*)"),
        ),
        (
            "主要检查为「核磁共振」时，返回就诊次数至少2次的科室。",
            ("exam='核磁共振'", "HAVING COUNT(*) >= 2", "GROUP BY department"),
        ),
        (
            "输出头实体类型为「Disease」、置信度不低于0.8的中文关系、尾实体和置信度。",
            ("head_type='Disease'", "confidence >= 0.8", "relation_zh"),
        ),
        (
            "按置信度从高到低取前3条三元组，输出头实体、中文关系、尾实体和置信度。",
            ("FROM kg_triples", "ORDER BY confidence DESC", "LIMIT 3"),
        ),
    ],
)
def test_schema_driven_generalisation_covers_unseen_structural_combinations(
    question: str,
    fragments: tuple[str, ...],
) -> None:
    result = compile_explicit_four_table_result(question)
    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert all(fragment in result.sql for fragment in fragments)


def test_two_relation_confidence_filter_preserves_every_declared_clause() -> None:
    question = (
        "筛选关系中文名属于临床表现或诊断检查且置信度不高于0.8的记录，"
        "返回head、relation_zh、tail、confidence，并按第一列降序排列。"
    )
    result = compile_explicit_four_table_result(question)
    assert result.disposition is CompileDisposition.SQL
    assert result.sql == (
        "SELECT head, relation_zh, tail, confidence FROM kg_triples "
        "WHERE relation_zh IN ('临床表现','诊断检查') AND confidence <= 0.8 "
        "ORDER BY 1 DESC"
    )


def test_live_domain_binding_accepts_natural_unquoted_values() -> None:
    result = compile_grounded_four_table_result(
        "查询科室为呼吸科且费用至少800元的就诊，返回就诊编号和费用。",
        {
            "department": ("呼吸科", "骨科"),
        },
    )
    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert "department='呼吸科'" in result.sql
    assert "cost >= 800" in result.sql
    assert "SELECT visit_id, cost" in result.sql


def test_unknown_unquoted_assignment_is_not_partially_rendered() -> None:
    result = compile_grounded_four_table_result(
        "查询科室为虚构科且费用至少800元的就诊，返回就诊编号和费用。",
        {
            "department": ("呼吸科", "骨科"),
        },
    )
    assert result.disposition is CompileDisposition.NOT_APPLICABLE
    assert result.sql is None


def test_all_natural_assignments_must_survive_rendering() -> None:
    result = compile_grounded_four_table_result(
        "连接患者与就诊信息，地区为南京且科室为呼吸科，返回姓名和费用。",
        {
            "region": ("南京",),
            "department": ("呼吸科",),
        },
    )
    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert "p.region='南京'" in result.sql
    assert "v.department='呼吸科'" in result.sql


@pytest.mark.parametrize(
    ("question", "fragments"),
    [
        (
            "查询费用介于300到900之间的就诊，返回就诊编号和费用。",
            ("cost BETWEEN 300 AND 900", "SELECT visit_id, cost"),
        ),
        (
            "查询2024年3月内的就诊，返回就诊编号和就诊日期。",
            ("visit_date LIKE '2024-03%'", "SELECT visit_id, visit_date"),
        ),
    ],
)
def test_generic_filters_compose_ranges_and_calendar_periods(
    question: str,
    fragments: tuple[str, ...],
) -> None:
    result = compile_explicit_four_table_result(question, {})
    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert all(fragment in result.sql for fragment in fragments)
