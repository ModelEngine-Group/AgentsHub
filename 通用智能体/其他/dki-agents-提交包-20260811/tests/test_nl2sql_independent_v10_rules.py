"""Synthetic regressions for the independent-v10 compositional compiler."""

import pytest

from dki.analysis.nl2sql_independent_v10 import compile_independent_v10_sql
from dki.analysis.nl2sql_protocol import validate_gold_sql


DOMAINS = {
    "region": ["测试城"],
    "department": ["测试科"],
    "disease": ["测试病"],
    "drug": ["测试药"],
    "exam": ["测试检查"],
    "kg_entity_type": ["Disease", "Drug"],
    "kg_relation": ["TEST_REL"],
    "kg_relation_zh": ["测试关系"],
}


@pytest.mark.parametrize(
    "question",
    [
        "查询测试病在2025-01-01到2025-02-01间不早于首日且不晚于末日的记录。",
        "按患者统计去过的不同科室数和疾病数以及最长住院天数。",
        "列出患者人数最多的前2个地区。",
        "关联患者地区与就诊科室，统计测试城在各科室的就诊次数和平均住院天数。",
        "按年-周统计就诊次数与平均住院天数。",
        "查看英文关系为TEST_REL且头类型为Disease的边，列出两端、置信度和证据次数。",
        "按地区统计患者数和平均年龄，只保留至少1位患者的地区。",
        "列出年龄高于总体平均年龄且确实有就诊记录的患者。",
        "先形成实际疾病与图谱边的公共集合，再按疾病汇总关系数和尾实体数。",
        "合并实际疾病与图谱头实体名称，按集合语义去重。",
        "在每个科室内计算就诊费用的百分位排名。",
    ],
)
def test_v10_operation_families_emit_safe_sql(question):
    sql = compile_independent_v10_sql(question, DOMAINS)
    assert sql is not None
    validate_gold_sql(sql)


def test_v10_row_count_wrapper_is_orthogonal_to_join_intent():
    question = (
        "关联患者地区与就诊科室，统计测试城在各科室的就诊次数和平均住院天数。"
        "此外，在每一行附上当前结果集的总行数。"
    )
    sql = compile_independent_v10_sql(question, DOMAINS)
    assert sql is not None
    assert sql.startswith("SELECT scoped.*,COUNT(*) OVER() AS result_row_count FROM (")
    assert "WHERE p.region='测试城'" in sql


def test_v10_cte_row_count_wrapper_preserves_base_ctes():
    question = (
        "先用两个公共表表达式汇总并筛出就诊至少2次的患者，再连接姓名。"
        "此外，在每一行附上当前结果集的总行数。"
    )
    sql = compile_independent_v10_sql(question, DOMAINS)
    assert sql is not None
    assert sql.startswith("SELECT scoped.*,COUNT(*) OVER() AS result_row_count FROM (WITH patient_stats")
    validate_gold_sql(sql)


@pytest.mark.parametrize(
    "question",
    [
        "删除患者表。",
        "随便查询一些医疗信息。",
        "查看英文关系为未知关系的边，列出两端、置信度和证据次数。",
        "统计测试城的数据。",
    ],
)
def test_v10_declines_unsafe_ambiguous_or_ungrounded_requests(question):
    assert compile_independent_v10_sql(question, DOMAINS) is None
