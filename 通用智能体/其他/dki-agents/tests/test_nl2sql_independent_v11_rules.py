"""Synthetic, non-evaluation regressions for the v11 bounded compiler."""

import pytest

from dki.analysis.nl2sql_independent_v11 import compile_independent_v11_sql
from dki.analysis.nl2sql_protocol import validate_gold_sql


DOMAINS = {
    "region": ["测试地区"],
    "department": ["测试科室"],
    "disease": ["测试疾病"],
    "drug": ["测试药物"],
    "exam": ["测试检查"],
    "kg_entity_type": ["Disease", "Drug"],
    "kg_relation": ["HAS_SYMPTOM", "TREATED_BY", "DIAGNOSED_BY"],
    "kg_relation_zh": ["症状"],
}


@pytest.mark.parametrize(
    ("family", "question", "required"),
    [
        (
            "filtering",
            "列出费用高于999.50且住院至少2天的就诊，并限定患者年龄低于70岁。"
            "结果列依次为就诊编号、患者编号、地区、科室、费用、住院天数。",
            "v.cost>999.5",
        ),
        (
            "aggregation",
            "汇总2025年费用不低于500的就诊，按地区统计就诊次数、患者数、"
            "总费用和平均住院天数。",
            "GROUP BY p.region",
        ),
        (
            "top_k",
            "返回平均单次费用最高的5个科室，但只考虑至少4次就诊的科室。"
            "输出平均住院天数。",
            "LIMIT 5",
        ),
        (
            "join",
            "连接图谱边与头、尾实体登记，筛选置信度至少0.7且两端提及次数均不少于2。"
            "输出两端实体、两端提及次数。",
            "JOIN kg_entities et",
        ),
        (
            "kg",
            "列出知识图谱中关系为HAS_SYMPTOM且置信度至少0.6的边。"
            "返回头实体、头类型、尾实体、尾类型、置信度和次数。",
            "relation='HAS_SYMPTOM'",
        ),
        (
            "temporal",
            "逐名患者列出相邻两次就诊间隔，排除第一次记录并仅保留间隔超过30天。"
            "输出患者编号、本次日期、上次日期。",
            "LAG(visit_date)",
        ),
        (
            "having",
            "在2025年费用至少400的记录中找出就诊次数不少于3的科室。"
            "输出患者数、总费用和平均住院天数。",
            "HAVING COUNT(*)>=3",
        ),
        (
            "nested",
            "找出提及次数至少2且高于同类型平均水平，并存在置信度至少0.8出边的实体。"
            "返回类型均值和最高出边置信度。",
            "SELECT AVG(e3.mention_count)",
        ),
        (
            "cte",
            "用分步CTE比较各科室2025年费用至少600的记录与全期同阈值费用。"
            "输出当年占比。",
            "WITH all_time AS",
        ),
        (
            "set",
            "求年龄至少20岁、在2025年和2024年都有就诊的患者交集。"
            "返回患者编号、姓名和地区。",
            " INTERSECT ",
        ),
        (
            "window",
            "按实体类型对图谱实体的提及次数做稠密排名，并同时给出类型内实体数"
            "与类型内提及总量。仅返回排名前4的实体。",
            "DENSE_RANK()",
        ),
    ],
)
def test_v11_operation_family_is_safe(family, question, required):
    sql = compile_independent_v11_sql(question, DOMAINS)
    assert sql is not None, family
    assert required in sql
    validate_gold_sql(sql)


@pytest.mark.parametrize(
    "question",
    [
        "删除所有患者数据。",
        "请随便分析一下。",
        "列出知识图谱中关系为UNKNOWN_REL且置信度至少0.6的边。"
        "返回头实体、头类型、尾实体、尾类型、置信度和次数。",
        "返回平均单次费用最高的五个科室。",
    ],
)
def test_v11_abstains_on_unsafe_incomplete_or_ungrounded_requests(question):
    assert compile_independent_v11_sql(question, DOMAINS) is None
