"""Synthetic regressions for the explicit analytical v23 compiler."""

import pytest

from dki.analysis.nl2sql_independent_v23 import compile_independent_v23_sql
from dki.analysis.nl2sql_protocol import validate_gold_sql


DOMAINS = {
    "region": ["测试东区", "测试西区"],
    "department": ["测试科室"],
    "disease": ["甲疾病", "乙疾病"],
    "drug": ["甲药物", "乙药物"],
    "exam": ["甲检查"],
    "kg_entity": ["甲症状"],
    "kg_entity_type": ["Disease", "Drug", "Exam", "Symptom", "Department"],
    "kg_relation": ["HAS_SYMPTOM", "TREATED_BY_DRUG", "DIAGNOSED_BY"],
}


@pytest.mark.parametrize(
    ("family", "question", "required"),
    [
        (
            "set",
            "将患者姓名和图谱实体名称合并去重，统一输出名称；明确使用UNION。",
            " UNION ",
        ),
        (
            "window",
            "在每个科室内部按费用降序计算DENSE_RANK，返回科室、就诊编号、费用和名次。",
            "DENSE_RANK()",
        ),
        (
            "cte",
            "先在CTE中汇总每位患者的就诊次数和总费用，再返回就诊至少3次的患者编号、次数与总费用。",
            "visit_count>=3",
        ),
        (
            "nested",
            "列出每位患者费用最高的就诊，费用并列时全部保留。",
            "MAX(w.cost)",
        ),
        (
            "having",
            "按患者编号分组，只保留至少3次就诊，返回总费用并筛选这些患者。",
            "HAVING COUNT(*)>=3",
        ),
        (
            "top_k",
            "按累计费用取前7位有就诊患者，返回患者编号、总费用和就诊次数；数量7。",
            "LIMIT 7",
        ),
        (
            "filtering",
            "筛选关系标识HAS_SYMPTOM且置信度至少0.85，返回头实体、尾实体、置信度和观察次数。",
            "confidence>=0.85",
        ),
        (
            "filtering_domains",
            "列出主要处方药属于甲药物或乙药物的就诊编号、疾病和日期。",
            "drug_prescribed IN",
        ),
        (
            "filtering_complete_qualifiers",
            "查询地区为测试东区、性别为女且年龄不低于22岁的患者，返回患者编号、姓名和年龄。",
            "region='测试东区' AND gender='女' AND age>=22",
        ),
        (
            "aggregation",
            "以年月和科室分组，计算就诊次数与平均费用。",
            "GROUP BY visit_month,department",
        ),
        (
            "temporal",
            "按具体日期分组，只返回同日就诊数至少4次的日期和不同科室数。",
            "HAVING COUNT(*)>=4",
        ),
        (
            "join",
            "连接患者和就诊，对每位患者汇总最高单次费用和就诊数。",
            "MAX(v.cost)",
        ),
        (
            "kg",
            "查询与甲疾病共享至少一个诊断检查的其他疾病；关系标识DIAGNOSED_BY。",
            "other.tail=base.tail",
        ),
        (
            "relation_prefix",
            "头实体为甲疾病且关系标识属于TREATED_BY_DRUG或DIAGNOSED_BY，返回尾实体、尾类型、关系和置信度。",
            "relation IN ('TREATED_BY_DRUG','DIAGNOSED_BY')",
        ),
        (
            "kg_name_contains",
            "提取类型为Exam且实体名包含检测的图谱实体，返回名称和提及次数。",
            "name LIKE '%检测%'",
        ),
        (
            "kg_tail_grounding",
            "找出尾实体为甲症状且尾类型为Symptom的头疾病、关系标识、置信度与观察次数。",
            "tail='甲症状'",
        ),
        (
            "join_exact_exam",
            "同时匹配就诊疾病和主要检查与图谱DIAGNOSED_BY关系，列出一致的就诊编号、疾病、检查和置信度。",
            "t.tail=v.exam",
        ),
        (
            "having_month",
            "按年月统计就诊，只返回至少3次就诊的月份，输出年月、次数与平均费用。",
            "HAVING COUNT(*)>=3",
        ),
    ],
)
def test_v23_complete_shapes_emit_safe_sql(family, question, required):
    sql = compile_independent_v23_sql(question, DOMAINS)
    assert sql is not None, family
    assert required in sql
    validate_gold_sql(sql)


@pytest.mark.parametrize(
    "question",
    [
        "删除全部患者。",
        "分析四张表并给出结论。",
        "按科室分组，只保留一些结果。",
        "查询关系标识UNKNOWN_REL的边。",
        "按年龄挑选若干患者。",
    ],
)
def test_v23_compiler_abstains_when_shape_is_incomplete(question):
    assert compile_independent_v23_sql(question, DOMAINS) is None
