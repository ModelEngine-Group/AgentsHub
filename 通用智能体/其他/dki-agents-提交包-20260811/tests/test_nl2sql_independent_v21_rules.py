"""Synthetic regressions for the bounded compositional v21 compiler."""

import pytest

from dki.analysis.nl2sql_independent_v21 import compile_independent_v21_sql
from dki.analysis.nl2sql_protocol import validate_gold_sql


DOMAINS = {
    "region": ["测试地区"],
    "department": ["测试科室"],
    "disease": ["甲疾病", "乙疾病"],
    "drug": ["测试药物"],
    "exam": ["测试检查"],
    "kg_entity_type": ["Disease", "Drug", "Department"],
    "kg_relation": ["HAS_SYMPTOM", "TREATED_BY_DRUG"],
}


@pytest.mark.parametrize(
    ("family", "question", "required"),
    [
        (
            "filtering",
            "查询测试地区年龄不低于37岁的患者姓名、性别和年龄，结果按年龄降序、姓名升序排列。",
            "age>=37",
        ),
        (
            "filtering_qualifiers",
            "查询地区为测试地区、性别为女且年龄不低于21岁的患者，返回患者编号、姓名和年龄，按年龄降序。",
            "gender='女'",
        ),
        (
            "aggregation",
            "以疾病分组统计住院就诊的就诊次数和平均费用（保留两位小数），并按就诊次数降序、疾病升序排列。住院天数 los_days > 0。",
            "GROUP BY disease",
        ),
        (
            "having",
            "对全部记录按科室分组，仅保留就诊不少于9次的组，返回科室、就诊次数、平均费用，并按就诊次数降序、科室升序排列。",
            "HAVING COUNT(*)>=9",
        ),
        (
            "top_k",
            "连接患者与就诊记录，按患者汇总累计住院天数，选指标最高的前6位，相同按编号升序；返回编号、姓名和指标。",
            "LIMIT 6",
        ),
        (
            "join",
            "连接患者与就诊记录，查询测试地区且曾在测试科室就诊的姓名、年龄、就诊日期、疾病和费用，按费用降序、就诊编号升序排列。",
            "p.region='测试地区'",
        ),
        (
            "cte",
            "先汇总各科室的就诊次数、平均费用、总费用、最长住院天数和住院数，再筛选最长住院至少6天的科室并按最长住院天数降序、科室升序排列。住院天数 los_days > 0。",
            "WITH dept_stats AS",
        ),
        (
            "nested",
            "在测试科室就诊中，找出费用高于该科室平均费用的编号、疾病和费用，按费用降序、编号升序。",
            "SELECT AVG(v2.cost)",
        ),
        (
            "set",
            "找出主诊断为甲疾病或乙疾病的患者编号，使用集合并集去重并按编号升序。",
            " UNION ",
        ),
        (
            "temporal",
            "从2027-01-01起，按自然月统计测试科室的就诊次数和当月费用总额（两位小数），按月份升序。",
            "visit_date>='2027-01-01'",
        ),
        (
            "kg",
            "对临床关系按头实体聚合边数和平均置信度，按边数降序取前4个；并列时按头实体升序。关系代码 relation = HAS_SYMPTOM。",
            "relation='HAS_SYMPTOM'",
        ),
        (
            "kg_entity_order",
            "列出疾病节点里提及次数达到11次的名称与次数，优先显示次数高者，同次按名称升序。实体类型 type = Disease。",
            "ORDER BY mention_count DESC,name ASC",
        ),
        (
            "kg_grouping",
            "将置信度不低于0.88的知识图谱边按关系中文名与头实体类型分组，统计每组的边数、最低置信度、观察次数总和，并按关系中文名升序、头实体类型升序排列。",
            "GROUP BY relation_zh,head_type",
        ),
        (
            "temporal_each",
            "对测试地区患者，找出2027年1月1日起各自最早一次就诊，显示患者编号、姓名、日期、疾病和费用，按日期及编号升序。就诊日期 visit_date >= 2027-01-01。",
            "SELECT MIN(v2.visit_date)",
        ),
        (
            "window",
            "先按科室和疾病汇总总费用，再在每个科室内按指标降序做稠密排名，保留前2名，按科室、名次、疾病升序。",
            "DENSE_RANK()",
        ),
    ],
)
def test_v21_bounded_shapes_emit_safe_sql(family, question, required):
    sql = compile_independent_v21_sql(question, DOMAINS)
    assert sql is not None, family
    assert required in sql
    validate_gold_sql(sql)


@pytest.mark.parametrize(
    "question",
    [
        "删除全部患者。",
        "连接患者与就诊记录后随便分析一下。",
        "按科室分组，仅保留一些组。",
        "关系代码 relation = UNKNOWN_REL，列出图谱边。",
        "查询未在值域中的地区年龄不低于30岁的患者姓名、性别和年龄。",
    ],
)
def test_v21_bounded_compiler_abstains_on_incomplete_or_ungrounded_input(question):
    assert compile_independent_v21_sql(question, DOMAINS) is None
