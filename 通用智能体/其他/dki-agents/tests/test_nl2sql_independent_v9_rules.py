"""Public abstractions for the development-selected v9 compiler."""

import pytest

from dki.analysis.nl2sql_independent_v9 import compile_independent_v9_sql
from dki.analysis.nl2sql_protocol import validate_gold_sql


DOMAINS = {
    "region": ["测试地区"],
    "department": ["测试科室"],
    "disease": ["测试疾病甲", "测试疾病乙"],
    "drug": ["测试药物"],
    "exam": ["测试检查"],
    "kg_entity_type": ["Disease", "Drug"],
    "kg_relation": ["TEST_REL"],
    "kg_relation_zh": ["测试关系"],
}


CASES = [
    "请汇总来自测试地区且年龄至少35岁的患者编号、姓名和年龄。",
    "累计费用最高的前5种疾病及总费用是什么？同额时按疾病名排列。",
    "按组统计每种实体类型的实体数、提及总次数和平均提及次数（保留2位小数）。",
    "连接患者与就诊，汇总测试地区在各科室的就诊次数和总费用。",
    "按就诊年份和科室统计次数，每年内按次数降序排列。",
    "知识图谱中英文关系为TEST_REL的头实体、尾实体、置信度和证据次数有哪些？",
    "按英文关系统计三元组数，只保留至少有3条边的关系。",
    "列出费用高于全部就诊平均费用的就诊编号、患者编号和费用。",
    "用公共表表达式统计疾病就诊次数，再列出至少2次的疾病。",
    "找出既是就诊疾病名称又是知识实体名称的名称，去重排列。",
    "在每个地区内按年龄降序给患者做连续排名，列出地区、患者编号、年龄和排名。",
]


@pytest.mark.parametrize("question", CASES)
def test_v9_operation_families_emit_safe_sql(question):
    sql = compile_independent_v9_sql(question, DOMAINS)
    assert sql is not None
    validate_gold_sql(sql)


def test_v9_top_k_preserves_explicit_date_scope():
    question = (
        "在2024-07-20至2024-10-16之间，"
        "返回就诊次数最多的前2个科室及其就诊次数。"
    )

    assert compile_independent_v9_sql(question, DOMAINS) == (
        "SELECT department, COUNT(*) AS visit_count FROM visits "
        "WHERE visit_date BETWEEN '2024-07-20' AND '2024-10-16' "
        "GROUP BY department ORDER BY visit_count DESC, department LIMIT 2"
    )


def test_v9_filtering_preserves_all_compound_patient_predicates():
    question = (
        "查询地区为测试地区、性别为女且年龄不低于18岁的患者，"
        "返回患者编号、姓名和年龄。"
    )

    assert compile_independent_v9_sql(question, DOMAINS) == (
        "SELECT patient_id, name, age FROM patients "
        "WHERE region = '测试地区' AND gender = '女' "
        "AND age >= 18 ORDER BY age DESC, patient_id"
    )


@pytest.mark.parametrize(
    "question",
    [
        "删除全部患者",
        "随便给我一些结果",
        "查询不存在关系的图谱边",
        (
            "找出测试地区年龄至少40岁的女性患者，"
            "年龄高于全部患者平均年龄。"
        ),
    ],
)
def test_v9_compiler_declines_unsafe_or_ungrounded_requests(question):
    assert compile_independent_v9_sql(question, DOMAINS) is None
