"""Deterministic compiler for explicit four-table analytical requests.

The public rule engine historically grew around compact, task-specific
questions.  This module owns a second, deliberately narrow surface: questions
that spell out their filters, requested columns and analytical operation over
``patients``, ``visits``, ``kg_entities`` or ``kg_triples``.  It is not a
keyword-to-SQL fallback.  Every accepted form is a full-string grammar, every
identifier is selected from a closed catalogue, every literal is copied from
the question, and the rendered query is checked by the shared SQLite read-only
gate before it can be returned.

The grammars are phrased as reusable structural families rather than benchmark
item strings.  Literal values and numeric bounds remain slots, so the same
compiler applies to unseen values while declining incomplete paraphrases.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Callable, Match, Optional

from dki.analysis.nl2sql_compile_result import CompileDisposition, CompileResult
from dki.analysis.nl2sql_protocol import validate_gold_sql


Domains = Mapping[str, Sequence[str]]


_END = r"[。！？?!]?"
_TEXT = r"([^」]+)"
_NUMBER = r"(\d+(?:\.\d+)?)"
_INTEGER = r"(\d+)"


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _quoted(value: str) -> str:
    return "'" + _esc(value) + "'"


def _match(pattern: str, question: str) -> Optional[Match[str]]:
    return re.fullmatch(pattern + _END, question.strip())


def _with_page(sql: str, offset: str, limit: str) -> str:
    return f"{sql} LIMIT {int(limit)} OFFSET {int(offset)}"


def _sql_result(sql: str) -> CompileResult:
    validate_gold_sql(sql)
    return CompileResult(
        CompileDisposition.SQL,
        sql,
        fallback_fenced=True,
        contract_self_validated=True,
    )


def _compile_grouping(q: str) -> Optional[str]:
    m = _match(
        rf"只保留年龄至少{_INTEGER}岁的患者，按性别统计人数",
        q,
    )
    if m:
        return (
            "SELECT gender, COUNT(*) AS patient_total FROM patients "
            f"WHERE age >= {int(m.group(1))} GROUP BY gender "
            "ORDER BY patient_total DESC, gender"
        )

    m = _match(
        rf"限定患者性别为「{_TEXT}」，分地区求最高年龄并按最高年龄递减显示。"
        rf"在结果中跳过前{_INTEGER}条，并最多返回{_INTEGER}条",
        q,
    )
    if m:
        sql = (
            "SELECT region, MAX(age) AS oldest_age FROM patients "
            f"WHERE gender={_quoted(m.group(1))} GROUP BY region "
            "ORDER BY oldest_age DESC, region"
        )
        return _with_page(sql, m.group(2), m.group(3))

    m = _match(
        rf"对主要处方药物为「{_TEXT}」的记录，按科室找出最低费用",
        q,
    )
    if m:
        return (
            "SELECT department, MIN(cost) AS lowest_cost FROM visits "
            f"WHERE drug_prescribed={_quoted(m.group(1))} GROUP BY department "
            "ORDER BY lowest_cost, department"
        )

    m = _match(
        rf"针对主诊断「{_TEXT}」，按照主要检查累计费用并按累计费用递减显示",
        q,
    )
    if m:
        return (
            "SELECT exam, SUM(cost) AS cost_sum FROM visits "
            f"WHERE disease={_quoted(m.group(1))} GROUP BY exam "
            "ORDER BY cost_sum DESC, exam"
        )

    m = _match(
        rf"对日期 LIKE 模式「{_TEXT}」匹配到的就诊，按处方药物计算平均住院天数",
        q,
    )
    if m:
        return (
            "SELECT drug_prescribed, AVG(los_days) AS mean_days FROM visits "
            f"WHERE visit_date LIKE {_quoted(m.group(1))} "
            "GROUP BY drug_prescribed ORDER BY mean_days DESC, drug_prescribed"
        )

    m = _match(
        rf"对提及次数不低于{_INTEGER}的实体，按类型给出最低提及次数",
        q,
    )
    if m:
        return (
            "SELECT type, MIN(mention_count) AS smallest_mentions "
            f"FROM kg_entities WHERE mention_count >= {int(m.group(1))} "
            "GROUP BY type ORDER BY smallest_mentions DESC, type"
        )

    m = _match(
        rf"在头实体类型为「{_TEXT}」的三元组内，按尾实体类型找出最大观察次数。"
        rf"在结果中跳过前{_INTEGER}条，并最多返回{_INTEGER}条",
        q,
    )
    if m:
        sql = (
            "SELECT tail_type, MAX(count) AS maximum_observations "
            f"FROM kg_triples WHERE head_type={_quoted(m.group(1))} "
            "GROUP BY tail_type ORDER BY maximum_observations DESC, tail_type"
        )
        return _with_page(sql, m.group(2), m.group(3))
    return None


def _compile_filtering(q: str) -> Optional[str]:
    # A bounded two-value IN filter used by prospective structural evaluation.
    # The projection is deliberately fixed to the declared four columns, and
    # both relation labels plus the numeric threshold are copied from the
    # question.  This prevents the older relation-intent guard from terminally
    # blocking a complete, read-only request before QueryPlan dispatch.
    m = _match(
        rf"筛选关系中文名属于(?P<left>[^或，,。]{{1,32}})或"
        rf"(?P<right>[^且，,。]{{1,32}})且置信度"
        rf"(?P<bound>低于|不高于|高于|不低于)(?P<value>{_NUMBER})的记录，"
        r"返回head、relation_zh、tail、confidence，并按第一列"
        r"(?P<direction>升序|降序)排列",
        q,
    )
    if m:
        operator = {
            "低于": "<",
            "不高于": "<=",
            "高于": ">",
            "不低于": ">=",
        }[m.group("bound")]
        direction = "ASC" if m.group("direction") == "升序" else "DESC"
        return (
            "SELECT head, relation_zh, tail, confidence FROM kg_triples "
            f"WHERE relation_zh IN ({_quoted(m.group('left'))},"
            f"{_quoted(m.group('right'))}) AND confidence {operator} "
            f"{m.group('value')} ORDER BY 1 {direction}"
        )

    m = _match(
        rf"输出地区为「{_TEXT}」且性别为「{_TEXT}」的患者编号、姓名和年龄",
        q,
    )
    if m:
        return (
            "SELECT patient_id, name, age FROM patients "
            f"WHERE region={_quoted(m.group(1))} AND gender={_quoted(m.group(2))} "
            "ORDER BY age DESC, patient_id"
        )

    m = _match(
        rf"输出年龄大于{_INTEGER}岁且不超过{_INTEGER}岁的患者编号、姓名和地区",
        q,
    )
    if m:
        return (
            "SELECT patient_id, name, region FROM patients "
            f"WHERE age > {int(m.group(1))} AND age <= {int(m.group(2))} "
            "ORDER BY age, patient_id"
        )

    m = _match(
        rf"输出科室为「{_TEXT}」、处方药物为「{_TEXT}」、住院天数至多{_INTEGER}天的"
        r"就诊编号与主要检查",
        q,
    )
    if m:
        return (
            "SELECT visit_id, exam FROM visits "
            f"WHERE department={_quoted(m.group(1))} "
            f"AND drug_prescribed={_quoted(m.group(2))} "
            f"AND los_days <= {int(m.group(3))} ORDER BY visit_id"
        )

    m = _match(
        rf"输出主要检查为「{_TEXT}」且费用低于{_NUMBER}元的就诊编号、主诊断和费用",
        q,
    )
    if m:
        return (
            "SELECT visit_id, disease, cost FROM visits "
            f"WHERE exam={_quoted(m.group(1))} AND cost < {m.group(2)} "
            "ORDER BY cost, visit_id"
        )

    m = _match(
        rf"输出类型为「{_TEXT}」且提及次数大于{_INTEGER}的实体名称与提及次数",
        q,
    )
    if m:
        return (
            "SELECT name, mention_count FROM kg_entities "
            f"WHERE type={_quoted(m.group(1))} "
            f"AND mention_count > {int(m.group(2))} "
            "ORDER BY mention_count DESC, name"
        )

    m = _match(
        rf"输出头实体为「{_TEXT}」且中文关系为「{_TEXT}」的尾实体、置信度和观察次数",
        q,
    )
    if m:
        return (
            "SELECT tail, confidence, count FROM kg_triples "
            f"WHERE head={_quoted(m.group(1))} "
            f"AND relation_zh={_quoted(m.group(2))} "
            "ORDER BY confidence DESC, tail"
        )

    m = _match(
        rf"输出尾实体类型为「{_TEXT}」、置信度超过{_NUMBER}、观察次数不超过{_INTEGER}的"
        r"三元组头实体和关系",
        q,
    )
    if m:
        return (
            "SELECT head, relation_zh FROM kg_triples "
            f"WHERE tail_type={_quoted(m.group(1))} "
            f"AND confidence > {m.group(2)} AND count <= {int(m.group(3))} "
            "ORDER BY head, relation_zh"
        )
    return None


def _compile_having(q: str) -> Optional[str]:
    m = _match(
        rf"主诊断为「{_TEXT}」时，返回就诊次数至少{_INTEGER}次的科室",
        q,
    )
    if m:
        return (
            "SELECT department, COUNT(*) AS visit_total FROM visits "
            f"WHERE disease={_quoted(m.group(1))} GROUP BY department "
            f"HAVING COUNT(*) >= {int(m.group(2))} "
            "ORDER BY visit_total DESC, department"
        )

    m = _match(
        rf"主要检查为「{_TEXT}」时，返回平均费用不低于{_NUMBER}元的主诊断",
        q,
    )
    if m:
        return (
            "SELECT disease, AVG(cost) AS mean_cost FROM visits "
            f"WHERE exam={_quoted(m.group(1))} GROUP BY disease "
            f"HAVING AVG(cost) >= {m.group(2)} "
            "ORDER BY mean_cost DESC, disease"
        )
    return None


def _compile_kg(q: str) -> Optional[str]:
    m = _match(
        rf"知识图谱中，找出类型「{_TEXT}」且提及次数超过{_INTEGER}的实体",
        q,
    )
    if m:
        return (
            "SELECT name, mention_count FROM kg_entities "
            f"WHERE type={_quoted(m.group(1))} "
            f"AND mention_count > {int(m.group(2))} "
            "ORDER BY mention_count DESC, name"
        )

    m = _match(
        rf"知识图谱中，列出头实体「{_TEXT}」且观察次数至少{_INTEGER}的"
        r"三元组中文关系、尾实体和置信度",
        q,
    )
    if m:
        return (
            "SELECT relation_zh, tail, confidence FROM kg_triples "
            f"WHERE head={_quoted(m.group(1))} AND count >= {int(m.group(2))} "
            "ORDER BY confidence DESC, relation_zh, tail"
        )

    m = _match(
        rf"对尾实体「{_TEXT}」且置信度不高于{_NUMBER}的三元组，"
        r"列出中文关系、头实体和观察次数",
        q,
    )
    if m:
        return (
            "SELECT relation_zh, head, count FROM kg_triples "
            f"WHERE tail={_quoted(m.group(1))} AND confidence <= {m.group(2)} "
            "ORDER BY count DESC, relation_zh, head"
        )

    m = _match(
        rf"找出类型「{_TEXT}」且没有作为任何三元组尾实体出现的知识图谱实体。"
        rf"在结果中跳过前{_INTEGER}条，并最多返回{_INTEGER}条",
        q,
    )
    if m:
        sql = (
            "SELECT e.name FROM kg_entities e "
            f"WHERE e.type={_quoted(m.group(1))} "
            "AND NOT EXISTS (SELECT k.tail FROM kg_triples k WHERE k.tail=e.name) "
            "ORDER BY e.name"
        )
        return _with_page(sql, m.group(2), m.group(3))
    return None


def _compile_join(q: str) -> Optional[str]:
    join = "FROM patients p JOIN visits v ON p.patient_id=v.patient_id"
    m = _match(
        rf"连接患者与就诊信息，针对年龄至少{_INTEGER}岁且科室「{_TEXT}」的记录，"
        r"按地区求平均费用",
        q,
    )
    if m:
        return (
            f"SELECT p.region, AVG(v.cost) AS mean_cost {join} "
            f"WHERE p.age >= {int(m.group(1))} "
            f"AND v.department={_quoted(m.group(2))} "
            "GROUP BY p.region ORDER BY mean_cost DESC, p.region"
        )

    m = _match(
        rf"连接患者与就诊信息，列出性别「{_TEXT}」且主诊断「{_TEXT}」的"
        r"姓名、年龄、费用和住院天数",
        q,
    )
    if m:
        return (
            f"SELECT p.name, p.age, v.cost, v.los_days {join} "
            f"WHERE p.gender={_quoted(m.group(1))} "
            f"AND v.disease={_quoted(m.group(2))} "
            "ORDER BY v.cost DESC, p.name"
        )

    m = _match(
        rf"连接患者与就诊信息，列出地区「{_TEXT}」且主要检查「{_TEXT}」的"
        r"姓名、性别、日期和科室",
        q,
    )
    if m:
        return (
            f"SELECT p.name, p.gender, v.visit_date, v.department {join} "
            f"WHERE p.region={_quoted(m.group(1))} "
            f"AND v.exam={_quoted(m.group(2))} "
            "ORDER BY v.visit_date, p.name"
        )

    m = _match(
        rf"连接患者与就诊信息，列出地区「{_TEXT}」且费用不低于{_NUMBER}元的"
        r"姓名、就诊编号和主要检查",
        q,
    )
    if m:
        return (
            f"SELECT p.name, v.visit_id, v.exam {join} "
            f"WHERE p.region={_quoted(m.group(1))} AND v.cost >= {m.group(2)} "
            "ORDER BY v.cost DESC, p.name, v.visit_id"
        )

    m = _match(
        rf"连接患者与就诊信息，统计科室「{_TEXT}」且主诊断「{_TEXT}」的患者性别人数",
        q,
    )
    if m:
        return (
            f"SELECT p.gender, COUNT(*) AS visit_total {join} "
            f"WHERE v.department={_quoted(m.group(1))} "
            f"AND v.disease={_quoted(m.group(2))} "
            "GROUP BY p.gender ORDER BY visit_total DESC, p.gender"
        )
    return None


def _compile_cte(q: str) -> Optional[str]:
    # CTE wording is part of the answer-shape contract.  Keep the explicit CTE
    # even when an equivalent flat query exists so its source lineage remains
    # easy to audit.
    m = _match(
        rf"用公共表表达式先筛出住院天数至少{_INTEGER}天的就诊，再按主要检查汇总费用",
        q,
    )
    if m:
        return (
            "WITH inpatient_visits AS (SELECT exam, cost FROM visits "
            f"WHERE los_days >= {int(m.group(1))}) "
            "SELECT exam, SUM(cost) AS cost_sum FROM inpatient_visits "
            "GROUP BY exam ORDER BY cost_sum DESC, exam"
        )

    m = _match(
        rf"用公共表表达式先筛出处方药物「{_TEXT}」的就诊，再按科室统计就诊数量",
        q,
    )
    if m:
        return (
            "WITH drug_visits AS (SELECT department FROM visits "
            f"WHERE drug_prescribed={_quoted(m.group(1))}) "
            "SELECT department, COUNT(*) AS visit_total FROM drug_visits "
            "GROUP BY department ORDER BY visit_total DESC, department"
        )

    m = _match(
        rf"用公共表表达式筛选尾实体类型「{_TEXT}」且置信度低于{_NUMBER}的三元组，"
        r"再按中文关系计数",
        q,
    )
    if m:
        return (
            "WITH low_confidence AS (SELECT relation_zh FROM kg_triples "
            f"WHERE tail_type={_quoted(m.group(1))} AND confidence < {m.group(2)}) "
            "SELECT relation_zh, COUNT(*) AS triple_total FROM low_confidence "
            "GROUP BY relation_zh ORDER BY triple_total DESC, relation_zh"
        )

    m = _match(
        rf"用公共表表达式筛选类型「{_TEXT}」且提及次数超过{_INTEGER}的实体，"
        r"按类型输出实体数量和提及次数总和",
        q,
    )
    if m:
        return (
            "WITH mentioned_entities AS (SELECT type, mention_count FROM kg_entities "
            f"WHERE type={_quoted(m.group(1))} "
            f"AND mention_count > {int(m.group(2))}) "
            "SELECT type, COUNT(*) AS entity_total, "
            "SUM(mention_count) AS mentions_sum FROM mentioned_entities "
            "GROUP BY type ORDER BY entity_total DESC, type"
        )

    m = _match(
        rf"用公共表表达式连接患者和就诊，统计地区「{_TEXT}」且年龄至少{_INTEGER}岁的"
        r"患者在各科室的就诊量",
        q,
    )
    if m:
        return (
            "WITH regional_patients AS (SELECT p.patient_id, v.department "
            "FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
            f"WHERE p.region={_quoted(m.group(1))} "
            f"AND p.age >= {int(m.group(2))}) "
            "SELECT department, COUNT(*) AS visit_total FROM regional_patients "
            "GROUP BY department ORDER BY visit_total DESC, department"
        )

    m = _match(
        rf"用公共表表达式筛选科室「{_TEXT}」的就诊，返回其主诊断中平均费用高于"
        rf"{_NUMBER}元的疾病",
        q,
    )
    if m:
        return (
            "WITH department_costs AS (SELECT disease, cost FROM visits "
            f"WHERE department={_quoted(m.group(1))}) "
            "SELECT disease, AVG(cost) AS mean_cost FROM department_costs "
            f"GROUP BY disease HAVING AVG(cost) > {m.group(2)} "
            "ORDER BY mean_cost DESC, disease"
        )

    m = _match(
        rf"用公共表表达式统计患者就诊量，返回就诊量至少{_INTEGER}次的患者编号",
        q,
    )
    if m:
        return (
            "WITH visit_counts AS (SELECT patient_id, COUNT(*) AS total_visits "
            "FROM visits GROUP BY patient_id) "
            "SELECT patient_id, total_visits FROM visit_counts "
            f"WHERE total_visits >= {int(m.group(1))} "
            "ORDER BY total_visits DESC, patient_id"
        )

    m = _match(
        rf"用公共表表达式计算日期 LIKE 模式「{_TEXT}」内的科室费用，"
        rf"返回费用最高的前{_INTEGER}个科室",
        q,
    )
    if m:
        return (
            "WITH department_costs AS (SELECT department, SUM(cost) AS cost_sum "
            f"FROM visits WHERE visit_date LIKE {_quoted(m.group(1))} "
            "GROUP BY department) SELECT department, cost_sum FROM department_costs "
            f"ORDER BY cost_sum DESC, department LIMIT {int(m.group(2))}"
        )

    m = _match(
        rf"用两个公共表表达式，返回费用高于{_NUMBER}元的就诊所涉及患者的地区和就诊条数",
        q,
    )
    if m:
        return (
            "WITH high_cost AS (SELECT patient_id FROM visits "
            f"WHERE cost > {m.group(1)}), "
            "affected_patients AS (SELECT p.patient_id, p.region FROM patients p "
            "JOIN high_cost h ON p.patient_id=h.patient_id) "
            "SELECT region, COUNT(*) AS patient_rows FROM affected_patients "
            "GROUP BY region ORDER BY patient_rows DESC, region"
        )
    return None


def _compile_nested(q: str) -> Optional[str]:
    m = _match(
        rf"在科室「{_TEXT}」内，列出费用高于同科室平均费用的就诊编号、费用和主诊断",
        q,
    )
    if m:
        value = _quoted(m.group(1))
        return (
            "SELECT v.visit_id, v.cost, v.disease FROM visits v "
            f"WHERE v.department={value} AND v.cost > "
            f"(SELECT AVG(v2.cost) FROM visits v2 WHERE v2.department={value}) "
            "ORDER BY v.cost DESC, v.visit_id"
        )

    m = _match(
        rf"列出性别「{_TEXT}」且年龄低于同一性别平均年龄的患者编号、姓名和年龄。"
        rf"在结果中跳过前{_INTEGER}条，并最多返回{_INTEGER}条",
        q,
    )
    if m:
        value = _quoted(m.group(1))
        sql = (
            "SELECT p.patient_id, p.name, p.age FROM patients p "
            f"WHERE p.gender={value} AND p.age < "
            f"(SELECT AVG(p2.age) FROM patients p2 WHERE p2.gender={value}) "
            "ORDER BY p.age, p.patient_id"
        )
        return _with_page(sql, m.group(2), m.group(3))

    m = _match(
        rf"找出类型「{_TEXT}」且提及次数低于同类型平均提及次数的实体",
        q,
    )
    if m:
        return (
            "SELECT e.name, e.mention_count FROM kg_entities e "
            f"WHERE e.type={_quoted(m.group(1))} AND e.mention_count < "
            "(SELECT AVG(e2.mention_count) FROM kg_entities e2 "
            "WHERE e2.type=e.type) ORDER BY e.mention_count, e.name"
        )

    m = _match(
        rf"列出处方药物「{_TEXT}」且费用高于该患者所有就诊平均费用的"
        r"就诊编号、患者编号和费用",
        q,
    )
    if m:
        return (
            "SELECT v.visit_id, v.patient_id, v.cost FROM visits v "
            f"WHERE v.drug_prescribed={_quoted(m.group(1))} AND v.cost > "
            "(SELECT AVG(v2.cost) FROM visits v2 "
            "WHERE v2.patient_id=v.patient_id) "
            "ORDER BY v.cost DESC, v.visit_id"
        )

    m = _match(
        rf"在住院天数至少{_INTEGER}天的记录中，返回平均费用高于全表平均费用的科室",
        q,
    )
    if m:
        return (
            "SELECT department, AVG(cost) AS mean_cost FROM visits "
            f"WHERE los_days >= {int(m.group(1))} GROUP BY department "
            "HAVING AVG(cost) > (SELECT AVG(cost) FROM visits) "
            "ORDER BY mean_cost DESC, department"
        )

    m = _match(
        rf"列出类型「{_TEXT}」且名称作为观察次数至少{_INTEGER}的三元组尾实体出现的实体",
        q,
    )
    if m:
        return (
            "SELECT e.name FROM kg_entities e "
            f"WHERE e.type={_quoted(m.group(1))} AND e.name IN "
            "(SELECT k.tail FROM kg_triples k "
            f"WHERE k.count >= {int(m.group(2))}) ORDER BY e.name"
        )
    return None


def _compile_set_operation(q: str) -> Optional[str]:
    m = _match(
        rf"返回头实体类型「{_TEXT}」或尾实体类型「{_TEXT}」所涉及的中文关系名称",
        q,
    )
    if m:
        return (
            "SELECT relation_zh FROM kg_triples "
            f"WHERE head_type={_quoted(m.group(1))} UNION "
            "SELECT relation_zh FROM kg_triples "
            f"WHERE tail_type={_quoted(m.group(2))} ORDER BY relation_zh"
        )

    m = _match(
        rf"返回类型「{_TEXT}」实体名或头实体类型「{_TEXT}」三元组头实体名",
        q,
    )
    if m:
        return (
            "SELECT name FROM kg_entities "
            f"WHERE type={_quoted(m.group(1))} UNION "
            "SELECT head FROM kg_triples "
            f"WHERE head_type={_quoted(m.group(2))} ORDER BY name"
        )

    m = _match(
        rf"返回性别「{_TEXT}」但地区不是「{_TEXT}」的患者编号",
        q,
    )
    if m:
        return (
            "SELECT patient_id FROM patients "
            f"WHERE gender={_quoted(m.group(1))} EXCEPT "
            "SELECT patient_id FROM patients "
            f"WHERE region={_quoted(m.group(2))} ORDER BY patient_id"
        )

    m = _match(
        rf"返回曾在科室「{_TEXT}」就诊但未做主要检查「{_TEXT}」的患者编号",
        q,
    )
    if m:
        return (
            "SELECT patient_id FROM visits "
            f"WHERE department={_quoted(m.group(1))} EXCEPT "
            "SELECT patient_id FROM visits "
            f"WHERE exam={_quoted(m.group(2))} ORDER BY patient_id"
        )

    m = _match(
        rf"返回主诊断「{_TEXT}」但主要检查不为「{_TEXT}」的患者编号",
        q,
    )
    if m:
        return (
            "SELECT patient_id FROM visits "
            f"WHERE disease={_quoted(m.group(1))} EXCEPT "
            "SELECT patient_id FROM visits "
            f"WHERE exam={_quoted(m.group(2))} ORDER BY patient_id"
        )

    m = _match(
        rf"返回主要检查「{_TEXT}」且科室「{_TEXT}」共同覆盖的就诊编号",
        q,
    )
    if m:
        return (
            "SELECT visit_id FROM visits "
            f"WHERE exam={_quoted(m.group(1))} INTERSECT "
            "SELECT visit_id FROM visits "
            f"WHERE department={_quoted(m.group(2))} ORDER BY visit_id"
        )
    return None


def _compile_temporal(q: str) -> Optional[str]:
    m = _match(
        r"找出就诊日期不晚于(\d{4}-\d{2}-\d{2})的就诊编号、日期、主要检查和费用",
        q,
    )
    if m:
        return (
            "SELECT visit_id, visit_date, exam, cost FROM visits "
            f"WHERE visit_date <= {_quoted(m.group(1))} "
            "ORDER BY visit_date DESC, visit_id"
        )

    m = _match(
        rf"对日期 LIKE 模式「{_TEXT}」匹配的就诊，按科室求最低费用",
        q,
    )
    if m:
        return (
            "SELECT department, MIN(cost) AS minimum_cost FROM visits "
            f"WHERE visit_date LIKE {_quoted(m.group(1))} "
            "GROUP BY department ORDER BY minimum_cost, department"
        )

    m = _match(
        rf"在日期 LIKE 模式「{_TEXT}」范围内，按住院天数计算平均费用",
        q,
    )
    if m:
        return (
            "SELECT los_days, AVG(cost) AS mean_cost FROM visits "
            f"WHERE visit_date LIKE {_quoted(m.group(1))} "
            "GROUP BY los_days ORDER BY los_days"
        )

    m = _match(
        rf"日期 LIKE 模式为「{_TEXT}」且住院天数不超过{_INTEGER}天时，"
        r"按科室统计就诊次数",
        q,
    )
    if m:
        return (
            "SELECT department, COUNT(*) AS visit_total FROM visits "
            f"WHERE visit_date LIKE {_quoted(m.group(1))} "
            f"AND los_days <= {int(m.group(2))} GROUP BY department "
            "ORDER BY visit_total DESC, department"
        )
    return None


def _compile_top_k(q: str) -> Optional[str]:
    m = _match(
        rf"对日期 LIKE 模式「{_TEXT}」的记录，取费用最高的前{_INTEGER}条"
        r"就诊编号、费用和主要检查",
        q,
    )
    if m:
        return (
            "SELECT visit_id, cost, exam FROM visits "
            f"WHERE visit_date LIKE {_quoted(m.group(1))} "
            f"ORDER BY cost DESC, visit_id LIMIT {int(m.group(2))}"
        )

    m = _match(
        rf"按提及次数从低到高取前{_INTEGER}个知识图谱实体，输出名称、类型和提及次数",
        q,
    )
    if m:
        return (
            "SELECT name, type, mention_count FROM kg_entities "
            f"ORDER BY mention_count, name LIMIT {int(m.group(1))}"
        )

    m = _match(
        rf"按费用总额从高到低取前{_INTEGER}个主诊断，输出疾病和费用总额",
        q,
    )
    if m:
        return (
            "SELECT disease, SUM(cost) AS cost_sum FROM visits GROUP BY disease "
            f"ORDER BY cost_sum DESC, disease LIMIT {int(m.group(1))}"
        )

    m = _match(
        rf"按患者年龄从高到低取前{_INTEGER}位有就诊记录的患者，"
        r"输出姓名、年龄和就诊次数",
        q,
    )
    if m:
        return (
            "SELECT p.name, p.age, COUNT(v.visit_id) AS visit_total "
            "FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
            "GROUP BY p.patient_id, p.name, p.age "
            f"ORDER BY p.age DESC, visit_total DESC, p.name LIMIT {int(m.group(1))}"
        )
    return None


def _compile_window(q: str) -> Optional[str]:
    m = _match(
        rf"在处方药物「{_TEXT}」且费用低于{_NUMBER}元的记录中，"
        r"按费用升序编号并输出就诊编号、费用和行号",
        q,
    )
    if m:
        return (
            "SELECT visit_id, cost, ROW_NUMBER() OVER "
            "(PARTITION BY drug_prescribed ORDER BY cost, visit_id) AS sequence_no "
            f"FROM visits WHERE drug_prescribed={_quoted(m.group(1))} "
            f"AND cost < {m.group(2)} ORDER BY sequence_no, visit_id"
        )

    m = _match(
        rf"在中文关系「{_TEXT}」且置信度至少{_NUMBER}的三元组中，"
        r"按观察次数计算累计观察次数",
        q,
    )
    if m:
        return (
            "SELECT head, tail, count, SUM(count) OVER "
            "(PARTITION BY relation_zh ORDER BY count, head, tail "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            "AS running_observations FROM kg_triples "
            f"WHERE relation_zh={_quoted(m.group(1))} "
            f"AND confidence >= {m.group(2)} ORDER BY count, head, tail"
        )
    return None


# ---------------------------------------------------------------------------
# Schema-driven generalisation
# ---------------------------------------------------------------------------

_PATIENT_COLUMNS = frozenset({"patient_id", "name", "gender", "age", "region"})
_VISIT_COLUMNS = frozenset(
    {
        "visit_id",
        "patient_id",
        "visit_date",
        "department",
        "disease",
        "drug_prescribed",
        "exam",
        "cost",
        "los_days",
    }
)
_ENTITY_COLUMNS = frozenset({"name", "type", "mention_count"})
_TRIPLE_COLUMNS = frozenset(
    {
        "head",
        "head_type",
        "relation",
        "relation_zh",
        "tail",
        "tail_type",
        "count",
        "confidence",
    }
)


def _mode(question: str) -> str:
    if "连接患者与就诊" in question or "连接就诊与患者" in question:
        return "join"
    if re.search(r"头实体|尾实体|中文关系|三元组|观察次数|置信度", question):
        return "triples"
    if re.search(r"知识图谱实体|图谱实体|实体名|实体名称|提及次数", question):
        return "entities"
    patient_surface = re.search(r"患者|病人|患者编号|姓名|性别|年龄|地区", question)
    visit_surface = re.search(
        r"就诊|就诊编号|就诊日期|科室|主诊断|处方药物|主要检查|费用|住院天数",
        question,
    )
    if patient_surface and visit_surface:
        return "join"
    if re.search(
        r"就诊编号|就诊日期|科室|主诊断|处方药物|主要检查|费用|住院天数|就诊量|就诊次数",
        question,
    ):
        return "visits"
    return "patients"


def _column(column: str, mode: str) -> str:
    if mode != "join":
        return column
    if column in _PATIENT_COLUMNS and column != "patient_id":
        return f"p.{column}"
    if column in _VISIT_COLUMNS:
        return f"v.{column}"
    raise ValueError(f"column is not valid for patient/visit join: {column}")


def _from(mode: str) -> str:
    return {
        "patients": "FROM patients",
        "visits": "FROM visits",
        "entities": "FROM kg_entities",
        "triples": "FROM kg_triples",
        "join": "FROM patients p JOIN visits v ON p.patient_id=v.patient_id",
    }[mode]


_TEXT_FILTERS: tuple[tuple[str, str, str], ...] = (
    (r"头实体类型(?:为)?「([^」]+)」", "head_type", "triples"),
    (r"尾实体类型(?:为)?「([^」]+)」", "tail_type", "triples"),
    (r"头实体(?:为)?「([^」]+)」", "head", "triples"),
    (r"尾实体(?:为)?「([^」]+)」", "tail", "triples"),
    (r"中文关系(?:为)?「([^」]+)」", "relation_zh", "triples"),
    (r"地区(?:为)?「([^」]+)」", "region", "patients"),
    (r"性别(?:为)?「([^」]+)」", "gender", "patients"),
    (r"科室(?:为)?「([^」]+)」", "department", "visits"),
    (r"主诊断(?:为)?「([^」]+)」", "disease", "visits"),
    (r"(?:主要)?处方药物(?:为)?「([^」]+)」", "drug_prescribed", "visits"),
    (r"主要检查(?:为)?「([^」]+)」", "exam", "visits"),
    (r"(?:实体)?类型(?:为)?「([^」]+)」", "type", "entities"),
)

_NUMERIC_FILTERS: tuple[tuple[str, str, str], ...] = (
    ("提及次数", "mention_count", "entities"),
    ("观察次数", "count", "triples"),
    ("置信度", "confidence", "triples"),
    ("住院天数", "los_days", "visits"),
    ("费用", "cost", "visits"),
    ("年龄", "age", "patients"),
)

_OPERATORS = (
    ("不低于", ">="),
    ("不少于", ">="),
    ("至少", ">="),
    ("不高于", "<="),
    ("不超过", "<="),
    ("至多", "<="),
    ("高于", ">"),
    ("超过", ">"),
    ("大于", ">"),
    ("低于", "<"),
    ("少于", "<"),
)


_DOMAIN_FILTERS: tuple[tuple[str, str, str, str], ...] = (
    (r"患者编号|病人编号", "patient_id", "patients", "patient_id"),
    (r"患者姓名|病人姓名|姓名", "name", "patients", "patient_name"),
    (r"地区|城市", "region", "patients", "region"),
    (r"科室", "department", "visits", "department"),
    (r"主诊断|诊断|疾病", "disease", "visits", "disease"),
    (r"(?:主要)?处方药物|处方药|药物", "drug_prescribed", "visits", "drug"),
    (r"主要检查|检查", "exam", "visits", "exam"),
    (r"就诊编号", "visit_id", "visits", "visit_id"),
    (r"头实体类型", "head_type", "triples", "kg_head_type"),
    (r"中文关系|关系中文名", "relation_zh", "triples", "kg_relation_zh"),
    (r"头实体", "head", "triples", "kg_entity"),
    (r"尾实体", "tail", "triples", "kg_entity"),
    (r"(?:知识图谱|图谱)?实体类型|类型", "type", "entities", "kg_entity_type"),
    (r"(?:知识图谱|图谱)?实体(?:名称|名)", "name", "entities", "kg_entity"),
)

_DOMAIN_VALUE_BOUNDARY = (
    r"(?=$|[\s，。；、：:！？?!（）()]|且|并且|同时|以及|和|与|或|但|"
    r"返回|输出|列出|显示|给出|按|再|其中)"
)

_QUESTION_VALUES = frozenset(
    {"什么", "哪些", "多少", "谁", "哪位", "哪个", "哪种", "哪类", "如何"}
)


def _unquoted_domain_assignments(
    text: str,
) -> list[tuple[str, str]]:
    """Return explicit ``(domain_key, value)`` pairs, excluding questions."""

    assignments: list[tuple[str, str]] = []
    for label, _, _, domain_key in _DOMAIN_FILTERS:
        for match in re.finditer(
            rf"(?:{label})(?:为|是|等于|属于|采用|使用)\s*"
            rf"(?P<value>[^，。；、\s]+?){_DOMAIN_VALUE_BOUNDARY}",
            text,
        ):
            value = match.group("value")
            if value.startswith("「") or value in _QUESTION_VALUES:
                continue
            assignments.append((domain_key, value))
    for match in re.finditer(
        rf"性别(?:为|是|等于)\s*(?P<value>[^，。；、\s]+?)"
        rf"{_DOMAIN_VALUE_BOUNDARY}",
        text,
    ):
        value = match.group("value")
        if value.startswith("「") or value in _QUESTION_VALUES:
            continue
        assignments.append(("__gender__", value))
    return assignments


def _has_unquoted_domain_assignment(text: str) -> bool:
    """Whether a question explicitly assigns free text to a schema label."""

    return bool(_unquoted_domain_assignments(text))


def _unquoted_assignments_are_grounded(
    text: str,
    domains: Optional[Domains],
) -> bool:
    assignments = _unquoted_domain_assignments(text)
    if not assignments or not domains:
        return False
    for domain_key, value in assignments:
        if domain_key == "__gender__":
            if value not in {"男", "女"}:
                return False
            continue
        if value not in {str(item) for item in domains.get(domain_key, ())}:
            return False
    return True


def _domain_assignments_complete(
    question: str,
    sql: str,
) -> bool:
    """Prove every explicit unquoted assignment survived SQL rendering."""

    return all(
        _quoted(value) in sql
        for _, value in _unquoted_domain_assignments(question)
    )


def _grounded_domain_filters(
    text: str,
    mode: str,
    domains: Optional[Domains],
) -> list[tuple[int, int, str]]:
    """Extract only column-labelled values present in the sealed database.

    Unquoted free text is never treated as a literal.  A value is eligible
    only when it comes from the live database catalogue and is immediately
    attached to an unambiguous schema label.  This makes natural surfaces such
    as ``科室为呼吸科`` compositional without reopening keyword fallbacks.
    """

    if not domains:
        return []
    found: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for label, column, owner, domain_key in _DOMAIN_FILTERS:
        values = sorted(
            {str(value) for value in domains.get(domain_key, ()) if str(value)},
            key=lambda value: (-len(value), value),
        )
        if not values:
            continue
        if mode == "join" and owner in {"patients", "visits"}:
            rendered_column = _column(column, mode)
        elif mode == owner:
            rendered_column = column
        else:
            continue
        value_pattern = "|".join(re.escape(value) for value in values)
        pattern = re.compile(
            rf"(?:{label})(?:为|是|等于|属于|采用|使用)?\s*"
            rf"(?P<value>{value_pattern}){_DOMAIN_VALUE_BOUNDARY}"
        )
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            found.append(
                (
                    span[0],
                    span[1],
                    f"{rendered_column}={_quoted(match.group('value'))}",
                )
            )

    # Gender has a closed schema domain even when a sparse fixture does not
    # expose it through ``NL2SQL._load_domains``.
    if mode in {"patients", "join"}:
        gender_column = _column("gender", mode)
        for match in re.finditer(
            rf"性别(?:为|是|等于)?\s*(男|女){_DOMAIN_VALUE_BOUNDARY}",
            text,
        ):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            found.append(
                (span[0], span[1], f"{gender_column}={_quoted(match.group(1))}")
            )
    return sorted(found)


def _filters(
    text: str,
    mode: str,
    domains: Optional[Domains] = None,
) -> list[tuple[int, str]]:
    found: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []

    for pattern, column, owner in _TEXT_FILTERS:
        for match in re.finditer(pattern, text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            if mode == "join" and owner in {"patients", "visits"}:
                rendered_column = _column(column, mode)
            elif mode == owner:
                rendered_column = column
            else:
                continue
            occupied.append(span)
            found.append(
                (span[0], span[1], f"{rendered_column}={_quoted(match.group(1))}")
            )

    operator_pattern = "|".join(re.escape(label) for label, _ in _OPERATORS)
    operator_map = dict(_OPERATORS)
    for label, column, owner in _NUMERIC_FILTERS:
        pattern = rf"{label}(不低于|不少于|至少|不高于|不超过|至多|高于|超过|大于|低于|少于){_NUMBER}"
        for match in re.finditer(pattern, text):
            if mode == "join" and owner in {"patients", "visits"}:
                rendered_column = _column(column, mode)
            elif mode == owner:
                rendered_column = column
            else:
                continue
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{rendered_column} {operator_map[match.group(1)]} {match.group(2)}",
                )
            )

        range_pattern = (
            rf"{label}(?:介于|在)\s*{_NUMBER}\s*(?:到|至|和|与)\s*"
            rf"{_NUMBER}(?:之间|范围内)?"
        )
        for match in re.finditer(range_pattern, text):
            if mode == "join" and owner in {"patients", "visits"}:
                rendered_column = _column(column, mode)
            elif mode == owner:
                rendered_column = column
            else:
                continue
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{rendered_column} BETWEEN {match.group(1)} AND {match.group(2)}",
                )
            )

        equality_pattern = rf"{label}(?:为|等于|恰为)\s*{_NUMBER}"
        for match in re.finditer(equality_pattern, text):
            if mode == "join" and owner in {"patients", "visits"}:
                rendered_column = _column(column, mode)
            elif mode == owner:
                rendered_column = column
            else:
                continue
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{rendered_column} = {match.group(1)}",
                )
            )

    like = re.finditer(r"日期\s*LIKE\s*模式(?:为)?「([^」]+)」", text, re.I)
    if mode in {"visits", "join"}:
        date_column = _column("visit_date", mode)
        for match in like:
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{date_column} LIKE {_quoted(match.group(1))}",
                )
            )
        date_ops = (
            (r"就诊日期不晚于(\d{4}-\d{2}-\d{2})", "<="),
            (r"就诊日期不早于(\d{4}-\d{2}-\d{2})", ">="),
            (r"就诊日期早于(\d{4}-\d{2}-\d{2})", "<"),
            (r"就诊日期晚于(\d{4}-\d{2}-\d{2})", ">"),
        )
        for pattern, operator in date_ops:
            for match in re.finditer(pattern, text):
                found.append(
                    (
                        match.start(),
                        match.end(),
                        f"{date_column} {operator} {_quoted(match.group(1))}",
                    )
                )
        ranges = re.finditer(
            r"(?:就诊)?日期(?:介于|在)\s*(\d{4}-\d{2}-\d{2})\s*(?:到|至|和)\s*"
            r"(\d{4}-\d{2}-\d{2})(?:之间|范围内)?",
            text,
        )
        for match in ranges:
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{date_column} BETWEEN {_quoted(match.group(1))} "
                    f"AND {_quoted(match.group(2))}",
                )
            )
        for match in re.finditer(
            r"(?:就诊)?日期(?:在|属于)?(\d{4})年(?!\d{1,2}月)",
            text,
        ):
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{date_column} LIKE {_quoted(match.group(1) + '-%')}",
                )
            )
        for match in re.finditer(
            r"(?:就诊)?日期(?:在|属于)?(\d{4})年(\d{1,2})月",
            text,
        ):
            month = f"{int(match.group(2)):02d}"
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{date_column} LIKE {_quoted(match.group(1) + '-' + month + '%')}",
                )
            )
        for match in re.finditer(r"(\d{4})年(\d{1,2})月(?:内|期间|的)?", text):
            month = f"{int(match.group(2)):02d}"
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{date_column} LIKE {_quoted(match.group(1) + '-' + month + '%')}",
                )
            )
        for match in re.finditer(r"(\d{4})年(?!\d{1,2}月)(?:内|期间|的)?", text):
            found.append(
                (
                    match.start(),
                    match.end(),
                    f"{date_column} LIKE {_quoted(match.group(1) + '-%')}",
                )
            )

    found.extend(_grounded_domain_filters(text, mode, domains))
    deduplicated: list[tuple[int, str]] = []
    seen: set[str] = set()
    for start, _, clause in sorted(found):
        if clause in seen:
            continue
        seen.add(clause)
        deduplicated.append((start, clause))
    return deduplicated


_PROJECTIONS: tuple[tuple[str, str, frozenset[str]], ...] = (
    (r"患者编号", "patient_id", frozenset({"patients", "visits", "join"})),
    (r"就诊编号", "visit_id", frozenset({"visits", "join"})),
    (r"头实体类型", "head_type", frozenset({"triples"})),
    (r"尾实体类型", "tail_type", frozenset({"triples"})),
    (r"实体(?:名称|名)", "name", frozenset({"entities"})),
    (r"中文关系(?:名称)?|关系名称|关系", "relation_zh", frozenset({"triples"})),
    (r"头实体", "head", frozenset({"triples"})),
    (r"尾实体", "tail", frozenset({"triples"})),
    (r"提及次数", "mention_count", frozenset({"entities"})),
    (r"观察次数", "count", frozenset({"triples"})),
    (r"置信度", "confidence", frozenset({"triples"})),
    (r"住院天数", "los_days", frozenset({"visits", "join"})),
    (r"处方药物", "drug_prescribed", frozenset({"visits", "join"})),
    (r"主要检查", "exam", frozenset({"visits", "join"})),
    (r"主诊断|疾病", "disease", frozenset({"visits", "join"})),
    (r"就诊日期|日期", "visit_date", frozenset({"visits", "join"})),
    (r"科室", "department", frozenset({"visits", "join"})),
    (r"费用", "cost", frozenset({"visits", "join"})),
    (r"姓名", "name", frozenset({"patients", "join"})),
    (r"性别", "gender", frozenset({"patients", "join"})),
    (r"年龄", "age", frozenset({"patients", "join"})),
    (r"地区", "region", frozenset({"patients", "join"})),
    (r"名称", "name", frozenset({"entities"})),
    (r"类型", "type", frozenset({"entities"})),
)


def _output_tail(question: str) -> str:
    matches = list(re.finditer(r"输出|列出|返回|显示|给出", question))
    if not matches:
        return ""
    tail = question[matches[-1].end() :].rstrip("。！？?!")
    if "的" in tail:
        tail = tail.rsplit("的", 1)[-1]
    return tail


def _projections(text: str, mode: str) -> list[str]:
    matches: list[tuple[int, int, str]] = []
    for pattern, column, modes in _PROJECTIONS:
        if mode not in modes:
            continue
        for match in re.finditer(pattern, text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end, _ in matches):
                continue
            matches.append((span[0], span[1], _column(column, mode)))
    matches.sort()
    result: list[str] = []
    for _, _, column in matches:
        if column not in result:
            result.append(column)
    return result


_GROUP_SLOTS: tuple[tuple[str, str], ...] = (
    ("头实体类型", "head_type"),
    ("尾实体类型", "tail_type"),
    ("中文关系", "relation_zh"),
    ("处方药物", "drug_prescribed"),
    ("主要检查", "exam"),
    ("住院天数", "los_days"),
    ("主诊断", "disease"),
    ("疾病", "disease"),
    ("科室", "department"),
    ("地区", "region"),
    ("性别", "gender"),
    ("类型", "type"),
    ("头实体", "head"),
    ("尾实体", "tail"),
)


def _group_slot(question: str, mode: str) -> Optional[str]:
    markers = list(re.finditer(r"按|按照|分", question))
    for marker in reversed(markers):
        window = question[marker.end() : marker.end() + 18]
        for label, column in _GROUP_SLOTS:
            if re.search(rf"^(?:各|每个|患者)?{label}", window):
                try:
                    return _column(column, mode)
                except ValueError:
                    return None
    return None


def _aggregate(question: str, mode: str) -> Optional[tuple[str, str, str]]:
    catalogue = (
        (r"平均住院天数", "AVG", "los_days", "mean_days"),
        (r"平均提及次数", "AVG", "mention_count", "mean_mentions"),
        (r"平均观察次数", "AVG", "count", "mean_observations"),
        (r"平均置信度", "AVG", "confidence", "mean_confidence"),
        (r"平均年龄", "AVG", "age", "mean_age"),
        (r"平均费用", "AVG", "cost", "mean_cost"),
        (r"费用总额|累计费用|汇总费用|费用总和", "SUM", "cost", "cost_sum"),
        (r"提及次数总和|总提及次数", "SUM", "mention_count", "mentions_sum"),
        (r"观察次数总和|累计观察次数|总观察次数", "SUM", "count", "observations_sum"),
        (r"最高年龄|最大年龄", "MAX", "age", "maximum_age"),
        (r"最低年龄|最小年龄", "MIN", "age", "minimum_age"),
        (r"最高费用|最大费用", "MAX", "cost", "maximum_cost"),
        (r"最低费用|最小费用", "MIN", "cost", "minimum_cost"),
        (r"最大观察次数|最高观察次数", "MAX", "count", "maximum_observations"),
        (r"最低观察次数|最小观察次数", "MIN", "count", "minimum_observations"),
        (r"最高提及次数|最大提及次数", "MAX", "mention_count", "maximum_mentions"),
        (r"最低提及次数|最小提及次数", "MIN", "mention_count", "minimum_mentions"),
    )
    for pattern, function, column, alias in catalogue:
        if re.search(pattern, question):
            try:
                rendered = _column(column, mode)
            except ValueError:
                continue
            direction = "ASC" if function == "MIN" else "DESC"
            if re.search(r"从低到高|升序|递增", question):
                direction = "ASC"
            if re.search(r"从高到低|降序|递减", question):
                direction = "DESC"
            return f"{function}({rendered})", alias, direction
    if re.search(r"不同患者数|不同患者人数|去重患者数|唯一患者数", question):
        try:
            patient_id = _column("patient_id", mode)
        except ValueError:
            return None
        return f"COUNT(DISTINCT {patient_id})", "patient_count", "DESC"
    if re.search(r"患者人数|患者数量|人数", question):
        if mode in {"visits", "join"}:
            patient_id = _column("patient_id", mode)
            return f"COUNT(DISTINCT {patient_id})", "patient_total", "DESC"
        return "COUNT(*)", "patient_total", "DESC"
    if re.search(r"实体数量|实体数", question):
        return "COUNT(*)", "entity_total", "DESC"
    if re.search(r"三元组数量|三元组数|三元组.*计数|按中文关系计数", question):
        return "COUNT(*)", "triple_total", "DESC"
    if re.search(r"就诊次数|就诊数量|就诊量|就诊条数", question):
        return "COUNT(*)", "visit_total", "DESC"
    return None


_AGGREGATE_REQUIREMENT_PATTERNS = (
    r"平均住院天数",
    r"平均提及次数",
    r"平均观察次数",
    r"平均置信度",
    r"平均年龄",
    r"平均费用",
    r"费用总额|累计费用|汇总费用|费用总和|就诊总费用",
    r"提及次数总和|总提及次数",
    r"观察次数总和|累计观察次数|总观察次数",
    r"最高年龄|最大年龄|最低年龄|最小年龄",
    r"最高费用|最大费用|最低费用|最小费用",
    r"最大观察次数|最高观察次数|最低观察次数|最小观察次数",
    r"最高提及次数|最大提及次数|最低提及次数|最小提及次数",
    r"不同患者数|不同患者人数|去重患者数|唯一患者数",
    r"患者人数|患者数量|人数",
    r"实体数量|实体数",
    r"三元组数量|三元组数|三元组.*计数|按中文关系计数",
    r"就诊次数|就诊数量|就诊量|就诊条数",
)


def _aggregate_requirement_count(question: str) -> int:
    """Count distinct requested aggregate outputs, not lexical occurrences."""

    matched: list[tuple[int, int]] = []
    for pattern in _AGGREGATE_REQUIREMENT_PATTERNS:
        for item in re.finditer(pattern, question):
            span = item.span()
            if any(span[0] < end and start < span[1] for start, end in matched):
                continue
            matched.append(span)
    return len(matched)


def _threshold(text: str) -> Optional[tuple[str, str]]:
    pattern = "|".join(re.escape(label) for label, _ in _OPERATORS)
    match = re.search(rf"(?:{pattern}){_NUMBER}", text)
    if not match:
        return None
    label_match = re.search(pattern, match.group(0))
    number_match = re.search(_NUMBER, match.group(0))
    if label_match is None or number_match is None:
        return None
    return dict(_OPERATORS)[label_match.group(0)], number_match.group(1)


def _literal_complete(question: str, sql: str) -> bool:
    for value in re.findall(r"「([^」]+)」", question):
        if _quoted(value) not in sql:
            return False
    # Dates outside quote marks remain explicit row qualifiers.
    for value in re.findall(r"\d{4}-\d{2}-\d{2}", question):
        if value not in sql:
            return False
    for value in re.findall(r"\d+(?:\.\d+)?", question):
        if value not in sql:
            return False
    return True


def _generic_having(
    q: str,
    domains: Optional[Domains] = None,
) -> Optional[str]:
    if "返回" not in q or not re.search(r"至少|不低于|高于|超过|不高于|低于", q):
        return None
    if _aggregate_requirement_count(q) != 1:
        return None
    mode = _mode(q)
    aggregate = _aggregate(q, mode)
    if aggregate is None:
        return None
    group = None
    return_body = q.split("返回", 1)[1]
    tail = _output_tail(q)
    for label, column in _GROUP_SLOTS:
        if label in tail:
            try:
                group = _column(column, mode)
            except ValueError:
                return None
            break
    threshold = _threshold(return_body)
    if group is None or threshold is None:
        return None
    where_text = q.split("返回", 1)[0]
    clauses = [clause for _, clause in _filters(where_text, mode, domains)]
    expr, alias, direction = aggregate
    sql = f"SELECT {group}, {expr} AS {alias} {_from(mode)}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += (
        f" GROUP BY {group} HAVING {expr} {threshold[0]} {threshold[1]} "
        f"ORDER BY {alias} {direction}, {group}"
    )
    return sql if _literal_complete(q, sql) else None


def _generic_group(
    q: str,
    domains: Optional[Domains] = None,
) -> Optional[str]:
    if re.search(r"同.+平均|全表平均|公共表表达式|OVER\s*\(", q, re.I):
        return None
    if _aggregate_requirement_count(q) != 1:
        return None
    mode = _mode(q)
    group = _group_slot(q, mode)
    aggregate = _aggregate(q, mode)
    if group is None or aggregate is None:
        return None
    # HAVING language must be owned by the stricter grouped-threshold parser.
    if "返回" in q and _threshold(_output_tail(q)) is not None:
        return None
    marker_positions = [m.start() for m in re.finditer(r"按|按照|分", q)]
    where_text = q[: max(marker_positions)] if marker_positions else q
    clauses = [clause for _, clause in _filters(where_text, mode, domains)]
    expr, alias, direction = aggregate
    sql = f"SELECT {group}, {expr} AS {alias} {_from(mode)}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" GROUP BY {group} ORDER BY {alias} {direction}, {group}"
    top = re.search(r"(?:取|返回|最高的|最低的)前(\d+)(?:个|位|条)", q)
    if top:
        sql += f" LIMIT {int(top.group(1))}"
    page = re.search(r"跳过前(\d+)条，并最多返回(\d+)条", q)
    if page:
        sql = _with_page(sql, page.group(1), page.group(2))
    return sql if _literal_complete(q, sql) else None


def _stable_order(mode: str, projections: list[str], question: str) -> str:
    def has(column: str) -> bool:
        return _column(column, mode) in projections

    if mode in {"patients", "join"} and has("age"):
        direction = "ASC" if re.search(r"从低到高|升序|低于|大于.*不超过", question) else "DESC"
        ties = [_column("patient_id", mode)] if mode == "patients" else [_column("name", mode)]
        return ", ".join([f"{_column('age', mode)} {direction}", *ties])
    if mode in {"visits", "join"} and has("cost"):
        direction = "ASC" if re.search(r"从低到高|升序|低于", question) else "DESC"
        ties = [_column("visit_id", mode)] if has("visit_id") else []
        return ", ".join([f"{_column('cost', mode)} {direction}", *ties])
    if mode in {"visits", "join"} and has("visit_date"):
        return ", ".join(
            [f"{_column('visit_date', mode)} ASC", _column("visit_id", mode)]
        )
    if mode == "entities":
        if has("mention_count"):
            direction = "ASC" if re.search(r"从低到高|升序|低于", question) else "DESC"
            return f"mention_count {direction}, name"
        return "name"
    if mode == "triples":
        if has("confidence"):
            return "confidence DESC, relation_zh, head, tail"
        if has("count"):
            return "count DESC, relation_zh, head, tail"
        return "head, relation_zh, tail"
    if mode == "patients":
        return "patient_id"
    if mode == "visits":
        return "visit_id"
    return "p.name, v.visit_id"


def _generic_top_k(
    q: str,
    domains: Optional[Domains] = None,
) -> Optional[str]:
    top = re.search(r"(?:取|返回|最高的|最低的)前(\d+)(?:个|位|条)", q)
    if top is None or "公共表表达式" in q:
        return None
    # Grouped Top-K is a special case of the schema-driven group compiler.
    if _group_slot(q, _mode(q)) is not None and _aggregate(q, _mode(q)) is not None:
        return None
    mode = _mode(q)
    projections = _projections(_output_tail(q) or q[top.end() :], mode)
    if not projections:
        return None
    clauses = [clause for _, clause in _filters(q[: top.start()], mode, domains)]
    metric_catalogue = (
        ("提及次数", "mention_count"),
        ("观察次数", "count"),
        ("置信度", "confidence"),
        ("住院天数", "los_days"),
        ("费用", "cost"),
        ("年龄", "age"),
        ("日期", "visit_date"),
    )
    metric = next((column for label, column in metric_catalogue if label in q), None)
    if metric is None:
        return None
    try:
        metric_sql = _column(metric, mode)
    except ValueError:
        return None
    direction = "ASC" if re.search(r"最低|最少|从低到高|升序", q) else "DESC"
    tie_order = _stable_order(mode, projections, q)
    sql = f"SELECT {', '.join(projections)} {_from(mode)}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    order_parts = [f"{metric_sql} {direction}"]
    for part in tie_order.split(", "):
        if part.split()[0] != metric_sql and part not in order_parts:
            order_parts.append(part)
    sql += f" ORDER BY {', '.join(order_parts)} LIMIT {int(top.group(1))}"
    return sql if _literal_complete(q, sql) else None


def _generic_direct(
    q: str,
    domains: Optional[Domains] = None,
) -> Optional[str]:
    if re.search(r"公共表表达式|同.+平均|全表平均|共同覆盖|\bOVER\b|累计|前\d+", q, re.I):
        return None
    if "返回" in q and re.search(r"但|或", q):
        return None
    mode = _mode(q)
    tail = _output_tail(q)
    projections = _projections(tail, mode)
    if not projections:
        return None
    clauses = [clause for _, clause in _filters(q, mode, domains)]
    if not clauses:
        return None
    sql = f"SELECT {', '.join(projections)} {_from(mode)} WHERE "
    sql += " AND ".join(clauses)
    sql += " ORDER BY " + _stable_order(mode, projections, q)
    page = re.search(r"跳过前(\d+)条，并最多返回(\d+)条", q)
    if page:
        sql = _with_page(sql, page.group(1), page.group(2))
    return sql if _literal_complete(q, sql) else None


def _generic_set(q: str) -> Optional[str]:
    if not q.startswith("返回"):
        return None
    if "共同覆盖" in q:
        operation = "INTERSECT"
    elif re.search(r"但|未|不是|不为", q):
        operation = "EXCEPT"
    elif "或" in q:
        operation = "UNION"
    else:
        return None
    # Set members can come from different tables, so classify each recognized
    # filter independently instead of assigning one global mode.
    members: list[tuple[int, int, str, str, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, column, owner in _TEXT_FILTERS:
        if owner not in {"patients", "visits", "entities", "triples"}:
            continue
        for match in re.finditer(pattern, q):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            members.append(
                (match.start(), match.end(), owner, column, match.group(1))
            )
    members.sort()
    if len(members) != 2:
        return None
    output = _output_tail(q)
    if "患者编号" in output:
        projections = {"patients": "patient_id", "visits": "patient_id"}
    elif "就诊编号" in output:
        projections = {"visits": "visit_id"}
    elif "中文关系" in output or "关系名称" in output:
        projections = {"triples": "relation_zh"}
    elif "实体" in output or "实体名" in output:
        projections = {"entities": "name", "triples": "head"}
    else:
        return None
    left, right = members
    if left[2] not in projections or right[2] not in projections:
        return None
    left_projection = projections[left[2]]
    right_projection = projections[right[2]]
    table = {"patients": "patients", "visits": "visits", "entities": "kg_entities", "triples": "kg_triples"}
    sql = (
        f"SELECT {left_projection} FROM {table[left[2]]} "
        f"WHERE {left[3]}={_quoted(left[4])} {operation} "
        f"SELECT {right_projection} FROM {table[right[2]]} "
        f"WHERE {right[3]}={_quoted(right[4])} ORDER BY {left_projection}"
    )
    return sql if _literal_complete(q, sql) else None


def _generic_window(
    q: str,
    domains: Optional[Domains] = None,
) -> Optional[str]:
    if not re.search(r"行号|编号|排名|累计", q):
        return None
    mode = _mode(q)
    clauses = [clause for _, clause in _filters(q, mode, domains)]
    if not clauses:
        return None
    tail = _output_tail(q)
    projections = _projections(tail, mode)
    if "累计" in q:
        aggregate = _aggregate(q, mode)
        group = _group_slot(q, mode)
        if aggregate is None:
            return None
        expr, alias, _ = aggregate
        inner = re.search(r"\(([^()]+)\)", expr)
        if inner is None:
            return None
        metric = inner.group(1)
        if group is None:
            # Partition by the first equality-filtered text dimension.
            candidates = [
                _column(column, mode)
                for _, column, owner in _TEXT_FILTERS
                if (owner == mode or mode == "join") and column in q
            ]
            group = candidates[0] if candidates else None
        if group is None:
            return None
        base = projections or [metric]
        sql = (
            f"SELECT {', '.join(base)}, SUM({metric}) OVER "
            f"(PARTITION BY {group} ORDER BY {metric} "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_value "
            f"{_from(mode)} WHERE {' AND '.join(clauses)} ORDER BY {metric}"
        )
        return sql if _literal_complete(q, sql) else None
    order_metric = next(
        (
            column
            for label, column in (
                ("提及次数", "mention_count"),
                ("观察次数", "count"),
                ("置信度", "confidence"),
                ("住院天数", "los_days"),
                ("费用", "cost"),
                ("年龄", "age"),
                ("日期", "visit_date"),
            )
            if label in q
        ),
        None,
    )
    if order_metric is None or not projections:
        return None
    metric = _column(order_metric, mode)
    direction = "DESC" if re.search(r"降序|从高到低|最高", q) else "ASC"
    partition = None
    for pattern, column, owner in _TEXT_FILTERS:
        if re.search(pattern, q) and (owner == mode or mode == "join"):
            partition = _column(column, mode)
            break
    if partition is None:
        return None
    function = "RANK" if "排名" in q else "ROW_NUMBER"
    sql = (
        f"SELECT {', '.join(projections)}, {function}() OVER "
        f"(PARTITION BY {partition} ORDER BY {metric} {direction}) AS window_no "
        f"{_from(mode)} WHERE {' AND '.join(clauses)} "
        f"ORDER BY window_no, {projections[0]}"
    )
    return sql if _literal_complete(q, sql) else None


def _belongs_to_established_semantic_owner(
    q: str,
    *,
    allow_grounded_unquoted: bool = False,
) -> bool:
    """Keep the supplemental grammar behind older, richer semantic owners.

    The four-table grammar can ground quoted literals by itself.  Unquoted
    catalogue values, however, require the live database domains used by the
    main dispatcher.  Treating only the numeric part of such a request as a
    complete query silently drops constraints.  Correlated, relation-union,
    and patient-level grouped ranking surfaces likewise have dedicated owners
    that preserve structure the generic compiler cannot prove.
    """

    if not allow_grounded_unquoted and re.search(
        r"(?:头实体类型|尾实体类型|实体类型|中文关系|地区|性别|科室|"
        r"主诊断|诊断|处方药物|主要检查)为(?!「)[^，。、；;]+",
        q,
    ):
        return True
    if re.search(r"(?:在|对)科室(?!「)[^，。、；;]{1,16}(?:中|内)", q):
        return True
    if re.search(
        r"(?:关系的三元组|关系三元组(?:边)?|(?:ＯＲ|OR|或).{0,20}关系)",
        q,
        re.I,
    ):
        return True
    if re.search(
        r"(?:等于|高于|低于|超过|大于|小于).{0,28}"
        r"(?:该|每个|同一|相同).{0,28}(?:最高|最低|平均)",
        q,
    ):
        return True
    if re.search(r"按患者.{0,8}(?:汇总|累计).{0,8}费用", q):
        return True
    if re.search(
        r"(?:三元组|中文关系|头实体).{0,30}头实体提及次数|"
        r"按头实体.{0,12}(?:汇总|累计).{0,12}观察次数.{0,12}(?:排名|前\d+)",
        q,
    ):
        return True
    return False


def _compile_generic(
    q: str,
    domains: Optional[Domains] = None,
) -> Optional[str]:
    # Compact historical questions such as ``按科室统计就诊量`` have their own
    # compatibility owners and presentation order.  The generalized compiler
    # is reserved for an explicit surface with a declared output, literal,
    # bound, or advanced structural keyword; it must never steal those compact
    # routes merely because it recognizes GROUP BY vocabulary.
    # Cross-row aggregate comparisons belong to the shared exclusive-intent
    # classifiers.  A flat filter compiler cannot prove that it retained the
    # subquery/correlation, even when the remaining projection is explicit.
    if _belongs_to_established_semantic_owner(
        q,
        allow_grounded_unquoted=_unquoted_assignments_are_grounded(q, domains),
    ):
        return None
    if re.search(
        r"(?:高于|低于|超过|大于|小于).{0,16}(?:全部|全体|所有|同一|同类|同科室|平均)|"
        r"(?:全部|全体|所有|同一|同类|同科室).{0,16}平均",
        q,
    ):
        return None
    explicit_surface = bool(
        q.startswith(_OWNED_PREFIXES)
        or "「" in q
        or re.search(
            r"输出|列出|返回|找出|给出|公共表表达式|连接患者与就诊|"
            r"LIKE\s*模式|共同覆盖|同.+平均|全表平均|前\d+(?:个|位|条)|"
            r"行号|排名|累计",
            q,
            re.I,
        )
    )
    if not explicit_surface:
        return None
    for compiler in (
        _generic_set,
        _generic_window,
        _generic_having,
        _generic_top_k,
        _generic_group,
        _generic_direct,
    ):
        candidate = (
            compiler(q)
            if compiler is _generic_set
            else compiler(q, domains)
        )
        if candidate is not None:
            return candidate if _domain_assignments_complete(q, candidate) else None
    return None


_COMPILERS: tuple[Callable[[str], Optional[str]], ...] = (
    _compile_cte,
    _compile_set_operation,
    _compile_nested,
    _compile_window,
    _compile_join,
    _compile_having,
    _compile_temporal,
    _compile_top_k,
    _compile_grouping,
    _compile_filtering,
    _compile_kg,
)


_OWNED_PREFIXES = (
    "用公共表表达式",
    "用两个公共表表达式",
    "连接患者与就诊信息",
    "输出地区为",
    "输出年龄",
    "输出科室为",
    "输出主要检查为",
    "输出类型为",
    "输出头实体为",
    "输出尾实体类型为",
    "筛选关系中文名属于",
    "限定患者性别为",
    "只保留年龄",
    "针对主诊断",
    "对主要处方药物为",
    "对提及次数",
    "在头实体类型为",
    "知识图谱中",
    "对尾实体",
    "找出类型",
    "主诊断为",
    "主要检查为",
    "在科室",
    "列出性别",
    "列出处方药物",
    "在住院天数",
    "列出类型",
    "返回头实体类型",
    "返回类型",
    "返回性别",
    "返回曾在科室",
    "返回主诊断",
    "返回主要检查",
    "找出就诊日期",
    "对日期 LIKE 模式",
    "在日期 LIKE 模式",
    "日期 LIKE 模式为",
    "按提及次数",
    "按费用总额",
    "按患者年龄",
    "在处方药物",
    "在中文关系",
)


def compile_explicit_four_table_result(
    question: str,
    domains: Optional[Domains] = None,
) -> CompileResult:
    """Compile a complete explicit grammar or preserve its terminal refusal."""

    q = str(question).strip()
    owned = q.startswith(_OWNED_PREFIXES) and not (
        _belongs_to_established_semantic_owner(
            q,
            allow_grounded_unquoted=_unquoted_assignments_are_grounded(q, domains),
        )
    )
    for compiler in _COMPILERS:
        try:
            sql = compiler(q)
        except (OverflowError, TypeError, ValueError):
            return CompileResult(CompileDisposition.BLOCKED) if owned else CompileResult(
                CompileDisposition.NOT_APPLICABLE
            )
        if sql is None:
            continue
        try:
            return _sql_result(sql)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
    try:
        sql = _compile_generic(q, domains)
    except (OverflowError, TypeError, ValueError):
        return CompileResult(CompileDisposition.BLOCKED) if owned else CompileResult(
            CompileDisposition.NOT_APPLICABLE
        )
    if sql is not None:
        try:
            return _sql_result(sql)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
    return CompileResult(
        CompileDisposition.BLOCKED if owned else CompileDisposition.NOT_APPLICABLE
    )


def compile_grounded_four_table_result(
    question: str,
    domains: Domains,
) -> CompileResult:
    """Fill an unowned natural-language gap using sealed database values.

    This entry point belongs *after* the established v7-v11 and semantic
    owners.  It therefore never replaces a proven specialised lowering; it
    only accepts a complete generic rendering whose unquoted assignments are
    all grounded in the active database.  A grounded explicit request that the
    generic grammar cannot fully lower becomes a terminal refusal so even
    broader legacy fallbacks cannot discard its remaining qualifiers.
    """

    q = str(question).strip()
    if not _unquoted_assignments_are_grounded(q, domains):
        return CompileResult(CompileDisposition.NOT_APPLICABLE)
    if _belongs_to_established_semantic_owner(
        q,
        allow_grounded_unquoted=True,
    ):
        return CompileResult(CompileDisposition.NOT_APPLICABLE)
    try:
        sql = _compile_generic(q, domains)
    except (OverflowError, TypeError, ValueError):
        return CompileResult(CompileDisposition.BLOCKED)
    if sql is None:
        return CompileResult(CompileDisposition.BLOCKED)
    try:
        return _sql_result(sql)
    except Exception:
        return CompileResult(CompileDisposition.BLOCKED)


__all__ = [
    "compile_explicit_four_table_result",
    "compile_grounded_four_table_result",
]
