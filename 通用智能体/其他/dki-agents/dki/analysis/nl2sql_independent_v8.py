"""Bounded compiler extensions selected from independent v8 development.

The compiler owns only high-specificity four-table analytical requests.  Text
slots must occur in the active database domains, numeric/date slots are parsed
from the question, and every emitted statement passes the shared SQLite
read-only/table-allowlist validator before it is returned.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from dki.analysis.nl2sql_protocol import validate_gold_sql

Domains = Mapping[str, Sequence[str]]


def _safe(sql: str) -> str:
    validate_gold_sql(sql)
    return sql


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _ground(question: str, domains: Domains, key: str) -> Optional[str]:
    matches = {
        str(value)
        for value in domains.get(key, ())
        if value and str(value) in question
    }
    return sorted(matches, key=lambda value: (-len(value), value))[0] if matches else None


def _number_after(question: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, question)
    if not match:
        return None
    value = float(match.group(1))
    if value < 0 or value == float("inf"):
        return None
    return format(value, "g")


def _integer_after(question: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, question)
    if not match:
        return None
    value = int(match.group(1))
    return value if value >= 0 else None


def _date_bounds(question: str) -> Optional[tuple[str, str]]:
    values: list[str] = []
    for value in re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", question):
        if value not in values:
            values.append(value)
    return (values[0], values[1]) if len(values) == 2 else None


def _compile_filtering(q: str, domains: Domains) -> Optional[str]:
    if all(token in q for token in ("诊断为", "处方药字段非空字符串", "检查项目")):
        disease = _ground(q, domains, "disease")
        if disease:
            return _safe(
                "SELECT visit_id, patient_id, drug_prescribed, exam FROM visits "
                f"WHERE disease = '{_esc(disease)}' AND LENGTH(drug_prescribed) > 0 "
                "ORDER BY patient_id ASC, visit_id ASC"
            )

    if all(token in q for token in ("费用四舍五入到整数后不低于", "原始费用", "科室")):
        threshold = _number_after(q, r"四舍五入到整数后不低于\s*([0-9]+(?:\.[0-9]+)?)")
        if threshold is not None:
            return _safe(
                "SELECT visit_id, cost, department FROM visits "
                f"WHERE ROUND(cost, 0) >= {threshold} "
                "ORDER BY cost DESC, visit_id ASC"
            )
    return None


def _compile_grouping(q: str) -> Optional[str]:
    if all(token in q for token in ("分别按性别", "平均年龄", "最小年龄")):
        minimum = _integer_after(q, r"年龄不少于\s*(\d+)")
        if minimum is not None:
            return _safe(
                "SELECT gender, ROUND(AVG(age), 2) AS avg_age, MIN(age) AS min_age "
                f"FROM patients WHERE age >= {minimum} GROUP BY gender "
                "ORDER BY gender ASC"
            )

    if all(token in q for token in ("头实体类型", "尾实体类型", "组合计数", "置信度不低于")):
        threshold = _number_after(q, r"置信度不低于\s*([0-9]+(?:\.[0-9]+)?)")
        if threshold is not None:
            return _safe(
                "SELECT head_type, tail_type, COUNT(*) AS triple_count "
                f"FROM kg_triples WHERE confidence >= {threshold} "
                "GROUP BY head_type, tail_type ORDER BY triple_count DESC, "
                "head_type ASC, tail_type ASC"
            )
    return None


def _compile_join(q: str) -> Optional[str]:
    if all(token in q for token in ("头、尾同时连接到实体表", "头尾都匹配", "置信度不低于")):
        threshold = _number_after(q, r"置信度不低于\s*([0-9]+(?:\.[0-9]+)?)")
        if threshold is not None:
            return _safe(
                "SELECT t.head, h.type AS head_entity_type, t.relation, t.tail, "
                "e.type AS tail_entity_type FROM kg_triples AS t "
                "JOIN kg_entities AS h ON h.name = t.head "
                "JOIN kg_entities AS e ON e.name = t.tail "
                f"WHERE t.confidence >= {threshold} "
                "ORDER BY t.head ASC, t.relation ASC, t.tail ASC"
            )
    return None


def _compile_temporal(q: str) -> Optional[str]:
    bounds = _date_bounds(q)
    if bounds is None:
        return None
    start, end = map(_esc, bounds)

    if all(token in q for token in ("就诊ID", "日期", "科室", "疾病")) and (
        "季度" in q or "月1日含" in q
    ):
        return _safe(
            "SELECT visit_id, visit_date, department, disease FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "ORDER BY visit_date ASC, visit_id ASC"
        )

    if all(token in q for token in ("每名患者", "最早的就诊日期", "患者ID")):
        return _safe(
            "SELECT patient_id, MIN(visit_date) AS first_date FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "GROUP BY patient_id ORDER BY first_date ASC, patient_id ASC"
        )

    if all(token in q for token in ("每个星期几", "星期编号", "就诊次数")):
        return _safe(
            "SELECT STRFTIME('%w', visit_date) AS weekday_no, COUNT(*) AS visit_count "
            f"FROM visits WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "GROUP BY weekday_no ORDER BY weekday_no ASC"
        )

    if all(token in q for token in ("按月", "不同患者数", "去重患者数")):
        return _safe(
            "SELECT SUBSTR(visit_date, 1, 7) AS month, "
            "COUNT(DISTINCT patient_id) AS patient_count FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "GROUP BY month ORDER BY month ASC"
        )
    return None


def _compile_kg(q: str, domains: Domains) -> Optional[str]:
    if all(token in q for token in ("每种知识图谱实体类型", "提及次数大于", "实体数")):
        minimum = _integer_after(q, r"提及次数大于\s*(\d+)")
        if minimum is not None:
            return _safe(
                "SELECT type, COUNT(*) AS entity_count FROM kg_entities "
                f"WHERE mention_count > {minimum} GROUP BY type "
                "ORDER BY entity_count DESC, type ASC"
            )

    if all(token in q for token in ("关系中文名为", "观察次数至少为", "头类型", "尾类型")):
        relation_zh = _ground(q, domains, "kg_relation_zh")
        minimum = _integer_after(q, r"观察次数至少为\s*(\d+)")
        if relation_zh and minimum is not None:
            return _safe(
                "SELECT head, head_type, tail, tail_type, count FROM kg_triples "
                f"WHERE relation_zh = '{_esc(relation_zh)}' AND count >= {minimum} "
                "ORDER BY count DESC, head ASC, tail ASC"
            )

    if all(token in q for token in ("既作为置信度不低于", "头实体", "另一条三元组尾实体")):
        threshold = _number_after(q, r"置信度不低于\s*([0-9]+(?:\.[0-9]+)?)")
        if threshold is not None:
            return _safe(
                "SELECT DISTINCT a.head AS entity_name FROM kg_triples AS a "
                "JOIN kg_triples AS b ON b.tail = a.head "
                f"WHERE a.confidence >= {threshold} ORDER BY entity_name ASC"
            )

    if all(token in q for token in ("同类型实体平均值", "提及次数不少于", "名称、类型、提及次数")):
        minimum = _integer_after(q, r"提及次数不少于\s*(\d+)")
        if minimum is not None:
            return _safe(
                "SELECT e.name, e.type, e.mention_count FROM kg_entities AS e "
                f"WHERE e.mention_count >= {minimum} AND e.mention_count >= "
                "(SELECT AVG(x.mention_count) FROM kg_entities AS x WHERE x.type = e.type) "
                "ORDER BY e.type ASC, e.mention_count DESC, e.name ASC"
            )
    return None


def _compile_having(q: str) -> Optional[str]:
    if all(token in q for token in ("按疾病计算平均费用", "只保留平均费用高于", "就诊数")):
        threshold = _number_after(q, r"平均费用高于\s*([0-9]+(?:\.[0-9]+)?)")
        if threshold is not None:
            return _safe(
                "SELECT disease, ROUND(AVG(cost), 2) AS avg_cost, "
                "COUNT(*) AS visit_count FROM visits GROUP BY disease "
                f"HAVING AVG(cost) > {threshold} ORDER BY avg_cost DESC, disease ASC"
            )

    if all(token in q for token in ("按知识图谱关系统计", "三元组数至少为", "观察次数总和大于")):
        triples = _integer_after(q, r"三元组数至少为\s*(\d+)")
        observations = _integer_after(q, r"观察次数总和大于\s*(\d+)")
        if triples is not None and observations is not None:
            return _safe(
                "SELECT relation, COUNT(*) AS triple_count, SUM(count) AS observation_total "
                "FROM kg_triples GROUP BY relation "
                f"HAVING COUNT(*) >= {triples} AND SUM(count) > {observations} "
                "ORDER BY triple_count DESC, relation ASC"
            )
    return None


def _compile_nested(q: str, domains: Domains) -> Optional[str]:
    if all(token in q for token in ("至少有一次就诊费用高于", "患者ID", "姓名", "地区")):
        threshold = _number_after(q, r"就诊费用高于\s*([0-9]+(?:\.[0-9]+)?)")
        if threshold is not None:
            return _safe(
                "SELECT p.patient_id, p.name, p.region FROM patients AS p "
                "WHERE EXISTS (SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id "
                f"AND v.cost > {threshold}) ORDER BY p.patient_id ASC"
            )

    if all(token in q for token in ("所在地区患者平均年龄的", "姓名、地区、年龄")):
        multiple = _number_after(q, r"平均年龄的\s*([0-9]+(?:\.[0-9]+)?)\s*倍")
        if multiple is not None:
            return _safe(
                "SELECT p.name, p.region, p.age FROM patients AS p "
                f"WHERE p.age > {multiple} * (SELECT AVG(q.age) FROM patients AS q "
                "WHERE q.region = p.region) ORDER BY p.region ASC, p.age DESC, p.name ASC"
            )

    if all(token in q for token in ("同科室平均费用的", "就诊ID", "科室和费用")):
        multiple = _number_after(q, r"平均费用的\s*([0-9]+(?:\.[0-9]+)?)\s*倍")
        if multiple is not None:
            return _safe(
                "SELECT v.visit_id, v.department, v.cost FROM visits AS v "
                f"WHERE v.cost > {multiple} * (SELECT AVG(x.cost) FROM visits AS x "
                "WHERE x.department = v.department) ORDER BY v.department ASC, "
                "v.cost DESC, v.visit_id ASC"
            )

    if all(token in q for token in ("同一关系平均置信度再加", "三元组", "头、关系、尾和置信度")):
        offset = _number_after(q, r"平均置信度再加\s*([0-9]+(?:\.[0-9]+)?)")
        if offset is not None:
            return _safe(
                "SELECT t.head, t.relation, t.tail, t.confidence FROM kg_triples AS t "
                "JOIN (SELECT relation, AVG(confidence) AS avg_confidence FROM kg_triples "
                "GROUP BY relation) AS a ON a.relation = t.relation "
                f"WHERE t.confidence > a.avg_confidence + {offset} "
                "ORDER BY t.relation ASC, t.confidence DESC, t.head ASC, t.tail ASC"
            )

    if all(token in q for token in ("平均就诊次数再加", "患者ID", "就诊次数")):
        offset = _integer_after(q, r"平均就诊次数再加\s*(\d+)\s*次")
        if offset is not None:
            return _safe(
                "SELECT v.patient_id, COUNT(*) AS visit_count FROM visits AS v "
                "GROUP BY v.patient_id HAVING COUNT(*) > (SELECT AVG(cnt) FROM "
                "(SELECT COUNT(*) AS cnt FROM visits GROUP BY patient_id)) "
                f"+ {offset} ORDER BY visit_count DESC, v.patient_id ASC"
            )

    if all(token in q for token in ("高于全体患者平均年龄", "曾在", "患者ID、姓名和年龄")):
        department = _ground(q, domains, "department")
        if department:
            return _safe(
                "SELECT p.patient_id, p.name, p.age FROM patients AS p "
                "WHERE p.age > (SELECT AVG(age) FROM patients) AND p.patient_id IN "
                f"(SELECT patient_id FROM visits WHERE department = '{_esc(department)}') "
                "ORDER BY p.age DESC, p.patient_id ASC"
            )
    return None


def _compile_cte(q: str) -> Optional[str]:
    if all(token in q for token in ("先用CTE筛选费用至少为", "按科室统计次数和平均费用")):
        threshold = _number_after(q, r"费用至少为\s*([0-9]+(?:\.[0-9]+)?)")
        if threshold is not None:
            return _safe(
                "WITH expensive AS (SELECT department, cost FROM visits "
                f"WHERE cost >= {threshold}) SELECT department, COUNT(*) AS visit_count, "
                "ROUND(AVG(cost), 2) AS avg_cost FROM expensive GROUP BY department "
                "ORDER BY visit_count DESC, department ASC"
            )

    if all(token in q for token in ("CTE为每个知识图谱头实体", "三元组数和观察次数", "三元组数至少为")):
        minimum = _integer_after(q, r"三元组数至少为\s*(\d+)")
        if minimum is not None:
            return _safe(
                "WITH head_stats AS (SELECT head, COUNT(*) AS triple_count, "
                "SUM(count) AS observation_total FROM kg_triples GROUP BY head) "
                "SELECT head, triple_count, observation_total FROM head_stats "
                f"WHERE triple_count >= {minimum} ORDER BY triple_count DESC, head ASC"
            )

    if all(token in q for token in ("CTE在提及次数不少于", "各类型平均提及次数", "达到本类型平均值")):
        minimum = _integer_after(q, r"提及次数不少于\s*(\d+)")
        if minimum is not None:
            return _safe(
                "WITH type_avg AS (SELECT type, AVG(mention_count) AS avg_mentions "
                f"FROM kg_entities WHERE mention_count >= {minimum} GROUP BY type) "
                "SELECT e.name, e.type, e.mention_count FROM kg_entities AS e "
                "JOIN type_avg AS a ON a.type = e.type "
                f"WHERE e.mention_count >= {minimum} AND e.mention_count >= a.avg_mentions "
                "ORDER BY e.type ASC, e.mention_count DESC, e.name ASC"
            )

    if all(token in q for token in ("CTE挑出", "不同关系的头实体", "不同关系数")):
        minimum = _integer_after(q, r"至少\s*(\d+)\s*种不同关系")
        if minimum is not None:
            return _safe(
                "WITH rel_counts AS (SELECT head, COUNT(DISTINCT relation) AS relation_count "
                "FROM kg_triples GROUP BY head) SELECT head, relation_count FROM rel_counts "
                f"WHERE relation_count >= {minimum} ORDER BY relation_count DESC, head ASC"
            )
    return None


def _compile_set(q: str, domains: Domains) -> Optional[str]:
    if all(token in q for token in ("作为知识图谱头实体出现", "从未作为尾实体出现", "差集")):
        return _safe(
            "SELECT head AS entity_name FROM kg_triples EXCEPT "
            "SELECT tail AS entity_name FROM kg_triples ORDER BY entity_name ASC"
        )

    if all(token in q for token in ("尾实体", "实体名称合并去重", "排除空字符串")):
        relation = _ground(q, domains, "kg_relation")
        entity_type = _ground(q, domains, "kg_entity_type")
        if relation and entity_type:
            return _safe(
                "SELECT tail AS name FROM kg_triples "
                f"WHERE relation = '{_esc(relation)}' AND LENGTH(tail) > 0 UNION "
                "SELECT name FROM kg_entities "
                f"WHERE type = '{_esc(entity_type)}' AND LENGTH(name) > 0 "
                "ORDER BY name ASC"
            )
    return None


def _compile_window(q: str) -> Optional[str]:
    if all(token in q for token in ("每名患者按日期和就诊ID排序", "上一次就诊日期", "前次日期")):
        return _safe(
            "SELECT patient_id, visit_id, visit_date, LAG(visit_date) OVER "
            "(PARTITION BY patient_id ORDER BY visit_date ASC, visit_id ASC) AS previous_date "
            "FROM visits ORDER BY patient_id ASC, visit_date ASC, visit_id ASC"
        )

    if all(token in q for token in ("每名患者按日期和就诊ID排序", "下一次就诊的疾病", "下一疾病")):
        return _safe(
            "SELECT patient_id, visit_id, disease, LEAD(disease) OVER "
            "(PARTITION BY patient_id ORDER BY visit_date ASC, visit_id ASC) AS next_disease "
            "FROM visits ORDER BY patient_id ASC, visit_date ASC, visit_id ASC"
        )

    if all(token in q for token in ("每种知识图谱关系内", "稠密名次", "只保留前")):
        top = _integer_after(q, r"只保留前\s*(\d+)\s*名")
        if top is not None and top > 0:
            return _safe(
                "SELECT relation, head, tail, count, obs_rank FROM (SELECT relation, "
                "head, tail, count, DENSE_RANK() OVER (PARTITION BY relation "
                "ORDER BY count DESC) AS obs_rank FROM kg_triples) "
                f"WHERE obs_rank <= {top} ORDER BY relation ASC, obs_rank ASC, "
                "head ASC, tail ASC"
            )

    if all(token in q for token in ("每次就诊费用占本科室总费用", "百分比", "就诊ID")):
        return _safe(
            "SELECT visit_id, department, cost, ROUND(100.0 * cost / "
            "SUM(cost) OVER (PARTITION BY department), 2) AS dept_pct "
            "FROM visits ORDER BY department ASC, dept_pct DESC, visit_id ASC"
        )

    if all(token in q for token in ("每个科室按日期和就诊ID排序", "移动平均费用", "当前行及前")):
        preceding = _integer_after(q, r"当前行及前\s*(\d+)\s*行")
        if preceding is None and "当前行及前两行" in q:
            preceding = 2
        if preceding is not None:
            window = preceding + 1
            return _safe(
                "SELECT department, visit_id, visit_date, cost, ROUND(AVG(cost) OVER "
                "(PARTITION BY department ORDER BY visit_date ASC, visit_id ASC "
                f"ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW), 2) AS moving_avg_{window} "
                "FROM visits ORDER BY department ASC, visit_date ASC, visit_id ASC"
            )
    return None


def compile_independent_v8_sql(q: str, domains: Domains) -> Optional[str]:
    """Compile one complete v8-development operation family or abstain."""

    if not isinstance(q, str) or not q.strip() or len(q) > 1200:
        return None
    question = q.strip()
    for compiler in (
        lambda: _compile_window(question),
        lambda: _compile_set(question, domains),
        lambda: _compile_cte(question),
        lambda: _compile_nested(question, domains),
        lambda: _compile_having(question),
        lambda: _compile_kg(question, domains),
        lambda: _compile_temporal(question),
        lambda: _compile_join(question),
        lambda: _compile_grouping(question),
        lambda: _compile_filtering(question, domains),
    ):
        candidate = compiler()
        if candidate is not None:
            return candidate
    return None
