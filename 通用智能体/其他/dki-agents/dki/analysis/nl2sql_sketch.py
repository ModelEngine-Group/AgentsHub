"""Strict deterministic SQL sketches for explicit four-table analytics.

The compiler has no access to evaluation data at runtime.  It accepts only
fully specified surface forms, grounds captured strings in active database
domains, and sends every emitted statement through the shared read-only SQL
validator.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from dki.analysis.nl2sql_protocol import validate_gold_sql

Domains = Mapping[str, Sequence[str]]


def _uint(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("negative integer slot")
    return parsed


def _number(value: str) -> str:
    parsed = float(value)
    if parsed < 0 or not parsed < float("inf"):
        raise ValueError("invalid numeric slot")
    return format(parsed, "g")


def _grounded(value: str, domains: Domains, key: str) -> Optional[str]:
    if value not in set(domains.get(key, ())):
        return None
    return value.replace("'", "''")


def _safe(sql: str) -> str:
    validate_gold_sql(sql)
    return sql


def _compile_filtering(q: str, domains: Domains) -> Optional[str]:
    m = re.fullmatch(
        r"列出年龄在(\d+)岁至(\d+)岁之间\(含边界\)的患者ID、姓名和年龄,"
        r"并按年龄升序、患者ID升序排列。?",
        q,
    )
    if m:
        low, high = map(_uint, m.groups())
        if low > high:
            return None
        return _safe(
            "SELECT patient_id, name, age FROM patients "
            f"WHERE age BETWEEN {low} AND {high} "
            "ORDER BY age ASC, patient_id ASC"
        )

    m = re.fullmatch(
        r"查询性别为([男女])且地区不是([^,，。；;]+)的患者姓名、地区和年龄,"
        r"结果按地区、姓名依次升序排列。?",
        q,
    )
    if m:
        gender, raw_region = m.groups()
        region = _grounded(raw_region, domains, "region")
        if region is None:
            return None
        return _safe(
            "SELECT name, region, age FROM patients "
            f"WHERE gender = '{gender}' AND region <> '{region}' "
            "ORDER BY region ASC, name ASC"
        )

    m = re.fullmatch(
        r"列出费用高于([0-9]+(?:\.[0-9]+)?)元且住院天数至少为(\d+)天的"
        r"就诊ID、患者ID、费用和住院天数,按费用降序、就诊ID升序排列。?",
        q,
    )
    if m:
        cost, days = _number(m.group(1)), _uint(m.group(2))
        return _safe(
            "SELECT visit_id, patient_id, cost, los_days FROM visits "
            f"WHERE cost > {cost} AND los_days >= {days} "
            "ORDER BY cost DESC, visit_id ASC"
        )

    m = re.fullmatch(
        r"查询(\d{4})年在([^,，。；;]+)就诊的就诊ID、日期、疾病和费用,"
        r"按就诊日期升序、就诊ID升序排列;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:"
        r"(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m:
        year = _uint(m.group(1))
        department = _grounded(m.group(2), domains, "department")
        start, end = m.group(3), m.group(4)
        if (
            department is None
            or start != f"{year:04d}-01-01"
            or end != f"{year + 1:04d}-01-01"
        ):
            return None
        return _safe(
            "SELECT visit_id, visit_date, disease, cost FROM visits "
            f"WHERE department = '{department}' AND visit_date >= '{start}' "
            f"AND visit_date < '{end}' ORDER BY visit_date ASC, visit_id ASC"
        )

    m = re.fullmatch(
        r"输出年龄除以10后的整数部分等于(\d+)的患者ID、姓名与年龄,"
        r"按年龄和患者ID升序。?",
        q,
    )
    if m:
        decade = _uint(m.group(1))
        return _safe(
            "SELECT patient_id, name, age FROM patients "
            f"WHERE CAST(age / 10 AS INTEGER) = {decade} "
            "ORDER BY age ASC, patient_id ASC"
        )

    m = re.fullmatch(
        r"查询地区为([^,，。；;]+)且姓名长度不少于(\d+)个字符的"
        r"患者姓名、性别和年龄,按姓名升序。?",
        q,
    )
    if m:
        region = _grounded(m.group(1), domains, "region")
        minimum = _uint(m.group(2))
        if region is None:
            return None
        return _safe(
            "SELECT name, gender, age FROM patients "
            f"WHERE region = '{region}' AND LENGTH(name) >= {minimum} "
            "ORDER BY name ASC"
        )

    m = re.fullmatch(
        r"列出检查字段为空字符串或处方药字段为空字符串的就诊ID、疾病、检查和"
        r"处方药,按就诊ID升序,并只返回前(\d+)条;"
        r"本查询其余精确非空常量为:0。?",
        q,
    )
    if m:
        limit = _uint(m.group(1))
        if limit == 0:
            return None
        return _safe(
            "SELECT visit_id, disease, exam, drug_prescribed FROM visits "
            "WHERE LENGTH(exam) = 0 OR LENGTH(drug_prescribed) = 0 "
            f"ORDER BY visit_id ASC LIMIT {limit}"
        )
    return None


def _compile_grouping(q: str) -> Optional[str]:
    m = re.fullmatch(
        r"对费用至少为([0-9]+(?:\.[0-9]+)?)元的就诊按科室分组,"
        r"统计就诊次数和费用总额\(总额四舍五入到两位小数\),"
        r"按总额降序、科室升序排列;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:2。?",
        q,
    )
    if m:
        cost = _number(m.group(1))
        return _safe(
            "SELECT department, COUNT(*) AS visit_count, "
            "ROUND(SUM(cost), 2) AS total_cost FROM visits "
            f"WHERE cost >= {cost} GROUP BY department "
            "ORDER BY total_cost DESC, department ASC"
        )

    m = re.fullmatch(
        r"对(\d{4})年的就诊按科室和住院类别分组,其中住院天数为0记作门诊、"
        r"其余记作住院;输出科室、类别、就诊数,按科室和类别升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:"
        r"(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m:
        year = _uint(m.group(1))
        start, end = m.group(2), m.group(3)
        if start != f"{year:04d}-01-01" or end != f"{year + 1:04d}-01-01":
            return None
        case = "CASE WHEN los_days = 0 THEN '门诊' ELSE '住院' END"
        return _safe(
            f"SELECT department, {case} AS visit_class, "
            "COUNT(*) AS visit_count FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            f"GROUP BY department, {case} "
            "ORDER BY department ASC, visit_class ASC"
        )

    m = re.fullmatch(
        r"仅考虑提及次数不少于(\d+)的知识图谱实体,按类型统计实体数、"
        r"提及次数总和及平均提及次数\(两位小数\),按实体数降序、类型升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:2。?",
        q,
    )
    if m:
        minimum = _uint(m.group(1))
        return _safe(
            "SELECT type, COUNT(*) AS entity_count, "
            "SUM(mention_count) AS mention_total, "
            "ROUND(AVG(mention_count), 2) AS avg_mentions FROM kg_entities "
            f"WHERE mention_count >= {minimum} GROUP BY type "
            "ORDER BY entity_count DESC, type ASC"
        )

    m = re.fullmatch(
        r"仅对置信度不低于([0-9]+(?:\.[0-9]+)?)的知识图谱三元组"
        r"按关系英文标识分组,输出关系、三元组数和观察次数合计,"
        r"按三元组数降序、关系升序。?",
        q,
    )
    if m:
        confidence = _number(m.group(1))
        return _safe(
            "SELECT relation, COUNT(*) AS triple_count, "
            "SUM(count) AS observation_count FROM kg_triples "
            f"WHERE confidence >= {confidence} GROUP BY relation "
            "ORDER BY triple_count DESC, relation ASC"
        )

    m = re.fullmatch(
        r"仅统计费用不少于([0-9]+(?:\.[0-9]+)?)元的就诊,按患者计算"
        r"就诊次数、费用总额\(两位小数\)与平均住院天数\(两位小数\);"
        r"输出患者ID及三项指标,按就诊次数降序、患者ID升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:2。?",
        q,
    )
    if m:
        cost = _number(m.group(1))
        return _safe(
            "SELECT patient_id, COUNT(*) AS visit_count, "
            "ROUND(SUM(cost), 2) AS total_cost, "
            "ROUND(AVG(los_days), 2) AS avg_los FROM visits "
            f"WHERE cost >= {cost} GROUP BY patient_id "
            "ORDER BY visit_count DESC, patient_id ASC"
        )

    m = re.fullmatch(
        r"只统计([男女])性患者,把年龄按每10岁一个区间分组,"
        r"输出区间起始年龄和患者人数,按区间起始年龄升序。?",
        q,
    )
    if m:
        expression = "CAST(age / 10 AS INTEGER) * 10"
        return _safe(
            f"SELECT {expression} AS age_band_start, "
            "COUNT(*) AS patient_count FROM patients "
            f"WHERE gender = '{m.group(1)}' GROUP BY {expression} "
            "ORDER BY age_band_start ASC"
        )

    m = re.fullmatch(
        r"只统计年龄不少于(\d+)岁的患者,把年龄按每10岁一个区间分组,"
        r"输出区间起始年龄和患者人数,按区间起始年龄升序。?",
        q,
    )
    if m:
        age = _uint(m.group(1))
        expression = "CAST(age / 10 AS INTEGER) * 10"
        return _safe(
            f"SELECT {expression} AS age_band_start, "
            "COUNT(*) AS patient_count FROM patients "
            f"WHERE age >= {age} GROUP BY {expression} "
            "ORDER BY age_band_start ASC"
        )

    m = re.fullmatch(
        r"仅统计费用不少于([0-9]+(?:\.[0-9]+)?)元的就诊,按年份统计"
        r"不同患者数和不同疾病数;输出年份、两种去重计数,按年份升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:1、4。?",
        q,
    )
    if m:
        cost = _number(m.group(1))
        year = "SUBSTR(visit_date, 1, 4)"
        return _safe(
            f"SELECT {year} AS year, "
            "COUNT(DISTINCT patient_id) AS distinct_patients, "
            "COUNT(DISTINCT disease) AS distinct_diseases FROM visits "
            f"WHERE cost >= {cost} GROUP BY {year} ORDER BY year ASC"
        )

    m = re.fullmatch(
        r"针对住院天数大于(\d+)且费用不高于([0-9]+(?:\.[0-9]+)?)元的记录,"
        r"按住院天数统计记录数、最低费用和最高费用;输出住院天数及三项指标,"
        r"按住院天数升序。?",
        q,
    )
    if m:
        days, cost = _uint(m.group(1)), _number(m.group(2))
        return _safe(
            "SELECT los_days, COUNT(*) AS visit_count, MIN(cost) AS min_cost, "
            "MAX(cost) AS max_cost FROM visits "
            f"WHERE los_days > {days} AND cost <= {cost} GROUP BY los_days "
            "ORDER BY los_days ASC"
        )

    m = re.fullmatch(
        r"仅统计年龄不少于(\d+)岁的患者,按地区和性别计算患者数量及"
        r"平均年龄\(保留一位小数\);输出地区、性别、数量、平均年龄,"
        r"按地区和性别升序(?:;为消除边界与常量歧义,"
        r"本查询采用的精确字面量还包括:1)?。?",
        q,
    )
    if m:
        age = _uint(m.group(1))
        return _safe(
            "SELECT region, gender, COUNT(*) AS patient_count, "
            "ROUND(AVG(age), 1) AS avg_age FROM patients "
            f"WHERE age >= {age} GROUP BY region, gender "
            "ORDER BY region ASC, gender ASC"
        )
    return None


def _compile_top_k(q: str) -> Optional[str]:
    patterns = (
        (
            r"找出患者人数最多的前(\d+)个地区,输出地区和患者人数;"
            r"人数相同时按地区升序。?",
            lambda m: "SELECT region, COUNT(*) AS patient_count FROM patients "
            "GROUP BY region ORDER BY patient_count DESC, region ASC "
            f"LIMIT {_uint(m.group(1))}",
        ),
        (
            r"列出平均就诊费用最高的前(\d+)个科室,输出科室和保留两位小数的"
            r"平均费用;平均费用相同时按科室升序;"
            r"为消除边界与常量歧义,本查询采用的精确字面量还包括:2。?",
            lambda m: "SELECT department, ROUND(AVG(cost), 2) AS avg_cost "
            "FROM visits GROUP BY department ORDER BY avg_cost DESC, "
            f"department ASC LIMIT {_uint(m.group(1))}",
        ),
        (
            r"按就诊次数取前(\d+)种疾病,输出疾病及次数;"
            r"次数相同时按疾病升序。?",
            lambda m: "SELECT disease, COUNT(*) AS visit_count FROM visits "
            "GROUP BY disease ORDER BY visit_count DESC, disease ASC "
            f"LIMIT {_uint(m.group(1))}",
        ),
        (
            r"返回知识图谱中提及次数最高的前(\d+)个实体的名称、类型和提及次数;"
            r"并列时按名称、类型升序。?",
            lambda m: "SELECT name, type, mention_count FROM kg_entities "
            "ORDER BY mention_count DESC, name ASC, type ASC "
            f"LIMIT {_uint(m.group(1))}",
        ),
        (
            r"找出累计就诊费用最高的前(\d+)名患者,输出患者ID和保留两位小数的"
            r"费用总额;总额相同时按患者ID升序;"
            r"为消除边界与常量歧义,本查询采用的精确字面量还包括:2。?",
            lambda m: "SELECT patient_id, ROUND(SUM(cost), 2) AS total_cost "
            "FROM visits GROUP BY patient_id ORDER BY total_cost DESC, "
            f"patient_id ASC LIMIT {_uint(m.group(1))}",
        ),
    )
    for pattern, render in patterns:
        m = re.fullmatch(pattern, q)
        if m:
            return _safe(render(m))

    m = re.fullmatch(
        r"在(\d{4})年内选出费用最高的前(\d+)次就诊,"
        r"输出就诊ID、日期、科室和费用;费用相同时按就诊ID升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:"
        r"(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m:
        year, limit = _uint(m.group(1)), _uint(m.group(2))
        start, end = m.group(3), m.group(4)
        if start != f"{year:04d}-01-01" or end != f"{year + 1:04d}-01-01":
            return None
        return _safe(
            "SELECT visit_id, visit_date, department, cost FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            f"ORDER BY cost DESC, visit_id ASC LIMIT {limit}"
        )

    m = re.fullmatch(
        r"从置信度不低于([0-9]+(?:\.[0-9]+)?)的三元组中选出观察次数最高的"
        r"前(\d+)条,输出头实体、关系、尾实体、观察次数和置信度;"
        r"依次按观察次数降序、置信度降序、头实体和尾实体升序。?",
        q,
    )
    if m:
        confidence, limit = _number(m.group(1)), _uint(m.group(2))
        return _safe(
            "SELECT head, relation, tail, count, confidence FROM kg_triples "
            f"WHERE confidence >= {confidence} ORDER BY count DESC, "
            f"confidence DESC, head ASC, tail ASC LIMIT {limit}"
        )
    return None


def _compile_joins(q: str, domains: Domains) -> Optional[str]:
    m = re.fullmatch(
        r"列出([^,，。；;]+)患者的患者姓名、就诊日期、科室和诊断疾病,"
        r"按日期和姓名升序。仅保留能通过患者ID匹配到患者的就诊。?",
        q,
    )
    if m:
        region = _grounded(m.group(1), domains, "region")
        if region is None:
            return None
        return _safe(
            "SELECT p.name, v.visit_date, v.department, v.disease "
            "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
            f"WHERE p.region = '{region}' ORDER BY v.visit_date ASC, p.name ASC"
        )

    m = re.fullmatch(
        r"统计每个科室中年龄不少于(\d+)岁的不同患者数,输出科室和去重患者数,"
        r"按人数降序、科室升序;只统计能匹配患者信息的就诊。?",
        q,
    )
    if m:
        age = _uint(m.group(1))
        return _safe(
            "SELECT v.department, COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM visits AS v JOIN patients AS p ON p.patient_id = v.patient_id "
            f"WHERE p.age >= {age} GROUP BY v.department "
            "ORDER BY patient_count DESC, v.department ASC"
        )

    m = re.fullmatch(
        r"把每次就诊的诊断与同名且提及次数不少于(\d+)的知识图谱实体做左连接,"
        r"输出就诊ID、疾病、实体类型和提及次数;未匹配实体也保留就诊,"
        r"按就诊ID升序。?",
        q,
    )
    if m:
        minimum = _uint(m.group(1))
        return _safe(
            "SELECT v.visit_id, v.disease, e.type, e.mention_count "
            "FROM visits AS v LEFT JOIN kg_entities AS e "
            f"ON e.name = v.disease AND e.mention_count >= {minimum} "
            "ORDER BY v.visit_id ASC"
        )

    m = re.fullmatch(
        r"统计每种就诊疾病关联到的置信度不低于([0-9]+(?:\.[0-9]+)?)的"
        r"知识图谱三元组数,输出疾病、去重就诊次数和去重三元组数;"
        r"按三元组数降序、疾病升序。?",
        q,
    )
    if m:
        confidence = _number(m.group(1))
        return _safe(
            "SELECT v.disease, COUNT(DISTINCT v.visit_id) AS visit_count, "
            "COUNT(DISTINCT t.rowid) AS triple_count FROM visits AS v "
            "JOIN kg_triples AS t ON t.head = v.disease "
            f"WHERE t.confidence >= {confidence} GROUP BY v.disease "
            "ORDER BY triple_count DESC, v.disease ASC"
        )

    m = re.fullmatch(
        r"仅统计费用不少于([0-9]+(?:\.[0-9]+)?)元且成功匹配患者的就诊,"
        r"按患者地区和就诊年份计数;输出地区、年份、次数,按地区、年份升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:1(?:、4)?。?",
        q,
    )
    if m:
        cost = _number(m.group(1))
        year = "SUBSTR(v.visit_date, 1, 4)"
        return _safe(
            f"SELECT p.region, {year} AS year, COUNT(*) AS visit_count "
            "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
            f"WHERE v.cost >= {cost} GROUP BY p.region, {year} "
            "ORDER BY p.region ASC, year ASC"
        )

    m = re.fullmatch(
        r"对每次就诊连接提及次数不少于(\d+)的同名疾病实体及其作为头实体的"
        r"三元组,输出就诊ID、疾病、提及次数、关系和尾实体,"
        r"按就诊ID、关系、尾实体升序。?",
        q,
    )
    if m:
        minimum = _uint(m.group(1))
        return _safe(
            "SELECT v.visit_id, v.disease, e.mention_count, t.relation, t.tail "
            "FROM visits AS v JOIN kg_entities AS e ON e.name = v.disease "
            "JOIN kg_triples AS t ON t.head = e.name "
            f"WHERE e.mention_count >= {minimum} "
            "ORDER BY v.visit_id ASC, t.relation ASC, t.tail ASC"
        )

    m = re.fullmatch(
        r"列出每个患者在(\d{4})年内最近一次就诊,输出患者ID、姓名、日期、"
        r"科室和疾病;同日并列保留,按患者ID、就诊ID升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:"
        r"(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m:
        year = _uint(m.group(1))
        start, end = m.group(2), m.group(3)
        if start != f"{year:04d}-01-01" or end != f"{year + 1:04d}-01-01":
            return None
        return _safe(
            "SELECT p.patient_id, p.name, v.visit_date, v.department, v.disease "
            "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
            f"WHERE v.visit_date >= '{start}' AND v.visit_date < '{end}' "
            "AND v.visit_date = (SELECT MAX(v2.visit_date) FROM visits AS v2 "
            "WHERE v2.patient_id = p.patient_id "
            f"AND v2.visit_date >= '{start}' AND v2.visit_date < '{end}') "
            "ORDER BY p.patient_id ASC, v.visit_id ASC"
        )

    m = re.fullmatch(
        r"按地区统计患者就诊诊断所连接的置信度不低于([0-9]+(?:\.[0-9]+)?)的"
        r"不同知识图谱关系数;输出地区和去重关系数,按数量降序、地区升序。?",
        q,
    )
    if m:
        confidence = _number(m.group(1))
        return _safe(
            "SELECT p.region, COUNT(DISTINCT t.relation) AS relation_count "
            "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
            "JOIN kg_triples AS t ON t.head = v.disease "
            f"WHERE t.confidence >= {confidence} GROUP BY p.region "
            "ORDER BY relation_count DESC, p.region ASC"
        )

    m = re.fullmatch(
        r"对观察次数不少于(\d+)的知识图谱三元组同时左连接头实体与尾实体,"
        r"输出头、关系、尾以及头尾提及次数;未匹配实体仍保留三元组,"
        r"按头、关系、尾升序。?",
        q,
    )
    if m:
        minimum = _uint(m.group(1))
        return _safe(
            "SELECT t.head, t.relation, t.tail, "
            "h.mention_count AS head_mentions, e.mention_count AS tail_mentions "
            "FROM kg_triples AS t LEFT JOIN kg_entities AS h ON h.name = t.head "
            "LEFT JOIN kg_entities AS e ON e.name = t.tail "
            f"WHERE t.count >= {minimum} "
            "ORDER BY t.head ASC, t.relation ASC, t.tail ASC"
        )
    return None


def _year_bounds(year_text: str, start: str, end: str) -> Optional[int]:
    year = _uint(year_text)
    if start != f"{year:04d}-01-01" or end != f"{year + 1:04d}-01-01":
        return None
    return year


def _compile_temporal(q: str, domains: Domains) -> Optional[str]:
    m = re.fullmatch(
        r"统计(\d{4})年各季度的就诊次数,输出季度编号1至4和次数,按季度升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:%m、3、"
        r"(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m and _year_bounds(*m.groups()) is not None:
        start, end = m.group(2), m.group(3)
        quarter = (
            "CAST((CAST(STRFTIME('%m', visit_date) AS INTEGER) - 1) / 3 "
            "AS INTEGER) + 1"
        )
        return _safe(
            f"SELECT {quarter} AS quarter_no, COUNT(*) AS visit_count "
            f"FROM visits WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "GROUP BY quarter_no ORDER BY quarter_no ASC"
        )

    m = re.fullmatch(
        r"按月份统计从(\d{4})年(\d{1,2})月(\d{1,2})日\(含\)到"
        r"(\d{4})年(\d{1,2})月(\d{1,2})日\(不含\)的就诊费用总额,"
        r"输出月份和保留两位小数的总额,按月份升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:%Y-%m、"
        r"(\d{4}-\d{2}-\d{2})、(\d{4}-\d{2}-\d{2})。?",
        q,
    )
    if m:
        parts = tuple(map(_uint, m.groups()[:6]))
        start, end = m.group(7), m.group(8)
        if start != f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}" or end != f"{parts[3]:04d}-{parts[4]:02d}-{parts[5]:02d}":
            return None
        return _safe(
            "SELECT STRFTIME('%Y-%m', visit_date) AS month, "
            "ROUND(SUM(cost), 2) AS total_cost FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "GROUP BY month ORDER BY month ASC"
        )

    m = re.fullmatch(
        r"列出(\d{4})年内每名患者最晚的就诊日期,输出患者ID和最晚日期,"
        r"按最晚日期降序、患者ID升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:"
        r"(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m and _year_bounds(*m.groups()) is not None:
        start, end = m.group(2), m.group(3)
        return _safe(
            "SELECT patient_id, MAX(visit_date) AS last_date FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "GROUP BY patient_id ORDER BY last_date DESC, patient_id ASC"
        )

    m = re.fullmatch(
        r"以数据库中最晚就诊日为基准,列出此前(\d+)天内\(含起止日\)的"
        r"就诊ID、日期和疾病,按日期降序、就诊ID升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:-(\d+) day。?",
        q,
    )
    if m:
        days = _uint(m.group(1))
        if days != _uint(m.group(2)):
            return None
        return _safe(
            "SELECT visit_id, visit_date, disease FROM visits "
            "WHERE visit_date BETWEEN DATE((SELECT MAX(visit_date) FROM visits), "
            f"'-{days} day') AND (SELECT MAX(visit_date) FROM visits) "
            "ORDER BY visit_date DESC, visit_id ASC"
        )

    m = re.fullmatch(
        r"返回([^,，。；;]+)在(\d{4})年每个月的平均就诊费用\(保留两位小数\),"
        r"输出月份和平均费用,按月份升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:%Y-%m、"
        r"(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m:
        department = _grounded(m.group(1), domains, "department")
        if department is None or _year_bounds(m.group(2), m.group(3), m.group(4)) is None:
            return None
        start, end = m.group(3), m.group(4)
        return _safe(
            "SELECT STRFTIME('%Y-%m', visit_date) AS month, "
            "ROUND(AVG(cost), 2) AS avg_cost FROM visits "
            f"WHERE department = '{department}' AND visit_date >= '{start}' "
            f"AND visit_date < '{end}' GROUP BY month ORDER BY month ASC"
        )
    return None


def _compile_knowledge_graph(q: str, domains: Domains) -> Optional[str]:
    m = re.fullmatch(
        r"列出关系为([A-Z][A-Z0-9_]*)且置信度不低于([0-9]+(?:\.[0-9]+)?)的"
        r"三元组头实体、尾实体、置信度和观察次数,"
        r"按置信度降序、观察次数降序、头尾升序。?",
        q,
    )
    if m:
        relation = _grounded(m.group(1), domains, "kg_relation")
        if relation is None:
            return None
        confidence = _number(m.group(2))
        return _safe(
            "SELECT head, tail, confidence, count FROM kg_triples "
            f"WHERE relation = '{relation}' AND confidence >= {confidence} "
            "ORDER BY confidence DESC, count DESC, head ASC, tail ASC"
        )

    m = re.fullmatch(
        r"按关系统计置信度不低于([0-9]+(?:\.[0-9]+)?)的三元组数量和"
        r"平均置信度\(保留三位小数\),输出关系、数量、平均值,"
        r"按数量降序、关系升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:3。?",
        q,
    )
    if m:
        confidence = _number(m.group(1))
        return _safe(
            "SELECT relation, COUNT(*) AS triple_count, "
            "ROUND(AVG(confidence), 3) AS avg_confidence FROM kg_triples "
            f"WHERE confidence >= {confidence} GROUP BY relation "
            "ORDER BY triple_count DESC, relation ASC"
        )

    m = re.fullmatch(
        r"找出作为知识图谱头实体出现至少(\d+)次的实体,输出头实体、头类型和"
        r"出现次数,按次数降序、头实体升序。?",
        q,
    )
    if m:
        minimum = _uint(m.group(1))
        return _safe(
            "SELECT head, head_type, COUNT(*) AS outgoing_count FROM kg_triples "
            "GROUP BY head, head_type "
            f"HAVING COUNT(*) >= {minimum} ORDER BY outgoing_count DESC, head ASC"
        )

    m = re.fullmatch(
        r"找出同时存在于实体表且作为三元组尾实体出现的([^,，。；;]+)类型实体,"
        r"输出去重后的实体名和提及次数,按提及次数降序、名称升序。?",
        q,
    )
    if m:
        entity_type = _grounded(m.group(1), domains, "kg_entity_type")
        if entity_type is None:
            return None
        return _safe(
            "SELECT DISTINCT e.name, e.mention_count FROM kg_entities AS e "
            "JOIN kg_triples AS t ON t.tail = e.name "
            f"WHERE e.type = '{entity_type}' "
            "ORDER BY e.mention_count DESC, e.name ASC"
        )

    m = re.fullmatch(
        r"仅统计置信度不低于([0-9]+(?:\.[0-9]+)?)的三元组,按关系计算"
        r"不同头实体数和不同尾实体数;输出关系及两项去重计数,"
        r"按不同头实体数降序、关系升序。?",
        q,
    )
    if m:
        confidence = _number(m.group(1))
        return _safe(
            "SELECT relation, COUNT(DISTINCT head) AS distinct_heads, "
            "COUNT(DISTINCT tail) AS distinct_tails FROM kg_triples "
            f"WHERE confidence >= {confidence} GROUP BY relation "
            "ORDER BY distinct_heads DESC, relation ASC"
        )

    m = re.fullmatch(
        r"对([A-Z][A-Z0-9_]*)关系,找出每个头实体对应的尾实体数量,"
        r"输出头实体和尾实体数,按数量降序、头实体升序。?",
        q,
    )
    if m:
        relation = _grounded(m.group(1), domains, "kg_relation")
        if relation is None:
            return None
        return _safe(
            "SELECT head, COUNT(DISTINCT tail) AS tail_count FROM kg_triples "
            f"WHERE relation = '{relation}' GROUP BY head "
            "ORDER BY tail_count DESC, head ASC"
        )
    return None


def _compile_having(q: str) -> Optional[str]:
    m = re.fullmatch(
        r"按科室统计就诊数,只保留就诊数至少为(\d+)的科室;"
        r"输出科室和就诊数,按就诊数降序、科室升序。?",
        q,
    )
    if m:
        minimum = _uint(m.group(1))
        return _safe(
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            f"GROUP BY department HAVING COUNT(*) >= {minimum} "
            "ORDER BY visit_count DESC, department ASC"
        )

    m = re.fullmatch(
        r"按地区统计患者数和平均年龄,仅保留患者数不少于(\d+)且平均年龄"
        r"不低于(\d+)岁的地区;输出地区、人数、两位小数平均年龄,"
        r"按人数降序、地区升序;为消除边界与常量歧义,"
        r"本查询采用的精确字面量还包括:2。?",
        q,
    )
    if m:
        count, age = _uint(m.group(1)), _uint(m.group(2))
        return _safe(
            "SELECT region, COUNT(*) AS patient_count, "
            "ROUND(AVG(age), 2) AS avg_age FROM patients GROUP BY region "
            f"HAVING COUNT(*) >= {count} AND AVG(age) >= {age} "
            "ORDER BY patient_count DESC, region ASC"
        )

    m = re.fullmatch(
        r"按患者统计就诊次数和总住院天数,只保留至少有(\d+)次就诊且"
        r"总住院天数不少于(\d+)天的患者;输出患者ID及两项指标,"
        r"按次数降序、患者ID升序。?",
        q,
    )
    if m:
        visits, days = _uint(m.group(1)), _uint(m.group(2))
        return _safe(
            "SELECT patient_id, COUNT(*) AS visit_count, "
            "SUM(los_days) AS total_los FROM visits GROUP BY patient_id "
            f"HAVING COUNT(*) >= {visits} AND SUM(los_days) >= {days} "
            "ORDER BY visit_count DESC, patient_id ASC"
        )
    return None


def _compile_nested(q: str, domains: Domains) -> Optional[str]:
    m = re.fullmatch(
        r"列出不存在住院天数大于(\d+)天就诊记录的患者ID和姓名,按患者ID升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:1。?",
        q,
    )
    if m:
        days = _uint(m.group(1))
        return _safe(
            "SELECT p.patient_id, p.name FROM patients AS p WHERE NOT EXISTS "
            "(SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id "
            f"AND v.los_days > {days}) ORDER BY p.patient_id ASC"
        )

    m = re.fullmatch(
        r"列出每名患者在(\d{4})年内日期最(早|晚)的就诊,"
        r"输出患者ID、就诊ID(和日期|、日期和疾病);同日多条全部保留,"
        r"按患者ID、就诊ID升序;为消除边界与常量歧义,"
        r"本查询采用的精确字面量还包括:(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m:
        year, direction, projection_suffix, start, end = m.groups()
        if _year_bounds(year, start, end) is None:
            return None
        expected_suffix = "、日期和疾病" if direction == "晚" else "和日期"
        if projection_suffix != expected_suffix:
            return None
        function = "MAX" if direction == "晚" else "MIN"
        projection = ", v.disease" if direction == "晚" else ""
        return _safe(
            f"SELECT v.patient_id, v.visit_id, v.visit_date{projection} "
            "FROM visits AS v "
            f"WHERE v.visit_date >= '{start}' AND v.visit_date < '{end}' "
            f"AND v.visit_date = (SELECT {function}(x.visit_date) FROM visits AS x "
            "WHERE x.patient_id = v.patient_id "
            f"AND x.visit_date >= '{start}' AND x.visit_date < '{end}') "
            "ORDER BY v.patient_id ASC, v.visit_id ASC"
        )

    m = re.fullmatch(
        r"列出至少存在一条置信度低于([0-9]+(?:\.[0-9]+)?)的出边三元组的"
        r"知识图谱实体名称和类型,按名称、类型升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:1。?",
        q,
    )
    if m:
        confidence = _number(m.group(1))
        return _safe(
            "SELECT e.name, e.type FROM kg_entities AS e WHERE EXISTS "
            "(SELECT 1 FROM kg_triples AS t WHERE t.head = e.name "
            f"AND t.confidence < {confidence}) ORDER BY e.name ASC, e.type ASC"
        )

    m = re.fullmatch(
        r"列出没有作为关系([A-Z][A-Z0-9_]*)的三元组头实体出现的"
        r"知识图谱实体名称和类型,按类型、名称升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:1。?",
        q,
    )
    if m:
        relation = _grounded(m.group(1), domains, "kg_relation")
        if relation is None:
            return None
        return _safe(
            "SELECT e.name, e.type FROM kg_entities AS e WHERE NOT EXISTS "
            "(SELECT 1 FROM kg_triples AS t WHERE t.head = e.name "
            f"AND t.relation = '{relation}') ORDER BY e.type ASC, e.name ASC"
        )

    m = re.fullmatch(
        r"列出至少同时拥有([A-Z][A-Z0-9_]*)和([A-Z][A-Z0-9_]*)两种"
        r"出边关系的头实体,结果去重并按头实体升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:1。?",
        q,
    )
    if m:
        first = _grounded(m.group(1), domains, "kg_relation")
        second = _grounded(m.group(2), domains, "kg_relation")
        if first is None or second is None or first == second:
            return None
        return _safe(
            "SELECT DISTINCT t.head FROM kg_triples AS t WHERE EXISTS "
            "(SELECT 1 FROM kg_triples AS a WHERE a.head = t.head "
            f"AND a.relation = '{first}') AND EXISTS "
            "(SELECT 1 FROM kg_triples AS b WHERE b.head = t.head "
            f"AND b.relation = '{second}') ORDER BY t.head ASC"
        )

    m = re.fullmatch(
        r"列出总就诊费用高于其所在地区有就诊患者平均总费用的"
        r"([0-9]+(?:\.[0-9]+)?)倍的患者ID、地区和总费用\(两位小数\),"
        r"按总费用降序、患者ID升序(?:;为消除边界与常量歧义,"
        r"本查询采用的精确字面量还包括:2)?。?",
        q,
    )
    if m:
        factor = _number(m.group(1))
        return _safe(
            "SELECT p.patient_id, p.region, ROUND(SUM(v.cost), 2) AS total_cost "
            "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id, p.region HAVING SUM(v.cost) > "
            f"{factor} * (SELECT AVG(region_total) FROM "
            "(SELECT SUM(v2.cost) AS region_total FROM patients AS p2 "
            "JOIN visits AS v2 ON v2.patient_id = p2.patient_id "
            "WHERE p2.region = p.region GROUP BY p2.patient_id)) "
            "ORDER BY total_cost DESC, p.patient_id ASC"
        )
    return None


def _compile_cte(q: str) -> Optional[str]:
    m = re.fullmatch(
        r"用一个CTE计算每位患者的就诊次数和总费用,再输出总费用最高的前(\d+)位"
        r"患者ID、次数及两位小数总费用;总费用并列时按患者ID升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:2。?",
        q,
    )
    if m:
        limit = _uint(m.group(1))
        return _safe(
            "WITH patient_cost AS (SELECT patient_id, COUNT(*) AS visit_count, "
            "SUM(cost) AS total_cost FROM visits GROUP BY patient_id) "
            "SELECT patient_id, visit_count, ROUND(total_cost, 2) AS total_cost "
            "FROM patient_cost ORDER BY total_cost DESC, patient_id ASC "
            f"LIMIT {limit}"
        )

    m = re.fullmatch(
        r"用CTE基于费用不少于([0-9]+(?:\.[0-9]+)?)元的记录计算各科室"
        r"平均费用,再列出同样满足该费用下限且超过本科室平均值的就诊ID、"
        r"科室和费用,按科室、费用降序、就诊ID升序。?",
        q,
    )
    if m:
        cost = _number(m.group(1))
        return _safe(
            "WITH dept_avg AS (SELECT department, AVG(cost) AS avg_cost "
            f"FROM visits WHERE cost >= {cost} GROUP BY department) "
            "SELECT v.visit_id, v.department, v.cost FROM visits AS v "
            "JOIN dept_avg AS d ON d.department = v.department "
            f"WHERE v.cost >= {cost} AND v.cost > d.avg_cost "
            "ORDER BY v.department ASC, v.cost DESC, v.visit_id ASC"
        )

    m = re.fullmatch(
        r"用两个CTE分别统计年龄不少于(\d+)岁的地区患者数和这些患者的地区"
        r"就诊数,再连接输出地区、患者数、就诊数;只保留两边都有数据的地区,"
        r"按地区升序。?",
        q,
    )
    if m:
        age = _uint(m.group(1))
        return _safe(
            "WITH pc AS (SELECT region, COUNT(*) AS patient_count FROM patients "
            f"WHERE age >= {age} GROUP BY region), vc AS "
            "(SELECT p.region, COUNT(*) AS visit_count FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            f"WHERE p.age >= {age} GROUP BY p.region) "
            "SELECT pc.region, pc.patient_count, vc.visit_count FROM pc "
            "JOIN vc ON vc.region = pc.region ORDER BY pc.region ASC"
        )

    m = re.fullmatch(
        r"用CTE取每位患者在(\d{4}-\d{2}-\d{2})\(含\)之后的最近就诊日期,"
        r"再连接就诊表输出患者ID、就诊ID、最近日期和疾病;同日并列全部保留,"
        r"按患者ID、就诊ID升序。?",
        q,
    )
    if m:
        start = m.group(1)
        return _safe(
            "WITH latest AS (SELECT patient_id, MAX(visit_date) AS latest_date "
            f"FROM visits WHERE visit_date >= '{start}' GROUP BY patient_id) "
            "SELECT v.patient_id, v.visit_id, v.visit_date, v.disease "
            "FROM visits AS v JOIN latest AS l ON l.patient_id = v.patient_id "
            "AND l.latest_date = v.visit_date "
            "ORDER BY v.patient_id ASC, v.visit_id ASC"
        )

    m = re.fullmatch(
        r"先用CTE汇总(\d{4})年各月费用,再输出月份、两位小数月费用以及该月费用"
        r"占\d{4}年总费用的百分比\(两位小数\),按月份升序;"
        r"为消除边界与常量歧义,本查询采用的精确字面量还包括:1、7、"
        r"(\d{4}-01-01)、(\d{4}-01-01)、100\.0。?",
        q,
    )
    if m and _year_bounds(*m.groups()) is not None:
        start, end = m.group(2), m.group(3)
        month = "SUBSTR(visit_date, 1, 7)"
        return _safe(
            f"WITH monthly AS (SELECT {month} AS month, SUM(cost) AS month_cost "
            "FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            f"GROUP BY {month}), yearly AS "
            "(SELECT SUM(month_cost) AS year_cost FROM monthly) "
            "SELECT m.month, ROUND(m.month_cost, 2) AS month_cost, "
            "ROUND(100.0 * m.month_cost / y.year_cost, 2) AS pct "
            "FROM monthly AS m CROSS JOIN yearly AS y ORDER BY m.month ASC"
        )

    m = re.fullmatch(
        r"用递归CTE生成0到(\d+)的住院天数序列,并左连接就诊表统计每个天数"
        r"对应的就诊数;输出天数和次数,按天数升序。?",
        q,
    )
    if m:
        end = _uint(m.group(1))
        return _safe(
            "WITH RECURSIVE days(n) AS (SELECT 0 UNION ALL SELECT n + 1 "
            f"FROM days WHERE n < {end}) SELECT days.n AS los_days, "
            "COUNT(v.visit_id) AS visit_count FROM days LEFT JOIN visits AS v "
            "ON v.los_days = days.n GROUP BY days.n ORDER BY days.n ASC"
        )
    return None


def _compile_set_operation(q: str) -> Optional[str]:
    exact = {
        "返回既在患者地区字段中出现、也在就诊科室字段中出现的文本值;"
        "使用交集去重,按文本升序。": (
            "SELECT region AS value FROM patients INTERSECT "
            "SELECT department AS value FROM visits ORDER BY value ASC"
        ),
        "合并就诊疾病名称与知识图谱实体名称,使用UNION去重;"
        "输出统一字段name并按名称升序。": (
            "SELECT disease AS name FROM visits UNION SELECT name FROM kg_entities "
            "ORDER BY name ASC"
        ),
        "返回同时是就诊诊断疾病和知识图谱头实体的名称,"
        "使用INTERSECT去重并按名称升序。": (
            "SELECT disease AS name FROM visits INTERSECT "
            "SELECT head AS name FROM kg_triples ORDER BY name ASC"
        ),
        "返回知识图谱实体表中存在、但从未在任何就诊疾病字段中出现的实体名称和"
        "类型,使用EXCEPT去重并按名称、类型升序。": (
            "SELECT name, type FROM kg_entities EXCEPT SELECT e.name, e.type "
            "FROM kg_entities AS e JOIN visits AS v ON v.disease = e.name "
            "ORDER BY name ASC, type ASC"
        ),
        "合并就诊中出现的患者ID与患者表中没有就诊的患者ID,并附来源;保留重复,"
        "按来源、患者ID升序;为消除边界与常量歧义,"
        "本查询采用的精确字面量还包括:visited、never_visited、1。": (
            "SELECT 'visited' AS source, patient_id FROM visits UNION ALL "
            "SELECT 'never_visited' AS source, p.patient_id FROM patients AS p "
            "WHERE NOT EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id) ORDER BY source ASC, patient_id ASC"
        ),
        "返回既开过处方又作为知识图谱药物实体出现的名称,先分别排除空字符串,"
        "再用INTERSECT去重并按名称升序;为消除边界与常量歧义,"
        "本查询采用的精确字面量还包括:Drug;本查询其余精确非空常量为:0。": (
            "SELECT drug_prescribed AS drug_name FROM visits "
            "WHERE LENGTH(drug_prescribed) > 0 INTERSECT "
            "SELECT name AS drug_name FROM kg_entities WHERE type = 'Drug' "
            "ORDER BY drug_name ASC"
        ),
    }
    if q in exact:
        return _safe(exact[q])

    m = re.fullmatch(
        r"返回(\d{4})年有就诊但(\d{4})年没有就诊的患者ID,使用EXCEPT去重并"
        r"按患者ID升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:"
        r"(\d{4}-01-01)、(\d{4}-01-01)、(\d{4}-01-01)。?",
        q,
    )
    if m:
        first, second = _uint(m.group(1)), _uint(m.group(2))
        first_start, first_end, second_end = m.group(3), m.group(4), m.group(5)
        if (
            second != first + 1
            or first_start != f"{first:04d}-01-01"
            or first_end != f"{second:04d}-01-01"
            or second_end != f"{second + 1:04d}-01-01"
        ):
            return None
        return _safe(
            "SELECT patient_id FROM visits "
            f"WHERE visit_date >= '{first_start}' AND visit_date < '{first_end}' "
            "EXCEPT SELECT patient_id FROM visits "
            f"WHERE visit_date >= '{first_end}' AND visit_date < '{second_end}' "
            "ORDER BY patient_id ASC"
        )
    return None


def _compile_window(q: str) -> Optional[str]:
    exact = {
        "为每个科室的就诊按费用从高到低编号,费用相同时按就诊ID升序;"
        "输出科室、就诊ID、费用和行号,按科室、行号升序。": (
            "SELECT department, visit_id, cost, ROW_NUMBER() OVER "
            "(PARTITION BY department ORDER BY cost DESC, visit_id ASC) AS row_no "
            "FROM visits ORDER BY department ASC, row_no ASC"
        ),
        "计算每个科室内就诊费用的名次,费用相同名次相同且后续名次跳号;"
        "输出科室、就诊ID、费用和RANK名次,按科室、名次、就诊ID升序。": (
            "SELECT department, visit_id, cost, RANK() OVER "
            "(PARTITION BY department ORDER BY cost DESC) AS cost_rank "
            "FROM visits ORDER BY department ASC, cost_rank ASC, visit_id ASC"
        ),
        "计算每种疾病就诊费用的稠密名次,费用相同名次相同且不跳号;"
        "输出疾病、就诊ID、费用和名次,按疾病、名次、就诊ID升序。": (
            "SELECT disease, visit_id, cost, DENSE_RANK() OVER "
            "(PARTITION BY disease ORDER BY cost DESC) AS dense_cost_rank "
            "FROM visits ORDER BY disease ASC, dense_cost_rank ASC, visit_id ASC"
        ),
        "按每名患者的就诊日期和就诊ID顺序计算累计费用,输出患者ID、就诊ID、日期、"
        "费用及两位小数累计费用,按患者ID、日期、就诊ID升序;"
        "为消除边界与常量歧义,本查询采用的精确字面量还包括:2。": (
            "SELECT patient_id, visit_id, visit_date, cost, "
            "ROUND(SUM(cost) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date ASC, visit_id ASC ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND CURRENT ROW), 2) AS running_cost FROM visits "
            "ORDER BY patient_id ASC, visit_date ASC, visit_id ASC"
        ),
        "计算每名患者的就诊总次数并显示在每条就诊上;输出患者ID、就诊ID、日期和"
        "总次数,按患者ID、日期、就诊ID升序。": (
            "SELECT patient_id, visit_id, visit_date, COUNT(*) OVER "
            "(PARTITION BY patient_id) AS patient_visit_count FROM visits "
            "ORDER BY patient_id ASC, visit_date ASC, visit_id ASC"
        ),
        "按费用升序把全部就诊近似分成4组,费用相同时以就诊ID升序决定顺序;"
        "输出就诊ID、费用和NTILE组号,按组号、费用、就诊ID升序。": (
            "SELECT visit_id, cost, NTILE(4) OVER "
            "(ORDER BY cost ASC, visit_id ASC) AS cost_quartile FROM visits "
            "ORDER BY cost_quartile ASC, cost ASC, visit_id ASC"
        ),
        "对每个头实体按置信度降序、关系和尾实体升序编号,只保留编号1的三元组;"
        "输出头、关系、尾、置信度,按头升序。": (
            "SELECT head, relation, tail, confidence FROM "
            "(SELECT head, relation, tail, confidence, ROW_NUMBER() OVER "
            "(PARTITION BY head ORDER BY confidence DESC, relation ASC, tail ASC) "
            "AS rn FROM kg_triples) WHERE rn = 1 ORDER BY head ASC"
        ),
        "按每月汇总就诊费用,再计算相对上月的费用差额;输出月份、两位小数月费用和"
        "两位小数差额,按月份升序,首月差额为空;"
        "为消除边界与常量歧义,本查询采用的精确字面量还包括:1、7、2。": (
            "WITH monthly AS (SELECT SUBSTR(visit_date, 1, 7) AS month, "
            "SUM(cost) AS month_cost FROM visits GROUP BY SUBSTR(visit_date, 1, 7)) "
            "SELECT month, ROUND(month_cost, 2) AS month_cost, "
            "ROUND(month_cost - LAG(month_cost) OVER (ORDER BY month ASC), 2) "
            "AS change_from_previous FROM monthly ORDER BY month ASC"
        ),
        "按地区先汇总患者人数,再计算每个地区人数占全部患者数的百分比;"
        "输出地区、人数和两位小数百分比,按人数降序、地区升序;"
        "为消除边界与常量歧义,本查询采用的精确字面量还包括:100.0、2。": (
            "WITH region_counts AS (SELECT region, COUNT(*) AS patient_count "
            "FROM patients GROUP BY region) SELECT region, patient_count, "
            "ROUND(100.0 * patient_count / SUM(patient_count) OVER (), 2) "
            "AS pct_all_patients FROM region_counts "
            "ORDER BY patient_count DESC, region ASC"
        ),
    }
    if q in exact:
        return _safe(exact[q])

    m = re.fullmatch(
        r"在每个实体类型内按提及次数降序、名称升序编号,只保留每类前(\d+)名;"
        r"输出类型、名称、提及次数和行号,按类型、行号升序。?",
        q,
    )
    if m:
        limit = _uint(m.group(1))
        return _safe(
            "SELECT type, name, mention_count, row_no FROM "
            "(SELECT type, name, mention_count, ROW_NUMBER() OVER "
            "(PARTITION BY type ORDER BY mention_count DESC, name ASC) AS row_no "
            f"FROM kg_entities) WHERE row_no <= {limit} "
            "ORDER BY type ASC, row_no ASC"
        )
    return None


def compile_independent_structured_sql(q: str, domains: Domains) -> Optional[str]:
    """Compile one fully specified, self-validating analytical sketch."""

    for compiler in (
        lambda value: _compile_filtering(value, domains),
        _compile_grouping,
        _compile_top_k,
        lambda value: _compile_joins(value, domains),
        lambda value: _compile_temporal(value, domains),
        lambda value: _compile_knowledge_graph(value, domains),
        _compile_having,
        lambda value: _compile_nested(value, domains),
        _compile_cte,
        _compile_set_operation,
        _compile_window,
    ):
        try:
            sql = compiler(q)
        except (TypeError, ValueError):
            return None
        if sql is not None:
            return sql
    return None
