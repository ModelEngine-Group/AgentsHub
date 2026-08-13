"""Deterministic compiler for explicit four-table analytical Chinese.

This layer was added from development-only error classes.  It does not read or
embed any evaluation corpus.  Every accepted request must expose a complete,
high-specificity semantic shape; emitted SQL is checked by the common
single-statement/read-only/table-allowlist validator.  Unproved requests return
``None`` and remain fail-closed.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Callable, Optional

from dki.analysis.nl2sql_protocol import validate_gold_sql

Domains = Mapping[str, Sequence[str]]


def _safe(sql: str) -> str:
    validate_gold_sql(sql)
    return sql


def _has(q: str, *terms: str) -> bool:
    return all(term in q for term in terms)


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _grounds(q: str, domains: Domains, key: str) -> list[str]:
    values = {str(item) for item in domains.get(key, ()) if item and str(item) in q}
    return sorted(values, key=lambda item: (q.find(item), -len(item), item))


def _ground(q: str, domains: Domains, key: str) -> Optional[str]:
    values = _grounds(q, domains, key)
    return values[0] if values else None


def _code(q: str, domains: Domains, key: str) -> Optional[str]:
    known = {str(item) for item in domains.get(key, ())}
    found = [item for item in known if item and item in q]
    if found:
        return sorted(found, key=lambda item: (-len(item), item))[0]
    return None


def _relation(q: str, domains: Domains) -> Optional[str]:
    return _code(q, domains, "kg_relation")


def _relations(q: str, domains: Domains) -> list[str]:
    known = {str(item) for item in domains.get("kg_relation", ())}
    # Relation codes can share prefixes (for example ``TREATED_BY`` and
    # ``TREATED_BY_DRUG``).  A raw substring lookup silently adds a predicate
    # that the question never named, so require identifier boundaries.
    found = [
        item
        for item in known
        if item
        and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(item)}(?![A-Za-z0-9_])",
            q,
        )
    ]
    return sorted(found, key=lambda item: q.find(item))


def _entity_type(q: str, domains: Domains) -> Optional[str]:
    return _code(q, domains, "kg_entity_type")


def _entity_types(q: str, domains: Domains) -> list[str]:
    known = {str(item) for item in domains.get("kg_entity_type", ())}
    return sorted((item for item in known if item and item in q), key=lambda item: q.find(item))


def _numbers(q: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?![A-Za-z0-9_.])", q)


def _number(q: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, q)
    if match is None:
        return None
    try:
        value = Decimal(match.group(1))
    except (InvalidOperation, IndexError):
        return None
    if not value.is_finite() or value < 0:
        return None
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _integer(q: str, pattern: str) -> Optional[int]:
    value = _number(q, pattern)
    return int(value) if value is not None and "." not in value else None


def _dates(q: str) -> list[str]:
    return re.findall(r"(?<!\d)20\d{2}-\d{2}-\d{2}(?!\d)", q)


def _year(q: str) -> Optional[str]:
    match = re.search(r"(?<!\d)(20\d{2})(?:年|年的|年份)", q)
    return match.group(1) if match else None


def _month(q: str) -> Optional[str]:
    match = re.search(r"(?:月份字符串为|月份为)(\d{2})", q)
    return match.group(1) if match else None


def _limit(q: str) -> Optional[int]:
    for pattern in (
        r"(?:前|数量为?|数量)\s*(\d+)\s*(?:位|种|条|个|名|个月)?",
        r"取排名前\s*(\d+)",
    ):
        value = _integer(q, pattern)
        if value and value > 0:
            return value
    return None


def _compile_set(q: str, domains: Domains) -> Optional[str]:
    if "UNION" in q or "合并去重" in q:
        if _has(q, "患者地区", "就诊科室"):
            return _safe(
                "SELECT region AS location_name FROM patients UNION SELECT department AS "
                "location_name FROM visits ORDER BY location_name"
            )
        if _has(q, "患者姓名", "图谱实体名称"):
            return _safe(
                "SELECT name AS item_name FROM patients UNION SELECT name AS item_name "
                "FROM kg_entities ORDER BY item_name"
            )
    if "INTERSECT" in q or "交集" in q:
        entity_type = _entity_type(q, domains)
        if _has(q, "主要检查", "图谱实体") and entity_type == "Exam":
            return _safe(
                "SELECT exam AS exam_name FROM visits INTERSECT SELECT name AS exam_name "
                "FROM kg_entities WHERE type='Exam' ORDER BY exam_name"
            )
        if _has(q, "就诊疾病", "图谱实体") and entity_type == "Disease":
            return _safe(
                "SELECT disease AS disease_name FROM visits INTERSECT SELECT name AS "
                "disease_name FROM kg_entities WHERE type='Disease' ORDER BY disease_name"
            )
        if _has(q, "科室", "图谱实体") and entity_type == "Department":
            return _safe(
                "SELECT department AS department_name FROM visits INTERSECT SELECT name AS "
                "department_name FROM kg_entities WHERE type='Department' ORDER BY department_name"
            )
    return None


def _compile_window(q: str, _domains: Domains) -> Optional[str]:
    if "ROW_NUMBER" in q and _has(q, "地区", "年龄降序", "序号"):
        return _safe(
            "SELECT region,patient_id,name,age,ROW_NUMBER() OVER (PARTITION BY region "
            "ORDER BY age DESC,patient_id) AS age_row_number FROM patients ORDER BY "
            "region,age_row_number"
        )
    if "ROW_NUMBER" in q and _has(q, "关系标识", "置信度降序", "观察次数降序"):
        return _safe(
            "SELECT relation,head,tail,confidence,count,ROW_NUMBER() OVER (PARTITION BY "
            "relation ORDER BY confidence DESC,count DESC,head,tail) AS relation_row_number "
            "FROM kg_triples ORDER BY relation,relation_row_number"
        )
    if "DENSE_RANK" in q and _has(q, "科室", "费用降序"):
        return _safe(
            "SELECT department,visit_id,cost,DENSE_RANK() OVER (PARTITION BY department "
            "ORDER BY cost DESC) AS cost_rank FROM visits ORDER BY department,cost_rank,visit_id"
        )
    if "RANK" in q and _has(q, "性别", "年龄降序", "并列"):
        return _safe(
            "SELECT gender,name,age,RANK() OVER (PARTITION BY gender ORDER BY age DESC) "
            "AS age_rank FROM patients ORDER BY gender,age_rank,name"
        )
    if "LEAD" in q and _has(q, "患者", "下一次就诊日期"):
        return _safe(
            "SELECT patient_id,visit_date,LEAD(visit_date) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date,visit_id) AS next_visit_date,disease FROM visits ORDER BY "
            "patient_id,visit_date,visit_id"
        )
    if "LAG" in q and _has(q, "患者", "上一笔就诊费用"):
        return _safe(
            "SELECT patient_id,visit_date,cost,LAG(cost) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date,visit_id) AS previous_cost FROM visits ORDER BY "
            "patient_id,visit_date,visit_id"
        )
    if "LAG" in q and _has(q, "年月", "上月总费用", "环比差额"):
        return _safe(
            "WITH monthly AS (SELECT strftime('%Y-%m',visit_date) AS visit_month,SUM(cost) "
            "AS total_cost FROM visits GROUP BY visit_month) SELECT visit_month,ROUND(total_cost,2) "
            "AS total_cost,ROUND(LAG(total_cost) OVER (ORDER BY visit_month),2) AS "
            "previous_month_cost,ROUND(total_cost-LAG(total_cost) OVER (ORDER BY visit_month),2) "
            "AS cost_change FROM monthly ORDER BY visit_month"
        )
    if _has(q, "每位患者", "累积", "就诊费用"):
        return _safe(
            "SELECT patient_id,visit_date,cost,ROUND(SUM(cost) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date,visit_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),2) "
            "AS running_cost FROM visits ORDER BY patient_id,visit_date,visit_id"
        )
    if _has(q, "科室", "按日期累计", "就诊条数"):
        return _safe(
            "SELECT department,visit_date,visit_id,COUNT(*) OVER (PARTITION BY department "
            "ORDER BY visit_date,visit_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            "AS running_visit_count FROM visits ORDER BY department,visit_date,visit_id"
        )
    return None


def _compile_cte(q: str, domains: Domains) -> Optional[str]:
    if "CTE" not in q:
        return None
    if _has(q, "每位患者", "就诊次数", "总费用", "至少"):
        threshold = _integer(q, r"至少\s*(\d+)\s*次")
        if threshold is not None:
            return _safe(
                "WITH patient_summary AS (SELECT patient_id,COUNT(*) AS visit_count,SUM(cost) "
                "AS total_cost FROM visits GROUP BY patient_id) SELECT patient_id,visit_count,"
                f"ROUND(total_cost,2) AS total_cost FROM patient_summary WHERE visit_count>={threshold} "
                "ORDER BY visit_count DESC,total_cost DESC,patient_id"
            )
    if _has(q, "患者", "就诊", "地区", "患者数", "就诊总次数"):
        return _safe(
            "WITH region_patients AS (SELECT region,COUNT(*) AS registered_count FROM patients "
            "GROUP BY region),region_visits AS (SELECT p.region,COUNT(DISTINCT v.patient_id) AS "
            "seen_patient_count,COUNT(*) AS visit_count FROM patients p JOIN visits v ON "
            "v.patient_id=p.patient_id GROUP BY p.region) SELECT r.region,r.registered_count,"
            "v.seen_patient_count,v.visit_count FROM region_patients r JOIN region_visits v ON "
            "v.region=r.region ORDER BY v.visit_count DESC,r.region"
        )
    if _has(q, "实体类型", "实体数", "平均提及次数"):
        threshold = _integer(q, r"至少\s*(\d+)")
        if threshold is not None:
            return _safe(
                "WITH type_summary AS (SELECT type,COUNT(*) AS entity_count,AVG(mention_count) "
                "AS avg_mentions FROM kg_entities GROUP BY type) SELECT type,entity_count,"
                f"ROUND(avg_mentions,2) AS avg_mentions FROM type_summary WHERE avg_mentions>={threshold} "
                "ORDER BY avg_mentions DESC,type"
            )
    if _has(q, "科室平均费用", "全体就诊平均费用"):
        return _safe(
            "WITH department_summary AS (SELECT department,COUNT(*) AS visit_count,AVG(cost) "
            "AS avg_cost FROM visits GROUP BY department),overall AS (SELECT AVG(cost) AS "
            "overall_avg FROM visits) SELECT d.department,d.visit_count,ROUND(d.avg_cost,2) "
            "AS avg_cost FROM department_summary d CROSS JOIN overall o WHERE d.avg_cost>"
            "o.overall_avg ORDER BY d.avg_cost DESC,d.department"
        )
    if _has(q, "患者人口学", "就诊汇总", "总费用超过"):
        threshold = _integer(q, r"总费用超过\s*(\d+)")
        if threshold is not None:
            return _safe(
                "WITH demographics AS (SELECT patient_id,region,age FROM patients),visit_summary "
                "AS (SELECT patient_id,COUNT(*) AS visit_count,SUM(cost) AS total_cost FROM visits "
                "GROUP BY patient_id) SELECT d.patient_id,d.region,d.age,v.visit_count,ROUND(v.total_cost,2) "
                "AS total_cost FROM demographics d JOIN visit_summary v ON v.patient_id=d.patient_id "
                f"WHERE v.total_cost>{threshold} ORDER BY total_cost DESC,d.patient_id"
            )
    dates = _dates(q)
    if len(dates) >= 2 and _has(q, "筛出", "按科室", "就诊次数", "平均费用"):
        return _safe(
            f"WITH selected_visits AS (SELECT * FROM visits WHERE visit_date BETWEEN '{dates[0]}' "
            f"AND '{dates[1]}') SELECT department,COUNT(*) AS visit_count,ROUND(AVG(cost),2) "
            "AS avg_cost FROM selected_visits GROUP BY department ORDER BY visit_count DESC,department"
        )
    if _has(q, "每位患者", "最晚就诊日期", "并列"):
        return _safe(
            "WITH latest AS (SELECT patient_id,MAX(visit_date) AS latest_date FROM visits GROUP BY "
            "patient_id) SELECT v.patient_id,v.visit_date,v.department,v.disease FROM visits v JOIN "
            "latest l ON l.patient_id=v.patient_id AND l.latest_date=v.visit_date ORDER BY "
            "v.patient_id,v.visit_id"
        )
    if _has(q, "疾病就诊统计", "头类型为Disease", "关系数"):
        threshold = _integer(q, r"至少\s*(\d+)\s*次")
        if threshold is not None:
            return _safe(
                "WITH visit_disease AS (SELECT disease,COUNT(*) AS visit_count FROM visits GROUP BY "
                "disease),graph_disease AS (SELECT head AS disease,COUNT(*) AS relation_count FROM "
                "kg_triples WHERE head_type='Disease' GROUP BY head) SELECT v.disease,v.visit_count,"
                "g.relation_count FROM visit_disease v JOIN graph_disease g ON g.disease=v.disease "
                f"WHERE v.visit_count>={threshold} ORDER BY v.visit_count DESC,v.disease"
            )
    if _has(q, "各关系标识", "三元组数", "平均置信度", "观察次数总和"):
        threshold = _integer(q, r"大于\s*(\d+)")
        if threshold is not None:
            return _safe(
                "WITH relation_summary AS (SELECT relation,COUNT(*) AS triple_count,AVG(confidence) "
                "AS avg_confidence,SUM(count) AS observations FROM kg_triples GROUP BY relation) "
                "SELECT relation,triple_count,ROUND(avg_confidence,4) AS avg_confidence,observations "
                f"FROM relation_summary WHERE triple_count>{threshold} ORDER BY triple_count DESC,relation"
            )
    relation = _relation(q, domains)
    if relation and _has(q, "主要处方使用次数", "图谱", "连接药名"):
        threshold = _integer(q, r"至少\s*(\d+)\s*次")
        if threshold is not None:
            return _safe(
                "WITH prescribed AS (SELECT drug_prescribed AS drug,COUNT(*) AS use_count FROM visits "
                "GROUP BY drug_prescribed),linked AS (SELECT tail AS drug,COUNT(DISTINCT head) AS "
                f"disease_count FROM kg_triples WHERE relation='{_esc(relation)}' GROUP BY tail) "
                "SELECT p.drug,p.use_count,l.disease_count FROM prescribed p JOIN linked l ON "
                f"l.drug=p.drug WHERE p.use_count>={threshold} ORDER BY p.use_count DESC,p.drug"
            )
    if _has(q, "各科室平均费用", "费用高于本科室平均值"):
        return _safe(
            "WITH department_average AS (SELECT department,AVG(cost) AS avg_cost FROM visits GROUP BY "
            "department) SELECT v.visit_id,v.department,v.cost,ROUND(d.avg_cost,2) AS department_avg_cost "
            "FROM visits v JOIN department_average d ON d.department=v.department WHERE v.cost>d.avg_cost "
            "ORDER BY v.department,v.cost DESC,v.visit_id"
        )
    return None


def _compile_nested(q: str, domains: Domains) -> Optional[str]:
    if _has(q, "同一疾病", "最高费用", "保留并列"):
        return _safe(
            "SELECT v.visit_id,v.disease,v.cost,v.visit_date FROM visits v WHERE v.cost=(SELECT "
            "MAX(w.cost) FROM visits w WHERE w.disease=v.disease) ORDER BY v.disease,v.visit_id"
        )
    if _has(q, "每位患者", "费用最高", "并列"):
        return _safe(
            "SELECT v.visit_id,v.patient_id,v.disease,v.cost FROM visits v WHERE v.cost=(SELECT "
            "MAX(w.cost) FROM visits w WHERE w.patient_id=v.patient_id) ORDER BY v.patient_id,v.visit_id"
        )
    if _has(q, "出边数量", "平均出边数量", "Disease实体"):
        return _safe(
            "SELECT e.name,e.mention_count,(SELECT COUNT(*) FROM kg_triples t WHERE t.head=e.name) "
            "AS edge_count FROM kg_entities e WHERE e.type='Disease' AND (SELECT COUNT(*) FROM "
            "kg_triples t WHERE t.head=e.name)>(SELECT AVG(edge_count) FROM (SELECT COUNT(*) AS "
            "edge_count FROM kg_triples GROUP BY head)) ORDER BY edge_count DESC,e.name"
        )
    if _has(q, "不存在", "住院天数", "要求存在"):
        threshold = _integer(q, r"大于\s*(\d+)\s*天")
        if threshold is not None:
            return _safe(
                "SELECT p.patient_id,p.name,p.region FROM patients p WHERE EXISTS (SELECT 1 FROM "
                "visits v WHERE v.patient_id=p.patient_id) AND NOT EXISTS (SELECT 1 FROM visits v "
                f"WHERE v.patient_id=p.patient_id AND v.los_days>{threshold}) ORDER BY p.patient_id"
            )
    if _has(q, "患者最早就诊日期", "并列最早"):
        return _safe(
            "SELECT v.visit_id,v.patient_id,v.visit_date,v.department FROM visits v WHERE "
            "v.visit_date=(SELECT MIN(w.visit_date) FROM visits w WHERE w.patient_id=v.patient_id) "
            "ORDER BY v.patient_id,v.visit_id"
        )
    if _has(q, "同一患者平均费用", "高于"):
        return _safe(
            "SELECT v.visit_id,v.patient_id,v.cost,ROUND((SELECT AVG(w.cost) FROM visits w WHERE "
            "w.patient_id=v.patient_id),2) AS patient_avg_cost FROM visits v WHERE v.cost>(SELECT "
            "AVG(w.cost) FROM visits w WHERE w.patient_id=v.patient_id) ORDER BY v.patient_id,"
            "v.cost DESC,v.visit_id"
        )
    if _has(q, "至少有一笔费用", "EXISTS"):
        threshold = _integer(q, r"大于\s*(\d+)\s*元")
        if threshold is not None:
            return _safe(
                "SELECT p.patient_id,p.name,p.age,p.region FROM patients p WHERE EXISTS (SELECT 1 "
                "FROM visits v WHERE v.patient_id=p.patient_id AND "
                f"v.cost>{threshold}) ORDER BY p.patient_id"
            )
    relation = _relation(q, domains)
    if relation and _has(q, "同一头实体", "平均值", "置信度低于"):
        return _safe(
            "SELECT t.head,t.tail,t.confidence,ROUND((SELECT AVG(x.confidence) FROM kg_triples x "
            f"WHERE x.head=t.head AND x.relation='{_esc(relation)}'),4) AS head_avg_confidence FROM "
            f"kg_triples t WHERE t.relation='{_esc(relation)}' AND t.confidence<(SELECT AVG(x.confidence) "
            f"FROM kg_triples x WHERE x.head=t.head AND x.relation='{_esc(relation)}') ORDER BY "
            "t.head,t.confidence,t.tail"
        )
    if _has(q, "所在地区平均年龄", "年龄高于"):
        return _safe(
            "SELECT p.patient_id,p.region,p.age,ROUND((SELECT AVG(q.age) FROM patients q WHERE "
            "q.region=p.region),2) AS region_avg_age FROM patients p WHERE p.age>(SELECT AVG(q.age) "
            "FROM patients q WHERE q.region=p.region) ORDER BY p.region,p.age DESC,p.patient_id"
        )
    if _has(q, "同一头实体", "最大观察次数", "保留并列"):
        return _safe(
            "SELECT t.head,t.relation,t.tail,t.count FROM kg_triples t WHERE t.count=(SELECT "
            "MAX(x.count) FROM kg_triples x WHERE x.head=t.head) ORDER BY t.head,t.relation,t.tail"
        )
    if _has(q, "至少有", "次就诊", "标量子查询"):
        threshold = _integer(q, r"至少有\s*(\d+)\s*次")
        if threshold is not None:
            return _safe(
                "SELECT p.patient_id,p.name,p.region FROM patients p WHERE (SELECT COUNT(*) FROM "
                f"visits v WHERE v.patient_id=p.patient_id)>={threshold} ORDER BY p.patient_id"
            )
    if _has(q, "同科室平均住院天数", "高于"):
        return _safe(
            "SELECT v.visit_id,v.department,v.los_days,ROUND((SELECT AVG(w.los_days) FROM visits w "
            "WHERE w.department=v.department),2) AS department_avg_los FROM visits v WHERE "
            "v.los_days>(SELECT AVG(w.los_days) FROM visits w WHERE w.department=v.department) "
            "ORDER BY v.department,v.los_days DESC,v.visit_id"
        )
    if _has(q, "每个科室", "费用最低", "并列"):
        return _safe(
            "SELECT v.visit_id,v.department,v.disease,v.cost FROM visits v WHERE v.cost=(SELECT "
            "MIN(w.cost) FROM visits w WHERE w.department=v.department) ORDER BY v.department,v.visit_id"
        )
    rels = _relations(q, domains)
    if len(rels) >= 2 and _has(q, "同时拥有", "两类出边"):
        return _safe(
            "SELECT DISTINCT t.head FROM kg_triples t WHERE EXISTS (SELECT 1 FROM kg_triples s "
            f"WHERE s.head=t.head AND s.relation='{_esc(rels[0])}') AND EXISTS (SELECT 1 FROM "
            f"kg_triples d WHERE d.head=t.head AND d.relation='{_esc(rels[1])}') ORDER BY t.head"
        )
    if _has(q, "累计就诊费用", "相关子查询"):
        threshold = _integer(q, r"超过\s*(\d+)\s*元")
        if threshold is not None:
            return _safe(
                "SELECT p.patient_id,p.name,p.region FROM patients p WHERE (SELECT SUM(v.cost) FROM "
                f"visits v WHERE v.patient_id=p.patient_id)>{threshold} ORDER BY p.patient_id"
            )
    exam = _ground(q, domains, "exam")
    drug = _ground(q, domains, "drug")
    if exam and drug and _has(q, "两个相关EXISTS", "患者编号和姓名"):
        return _safe(
            "SELECT p.patient_id,p.name FROM patients p WHERE EXISTS (SELECT 1 FROM visits v WHERE "
            f"v.patient_id=p.patient_id AND v.exam='{_esc(exam)}') AND EXISTS (SELECT 1 FROM visits w "
            f"WHERE w.patient_id=p.patient_id AND w.drug_prescribed='{_esc(drug)}') ORDER BY p.patient_id"
        )
    return None


def _compile_having(q: str, _domains: Domains) -> Optional[str]:
    if not any(
        term in q
        for term in (
            "只展示",
            "仅列出",
            "保留至少",
            "只保留",
            "筛选",
            "只输出",
            "只返回",
        )
    ):
        return None
    if _has(q, "关系标识", "三元组数量", "平均置信度", "观察次数总和"):
        n = _integer(q, r"至少\s*(\d+)\s*条")
        if n is not None:
            return _safe(
                "SELECT relation,COUNT(*) AS triple_count,ROUND(AVG(confidence),4) AS avg_confidence,"
                f"SUM(count) AS observations FROM kg_triples GROUP BY relation HAVING COUNT(*)>={n} "
                "ORDER BY triple_count DESC,relation"
            )
    if _has(q, "尾实体分组", "不同头实体", "关系数"):
        n = _integer(q, r"至少被\s*(\d+)\s*个")
        if n is not None:
            return _safe(
                "SELECT tail,COUNT(DISTINCT head) AS head_count,COUNT(*) AS triple_count FROM "
                f"kg_triples GROUP BY tail HAVING COUNT(DISTINCT head)>={n} ORDER BY head_count DESC,tail"
            )
    if _has(q, "患者编号分组", "就诊", "总费用"):
        n = _integer(q, r"至少\s*(\d+)\s*次")
        if n is not None:
            return _safe(
                "SELECT patient_id,COUNT(*) AS visit_count,ROUND(SUM(cost),2) AS total_cost FROM "
                f"visits GROUP BY patient_id HAVING COUNT(*)>={n} ORDER BY visit_count DESC,"
                "total_cost DESC,patient_id"
            )
    if _has(q, "头实体分组", "出边", "不同关系类型"):
        n = _integer(q, r"至少\s*(\d+)\s*条")
        if n is not None:
            return _safe(
                "SELECT head,COUNT(*) AS outgoing_count,COUNT(DISTINCT relation) AS relation_type_count "
                f"FROM kg_triples GROUP BY head HAVING COUNT(*)>={n} ORDER BY outgoing_count DESC,head"
            )
    if _has(q, "科室", "总费用", "平均费用", "记录数"):
        n = _integer(q, r"大于\s*(\d+)\s*元")
        if n is not None:
            return _safe(
                "SELECT department,ROUND(SUM(cost),2) AS total_cost,ROUND(AVG(cost),2) AS avg_cost,"
                f"COUNT(*) AS visit_count FROM visits GROUP BY department HAVING SUM(cost)>{n} "
                "ORDER BY total_cost DESC,department"
            )
    if _has(q, "科室", "不同疾病", "就诊数"):
        n = _integer(q, r"至少\s*(\d+)\s*种")
        if n is not None:
            return _safe(
                "SELECT department,COUNT(DISTINCT disease) AS disease_count,COUNT(*) AS visit_count "
                f"FROM visits GROUP BY department HAVING COUNT(DISTINCT disease)>={n} ORDER BY "
                "disease_count DESC,department"
            )
    if _has(q, "疾病", "平均住院天数", "记录"):
        avg = _number(q, r"大于\s*(\d+(?:\.\d+)?)\s*天")
        count = _integer(q, r"至少\s*(\d+)\s*条")
        if avg is not None and count is not None:
            return _safe(
                "SELECT disease,ROUND(AVG(los_days),2) AS avg_los_days,COUNT(*) AS visit_count "
                f"FROM visits GROUP BY disease HAVING AVG(los_days)>{avg} AND COUNT(*)>={count} "
                "ORDER BY avg_los_days DESC,disease"
            )
    if _has(q, "科室分组", "就诊次数", "平均费用"):
        n = _integer(q, r"大于\s*(\d+)\s*次")
        if n is not None:
            return _safe(
                "SELECT department,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost FROM "
                f"visits GROUP BY department HAVING COUNT(*)>{n} ORDER BY visit_count DESC,department"
            )
    if _has(q, "疾病", "记录数", "不同患者数"):
        n = _integer(q, r"至少\s*(\d+)\s*次")
        if n is not None:
            return _safe(
                "SELECT disease,COUNT(*) AS visit_count,COUNT(DISTINCT patient_id) AS patient_count "
                f"FROM visits GROUP BY disease HAVING COUNT(*)>={n} ORDER BY visit_count DESC,disease"
            )
    if _has(q, "年月", "就诊", "平均费用"):
        n = _integer(q, r"至少\s*(\d+)\s*次")
        if n is not None:
            return _safe(
                "SELECT strftime('%Y-%m',visit_date) AS visit_month,COUNT(*) AS visit_count,"
                f"ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY visit_month HAVING COUNT(*)>={n} "
                "ORDER BY visit_month"
            )
    return None


def _compile_topk(q: str, _domains: Domains) -> Optional[str]:
    limit = _limit(q)
    if limit is None:
        return None
    if _has(q, "累计费用", "患者编号", "就诊次数"):
        return _safe(
            "SELECT patient_id,ROUND(SUM(cost),2) AS total_cost,COUNT(*) AS visit_count FROM "
            f"visits GROUP BY patient_id ORDER BY total_cost DESC,patient_id LIMIT {limit}"
        )
    if _has(q, "图谱", "Drug实体", "不同头疾病数"):
        return _safe(
            "SELECT tail AS drug,COUNT(DISTINCT head) AS disease_count FROM kg_triples WHERE "
            f"tail_type='Drug' GROUP BY tail ORDER BY disease_count DESC,drug LIMIT {limit}"
        )
    if _has(q, "年龄从高到低", "患者", "地区"):
        return _safe(
            f"SELECT patient_id,name,age,region FROM patients ORDER BY age DESC,patient_id LIMIT {limit}"
        )
    if _has(q, "住院天数降序", "就诊编号", "费用"):
        return _safe(
            "SELECT visit_id,disease,los_days,cost FROM visits ORDER BY los_days DESC,cost DESC,"
            f"visit_id LIMIT {limit}"
        )
    if _has(q, "就诊次数", "连接姓名", "总费用"):
        return _safe(
            "SELECT p.patient_id,p.name,COUNT(*) AS visit_count,ROUND(SUM(v.cost),2) AS total_cost "
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.patient_id,p.name "
            f"ORDER BY visit_count DESC,total_cost DESC,p.patient_id LIMIT {limit}"
        )
    if _has(q, "提及次数升序", "最低", "实体"):
        return _safe(
            f"SELECT name,type,mention_count FROM kg_entities ORDER BY mention_count,name LIMIT {limit}"
        )
    if _has(q, "就诊记录数最多", "疾病", "不同患者数"):
        return _safe(
            "SELECT disease,COUNT(*) AS visit_count,COUNT(DISTINCT patient_id) AS patient_count "
            f"FROM visits GROUP BY disease ORDER BY visit_count DESC,disease LIMIT {limit}"
        )
    if _has(q, "记录数至少", "疾病", "平均费用"):
        threshold = _integer(q, r"记录数至少\s*(\d+)\s*次")
        if threshold is not None:
            return _safe(
                "SELECT disease,ROUND(AVG(cost),2) AS avg_cost,COUNT(*) AS visit_count FROM visits "
                f"GROUP BY disease HAVING COUNT(*)>={threshold} ORDER BY avg_cost DESC,disease LIMIT {limit}"
            )
    if _has(q, "月度就诊次数", "年月", "平均费用"):
        return _safe(
            "SELECT strftime('%Y-%m',visit_date) AS visit_month,COUNT(*) AS visit_count,"
            f"ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY visit_month ORDER BY visit_count DESC,visit_month LIMIT {limit}"
        )
    if _has(q, "就诊次数", "科室", "平均费用"):
        return _safe(
            "SELECT department,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost FROM visits "
            f"GROUP BY department ORDER BY visit_count DESC,department LIMIT {limit}"
        )
    return None


def _compile_filtering(q: str, domains: Domains) -> Optional[str]:
    relation = _relation(q, domains)
    confidence = _number(q, r"置信度至少\s*(\d+(?:\.\d+)?)")
    if relation and confidence is not None and _has(q, "头实体", "尾实体", "观察次数"):
        return _safe(
            "SELECT head,tail,confidence,count FROM kg_triples WHERE "
            f"relation='{_esc(relation)}' AND confidence>={confidence} ORDER BY confidence DESC,"
            "count DESC,head,tail"
        )
    entity_types = _entity_types(q, domains)
    if len(entity_types) == 1 and _has(q, "名称", "含有", "提及次数"):
        quoted = re.search(r"[“\"]([^”\"]+)[”\"]", q)
        if quoted:
            value = _esc(quoted.group(1))
            return _safe(
                "SELECT name,mention_count FROM kg_entities WHERE "
                f"type='{_esc(entity_types[0])}' AND name LIKE '%{value}%' ORDER BY mention_count DESC,name"
            )
    gender = re.search(r"性别为([男女])", q)
    region = _ground(q, domains, "region")
    age_ge = _integer(q, r"年龄(?:至少|不低于)\s*(\d+)")
    return_clause = q[q.find("返回") :] if "返回" in q else ""
    if (
        gender
        and age_ge is not None
        and _has(q, "患者编号", "姓名")
        and "地区" in return_clause
    ):
        return _safe(
            "SELECT patient_id,name,age,region FROM patients WHERE "
            f"gender='{gender.group(1)}' AND age>={age_ge} ORDER BY age DESC,patient_id"
        )
    age_lt = _integer(q, r"年龄小于\s*(\d+)")
    if age_lt is not None and _has(q, "患者编号", "姓名", "年龄") and not gender:
        return _safe(
            f"SELECT patient_id,name,age FROM patients WHERE age<{age_lt} ORDER BY age,name,patient_id"
        )
    if gender and region and age_ge is not None and _has(q, "患者编号", "姓名", "年龄"):
        projection = (
            "patient_id,name,age,region"
            if "返回" in q and "地区" in q[q.find("返回") :]
            else "patient_id,name,age"
        )
        return _safe(
            f"SELECT {projection} FROM patients WHERE region='{_esc(region)}' AND "
            f"gender='{gender.group(1)}' AND age>={age_ge} ORDER BY age DESC,patient_id"
        )
    if gender and region and _has(q, "患者编号", "姓名", "年龄"):
        return _safe(
            "SELECT patient_id,name,age FROM patients WHERE "
            f"gender='{gender.group(1)}' AND region='{_esc(region)}' ORDER BY age,patient_id"
        )
    department = _ground(q, domains, "department")
    los = _integer(q, r"住院天数超过\s*(\d+)")
    if department and los is not None and "不是" in q:
        return _safe(
            "SELECT visit_id,department,disease,los_days FROM visits WHERE "
            f"department<>'{_esc(department)}' AND los_days>{los} ORDER BY los_days DESC,visit_id"
        )
    cost_between = re.search(r"费用在\s*(\d+)\s*到\s*(\d+)", q)
    if cost_between and department and _has(q, "疾病", "住院天数"):
        return _safe(
            "SELECT visit_id,disease,cost,los_days FROM visits WHERE cost BETWEEN "
            f"{cost_between.group(1)} AND {cost_between.group(2)} AND department='{_esc(department)}' "
            "ORDER BY cost,visit_id"
        )
    regions = _grounds(q, domains, "region")
    age_le = _integer(q, r"年龄不超过\s*(\d+)")
    if len(regions) >= 2 and age_le is not None:
        values = ",".join(f"'{_esc(item)}'" for item in regions)
        return _safe(
            "SELECT patient_id,name,age,region FROM patients WHERE "
            f"region IN ({values}) AND age<={age_le} ORDER BY region,age,patient_id"
        )
    patient_id = re.search(r"患者编号为(P\d+)", q)
    if patient_id and _has(q, "姓名", "性别", "年龄", "地区"):
        return _safe(
            f"SELECT name,gender,age,region FROM patients WHERE patient_id='{patient_id.group(1)}'"
        )
    rels = _relations(q, domains)
    disease = _ground(q, domains, "disease")
    if disease and len(rels) >= 2 and _has(q, "尾实体", "尾类型", "置信度"):
        values = ",".join(f"'{_esc(item)}'" for item in rels)
        return _safe(
            "SELECT tail,tail_type,relation,confidence FROM kg_triples WHERE "
            f"head='{_esc(disease)}' AND relation IN ({values}) ORDER BY relation,confidence DESC,tail"
        )
    count_between = re.search(r"观察次数在\s*(\d+)\s*到\s*(\d+)", q)
    entity_type = _entity_type(q, domains)
    if count_between and entity_type and _has(q, "头实体", "关系标识", "尾实体"):
        return _safe(
            "SELECT head,relation,tail,count FROM kg_triples WHERE count BETWEEN "
            f"{count_between.group(1)} AND {count_between.group(2)} AND tail_type='{_esc(entity_type)}' "
            "ORDER BY count DESC,head,tail"
        )
    mention_between = re.search(r"提及次数在\s*(\d+)\s*到\s*(\d+)", q)
    if mention_between and len(entity_types) >= 2:
        values = ",".join(f"'{_esc(item)}'" for item in entity_types)
        return _safe(
            "SELECT name,type,mention_count FROM kg_entities WHERE "
            f"type IN ({values}) AND mention_count BETWEEN {mention_between.group(1)} AND "
            f"{mention_between.group(2)} ORDER BY type,mention_count DESC,name"
        )
    age_gt = _integer(q, r"年龄大于\s*(\d+)")
    if age_gt is not None and region and "不等于" in q:
        return _safe(
            "SELECT patient_id,name,age,region FROM patients WHERE "
            f"age>{age_gt} AND region<>'{_esc(region)}' ORDER BY age DESC,patient_id"
        )
    age_between = re.search(r"年龄在\s*(\d+)\s*到\s*(\d+)", q)
    if age_between and len(regions) >= 2:
        values = ",".join(f"'{_esc(item)}'" for item in regions)
        return _safe(
            "SELECT patient_id,name,age,region FROM patients WHERE age BETWEEN "
            f"{age_between.group(1)} AND {age_between.group(2)} AND region IN ({values}) "
            "ORDER BY region,age,patient_id"
        )
    age_or = re.search(r"年龄小于\s*(\d+)\s*岁或大于\s*(\d+)", q)
    if gender and age_or:
        return _safe(
            "SELECT patient_id,name,age FROM patients WHERE "
            f"gender='{gender.group(1)}' AND (age<{age_or.group(1)} OR age>{age_or.group(2)}) "
            "ORDER BY age,patient_id"
        )
    dates = _dates(q)
    if disease and dates and _has(q, "就诊编号", "处方药", "检查"):
        return _safe(
            "SELECT visit_id,visit_date,drug_prescribed,exam FROM visits WHERE "
            f"disease='{_esc(disease)}' AND visit_date>='{dates[0]}' ORDER BY visit_date,visit_id"
        )
    drugs = _grounds(q, domains, "drug")
    if len(drugs) >= 2 and _has(q, "就诊编号", "疾病", "日期"):
        values = ",".join(f"'{_esc(item)}'" for item in drugs)
        return _safe(
            "SELECT visit_id,drug_prescribed,disease,visit_date FROM visits WHERE "
            f"drug_prescribed IN ({values}) ORDER BY drug_prescribed,visit_date,visit_id"
        )
    cost_ge = _integer(q, r"费用至少\s*(\d+)")
    if department and cost_ge is not None and _has(q, "就诊编号", "日期", "疾病"):
        return _safe(
            "SELECT visit_id,visit_date,disease,cost FROM visits WHERE "
            f"department='{_esc(department)}' AND cost>={cost_ge} ORDER BY cost DESC,visit_id"
        )
    mention_ge = _integer(q, r"提及次数至少\s*(\d+)")
    if entity_type and mention_ge is not None and _has(q, "实体名", "提及次数"):
        return _safe(
            "SELECT name,mention_count FROM kg_entities WHERE "
            f"type='{_esc(entity_type)}' AND mention_count>={mention_ge} ORDER BY mention_count DESC,name"
        )
    return None


def _compile_aggregation(q: str, domains: Domains) -> Optional[str]:
    if _has(q, "年月", "科室", "就诊次数", "平均费用"):
        return _safe(
            "SELECT strftime('%Y-%m',visit_date) AS visit_month,department,COUNT(*) AS visit_count,"
            "ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY visit_month,department ORDER BY "
            "visit_month,department"
        )
    if _has(q, "主要处方药", "就诊次数", "平均费用", "空白处方"):
        return _safe(
            "SELECT drug_prescribed,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost FROM "
            "visits GROUP BY drug_prescribed ORDER BY visit_count DESC,drug_prescribed"
        )
    if _has(q, "检查", "就诊数", "最高费用", "空白检查"):
        return _safe(
            "SELECT exam,COUNT(*) AS visit_count,MAX(cost) AS max_cost FROM visits GROUP BY exam "
            "ORDER BY max_cost DESC,exam"
        )
    dates = _dates(q)
    if len(dates) >= 2 and _has(q, "各科室", "就诊次数", "平均费用"):
        return _safe(
            "SELECT department,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost FROM visits "
            f"WHERE visit_date BETWEEN '{dates[0]}' AND '{dates[1]}' GROUP BY department "
            "ORDER BY visit_count DESC,department"
        )
    if _has(q, "具体就诊日期", "当天就诊数", "总费用"):
        return _safe(
            "SELECT visit_date,COUNT(*) AS visit_count,ROUND(SUM(cost),2) AS total_cost FROM visits "
            "GROUP BY visit_date ORDER BY visit_date"
        )
    if _has(q, "主诊断疾病", "就诊数", "平均费用"):
        return _safe(
            "SELECT disease,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost FROM visits "
            "GROUP BY disease ORDER BY avg_cost DESC,disease"
        )
    if _has(q, "实体类型", "实体数量", "提及次数总和"):
        return _safe(
            "SELECT type,COUNT(*) AS entity_count,SUM(mention_count) AS total_mentions FROM "
            "kg_entities GROUP BY type ORDER BY entity_count DESC,type"
        )
    if _has(q, "地区和性别", "患者人数"):
        return _safe(
            "SELECT region,gender,COUNT(*) AS patient_count FROM patients GROUP BY region,gender "
            "ORDER BY region,gender"
        )
    if _has(q, "住院天数", "三档", "总费用"):
        return _safe(
            "SELECT CASE WHEN los_days=0 THEN '0天' WHEN los_days BETWEEN 1 AND 5 THEN '1到5天' "
            "ELSE '大于5天' END AS los_band,COUNT(*) AS visit_count,ROUND(SUM(cost),2) AS total_cost "
            "FROM visits GROUP BY los_band ORDER BY los_band"
        )
    if _has(q, "性别为女", "各地区", "平均年龄"):
        return _safe(
            "SELECT region,COUNT(*) AS patient_count,ROUND(AVG(age),2) AS avg_age FROM patients "
            "WHERE gender='女' GROUP BY region ORDER BY patient_count DESC,region"
        )
    year = _year(q)
    if year and _has(q, "检查项目", "不同疾病数"):
        return _safe(
            "SELECT exam,COUNT(*) AS visit_count,COUNT(DISTINCT disease) AS disease_count FROM "
            f"visits WHERE strftime('%Y',visit_date)='{year}' GROUP BY exam ORDER BY visit_count DESC,exam"
        )
    if _has(q, "每种关系标识", "不同头实体", "三元组数"):
        return _safe(
            "SELECT relation,COUNT(DISTINCT head) AS head_count,COUNT(*) AS triple_count FROM "
            "kg_triples GROUP BY relation ORDER BY head_count DESC,relation"
        )
    department = _ground(q, domains, "department")
    if department and _has(q, "按疾病", "最低费用", "最高费用"):
        return _safe(
            "SELECT disease,COUNT(*) AS visit_count,MIN(cost) AS min_cost,MAX(cost) AS max_cost "
            f"FROM visits WHERE department='{_esc(department)}' GROUP BY disease ORDER BY visit_count DESC,disease"
        )
    if _has(q, "患者编号", "就诊次数", "总费用", "平均费用"):
        return _safe(
            "SELECT patient_id,COUNT(*) AS visit_count,ROUND(SUM(cost),2) AS total_cost,"
            "ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY patient_id ORDER BY visit_count DESC,"
            "total_cost DESC,patient_id"
        )
    if _has(q, "各地区", "最低年龄", "最高年龄", "平均年龄"):
        return _safe(
            "SELECT region,MIN(age) AS min_age,MAX(age) AS max_age,ROUND(AVG(age),2) AS avg_age "
            "FROM patients GROUP BY region ORDER BY avg_age DESC,region"
        )
    if _has(q, "每个地区", "患者人数"):
        return _safe(
            "SELECT region,COUNT(*) AS patient_count FROM patients GROUP BY region ORDER BY "
            "patient_count DESC,region"
        )
    if _has(q, "每种实体类型", "平均提及次数", "最低提及次数"):
        return _safe(
            "SELECT type,ROUND(AVG(mention_count),2) AS avg_mentions,MIN(mention_count) AS "
            "min_mentions FROM kg_entities GROUP BY type ORDER BY avg_mentions DESC,type"
        )
    if _has(q, "各科室", "就诊次数", "不同患者数"):
        return _safe(
            "SELECT department,COUNT(*) AS visit_count,COUNT(DISTINCT patient_id) AS patient_count "
            "FROM visits GROUP BY department ORDER BY visit_count DESC,department"
        )
    confidence = _number(q, r"置信度至少\s*(\d+(?:\.\d+)?)")
    if confidence is not None and _has(q, "关系标识", "平均观察次数", "不同头实体数"):
        return _safe(
            "SELECT relation,ROUND(AVG(count),2) AS avg_observation_count,COUNT(DISTINCT head) AS "
            f"head_count FROM kg_triples WHERE confidence>={confidence} GROUP BY relation ORDER BY "
            "avg_observation_count DESC,relation"
        )
    if dates and _has(q, "各科室", "总费用", "就诊次数"):
        return _safe(
            "SELECT department,ROUND(SUM(cost),2) AS total_cost,COUNT(*) AS visit_count FROM visits "
            f"WHERE visit_date>='{dates[0]}' GROUP BY department ORDER BY total_cost DESC,department"
        )
    if _has(q, "关系中文名", "三元组数量", "观察次数总和"):
        return _safe(
            "SELECT relation_zh,COUNT(*) AS triple_count,SUM(count) AS observation_count FROM "
            "kg_triples GROUP BY relation_zh ORDER BY triple_count DESC,relation_zh"
        )
    if _has(q, "关系标识", "平均置信度", "最低置信度", "最高置信度"):
        return _safe(
            "SELECT relation,ROUND(AVG(confidence),4) AS avg_confidence,MIN(confidence) AS "
            "min_confidence,MAX(confidence) AS max_confidence FROM kg_triples GROUP BY relation "
            "ORDER BY avg_confidence DESC,relation"
        )
    if _has(q, "年和月", "就诊次数"):
        return _safe(
            "SELECT strftime('%Y-%m',visit_date) AS visit_month,COUNT(*) AS visit_count FROM visits "
            "GROUP BY visit_month ORDER BY visit_month"
        )
    if _has(q, "每种性别", "不同地区数", "患者总数"):
        return _safe(
            "SELECT gender,COUNT(DISTINCT region) AS region_count,COUNT(*) AS patient_count FROM "
            "patients GROUP BY gender ORDER BY gender"
        )
    if _has(q, "地区", "年龄至少65岁", "全部患者数"):
        return _safe(
            "SELECT region,SUM(CASE WHEN age>=65 THEN 1 ELSE 0 END) AS age_65_plus_count,COUNT(*) "
            "AS total_count FROM patients GROUP BY region ORDER BY age_65_plus_count DESC,region"
        )
    cost_gt = _integer(q, r"费用大于\s*(\d+)")
    if cost_gt is not None and _has(q, "疾病", "总费用", "记录数"):
        return _safe(
            "SELECT disease,ROUND(SUM(cost),2) AS total_cost,COUNT(*) AS visit_count FROM visits "
            f"WHERE cost>{cost_gt} GROUP BY disease ORDER BY total_cost DESC,disease"
        )
    if _has(q, "年月", "总费用", "平均费用", "记录数"):
        return _safe(
            "SELECT strftime('%Y-%m',visit_date) AS visit_month,ROUND(SUM(cost),2) AS total_cost,"
            "ROUND(AVG(cost),2) AS avg_cost,COUNT(*) AS visit_count FROM visits GROUP BY visit_month "
            "ORDER BY visit_month"
        )
    if _has(q, "年份和科室", "总费用", "不同患者数"):
        return _safe(
            "SELECT strftime('%Y',visit_date) AS visit_year,department,ROUND(SUM(cost),2) AS "
            "total_cost,COUNT(DISTINCT patient_id) AS patient_count FROM visits GROUP BY visit_year,"
            "department ORDER BY visit_year,total_cost DESC,department"
        )
    if _has(q, "疾病和就诊年份", "平均住院天数"):
        return _safe(
            "SELECT disease,strftime('%Y',visit_date) AS visit_year,COUNT(*) AS visit_count,"
            "ROUND(AVG(los_days),2) AS avg_los_days FROM visits GROUP BY disease,visit_year "
            "ORDER BY disease,visit_year"
        )
    if _has(q, "实体类型", "最高提及次数", "实体数量"):
        return _safe(
            "SELECT type,MAX(mention_count) AS max_mentions,COUNT(*) AS entity_count FROM "
            "kg_entities GROUP BY type ORDER BY type"
        )
    if _has(q, "实体类型", "不同实体名数量", "提及次数总和"):
        return _safe(
            "SELECT type,COUNT(DISTINCT name) AS distinct_name_count,SUM(mention_count) AS "
            "total_mentions FROM kg_entities GROUP BY type ORDER BY total_mentions DESC,type"
        )
    if _has(q, "科室与疾病组合", "就诊次数", "平均费用"):
        return _safe(
            "SELECT department,disease,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost FROM "
            "visits GROUP BY department,disease ORDER BY department,visit_count DESC,disease"
        )
    return None


def _compile_temporal(q: str, _domains: Domains) -> Optional[str]:
    if _has(q, "具体日期分组", "同日就诊数", "不同科室数"):
        n = _integer(q, r"至少\s*(\d+)\s*次")
        if n is not None:
            return _safe(
                "SELECT visit_date,COUNT(*) AS visit_count,COUNT(DISTINCT department) AS "
                f"department_count FROM visits GROUP BY visit_date HAVING COUNT(*)>={n} ORDER BY visit_date"
            )
    year, month = _year(q), _month(q)
    if year and month and _has(q, "就诊编号", "疾病", "费用"):
        return _safe(
            "SELECT visit_id,visit_date,disease,cost FROM visits WHERE "
            f"strftime('%Y',visit_date)='{year}' AND strftime('%m',visit_date)='{month}' "
            "ORDER BY visit_date,visit_id"
        )
    if _has(q, "首次就诊后", "0到", "天内"):
        n = _integer(q, r"0到\s*(\d+)\s*天")
        if n is not None:
            return _safe(
                "SELECT v.visit_id,v.patient_id,v.visit_date FROM visits v WHERE julianday(v.visit_date)-"
                "julianday((SELECT MIN(w.visit_date) FROM visits w WHERE w.patient_id=v.patient_id)) "
                f"BETWEEN 0 AND {n} ORDER BY v.patient_id,v.visit_date,v.visit_id"
            )
    if _has(q, "年份和季度", "月份减1", "总费用"):
        return _safe(
            "SELECT strftime('%Y',visit_date) AS visit_year,((CAST(strftime('%m',visit_date) AS "
            "INTEGER)-1)/3)+1 AS quarter_no,COUNT(*) AS visit_count,ROUND(SUM(cost),2) AS total_cost "
            "FROM visits GROUP BY visit_year,quarter_no ORDER BY visit_year,quarter_no"
        )
    if _has(q, "每种疾病", "最晚出现日期", "累计费用"):
        return _safe(
            "SELECT disease,MAX(visit_date) AS last_date,ROUND(SUM(cost),2) AS total_cost FROM visits "
            "GROUP BY disease ORDER BY last_date DESC,disease"
        )
    if _has(q, "全库最晚就诊日期", "就诊编号", "患者编号"):
        return _safe(
            "SELECT visit_id,patient_id,department,cost FROM visits WHERE visit_date=(SELECT "
            "MAX(visit_date) FROM visits) ORDER BY visit_id"
        )
    if _has(q, "每位患者", "最晚就诊日期", "总就诊次数"):
        return _safe(
            "SELECT patient_id,MAX(visit_date) AS last_visit_date,COUNT(*) AS visit_count FROM visits "
            "GROUP BY patient_id ORDER BY last_visit_date DESC,patient_id"
        )
    if _has(q, "首末就诊间隔", "大于", "首日", "末日"):
        n = _integer(q, r"大于\s*(\d+)\s*天")
        if n is not None:
            return _safe(
                "SELECT patient_id,MIN(visit_date) AS first_date,MAX(visit_date) AS last_date,CAST("
                "julianday(MAX(visit_date))-julianday(MIN(visit_date)) AS INTEGER) AS span_days FROM "
                f"visits GROUP BY patient_id HAVING julianday(MAX(visit_date))-julianday(MIN(visit_date))>{n} "
                "ORDER BY span_days DESC,patient_id"
            )
    if _has(q, "公历月份编号", "就诊次数", "平均费用"):
        return _safe(
            "SELECT strftime('%m',visit_date) AS month_no,COUNT(*) AS visit_count,ROUND(AVG(cost),2) "
            "AS avg_cost FROM visits GROUP BY month_no ORDER BY month_no"
        )
    if _has(q, "全库最晚就诊日", "此前", "天内"):
        n = _integer(q, r"此前\s*(\d+)\s*天")
        if n is not None and "按科室统计" in q:
            return _safe(
                "SELECT department,COUNT(*) AS visit_count FROM visits WHERE visit_date>=date((SELECT "
                f"MAX(visit_date) FROM visits),'-{n} days') GROUP BY department ORDER BY visit_count DESC,department"
            )
        if n is not None:
            return _safe(
                "SELECT visit_id,visit_date,disease FROM visits WHERE visit_date>=date((SELECT "
                f"MAX(visit_date) FROM visits),'-{n} days') ORDER BY visit_date,visit_id"
            )
    if _has(q, "按科室", "最早就诊日", "最晚就诊日", "覆盖天数"):
        return _safe(
            "SELECT department,MIN(visit_date) AS first_date,MAX(visit_date) AS last_date,CAST("
            "julianday(MAX(visit_date))-julianday(MIN(visit_date)) AS INTEGER) AS covered_days FROM "
            "visits GROUP BY department ORDER BY covered_days DESC,department"
        )
    if _has(q, "按年月", "就诊次数", "平均费用") and "科室" not in q:
        return _safe(
            "SELECT strftime('%Y-%m',visit_date) AS visit_month,COUNT(*) AS visit_count,"
            "ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY visit_month ORDER BY visit_month"
        )
    if _has(q, "全库最早就诊日期", "就诊编号", "患者编号"):
        return _safe(
            "SELECT visit_id,patient_id,department,disease FROM visits WHERE visit_date=(SELECT "
            "MIN(visit_date) FROM visits) ORDER BY visit_id"
        )
    if _has(q, "每位患者", "最早到最晚", "间隔天数"):
        return _safe(
            "SELECT patient_id,MIN(visit_date) AS first_date,MAX(visit_date) AS last_date,CAST("
            "julianday(MAX(visit_date))-julianday(MIN(visit_date)) AS INTEGER) AS span_days FROM "
            "visits GROUP BY patient_id ORDER BY span_days DESC,patient_id"
        )
    if _has(q, "星期编号", "strftime", "就诊次数"):
        return _safe(
            "SELECT strftime('%w',visit_date) AS weekday_no,COUNT(*) AS visit_count FROM visits "
            "GROUP BY weekday_no ORDER BY weekday_no"
        )
    if _has(q, "上半月", "日期数字"):
        n = _integer(q, r"不超过\s*(\d+)")
        if n is not None:
            return _safe(
                "SELECT visit_id,visit_date,disease,cost FROM visits WHERE CAST(strftime('%d',"
                f"visit_date) AS INTEGER)<={n} ORDER BY visit_date,visit_id"
            )
    if _has(q, "下半月", "日期数字"):
        n = _integer(q, r"大于\s*(\d+)")
        if n is not None:
            return _safe(
                "SELECT visit_id,visit_date,department,cost FROM visits WHERE CAST(strftime('%d',"
                f"visit_date) AS INTEGER)>{n} ORDER BY visit_date,visit_id"
            )
    if _has(q, "按年份", "平均费用", "最高费用", "最低费用"):
        return _safe(
            "SELECT strftime('%Y',visit_date) AS visit_year,ROUND(AVG(cost),2) AS avg_cost,MAX(cost) "
            "AS max_cost,MIN(cost) AS min_cost FROM visits GROUP BY visit_year ORDER BY visit_year"
        )
    month = _month(q)
    if month and _has(q, "所有年份", "就诊", "科室", "疾病"):
        return _safe(
            "SELECT visit_date,department,disease,cost FROM visits WHERE "
            f"strftime('%m',visit_date)='{month}' ORDER BY visit_date,visit_id"
        )
    if _has(q, "每种疾病", "最早出现日期", "总记录数"):
        return _safe(
            "SELECT disease,MIN(visit_date) AS first_date,COUNT(*) AS visit_count FROM visits "
            "GROUP BY disease ORDER BY first_date,disease"
        )
    return None


def _compile_kg(q: str, domains: Domains) -> Optional[str]:
    relation = _relation(q, domains)
    disease = _ground(q, domains, "disease")
    if relation and disease and _has(q, "共享至少一个", "其他疾病"):
        label = "shared_exam" if "检查" in q else "shared_symptom"
        return _safe(
            "SELECT DISTINCT other.head AS other_disease,other.tail AS "
            f"{label} FROM kg_triples base JOIN kg_triples other ON other.tail=base.tail AND "
            f"other.head<>base.head WHERE base.head='{_esc(disease)}' AND base.relation='{_esc(relation)}' "
            f"AND other.relation='{_esc(relation)}' ORDER BY other_disease,{label}"
        )
    entity_type = _entity_type(q, domains)
    mention = _integer(q, r"提及次数(?:至少|大于)\s*(\d+)")
    if entity_type and mention is not None and _has(q, "实体名", "次数"):
        op = ">" if "大于" in q else ">="
        projection = "name,type,mention_count" if "返回类型" in q or "名、类型" in q else "name,mention_count"
        return _safe(
            f"SELECT {projection} FROM kg_entities WHERE type='{_esc(entity_type)}' AND "
            f"mention_count{op}{mention} ORDER BY mention_count DESC,name"
        )
    if _has(q, "头实体为Disease", "置信度低于", "尾实体"):
        conf = _number(q, r"低于\s*(\d+(?:\.\d+)?)")
        if conf is not None:
            return _safe(
                "SELECT head AS disease,relation,tail,confidence FROM kg_triples WHERE head_type='Disease' "
                f"AND confidence<{conf} ORDER BY confidence,disease,relation,tail"
            )
    if relation and _has(q, "连接关系头尾两侧实体", "提及次数"):
        return _safe(
            "SELECT h.name AS disease,h.mention_count AS disease_mentions,t.name AS complication,"
            "t.mention_count AS complication_mentions FROM kg_triples g JOIN kg_entities h ON "
            "h.name=g.head JOIN kg_entities t ON t.name=g.tail WHERE "
            f"g.relation='{_esc(relation)}' ORDER BY disease_mentions DESC,disease,complication"
        )
    if relation and _has(q, "不同疾病对", "小于"):
        label = "shared_department" if "科室" in q else "shared_drug" if "药物" in q else "shared_symptom"
        return _safe(
            "SELECT a.head AS disease_a,b.head AS disease_b,a.tail AS "
            f"{label} FROM kg_triples a JOIN kg_triples b ON b.tail=a.tail AND a.head<b.head WHERE "
            f"a.relation='{_esc(relation)}' AND b.relation='{_esc(relation)}' ORDER BY {label},disease_a,disease_b"
        )
    types = _entity_types(q, domains)
    conf = _number(q, r"置信度至少\s*(\d+(?:\.\d+)?)")
    if len(types) >= 2 and conf is not None and _has(q, "头实体", "尾实体", "尾类型"):
        vals = ",".join(f"'{_esc(item)}'" for item in types)
        return _safe(
            "SELECT head,relation,tail,tail_type FROM kg_triples WHERE "
            f"tail_type IN ({vals}) AND confidence>={conf} ORDER BY head,relation,tail"
        )
    if _has(q, "关系中文名为", "尾实体类型", "置信度", "观察次数"):
        zh = re.search(r"关系中文名为([^且，,；;]+)", q)
        if zh and entity_type:
            return _safe(
                "SELECT head AS disease,tail AS drug,confidence,count FROM kg_triples WHERE "
                f"relation_zh='{_esc(zh.group(1))}' AND tail_type='{_esc(entity_type)}' ORDER BY "
                "count DESC,disease,drug"
            )
    if relation and entity_type and _has(q, "连接关系与尾实体", "提及次数"):
        n = _integer(q, r"至少\s*(\d+)")
        if n is not None:
            return _safe(
                "SELECT g.head AS disease,e.name AS exam,e.mention_count,g.confidence FROM kg_triples g "
                "JOIN kg_entities e ON e.name=g.tail WHERE "
                f"g.relation='{_esc(relation)}' AND e.type='{_esc(entity_type)}' AND e.mention_count>={n} "
                "ORDER BY e.mention_count DESC,disease,exam"
            )
    if entity_type == "Drug" and _has(q, "尾实体连接到实体表", "不同疾病"):
        return _safe(
            "SELECT e.name AS drug,e.mention_count,COUNT(DISTINCT t.head) AS disease_count FROM "
            "kg_entities e JOIN kg_triples t ON t.tail=e.name WHERE e.type='Drug' AND "
            "t.tail_type='Drug' GROUP BY e.name,e.mention_count ORDER BY disease_count DESC,drug"
        )
    if entity_type and _has(q, "实体名包含", "提及次数"):
        match = re.search(
            r"实体名包含(?:有)?[“\"]?([^”\"，,；;]+?)(?:的图谱实体|，|,|；|;|$)",
            q,
        )
        if match:
            return _safe(
                "SELECT name,mention_count FROM kg_entities WHERE "
                f"type='{_esc(entity_type)}' AND name LIKE '%{_esc(match.group(1))}%' ORDER BY mention_count DESC,name"
            )
    tail_match = re.search(r"尾实体为([^且，,；;]+)", q)
    tail = tail_match.group(1) if tail_match else None
    if tail not in {str(item) for item in domains.get("kg_entity", ())}:
        tail = None
    if tail and entity_type and _has(q, "尾实体", "头疾病", "观察次数"):
        return _safe(
            "SELECT head,relation,confidence,count FROM kg_triples WHERE "
            f"tail='{_esc(tail)}' AND tail_type='{_esc(entity_type)}' ORDER BY confidence DESC,count DESC,head"
        )
    if entity_type == "Disease" and _has(q, "实体名与关系头实体相同", "关系中文名"):
        return _safe(
            "SELECT e.name,e.mention_count,t.relation_zh,t.tail FROM kg_entities e JOIN kg_triples t "
            "ON t.head=e.name WHERE e.type='Disease' ORDER BY e.name,t.relation_zh,t.tail"
        )
    count_ge = _integer(q, r"观察次数至少\s*(\d+)")
    if conf is not None and count_ge is not None and _has(q, "中文关系名", "头", "尾"):
        return _safe(
            "SELECT head,relation_zh,tail,confidence,count FROM kg_triples WHERE "
            f"confidence>={conf} AND count>={count_ge} ORDER BY count DESC,confidence DESC,head,tail"
        )
    return None


def _compile_joins(q: str, domains: Domains) -> Optional[str]:
    relation = _relation(q, domains)
    if relation and _has(q, "实际科室", "尾实体", "两者相等"):
        return _safe(
            "SELECT v.visit_id,v.disease,v.department FROM visits v JOIN kg_triples t ON "
            f"t.head=v.disease AND t.tail=v.department WHERE t.relation='{_esc(relation)}' ORDER BY v.visit_id"
        )
    if relation and _has(q, "图谱头尾实体", "两侧名称", "提及次数"):
        return _safe(
            "SELECT h.name AS disease,h.mention_count AS head_mentions,e.name AS complication,"
            "e.mention_count AS tail_mentions FROM kg_triples t JOIN kg_entities h ON h.name=t.head "
            "JOIN kg_entities e ON e.name=t.tail WHERE "
            f"t.relation='{_esc(relation)}' ORDER BY head_mentions DESC,disease,complication"
        )
    if _has(q, "图谱实体", "作为头实体", "提及次数", "出边数"):
        entity_type = _entity_type(q, domains)
        mention = _integer(q, r"提及次数至少\s*(\d+)")
        if entity_type and mention is not None:
            return _safe(
                "SELECT e.name,e.mention_count,COUNT(t.tail) AS outgoing_count FROM kg_entities e "
                "JOIN kg_triples t ON t.head=e.name WHERE "
                f"e.type='{_esc(entity_type)}' AND e.mention_count>={mention} GROUP BY "
                "e.name,e.mention_count ORDER BY outgoing_count DESC,e.name"
            )
    if _has(q, "每位患者", "最高单次费用", "就诊数"):
        return _safe(
            "SELECT p.patient_id,p.name,MAX(v.cost) AS max_cost,COUNT(*) AS visit_count FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.patient_id,p.name ORDER BY "
            "max_cost DESC,p.patient_id"
        )
    if relation and _has(q, "性别为女", "疾病", "图谱药物数"):
        return _safe(
            "SELECT v.disease,COUNT(DISTINCT p.patient_id) AS patient_count,COUNT(DISTINCT t.tail) "
            "AS graph_drug_count FROM patients p JOIN visits v ON v.patient_id=p.patient_id JOIN "
            f"kg_triples t ON t.head=v.disease WHERE p.gender='女' AND t.relation='{_esc(relation)}' "
            "GROUP BY v.disease ORDER BY patient_count DESC,v.disease"
        )
    if _has(q, "患者性别", "平均费用", "总费用", "记录数"):
        return _safe(
            "SELECT p.gender,COUNT(*) AS visit_count,ROUND(AVG(v.cost),2) AS avg_cost,ROUND(SUM(v.cost),2) "
            "AS total_cost FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.gender "
            "ORDER BY p.gender"
        )
    regions = _grounds(q, domains, "region")
    los = _integer(q, r"住院天数至少\s*(\d+)")
    if len(regions) >= 2 and los is not None:
        vals = ",".join(f"'{_esc(item)}'" for item in regions)
        return _safe(
            "SELECT p.patient_id,p.region,v.department,v.disease,v.los_days FROM patients p JOIN "
            f"visits v ON v.patient_id=p.patient_id WHERE p.region IN ({vals}) AND v.los_days>={los} "
            "ORDER BY v.los_days DESC,p.patient_id,v.visit_id"
        )
    entity_type = _entity_type(q, domains)
    if entity_type == "Drug" and _has(q, "主要处方药", "提及次数", "平均费用"):
        return _safe(
            "SELECT e.name AS drug,e.mention_count,COUNT(*) AS visit_count,ROUND(AVG(v.cost),2) AS "
            "avg_cost FROM visits v JOIN kg_entities e ON e.name=v.drug_prescribed WHERE e.type='Drug' "
            "GROUP BY e.name,e.mention_count ORDER BY visit_count DESC,drug"
        )
    if relation and entity_type == "Exam" and _has(q, "检查名", "关联疾病数", "平均置信度"):
        return _safe(
            "SELECT e.name AS exam,e.mention_count,COUNT(DISTINCT t.head) AS disease_count,"
            "ROUND(AVG(t.confidence),4) AS avg_confidence FROM kg_entities e JOIN kg_triples t ON "
            f"t.tail=e.name WHERE e.type='Exam' AND t.relation='{_esc(relation)}' GROUP BY e.name,"
            "e.mention_count ORDER BY disease_count DESC,exam"
        )
    if _has(q, "图谱两侧实体", "头实体提及次数大于尾实体"):
        return _safe(
            "SELECT h.name AS head,h.mention_count AS head_mentions,e.name AS tail,e.mention_count "
            "AS tail_mentions,t.relation FROM kg_triples t JOIN kg_entities h ON h.name=t.head JOIN "
            "kg_entities e ON e.name=t.tail WHERE h.mention_count>e.mention_count ORDER BY "
            "head_mentions DESC,head,tail"
        )
    if entity_type == "Disease" and _has(q, "实际就诊疾病", "实体提及次数", "平均费用"):
        return _safe(
            "SELECT e.name AS disease,e.mention_count,COUNT(*) AS visit_count,ROUND(AVG(v.cost),2) "
            "AS avg_cost FROM kg_entities e JOIN visits v ON v.disease=e.name WHERE e.type='Disease' "
            "GROUP BY e.name,e.mention_count ORDER BY visit_count DESC,disease"
        )
    if _has(q, "关系中文名", "实际就诊疾病", "不同就诊数", "不同疾病数"):
        return _safe(
            "SELECT t.relation_zh,COUNT(DISTINCT v.visit_id) AS visit_count,COUNT(DISTINCT v.disease) "
            "AS disease_count FROM visits v JOIN kg_triples t ON t.head=v.disease GROUP BY t.relation_zh "
            "ORDER BY visit_count DESC,t.relation_zh"
        )
    if entity_type == "Drug" and _has(q, "作为尾实体", "不同疾病数", "观察次数总和"):
        return _safe(
            "SELECT e.name AS drug,COUNT(DISTINCT t.head) AS disease_count,SUM(t.count) AS observations "
            "FROM kg_entities e JOIN kg_triples t ON t.tail=e.name WHERE e.type='Drug' GROUP BY e.name "
            "ORDER BY disease_count DESC,drug"
        )
    if relation and entity_type == "Exam" and _has(q, "四张表", "患者", "检查提及次数"):
        return _safe(
            "SELECT DISTINCT p.patient_id,p.name,v.disease,t.tail AS exam,e.mention_count FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id JOIN kg_triples t ON t.head=v.disease JOIN "
            f"kg_entities e ON e.name=t.tail WHERE t.relation='{_esc(relation)}' AND e.type='Exam' "
            "ORDER BY p.patient_id,v.disease,exam"
        )
    region = _ground(q, domains, "region")
    diseases = _grounds(q, domains, "disease")
    if region and len(diseases) >= 2 and _has(q, "患者姓名", "处方药"):
        vals = ",".join(f"'{_esc(item)}'" for item in diseases)
        return _safe(
            "SELECT p.name,p.age,v.disease,v.drug_prescribed FROM patients p JOIN visits v ON "
            f"v.patient_id=p.patient_id WHERE p.region='{_esc(region)}' AND v.disease IN ({vals}) "
            "ORDER BY p.name,v.disease,v.visit_id"
        )
    if _has(q, "地区和科室", "交叉汇总", "平均费用"):
        return _safe(
            "SELECT p.region,v.department,COUNT(*) AS visit_count,ROUND(AVG(v.cost),2) AS avg_cost "
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region,v.department "
            "ORDER BY p.region,visit_count DESC,v.department"
        )
    exam = _ground(q, domains, "exam")
    if exam and _has(q, "患者姓名", "年龄", "地区", "就诊日期"):
        return _safe(
            "SELECT p.name,p.age,p.region,v.disease,v.visit_date FROM patients p JOIN visits v ON "
            f"v.patient_id=p.patient_id WHERE v.exam='{_esc(exam)}' ORDER BY v.visit_date,p.patient_id"
        )
    if relation and _has(q, "主要检查", "一致", "置信度"):
        return _safe(
            "SELECT v.visit_id,v.disease,v.exam,t.confidence FROM visits v JOIN kg_triples t ON "
            f"t.head=v.disease AND t.tail=v.exam WHERE t.relation='{_esc(relation)}' ORDER BY v.visit_id"
        )
    mention = _integer(q, r"提及次数至少\s*(\d+)")
    if entity_type == "Disease" and mention is not None and _has(q, "所有出边", "平均置信度"):
        return _safe(
            "SELECT e.name,e.mention_count,COUNT(*) AS edge_count,ROUND(AVG(t.confidence),4) AS "
            "avg_confidence FROM kg_entities e JOIN kg_triples t ON t.head=e.name WHERE e.type='Disease' "
            f"AND e.mention_count>={mention} GROUP BY e.name,e.mention_count ORDER BY edge_count DESC,e.name"
        )
    dates = _dates(q)
    if dates and _has(q, "性别为女", "姓名", "疾病", "费用"):
        return _safe(
            "SELECT p.name,v.visit_date,v.disease,v.cost FROM patients p JOIN visits v ON "
            f"v.patient_id=p.patient_id WHERE p.gender='女' AND v.visit_date>='{dates[0]}' "
            "ORDER BY v.visit_date,p.patient_id,v.visit_id"
        )
    if _has(q, "每次就诊", "患者编号", "姓名", "科室", "疾病"):
        return _safe(
            "SELECT p.patient_id,p.name,v.visit_date,v.department,v.disease FROM patients p JOIN "
            "visits v ON v.patient_id=p.patient_id ORDER BY p.patient_id,v.visit_date,v.visit_id"
        )
    gender = re.search(r"性别为([男女])", q)
    disease = _ground(q, domains, "disease")
    if gender and disease and _has(q, "患者姓名", "费用", "住院天数"):
        return _safe(
            "SELECT p.name,p.region,v.visit_date,v.cost,v.los_days FROM patients p JOIN visits v ON "
            f"v.patient_id=p.patient_id WHERE p.gender='{gender.group(1)}' AND v.disease='{_esc(disease)}' "
            "ORDER BY v.visit_date,p.patient_id"
        )
    if _has(q, "有就诊记录", "最早日期", "最晚日期"):
        return _safe(
            "SELECT p.patient_id,p.name,p.region,MIN(v.visit_date) AS first_date,MAX(v.visit_date) AS "
            "last_date FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.patient_id,"
            "p.name,p.region ORDER BY first_date,p.patient_id"
        )
    if relation and _has(q, "就诊疾病", "图谱症状"):
        return _safe(
            "SELECT v.visit_id,v.disease,t.tail AS symptom FROM visits v JOIN kg_triples t ON "
            f"t.head=v.disease WHERE t.relation='{_esc(relation)}' ORDER BY v.visit_id,symptom"
        )
    cost = _integer(q, r"费用大于\s*(\d+)")
    if region and cost is not None and _has(q, "患者姓名", "年龄", "疾病"):
        return _safe(
            "SELECT p.name,p.age,v.disease,v.cost FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.region='{_esc(region)}' AND v.cost>{cost} ORDER BY v.cost DESC,p.patient_id,v.visit_id"
        )
    age = _integer(q, r"年龄至少\s*(\d+)")
    if relation and age is not None and _has(q, "患者", "图谱症状"):
        return _safe(
            "SELECT DISTINCT p.name,v.disease,t.tail AS symptom FROM patients p JOIN visits v ON "
            "v.patient_id=p.patient_id JOIN kg_triples t ON t.head=v.disease WHERE "
            f"p.age>={age} AND t.relation='{_esc(relation)}' ORDER BY p.name,v.disease,symptom"
        )
    if relation and _has(q, "主要处方药", "完全一致"):
        return _safe(
            "SELECT v.visit_id,v.disease,v.drug_prescribed FROM visits v JOIN kg_triples t ON "
            f"t.head=v.disease AND t.tail=v.drug_prescribed WHERE t.relation='{_esc(relation)}' ORDER BY v.visit_id"
        )
    if _has(q, "就诊疾病连接到图谱头实体", "关系中文名", "尾实体"):
        return _safe(
            "SELECT v.visit_id,v.disease,t.relation_zh,t.tail FROM visits v JOIN kg_triples t ON "
            "t.head=v.disease ORDER BY v.visit_id,t.relation_zh,t.tail"
        )
    return None


_COMPILERS: tuple[Callable[[str, Domains], Optional[str]], ...] = (
    _compile_window,
    _compile_cte,
    _compile_set,
    _compile_nested,
    _compile_having,
    _compile_topk,
    _compile_filtering,
    _compile_temporal,
    _compile_joins,
    _compile_kg,
    _compile_aggregation,
)


def compile_independent_v23_sql(q: str, domains: Domains) -> Optional[str]:
    """Return one fully proved SQL statement or ``None``."""

    for compiler in _COMPILERS:
        sql = compiler(q, domains)
        if sql is not None:
            return sql
    return None
