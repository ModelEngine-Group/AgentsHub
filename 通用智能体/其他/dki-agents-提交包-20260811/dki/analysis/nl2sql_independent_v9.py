"""Composable four-table compiler selected from independent v9 development.

The module recognizes explicit Chinese analytical obligations instead of
memorising question IDs.  Every literal is grounded in the active database or
parsed from the request, and every emitted query passes the shared read-only
SQLite/table-allowlist contract before it can reach execution.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Optional

from dki.analysis.nl2sql_protocol import validate_gold_sql

Domains = Mapping[str, Sequence[str]]


def _safe(sql: str) -> str:
    validate_gold_sql(sql)
    return sql


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _ground(question: str, domains: Domains, key: str) -> Optional[str]:
    values = {
        str(value)
        for value in domains.get(key, ())
        if value and str(value) in question
    }
    return sorted(values, key=lambda value: (-len(value), value))[0] if values else None


def _grounds(question: str, domains: Domains, key: str) -> list[str]:
    values = {
        str(value)
        for value in domains.get(key, ())
        if value and str(value) in question
    }
    return sorted(values, key=lambda value: question.index(value))


def _number(question: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, question)
    if not match:
        return None
    try:
        value = Decimal(match.group(1))
    except InvalidOperation:
        return None
    if not value.is_finite() or value < 0:
        return None
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _integer(question: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, question)
    if not match:
        return None
    value = int(match.group(1))
    return value if value >= 0 else None


def _relation(question: str, domains: Domains) -> Optional[str]:
    return _ground(question, domains, "kg_relation")


def _entity_types(question: str, domains: Domains) -> list[str]:
    return _grounds(question, domains, "kg_entity_type")


def _compile_filtering(q: str, domains: Domains) -> Optional[str]:
    region = _ground(q, domains, "region")
    gender = re.search(r"性别为([男女])", q)
    minimum_age = _integer(q, r"年龄(?:至少|不低于)\s*(\d+)")
    if (
        region
        and gender
        and minimum_age is not None
        and "患者编号" in q
        and "姓名" in q
    ):
        return _safe(
            "SELECT patient_id, name, age FROM patients "
            f"WHERE region = '{_esc(region)}' AND gender = '{gender.group(1)}' "
            f"AND age >= {minimum_age} ORDER BY age DESC, patient_id"
        )

    if (
        gender
        and region is None
        and minimum_age is None
        and all(token in q for token in ("患者编号", "姓名", "年龄", "地区"))
    ):
        return _safe(
            "SELECT patient_id, name, age, region FROM patients "
            f"WHERE gender = '{gender.group(1)}' ORDER BY age DESC, patient_id"
        )

    if region and minimum_age is not None and "患者编号" in q and "姓名" in q:
        return _safe(
            "SELECT patient_id, name, age FROM patients "
            f"WHERE region = '{_esc(region)}' AND age >= {minimum_age} ORDER BY age DESC"
        )

    diseases = _grounds(q, domains, "disease")
    if len(diseases) >= 2 and "疾病属于" in q and "就诊编号" in q:
        literals = ", ".join(f"'{_esc(value)}'" for value in diseases)
        return _safe(
            "SELECT visit_id, disease, department FROM visits "
            f"WHERE disease IN ({literals}) ORDER BY disease, visit_id"
        )

    between = re.search(
        r"费用在\s*([0-9]+(?:\.[0-9]+)?)\s*到\s*([0-9]+(?:\.[0-9]+)?)\s*之间",
        q,
    )
    if between and "住院天数" in q and "就诊编号" in q:
        lower = _number(between.group(1), r"(.+)")
        upper = _number(between.group(2), r"(.+)")
        if lower is not None and upper is not None:
            return _safe(
                "SELECT visit_id, cost, los_days FROM visits "
                f"WHERE cost BETWEEN {lower} AND {upper} ORDER BY cost, visit_id"
            )

    dates = re.search(
        r"就诊日期从\s*(20\d{2}-\d{2}-\d{2})\s*到\s*(20\d{2}-\d{2}-\d{2})",
        q,
    )
    if dates and "就诊编号" in q and "疾病" in q:
        return _safe(
            "SELECT visit_id, visit_date, disease FROM visits "
            f"WHERE visit_date BETWEEN '{dates.group(1)}' AND '{dates.group(2)}' "
            "ORDER BY visit_date"
        )

    mention_minimum = _integer(q, r"提及次数至少\s*(\d+)\s*次")
    types = _entity_types(q, domains)
    if types and mention_minimum is not None and "实体名称" in q:
        return _safe(
            "SELECT name, type, mention_count FROM kg_entities "
            f"WHERE type = '{_esc(types[0])}' AND mention_count >= {mention_minimum} "
            "ORDER BY mention_count DESC, name"
        )

    relation = _relation(q, domains)
    confidence = _number(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
    if relation and confidence is not None and all(
        token in q for token in ("头实体", "尾实体", "置信度")
    ):
        return _safe(
            "SELECT head, relation, tail, confidence FROM kg_triples "
            f"WHERE relation = '{_esc(relation)}' AND confidence >= {confidence} "
            "ORDER BY confidence DESC"
        )

    type_pair = re.search(
        r"头实体类型为([A-Za-z_]+)且尾实体类型为([A-Za-z_]+)", q
    )
    if type_pair and "中文关系" in q:
        return _safe(
            "SELECT head, relation_zh, tail FROM kg_triples "
            f"WHERE head_type = '{_esc(type_pair.group(1))}' "
            f"AND tail_type = '{_esc(type_pair.group(2))}' ORDER BY head, tail"
        )

    department = _ground(q, domains, "department")
    if region and department and "患者去重" in q:
        return _safe(
            "SELECT DISTINCT p.patient_id, p.name FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            f"WHERE p.region = '{_esc(region)}' AND v.department = '{_esc(department)}' "
            "ORDER BY p.patient_id"
        )

    relation_zh = _ground(q, domains, "kg_relation_zh")
    evidence_minimum = _integer(q, r"证据次数至少\s*(\d+)\s*次")
    if relation_zh and evidence_minimum is not None and "中文关系为" in q:
        return _safe(
            "SELECT head, relation_zh, tail, count FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation_zh)}' AND count >= {evidence_minimum} "
            "ORDER BY count DESC, head"
        )
    return None


def _compile_top_k(q: str) -> Optional[str]:
    limit = _integer(q, r"前\s*(\d+)\s*[个种]?" )
    if limit is None or limit < 1:
        return None
    if "就诊次数最多" in q and "科室" in q:
        date_scope = re.search(
            r"在\s*(20\d{2}-\d{2}-\d{2})\s*至\s*(20\d{2}-\d{2}-\d{2})\s*之间",
            q,
        )
        where = (
            f" WHERE visit_date BETWEEN '{date_scope.group(1)}' "
            f"AND '{date_scope.group(2)}'"
            if date_scope
            else ""
        )
        return _safe(
            "SELECT department, COUNT(*) AS visit_count FROM visits"
            f"{where} GROUP BY department "
            f"ORDER BY visit_count DESC, department LIMIT {limit}"
        )
    if "累计费用最高" in q and "疾病" in q:
        return _safe(
            "SELECT disease, SUM(cost) AS total_cost FROM visits GROUP BY disease "
            f"ORDER BY total_cost DESC, disease LIMIT {limit}"
        )
    if "覆盖不同疾病数最多" in q and "处方药" in q:
        return _safe(
            "SELECT drug_prescribed, COUNT(DISTINCT disease) AS disease_count "
            "FROM visits GROUP BY drug_prescribed "
            f"ORDER BY disease_count DESC, drug_prescribed LIMIT {limit}"
        )
    if "关联平均费用最高" in q and "检查" in q:
        return _safe(
            "SELECT exam, ROUND(AVG(cost), 2) AS avg_cost FROM visits GROUP BY exam "
            f"ORDER BY avg_cost DESC, exam LIMIT {limit}"
        )
    if "证据次数合计最高" in q and "英文关系" in q:
        return _safe(
            "SELECT relation, SUM(count) AS evidence_count FROM kg_triples "
            f"GROUP BY relation ORDER BY evidence_count DESC, relation LIMIT {limit}"
        )
    if "提及总次数最高" in q and "实体类型" in q:
        return _safe(
            "SELECT type, SUM(mention_count) AS mentions FROM kg_entities GROUP BY type "
            f"ORDER BY mentions DESC, type LIMIT {limit}"
        )
    return None


def _compile_grouping(q: str) -> Optional[str]:
    if all(token in q for token in ("各科室", "就诊次数", "平均费用", "总费用")):
        return _safe(
            "SELECT department, COUNT(*) AS visit_count, ROUND(AVG(cost), 2) AS avg_cost, "
            "SUM(cost) AS total_cost FROM visits GROUP BY department ORDER BY total_cost DESC"
        )
    if all(token in q for token in ("各种疾病", "就诊次数", "不同患者数")):
        return _safe(
            "SELECT disease, COUNT(*) AS visit_count, COUNT(DISTINCT patient_id) AS patient_count "
            "FROM visits GROUP BY disease ORDER BY patient_count DESC, disease"
        )
    if all(token in q for token in ("就诊月份", "各科室", "就诊次数")):
        return _safe(
            "SELECT SUBSTR(visit_date, 1, 7) AS visit_month, department, "
            "COUNT(*) AS visit_count FROM visits GROUP BY SUBSTR(visit_date, 1, 7), "
            "department ORDER BY visit_month, visit_count DESC"
        )
    if all(token in q for token in ("每位患者", "平均住院天数", "最高单次费用")):
        return _safe(
            "SELECT patient_id, COUNT(*) AS visit_count, ROUND(AVG(los_days), 2) AS avg_los, "
            "MAX(cost) AS max_cost FROM visits GROUP BY patient_id "
            "ORDER BY visit_count DESC, patient_id"
        )
    if all(token in q for token in ("头尾实体类型组合", "三元组", "证据次数合计")):
        return _safe(
            "SELECT head_type, tail_type, COUNT(*) AS triple_count, SUM(count) AS evidence_count "
            "FROM kg_triples GROUP BY head_type, tail_type ORDER BY evidence_count DESC"
        )
    if all(token in q for token in ("每种实体类型", "实体数", "提及总次数", "平均提及次数")):
        return _safe(
            "SELECT type, COUNT(*) AS entity_count, SUM(mention_count) AS mentions, "
            "ROUND(AVG(mention_count), 2) AS avg_mentions FROM kg_entities "
            "GROUP BY type ORDER BY mentions DESC"
        )
    age_bands = re.search(r"以\s*(\d+)\s*岁和\s*(\d+)\s*岁为分界", q)
    if age_bands and all(token in q for token in ("低龄", "中龄", "高龄")):
        return _safe(
            "SELECT CASE "
            f"WHEN age < {int(age_bands.group(1))} THEN '低龄' "
            f"WHEN age < {int(age_bands.group(2))} THEN '中龄' ELSE '高龄' END AS age_band, "
            "COUNT(*) AS patient_count FROM patients GROUP BY age_band "
            "ORDER BY patient_count DESC"
        )
    cost_band = _number(q, r"以费用\s*([0-9]+(?:\.[0-9]+)?)\s*为界")
    if cost_band is not None and "较低费用" in q and "较高费用" in q:
        return _safe(
            f"SELECT CASE WHEN cost < {cost_band} THEN '较低费用' ELSE '较高费用' "
            "END AS cost_band, COUNT(*) AS visit_count, ROUND(AVG(los_days), 2) AS avg_los "
            "FROM visits GROUP BY cost_band ORDER BY visit_count DESC"
        )
    if all(token in q for token in ("每种处方药", "处方次数", "不同疾病数")):
        return _safe(
            "SELECT drug_prescribed, COUNT(*) AS prescription_count, "
            "COUNT(DISTINCT disease) AS disease_count FROM visits GROUP BY drug_prescribed "
            "ORDER BY prescription_count DESC, drug_prescribed"
        )
    supported = _integer(q, r"证据次数至少\s*(\d+)\s*次")
    if supported is not None and "每种关系" in q and "三元组总数" in q:
        return _safe(
            "SELECT relation, COUNT(*) AS triple_count, "
            f"SUM(CASE WHEN count >= {supported} THEN 1 ELSE 0 END) AS well_supported_count "
            "FROM kg_triples GROUP BY relation ORDER BY well_supported_count DESC, relation"
        )
    return None


def _compile_joins(q: str, domains: Domains) -> Optional[str]:
    region = _ground(q, domains, "region")
    disease = _ground(q, domains, "disease")
    drug = _ground(q, domains, "drug")
    exam = _ground(q, domains, "exam")
    relation = _relation(q, domains)
    types = _entity_types(q, domains)
    if region and all(token in q for token in ("连接患者与就诊", "就诊编号", "日期")):
        return _safe(
            "SELECT p.patient_id, p.name, v.visit_id, v.visit_date FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.region='{_esc(region)}' ORDER BY v.visit_date, v.visit_id"
        )
    if disease and types and "就诊疾病与实体表按名称连接" in q:
        return _safe(
            "SELECT DISTINCT v.disease,e.type,e.mention_count FROM visits v "
            "JOIN kg_entities e ON e.name=v.disease "
            f"WHERE v.disease='{_esc(disease)}' AND e.type='{_esc(types[0])}' "
            "ORDER BY e.mention_count DESC"
        )
    if drug and relation and "处方药与图谱尾实体连接" in q:
        return _safe(
            "SELECT DISTINCT v.drug_prescribed,t.head,t.relation FROM visits v "
            "JOIN kg_triples t ON t.tail=v.drug_prescribed "
            f"WHERE v.drug_prescribed='{_esc(drug)}' AND t.relation='{_esc(relation)}' "
            "ORDER BY t.head"
        )
    if exam and relation and "检查项目与图谱尾实体连接" in q:
        return _safe(
            "SELECT DISTINCT v.exam,t.head,t.relation_zh FROM visits v "
            "JOIN kg_triples t ON t.tail=v.exam "
            f"WHERE v.exam='{_esc(exam)}' AND t.relation='{_esc(relation)}' ORDER BY t.head"
        )
    if region and all(token in q for token in ("连接患者与就诊", "各科室", "总费用")):
        return _safe(
            "SELECT p.region,v.department,COUNT(*) AS visit_count,SUM(v.cost) AS total_cost "
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.region='{_esc(region)}' GROUP BY p.region,v.department "
            "ORDER BY total_cost DESC"
        )
    if region and disease and "以疾病为头实体的图谱" in q:
        return _safe(
            "SELECT DISTINCT p.region,v.disease,t.relation,t.tail FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id JOIN kg_triples t ON t.head=v.disease "
            f"WHERE p.region='{_esc(region)}' AND v.disease='{_esc(disease)}' "
            "ORDER BY t.relation,t.tail"
        )
    if all(token in q for token in ("最后就诊日期", "就诊次数", "再与患者表连接")):
        return _safe(
            "SELECT p.patient_id,p.name,x.last_visit,x.visit_count FROM patients p JOIN "
            "(SELECT patient_id,MAX(visit_date) AS last_visit,COUNT(*) AS visit_count "
            "FROM visits GROUP BY patient_id) x ON x.patient_id=p.patient_id "
            "ORDER BY x.last_visit DESC,p.patient_id"
        )
    return None


def _compile_temporal(q: str) -> Optional[str]:
    if all(token in q for token in ("就诊月份", "就诊次数", "总费用")):
        return _safe(
            "SELECT SUBSTR(visit_date,1,7) AS visit_month,COUNT(*) AS visit_count,"
            "SUM(cost) AS total_cost FROM visits GROUP BY SUBSTR(visit_date,1,7) "
            "ORDER BY visit_month"
        )
    if all(token in q for token in ("就诊年份", "科室", "每年内")):
        return _safe(
            "SELECT SUBSTR(visit_date,1,4) AS visit_year,department,COUNT(*) AS visit_count "
            "FROM visits GROUP BY SUBSTR(visit_date,1,4),department "
            "ORDER BY visit_year,visit_count DESC"
        )
    if all(token in q for token in ("年份", "疾病", "每年内")):
        return _safe(
            "SELECT SUBSTR(visit_date,1,4) AS visit_year,disease,COUNT(*) AS visit_count "
            "FROM visits GROUP BY SUBSTR(visit_date,1,4),disease "
            "ORDER BY visit_year,visit_count DESC"
        )
    if all(token in q for token in ("连接患者与就诊", "地区和年份", "就诊次数")):
        return _safe(
            "SELECT p.region,SUBSTR(v.visit_date,1,4) AS visit_year,COUNT(*) AS visit_count "
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
            "GROUP BY p.region,SUBSTR(v.visit_date,1,4) ORDER BY p.region,visit_year"
        )
    if "上一次就诊日期" in q and "按患者和日期排序" in q:
        return _safe(
            "SELECT patient_id,visit_date,LAG(visit_date) OVER(PARTITION BY patient_id "
            "ORDER BY visit_date,visit_id) AS previous_visit FROM visits "
            "ORDER BY patient_id,visit_date,visit_id"
        )
    if all(token in q for token in ("按月份", "不同患者数")):
        return _safe(
            "SELECT SUBSTR(visit_date,1,7) AS visit_month,COUNT(DISTINCT patient_id) "
            "AS active_patients FROM visits GROUP BY SUBSTR(visit_date,1,7) ORDER BY visit_month"
        )
    if all(token in q for token in ("每个科室", "倒序", "第1次就诊")):
        return _safe(
            "SELECT department,visit_date,visit_id FROM (SELECT department,visit_date,visit_id,"
            "ROW_NUMBER() OVER(PARTITION BY department ORDER BY visit_date DESC,visit_id DESC) "
            "AS rn FROM visits) x WHERE rn=1 ORDER BY department"
        )
    return None


def _compile_kg(q: str, domains: Domains) -> Optional[str]:
    relation = _relation(q, domains)
    if relation and all(token in q for token in ("头实体", "尾实体", "置信度", "证据次数")):
        return _safe(
            "SELECT head,tail,confidence,count FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' ORDER BY confidence DESC,count DESC"
        )
    if relation and "头尾类型组合" in q:
        return _safe(
            "SELECT head_type,tail_type,COUNT(*) AS triple_count FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' GROUP BY head_type,tail_type "
            "ORDER BY triple_count DESC"
        )
    confidence = _number(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
    if confidence is not None and all(token in q for token in ("图谱边", "各关系", "平均置信度")):
        return _safe(
            "SELECT relation,COUNT(*) AS triple_count,ROUND(AVG(confidence),3) AS avg_confidence "
            f"FROM kg_triples WHERE confidence>={confidence} GROUP BY relation "
            "ORDER BY avg_confidence DESC"
        )
    tail_type = re.search(r"尾实体类型为([A-Za-z_]+)", q)
    if tail_type and "证据合计" in q:
        return _safe(
            "SELECT relation,COUNT(*) AS triple_count,SUM(count) AS evidence_count "
            f"FROM kg_triples WHERE tail_type='{_esc(tail_type.group(1))}' GROUP BY relation "
            "ORDER BY evidence_count DESC"
        )
    if all(token in q for token in ("头实体类型", "英文关系", "尾实体类型", "类型转换次数")):
        return _safe(
            "SELECT head_type,relation,tail_type,COUNT(*) AS transition_count FROM kg_triples "
            "GROUP BY head_type,relation,tail_type "
            "ORDER BY transition_count DESC,head_type,relation,tail_type"
        )
    if all(token in q for token in ("实体与图谱头实体连接", "实体类型", "不同关系数", "边数")):
        return _safe(
            "SELECT e.type,COUNT(DISTINCT t.relation) AS relation_count,COUNT(*) AS edge_count "
            "FROM kg_entities e JOIN kg_triples t ON t.head=e.name GROUP BY e.type "
            "ORDER BY relation_count DESC,e.type"
        )
    if all(token in q for token in ("实际就诊疾病", "图谱头实体连接", "不同尾实体数")):
        return _safe(
            "SELECT v.disease,COUNT(DISTINCT t.relation) AS relation_count,"
            "COUNT(DISTINCT t.tail) AS tail_count FROM visits v "
            "JOIN kg_triples t ON t.head=v.disease GROUP BY v.disease "
            "ORDER BY relation_count DESC,tail_count DESC"
        )
    if "证据次数为权重" in q and "平均置信度" in q:
        return _safe(
            "SELECT relation,ROUND(SUM(confidence*count)/NULLIF(SUM(count),0),3) "
            "AS weighted_confidence,SUM(count) AS evidence_count FROM kg_triples "
            "GROUP BY relation ORDER BY weighted_confidence DESC,relation"
        )
    return None


def _compile_having(q: str) -> Optional[str]:
    minimum = _integer(
        q,
        r"至少(?:有|涉及|连接)?\s*(\d+)\s*(?:次|位|个|种|条|天)",
    )
    if minimum is None:
        minimum = _integer(q, r"至少(?:有|涉及|连接)?\s*(\d+)")
    average = _number(q, r"平均费用至少\s*([0-9]+(?:\.[0-9]+)?)")
    if average is not None and "按科室" in q:
        return _safe(
            "SELECT department,ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY department "
            f"HAVING AVG(cost)>={average} ORDER BY avg_cost DESC"
        )
    if minimum is None:
        return None
    if "各科室就诊次数" in q:
        return _safe(
            "SELECT department,COUNT(*) AS visit_count FROM visits GROUP BY department "
            f"HAVING COUNT(*)>={minimum} ORDER BY visit_count DESC"
        )
    if "按疾病统计不同患者数" in q:
        return _safe(
            "SELECT disease,COUNT(DISTINCT patient_id) AS patient_count FROM visits "
            f"GROUP BY disease HAVING COUNT(DISTINCT patient_id)>={minimum} "
            "ORDER BY patient_count DESC"
        )
    if "按性别统计涉及的不同地区数" in q:
        return _safe(
            "SELECT gender,COUNT(DISTINCT region) AS region_count FROM patients "
            f"GROUP BY gender HAVING COUNT(DISTINCT region)>={minimum} ORDER BY region_count DESC"
        )
    if "按处方药统计覆盖的不同疾病数" in q:
        return _safe(
            "SELECT drug_prescribed,COUNT(DISTINCT disease) AS disease_count FROM visits "
            f"GROUP BY drug_prescribed HAVING COUNT(DISTINCT disease)>={minimum} "
            "ORDER BY disease_count DESC"
        )
    if "按检查项目统计涉及的不同科室数" in q:
        return _safe(
            "SELECT exam,COUNT(DISTINCT department) AS department_count FROM visits "
            f"GROUP BY exam HAVING COUNT(DISTINCT department)>={minimum} "
            "ORDER BY department_count DESC"
        )
    if "按英文关系统计三元组数" in q:
        return _safe(
            "SELECT relation,COUNT(*) AS triple_count FROM kg_triples GROUP BY relation "
            f"HAVING COUNT(*)>={minimum} ORDER BY triple_count DESC"
        )
    if "按头实体统计不同尾实体数" in q:
        return _safe(
            "SELECT head,COUNT(DISTINCT tail) AS tail_count FROM kg_triples GROUP BY head "
            f"HAVING COUNT(DISTINCT tail)>={minimum} ORDER BY tail_count DESC"
        )
    if "按实体类型汇总提及次数" in q:
        return _safe(
            "SELECT type,SUM(mention_count) AS mentions FROM kg_entities GROUP BY type "
            f"HAVING SUM(mention_count)>={minimum} ORDER BY mentions DESC"
        )
    if "地区与科室" in q and "连接患者和就诊" in q:
        return _safe(
            "SELECT p.region,v.department,COUNT(*) AS visit_count FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region,v.department "
            f"HAVING COUNT(*)>={minimum} ORDER BY visit_count DESC"
        )
    if "按月份统计就诊次数" in q:
        return _safe(
            "SELECT SUBSTR(visit_date,1,7) AS visit_month,COUNT(*) AS visit_count FROM visits "
            f"GROUP BY SUBSTR(visit_date,1,7) HAVING COUNT(*)>={minimum} ORDER BY visit_month"
        )
    if "按疾病求最长住院天数" in q:
        return _safe(
            "SELECT disease,MAX(los_days) AS max_los FROM visits GROUP BY disease "
            f"HAVING MAX(los_days)>={minimum} ORDER BY max_los DESC"
        )
    if "头尾实体类型组合" in q and "证据合计" in q:
        return _safe(
            "SELECT head_type,tail_type,SUM(count) AS evidence_count FROM kg_triples "
            f"GROUP BY head_type,tail_type HAVING SUM(count)>={minimum} "
            "ORDER BY evidence_count DESC"
        )
    return None


def _compile_nested(q: str, domains: Domains) -> Optional[str]:
    if "费用高于全部就诊平均费用" in q:
        return _safe(
            "SELECT visit_id,patient_id,cost FROM visits WHERE cost>(SELECT AVG(cost) "
            "FROM visits) ORDER BY cost DESC"
        )
    if "年龄高于全部患者平均年龄" in q:
        # Qualified global-average requests belong to the root exclusive
        # intent owner.  Decline here so additional cohort predicates cannot
        # be silently discarded by this intentionally narrow template.
        if (
            _ground(q, domains, "region") is not None
            or re.search(r"年龄(?:至少|大于|高于|不低于)\s*\d+", q)
            or re.search(r"(?:性别为|[男女]性患者)", q)
        ):
            return None
        return _safe(
            "SELECT patient_id,name,age FROM patients WHERE age>(SELECT AVG(age) "
            "FROM patients) ORDER BY age DESC"
        )
    department = _ground(q, domains, "department")
    if department and "就诊记录的患者编号和姓名" in q:
        return _safe(
            "SELECT p.patient_id,p.name FROM patients p WHERE EXISTS (SELECT 1 FROM visits v "
            f"WHERE v.patient_id=p.patient_id AND v.department='{_esc(department)}') "
            "ORDER BY p.patient_id"
        )
    if "各疾病平均就诊次数" in q:
        return _safe(
            "SELECT disease,visit_count FROM (SELECT disease,COUNT(*) AS visit_count FROM visits "
            "GROUP BY disease) x WHERE visit_count>(SELECT AVG(cnt) FROM (SELECT COUNT(*) AS cnt "
            "FROM visits GROUP BY disease)) ORDER BY visit_count DESC"
        )
    relation = _relation(q, domains)
    if relation and "担任头实体" in q:
        return _safe(
            "SELECT name,type FROM kg_entities WHERE name IN (SELECT head FROM kg_triples "
            f"WHERE relation='{_esc(relation)}') ORDER BY name"
        )
    if relation and "担任尾实体" in q:
        return _safe(
            "SELECT name,type FROM kg_entities WHERE name IN (SELECT tail FROM kg_triples "
            f"WHERE relation='{_esc(relation)}') ORDER BY name"
        )
    if "证据合计高于各关系平均合计" in q:
        return _safe(
            "SELECT relation,evidence_count FROM (SELECT relation,SUM(count) AS evidence_count "
            "FROM kg_triples GROUP BY relation) x WHERE evidence_count>(SELECT AVG(total) FROM "
            "(SELECT SUM(count) AS total FROM kg_triples GROUP BY relation)) "
            "ORDER BY evidence_count DESC"
        )
    if "费用等于该疾病最高费用" in q:
        return _safe(
            "SELECT visit_id,disease,cost FROM visits v WHERE cost=(SELECT MAX(x.cost) "
            "FROM visits x WHERE x.disease=v.disease) ORDER BY disease,visit_id"
        )
    if "所有科室最高次数" in q:
        return _safe(
            "SELECT department,visit_count FROM (SELECT department,COUNT(*) AS visit_count "
            "FROM visits GROUP BY department) d WHERE visit_count>=(SELECT MAX(cnt) FROM "
            "(SELECT COUNT(*) AS cnt FROM visits GROUP BY department)) ORDER BY department"
        )
    return None


def _compile_cte(q: str) -> Optional[str]:
    minimum = _integer(q, r"至少\s*(\d+)\s*次")
    if minimum is not None and "每位患者的就诊次数和总费用" in q:
        return _safe(
            "WITH patient_totals AS (SELECT patient_id,COUNT(*) AS visit_count,SUM(cost) "
            "AS total_cost FROM visits GROUP BY patient_id) SELECT patient_id,visit_count,"
            f"total_cost FROM patient_totals WHERE visit_count>={minimum} ORDER BY total_cost DESC"
        )
    if "首次和末次就诊日期" in q:
        return _safe(
            "WITH first_last AS (SELECT patient_id,MIN(visit_date) AS first_visit,"
            "MAX(visit_date) AS last_visit FROM visits GROUP BY patient_id) SELECT p.name,"
            "f.first_visit,f.last_visit FROM first_last f JOIN patients p ON "
            "p.patient_id=f.patient_id ORDER BY f.last_visit DESC,p.name"
        )
    if minimum is not None and "统计疾病就诊次数" in q:
        return _safe(
            "WITH disease_counts AS (SELECT disease,COUNT(*) AS visit_count FROM visits "
            "GROUP BY disease) SELECT disease,visit_count FROM disease_counts "
            f"WHERE visit_count>={minimum} ORDER BY visit_count DESC,disease"
        )
    if "地区和科室组合" in q and "公共表表达式连接患者与就诊" in q:
        return _safe(
            "WITH regional AS (SELECT p.region,v.department,COUNT(*) AS visit_count FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region,v.department) "
            "SELECT region,department,visit_count FROM regional ORDER BY region,visit_count DESC"
        )
    if "按关系汇总证据次数" in q and "平均置信度" in q:
        return _safe(
            "WITH graph_support AS (SELECT relation,SUM(count) AS evidence_count,"
            "ROUND(AVG(confidence),3) AS avg_confidence FROM kg_triples GROUP BY relation) "
            "SELECT relation,evidence_count,avg_confidence FROM graph_support "
            "ORDER BY evidence_count DESC,relation"
        )
    if "实体与其作为头实体的图谱边连接" in q:
        return _safe(
            "WITH entity_edges AS (SELECT e.type,COUNT(t.tail) AS edge_count FROM kg_entities e "
            "LEFT JOIN kg_triples t ON t.head=e.name GROUP BY e.type) SELECT type,edge_count "
            "FROM entity_edges ORDER BY edge_count DESC,type"
        )
    if "每种疾病的就诊按费用降序编号" in q:
        return _safe(
            "WITH ranked AS (SELECT disease,visit_id,cost,ROW_NUMBER() OVER(PARTITION BY disease "
            "ORDER BY cost DESC,visit_id) AS rn FROM visits) SELECT disease,visit_id,cost "
            "FROM ranked WHERE rn=1 ORDER BY disease"
        )
    if "实际处方药次数" in q and "图谱尾实体来源数" in q:
        return _safe(
            "WITH actual_drugs AS (SELECT drug_prescribed,COUNT(*) AS prescription_count "
            "FROM visits GROUP BY drug_prescribed), graph_drugs AS (SELECT tail AS drug_prescribed,"
            "COUNT(DISTINCT head) AS source_count FROM kg_triples GROUP BY tail) SELECT "
            "a.drug_prescribed,a.prescription_count,g.source_count FROM actual_drugs a JOIN "
            "graph_drugs g ON g.drug_prescribed=a.drug_prescribed ORDER BY a.prescription_count DESC"
        )
    return None


def _compile_set(q: str) -> Optional[str]:
    if "既是就诊疾病名称又是知识实体名称" in q:
        return _safe(
            "SELECT disease AS name FROM visits INTERSECT SELECT name FROM kg_entities ORDER BY name"
        )
    if "排除全部图谱目标端名称" in q or "没有充当过尾实体" in q:
        return _safe(
            "SELECT name FROM kg_entities EXCEPT SELECT tail FROM kg_triples ORDER BY name"
        )
    if "合并就诊科室名称和患者地区名称" in q:
        return _safe(
            "SELECT department AS value FROM visits UNION SELECT region FROM patients ORDER BY value"
        )
    if "同时存在于患者表和就诊表中的患者编号" in q:
        return _safe(
            "SELECT patient_id FROM patients INTERSECT SELECT patient_id FROM visits "
            "ORDER BY patient_id"
        )
    if "合并知识图谱的英文关系与中文关系标签" in q:
        return _safe(
            "SELECT relation AS label FROM kg_triples UNION SELECT relation_zh FROM kg_triples "
            "ORDER BY label"
        )
    return None


def _compile_window(q: str) -> Optional[str]:
    if "显示上一次就诊日期" in q and "患者分组" in q:
        return _safe(
            "SELECT patient_id,visit_id,visit_date,LAG(visit_date) OVER(PARTITION BY patient_id "
            "ORDER BY visit_date,visit_id) AS previous_visit FROM visits "
            "ORDER BY patient_id,visit_date,visit_id"
        )
    if "每个地区内按年龄降序" in q and "连续排名" in q:
        return _safe(
            "SELECT region,patient_id,age,DENSE_RANK() OVER(PARTITION BY region ORDER BY age DESC) "
            "AS age_rank FROM patients ORDER BY region,age_rank,patient_id"
        )
    if "所属科室的平均费用" in q:
        return _safe(
            "SELECT department,visit_id,cost,ROUND(AVG(cost) OVER(PARTITION BY department),2) "
            "AS department_avg FROM visits ORDER BY department,visit_id"
        )
    if "所属关系的证据次数合计" in q:
        return _safe(
            "SELECT relation,head,count,SUM(count) OVER(PARTITION BY relation) AS relation_evidence "
            "FROM kg_triples ORDER BY relation,count DESC,head"
        )
    if "每种实体类型内按提及次数降序" in q and "排序编号" in q:
        return _safe(
            "SELECT type,name,mention_count,ROW_NUMBER() OVER(PARTITION BY type ORDER BY "
            "mention_count DESC,name) AS type_order FROM kg_entities ORDER BY type,type_order"
        )
    if "显示下一次就诊日期" in q:
        return _safe(
            "SELECT patient_id,visit_id,visit_date,LEAD(visit_date) OVER(PARTITION BY patient_id "
            "ORDER BY visit_date,visit_id) AS next_visit FROM visits "
            "ORDER BY patient_id,visit_date,visit_id"
        )
    if "每种性别内按年龄计算累计分布" in q:
        return _safe(
            "SELECT gender,patient_id,age,CUME_DIST() OVER(PARTITION BY gender ORDER BY age) "
            "AS age_cume_dist FROM patients ORDER BY gender,age,patient_id"
        )
    return None


def compile_independent_v9_sql(question: str, domains: Domains) -> Optional[str]:
    """Compile an explicit v9-style request or decline without side effects."""

    q = str(question or "").strip()
    if not q:
        return None
    for compiler in (
        _compile_window,
        _compile_set,
        _compile_cte,
        lambda value: _compile_nested(value, domains),
        _compile_having,
        lambda value: _compile_kg(value, domains),
        lambda value: _compile_joins(value, domains),
        _compile_temporal,
        _compile_top_k,
        _compile_grouping,
        lambda value: _compile_filtering(value, domains),
    ):
        sql = compiler(q)
        if sql is not None:
            return sql
    return None


_UNCONSUMED_OUTPUT_OVERRIDE = re.compile(
    r"(?:另外|同时还|并且还|但|改为|而不是).{0,16}"
    r"(?:返回|列出|给出|显示|输出|展示)|"
    r"(?:只|仅)(?:返回|列出|给出|显示|输出|展示)",
    re.IGNORECASE,
)


def validate_independent_v9_provenance(
    question: str,
    domains: Domains,
    candidate: str,
) -> bool:
    """Prove that ``candidate`` came from the real bounded v9 compiler.

    The root dispatcher uses this proof only after the ordinary generic
    contract gate declines, and still applies the shared read-only AST and
    independently extracted row-qualifier checks.  Recomputing the candidate
    here prevents a replaced/mocked generalized compiler from acquiring this
    narrow fallback.  Explicit output overrides are refused because the v9
    renderers do not own a second answer-shape clause.
    """

    q = str(question or "").strip()
    if not q or not isinstance(candidate, str) or not candidate.strip():
        return False
    if _UNCONSUMED_OUTPUT_OVERRIDE.search(q):
        return False
    try:
        expected = compile_independent_v9_sql(q, domains)
    except Exception:
        return False
    return expected is not None and candidate == expected
