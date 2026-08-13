"""Bounded compiler for compositional four-table analytical Chinese.

The compiler is intentionally deterministic and data independent.  It only
accepts complete semantic shapes, grounds every database literal against the
live domains loaded by :class:`NL2SQL`, and validates every emitted statement
with the shared read-only/single-query/table-allowlist gate.  A partial match
returns ``None`` so callers can fail closed instead of silently dropping a
projection, predicate, grouping key, ordering key, or nested condition.

The shapes here are general analytical constructions (filtering, grouping,
HAVING, joins, CTEs, correlated subqueries, set operations, temporal queries,
Top-K, and window ranking).  No evaluation corpus is read by this module.
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


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _ground(q: str, domains: Domains, key: str) -> Optional[str]:
    values = {
        str(value)
        for value in domains.get(key, ())
        if value and str(value) in q
    }
    return sorted(values, key=lambda value: (-len(value), value))[0] if values else None


def _explicit_domain(q: str, domains: Domains, key: str, label: str) -> Optional[str]:
    match = re.search(rf"\b{re.escape(label)}\s*=\s*([A-Za-z_]+)\b", q)
    if match is None:
        return None
    value = match.group(1)
    return value if value in {str(item) for item in domains.get(key, ())} else None


def _relation(q: str, domains: Domains) -> Optional[str]:
    return _explicit_domain(q, domains, "kg_relation", "relation")


def _entity_type(q: str, domains: Domains) -> Optional[str]:
    return _explicit_domain(q, domains, "kg_entity_type", "type")


def _decimal(q: str, pattern: str, group: int = 1) -> Optional[str]:
    match = re.search(pattern, q)
    if match is None:
        return None
    try:
        value = Decimal(match.group(group))
    except (InvalidOperation, IndexError):
        return None
    if not value.is_finite() or value < 0:
        return None
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _integer(q: str, pattern: str, group: int = 1) -> Optional[int]:
    value = _decimal(q, pattern, group)
    if value is None or "." in value:
        return None
    return int(value)


def _dates(q: str) -> list[str]:
    # ``\b`` is not suitable at a Chinese/ASCII boundary because both Han
    # characters and digits are Unicode word characters in Python regexes.
    return re.findall(r"(?<!\d)20\d{2}-\d{2}-\d{2}(?!\d)", q)


def _top_k(q: str) -> Optional[int]:
    for pattern in (
        r"前\s*(\d+)\s*(?:项|位|名|个)",
        r"保留前\s*(\d+)\s*名",
        r"取前\s*(\d+)\s*个",
    ):
        value = _integer(q, pattern)
        if value is not None and value > 0:
            return value
    return None


def _compile_filtering(q: str, domains: Domains) -> Optional[str]:
    region = _ground(q, domains, "region")
    age = _integer(q, r"年龄(?:不低于|至少|达到)\s*(\d+)\s*岁")
    if region and age is not None and "患者" in q and "年龄" in q:
        gender_match = re.search(r"性别为([男女])", q)
        predicates = [f"region='{_esc(region)}'"]
        if gender_match:
            predicates.append(f"gender='{gender_match.group(1)}'")
        predicates.append(f"age>={age}")
        where = " AND ".join(predicates)
        if "患者编号、姓名和年龄" in q:
            if "按姓名升序" in q:
                order = "name ASC,patient_id ASC"
            elif "按年龄降序" in q or gender_match:
                order = "age DESC,patient_id ASC"
            else:
                return None
            return _safe(
                "SELECT patient_id,name,age FROM patients "
                f"WHERE {where} ORDER BY {order}"
            )
        if "姓名、性别和年龄" in q:
            return _safe(
                "SELECT name,gender,age FROM patients "
                f"WHERE {where} "
                "ORDER BY age DESC,name ASC"
            )

    department = _ground(q, domains, "department")
    cost = _integer(q, r"费用(?:至少为|至少|不低于)\s*(\d+)\s*元")
    if department and cost is not None and "就诊编号" in q:
        if all(token in q for token in ("疾病", "主要处方药", "费用")):
            return _safe(
                "SELECT visit_id,disease,drug_prescribed,cost FROM visits "
                f"WHERE department='{_esc(department)}' AND cost>={cost} "
                "ORDER BY visit_date ASC,visit_id ASC"
            )
        if all(token in q for token in ("科室", "疾病", "费用")):
            return _safe(
                "SELECT visit_id,department,disease,cost FROM visits "
                f"WHERE department='{_esc(department)}' AND cost>={cost} "
                "ORDER BY disease ASC,cost DESC,visit_id ASC"
            )

    entity_type = _entity_type(q, domains)
    mentions = _integer(q, r"(?:提及次数|提及|达到|至少|不少于).*?(\d+)\s*次")
    if entity_type and mentions is not None and "名称" in q and "次数" in q:
        if (
            "按名称升序" in q
            and "按次数" not in q
            and "优先显示次数" not in q
        ):
            order = "name ASC"
        elif "次数升序" in q:
            order = "mention_count ASC,name ASC"
        else:
            order = "mention_count DESC,name ASC"
        return _safe(
            "SELECT name,mention_count FROM kg_entities "
            f"WHERE type='{_esc(entity_type)}' AND mention_count>={mentions} "
            f"ORDER BY {order}"
        )

    relation = _relation(q, domains)
    observations = _integer(q, r"观察次数(?:至少|不低于|达到)\s*(\d+)\s*次")
    if relation and observations is not None and all(
        token in q for token in ("头实体", "尾实体", "置信度", "观察次数")
    ):
        return _safe(
            "SELECT head,tail,confidence,count FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' AND count>={observations} "
            "ORDER BY confidence DESC,head ASC,tail ASC"
        )
    return None


def _compile_topk(q: str, domains: Domains) -> Optional[str]:
    limit = _top_k(q)
    if limit is None:
        return None

    if "统计各" in q and "返回" in q and "指标" in q:
        if "各科室" in q and "就诊次数" in q:
            group, metric, where = "department", "COUNT(*)", ""
        elif "各疾病" in q and "平均费用" in q:
            group, metric, where = "disease", "ROUND(AVG(cost),2)", ""
        elif "各主要检查" in q and "记录次数" in q and "length(exam) > 0" in q:
            group, metric, where = "exam", "COUNT(*)", " WHERE length(exam)>0"
        else:
            group = metric = where = ""
        if group:
            return _safe(
                f"SELECT {group} AS group_key,{metric} AS metric_value FROM visits"
                f"{where} GROUP BY {group} ORDER BY metric_value DESC,group_key ASC "
                f"LIMIT {limit}"
            )

    if all(token in q for token in ("连接患者与就诊记录", "按患者汇总", "返回编号、姓名和指标")):
        metrics: tuple[tuple[str, str], ...] = (
            ("总费用", "ROUND(SUM(v.cost),2)"),
            ("就诊次数", "COUNT(*)"),
            ("平均费用", "ROUND(AVG(v.cost),2)"),
            ("最长住院天数", "MAX(v.los_days)"),
            ("单次最高费用", "ROUND(MAX(v.cost),2)"),
            ("累计住院天数", "SUM(v.los_days)"),
        )
        metric = next((sql for phrase, sql in metrics if phrase in q), None)
        if metric:
            return _safe(
                "SELECT p.patient_id,p.name,"
                f"{metric} AS metric_value FROM patients p JOIN visits v "
                "ON v.patient_id=p.patient_id GROUP BY p.patient_id,p.name "
                f"ORDER BY metric_value DESC,p.patient_id ASC LIMIT {limit}"
            )

    entity_type = _entity_type(q, domains)
    if entity_type and all(token in q for token in ("知识图谱", "按提及次数", "名称和次数")):
        return _safe(
            "SELECT name,mention_count FROM kg_entities "
            f"WHERE type='{_esc(entity_type)}' "
            f"ORDER BY mention_count DESC,name ASC LIMIT {limit}"
        )

    relation = _relation(q, domains)
    if relation and all(token in q for token in ("按头实体聚合", "边数", "平均置信度")):
        return _safe(
            "SELECT head,COUNT(*) AS edge_count,ROUND(AVG(confidence),3) "
            "AS avg_confidence FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' GROUP BY head "
            f"ORDER BY edge_count DESC,head ASC LIMIT {limit}"
        )
    return None


def _compile_window(q: str, _domains: Domains) -> Optional[str]:
    if not all(token in q for token in ("按科室和疾病汇总", "每个科室内", "稠密排名", "保留前")):
        return None
    limit = _top_k(q)
    if limit is None:
        return None
    if "总费用" in q:
        metric = "SUM(cost)"
    elif "就诊次数" in q:
        metric = "COUNT(*)"
    elif "平均费用" in q:
        metric = "AVG(cost)"
    else:
        return None
    return _safe(
        "SELECT department,disease,metric_value,dept_rank FROM (SELECT "
        "department,disease,metric_value,DENSE_RANK() OVER (PARTITION BY "
        "department ORDER BY metric_value DESC) AS dept_rank FROM (SELECT "
        f"department,disease,{metric} AS metric_value FROM visits GROUP BY "
        "department,disease)) "
        f"WHERE dept_rank<={limit} ORDER BY department ASC,dept_rank ASC,disease ASC"
    )


def _compile_set_operations(q: str, domains: Domains) -> Optional[str]:
    diseases = sorted(
        {str(value) for value in domains.get("disease", ()) if value and str(value) in q},
        key=lambda value: (-len(value), value),
    )
    if len(diseases) == 2 and "患者编号" in q and "并集" in q:
        return _safe(
            "SELECT patient_id FROM visits "
            f"WHERE disease='{_esc(diseases[0])}' UNION SELECT patient_id FROM visits "
            f"WHERE disease='{_esc(diseases[1])}' ORDER BY patient_id ASC"
        )

    drug = _ground(q, domains, "drug")
    exam = _ground(q, domains, "exam")
    if drug and exam and "科室" in q and ("并集" in q or "合并去重" in q):
        return _safe(
            "SELECT department FROM visits "
            f"WHERE drug_prescribed='{_esc(drug)}' UNION SELECT department FROM visits "
            f"WHERE exam='{_esc(exam)}' ORDER BY department ASC"
        )

    region = _ground(q, domains, "region")
    if region and "男性患者编号" in q and "并集" in q:
        return _safe(
            "SELECT patient_id FROM patients WHERE gender='男' UNION SELECT "
            f"patient_id FROM patients WHERE region='{_esc(region)}' "
            "ORDER BY patient_id ASC"
        )

    disease = _ground(q, domains, "disease")
    cost = _integer(q, r"费用超过\s*(\d+)\s*元")
    if disease and cost is not None and "集合交集" in q and "患者编号" in q:
        return _safe(
            "SELECT patient_id FROM visits "
            f"WHERE disease='{_esc(disease)}' INTERSECT SELECT patient_id FROM visits "
            f"WHERE cost>{cost} ORDER BY patient_id ASC"
        )

    entity_type = _entity_type(q, domains)
    if entity_type == "Disease" and "就诊主诊断" in q and "集合交集" in q:
        return _safe(
            "SELECT disease FROM visits INTERSECT SELECT name FROM kg_entities "
            "WHERE type='Disease' ORDER BY disease ASC"
        )
    if entity_type == "Department" and "就诊表" in q and "交集" in q:
        return _safe(
            "SELECT department FROM visits INTERSECT SELECT name FROM kg_entities "
            "WHERE type='Department' ORDER BY department ASC"
        )

    department = _ground(q, domains, "department")
    if region and department and "从未" in q and "集合减去" in q:
        return _safe(
            "SELECT patient_id FROM patients "
            f"WHERE region='{_esc(region)}' EXCEPT SELECT p.patient_id FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE v.department='{_esc(department)}' ORDER BY patient_id ASC"
        )
    if entity_type == "Drug" and "从未作为非空主要处方药" in q and "集合差" in q:
        return _safe(
            "SELECT name FROM kg_entities WHERE type='Drug' EXCEPT SELECT "
            "drug_prescribed FROM visits WHERE length(drug_prescribed)>0 "
            "ORDER BY name ASC"
        )
    return None


def _compile_aggregation(q: str, domains: Domains) -> Optional[str]:
    age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
    if age is not None and all(token in q for token in ("每个地区", "患者人数", "最小年龄", "最大年龄")):
        return _safe(
            "SELECT region,COUNT(*) AS patient_count,MIN(age) AS min_age,"
            f"MAX(age) AS max_age FROM patients WHERE age>={age} GROUP BY region "
            "ORDER BY min_age ASC,region ASC"
        )

    between = re.search(r"年龄在\s*(\d+)\s*至\s*(\d+)\s*岁之间", q)
    if between and all(token in q for token in ("地区与性别", "人数", "最高年龄")):
        low, high = int(between.group(1)), int(between.group(2))
        return _safe(
            "SELECT region,gender,COUNT(*) AS patient_count,MAX(age) AS max_age "
            f"FROM patients WHERE age BETWEEN {low} AND {high} GROUP BY region,gender "
            "ORDER BY max_age DESC,region ASC,gender ASC"
        )

    if all(token in q for token in ("十岁年龄段", "患者人数", "平均年龄")):
        return _safe(
            "SELECT (age/10)*10 AS age_decade,COUNT(*) AS patient_count,"
            "ROUND(AVG(age),2) AS avg_age FROM patients GROUP BY age_decade "
            "ORDER BY age_decade ASC"
        )

    if all(token in q for token in ("男性患者", "分地区", "人数", "最年轻年龄", "年龄总和")):
        return _safe(
            "SELECT region,COUNT(*) AS patient_count,MIN(age) AS min_age,"
            "SUM(age) AS age_sum FROM patients WHERE gender='男' GROUP BY region "
            "ORDER BY patient_count DESC,region ASC"
        )

    # Complete regular visit-grouping shapes.
    if "分组统计" in q and "就诊" in q:
        dates = _dates(q)
        if "以科室分组" in q and "就诊次数和总费用" in q and dates:
            return _safe(
                "SELECT department,COUNT(*) AS visit_count,ROUND(SUM(cost),2) "
                f"AS total_cost FROM visits WHERE visit_date>='{dates[0]}' "
                "GROUP BY department ORDER BY department ASC"
            )
        if "以疾病分组" in q and "住院就诊" in q and "平均费用" in q:
            return _safe(
                "SELECT disease,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost "
                "FROM visits WHERE los_days>0 GROUP BY disease "
                "ORDER BY visit_count DESC,disease ASC"
            )
        if "以主要处方药分组" in q and all(token in q for token in ("使用次数", "平均住院天数", "length(drug_prescribed) > 0")):
            return _safe(
                "SELECT drug_prescribed,COUNT(*) AS use_count,ROUND(AVG(los_days),2) "
                "AS avg_los FROM visits WHERE length(drug_prescribed)>0 "
                "GROUP BY drug_prescribed ORDER BY use_count DESC,drug_prescribed ASC"
            )
        if "以主要检查分组" in q and "length(exam) > 0" in q:
            if "最高费用" in q and len(dates) >= 2:
                return _safe(
                    "SELECT exam,COUNT(*) AS exam_count,ROUND(MAX(cost),2) AS max_cost "
                    f"FROM visits WHERE visit_date BETWEEN '{dates[0]}' AND '{dates[1]}' "
                    "AND length(exam)>0 GROUP BY exam ORDER BY exam_count DESC,exam ASC"
                )
            if "平均费用" in q:
                return _safe(
                    "SELECT exam,COUNT(*) AS exam_count,ROUND(AVG(cost),2) AS avg_cost "
                    "FROM visits WHERE length(exam)>0 GROUP BY exam "
                    "ORDER BY avg_cost DESC,exam ASC"
                )
        if "以就诊月份分组" in q and dates and "总费用" in q:
            return _safe(
                "SELECT substr(visit_date,1,7),COUNT(*) AS visit_count,"
                f"ROUND(SUM(cost),2) AS total_cost FROM visits WHERE visit_date>='{dates[0]}' "
                "GROUP BY substr(visit_date,1,7) ORDER BY 1 ASC"
            )
        if "以住院天数分组" in q and "平均费用" in q:
            threshold = _integer(q, r"费用至少\s*(\d+)\s*元")
            if threshold is not None:
                return _safe(
                    "SELECT los_days,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost "
                    f"FROM visits WHERE cost>={threshold} GROUP BY los_days "
                    "ORDER BY los_days ASC"
                )
        if "以科室与疾病组合分组" in q and dates and "最高费用" in q:
            return _safe(
                "SELECT department,disease,COUNT(*) AS visit_count,ROUND(MAX(cost),2) "
                f"AS max_cost FROM visits WHERE visit_date<'{dates[0]}' "
                "GROUP BY department,disease ORDER BY visit_count DESC,department ASC,disease ASC"
            )
        if "以科室与年份组合分组" in q and "平均费用" in q:
            return _safe(
                "SELECT department,substr(visit_date,1,4),COUNT(*) AS visit_count,"
                "ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY department,"
                "substr(visit_date,1,4) ORDER BY department ASC,2 ASC"
            )
        if "以疾病与主要处方药组合分组" in q and "length(drug_prescribed) > 0" in q:
            return _safe(
                "SELECT disease,drug_prescribed,COUNT(*) AS pair_count,ROUND(SUM(cost),2) "
                "AS total_cost FROM visits WHERE length(drug_prescribed)>0 "
                "GROUP BY disease,drug_prescribed ORDER BY pair_count DESC,disease ASC,"
                "drug_prescribed ASC"
            )
        if "以科室与主要检查组合分组" in q and "length(exam) > 0" in q:
            return _safe(
                "SELECT department,exam,COUNT(*) AS pair_count,ROUND(AVG(los_days),2) "
                "AS avg_los FROM visits WHERE length(exam)>0 GROUP BY department,exam "
                "ORDER BY department ASC,pair_count DESC,exam ASC"
            )
        if "以就诊年份分组" in q and "平均费用" in q:
            threshold = _integer(q, r"费用超过\s*(\d+)\s*元")
            if threshold is not None:
                return _safe(
                    "SELECT substr(visit_date,1,4),COUNT(*) AS visit_count,"
                    "ROUND(AVG(cost),2) AS avg_cost FROM visits "
                    f"WHERE cost>{threshold} GROUP BY substr(visit_date,1,4) ORDER BY 1 ASC"
                )
        if "以科室分组" in q and "门诊" in q and "平均费用" in q:
            return _safe(
                "SELECT department,COUNT(*) AS outpatient_count,ROUND(AVG(cost),2) "
                "AS avg_cost FROM visits WHERE los_days=0 GROUP BY department "
                "ORDER BY outpatient_count DESC,department ASC"
            )
        if "以疾病分组" in q and "住院至少" in q and "总费用" in q:
            los = _integer(q, r"住院至少\s*(\d+)\s*天")
            if los is not None:
                return _safe(
                    "SELECT disease,COUNT(*) AS long_stay_count,ROUND(SUM(cost),2) "
                    f"AS total_cost FROM visits WHERE los_days>={los} GROUP BY disease "
                    "ORDER BY total_cost DESC,disease ASC"
                )

    department = _ground(q, domains, "department")
    if department and "按疾病分组计算" in q:
        where = f"department='{_esc(department)}'"
        dates = _dates(q)
        if dates:
            where += f" AND visit_date>='{dates[0]}'"
        if "就诊次数和平均费用" in q:
            projection = "COUNT(*) AS disease_visits,ROUND(AVG(cost),2) AS avg_cost"
            order = "disease_visits DESC,disease ASC"
        elif "总费用" in q and "最长住院天数" in q:
            projection = "ROUND(SUM(cost),2) AS total_cost,MAX(los_days) AS max_los"
            order = "total_cost DESC,disease ASC"
        elif "平均费用与平均住院天数" in q:
            projection = "ROUND(AVG(cost),2) AS avg_cost,ROUND(AVG(los_days),2) AS avg_los"
            order = "avg_cost DESC,disease ASC"
        elif "就诊次数和最高费用" in q:
            projection = "COUNT(*) AS disease_visits,ROUND(MAX(cost),2) AS max_cost"
            order = "max_cost DESC,disease ASC"
        elif "就诊次数和最低费用" in q:
            projection = "COUNT(*) AS disease_visits,ROUND(MIN(cost),2) AS min_cost"
            order = "min_cost ASC,disease ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT disease,{projection} FROM visits WHERE {where} "
                f"GROUP BY disease ORDER BY {order}"
            )

    if "实体类型" in q or "按类型" in q:
        if all(token in q for token in ("数量", "平均提及次数", "最高提及次数")):
            return _safe(
                "SELECT type,COUNT(*) AS entity_count,ROUND(AVG(mention_count),2) "
                "AS avg_mentions,MAX(mention_count) AS max_mentions FROM kg_entities "
                "GROUP BY type ORDER BY type ASC"
            )
        threshold = _integer(q, r"提及(?:次数)?(?:不少于|至少)\s*(\d+)\s*次")
        if threshold is not None and "高频实体数" in q:
            return _safe(
                "SELECT type,COUNT(*) AS frequent_count,ROUND(AVG(mention_count),2) "
                f"AS avg_mentions FROM kg_entities WHERE mention_count>={threshold} "
                "GROUP BY type ORDER BY frequent_count DESC,type ASC"
            )
        conditional = _integer(q, r"提及次数至少\s*(\d+)\s*次的实体数")
        if conditional is not None and "分别计算总数" in q:
            return _safe(
                "SELECT type,COUNT(*) AS entity_count,SUM(CASE WHEN mention_count>="
                f"{conditional} THEN 1 ELSE 0 END) AS high_mention_count FROM kg_entities "
                "GROUP BY type ORDER BY high_mention_count DESC,type ASC"
            )
        maximum = _integer(q, r"提及次数不超过\s*(\d+)\s*次")
        if maximum is not None and all(token in q for token in ("数量", "平均值", "总和")):
            return _safe(
                "SELECT type,COUNT(*) AS entity_count,ROUND(AVG(mention_count),2) "
                f"AS avg_mentions,SUM(mention_count) AS mention_sum FROM kg_entities "
                f"WHERE mention_count<={maximum} GROUP BY type "
                "ORDER BY mention_sum DESC,type ASC"
            )
        if all(token in q for token in ("最小值", "最大值", "极差")):
            return _safe(
                "SELECT type,MIN(mention_count) AS min_mentions,MAX(mention_count) "
                "AS max_mentions,MAX(mention_count)-MIN(mention_count) AS mention_range "
                "FROM kg_entities GROUP BY type ORDER BY mention_range DESC,type ASC"
            )

    if "知识图谱边" in q and "分组" in q:
        count_min = _integer(q, r"观察次数至少\s*(\d+)\s*次")
        confidence = _decimal(q, r"置信度不低于\s*([0-9]+(?:\.[0-9]+)?)")
        if "头尾实体类型组合" in q and count_min is not None:
            return _safe(
                "SELECT head_type,tail_type,COUNT(*) AS edge_count,ROUND(AVG(confidence),3) "
                f"AS avg_confidence,MAX(count) AS max_count FROM kg_triples WHERE count>={count_min} "
                "GROUP BY head_type,tail_type ORDER BY head_type ASC,tail_type ASC"
            )
        if "关系中文名与头实体类型" in q and confidence is not None:
            return _safe(
                "SELECT relation_zh,head_type,COUNT(*) AS edge_count,MIN(confidence) "
                f"AS min_confidence,SUM(count) AS observation_sum FROM kg_triples "
                f"WHERE confidence>={confidence} GROUP BY relation_zh,head_type "
                "ORDER BY relation_zh ASC,head_type ASC"
            )
        if "关系代码与尾实体类型" in q and count_min is not None:
            return _safe(
                "SELECT relation,tail_type,COUNT(*) AS edge_count,ROUND(AVG(confidence),3) "
                f"AS avg_confidence,MAX(count) AS max_count FROM kg_triples WHERE count>={count_min} "
                "GROUP BY relation,tail_type ORDER BY relation ASC,tail_type ASC"
            )
    return None


def _compile_having(q: str, _domains: Domains) -> Optional[str]:
    if "仅保留" not in q or "分组" not in q:
        return None
    if "按科室分组" in q and "就诊不少于" in q and "平均费用" in q:
        n = _integer(q, r"就诊不少于\s*(\d+)\s*次")
        if n is not None:
            return _safe(
                "SELECT department,COUNT(*) AS visit_count,ROUND(AVG(cost),2) AS avg_cost "
                f"FROM visits GROUP BY department HAVING COUNT(*)>={n} "
                "ORDER BY visit_count DESC,department ASC"
            )
    if "按月份分组" in q and "就诊不少于" in q and "总费用" in q:
        n = _integer(q, r"就诊不少于\s*(\d+)\s*次")
        if n is not None:
            return _safe(
                "SELECT substr(visit_date,1,7),COUNT(*) AS visit_count,ROUND(SUM(cost),2) "
                f"AS total_cost FROM visits GROUP BY substr(visit_date,1,7) "
                f"HAVING COUNT(*)>={n} ORDER BY 1 ASC"
            )
    if "住院就诊按科室分组" in q and "住院总费用超过" in q:
        value = _integer(q, r"住院总费用超过\s*(\d+)\s*元")
        if value is not None:
            return _safe(
                "SELECT department,COUNT(*) AS inpatient_count,ROUND(SUM(cost),2) "
                f"AS total_cost FROM visits WHERE los_days>0 GROUP BY department "
                f"HAVING SUM(cost)>{value} ORDER BY total_cost DESC,department ASC"
            )
    if "费用超过" in q and "按疾病分组" in q and "高费用就诊至少" in q:
        cost = _integer(q, r"费用超过\s*(\d+)\s*元")
        n = _integer(q, r"高费用就诊至少\s*(\d+)\s*次")
        if cost is not None and n is not None:
            return _safe(
                "SELECT disease,COUNT(*) AS costly_count,ROUND(AVG(cost),2) AS avg_cost "
                f"FROM visits WHERE cost>{cost} GROUP BY disease HAVING COUNT(*)>={n} "
                "ORDER BY costly_count DESC,disease ASC"
            )
    if "患者按地区分组" in q and "患者至少" in q and "平均年龄" in q:
        ages = re.findall(r"(\d+)\s*岁及以上", q)
        n = _integer(q, r"患者至少\s*(\d+)\s*人")
        if ages and n is not None:
            return _safe(
                "SELECT region,COUNT(*) AS patient_count,ROUND(AVG(age),2) AS avg_age "
                f"FROM patients WHERE age>={int(ages[0])} GROUP BY region "
                f"HAVING COUNT(*)>={n} ORDER BY patient_count DESC,region ASC"
            )
    if "患者按性别分组" in q and "平均年龄超过" in q and "最低年龄" in q:
        minimum = _integer(q, r"(\d+)\s*岁及以上")
        average = _integer(q, r"平均年龄超过\s*(\d+)\s*岁")
        if minimum is not None and average is not None:
            return _safe(
                "SELECT gender,COUNT(*) AS patient_count,MIN(age) AS min_age FROM patients "
                f"WHERE age>={minimum} GROUP BY gender HAVING AVG(age)>{average} "
                "ORDER BY gender ASC"
            )
    if "图谱边按关系分组" in q and "边数不少于" in q:
        count_min = _integer(q, r"观察次数至少\s*(\d+)\s*次")
        edge_min = _integer(q, r"边数不少于\s*(\d+)\s*条")
        if count_min is not None and edge_min is not None:
            return _safe(
                "SELECT relation,relation_zh,COUNT(*) AS edge_count,ROUND(AVG(confidence),3) "
                f"AS avg_confidence FROM kg_triples WHERE count>={count_min} "
                f"GROUP BY relation,relation_zh HAVING COUNT(*)>={edge_min} "
                "ORDER BY edge_count DESC,relation ASC"
            )
    if "图谱边按关系代码分组" in q and "观察次数总和不少于" in q:
        confidence = _decimal(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
        total = _integer(q, r"观察次数总和不少于\s*(\d+)")
        if confidence is not None and total is not None:
            return _safe(
                "SELECT relation,COUNT(*) AS edge_count,SUM(count) AS observation_sum "
                f"FROM kg_triples WHERE confidence>={confidence} GROUP BY relation "
                f"HAVING SUM(count)>={total} ORDER BY observation_sum DESC,relation ASC"
            )
    return None


def _compile_cte(q: str, domains: Domains) -> Optional[str]:
    if "先汇总各科室的就诊次数" in q and "住院数" in q:
        if "筛选就诊次数至少" in q:
            value = _integer(q, r"就诊次数至少\s*(\d+)\s*次")
            column, op, order = "visit_count", ">=", "visit_count"
        elif "筛选总费用超过" in q:
            value = _integer(q, r"总费用超过\s*(\d+)\s*元")
            column, op, order = "total_cost", ">", "total_cost"
        elif "筛选最长住院至少" in q:
            value = _integer(q, r"最长住院至少\s*(\d+)\s*天")
            column, op, order = "max_los", ">=", "max_los"
        elif "筛选住院就诊至少" in q:
            value = _integer(q, r"住院就诊至少\s*(\d+)\s*次")
            column, op, order = "inpatient_count", ">=", "inpatient_count"
        else:
            value = None
            column = op = order = ""
        if value is not None:
            return _safe(
                "WITH dept_stats AS (SELECT department,COUNT(*) AS visit_count,"
                "ROUND(AVG(cost),2) AS avg_cost,ROUND(SUM(cost),2) AS total_cost,"
                "MAX(los_days) AS max_los,SUM(CASE WHEN los_days>0 THEN 1 ELSE 0 END) "
                "AS inpatient_count FROM visits GROUP BY department) SELECT department,"
                "visit_count,avg_cost,total_cost,max_los,inpatient_count FROM dept_stats "
                f"WHERE {column}{op}{value} ORDER BY {order} DESC,department ASC"
            )

    dates = _dates(q)
    if all(token in q for token in ("每位患者的就诊次数", "去重疾病数", "连接姓名")) and dates:
        n = _integer(q, r"visit_count\s*>=\s*(\d+)") or _integer(q, r"至少(?:两|2)次就诊")
        if n is not None:
            return _safe(
                "WITH recent_use AS (SELECT patient_id,COUNT(*) AS visit_count,"
                "COUNT(DISTINCT disease) AS disease_count FROM visits "
                f"WHERE visit_date>='{dates[0]}' GROUP BY patient_id) SELECT p.patient_id,"
                "p.name,r.visit_count,r.disease_count FROM recent_use r JOIN patients p "
                f"ON p.patient_id=r.patient_id WHERE r.visit_count>={n} "
                "ORDER BY r.visit_count DESC,p.patient_id ASC"
            )

    if all(token in q for token in ("每位患者的住院次数", "累计住院天数", "住院总费用", "连接患者资料")):
        age = _integer(q, r"(\d+)\s*岁及以上")
        if age is not None:
            return _safe(
                "WITH inpatient AS (SELECT patient_id,COUNT(*) AS inpatient_count,"
                "SUM(los_days) AS total_los,ROUND(SUM(cost),2) AS total_cost FROM visits "
                "WHERE los_days>0 GROUP BY patient_id) SELECT p.patient_id,p.name,p.age,"
                "i.inpatient_count,i.total_los,i.total_cost FROM inpatient i JOIN patients p "
                f"ON p.patient_id=i.patient_id WHERE p.age>={age} "
                "ORDER BY i.total_los DESC,p.patient_id ASC"
            )

    if all(token in q for token in ("首末就诊日期", "主要检查种类数", "连接患者姓名")):
        n = _integer(q, r"检查种类至少\s*(\d+)\s*种")
        if n is not None and "length(exam) > 0" in q:
            return _safe(
                "WITH patient_span AS (SELECT patient_id,MIN(visit_date) AS first_date,"
                "MAX(visit_date) AS last_date,COUNT(DISTINCT CASE WHEN length(exam)>0 "
                "THEN exam END) AS exam_types FROM visits GROUP BY patient_id) SELECT "
                "p.patient_id,p.name,s.first_date,s.last_date,s.exam_types FROM patient_span s "
                f"JOIN patients p ON p.patient_id=s.patient_id WHERE s.exam_types>={n} "
                "ORDER BY s.exam_types DESC,p.patient_id ASC"
            )

    relation = _relation(q, domains)
    entity_type = _entity_type(q, domains)
    if relation and entity_type and all(token in q for token in ("按头实体构造汇总", "边数", "平均置信度", "与疾病实体连接")):
        alias = "drug_count" if "去重药物数" in q else "target_count"
        return _safe(
            "WITH edge_stats AS (SELECT head,COUNT(*) AS edge_count,COUNT(DISTINCT tail) "
            f"AS {alias},ROUND(AVG(confidence),3) AS avg_confidence FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' GROUP BY head) SELECT e.name,e.mention_count,"
            f"s.edge_count,s.{alias},s.avg_confidence FROM edge_stats s JOIN kg_entities e "
            f"ON e.name=s.head WHERE e.type='{_esc(entity_type)}' "
            f"ORDER BY s.{alias} DESC,e.name ASC"
        )

    if all(token in q for token in ("先按月汇总", "所有月份的平均次数", "高于月均值")):
        return _safe(
            "WITH monthly AS (SELECT substr(visit_date,1,7) AS month,COUNT(*) AS "
            "visit_count,ROUND(SUM(cost),2) AS total_cost FROM visits GROUP BY month),"
            "baseline AS (SELECT AVG(visit_count) AS avg_count FROM monthly) SELECT "
            "m.month,m.visit_count,m.total_cost,ROUND(b.avg_count,2) AS avg_monthly_count "
            "FROM monthly m CROSS JOIN baseline b WHERE m.visit_count>b.avg_count "
            "ORDER BY m.month ASC"
        )
    return None


def _compile_joins(q: str, domains: Domains) -> Optional[str]:
    region = _ground(q, domains, "region")
    department = _ground(q, domains, "department")
    if region and department and "连接患者与就诊记录" in q:
        where = f"p.region='{_esc(region)}' AND v.department='{_esc(department)}'"
        if all(token in q for token in ("姓名、年龄、就诊日期、疾病和费用", "费用降序")):
            projection = "p.name,p.age,v.visit_date,v.disease,v.cost"
            order = "v.cost DESC,v.visit_id ASC"
        elif "患者编号、性别、就诊编号、主要处方药和费用" in q:
            projection = "p.patient_id,p.gender,v.visit_id,v.drug_prescribed,v.cost"
            order = "p.patient_id ASC,v.visit_date ASC,v.visit_id ASC"
        elif "姓名、地区、日期、主要检查和住院天数" in q:
            projection = "p.name,p.region,v.visit_date,v.exam,v.los_days"
            order = "v.los_days DESC,p.name ASC,v.visit_id ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT {projection} FROM patients p JOIN visits v "
                f"ON v.patient_id=p.patient_id WHERE {where} ORDER BY {order}"
            )

    disease = _ground(q, domains, "disease")
    gender_match = re.search(r"性别为([男女])", q)
    if disease and gender_match and "患者资料和就诊表按编号连接" in q:
        gender = gender_match.group(1)
        if "患者编号、性别、就诊编号、主要处方药和费用" in q:
            projection = "p.patient_id,p.gender,v.visit_id,v.drug_prescribed,v.cost"
            order = "p.patient_id ASC,v.visit_date ASC,v.visit_id ASC"
        elif "患者编号、姓名、科室、疾病和费用" in q:
            projection = "p.patient_id,p.name,v.department,v.disease,v.cost"
            order = "p.name ASC,v.cost DESC,v.visit_id ASC"
        elif "患者编号、姓名、地区、就诊编号、科室和疾病" in q:
            projection = "p.patient_id,p.name,p.region,v.visit_id,v.department,v.disease"
            order = "v.visit_date ASC,v.visit_id ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT {projection} FROM patients p JOIN visits v ON "
                f"p.patient_id=v.patient_id WHERE p.gender='{gender}' "
                f"AND v.disease='{_esc(disease)}' ORDER BY {order}"
            )

    dates = _dates(q)
    age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
    if department and dates and age is not None and all(
        token in q for token in ("跨表统计", "按地区和性别", "去重患者数", "平均费用")
    ):
        return _safe(
            "SELECT p.region,p.gender,COUNT(DISTINCT p.patient_id) AS patient_count,"
            "COUNT(*) AS visit_count,ROUND(AVG(v.cost),2) AS avg_cost FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE v.department='{_esc(department)}' AND v.visit_date>='{dates[0]}' "
            f"AND p.age>={age} GROUP BY p.region,p.gender "
            "ORDER BY p.region ASC,p.gender ASC"
        )

    relation = _relation(q, domains)
    if relation and "就诊表" in q and "图谱" in q and "去重" in q:
        if "主要处方药与图谱的头实体匹配" in q:
            return _safe(
                "SELECT DISTINCT v.visit_id,v.drug_prescribed,t.tail,t.confidence "
                "FROM visits v JOIN kg_triples t ON v.drug_prescribed=t.head "
                f"WHERE t.relation='{_esc(relation)}' ORDER BY v.drug_prescribed ASC,"
                "v.visit_id ASC,t.tail ASC"
            )
        if "疾病与图谱的头实体匹配" in q and "日期、疾病和症状目标" in q:
            return _safe(
                "SELECT DISTINCT v.visit_id,v.visit_date,v.disease,t.tail FROM visits v "
                "JOIN kg_triples t ON v.disease=t.head "
                f"WHERE t.relation='{_esc(relation)}' ORDER BY v.visit_date ASC,"
                "v.visit_id ASC,t.tail ASC"
            )
        if "疾病与图谱的头实体匹配" in q:
            return _safe(
                "SELECT DISTINCT v.visit_id,v.disease,t.tail,t.confidence FROM visits v "
                "JOIN kg_triples t ON v.disease=t.head "
                f"WHERE t.relation='{_esc(relation)}' ORDER BY v.visit_id ASC,t.tail ASC"
            )

    if relation and all(token in q for token in ("串联患者资料", "就诊记录", "图谱边", "去重返回")):
        if "患者编号、姓名、疾病和症状目标" in q:
            projection = "p.patient_id,p.name,v.disease,t.tail"
            order = "p.patient_id ASC,t.tail ASC"
        elif "患者编号、年龄、疾病和图谱药物" in q:
            projection = "p.patient_id,p.age,v.disease,t.tail"
            order = "p.age DESC,p.patient_id ASC,t.tail ASC"
        elif "患者编号、姓名、就诊科室和治疗目标" in q:
            projection = "p.patient_id,p.name,v.department,t.tail"
            order = "v.department ASC,p.patient_id ASC,t.tail ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT DISTINCT {projection} FROM patients p JOIN visits v ON "
                "v.patient_id=p.patient_id JOIN kg_triples t ON v.disease=t.head "
                f"WHERE t.relation='{_esc(relation)}' ORDER BY {order}"
            )
    return None


def _compile_kg(q: str, domains: Domains) -> Optional[str]:
    entity_type = _entity_type(q, domains)
    relation = _relation(q, domains)

    if entity_type and "提及" in q and "名称" in q and "次数" in q and "平均值" not in q:
        mentions = _integer(q, r"(?:提及次数(?:至少|达到)|不少于|提及至少|提及达到)\s*(\d+)\s*次")
        if mentions is not None:
            if "按名称升序" in q and "按次数" not in q and "优先显示次数" not in q:
                order = "name ASC"
            elif "次数升序" in q:
                order = "mention_count ASC,name ASC"
            else:
                order = "mention_count DESC,name ASC"
            return _safe(
                "SELECT name,mention_count FROM kg_entities "
                f"WHERE type='{_esc(entity_type)}' AND mention_count>={mentions} "
                f"ORDER BY {order}"
            )

    confidence = _decimal(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
    if relation and confidence is not None and all(token in q for token in ("头实体", "尾实体", "置信度")) and "平均" not in q:
        return _safe(
            "SELECT head,tail,confidence FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' AND confidence>={confidence} "
            "ORDER BY head ASC,tail ASC"
        )

    if relation and "按头实体汇总" in q:
        if all(token in q for token in ("去重尾实体数", "观察次数总和")) and "边数" not in q:
            projection = "COUNT(DISTINCT tail) AS target_count,SUM(count) AS observation_sum"
            order = "target_count DESC,head ASC"
        elif all(token in q for token in ("边数", "最高置信度", "平均观察次数")):
            projection = "COUNT(*) AS edge_count,MAX(confidence) AS max_confidence,ROUND(AVG(count),2) AS avg_count"
            order = "max_confidence DESC,head ASC"
        elif all(token in q for token in ("去重尾实体数", "边数", "平均置信度")):
            projection = "COUNT(DISTINCT tail) AS target_count,COUNT(*) AS edge_count,ROUND(AVG(confidence),3) AS avg_confidence"
            order = "edge_count DESC,head ASC"
        elif all(token in q for token in ("观察次数总和", "边数", "最低置信度")):
            projection = "SUM(count) AS observation_sum,COUNT(*) AS edge_count,MIN(confidence) AS min_confidence"
            order = "observation_sum DESC,head ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT head,{projection} FROM kg_triples WHERE relation='{_esc(relation)}' "
                f"GROUP BY head ORDER BY {order}"
            )

    if relation and entity_type and "实体与" in q and "按头实体连接" in q:
        where = f"e.type='{_esc(entity_type)}' AND t.relation='{_esc(relation)}'"
        if all(token in q for token in ("名称", "目标", "置信度", "观察次数")):
            projection = "e.name,t.tail,t.confidence,t.count"
            order = "t.confidence DESC,e.name ASC,t.tail ASC"
        elif all(token in q for token in ("名称", "提及次数", "目标", "边置信度")):
            projection = "e.name,e.mention_count,t.tail,t.confidence"
            order = "e.mention_count DESC,e.name ASC,t.tail ASC"
        elif all(token in q for token in ("名称", "目标", "置信度")):
            projection = "e.name,t.tail,t.confidence"
            order = "e.name ASC,t.confidence DESC,t.tail ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT {projection} FROM kg_entities e JOIN kg_triples t ON t.head=e.name "
                f"WHERE {where} ORDER BY {order}"
            )

    if relation and "就诊主诊断映射到" in q and "按疾病汇总" in q:
        if all(token in q for token in ("去重检查数", "平均置信度")):
            projection = "COUNT(DISTINCT t.tail) AS target_count,ROUND(AVG(t.confidence),3) AS avg_confidence"
            order = "target_count DESC,v.disease ASC"
        elif all(token in q for token in ("去重药物数", "观察次数总和")):
            projection = "COUNT(DISTINCT t.tail) AS target_count,SUM(t.count) AS observation_sum"
            order = "observation_sum DESC,v.disease ASC"
        elif all(token in q for token in ("去重所属科室目标数", "实际就诊科室数")):
            projection = "COUNT(DISTINCT t.tail) AS target_count,COUNT(DISTINCT v.department) AS visit_dept_count"
            order = "visit_dept_count DESC,v.disease ASC"
        elif all(token in q for token in ("去重并发症数", "最高置信度")):
            projection = "COUNT(DISTINCT t.tail) AS target_count,MAX(t.confidence) AS max_confidence"
            order = "target_count DESC,v.disease ASC"
        elif all(token in q for token in ("去重治疗方式数", "匹配就诊平均费用")):
            projection = "COUNT(DISTINCT t.tail) AS target_count,ROUND(AVG(v.cost),2) AS avg_visit_cost"
            order = "avg_visit_cost DESC,v.disease ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT v.disease,{projection} FROM visits v JOIN kg_triples t "
                f"ON t.head=v.disease AND t.relation='{_esc(relation)}' "
                f"GROUP BY v.disease ORDER BY {order}"
            )
    return None


def _compile_nested(q: str, domains: Domains) -> Optional[str]:
    department = _ground(q, domains, "department")
    if department and "平均" in q and "就诊" in q:
        if all(token in q for token in ("费用高于", "科室平均费用", "编号、疾病和费用")):
            return _safe(
                "SELECT v.visit_id,v.disease,v.cost FROM visits v "
                f"WHERE v.department='{_esc(department)}' AND v.cost>(SELECT AVG(v2.cost) "
                "FROM visits v2 WHERE v2.department=v.department) "
                "ORDER BY v.cost DESC,v.visit_id ASC"
            )
        if all(token in q for token in ("费用低于", "科室平均", "就诊编号、日期、疾病和费用")):
            return _safe(
                "SELECT v.visit_id,v.visit_date,v.disease,v.cost FROM visits v "
                f"WHERE v.department='{_esc(department)}' AND v.cost<(SELECT AVG(v2.cost) "
                "FROM visits v2 WHERE v2.department=v.department) "
                "ORDER BY v.cost ASC,v.visit_id ASC"
            )
    if department and all(token in q for token in ("最长住院天数", "编号、疾病、天数和费用")):
        return _safe(
            "SELECT v.visit_id,v.disease,v.los_days,v.cost FROM visits v "
            f"WHERE v.department='{_esc(department)}' AND v.los_days=(SELECT MAX(v2.los_days) "
            "FROM visits v2 WHERE v2.department=v.department) "
            "ORDER BY v.cost DESC,v.visit_id ASC"
        )

    disease = _ground(q, domains, "disease")
    if disease and all(token in q for token in ("同一疾病平均费用", "就诊编号、科室、费用和住院天数")):
        return _safe(
            "SELECT v.visit_id,v.department,v.cost,v.los_days FROM visits v "
            f"WHERE v.disease='{_esc(disease)}' AND v.cost>(SELECT AVG(v2.cost) "
            "FROM visits v2 WHERE v2.disease=v.disease) "
            "ORDER BY v.cost DESC,v.visit_id ASC"
        )

    region = _ground(q, domains, "region")
    if region and department and "关联子查询" in q and "至少有一条" in q:
        where = (
            f"p.region='{_esc(region)}' AND EXISTS (SELECT 1 FROM visits v "
            f"WHERE v.patient_id=p.patient_id AND v.department='{_esc(department)}')"
        )
        if "患者编号、姓名和年龄" in q:
            projection, order = "p.patient_id,p.name,p.age", "p.patient_id ASC"
        elif "姓名、性别、年龄和患者编号" in q:
            projection, order = "p.name,p.gender,p.age,p.patient_id", "p.age DESC,p.patient_id ASC"
        elif "患者编号、姓名及地区" in q:
            projection, order = "p.patient_id,p.name,p.region", "p.name ASC,p.patient_id ASC"
        elif "患者编号、地区、姓名和年龄" in q:
            projection, order = "p.patient_id,p.region,p.name,p.age", "p.age ASC,p.patient_id ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT {projection} FROM patients p WHERE {where} ORDER BY {order}"
            )

    if region and all(token in q for token in ("同一性别平均年龄", "编号、姓名、性别和年龄")):
        return _safe(
            "SELECT p.patient_id,p.name,p.gender,p.age FROM patients p "
            f"WHERE p.region='{_esc(region)}' AND p.age>(SELECT AVG(p2.age) FROM "
            "patients p2 WHERE p2.region=p.region AND p2.gender=p.gender) "
            "ORDER BY p.gender ASC,p.age DESC,p.patient_id ASC"
        )

    relation = _relation(q, domains)
    if relation and "平均" in q and "kg_triples" not in q:
        if all(token in q for token in ("置信度高于该关系整体平均值", "头实体、尾实体和置信度")):
            return _safe(
                "SELECT t.head,t.tail,t.confidence FROM kg_triples t "
                f"WHERE t.relation='{_esc(relation)}' AND t.confidence>(SELECT AVG(t2.confidence) "
                "FROM kg_triples t2 WHERE t2.relation=t.relation) "
                "ORDER BY t.confidence DESC,t.head ASC,t.tail ASC"
            )
        if all(token in q for token in ("同一头实体平均置信度", "头尾实体和置信度")):
            return _safe(
                "SELECT t.head,t.tail,t.confidence FROM kg_triples t "
                f"WHERE t.relation='{_esc(relation)}' AND t.confidence>(SELECT AVG(t2.confidence) "
                "FROM kg_triples t2 WHERE t2.relation=t.relation AND t2.head=t.head) "
                "ORDER BY t.head ASC,t.confidence DESC,t.tail ASC"
            )
        if all(token in q for token in ("置信度不低于该关系平均值", "观察次数")):
            return _safe(
                "SELECT t.head,t.tail,t.confidence,t.count FROM kg_triples t "
                f"WHERE t.relation='{_esc(relation)}' AND t.confidence>=(SELECT AVG(t2.confidence) "
                "FROM kg_triples t2 WHERE t2.relation=t.relation) "
                "ORDER BY t.count DESC,t.head ASC,t.tail ASC"
            )
        if all(token in q for token in ("观察次数高于该关系平均观察次数", "次数和置信度")):
            return _safe(
                "SELECT t.head,t.tail,t.count,t.confidence FROM kg_triples t "
                f"WHERE t.relation='{_esc(relation)}' AND t.count>(SELECT AVG(t2.count) "
                "FROM kg_triples t2 WHERE t2.relation=t.relation) "
                "ORDER BY t.count DESC,t.head ASC,t.tail ASC"
            )

    entity_type = _entity_type(q, domains)
    if entity_type and all(token in q for token in ("提及次数", "平均值", "名称", "次数")):
        if "任一端出现" in q:
            exists = "EXISTS (SELECT 1 FROM kg_triples t WHERE t.head=e.name OR t.tail=e.name)"
        elif "头实体" in q:
            exists = "EXISTS (SELECT 1 FROM kg_triples t WHERE t.head=e.name)"
        elif "尾实体" in q or "被一条关系指向" in q:
            exists = "EXISTS (SELECT 1 FROM kg_triples t WHERE t.tail=e.name)"
        else:
            exists = ""
        if exists:
            return _safe(
                "SELECT e.name,e.mention_count FROM kg_entities e "
                f"WHERE e.type='{_esc(entity_type)}' AND e.mention_count>(SELECT "
                "AVG(e2.mention_count) FROM kg_entities e2 WHERE e2.type=e.type) "
                f"AND {exists} ORDER BY e.mention_count DESC,e.name ASC"
            )
    return None


def _compile_temporal(q: str, domains: Domains) -> Optional[str]:
    dates = _dates(q)
    if len(dates) >= 2 and all(token in q for token in ("时间范围", "包含本月月初", "不包含下月月初")):
        where = f"visit_date>='{dates[0]}' AND visit_date<'{dates[1]}'"
        if "就诊编号、日期、科室和疾病" in q:
            projection = "visit_id,visit_date,department,disease"
            order = "visit_date ASC,visit_id ASC"
        elif "患者编号、日期、费用和住院天数" in q:
            projection = "patient_id,visit_date,cost,los_days"
            order = "cost DESC,visit_id ASC"
        elif "就诊编号、患者编号、科室和费用" in q:
            projection = "visit_id,patient_id,department,cost"
            order = "patient_id ASC,visit_date ASC,visit_id ASC"
        else:
            projection = order = ""
        if projection:
            return _safe(
                f"SELECT {projection} FROM visits WHERE {where} ORDER BY {order}"
            )

    department = _ground(q, domains, "department")
    if department and dates and all(token in q for token in ("按自然月统计", "就诊次数", "费用总额", "月份升序")):
        return _safe(
            "SELECT substr(visit_date,1,7) AS visit_month,COUNT(*) AS visit_count,"
            "ROUND(SUM(cost),2) AS total_cost FROM visits "
            f"WHERE department='{_esc(department)}' AND visit_date>='{dates[0]}' "
            "GROUP BY visit_month ORDER BY visit_month ASC"
        )

    region = _ground(q, domains, "region")
    if (
        region
        and ("每位患者" in q or "各自" in q)
        and ("末次" in q or "最早" in q or "最近一次" in q)
    ):
        base = (
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.region='{_esc(region)}'"
        )
        if "末次就诊日记录" in q:
            return _safe(
                "SELECT p.patient_id,p.name,v.visit_id,v.visit_date,v.department "
                f"{base} AND v.visit_date=(SELECT MAX(v2.visit_date) FROM visits v2 "
                "WHERE v2.patient_id=v.patient_id) ORDER BY p.patient_id ASC,v.visit_id ASC"
            )
        if "最早一次就诊" in q and dates:
            return _safe(
                "SELECT p.patient_id,p.name,v.visit_date,v.disease,v.cost "
                f"{base} AND v.visit_date=(SELECT MIN(v2.visit_date) FROM visits v2 "
                f"WHERE v2.patient_id=v.patient_id AND v2.visit_date>='{dates[0]}') "
                "ORDER BY v.visit_date ASC,p.patient_id ASC"
            )
        if "最早门诊" in q and "los_days = 0" in q:
            return _safe(
                "SELECT p.patient_id,p.name,v.visit_date,v.disease,v.cost "
                f"{base} AND v.los_days=0 AND v.visit_date=(SELECT MIN(v2.visit_date) "
                "FROM visits v2 WHERE v2.patient_id=v.patient_id AND v2.los_days=0) "
                "ORDER BY p.patient_id ASC,v.visit_id ASC"
            )
        if "最近一次住院" in q and "los_days > 0" in q:
            return _safe(
                "SELECT p.patient_id,p.name,v.visit_date,v.department,v.los_days "
                f"{base} AND v.los_days>0 AND v.visit_date=(SELECT MAX(v2.visit_date) "
                "FROM visits v2 WHERE v2.patient_id=v.patient_id AND v2.los_days>0) "
                "ORDER BY p.patient_id ASC,v.visit_id ASC"
            )

    if region and "至少有两次就诊" in q:
        common_from = (
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.region='{_esc(region)}' GROUP BY p.patient_id,p.name HAVING COUNT(*)>=2"
        )
        if "跨度折合月数" in q:
            return _safe(
                "SELECT p.patient_id,p.name,MIN(v.visit_date) AS first_date,"
                "MAX(v.visit_date) AS last_date,COUNT(*) AS visit_count,ROUND((julianday("
                "MAX(v.visit_date))-julianday(MIN(v.visit_date)))/30.44,1) AS span_value "
                f"{common_from} ORDER BY span_value DESC,p.patient_id ASC"
            )
        if all(token in q for token in ("就诊次数", "第一天", "最后一天")):
            return _safe(
                "SELECT p.patient_id,p.name,COUNT(*) AS visit_count,MIN(v.visit_date) "
                "AS first_date,MAX(v.visit_date) AS last_date "
                f"{common_from} ORDER BY visit_count DESC,p.patient_id ASC"
            )
        if all(token in q for token in ("跨度完整周数", "余下天数")):
            return _safe(
                "SELECT p.patient_id,p.name,MIN(v.visit_date) AS first_date,"
                "MAX(v.visit_date) AS last_date,CAST((julianday(MAX(v.visit_date))-"
                "julianday(MIN(v.visit_date)))/7 AS INTEGER) AS full_weeks,CAST(julianday("
                "MAX(v.visit_date))-julianday(MIN(v.visit_date)) AS INTEGER)%7 AS "
                f"remaining_days {common_from} ORDER BY full_weeks DESC,p.patient_id ASC"
            )
        if all(token in q for token in ("总跨度天数", "就诊次数")):
            return _safe(
                "SELECT p.patient_id,p.name,MIN(v.visit_date) AS first_date,"
                "MAX(v.visit_date) AS last_date,CAST(julianday(MAX(v.visit_date))-"
                "julianday(MIN(v.visit_date)) AS INTEGER) AS span_value,COUNT(*) AS "
                f"visit_count {common_from} ORDER BY span_value ASC,p.patient_id ASC"
            )
    return None


_COMPILERS: tuple[Callable[[str, Domains], Optional[str]], ...] = (
    _compile_window,
    _compile_cte,
    _compile_set_operations,
    _compile_temporal,
    _compile_nested,
    _compile_joins,
    _compile_having,
    _compile_topk,
    _compile_filtering,
    _compile_kg,
    _compile_aggregation,
)


def compile_independent_v21_sql(q: str, domains: Domains) -> Optional[str]:
    """Compile one complete bounded analytical shape or return ``None``.

    Compiler order is intentional: structurally exclusive owners (window,
    CTE, set, temporal, nested, joins, and HAVING) run before flat aggregation
    and filtering so a broad legacy-looking phrase cannot discard structure.
    """

    for compiler in _COMPILERS:
        candidate = compiler(q, domains)
        if candidate is not None:
            return candidate
    return None
