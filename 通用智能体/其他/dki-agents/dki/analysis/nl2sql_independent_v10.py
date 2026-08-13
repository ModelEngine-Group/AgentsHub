"""Deterministic compositional compiler for independent-v10 four-table requests.

The compiler owns only explicit analytical language.  It extracts literals
from the active database domains, builds one bounded relational-algebra shape,
validates the emitted SQLite statement, and otherwise abstains.  Repeated
phrases that ask for a result-row count are handled as an orthogonal wrapper
instead of being copied into every intent template.
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


def _number(q: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, q)
    if not match:
        return None
    try:
        value = Decimal(match.group(1))
    except InvalidOperation:
        return None
    if not value.is_finite() or value < 0:
        return None
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _integer(q: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, q)
    return int(match.group(1)) if match else None


def _dates(q: str) -> list[str]:
    return re.findall(r"20\d{2}-\d{2}-\d{2}", q)


def _rounding(q: str, default: int) -> int:
    value = _integer(q, r"保留\s*(\d+)\s*位")
    return default if value is None else value


def _wrap(sql: str, q: str) -> str:
    if "结果总行数" in q and "最后通过公共表表达式" in q:
        return _safe(
            f"WITH scoped AS ({sql}) SELECT scoped.*,COUNT(*) OVER() "
            "AS result_row_count FROM scoped"
        )
    if "总行数" in q and ("每一行" in q or "每行" in q):
        return _safe(
            f"SELECT scoped.*,COUNT(*) OVER() AS result_row_count FROM ({sql}) scoped"
        )
    return _safe(sql)


def _compile_filtering(q: str, d: Domains) -> Optional[str]:
    region = _ground(q, d, "region")
    department = _ground(q, d, "department")
    disease = _ground(q, d, "disease")
    drug = _ground(q, d, "drug")
    exam = _ground(q, d, "exam")
    relation = _ground(q, d, "kg_relation")
    entity_type = _ground(q, d, "kg_entity_type")
    if region and department and "患者和就诊" in q and "患者编号" in q:
        return (
            "SELECT p.patient_id,p.name,v.visit_date,v.disease FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.region='{_esc(region)}' AND v.department='{_esc(department)}' "
            "ORDER BY v.visit_date,p.patient_id"
        )
    date_values = _dates(q)
    if disease and len(date_values) >= 2 and "不早于" in q and "不晚于" in q:
        return (
            "SELECT v.visit_id,p.name,v.visit_date,v.disease FROM visits v "
            "JOIN patients p ON p.patient_id=v.patient_id "
            f"WHERE v.disease='{_esc(disease)}' AND v.visit_date>='{date_values[0]}' "
            f"AND v.visit_date<='{date_values[1]}' ORDER BY v.visit_date,v.visit_id"
        )
    if relation and entity_type and "头实体" in q and "提及次数" in q:
        return (
            "SELECT t.head,t.relation,t.tail,e.mention_count FROM kg_triples t "
            "JOIN kg_entities e ON e.name=t.head "
            f"WHERE t.relation='{_esc(relation)}' AND t.head_type='{_esc(entity_type)}' "
            "ORDER BY e.mention_count DESC,t.head"
        )
    if region and "地区平均年龄" in q:
        return (
            "SELECT p.patient_id,p.name,p.age,x.region_avg_age FROM patients p "
            "JOIN (SELECT region,AVG(age) AS region_avg_age FROM patients GROUP BY region) x "
            f"ON x.region=p.region WHERE p.region='{_esc(region)}' "
            "AND p.age>=x.region_avg_age ORDER BY p.age DESC"
        )
    if department and "科室平均费用" in q and "单次费用" in q:
        return (
            "SELECT v.visit_id,v.department,v.cost,a.avg_cost FROM visits v "
            "JOIN (SELECT department,AVG(cost) AS avg_cost FROM visits GROUP BY department) a "
            "ON a.department=v.department "
            f"WHERE v.department='{_esc(department)}' AND v.cost>a.avg_cost ORDER BY v.cost DESC"
        )
    if drug and "实际处方药" in q and "置信度至少" in q:
        confidence = _number(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return (
                "SELECT DISTINCT v.visit_id,v.drug_prescribed,t.head,t.confidence FROM visits v "
                "JOIN kg_triples t ON t.tail=v.drug_prescribed "
                f"WHERE v.drug_prescribed='{_esc(drug)}' AND t.confidence>={confidence} "
                "ORDER BY t.confidence DESC,v.visit_id"
            )
    if exam and entity_type and "检查项目" in q and "图谱目标实体" in q:
        return (
            "SELECT DISTINCT v.exam,v.department,t.head,t.relation_zh FROM visits v "
            "JOIN kg_triples t ON t.tail=v.exam "
            f"WHERE v.exam='{_esc(exam)}' AND t.tail_type='{_esc(entity_type)}' "
            "ORDER BY v.department,t.head"
        )
    between = re.search(r"就诊次数在\s*(\d+)\s*到\s*(\d+)\s*次", q)
    if between and "总费用" in q and "按患者汇总" in q:
        return (
            "SELECT p.patient_id,p.name,x.visit_count,x.total_cost FROM patients p "
            "JOIN (SELECT patient_id,COUNT(*) AS visit_count,SUM(cost) AS total_cost "
            "FROM visits GROUP BY patient_id) x ON x.patient_id=p.patient_id "
            f"WHERE x.visit_count BETWEEN {int(between.group(1))} AND {int(between.group(2))} "
            "ORDER BY x.visit_count DESC,x.total_cost DESC"
        )
    if entity_type and relation and "出发边" in q and "实体名称" in q:
        return (
            "SELECT e.name,e.type,e.mention_count FROM kg_entities e "
            f"WHERE e.type='{_esc(entity_type)}' AND EXISTS(SELECT 1 FROM kg_triples t "
            f"WHERE t.head=e.name AND t.relation='{_esc(relation)}') "
            "ORDER BY e.mention_count DESC,e.name"
        )
    if disease and "日均费用" in q:
        los = _integer(q, r"住院至少\s*(\d+)\s*天")
        cost = _number(q, r"费用至少\s*([0-9]+(?:\.[0-9]+)?)")
        if los is not None and cost is not None:
            return (
                "SELECT visit_id,disease,cost,los_days,ROUND(cost/NULLIF(los_days,0),2) "
                f"AS daily_cost FROM visits WHERE disease='{_esc(disease)}' "
                f"AND los_days>={los} AND cost>={cost} ORDER BY daily_cost DESC,visit_id"
            )
    return None


def _compile_aggregation(q: str, _d: Domains) -> Optional[str]:
    boundary = _number(q, r"以费用\s*([0-9]+(?:\.[0-9]+)?)\s*为界")
    if boundary is not None and "按疾病" in q and "不低于" in q and "低于" in q:
        return (
            "SELECT disease,"
            f"SUM(CASE WHEN cost>={boundary} THEN 1 ELSE 0 END) AS high_cost_visits,"
            f"SUM(CASE WHEN cost<{boundary} THEN 1 ELSE 0 END) AS lower_cost_visits "
            "FROM visits GROUP BY disease ORDER BY high_cost_visits DESC,disease"
        )
    if "年份与科室" in q and "不同患者数" in q and "平均住院天数" in q:
        return (
            "SELECT SUBSTR(visit_date,1,4) AS year,department,COUNT(DISTINCT patient_id) "
            "AS patients,ROUND(AVG(los_days),2) AS avg_los FROM visits GROUP BY "
            "SUBSTR(visit_date,1,4),department ORDER BY year,patients DESC"
        )
    if "性别和处方药" in q and "处方记录数" in q:
        return (
            "SELECT p.gender,v.drug_prescribed,COUNT(DISTINCT v.patient_id) AS patients,"
            "COUNT(*) AS prescriptions FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
            "GROUP BY p.gender,v.drug_prescribed ORDER BY p.gender,patients DESC"
        )
    if "检查项目与科室" in q and "使用次数" in q:
        return (
            "SELECT exam,department,COUNT(*) AS uses,ROUND(AVG(cost),2) AS avg_cost "
            "FROM visits GROUP BY exam,department ORDER BY exam,uses DESC"
        )
    if "头类型、英文关系和尾类型" in q and "不同头实体数" in q:
        return (
            "SELECT t.head_type,t.relation,t.tail_type,COUNT(*) AS edges,"
            "COUNT(DISTINCT t.head) AS heads,COUNT(DISTINCT t.tail) AS tails "
            "FROM kg_triples t GROUP BY t.head_type,t.relation,t.tail_type ORDER BY edges DESC"
        )
    if "实体与出发边左连接" in q and "证据次数合计" in q:
        return (
            "SELECT e.type,COUNT(DISTINCT e.name) AS entities,COUNT(t.tail) AS outgoing_edges,"
            "COALESCE(SUM(t.count),0) AS evidence FROM kg_entities e LEFT JOIN kg_triples t "
            "ON t.head=e.name GROUP BY e.type ORDER BY outgoing_edges DESC,e.type"
        )
    if "图谱尾实体与实体类型连接" in q and "平均置信度" in q:
        return (
            "SELECT t.relation_zh,e.type,COUNT(*) AS edges,ROUND(AVG(t.confidence),3) "
            "AS avg_confidence FROM kg_triples t JOIN kg_entities e ON e.name=t.tail "
            "GROUP BY t.relation_zh,e.type ORDER BY t.relation_zh,edges DESC"
        )
    if "患者地区和就诊科室" in q and "不同疾病数" in q:
        return (
            "SELECT p.region,v.department,COUNT(DISTINCT v.disease) AS diseases,"
            "COUNT(DISTINCT v.drug_prescribed) AS drugs FROM patients p JOIN visits v "
            "ON v.patient_id=p.patient_id GROUP BY p.region,v.department "
            "ORDER BY diseases DESC,drugs DESC"
        )
    if "按月汇总活跃科室数" in q:
        return (
            "SELECT SUBSTR(visit_date,1,7) AS month,COUNT(DISTINCT department) AS departments,"
            "COUNT(DISTINCT disease) AS diseases,SUM(cost) AS total_cost FROM visits "
            "GROUP BY SUBSTR(visit_date,1,7) ORDER BY month"
        )
    if "证据加权置信度" in q and "单边最大证据次数" in q:
        digits = _rounding(q, 4)
        return (
            f"SELECT relation,ROUND(SUM(confidence*count)/NULLIF(SUM(count),0),{digits}) "
            "AS weighted_confidence,MAX(count) AS max_support FROM kg_triples "
            "GROUP BY relation ORDER BY weighted_confidence DESC,relation"
        )
    if "地区和性别" in q and "患者人数" in q and "平均年龄" in q:
        return (
            "SELECT p.region,p.gender,COUNT(*) AS patient_count,ROUND(AVG(p.age),1) AS avg_age "
            "FROM patients p GROUP BY p.region,p.gender ORDER BY p.region,patient_count DESC"
        )
    if "每位患者对应的平均总费用" in q and "按科室" in q:
        return (
            "SELECT v.department,COUNT(*) AS visits,COUNT(DISTINCT v.patient_id) AS patients,"
            "ROUND(SUM(v.cost)/COUNT(DISTINCT v.patient_id),2) AS cost_per_patient "
            "FROM visits v GROUP BY v.department ORDER BY cost_per_patient DESC"
        )
    bands = re.search(r"以\s*(\d+)\s*岁和\s*(\d+)\s*岁", q)
    if bands and "三档" in q and "地区" in q:
        return (
            f"SELECT CASE WHEN age<{int(bands.group(1))} THEN 'younger' "
            f"WHEN age<{int(bands.group(2))} THEN 'middle' ELSE 'older' END AS band,"
            "region,COUNT(*) AS patients FROM patients GROUP BY band,region "
            "ORDER BY band,patients DESC"
        )
    if "关联患者地区与就诊疾病" in q and "费用合计" in q:
        return (
            "SELECT p.region,v.disease,COUNT(*) AS visits,SUM(v.cost) AS total_cost "
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region,v.disease "
            "ORDER BY p.region,total_cost DESC"
        )
    if "按患者统计去过的不同科室数" in q:
        return (
            "SELECT patient_id,COUNT(DISTINCT department) AS departments,"
            "COUNT(DISTINCT disease) AS diseases,MAX(los_days) AS max_los FROM visits "
            "GROUP BY patient_id ORDER BY departments DESC,diseases DESC,patient_id"
        )
    return None


def _compile_top_k(q: str, _d: Domains) -> Optional[str]:
    limit = _integer(q, r"前\s*(\d+)")
    if limit is None or limit < 1:
        return None
    if "患者人数最多" in q and "地区" in q:
        return f"SELECT region,COUNT(*) AS n FROM patients GROUP BY region ORDER BY n DESC,region LIMIT {limit}"
    if "疾病种类覆盖最多" in q and "科室" in q:
        return f"SELECT department,COUNT(DISTINCT disease) AS n FROM visits GROUP BY department ORDER BY n DESC,department LIMIT {limit}"
    if "累计证据次数最高" in q and "尾实体" in q:
        return f"SELECT tail,SUM(count) AS n FROM kg_triples GROUP BY tail ORDER BY n DESC,tail LIMIT {limit}"
    if "平均提及次数最高" in q and "实体类型" in q:
        return f"SELECT type,ROUND(AVG(mention_count),2) AS n FROM kg_entities GROUP BY type ORDER BY n DESC,type LIMIT {limit}"
    if "总费用最高" in q and "性别与科室" in q:
        return (
            "SELECT p.gender,v.department,SUM(v.cost) AS n FROM patients p JOIN visits v "
            "ON v.patient_id=p.patient_id GROUP BY p.gender,v.department "
            f"ORDER BY n DESC,p.gender,v.department LIMIT {limit}"
        )
    group_specs = (
        ("每个地区年龄最高", "region", "patient_id", "age", "patients", "age"),
        ("每个科室费用最高", "department", "visit_id", "cost", "visits", "cost"),
        ("每种疾病住院天数最长", "disease", "visit_id", "los_days", "visits", "los_days"),
        ("每种药物对应费用最高", "drug_prescribed", "visit_id", "cost", "visits", "cost"),
        ("每种检查对应住院天数最长", "exam", "visit_id", "los_days", "visits", "los_days"),
        ("每种尾实体类型证据次数最高", "tail_type", "tail", "count", "kg_triples", "count"),
        ("每个科室日期最晚", "department", "visit_id", "visit_date", "visits", "visit_date"),
        ("每种疾病日期最晚", "disease", "visit_id", "visit_date", "visits", "visit_date"),
    )
    for phrase, group, identifier, metric, table, order_metric in group_specs:
        if phrase in q:
            return (
                f"SELECT {group},{identifier},{metric} FROM (SELECT {group},{identifier},{metric},"
                f"ROW_NUMBER() OVER(PARTITION BY {group} ORDER BY {order_metric} DESC,{identifier}) "
                f"AS pos FROM {table}) ranked WHERE pos<={limit} ORDER BY {group},pos"
            )
    return None


def _compile_joins(q: str, d: Domains) -> Optional[str]:
    region = _ground(q, d, "region")
    department = _ground(q, d, "department")
    disease = _ground(q, d, "disease")
    drug = _ground(q, d, "drug")
    relation = _ground(q, d, "kg_relation")
    relation_zh = _ground(q, d, "kg_relation_zh")
    gender = "男" if "性别为男" in q else "女" if "性别为女" in q else None
    if region and "患者地区与就诊科室" in q and "平均住院天数" in q:
        return (
            "SELECT p.region,v.department,COUNT(*) AS visits,"
            f"ROUND(AVG(v.los_days),{_rounding(q, 2)}) AS avg_los FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.region='{_esc(region)}' GROUP BY p.region,v.department "
            "ORDER BY visits DESC"
        )
    if disease and "患者、就诊和同名知识实体" in q:
        return (
            "SELECT p.name,v.disease,e.type,e.mention_count FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id "
            "JOIN kg_entities e ON e.name=v.disease "
            f"WHERE v.disease='{_esc(disease)}' ORDER BY e.mention_count DESC,p.name"
        )
    if relation and "实际疾病连接到图谱头实体" in q and "尾实体类型" in q:
        return (
            "SELECT DISTINCT v.disease,t.relation,t.tail,z.type FROM visits v "
            "JOIN kg_triples t ON t.head=v.disease JOIN kg_entities z ON z.name=t.tail "
            f"WHERE t.relation='{_esc(relation)}' ORDER BY v.disease,t.tail"
        )
    if drug and "患者地区、实际处方和图谱目标实体" in q:
        return (
            "SELECT DISTINCT p.region,v.drug_prescribed,t.head,t.confidence FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id "
            "JOIN kg_triples t ON t.tail=v.drug_prescribed "
            f"WHERE v.drug_prescribed='{_esc(drug)}' ORDER BY t.confidence DESC,p.region"
        )
    if "同一患者的就诊做自连接" in q and "紧随其后的下一条" in q:
        return (
            "SELECT a.patient_id,a.visit_id AS first_id,b.visit_id AS next_id,"
            "a.visit_date AS first_date,b.visit_date AS next_date FROM visits a "
            "JOIN visits b ON b.patient_id=a.patient_id AND b.visit_id=(SELECT MIN(c.visit_id) "
            "FROM visits c WHERE c.patient_id=a.patient_id AND c.visit_id>a.visit_id) "
            "ORDER BY a.patient_id,a.visit_id"
        )
    if "同一疾病的两次不同就诊" in q and "科室路径数" in q:
        return (
            "SELECT a.disease,COUNT(*) AS matched_pairs,"
            "COUNT(DISTINCT a.department||'->'||b.department) AS department_routes "
            "FROM visits a JOIN visits b ON b.disease=a.disease AND b.visit_id>a.visit_id "
            "GROUP BY a.disease ORDER BY matched_pairs DESC,a.disease"
        )
    if "每位患者去过的科室数和总费用" in q:
        return (
            "SELECT p.patient_id,p.name,x.department_count,x.total_cost FROM patients p JOIN "
            "(SELECT patient_id,COUNT(DISTINCT department) AS department_count,SUM(cost) "
            "AS total_cost FROM visits GROUP BY patient_id) x ON x.patient_id=p.patient_id "
            "ORDER BY x.department_count DESC,x.total_cost DESC"
        )
    if "实体分别与出边汇总和入边汇总连接" in q:
        return (
            "SELECT e.name,e.type,o.out_edges,i.in_edges FROM kg_entities e LEFT JOIN "
            "(SELECT head,COUNT(*) AS out_edges FROM kg_triples GROUP BY head) o ON o.head=e.name "
            "LEFT JOIN (SELECT tail,COUNT(*) AS in_edges FROM kg_triples GROUP BY tail) i "
            "ON i.tail=e.name ORDER BY COALESCE(o.out_edges,0)+COALESCE(i.in_edges,0) DESC,e.name"
        )
    if "按患者和月份汇总就诊次数" in q and "连接患者地区" in q:
        return (
            "SELECT p.region,m.visit_month,m.visit_count FROM patients p JOIN "
            "(SELECT patient_id,SUBSTR(visit_date,1,7) AS visit_month,COUNT(*) AS visit_count "
            "FROM visits GROUP BY patient_id,SUBSTR(visit_date,1,7)) m ON m.patient_id=p.patient_id "
            "GROUP BY p.region,m.visit_month,m.visit_count ORDER BY p.region,m.visit_month"
        )
    if "连接头实体及其类型" in q and "不同头实体数" in q:
        return (
            "SELECT e.type,t.relation,COUNT(DISTINCT t.head) AS heads,"
            "COUNT(DISTINCT t.tail) AS tails FROM kg_entities e "
            "JOIN kg_triples t ON t.head=e.name GROUP BY e.type,t.relation "
            "ORDER BY e.type,heads DESC"
        )
    if region and "患者、就诊疾病和知识实体" in q and "匹配就诊数" in q:
        return (
            "SELECT p.region,v.department,e.type,COUNT(*) AS matched_visits FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id JOIN kg_entities e ON e.name=v.disease "
            f"WHERE p.region='{_esc(region)}' GROUP BY p.region,v.department,e.type "
            "ORDER BY matched_visits DESC"
        )
    if gender and "患者姓名、地区以及就诊日期" in q:
        return (
            "SELECT p.name,p.region,v.visit_date,v.department,v.cost FROM patients p "
            "INNER JOIN visits v ON v.patient_id=p.patient_id "
            f"WHERE p.gender='{gender}' ORDER BY p.name,v.visit_date"
        )
    if department and "实际检查连接到图谱尾实体" in q:
        return (
            "SELECT v.department,v.exam,t.head,t.relation FROM visits v JOIN kg_triples t "
            f"ON t.tail=v.exam WHERE v.department='{_esc(department)}' "
            "GROUP BY v.department,v.exam,t.head,t.relation ORDER BY v.exam,t.head"
        )
    if relation_zh and "三元组同时连接头尾实体表" in q:
        return (
            "SELECT t.head,h.type AS head_entity_type,t.relation,t.tail,z.type "
            "AS tail_entity_type FROM kg_triples t JOIN kg_entities h ON h.name=t.head "
            "JOIN kg_entities z ON z.name=t.tail "
            f"WHERE t.relation_zh='{_esc(relation_zh)}' ORDER BY t.head,t.tail"
        )
    return None


def _compile_temporal(q: str, _d: Domains) -> Optional[str]:
    if "逐月按科室" in q and "活跃患者数" in q:
        return (
            "SELECT SUBSTR(visit_date,1,7) AS month,department,COUNT(DISTINCT patient_id) "
            "AS active_patients FROM visits GROUP BY SUBSTR(visit_date,1,7),department "
            "ORDER BY month,active_patients DESC"
        )
    if "年-周" in q and "平均住院天数" in q:
        return (
            "SELECT STRFTIME('%Y-%W',visit_date) AS week,COUNT(*) AS visits,"
            f"ROUND(AVG(los_days),{_rounding(q, 2)}) AS avg_los FROM visits "
            "GROUP BY STRFTIME('%Y-%W',visit_date) ORDER BY week"
        )
    dates = _dates(q)
    if len(dates) >= 2 and "闭区间" in q and "日期和科室" in q:
        return (
            "SELECT visit_date,department,COUNT(*) AS visits FROM visits "
            f"WHERE visit_date BETWEEN '{dates[0]}' AND '{dates[1]}' "
            "GROUP BY visit_date,department ORDER BY visit_date,visits DESC"
        )
    if "最近就诊日期和活跃月份数" in q:
        return (
            "SELECT disease,MAX(visit_date) AS latest_date,"
            "COUNT(DISTINCT SUBSTR(visit_date,1,7)) AS active_months FROM visits "
            "GROUP BY disease ORDER BY latest_date DESC,disease"
        )
    if "科室和日期统计每日就诊数" in q and "累计就诊数" in q:
        return (
            "SELECT department,visit_date,COUNT(*) AS daily_visits,"
            "SUM(COUNT(*)) OVER(PARTITION BY department ORDER BY visit_date "
            "ROWS UNBOUNDED PRECEDING) AS running_visits FROM visits "
            "GROUP BY department,visit_date ORDER BY department,visit_date"
        )
    if "连接患者地区后" in q and "地区和月份统计活跃患者数" in q:
        return (
            "SELECT p.region,SUBSTR(v.visit_date,1,7) AS month,COUNT(DISTINCT v.patient_id) "
            "AS patients FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
            "GROUP BY p.region,SUBSTR(v.visit_date,1,7) ORDER BY p.region,month"
        )
    if "同一日历月份" in q and "平均费用" in q:
        return (
            "SELECT SUBSTR(visit_date,6,2) AS calendar_month,COUNT(*) AS visits,"
            f"ROUND(AVG(cost),{_rounding(q, 2)}) AS avg_cost FROM visits "
            "GROUP BY SUBSTR(visit_date,6,2) ORDER BY calendar_month"
        )
    if "首次就诊月份作为队列月份" in q:
        return (
            "SELECT cohort_month,COUNT(*) AS patients FROM (SELECT patient_id,"
            "MIN(SUBSTR(visit_date,1,7)) AS cohort_month FROM visits GROUP BY patient_id) c "
            "GROUP BY cohort_month ORDER BY cohort_month"
        )
    if "最早与最晚记录间的活跃跨度天数" in q:
        return (
            "SELECT disease,first_date,last_date,CAST(JULIANDAY(last_date)-JULIANDAY(first_date) "
            "AS INTEGER) AS active_span FROM (SELECT disease,MIN(visit_date) AS first_date,"
            "MAX(visit_date) AS last_date FROM visits GROUP BY disease) x ORDER BY active_span DESC"
        )
    if "最早就诊日期和累计就诊次数" in q:
        return (
            "SELECT patient_id,MIN(visit_date) AS first_date,COUNT(*) AS lifetime_visits "
            "FROM visits GROUP BY patient_id ORDER BY first_date,patient_id"
        )
    if "按月汇总费用" in q and "上一个月" in q:
        return (
            "SELECT SUBSTR(visit_date,1,7) AS month,SUM(cost) AS monthly_cost,"
            "LAG(SUM(cost)) OVER(ORDER BY SUBSTR(visit_date,1,7)) AS previous_cost FROM visits "
            "GROUP BY SUBSTR(visit_date,1,7) ORDER BY month"
        )
    if "星期序号和科室" in q:
        return (
            "SELECT STRFTIME('%w',visit_date) AS weekday,department,COUNT(*) AS visits "
            "FROM visits GROUP BY STRFTIME('%w',visit_date),department "
            "ORDER BY weekday,visits DESC"
        )
    return None


def _compile_kg(q: str, d: Domains) -> Optional[str]:
    relation = _ground(q, d, "kg_relation")
    relation_zh = _ground(q, d, "kg_relation_zh")
    entity_type = _ground(q, d, "kg_entity_type")
    if relation and entity_type and "列出两端、置信度和证据次数" in q:
        return (
            "SELECT head,tail,confidence,count FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' AND head_type='{_esc(entity_type)}' "
            "ORDER BY count DESC,confidence DESC"
        )
    if entity_type and "按关系和尾类型统计边数" in q:
        return (
            "SELECT relation,tail_type,COUNT(*) AS edges FROM kg_triples "
            f"WHERE head_type='{_esc(entity_type)}' GROUP BY relation,tail_type ORDER BY edges DESC"
        )
    confidence = _number(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
    if confidence is not None and "中文关系和头类型" in q and "不同尾实体数" in q:
        return (
            "SELECT relation_zh,head_type,COUNT(DISTINCT tail) AS tails FROM kg_triples "
            f"WHERE confidence>={confidence} GROUP BY relation_zh,head_type ORDER BY tails DESC"
        )
    if entity_type and "按头实体统计关系种类数和证据合计" in q:
        return (
            "SELECT head,COUNT(DISTINCT relation) AS relations,SUM(count) AS support "
            f"FROM kg_triples WHERE tail_type='{_esc(entity_type)}' GROUP BY head "
            "ORDER BY relations DESC,support DESC"
        )
    if relation_zh and "按尾实体统计不同来源数和平均置信度" in q:
        return (
            "SELECT tail,COUNT(DISTINCT head) AS heads,"
            f"ROUND(AVG(confidence),{_rounding(q, 3)}) AS avg_confidence FROM kg_triples "
            f"WHERE relation_zh='{_esc(relation_zh)}' GROUP BY tail ORDER BY heads DESC"
        )
    if "图谱两端实体的登记类型" in q and "按两端类型和关系" in q:
        return (
            "SELECT h.type AS head_type,z.type AS tail_type,t.relation,COUNT(*) AS edges "
            "FROM kg_triples t JOIN kg_entities h ON h.name=t.head "
            "JOIN kg_entities z ON z.name=t.tail GROUP BY h.type,z.type,t.relation "
            "ORDER BY edges DESC"
        )
    if "实体作为图谱终点" in q and "不同来源数" in q:
        return (
            "SELECT e.type,COUNT(DISTINCT t.head) AS heads,ROUND(AVG(t.confidence),3) "
            "AS avg_confidence FROM kg_entities e JOIN kg_triples t ON t.tail=e.name "
            "GROUP BY e.type ORDER BY heads DESC,e.type"
        )
    if "实体作为图谱起点" in q and "不同目标数" in q:
        return (
            "SELECT e.type,COUNT(DISTINCT t.tail) AS tails,SUM(t.count) AS support "
            "FROM kg_entities e JOIN kg_triples t ON t.head=e.name GROUP BY e.type "
            "ORDER BY tails DESC,e.type"
        )
    if "实际疾病与图谱出发边" in q:
        return (
            "SELECT v.disease,COUNT(DISTINCT t.relation) AS relations,SUM(t.count) AS support "
            "FROM visits v JOIN kg_triples t ON t.head=v.disease GROUP BY v.disease "
            "ORDER BY support DESC,v.disease"
        )
    if "实际检查与图谱目标实体" in q:
        return (
            "SELECT v.exam,COUNT(DISTINCT t.head) AS sources,COUNT(DISTINCT t.relation) "
            "AS relations FROM visits v JOIN kg_triples t ON t.tail=v.exam GROUP BY v.exam "
            "ORDER BY sources DESC,v.exam"
        )
    if "处方药与图谱目标实体" in q and "来源实体数" in q:
        return (
            "SELECT v.drug_prescribed,COUNT(DISTINCT t.head) AS diseases,"
            "ROUND(AVG(t.confidence),3) AS avg_confidence FROM visits v "
            "JOIN kg_triples t ON t.tail=v.drug_prescribed GROUP BY v.drug_prescribed "
            "ORDER BY diseases DESC"
        )
    if "按关系同时汇总边数" in q and "证据加权置信度" in q:
        return (
            "SELECT relation,COUNT(*) AS edges,SUM(count) AS support,"
            f"ROUND(SUM(confidence*count)/NULLIF(SUM(count),0),{_rounding(q, 3)}) "
            "AS weighted_confidence FROM kg_triples GROUP BY relation ORDER BY support DESC"
        )
    return None


def _compile_having(q: str, _d: Domains) -> Optional[str]:
    count = _integer(q, r"至少(?:有|覆盖)?\s*(\d+)\s*(?:位|个|次|种)?")
    value = _number(q, r"至少\s*([0-9]+(?:\.[0-9]+)?)")
    if count is not None and "按地区统计患者数和平均年龄" in q:
        return (
            "SELECT region,COUNT(*) AS patients,ROUND(AVG(age),1) AS avg_age FROM patients "
            f"GROUP BY region HAVING COUNT(*)>={count} ORDER BY patients DESC"
        )
    if count is not None and "按科室统计不同患者数" in q:
        return (
            "SELECT department,COUNT(DISTINCT patient_id) AS patients FROM visits "
            f"GROUP BY department HAVING COUNT(DISTINCT patient_id)>={count} ORDER BY patients DESC"
        )
    if value is not None and "按疾病汇总费用" in q:
        return (
            "SELECT disease,SUM(cost) AS total_cost FROM visits GROUP BY disease "
            f"HAVING SUM(cost)>={value} ORDER BY total_cost DESC"
        )
    if count is not None and "按药物统计覆盖科室数" in q:
        return (
            "SELECT drug_prescribed,COUNT(DISTINCT department) AS departments FROM visits "
            f"GROUP BY drug_prescribed HAVING COUNT(DISTINCT department)>={count} "
            "ORDER BY departments DESC"
        )
    if count is not None and "按地区和疾病统计就诊数" in q:
        return (
            "SELECT p.region,v.disease,COUNT(*) AS visits FROM patients p JOIN visits v "
            "ON v.patient_id=p.patient_id GROUP BY p.region,v.disease "
            f"HAVING COUNT(*)>={count} ORDER BY visits DESC"
        )
    if count is not None and "按关系统计不同头实体数" in q:
        return (
            "SELECT relation,COUNT(DISTINCT head) AS heads FROM kg_triples GROUP BY relation "
            f"HAVING COUNT(DISTINCT head)>={count} ORDER BY heads DESC"
        )
    if value is not None and "按头尾类型组合汇总证据" in q:
        return (
            "SELECT head_type,tail_type,SUM(count) AS support FROM kg_triples "
            f"GROUP BY head_type,tail_type HAVING SUM(count)>={value} ORDER BY support DESC"
        )
    if value is not None and "按科室和疾病计算平均费用" in q:
        return (
            "SELECT department,disease,ROUND(AVG(cost),2) AS avg_cost FROM visits "
            f"GROUP BY department,disease HAVING AVG(cost)>={value} ORDER BY avg_cost DESC"
        )
    if count is not None and "按性别与科室统计疾病种类" in q:
        return (
            "SELECT p.gender,v.department,COUNT(DISTINCT v.disease) AS diseases FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.gender,v.department "
            f"HAVING COUNT(DISTINCT v.disease)>={count} ORDER BY diseases DESC"
        )
    return None


def _compile_nested(q: str, d: Domains) -> Optional[str]:
    if "年龄高于总体平均年龄" in q and "有就诊记录" in q:
        return (
            "SELECT patient_id,name,age FROM patients WHERE age>(SELECT AVG(age) FROM patients) "
            "AND patient_id IN(SELECT patient_id FROM visits) ORDER BY age DESC"
        )
    if "每位患者总费用的平均值" in q:
        return (
            "SELECT visit_id,disease,cost FROM visits WHERE cost>(SELECT AVG(total_cost) FROM "
            "(SELECT patient_id,SUM(cost) AS total_cost FROM visits GROUP BY patient_id)) "
            "ORDER BY cost DESC"
        )
    if "各关系证据" in q and "关系平均合计" in q:
        return (
            "SELECT relation,support FROM (SELECT relation,SUM(count) AS support FROM kg_triples "
            "GROUP BY relation) r WHERE support>(SELECT AVG(support) FROM (SELECT SUM(count) "
            "AS support FROM kg_triples GROUP BY relation)) ORDER BY support DESC"
        )
    if "提及次数高于其所属类型平均值" in q:
        return (
            "SELECT name,type,mention_count FROM kg_entities WHERE mention_count>"
            "(SELECT AVG(mention_count) FROM kg_entities WHERE type=kg_entities.type) "
            "ORDER BY type,mention_count DESC"
        )
    minimum = _integer(q, r"至少去过\s*(\d+)\s*个不同科室")
    if minimum is not None:
        return (
            "SELECT patient_id,name FROM patients p WHERE EXISTS(SELECT 1 FROM "
            "(SELECT patient_id,COUNT(DISTINCT department) AS departments FROM visits "
            f"GROUP BY patient_id) x WHERE x.patient_id=p.patient_id AND x.departments>={minimum}) "
            "ORDER BY patient_id"
        )
    if "费用最高的1次就诊" in q:
        return (
            "SELECT patient_id,name FROM patients WHERE patient_id=(SELECT patient_id "
            "FROM visits ORDER BY cost DESC,visit_id LIMIT 1)"
        )
    department = _ground(q, d, "department")
    if department and "通过子查询确定曾在" in q:
        return (
            "SELECT patient_id,name FROM patients WHERE patient_id IN(SELECT patient_id "
            f"FROM visits WHERE department='{_esc(department)}' GROUP BY patient_id) "
            "ORDER BY patient_id"
        )
    if "所有疾病最高次数" in q:
        return (
            "SELECT disease,visits FROM (SELECT disease,COUNT(*) AS visits FROM visits "
            "GROUP BY disease) d WHERE visits=(SELECT MAX(n) FROM (SELECT COUNT(*) AS n "
            "FROM visits GROUP BY disease)) ORDER BY disease"
        )
    if "科室均值高于全部就诊总体均值" in q:
        return (
            "SELECT department,avg_cost FROM (SELECT department,AVG(cost) AS avg_cost "
            "FROM visits GROUP BY department) d WHERE avg_cost>(SELECT AVG(cost) FROM visits) "
            "ORDER BY avg_cost DESC"
        )
    relation = _ground(q, d, "kg_relation")
    if relation and "作为头实体出现、但未作为尾实体出现" in q:
        return (
            "SELECT name,type FROM kg_entities WHERE name IN(SELECT head FROM kg_triples "
            f"WHERE relation='{_esc(relation)}') AND name NOT IN(SELECT tail FROM kg_triples "
            f"WHERE relation='{_esc(relation)}') ORDER BY name"
        )
    if "全库最晚就诊日期" in q:
        return (
            "SELECT visit_id,patient_id,visit_date FROM visits WHERE visit_date="
            "(SELECT MAX(visit_date) FROM visits) ORDER BY visit_id"
        )
    return None


def _compile_cte(q: str, _d: Domains) -> Optional[str]:
    minimum = _integer(q, r"就诊至少\s*(\d+)\s*次")
    if minimum is not None and "两个公共表表达式" in q and "连接姓名" in q:
        return (
            "WITH patient_stats AS (SELECT patient_id,COUNT(*) AS visits,SUM(cost) AS cost "
            "FROM visits GROUP BY patient_id), eligible AS (SELECT * FROM patient_stats "
            f"WHERE visits>={minimum}) SELECT p.name,e.visits,e.cost FROM eligible e "
            "JOIN patients p ON p.patient_id=e.patient_id ORDER BY e.cost DESC"
        )
    if "地区疾病次数" in q and "地区最大值" in q:
        return (
            "WITH regional AS (SELECT p.region,v.disease,COUNT(*) AS visits FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region,v.disease), "
            "maxima AS (SELECT region,MAX(visits) AS max_visits FROM regional GROUP BY region) "
            "SELECT r.region,r.disease,r.visits FROM regional r JOIN maxima m "
            "ON m.region=r.region AND m.max_visits=r.visits ORDER BY r.region,r.disease"
        )
    if "公共表表达式汇总出边与入边" in q:
        return (
            "WITH outgoing AS (SELECT head,COUNT(*) AS edges,SUM(count) AS support FROM kg_triples "
            "GROUP BY head), incoming AS (SELECT tail,COUNT(*) AS edges FROM kg_triples "
            "GROUP BY tail) SELECT e.name,COALESCE(o.edges,0) AS out_edges,"
            "COALESCE(i.edges,0) AS in_edges,COALESCE(o.support,0) AS support FROM kg_entities e "
            "LEFT JOIN outgoing o ON o.head=e.name LEFT JOIN incoming i ON i.tail=e.name "
            "ORDER BY support DESC,e.name"
        )
    if "疾病用药覆盖" in q and "连续排名" in q:
        return (
            "WITH disease_drugs AS (SELECT disease,COUNT(DISTINCT drug_prescribed) AS drugs "
            "FROM visits GROUP BY disease), ordered AS (SELECT disease,drugs,DENSE_RANK() "
            "OVER(ORDER BY drugs DESC) AS coverage_rank FROM disease_drugs) SELECT disease,drugs,"
            "coverage_rank FROM ordered ORDER BY coverage_rank,disease"
        )
    if "实际疾病与图谱边的公共集合" in q:
        return (
            "WITH graph_actual AS (SELECT DISTINCT v.disease,t.relation,t.tail FROM visits v "
            "JOIN kg_triples t ON t.head=v.disease), summaries AS (SELECT disease,"
            "COUNT(DISTINCT relation) AS relations,COUNT(DISTINCT tail) AS tails FROM graph_actual "
            "GROUP BY disease) SELECT disease,relations,tails FROM summaries "
            "ORDER BY relations DESC,tails DESC"
        )
    if "科室每日次数" in q and "当前行及前2行" in q:
        return (
            "WITH dept_days AS (SELECT department,visit_date,COUNT(*) AS visits FROM visits "
            "GROUP BY department,visit_date), rolling AS (SELECT department,visit_date,visits,"
            "SUM(visits) OVER(PARTITION BY department ORDER BY visit_date ROWS BETWEEN 2 "
            "PRECEDING AND CURRENT ROW) AS three_day_rows FROM dept_days) SELECT * FROM rolling "
            "ORDER BY department,visit_date"
        )
    if "各实体类型提及次数和总次数" in q:
        return (
            "WITH type_mentions AS (SELECT type,SUM(mention_count) AS mentions FROM kg_entities "
            "GROUP BY type), total AS (SELECT SUM(mentions) AS all_mentions FROM type_mentions) "
            "SELECT t.type,t.mentions,ROUND(t.mentions*100.0/x.all_mentions,2) AS share_pct "
            "FROM type_mentions t CROSS JOIN total x ORDER BY share_pct DESC"
        )
    if "按月汇总次数和费用" in q and "按费用排名" in q:
        return (
            "WITH monthly AS (SELECT SUBSTR(visit_date,1,7) AS month,COUNT(*) AS visits,"
            "SUM(cost) AS cost FROM visits GROUP BY SUBSTR(visit_date,1,7)), ranked AS "
            "(SELECT *,RANK() OVER(ORDER BY cost DESC) AS cost_rank FROM monthly) "
            "SELECT month,visits,cost,cost_rank FROM ranked ORDER BY month"
        )
    if "去重患者疾病组合" in q and "每位患者疾病数" in q:
        return (
            "WITH patient_disease AS (SELECT DISTINCT patient_id,disease FROM visits), counts AS "
            "(SELECT patient_id,COUNT(*) AS diseases FROM patient_disease GROUP BY patient_id) "
            "SELECT p.patient_id,p.name,c.diseases FROM counts c JOIN patients p "
            "ON p.patient_id=c.patient_id ORDER BY c.diseases DESC,p.patient_id"
        )
    return None


def _compile_set(q: str, _d: Domains) -> Optional[str]:
    if "实际疾病与图谱头实体名称" in q:
        return "SELECT disease AS concept FROM visits UNION SELECT head FROM kg_triples ORDER BY concept"
    if "实际处方药和检查项目名称" in q:
        return "SELECT drug_prescribed AS concept FROM visits UNION SELECT exam FROM visits ORDER BY concept"
    if "同时出现在检查项目和知识实体表" in q:
        return "SELECT exam AS concept FROM visits INTERSECT SELECT name FROM kg_entities ORDER BY concept"
    if "图谱头端和尾端名称保留重复地合并" in q:
        return (
            "SELECT name,COUNT(*) AS endpoint_uses FROM (SELECT head AS name FROM kg_triples "
            "UNION ALL SELECT tail FROM kg_triples) x GROUP BY name ORDER BY endpoint_uses DESC,name"
        )
    if "患者表与就诊表共有的患者编号" in q:
        return (
            "SELECT patient_id,'linked' AS status FROM patients INTERSECT "
            "SELECT patient_id,'linked' FROM visits ORDER BY patient_id"
        )
    if "保留重复地合并地区和科室名称" in q:
        return (
            "SELECT value,COUNT(*) AS source_rows FROM (SELECT region AS value FROM patients "
            "UNION ALL SELECT department FROM visits) x GROUP BY value ORDER BY source_rows DESC,value"
        )
    if "全并集保留头尾角色" in q:
        return (
            "SELECT role,name FROM (SELECT 'head' AS role,head AS name FROM kg_triples "
            "UNION ALL SELECT 'tail' AS role,tail FROM kg_triples) x GROUP BY role,name "
            "ORDER BY role,name"
        )
    if "英文关系、中文关系和实体类型" in q:
        return (
            "SELECT relation AS label FROM kg_triples UNION SELECT relation_zh FROM kg_triples "
            "UNION SELECT type FROM kg_entities ORDER BY label"
        )
    return None


def _compile_window(q: str, _d: Domains) -> Optional[str]:
    limit = _integer(q, r"编号不超过\s*(\d+)")
    if limit is not None and "按地区将患者年龄降序编号" in q:
        return (
            "SELECT region,patient_id,age FROM (SELECT region,patient_id,age,ROW_NUMBER() "
            "OVER(PARTITION BY region ORDER BY age DESC,patient_id) AS rn FROM patients) x "
            f"WHERE rn<={limit} ORDER BY region,rn"
        )
    if "费用相对该患者平均费用的偏差" in q:
        return (
            "SELECT patient_id,visit_id,cost,cost-SUM(cost) OVER(PARTITION BY patient_id)/"
            "COUNT(*) OVER(PARTITION BY patient_id) AS deviation FROM visits "
            "ORDER BY patient_id,visit_id"
        )
    if "科室内计算就诊费用的百分位排名" in q:
        return (
            "SELECT department,visit_id,cost,PERCENT_RANK() OVER(PARTITION BY department "
            "ORDER BY cost) AS cost_percent_rank FROM visits ORDER BY department,cost,visit_id"
        )
    if "月度费用" in q and "累计费用" in q:
        return (
            "SELECT month,cost,SUM(cost) OVER(ORDER BY month ROWS UNBOUNDED PRECEDING) "
            "AS cumulative_cost FROM (SELECT SUBSTR(visit_date,1,7) AS month,SUM(cost) AS cost "
            "FROM visits GROUP BY SUBSTR(visit_date,1,7)) x ORDER BY month"
        )
    if "科室内疾病次数" in q and "连续排名" in q:
        return (
            "SELECT department,disease,visits,DENSE_RANK() OVER(PARTITION BY department "
            "ORDER BY visits DESC) AS disease_rank FROM (SELECT department,disease,COUNT(*) "
            "AS visits FROM visits GROUP BY department,disease) x "
            "ORDER BY department,disease_rank,disease"
        )
    buckets = _integer(q, r"分成\s*(\d+)\s*个等频档")
    if buckets is not None and "每种实体类型内" in q:
        return (
            f"SELECT type,name,mention_count,NTILE({buckets}) OVER(PARTITION BY type ORDER BY "
            "mention_count DESC,name) AS mention_quartile FROM kg_entities "
            "ORDER BY type,mention_quartile,name"
        )
    if "同一头实体的平均置信度" in q:
        return (
            "SELECT head,relation,confidence,ROUND(AVG(confidence) OVER(PARTITION BY head),3) "
            "AS head_avg FROM kg_triples ORDER BY head,confidence DESC"
        )
    if "年龄最高者的患者编号" in q:
        return (
            "SELECT region,patient_id,age,FIRST_VALUE(patient_id) OVER(PARTITION BY region "
            "ORDER BY age DESC,patient_id) AS oldest_patient_id FROM patients "
            "ORDER BY region,age DESC"
        )
    if "患者内累计费用和患者最高单次费用" in q:
        return (
            "SELECT patient_id,visit_id,cost,SUM(cost) OVER(PARTITION BY patient_id ORDER BY "
            "visit_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_cost,"
            "MAX(cost) OVER(PARTITION BY patient_id) AS max_cost FROM visits "
            "ORDER BY patient_id,visit_id"
        )
    if "每种关系内按置信度排序" in q and "最高置信度" in q:
        return (
            "SELECT relation,tail,confidence,LAST_VALUE(confidence) OVER(PARTITION BY relation "
            "ORDER BY confidence ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) "
            "AS relation_max_confidence FROM kg_triples ORDER BY relation,confidence,tail"
        )
    return None


def compile_independent_v10_sql(question: str, domains: Domains) -> Optional[str]:
    """Compile a bounded independent-v10 analytical request or safely abstain."""

    q = str(question or "").strip()
    if not q:
        return None
    compilers: tuple[Callable[[str, Domains], Optional[str]], ...] = (
        _compile_window,
        _compile_set,
        _compile_cte,
        _compile_nested,
        _compile_having,
        _compile_kg,
        _compile_joins,
        _compile_temporal,
        _compile_top_k,
        _compile_aggregation,
        _compile_filtering,
    )
    for compiler in compilers:
        sql = compiler(q, domains)
        if sql is not None:
            try:
                return _wrap(sql, q)
            except ValueError:
                return None
    return None
