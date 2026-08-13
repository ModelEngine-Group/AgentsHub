"""Bounded deterministic compiler for natural four-table analytical Chinese.

This module implements compositional operation families selected only from the
independent development member.  It never reads an evaluation file at runtime.
Every text slot is grounded in the active database domains, every numeric slot
is parsed, and every emitted statement passes the shared one-statement,
read-only, four-table SQL validator.
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


def _values(question: str, domains: Domains, key: str) -> list[str]:
    return sorted(
        {str(value) for value in domains.get(key, ()) if value and str(value) in question},
        key=lambda value: (-len(value), value),
    )


def _value(question: str, domains: Domains, key: str) -> Optional[str]:
    values = _values(question, domains, key)
    return values[0] if values else None


def _relation(question: str, domains: Domains) -> Optional[str]:
    return _value(question, domains, "kg_relation")


def _outer_limit(question: str) -> Optional[int]:
    matches: list[str] = []
    for pattern in (
        r"只保留既定排序后的前(\d+)行",
        r"输出窗口覆盖排序后前(\d+)项",
        r"审计截取上限设为(\d+)行",
    ):
        matches.extend(re.findall(pattern, question))
    if not matches:
        return None
    values = {_uint(value) for value in matches}
    if len(values) != 1 or next(iter(values)) <= 0:
        raise ValueError("ambiguous outer limit")
    return next(iter(values))


def _suffix(question: str) -> str:
    limit = _outer_limit(question)
    return f" LIMIT {limit}" if limit is not None else ""


def _top_k(question: str, *patterns: str) -> Optional[int]:
    values: list[int] = []
    for pattern in patterns:
        values.extend(_uint(value) for value in re.findall(pattern, question))
    if not values:
        return None
    if len(set(values)) != 1 or values[0] <= 0:
        raise ValueError("ambiguous top-k")
    return values[0]


def _year(question: str) -> Optional[int]:
    years = {_uint(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)}
    return next(iter(years)) if len(years) == 1 else None


def _decimal_after(question: str, pattern: str) -> Optional[str]:
    values = re.findall(pattern, question)
    if not values:
        return None
    normalized = {_number(value) for value in values}
    return next(iter(normalized)) if len(normalized) == 1 else None


def _integer_after(question: str, pattern: str) -> Optional[int]:
    values = re.findall(pattern, question)
    if not values:
        return None
    normalized = {_uint(value) for value in values}
    return next(iter(normalized)) if len(normalized) == 1 else None


def _compile_filtering(q: str, domains: Domains) -> Optional[str]:
    if all(token in q for token in ("年龄在", "之间", "患者", "编号", "年龄")):
        bounds = re.search(r"年龄在(\d+)至(\d+)岁之间", q)
        gender = "男" if "男性" in q or "性别为男" in q else "女" if "女性" in q or "性别为女" in q else None
        region = _value(q, domains, "region")
        if bounds and gender and region:
            low, high = map(_uint, bounds.groups())
            if low <= high:
                # 投影列由问句驱动：仅当问句显式请求姓名时才返回姓名列，
                # 否则保持"编号 + 年龄"的最小形状（与既有回归一致）。
                include_name = bool(re.search(r"姓名|名字", q))
                projection = "patient_id, name, age" if include_name else "patient_id, age"
                # 排序方向也由问句驱动："年龄较大者在前" 等价于年龄降序，
                # "年龄较小者在前" 为升序；无明确方向时保持最小升序排序。
                if re.search(r"年龄较大|年长|先列年龄大", q):
                    order = (
                        "age DESC, name ASC, patient_id ASC"
                        if include_name
                        else "age DESC, patient_id"
                    )
                elif re.search(r"年龄较小|年龄最小|年幼|低龄", q):
                    order = (
                        "age ASC, name ASC, patient_id ASC"
                        if include_name
                        else "age ASC, patient_id"
                    )
                else:
                    order = "age, patient_id"
                return _safe(
                    f"SELECT {projection} FROM patients "
                    f"WHERE gender='{gender}' AND region='{_esc(region)}' "
                    f"AND age BETWEEN {low} AND {high} ORDER BY {order}{_suffix(q)}"
                )

    if all(token in q for token in ("费用高于", "就诊编号", "日期", "住院天数")):
        department = _value(q, domains, "department")
        cost = _decimal_after(q, r"费用高于([0-9]+(?:\.[0-9]+)?)")
        days = _integer_after(q, r"住院天数(?:不少于|达到)(\d+)天")
        if department and cost is not None:
            day_sql = f" AND los_days>={days}" if days is not None else ""
            return _safe(
                "SELECT visit_id, visit_date, cost, los_days FROM visits "
                f"WHERE department='{_esc(department)}' AND cost>{cost}{day_sql} "
                f"ORDER BY cost DESC, visit_id{_suffix(q)}"
            )

    month = re.search(r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])月", q)
    if month and all(token in q for token in ("就诊", "日期", "疾病")):
        department = _value(q, domains, "department")
        if department:
            month_value = f"{month.group(1)}-{month.group(2)}"
            return _safe(
                "SELECT visit_date, visit_id, disease FROM visits "
                f"WHERE substr(visit_date,1,7)='{month_value}' "
                f"AND department='{_esc(department)}' ORDER BY visit_date, visit_id{_suffix(q)}"
            )

    if all(token in q for token in ("知识图谱实体", "提及次数至少")):
        entity_type = _value(q, domains, "kg_entity_type")
        minimum = _integer_after(q, r"提及次数至少(?:为)?(\d+)")
        if entity_type and minimum is not None:
            return _safe(
                "SELECT name, mention_count FROM kg_entities "
                f"WHERE type='{_esc(entity_type)}' AND mention_count>={minimum} "
                f"ORDER BY mention_count DESC, name{_suffix(q)}"
            )

    if all(token in q for token in ("头尾实体", "置信度")):
        relation = _relation(q, domains)
        confidence = _decimal_after(q, r"置信度(?:不低于|至少为?)([0-9]+(?:\.[0-9]+)?)")
        if relation and confidence is not None:
            return _safe(
                "SELECT head, tail, confidence FROM kg_triples "
                f"WHERE relation='{_esc(relation)}' AND confidence>={confidence} "
                f"ORDER BY confidence DESC, head, tail{_suffix(q)}"
            )

    if all(token in q for token in ("检查项目为", "住院天数", "就诊编号", "疾病", "费用")):
        exam = _value(q, domains, "exam")
        days = _integer_after(q, r"住院天数(?:达到|不少于)(\d+)天")
        if exam and days is not None:
            return _safe(
                "SELECT visit_id, disease, cost FROM visits "
                f"WHERE exam='{_esc(exam)}' AND los_days>={days} "
                f"ORDER BY cost DESC, visit_id{_suffix(q)}"
            )

    if "来自" in q and "或" in q and "患者编号、地区和年龄" in q:
        regions = _values(q, domains, "region")
        age = _integer_after(q, r"年龄高于(\d+)岁")
        gender = "男" if "性别为男" in q else "女" if "性别为女" in q else None
        if len(regions) == 2 and age is not None and gender:
            ordered = sorted(_esc(value) for value in regions)
            return _safe(
                "SELECT patient_id, region, age FROM patients "
                f"WHERE region IN ('{ordered[0]}','{ordered[1]}') "
                f"AND gender='{gender}' AND age>{age} "
                f"ORDER BY region, age DESC, patient_id{_suffix(q)}"
            )
    return None


def _compile_grouping(q: str, domains: Domains) -> Optional[str]:
    suffix = _suffix(q)
    if all(token in q for token in ("按地区统计患者人数", "平均年龄", "最高年龄")):
        return _safe(
            "SELECT region, COUNT(*) AS patient_count, ROUND(AVG(age),2) AS avg_age, "
            "MAX(age) AS max_age FROM patients GROUP BY region "
            f"ORDER BY patient_count DESC, region{suffix}"
        )

    if all(token in q for token in ("每种疾病", "就诊量", "平均住院天数", "平均费用")):
        extra = ", COUNT(DISTINCT department) AS department_count" if "科室数" in q else ""
        return _safe(
            "SELECT disease, COUNT(*) AS visit_count, ROUND(AVG(los_days),2) AS avg_los, "
            f"ROUND(AVG(cost),2) AS avg_cost{extra} FROM visits GROUP BY disease "
            f"ORDER BY visit_count DESC, disease{suffix}"
        )

    if all(token in q for token in ("按实体类型统计", "实体数", "提及次数总和", "平均提及次数")):
        peak = ", MAX(mention_count) AS peak_mentions" if "最高提及次数" in q else ""
        return _safe(
            "SELECT type, COUNT(*) AS entity_count, SUM(mention_count) AS mention_total, "
            f"ROUND(AVG(mention_count),2) AS avg_mentions{peak} FROM kg_entities "
            f"GROUP BY type ORDER BY entity_count DESC, type{suffix}"
        )

    if all(token in q for token in ("按关系类型汇总", "三元组数量", "观察次数合计", "平均置信度")):
        high = ""
        if "另列置信度至少" in q:
            threshold = _decimal_after(q, r"置信度至少([0-9]+(?:\.[0-9]+)?)")
            if threshold is None:
                return None
            high = f", SUM(CASE WHEN confidence>={threshold} THEN 1 ELSE 0 END) AS high_confidence_count"
        return _safe(
            "SELECT relation, COUNT(*) AS triple_count, SUM(count) AS support_total, "
            f"ROUND(AVG(confidence),4) AS avg_confidence{high} FROM kg_triples "
            f"GROUP BY relation ORDER BY triple_count DESC, relation{suffix}"
        )

    if all(token in q for token in ("按疾病计算", "不同患者数", "就诊总费用", "最高单次费用")):
        return _safe(
            "SELECT disease, COUNT(DISTINCT patient_id) AS patient_count, "
            "ROUND(SUM(cost),2) AS total_cost, MAX(cost) AS max_cost FROM visits "
            f"GROUP BY disease ORDER BY patient_count DESC, disease{suffix}"
        )

    if all(token in q for token in ("逐月汇总", "去重患者数", "费用合计")):
        return _safe(
            "SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT patient_id) AS patient_count, ROUND(SUM(cost),2) AS total_cost "
            f"FROM visits GROUP BY month ORDER BY month{suffix}"
        )

    if all(token in q for token in ("按星期几", "就诊次数", "平均费用")):
        return _safe(
            "SELECT strftime('%w',visit_date) AS weekday, COUNT(*) AS visit_count, "
            "ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY weekday ORDER BY weekday"
            f"{suffix}"
        )

    if all(token in q for token in ("按图谱所属科室", "疾病数量", "平均置信度")):
        return _safe(
            "SELECT tail AS department, COUNT(DISTINCT head) AS disease_count, "
            "ROUND(AVG(confidence),4) AS avg_confidence FROM kg_triples "
            "WHERE relation='BELONGS_TO_DEPT' GROUP BY tail "
            f"ORDER BY disease_count DESC, department{suffix}"
        )

    if all(token in q for token in ("按关系类型统计", "三元组数量", "平均观察次数")):
        confidence = _decimal_after(q, r"置信度不低于([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return _safe(
                "SELECT relation, COUNT(*) AS triple_count, ROUND(AVG(count),2) AS avg_support "
                f"FROM kg_triples WHERE confidence>={confidence} GROUP BY relation "
                f"ORDER BY triple_count DESC, relation{suffix}"
            )

    if all(token in q for token in ("每种疾病", "并发症数量", "观察次数合计")):
        return _safe(
            "SELECT head AS disease, COUNT(DISTINCT tail) AS complication_count, "
            "SUM(count) AS support_total FROM kg_triples WHERE relation='HAS_COMPLICATION' "
            f"GROUP BY head ORDER BY complication_count DESC, disease{suffix}"
        )

    if all(token in q for token in ("按科室统计就诊次数", "门诊次数", "住院次数")):
        year = _year(q)
        if year is not None:
            return _safe(
                "SELECT department, COUNT(*) AS visit_count, "
                "SUM(CASE WHEN los_days=0 THEN 1 ELSE 0 END) AS outpatient_count, "
                "SUM(CASE WHEN los_days>0 THEN 1 ELSE 0 END) AS inpatient_count "
                f"FROM visits WHERE substr(visit_date,1,4)='{year}' GROUP BY department "
                f"ORDER BY visit_count DESC, department{suffix}"
            )

    if all(token in q for token in ("各科室", "就诊次数", "总费用", "平均费用", "住院就诊量")):
        return _safe(
            "SELECT department, COUNT(*) AS visit_count, ROUND(SUM(cost),2) AS total_cost, "
            "ROUND(AVG(cost),2) AS avg_cost, "
            "SUM(CASE WHEN los_days>0 THEN 1 ELSE 0 END) AS inpatient_count "
            f"FROM visits GROUP BY department ORDER BY total_cost DESC, department{suffix}"
        )

    if all(token in q for token in ("按患者性别", "就诊次数", "就诊患者数", "平均单次费用")):
        return _safe(
            "SELECT p.gender, COUNT(v.visit_id) AS visit_count, "
            "COUNT(DISTINCT p.patient_id) AS patient_count, "
            "ROUND(AVG(v.cost),2) AS avg_cost FROM patients p JOIN visits v "
            "ON v.patient_id=p.patient_id GROUP BY p.gender "
            f"ORDER BY p.gender{suffix}"
        )

    if all(token in q for token in ("平均置信度从高到低", "前", "关系", "三元组数量")):
        top = _top_k(q, r"前(\d+)种关系")
        if top:
            return _safe(
                "SELECT relation, COUNT(*) AS triple_count, ROUND(AVG(confidence),4) AS avg_confidence "
                "FROM kg_triples GROUP BY relation ORDER BY avg_confidence DESC, relation "
                f"LIMIT {top}"
            )
    return None


def _compile_top_having(q: str, domains: Domains) -> Optional[str]:
    if all(token in q for token in ("平均就诊费用最高", "科室", "就诊次数")):
        top = _top_k(q, r"最高的(\d+)个科室")
        if top:
            return _safe(
                "SELECT department, COUNT(*) AS visit_count, ROUND(AVG(cost),2) AS avg_cost "
                "FROM visits GROUP BY department ORDER BY avg_cost DESC, department "
                f"LIMIT {top}"
            )

    if all(token in q for token in ("按总费用", "疾病", "就诊量", "平均住院天数")):
        top = _top_k(q, r"前(\d+)的疾病")
        if top:
            return _safe(
                "SELECT disease, ROUND(SUM(cost),2) AS total_cost, COUNT(*) AS visit_count, "
                "ROUND(AVG(los_days),2) AS avg_los FROM visits GROUP BY disease "
                f"ORDER BY total_cost DESC, disease LIMIT {top}"
            )

    if all(token in q for token in ("每个头实体", "三元组数", "关联数最多")):
        top = _top_k(q, r"最多的(\d+)个实体")
        if top:
            return _safe(
                "SELECT head, head_type, COUNT(*) AS triple_count FROM kg_triples "
                f"GROUP BY head, head_type ORDER BY triple_count DESC, head LIMIT {top}"
            )

    if all(token in q for token in ("使用次数最多", "非空处方药", "疾病种数")):
        top = _top_k(q, r"最多的(\d+)种")
        if top:
            return _safe(
                "SELECT drug_prescribed, COUNT(*) AS use_count, "
                "COUNT(DISTINCT disease) AS disease_count FROM visits "
                "WHERE LENGTH(drug_prescribed) GROUP BY drug_prescribed "
                f"ORDER BY use_count DESC, drug_prescribed LIMIT {top}"
            )

    if all(token in q for token in ("按提及次数", "知识图谱实体", "实体类型")):
        top = _top_k(q, r"关注的(\d+)个")
        if top:
            return _safe(
                "SELECT name, type, mention_count FROM kg_entities "
                f"ORDER BY mention_count DESC, name LIMIT {top}"
            )

    suffix = _suffix(q)
    if all(token in q for token in ("就诊次数超过", "疾病", "患者数", "平均费用")):
        minimum = _integer_after(q, r"就诊次数超过(\d+)次")
        if minimum is not None:
            return _safe(
                "SELECT disease, COUNT(*) AS visit_count, COUNT(DISTINCT patient_id) AS patient_count, "
                "ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY disease "
                f"HAVING COUNT(*)>{minimum} ORDER BY visit_count DESC, disease{suffix}"
            )

    if all(token in q for token in ("覆盖", "种疾病", "处方药", "使用次数")):
        minimum = _integer_after(q, r"至少覆盖(\d+)种疾病")
        if minimum is not None:
            return _safe(
                "SELECT drug_prescribed, COUNT(*) AS use_count, "
                "COUNT(DISTINCT disease) AS disease_count FROM visits WHERE LENGTH(drug_prescribed) "
                "GROUP BY drug_prescribed "
                f"HAVING COUNT(DISTINCT disease)>={minimum} "
                f"ORDER BY disease_count DESC, drug_prescribed{suffix}"
            )

    if all(token in q for token in ("不同科室出现", "疾病", "就诊量", "科室数")):
        minimum = _integer_after(q, r"不少于(\d+)个不同科室")
        if minimum is not None:
            return _safe(
                "SELECT disease, COUNT(*) AS visit_count, "
                "COUNT(DISTINCT department) AS department_count FROM visits GROUP BY disease "
                f"HAVING COUNT(DISTINCT department)>={minimum} "
                f"ORDER BY department_count DESC, disease{suffix}"
            )

    if all(token in q for token in ("按月份筛出", "就诊量至少", "费用总额超过")):
        count = _integer_after(q, r"就诊量至少(\d+)次")
        total = _decimal_after(q, r"费用总额超过([0-9]+(?:\.[0-9]+)?)")
        if count is not None and total is not None:
            return _safe(
                "SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visit_count, "
                "ROUND(SUM(cost),2) AS total_cost FROM visits GROUP BY month "
                f"HAVING COUNT(*)>={count} AND SUM(cost)>{total} ORDER BY month{suffix}"
            )

    if all(token in q for token in ("平均费用高于", "至少有", "科室", "就诊量")):
        average = _decimal_after(q, r"平均费用高于([0-9]+(?:\.[0-9]+)?)")
        count = _integer_after(q, r"至少有(\d+)次记录")
        if average is not None and count is not None:
            return _safe(
                "SELECT department, COUNT(*) AS visit_count, ROUND(AVG(cost),2) AS avg_cost "
                "FROM visits GROUP BY department "
                f"HAVING COUNT(*)>={count} AND AVG(cost)>{average} "
                f"ORDER BY avg_cost DESC, department{suffix}"
            )

    if all(token in q for token in ("总住院天数不少于", "患者", "就诊次数", "费用总额")):
        minimum = _integer_after(q, r"总住院天数不少于(\d+)天")
        if minimum is not None:
            return _safe(
                "SELECT patient_id, COUNT(*) AS visit_count, SUM(los_days) AS total_los, "
                "ROUND(SUM(cost),2) AS total_cost FROM visits GROUP BY patient_id "
                f"HAVING SUM(los_days)>={minimum} ORDER BY total_los DESC, patient_id{suffix}"
            )

    if all(token in q for token in ("不同症状", "疾病", "观察次数")):
        minimum = _integer_after(q, r"至少(\d+)个不同症状")
        if minimum is not None:
            return _safe(
                "SELECT head AS disease, COUNT(DISTINCT tail) AS symptom_count, "
                "SUM(count) AS support_total FROM kg_triples WHERE relation='HAS_SYMPTOM' "
                f"GROUP BY head HAVING COUNT(DISTINCT tail)>={minimum} "
                f"ORDER BY symptom_count DESC, disease{suffix}"
            )

    if all(token in q for token in ("提及次数总和至少", "实体类型", "实体数", "平均提及次数")):
        minimum = _integer_after(q, r"提及次数总和至少(?:为)?(\d+)")
        if minimum is not None:
            return _safe(
                "SELECT type, COUNT(*) AS entity_count, SUM(mention_count) AS total_mentions, "
                "ROUND(AVG(mention_count),2) AS avg_mentions FROM kg_entities GROUP BY type "
                f"HAVING SUM(mention_count)>={minimum} ORDER BY total_mentions DESC, type{suffix}"
            )
    return None


def _compile_temporal(q: str, domains: Domains) -> Optional[str]:
    suffix = _suffix(q)
    if all(token in q for token in ("按自然月", "就诊量", "费用总额", "时间序列")):
        return _safe(
            "SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visit_count, "
            "ROUND(SUM(cost),2) AS total_cost FROM visits GROUP BY month "
            f"ORDER BY month{suffix}"
        )
    if all(token in q for token in ("首次就诊日期", "全库起始日", "天数")):
        return _safe(
            "SELECT patient_id, MIN(visit_date) AS first_visit, "
            "CAST(julianday(MIN(visit_date))-julianday((SELECT MIN(visit_date) FROM visits)) "
            "AS INTEGER) AS days_from_start FROM visits GROUP BY patient_id "
            f"ORDER BY first_visit, patient_id{suffix}"
        )
    if all(token in q for token in ("末次就诊日期", "就诊次数", "首末次间隔天数")):
        return _safe(
            "SELECT patient_id, MAX(visit_date) AS last_visit, COUNT(*) AS visit_count, "
            "CAST(julianday(MAX(visit_date))-julianday(MIN(visit_date)) AS INTEGER) "
            f"AS active_span_days FROM visits GROUP BY patient_id ORDER BY last_visit DESC, patient_id{suffix}"
        )
    if all(token in q for token in ("按年份统计", "最早", "最晚", "间隔天数")):
        department = _value(q, domains, "department")
        if department:
            return _safe(
                "SELECT substr(visit_date,1,4) AS year, MIN(visit_date) AS first_date, "
                "MAX(visit_date) AS last_date, "
                "CAST(julianday(MAX(visit_date))-julianday(MIN(visit_date)) AS INTEGER) "
                f"AS span_days FROM visits WHERE department='{_esc(department)}' "
                f"GROUP BY year ORDER BY year{suffix}"
            )
    if all(token in q for token in ("按年份和季度", "就诊次数", "去重患者数")):
        return _safe(
            "SELECT substr(visit_date,1,4) AS year, "
            "((CAST(substr(visit_date,6,2) AS INTEGER)-1)/3)+1 AS quarter, "
            "COUNT(*) AS visit_count, COUNT(DISTINCT patient_id) AS patient_count "
            f"FROM visits GROUP BY year, quarter ORDER BY year, quarter{suffix}"
        )
    if all(token in q for token in ("逐月统计不同患者数", "住院总天数")):
        year = _year(q)
        if year is not None:
            return _safe(
                "SELECT substr(visit_date,1,7) AS month, "
                "COUNT(DISTINCT patient_id) AS patient_count, SUM(los_days) AS total_los "
                f"FROM visits WHERE substr(visit_date,1,4)='{year}' "
                f"GROUP BY month ORDER BY month{suffix}"
            )
    if all(token in q for token in ("至少两次就诊", "首次", "末次", "随访跨度")):
        return _safe(
            "SELECT patient_id, MIN(visit_date) AS first_visit, MAX(visit_date) AS last_visit, "
            "CAST(julianday(MAX(visit_date))-julianday(MIN(visit_date)) AS INTEGER) "
            f"AS span_days FROM visits GROUP BY patient_id HAVING COUNT(*)>=2 ORDER BY span_days DESC, patient_id{suffix}"
        )
    if all(token in q for token in ("逐年比较", "就诊次数", "住院就诊次数", "平均费用")):
        return _safe(
            "SELECT substr(visit_date,1,4) AS year, COUNT(*) AS visit_count, "
            "SUM(CASE WHEN los_days>0 THEN 1 ELSE 0 END) AS inpatient_count, "
            "ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY year "
            f"ORDER BY year{suffix}"
        )
    return None


def _compile_kg(q: str, domains: Domains) -> Optional[str]:
    suffix = _suffix(q)
    relation = _relation(q, domains)
    if all(token in q for token in ("三元组头实体", "尾实体", "置信度", "观察次数")) and relation:
        return _safe(
            "SELECT head, tail, confidence, count AS support_count FROM kg_triples "
            f"WHERE relation='{_esc(relation)}' ORDER BY confidence DESC, head, tail{suffix}"
        )
    if all(token in q for token in ("每种疾病", "不同症状", "症状更丰富")):
        return _safe(
            "SELECT head AS disease, COUNT(DISTINCT tail) AS symptom_count FROM kg_triples "
            "WHERE relation='HAS_SYMPTOM' GROUP BY head "
            f"ORDER BY symptom_count DESC, disease{suffix}"
        )
    if all(token in q for token in ("每种疾病", "药物治疗选项数", "平均关系置信度")):
        return _safe(
            "SELECT head AS disease, COUNT(DISTINCT tail) AS drug_count, "
            "ROUND(AVG(confidence),4) AS avg_confidence FROM kg_triples "
            "WHERE relation='TREATED_BY_DRUG' GROUP BY head "
            f"ORDER BY drug_count DESC, disease{suffix}"
        )
    if all(token in q for token in ("哪些检查辅助诊断", "置信度", "观察次数")):
        disease = _value(q, domains, "disease") or _value(q, domains, "kg_entity")
        if disease:
            return _safe(
                "SELECT tail AS exam, confidence, count AS support_count FROM kg_triples "
                f"WHERE head='{_esc(disease)}' AND relation='DIAGNOSED_BY' "
                f"ORDER BY confidence DESC, exam{suffix}"
            )
    if all(token in q for token in ("类型实体", "数量", "提及总量", "最小值", "最大值")):
        entity_type = _value(q, domains, "kg_entity_type")
        if entity_type:
            return _safe(
                "SELECT COUNT(*) AS entity_count, SUM(mention_count) AS total_mentions, "
                "MIN(mention_count) AS min_mentions, MAX(mention_count) AS max_mentions "
                f"FROM kg_entities WHERE type='{_esc(entity_type)}'{suffix}"
            )
    if all(token in q for token in ("按药物", "禁忌疾病数量", "最低置信度")):
        return _safe(
            "SELECT head AS drug, COUNT(DISTINCT tail) AS contraindication_count, "
            "MIN(confidence) AS min_confidence FROM kg_triples "
            "WHERE relation='CONTRAINDICATED_FOR' GROUP BY head "
            f"ORDER BY contraindication_count DESC, drug{suffix}"
        )
    if all(token in q for token in ("观察次数至少", "置信度至少", "图谱关系")):
        count = _integer_after(q, r"观察次数至少(?:为)?(\d+)")
        confidence = _decimal_after(q, r"置信度至少(?:为)?([0-9]+(?:\.[0-9]+)?)")
        if count is not None and confidence is not None:
            return _safe(
                "SELECT head, relation, tail, count AS support_count, confidence FROM kg_triples "
                f"WHERE count>={count} AND confidence>={confidence} "
                f"ORDER BY support_count DESC, confidence DESC, head, tail{suffix}"
            )
    return None


def _compile_joins(q: str, domains: Domains) -> Optional[str]:
    suffix = _suffix(q)
    if all(token in q for token in ("就诊疾病", "知识图谱实体", "提及次数", "费用")):
        department = _value(q, domains, "department")
        if department:
            return _safe(
                "SELECT v.visit_id, v.disease, e.mention_count, v.cost FROM visits v "
                "JOIN kg_entities e ON e.name=v.disease AND e.type='Disease' "
                f"WHERE v.department='{_esc(department)}' ORDER BY v.cost DESC, v.visit_id{suffix}"
            )
    if all(token in q for token in ("诊断为", "药物治疗关系", "处方药", "图谱药物")):
        disease = _value(q, domains, "disease")
        if disease:
            return _safe(
                "SELECT v.visit_id, v.drug_prescribed, t.tail AS graph_drug FROM visits v "
                "JOIN kg_triples t ON t.head=v.disease AND t.relation='TREATED_BY_DRUG' "
                f"WHERE v.disease='{_esc(disease)}' ORDER BY v.visit_id, graph_drug{suffix}"
            )
    if all(token in q for token in ("连接图谱头实体与实体表", "三元组数", "平均提及次数")):
        return _safe(
            "SELECT e.type, COUNT(*) AS triple_count, ROUND(AVG(e.mention_count),2) AS avg_head_mentions "
            "FROM kg_entities e JOIN kg_triples t ON t.head=e.name AND t.head_type=e.type "
            f"GROUP BY e.type ORDER BY triple_count DESC, e.type{suffix}"
        )
    if all(token in q for token in ("连接图谱尾实体与实体表", "关系条数", "提及次数总和")):
        return _safe(
            "SELECT e.type, COUNT(*) AS triple_count, SUM(e.mention_count) AS tail_mentions "
            "FROM kg_entities e JOIN kg_triples t ON t.tail=e.name AND t.tail_type=e.type "
            f"GROUP BY e.type ORDER BY triple_count DESC, e.type{suffix}"
        )
    if all(token in q for token in ("实际就诊疾病", "不同关系类型数量")):
        return _safe(
            "SELECT v.disease, COUNT(DISTINCT t.relation) AS relation_count, "
            "COUNT(DISTINCT v.visit_id) AS visit_count FROM visits v "
            "JOIN kg_triples t ON t.head=v.disease GROUP BY v.disease "
            f"ORDER BY relation_count DESC, v.disease{suffix}"
        )
    if all(token in q for token in ("同一患者", "不同就诊对", "日期间隔", "时间正向")):
        return _safe(
            "SELECT a.patient_id, a.visit_id AS earlier_visit, b.visit_id AS later_visit, "
            "CAST(julianday(b.visit_date)-julianday(a.visit_date) AS INTEGER) AS gap_days "
            "FROM visits a JOIN visits b ON b.patient_id=a.patient_id AND b.visit_date>a.visit_date "
            f"ORDER BY a.patient_id, a.visit_date, b.visit_date, a.visit_id, b.visit_id{suffix}"
        )
    if all(token in q for token in ("各地区患者", "就诊次数", "患者数", "平均费用")):
        return _safe(
            "SELECT p.region, COUNT(v.visit_id) AS visit_count, "
            "COUNT(DISTINCT p.patient_id) AS patient_count, ROUND(AVG(v.cost),2) AS avg_cost "
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region "
            f"ORDER BY visit_count DESC, p.region{suffix}"
        )
    if all(token in q for token in ("疾病连接到图谱诊断检查关系", "实际检查", "图谱建议检查")):
        return _safe(
            "SELECT v.visit_id, v.disease, v.exam AS recorded_exam, t.tail AS graph_exam "
            "FROM visits v JOIN kg_triples t ON t.head=v.disease AND t.relation='DIAGNOSED_BY' "
            f"ORDER BY v.visit_id, graph_exam{suffix}"
        )
    if all(token in q for token in ("不同就诊患者的地区", "就诊费用")) or all(
        token in q for token in ("不同就诊患者", "地区", "汇总", "就诊费用")
    ):
        minimum = _integer_after(q, r"不少于(\d+)位不同就诊患者")
        if minimum is not None:
            return _safe(
                "SELECT p.region, COUNT(DISTINCT p.patient_id) AS patient_count, "
                "ROUND(SUM(v.cost),2) AS total_cost FROM patients p JOIN visits v "
                "ON v.patient_id=p.patient_id GROUP BY p.region "
                f"HAVING COUNT(DISTINCT p.patient_id)>={minimum} "
                f"ORDER BY patient_count DESC, p.region{suffix}"
            )
    return None


def _compile_set_window_nested(q: str, domains: Domains) -> Optional[str]:
    suffix = _suffix(q)
    if all(token in q for token in ("就诊表疾病", "图谱疾病实体", "去重", "统一来源集合")):
        return _safe(
            "SELECT disease AS name FROM visits UNION SELECT name FROM kg_entities "
            f"WHERE type='Disease' ORDER BY name{suffix}"
        )
    if all(token in q for token in ("实际处方药", "图谱治疗药物", "交集")):
        return _safe(
            "SELECT drug_prescribed AS drug FROM visits WHERE LENGTH(drug_prescribed) "
            "INTERSECT SELECT tail FROM kg_triples WHERE relation='TREATED_BY_DRUG' "
            f"ORDER BY drug{suffix}"
        )
    if all(token in q for token in ("实际检查", "图谱诊断检查", "来源标签")):
        return _safe(
            "SELECT exam AS item, 'visit' AS source FROM visits WHERE LENGTH(exam) "
            "UNION SELECT tail AS item, 'graph' AS source FROM kg_triples "
            f"WHERE relation='DIAGNOSED_BY' ORDER BY item, source{suffix}"
        )
    if all(token in q for token in ("患者所在地区", "就诊科室", "地点清单", "类别标签")):
        return _safe(
            "SELECT region AS place, 'region' AS place_type FROM patients UNION "
            "SELECT department AS place, 'department' AS place_type FROM visits "
            f"ORDER BY place_type, place{suffix}"
        )
    if all(token in q for token in ("门诊与住院疾病清单", "就诊类型标签")):
        return _safe(
            "SELECT disease, 'outpatient' AS visit_type FROM visits WHERE los_days=0 UNION "
            "SELECT disease, 'inpatient' AS visit_type FROM visits WHERE los_days>0 "
            f"ORDER BY disease, visit_type{suffix}"
        )
    if all(token in q for token in ("实际诊断疾病", "归属于该科室疾病", "交集")):
        department = _value(q, domains, "department")
        if department:
            escaped = _esc(department)
            return _safe(
                f"SELECT disease FROM visits WHERE department='{escaped}' INTERSECT "
                "SELECT head FROM kg_triples WHERE relation='BELONGS_TO_DEPT' "
                f"AND tail='{escaped}' ORDER BY disease{suffix}"
            )
    if all(token in q for token in ("患者的疾病集合", "其他地区疾病集合", "交集")):
        region = _value(q, domains, "region")
        if region:
            escaped = _esc(region)
            return _safe(
                "SELECT v.disease FROM visits v JOIN patients p ON p.patient_id=v.patient_id "
                f"WHERE p.region='{escaped}' INTERSECT SELECT v.disease FROM visits v "
                "JOIN patients p ON p.patient_id=v.patient_id "
                f"WHERE p.region<>'{escaped}' ORDER BY disease{suffix}"
            )
    if all(token in q for token in ("图谱头实体", "尾实体", "去重合并", "实体名称清单")):
        return _safe(
            "SELECT head AS entity_name FROM kg_triples UNION SELECT tail FROM kg_triples "
            f"ORDER BY entity_name{suffix}"
        )
    if all(token in q for token in ("图谱实体存在", "从未充当", "三元组头实体", "名称与类型")):
        return _safe(
            "SELECT name, type FROM kg_entities EXCEPT SELECT head, head_type FROM kg_triples "
            f"ORDER BY type, name{suffix}"
        )
    if all(token in q for token in ("症状实体名称", "排除", "三元组头实体")):
        return _safe(
            "SELECT name FROM kg_entities WHERE type='Symptom' EXCEPT "
            f"SELECT head FROM kg_triples ORDER BY name{suffix}"
        )

    if all(token in q for token in ("每个科室内部", "费用", "编号", "最高的前三次")):
        return _safe(
            "WITH ranked AS (SELECT department, visit_id, cost, ROW_NUMBER() OVER "
            "(PARTITION BY department ORDER BY cost DESC, visit_id) AS rn FROM visits) "
            "SELECT department, visit_id, cost, rn FROM ranked WHERE rn<=3 "
            f"ORDER BY department, rn{suffix}"
        )
    if all(token in q for token in ("实体按类型分组", "提及次数排名", "最高的两个实体")):
        return _safe(
            "WITH ranked AS (SELECT type, name, mention_count, ROW_NUMBER() OVER "
            "(PARTITION BY type ORDER BY mention_count DESC, name) AS rn FROM kg_entities) "
            "SELECT type, name, mention_count, rn FROM ranked WHERE rn<=2 "
            f"ORDER BY type, rn{suffix}"
        )
    if all(token in q for token in ("每种图谱关系", "置信度从高到低排名", "每类前三条")):
        return _safe(
            "WITH ranked AS (SELECT relation, head, tail, confidence, ROW_NUMBER() OVER "
            "(PARTITION BY relation ORDER BY confidence DESC, head, tail) AS rn FROM kg_triples) "
            "SELECT relation, head, tail, confidence, rn FROM ranked WHERE rn<=3 "
            f"ORDER BY relation, rn{suffix}"
        )
    if all(token in q for token in ("按日期", "截至当次", "累计费用")):
        department = _value(q, domains, "department")
        if department:
            return _safe(
                "SELECT visit_date, visit_id, cost, ROUND(SUM(cost) OVER "
                "(ORDER BY visit_date, visit_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),2) "
                f"AS running_cost FROM visits WHERE department='{_esc(department)}' "
                f"ORDER BY visit_date, visit_id{suffix}"
            )
    if all(token in q for token in ("患者总费用", "划分为四组", "所属分组")):
        return _safe(
            "WITH totals AS (SELECT patient_id, SUM(cost) AS total_cost FROM visits GROUP BY patient_id) "
            "SELECT patient_id, ROUND(total_cost,2) AS total_cost, NTILE(4) OVER "
            "(ORDER BY total_cost DESC, patient_id) AS cost_quartile FROM totals "
            f"ORDER BY cost_quartile, total_cost DESC, patient_id{suffix}"
        )
    if all(token in q for token in ("按科室日期顺序", "第一条就诊", "对照基线")):
        return _safe(
            "SELECT department, visit_id, visit_date, FIRST_VALUE(visit_date) OVER "
            "(PARTITION BY department ORDER BY visit_date, visit_id ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND UNBOUNDED FOLLOWING) AS first_dept_date FROM visits "
            f"ORDER BY department, visit_date, visit_id{suffix}"
        )

    if all(token in q for token in ("费用高于其所在科室平均费用", "科室", "就诊编号")):
        return _safe(
            "SELECT v.department, v.visit_id, v.cost FROM visits v WHERE v.cost>"
            "(SELECT AVG(x.cost) FROM visits x WHERE x.department=v.department) "
            f"ORDER BY v.department, v.cost DESC, v.visit_id{suffix}"
        )
    if all(token in q for token in ("年龄高于同地区患者平均年龄", "地区均值")):
        return _safe(
            "SELECT p.patient_id, p.region, p.age, ROUND((SELECT AVG(x.age) FROM patients x "
            "WHERE x.region=p.region),2) AS region_avg_age FROM patients p WHERE p.age>"
            "(SELECT AVG(x.age) FROM patients x WHERE x.region=p.region) "
            f"ORDER BY p.region, p.age DESC, p.patient_id{suffix}"
        )
    if all(token in q for token in ("总就诊费用", "所有患者平均总费用", "个人总费用")):
        return _safe(
            "SELECT v.patient_id, ROUND(SUM(v.cost),2) AS patient_total FROM visits v "
            "GROUP BY v.patient_id HAVING SUM(v.cost)>(SELECT AVG(t.patient_total) FROM "
            "(SELECT SUM(cost) AS patient_total FROM visits GROUP BY patient_id) t) "
            f"ORDER BY patient_total DESC, v.patient_id{suffix}"
        )
    if all(token in q for token in ("就诊表中出现", "知识图谱实体表", "同名疾病实体")):
        return _safe(
            "SELECT DISTINCT v.disease FROM visits v WHERE EXISTS (SELECT 1 FROM kg_entities e "
            "WHERE e.name=v.disease AND e.type='Disease') "
            f"ORDER BY v.disease{suffix}"
        )
    if all(token in q for token in ("患者中", "费用超过该患者自身平均费用")):
        region = _value(q, domains, "region")
        if region:
            return _safe(
                "SELECT p.patient_id, v.visit_id, v.cost FROM patients p JOIN visits v "
                f"ON v.patient_id=p.patient_id WHERE p.region='{_esc(region)}' AND v.cost>"
                "(SELECT AVG(x.cost) FROM visits x WHERE x.patient_id=v.patient_id) "
                f"ORDER BY p.patient_id, v.cost DESC, v.visit_id{suffix}"
            )
    if all(token in q for token in ("相同头实体平均观察次数", "三元组")) and _relation(q, domains):
        relation = _esc(_relation(q, domains) or "")
        return _safe(
            "SELECT t.head, t.tail, t.count AS support_count FROM kg_triples t "
            f"WHERE t.relation='{relation}' AND t.count>(SELECT AVG(x.count) FROM kg_triples x "
            f"WHERE x.head=t.head) ORDER BY support_count DESC, t.head, t.tail{suffix}"
        )
    if all(token in q for token in ("每个图谱关系", "置信度高于", "自身平均值")):
        return _safe(
            "SELECT t.relation, t.head, t.tail, t.confidence FROM kg_triples t JOIN "
            "(SELECT relation, AVG(confidence) AS avg_confidence FROM kg_triples GROUP BY relation) a "
            "ON a.relation=t.relation WHERE t.confidence>a.avg_confidence "
            f"ORDER BY t.relation, t.confidence DESC, t.head, t.tail{suffix}"
        )
    if all(token in q for token in ("同类型实体平均提及次数", "知识图谱实体")):
        return _safe(
            "SELECT e.type, e.name, e.mention_count FROM kg_entities e WHERE e.mention_count>"
            "(SELECT AVG(x.mention_count) FROM kg_entities x WHERE x.type=e.type) "
            f"ORDER BY e.type, e.mention_count DESC, e.name{suffix}"
        )
    if all(token in q for token in ("疾病关系数", "平均关系数", "图谱疾病")):
        return _safe(
            "SELECT t.head AS disease, COUNT(*) AS relation_count FROM kg_triples t "
            "WHERE t.head_type='Disease' GROUP BY t.head HAVING COUNT(*)>(SELECT AVG(x.cnt) FROM "
            "(SELECT COUNT(*) AS cnt FROM kg_triples WHERE head_type='Disease' GROUP BY head) x) "
            f"ORDER BY relation_count DESC, disease{suffix}"
        )

    if all(token in q for token in ("按月计算费用总额", "累计费用")):
        return _safe(
            "WITH monthly AS (SELECT substr(visit_date,1,7) AS month, SUM(cost) AS monthly_cost "
            "FROM visits GROUP BY month) SELECT month, ROUND(monthly_cost,2) AS monthly_cost, "
            "ROUND(SUM(monthly_cost) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND CURRENT ROW),2) AS running_cost FROM monthly "
            f"ORDER BY month{suffix}"
        )
    if all(token in q for token in ("疾病按就诊次数", "稠密排名", "共享名次")):
        return _safe(
            "WITH disease_counts AS (SELECT disease, COUNT(*) AS visit_count FROM visits GROUP BY disease) "
            "SELECT disease, visit_count, DENSE_RANK() OVER (ORDER BY visit_count DESC) AS visit_rank "
            f"FROM disease_counts ORDER BY visit_rank, disease{suffix}"
        )
    if all(token in q for token in ("按月统计就诊量", "上月就诊量", "环比数量变化")):
        year = _year(q)
        if year is not None:
            return _safe(
                "WITH monthly AS (SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visit_count "
                f"FROM visits WHERE substr(visit_date,1,4)='{year}' GROUP BY month), "
                "lagged AS (SELECT month, visit_count, LAG(visit_count) OVER (ORDER BY month) AS prev_count "
                "FROM monthly) SELECT month, visit_count, prev_count, visit_count-prev_count AS count_change "
                f"FROM lagged ORDER BY month{suffix}"
            )
    return None


def _compile_cte(q: str) -> Optional[str]:
    suffix = _suffix(q)
    if all(token in q for token in ("每位患者首末次就诊日期", "随访跨度大于零")):
        return _safe(
            "WITH span AS (SELECT patient_id, MIN(visit_date) AS first_visit, "
            "MAX(visit_date) AS last_visit FROM visits GROUP BY patient_id) "
            "SELECT patient_id, first_visit, last_visit, "
            "CAST(julianday(last_visit)-julianday(first_visit) AS INTEGER) AS span_days "
            f"FROM span WHERE last_visit>first_visit ORDER BY span_days DESC, patient_id{suffix}"
        )
    if all(token in q for token in ("每个疾病", "图谱关系数", "就诊数", "两边都存在")):
        return _safe(
            "WITH graph_stats AS (SELECT head AS disease, COUNT(*) AS graph_relations FROM kg_triples "
            "WHERE head_type='Disease' GROUP BY head), visit_stats AS (SELECT disease, COUNT(*) AS visit_count "
            "FROM visits GROUP BY disease) SELECT v.disease, v.visit_count, g.graph_relations "
            "FROM visit_stats v JOIN graph_stats g ON g.disease=v.disease "
            f"ORDER BY v.visit_count DESC, v.disease{suffix}"
        )
    if all(token in q for token in ("汇总各科室费用", "占全库费用的比例")):
        return _safe(
            "WITH dept_cost AS (SELECT department, SUM(cost) AS total_cost FROM visits GROUP BY department), "
            "grand AS (SELECT SUM(total_cost) AS all_cost FROM dept_cost) SELECT d.department, "
            "ROUND(d.total_cost,2) AS total_cost, ROUND(d.total_cost*100.0/g.all_cost,2) AS cost_pct "
            f"FROM dept_cost d CROSS JOIN grand g ORDER BY cost_pct DESC, d.department{suffix}"
        )
    if all(token in q for token in ("每类图谱关系", "三元组数高于关系类别平均值", "平均置信度")):
        return _safe(
            "WITH rel_stats AS (SELECT relation, COUNT(*) AS triple_count, AVG(confidence) AS avg_conf "
            "FROM kg_triples GROUP BY relation), baseline AS (SELECT AVG(triple_count) AS avg_count "
            "FROM rel_stats) SELECT r.relation, r.triple_count, ROUND(r.avg_conf,4) AS avg_confidence "
            "FROM rel_stats r CROSS JOIN baseline b WHERE r.triple_count>b.avg_count "
            f"ORDER BY r.triple_count DESC, r.relation{suffix}"
        )
    if all(token in q for token in ("地区患者数", "地区就诊量")) and (
        "全部地区指标" in q or "每位患者" in q
    ):
        return _safe(
            "WITH patient_region AS (SELECT region, COUNT(*) AS patient_count FROM patients GROUP BY region), "
            "visit_region AS (SELECT p.region, COUNT(v.visit_id) AS visit_count FROM patients p "
            "JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region) SELECT pr.region, pr.patient_count, "
            "vr.visit_count, ROUND(vr.visit_count*1.0/pr.patient_count,2) AS visits_per_patient "
            "FROM patient_region pr JOIN visit_region vr ON vr.region=pr.region "
            f"ORDER BY visits_per_patient DESC, pr.region{suffix}"
        )
    if all(token in q for token in ("实际处方次数", "图谱治疗关系数", "同名药物")):
        return _safe(
            "WITH prescribed AS (SELECT drug_prescribed AS drug, COUNT(*) AS prescribed_count FROM visits "
            "WHERE LENGTH(drug_prescribed) GROUP BY drug_prescribed), graph_drugs AS "
            "(SELECT tail AS drug, COUNT(*) AS graph_count FROM kg_triples WHERE relation='TREATED_BY_DRUG' "
            "GROUP BY tail) SELECT p.drug, p.prescribed_count, g.graph_count FROM prescribed p "
            f"JOIN graph_drugs g ON g.drug=p.drug ORDER BY p.prescribed_count DESC, p.drug{suffix}"
        )
    if all(token in q for token in ("科室住院与门诊费用", "住院费用占", "百分比")):
        return _safe(
            "WITH split AS (SELECT department, SUM(CASE WHEN los_days>0 THEN cost ELSE 0 END) "
            "AS inpatient_cost, SUM(CASE WHEN los_days=0 THEN cost ELSE 0 END) AS outpatient_cost "
            "FROM visits GROUP BY department) SELECT department, ROUND(inpatient_cost,2) AS inpatient_cost, "
            "ROUND(outpatient_cost,2) AS outpatient_cost, ROUND(inpatient_cost*100.0/"
            "NULLIF(inpatient_cost+outpatient_cost,0),2) AS inpatient_pct FROM split "
            f"ORDER BY inpatient_pct DESC, department{suffix}"
        )
    if all(token in q for token in ("公用表表达式", "患者总费用", "高于患者总费用平均值")):
        return _safe(
            "WITH patient_cost AS (SELECT patient_id, SUM(cost) AS total_cost FROM visits GROUP BY patient_id), "
            "benchmark AS (SELECT AVG(total_cost) AS avg_total FROM patient_cost) SELECT p.patient_id, p.region, "
            "ROUND(pc.total_cost,2) AS total_cost, ROUND(b.avg_total,2) AS avg_total FROM patient_cost pc "
            "JOIN patients p ON p.patient_id=pc.patient_id CROSS JOIN benchmark b WHERE pc.total_cost>b.avg_total "
            f"ORDER BY total_cost DESC, p.patient_id{suffix}"
        )
    return None


def compile_independent_v7_sql(q: str, domains: Domains) -> Optional[str]:
    """Compile one fully owned natural analytical request or abstain."""

    if not isinstance(q, str) or not q.strip() or len(q) > 1200:
        return None
    question = q.strip()
    for compiler in (
        lambda: _compile_cte(question),
        lambda: _compile_set_window_nested(question, domains),
        lambda: _compile_joins(question, domains),
        lambda: _compile_temporal(question, domains),
        lambda: _compile_top_having(question, domains),
        lambda: _compile_grouping(question, domains),
        lambda: _compile_kg(question, domains),
        lambda: _compile_filtering(question, domains),
    ):
        candidate = compiler()
        if candidate is not None:
            return candidate
    return None
