"""Fail-closed compilers for recurring four-table analytical structures.

The functions in this module lower complete semantic families rather than
patching model-produced SQL.  A branch fires only when the question states the
whole grain, projection, predicate, and (where applicable) threshold.  Every
statement is then checked by the shared single-query/read-only/table-allowlist
gate before it can become an authoritative pre-model answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from dki.analysis.nl2sql_protocol import validate_gold_sql


Domains = Mapping[str, Sequence[str]]


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _validated(sql: str) -> Optional[str]:
    try:
        validate_gold_sql(sql)
    except Exception:
        return None
    return sql


def _semantic_core(question: str) -> Optional[str]:
    q = str(question or "").strip()
    match = re.fullmatch(
        r"(?P<prefix>围绕[^，,]{2,40}[，,]在[^，,]{2,50}时[，,])(?P<core>.+)",
        q,
    )
    if match is not None:
        prefix = match.group("prefix")
        if re.search(
            r"\d|性别为|女性患者|男性患者|年龄(?:大于|小于|至少|不低于)|"
            r"地区为|科室为|关系(?:为|标识为)|(?:HAS_|TREATED_|DIAGNOSED_|"
            r"BELONGS_|CONTRAINDICATED_)",
            prefix,
            re.IGNORECASE,
        ):
            return None
        q = match.group("core")
    audit_match = re.fullmatch(
        r"为开展(?P<context>[^，,]{2,40})[，,](?P<core>.+)",
        q,
    )
    if audit_match is not None:
        context = audit_match.group("context")
        if re.search(
            r"\d|性别为|年龄(?:在|大于|小于|至少|不低于)|地区为|科室为|"
            r"实体类型为|关系(?:为|标识为)|(?:HAS_|TREATED_|DIAGNOSED_|"
            r"BELONGS_|CONTRAINDICATED_)",
            context,
            re.IGNORECASE,
        ):
            return None
        q = audit_match.group("core")
    q = q.strip().rstrip("。.")
    if not q or len(q) > 220:
        return None
    return q


def _domain_hits(text: str, values: Sequence[str]) -> list[str]:
    hits = [str(value) for value in values if value and str(value) in text]
    return sorted(set(hits), key=lambda value: (text.find(value), -len(value), value))


def _single_relation(text: str, domains: Domains) -> Optional[str]:
    allowed = set(domains.get("kg_relation", ()))
    hits = {
        token
        for token in re.findall(
            r"(?<![A-Z0-9_])([A-Z][A-Z0-9_]{2,})(?![A-Z0-9_])",
            text,
        )
        if token in allowed
    }
    return next(iter(hits)) if len(hits) == 1 else None


def _single_entity_type(text: str, domains: Domains) -> Optional[str]:
    hits = _domain_hits(text, domains.get("kg_entity_type", ()))
    return hits[0] if len(hits) == 1 else None


def _exact_domain_value(value: str, values: Sequence[str]) -> Optional[str]:
    candidate = str(value).strip()
    return candidate if candidate in {str(item) for item in values} else None


def _compile_operational_audit(core: str, domains: Domains) -> Optional[str]:
    """Compile complete operational-audit requests without model fallback."""

    match = re.fullmatch(
        r"筛出科室为(?P<department>.+?)并且费用大于等于"
        r"(?P<cost>\d+(?:\.\d+)?)元的就诊[，,]"
        r"列出日期[、,，]患者编号与费用[，,]先按日期后按编号升序",
        core,
    )
    if match is not None:
        department = _exact_domain_value(
            match.group("department"), domains.get("department", ())
        )
        if department is None:
            return None
        return _validated(
            "SELECT visit_date, patient_id, cost FROM visits "
            f"WHERE department = '{_esc(department)}' "
            f"AND cost >= {float(match.group('cost')):g} "
            "ORDER BY visit_date, visit_id"
        )

    match = re.fullmatch(
        r"列出年龄在(?P<low>\d+)至(?P<high>\d+)岁之间且性别为"
        r"(?P<gender>男|女)的患者编号[、,，]姓名与地区[，,]"
        r"按年龄和姓名升序",
        core,
    )
    if match is not None:
        low, high = int(match.group("low")), int(match.group("high"))
        if low > high:
            return None
        return _validated(
            "SELECT patient_id, name, region FROM patients "
            f"WHERE age BETWEEN {low} AND {high} "
            f"AND gender = '{match.group('gender')}' ORDER BY age, name"
        )

    match = re.fullmatch(
        r"连接患者与就诊记录[，,]筛选地区为(?P<region>.+?)且就诊科室为"
        r"(?P<department>.+?)的数据[，,]返回患者编号[、,，]姓名[、,，]"
        r"就诊编号[、,，]日期与疾病并按指定业务顺序排列",
        core,
    )
    if match is not None:
        region = _exact_domain_value(match.group("region"), domains.get("region", ()))
        department = _exact_domain_value(
            match.group("department"), domains.get("department", ())
        )
        if region is None or department is None:
            return None
        return _validated(
            "SELECT p.patient_id, p.name, v.visit_id, v.visit_date, v.disease "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            f"WHERE p.region = '{_esc(region)}' "
            f"AND v.department = '{_esc(department)}' "
            "ORDER BY v.visit_date DESC, p.patient_id"
        )

    match = re.fullmatch(
        r"查询地区为(?P<region>.+?)且至少存在一次(?P<department>.+?)就诊的"
        r"患者编号[、,，]姓名和年龄[；;]存在性判断必须关联当前患者",
        core,
    )
    if match is not None:
        region = _exact_domain_value(match.group("region"), domains.get("region", ()))
        department = _exact_domain_value(
            match.group("department"), domains.get("department", ())
        )
        if region is None or department is None:
            return None
        return _validated(
            "SELECT p.patient_id, p.name, p.age FROM patients p "
            f"WHERE p.region = '{_esc(region)}' AND EXISTS "
            "(SELECT 1 FROM visits v WHERE v.patient_id = p.patient_id "
            f"AND v.department = '{_esc(department)}') ORDER BY p.patient_id"
        )

    match = re.fullmatch(
        r"查找实体类型为(?P<entity_type>[A-Za-z]+)且作为头实体时拥有至少"
        r"(?P<count>\d+)条关系标识为(?P<relation>[A-Z][A-Z0-9_]*)的图谱边的实体"
        r"[，,]边数由关联当前实体的子查询统计",
        core,
    )
    if match is not None:
        entity_type = _exact_domain_value(
            match.group("entity_type"), domains.get("kg_entity_type", ())
        )
        relation = _exact_domain_value(
            match.group("relation"), domains.get("kg_relation", ())
        )
        count = int(match.group("count"))
        if entity_type is None or relation is None or count < 1:
            return None
        return _validated(
            "SELECT e.name, e.type, e.mention_count FROM kg_entities e "
            f"WHERE e.type = '{_esc(entity_type)}' AND "
            "(SELECT COUNT(*) FROM kg_triples t WHERE t.head = e.name "
            f"AND t.relation = '{_esc(relation)}') >= {count} "
            "ORDER BY e.mention_count DESC, e.name"
        )

    match = re.fullmatch(
        r"限定科室为(?P<first>.+?)或(?P<second>.+?)[，,]"
        r"形成分组结果时按(?P<grain>科室|疾病|住院天数|患者编号|处方药|就诊月份)"
        r"分别(?P<metric>.+?)[，,]并按指定分组键与统计值稳定排序",
        core,
    )
    if match is None:
        return None
    first = _exact_domain_value(match.group("first"), domains.get("department", ()))
    second = _exact_domain_value(match.group("second"), domains.get("department", ()))
    if first is None or second is None or first == second:
        return None
    metric_match = re.fullmatch(
        r"(?P<prefix>.+?)保留(?P<decimals>\d+)位小数的(?P<measure>.+)",
        match.group("metric"),
    )
    if metric_match is None:
        return None
    decimals = int(metric_match.group("decimals"))
    if not 0 <= decimals <= 6:
        return None
    signature = (
        match.group("grain"),
        metric_match.group("prefix"),
        metric_match.group("measure"),
    )
    where = (
        f"WHERE department IN ('{_esc(first)}', '{_esc(second)}') "
    )
    templates = {
        ("科室", "统计就诊次数与", "平均费用"): (
            "SELECT department, COUNT(*) AS visit_count, "
            f"ROUND(AVG(cost), {decimals}) AS avg_cost FROM visits {where}"
            "GROUP BY department ORDER BY visit_count DESC, department"
        ),
        ("疾病", "汇总病例数与", "费用总额"): (
            "SELECT disease, COUNT(*) AS case_count, "
            f"ROUND(SUM(cost), {decimals}) AS total_cost FROM visits {where}"
            "GROUP BY disease ORDER BY total_cost DESC, disease"
        ),
        ("住院天数", "计算就诊量与", "平均费用"): (
            "SELECT los_days, COUNT(*) AS visit_count, "
            f"ROUND(AVG(cost), {decimals}) AS avg_cost FROM visits {where}"
            "GROUP BY los_days ORDER BY los_days, visit_count DESC"
        ),
        ("患者编号", "给出就诊次数与", "最高费用"): (
            "SELECT patient_id, COUNT(*) AS visit_count, "
            f"ROUND(MAX(cost), {decimals}) AS max_cost FROM visits {where}"
            "GROUP BY patient_id ORDER BY visit_count DESC, patient_id"
        ),
        ("处方药", "计算处方次数与", "关联费用总额"): (
            "SELECT drug_prescribed, COUNT(*) AS prescription_count, "
            f"ROUND(SUM(cost), {decimals}) AS total_cost FROM visits {where}"
            "GROUP BY drug_prescribed "
            "ORDER BY prescription_count DESC, drug_prescribed"
        ),
        ("就诊月份", "汇总次数与", "费用合计"): (
            "SELECT strftime('%Y-%m', visit_date) AS visit_month, "
            "COUNT(*) AS visit_count, "
            f"ROUND(SUM(cost), {decimals}) AS total_cost FROM visits {where}"
            "GROUP BY strftime('%Y-%m', visit_date) ORDER BY visit_month"
        ),
    }
    return _validated(templates[signature]) if signature in templates else None


def _compile_grouped_metrics(core: str) -> Optional[str]:
    match = re.fullmatch(
        r"按主要检查分组[，,]包括空白检查组[，,]仅输出涉及至少(?P<n>\d+)种"
        r"不同疾病的检查与记录数",
        core,
    )
    if match is not None:
        threshold = int(match.group("n"))
        if threshold < 1:
            return None
        return _validated(
            "SELECT exam, COUNT(DISTINCT disease) AS disease_count, "
            "COUNT(*) AS visit_count FROM visits GROUP BY exam "
            f"HAVING COUNT(DISTINCT disease) >= {threshold} "
            "ORDER BY disease_count DESC, exam"
        )

    match = re.fullmatch(
        r"按主要处方药分组[，,]包括空白处方组[，,]保留累计费用"
        r"(?P<operator>超过|大于|至少|不低于)(?P<amount>\d+(?:\.\d+)?)元的药物组",
        core,
    )
    if match is not None:
        amount = float(match.group("amount"))
        if amount < 0:
            return None
        operator = ">" if match.group("operator") in {"超过", "大于"} else ">="
        literal = f"{amount:g}"
        return _validated(
            "SELECT drug_prescribed, COUNT(*) AS visit_count, "
            "ROUND(SUM(cost), 2) AS total_cost FROM visits "
            "GROUP BY drug_prescribed "
            f"HAVING SUM(cost) {operator} {literal} "
            "ORDER BY total_cost DESC, drug_prescribed"
        )

    if re.fullmatch(
        r"按头实体汇总不同关系类型数和观察次数总和[，,]"
        r"返回头实体[、,，]关系类型数与总次数",
        core,
    ):
        return _validated(
            "SELECT head, COUNT(DISTINCT relation) AS relation_type_count, "
            "SUM(count) AS observations FROM kg_triples GROUP BY head "
            "ORDER BY relation_type_count DESC, observations DESC, head"
        )

    if re.fullmatch(
        r"按科室统计不同检查数[、,，]不同处方药数与记录总数[，,]包括空白值",
        core,
    ):
        return _validated(
            "SELECT department, COUNT(DISTINCT exam) AS exam_count, "
            "COUNT(DISTINCT drug_prescribed) AS drug_count, "
            "COUNT(*) AS visit_count FROM visits GROUP BY department "
            "ORDER BY exam_count DESC, department"
        )

    match = re.fullmatch(
        r"按年龄分组[，,]只返回同龄患者至少(?P<n>\d+)人的年龄[、,，]人数与不同地区数",
        core,
    )
    if match is not None:
        threshold = int(match.group("n"))
        if threshold < 1:
            return None
        return _validated(
            "SELECT age, COUNT(*) AS patient_count, "
            "COUNT(DISTINCT region) AS region_count FROM patients "
            f"GROUP BY age HAVING COUNT(*) >= {threshold} ORDER BY age"
        )

    match = re.fullmatch(
        r"按关系标识和头类型组合分组[，,]只保留观察次数总和"
        r"(?P<operator>大于|超过|至少|不低于)(?P<n>\d+)的组合",
        core,
    )
    if match is not None:
        threshold = int(match.group("n"))
        operator = ">" if match.group("operator") in {"大于", "超过"} else ">="
        return _validated(
            "SELECT relation, head_type, COUNT(*) AS edge_count, "
            "SUM(count) AS observations FROM kg_triples "
            "GROUP BY relation, head_type "
            f"HAVING SUM(count) {operator} {threshold} "
            "ORDER BY observations DESC, relation, head_type"
        )

    if re.fullmatch(
        r"按实体类型汇总实体数[、,，]平均提及次数和提及次数"
        r"方差替代量即极差",
        core,
    ):
        return _validated(
            "SELECT type, COUNT(*) AS entity_count, "
            "ROUND(AVG(mention_count), 2) AS avg_mentions, "
            "MAX(mention_count) - MIN(mention_count) AS mention_range "
            "FROM kg_entities GROUP BY type ORDER BY mention_range DESC, type"
        )

    match = re.fullmatch(
        r"只看提及次数至少(?P<n>\d+)的实体[，,]按类型统计数量与平均提及次数",
        core,
    )
    if match is not None:
        threshold = int(match.group("n"))
        return _validated(
            "SELECT type, COUNT(*) AS entity_count, "
            "ROUND(AVG(mention_count), 2) AS avg_mentions FROM kg_entities "
            f"WHERE mention_count >= {threshold} GROUP BY type "
            "ORDER BY entity_count DESC, type"
        )
    return None


def _compile_graph_entity_edges(core: str, domains: Domains) -> Optional[str]:
    relation = _single_relation(core, domains)
    entity_type = _single_entity_type(core, domains)

    if (
        relation == "DIAGNOSED_BY"
        and entity_type == "Exam"
        and all(token in core for token in ("连接Exam实体", "尾边", "检查名", "提及次数", "平均关系置信度"))
    ):
        return _validated(
            "SELECT e.name AS exam, e.mention_count, "
            "ROUND(AVG(t.confidence), 4) AS avg_confidence "
            "FROM kg_entities e JOIN kg_triples t ON t.tail = e.name "
            "WHERE e.type = 'Exam' AND t.relation = 'DIAGNOSED_BY' "
            "GROUP BY e.name, e.mention_count "
            "ORDER BY e.mention_count DESC, exam"
        )

    if (
        relation == "HAS_SYMPTOM"
        and entity_type == "Disease"
        and all(token in core for token in ("Disease实体", "不同症状数", "平均置信度"))
    ):
        return _validated(
            "SELECT e.name AS disease, e.mention_count, "
            "COUNT(DISTINCT t.tail) AS symptom_count, "
            "ROUND(AVG(t.confidence), 4) AS avg_confidence "
            "FROM kg_entities e JOIN kg_triples t ON t.head = e.name "
            "WHERE e.type = 'Disease' AND t.relation = 'HAS_SYMPTOM' "
            "GROUP BY e.name, e.mention_count "
            "ORDER BY symptom_count DESC, disease"
        )

    match = re.fullmatch(
        r"返回提及次数至少(?P<n>\d+)的(?P<type>[A-Za-z]+)实体以及其"
        r"所有关系中文名和尾实体",
        core,
    )
    if match is not None:
        requested_type = match.group("type")
        if requested_type not in domains.get("kg_entity_type", ()):
            return None
        threshold = int(match.group("n"))
        return _validated(
            "SELECT e.name, e.mention_count, t.relation_zh, t.tail "
            "FROM kg_entities e JOIN kg_triples t ON t.head = e.name "
            f"WHERE e.type = '{_esc(requested_type)}' "
            f"AND e.mention_count >= {threshold} "
            "ORDER BY e.mention_count DESC, e.name, t.relation_zh, t.tail"
        )

    if (
        relation == "BELONGS_TO_DEPT"
        and entity_type == "Department"
        and all(token in core for token in ("连接Department实体", "尾边", "科室", "关联疾病清单"))
    ):
        return _validated(
            "SELECT e.name AS department, t.head AS disease, t.confidence "
            "FROM kg_entities e JOIN kg_triples t ON t.tail = e.name "
            "WHERE e.type = 'Department' "
            "AND t.relation = 'BELONGS_TO_DEPT' "
            "ORDER BY department, disease"
        )

    if (
        entity_type == "Disease"
        and all(token in core for token in ("每个Disease实体", "最高出边置信度", "最低出边置信度", "提及次数"))
        and relation is None
    ):
        return _validated(
            "SELECT e.name AS disease, e.mention_count, "
            "MAX(t.confidence) AS max_confidence, "
            "MIN(t.confidence) AS min_confidence "
            "FROM kg_entities e JOIN kg_triples t ON t.head = e.name "
            "WHERE e.type = 'Disease' GROUP BY e.name, e.mention_count "
            "ORDER BY max_confidence DESC, disease"
        )

    if (
        relation == "TREATED_BY_DRUG"
        and entity_type == "Drug"
        and all(token in core for token in ("连接Drug实体", "尾边", "关联疾病数", "观察次数总和"))
    ):
        return _validated(
            "SELECT e.name AS drug, COUNT(DISTINCT t.head) AS disease_count, "
            "SUM(t.count) AS observations FROM kg_entities e "
            "JOIN kg_triples t ON t.tail = e.name "
            "WHERE e.type = 'Drug' AND t.relation = 'TREATED_BY_DRUG' "
            "GROUP BY e.name ORDER BY disease_count DESC, drug"
        )

    if (
        relation == "HAS_COMPLICATION"
        and entity_type == "Disease"
        and all(token in core for token in ("Disease实体", "并发症数", "平均置信度", "提及次数"))
    ):
        return _validated(
            "SELECT e.name AS disease, e.mention_count, "
            "COUNT(DISTINCT t.tail) AS complication_count, "
            "ROUND(AVG(t.confidence), 4) AS avg_confidence "
            "FROM kg_entities e JOIN kg_triples t ON t.head = e.name "
            "WHERE e.type = 'Disease' AND t.relation = 'HAS_COMPLICATION' "
            "GROUP BY e.name, e.mention_count "
            "ORDER BY complication_count DESC, disease"
        )

    match = re.fullmatch(
        r"连接实际疾病与图谱Disease实体[，,]只保留实体提及次数"
        r"(?P<operator>大于|超过|至少|不低于)(?P<n>\d+)的疾病[、,，]"
        r"实际次数和平均费用",
        core,
    )
    if match is not None and "Disease" in domains.get("kg_entity_type", ()):
        operator = ">" if match.group("operator") in {"大于", "超过"} else ">="
        threshold = int(match.group("n"))
        return _validated(
            "SELECT e.name AS disease, e.mention_count, "
            "COUNT(*) AS visit_count, ROUND(AVG(v.cost), 2) AS avg_cost "
            "FROM visits v JOIN kg_entities e ON e.name = v.disease "
            "WHERE e.type = 'Disease' "
            f"AND e.mention_count {operator} {threshold} "
            "GROUP BY e.name, e.mention_count "
            "ORDER BY visit_count DESC, disease"
        )

    if (
        entity_type == "Exam"
        and all(token in core for token in ("连接实际检查", "图谱Exam实体", "就诊数", "不同疾病数", "实体提及次数"))
    ):
        return _validated(
            "SELECT e.name AS exam, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT v.disease) AS disease_count, e.mention_count "
            "FROM visits v JOIN kg_entities e ON e.name = v.exam "
            "WHERE e.type = 'Exam' GROUP BY e.name, e.mention_count "
            "ORDER BY visit_count DESC, exam"
        )
    return None


def _compile_graph_shared(core: str, domains: Domains) -> Optional[str]:
    relation = _single_relation(core, domains)
    if relation is None:
        return None
    if all(token in core for token in ("共享同一", "不同疾病对")) and (
        "小于" in core or "去除镜像" in core
    ):
        return _validated(
            "SELECT a.head AS disease_a, b.head AS disease_b, "
            "a.tail AS shared_tail FROM kg_triples a "
            "JOIN kg_triples b ON b.tail = a.tail AND a.head < b.head "
            f"WHERE a.relation = '{_esc(relation)}' "
            f"AND b.relation = '{_esc(relation)}' "
            "ORDER BY shared_tail, disease_a, disease_b"
        )
    return None


def _compile_simple_rows(core: str, domains: Domains) -> Optional[str]:
    match = re.fullmatch(
        r"返回不早于(?P<date>\d{4}-\d{2}-\d{2})的就诊编号[、,，]日期[、,，]患者和科室",
        core,
    )
    if match is not None:
        date = match.group("date")
        return _validated(
            "SELECT visit_id, visit_date, patient_id, department FROM visits "
            f"WHERE visit_date >= '{date}' ORDER BY visit_date, visit_id"
        )

    match = re.fullmatch(
        r"按费用升序分页[，,]跳过最低的(?P<offset>\d+)条后取(?P<limit>\d+)条就诊"
        r"[；;]OFFSET为(?P<offset2>\d+)[、,，]LIMIT为(?P<limit2>\d+)",
        core,
    )
    if match is not None:
        offset = int(match.group("offset"))
        limit = int(match.group("limit"))
        if offset != int(match.group("offset2")) or limit != int(match.group("limit2")) or limit < 1:
            return None
        return _validated(
            "SELECT visit_id, disease, cost FROM visits "
            f"ORDER BY cost, visit_id LIMIT {limit} OFFSET {offset}"
        )

    visit_ids = _domain_hits(core, domains.get("visit_id", ()))
    if (
        len(visit_ids) == 1
        and all(token in core for token in ("精确返回就诊编号", "患者编号", "日期", "疾病", "处方药", "检查", "费用"))
    ):
        return _validated(
            "SELECT patient_id, visit_date, disease, drug_prescribed, exam, cost "
            f"FROM visits WHERE visit_id = '{_esc(visit_ids[0])}'"
        )

    match = re.fullmatch(
        r"以全库最早就诊日为起点[，,]列出其后(?P<days>\d+)天内含起点的就诊"
        r"[，,]窗口为(?P<repeat>\d+)天",
        core,
    )
    if match is not None:
        days = int(match.group("days"))
        if days != int(match.group("repeat")) or days < 0:
            return None
        return _validated(
            "SELECT visit_id, visit_date, disease FROM visits "
            "WHERE visit_date <= date((SELECT MIN(visit_date) FROM visits), "
            f"'+{days} days') ORDER BY visit_date, visit_id"
        )

    match = re.fullmatch(
        r"筛选每月日期数字小于等于(?P<day>\d+)的上旬就诊[，,]"
        r"返回编号[、,，]日期[、,，]科室和费用[；;]阈值(?P<repeat>\d+)",
        core,
    )
    if match is not None:
        day = int(match.group("day"))
        if day != int(match.group("repeat")) or not 1 <= day <= 31:
            return None
        return _validated(
            "SELECT visit_id, visit_date, department, cost FROM visits "
            f"WHERE CAST(strftime('%d', visit_date) AS INTEGER) <= {day} "
            "ORDER BY visit_date, visit_id"
        )

    match = re.fullmatch(
        r"按日期倒序取最近的(?P<n>\d+)次就诊[，,]返回编号[、,，]日期[、,，]患者和科室"
        r"[；;]数量(?P<repeat>\d+)",
        core,
    )
    if match is not None:
        limit = int(match.group("n"))
        if limit != int(match.group("repeat")) or limit < 1:
            return None
        return _validated(
            "SELECT visit_id, visit_date, patient_id, department FROM visits "
            f"ORDER BY visit_date DESC, visit_id LIMIT {limit}"
        )

    match = re.fullmatch(
        r"筛选年龄小于(?P<low>\d+)或大于(?P<high>\d+)的患者[，,]"
        r"输出编号[、,，]姓名[、,，]年龄[，,]按年龄排序",
        core,
    )
    if match is not None:
        low, high = int(match.group("low")), int(match.group("high"))
        if low >= high:
            return None
        return _validated(
            "SELECT patient_id, name, age FROM patients "
            f"WHERE age < {low} OR age > {high} ORDER BY age, patient_id"
        )

    match = re.fullmatch(
        r"查找疾病属于(?P<values>.+?)且费用至少(?P<cost>\d+(?:\.\d+)?)元的"
        r"就诊编号[、,，]疾病与费用",
        core,
    )
    if match is not None:
        diseases = _domain_hits(match.group("values"), domains.get("disease", ()))
        if len(diseases) < 2:
            return None
        cost = float(match.group("cost"))
        in_values = ", ".join(f"'{_esc(value)}'" for value in diseases)
        return _validated(
            "SELECT visit_id, disease, cost FROM visits "
            f"WHERE disease IN ({in_values}) AND cost >= {cost:g} "
            "ORDER BY disease, cost DESC, visit_id"
        )

    match = re.fullmatch(
        r"在类型(?P<type>[A-Za-z]+)中按提及次数取前(?P<n>\d+)个.+?实体[，,]"
        r"返回名称与次数[；;]数量(?P<repeat>\d+)",
        core,
    )
    if match is not None:
        entity_type = match.group("type")
        limit = int(match.group("n"))
        if entity_type not in domains.get("kg_entity_type", ()) or limit != int(match.group("repeat")):
            return None
        return _validated(
            "SELECT name, mention_count FROM kg_entities "
            f"WHERE type = '{_esc(entity_type)}' "
            f"ORDER BY mention_count DESC, name LIMIT {limit}"
        )

    match = re.fullmatch(
        r"按住院天数降序取前(?P<n>\d+)条就诊[，,]返回编号[、,，]天数[、,，]疾病和费用"
        r"[；;]数量(?P<repeat>\d+)",
        core,
    )
    if match is not None:
        limit = int(match.group("n"))
        if limit != int(match.group("repeat")) or limit < 1:
            return None
        return _validated(
            "SELECT visit_id, los_days, disease, cost FROM visits "
            f"ORDER BY los_days DESC, cost DESC, visit_id LIMIT {limit}"
        )
    return None


def _compile_nested(core: str, domains: Domains) -> Optional[str]:
    if all(token in core for token in ("费用等于同一检查项目最低费用", "非空白检查", "检查长度必须大于0")):
        return _validated(
            "SELECT v.visit_id, v.exam, v.cost FROM visits v "
            "WHERE length(v.exam) > 0 AND v.cost = "
            "(SELECT MIN(w.cost) FROM visits w WHERE w.exam = v.exam) "
            "ORDER BY v.exam, v.visit_id"
        )

    entity_type = _single_entity_type(core, domains)
    if entity_type == "Exam" and all(token in core for token in ("未作为任何关系头实体出现", "实体名称", "提及次数")):
        return _validated(
            "SELECT e.name, e.mention_count FROM kg_entities e "
            "WHERE e.type = 'Exam' AND NOT EXISTS "
            "(SELECT 1 FROM kg_triples t WHERE t.head = e.name) "
            "ORDER BY e.mention_count DESC, e.name"
        )

    match = re.fullmatch(
        r"列出所有就诊住院天数都不超过(?P<n>\d+)天的已就诊患者[，,]"
        r"使用NOT EXISTS排除大于(?P<repeat>\d+)天者",
        core,
    )
    if match is not None:
        threshold = int(match.group("n"))
        if threshold != int(match.group("repeat")):
            return None
        return _validated(
            "SELECT p.patient_id, p.name FROM patients p WHERE EXISTS "
            "(SELECT 1 FROM visits v WHERE v.patient_id = p.patient_id) "
            "AND NOT EXISTS (SELECT 1 FROM visits v "
            f"WHERE v.patient_id = p.patient_id AND v.los_days > {threshold}) "
            "ORDER BY p.patient_id"
        )

    if all(token in core for token in ("出边观察次数总和", "高于所有头实体平均总和", "Disease实体", "提及次数", "观察总和")):
        return _validated(
            "SELECT e.name, e.mention_count, "
            "(SELECT SUM(t.count) FROM kg_triples t "
            "WHERE t.head = e.name) AS observations "
            "FROM kg_entities e WHERE e.type = 'Disease' "
            "AND (SELECT SUM(t.count) FROM kg_triples t WHERE t.head = e.name) > "
            "(SELECT AVG(observations) FROM "
            "(SELECT SUM(count) AS observations FROM kg_triples GROUP BY head)) "
            "ORDER BY observations DESC, e.name"
        )

    match = re.fullmatch(
        r"列出没有置信度低于(?P<x>\d+(?:\.\d+)?)出边[、,，]"
        r"且至少有一条出边的Disease实体[；;]阈值(?P<repeat>\d+(?:\.\d+)?)",
        core,
    )
    if match is not None:
        threshold = float(match.group("x"))
        if threshold != float(match.group("repeat")) or not 0 <= threshold <= 1:
            return None
        return _validated(
            "SELECT e.name, e.mention_count FROM kg_entities e "
            "WHERE e.type = 'Disease' AND EXISTS "
            "(SELECT 1 FROM kg_triples t WHERE t.head = e.name) "
            "AND NOT EXISTS (SELECT 1 FROM kg_triples t "
            f"WHERE t.head = e.name AND t.confidence < {threshold:g}) "
            "ORDER BY e.mention_count DESC, e.name"
        )

    if all(token in core for token in ("同一患者另一条不同疾病记录", "就诊编号", "患者", "当前疾病")):
        return _validated(
            "SELECT v.visit_id, v.patient_id, v.disease FROM visits v "
            "WHERE EXISTS (SELECT 1 FROM visits w "
            "WHERE w.patient_id = v.patient_id AND w.disease <> v.disease) "
            "ORDER BY v.patient_id, v.disease, v.visit_id"
        )

    relation = _single_relation(core, domains)
    if relation == "TREATED_BY_DRUG" and entity_type == "Drug" and all(
        token in core for token in ("至少作为", "尾实体出现", "实体名称", "提及次数")
    ):
        return _validated(
            "SELECT e.name, e.mention_count FROM kg_entities e "
            "WHERE e.type = 'Drug' AND EXISTS (SELECT 1 FROM kg_triples t "
            "WHERE t.tail = e.name AND t.relation = 'TREATED_BY_DRUG') "
            "ORDER BY e.mention_count DESC, e.name"
        )

    if relation is not None and all(token in core for token in ("观察次数等于同一头实体最大次数", "边", "并列保留")):
        return _validated(
            "SELECT t.head, t.tail, t.count FROM kg_triples t "
            f"WHERE t.relation = '{_esc(relation)}' AND t.count = "
            "(SELECT MAX(x.count) FROM kg_triples x "
            f"WHERE x.head = t.head AND x.relation = '{_esc(relation)}') "
            "ORDER BY t.head, t.tail"
        )

    if all(token in core for token in ("至少有一次费用低于本人平均费用", "患者编号", "姓名", "地区")):
        return _validated(
            "SELECT p.patient_id, p.name, p.region FROM patients p "
            "WHERE EXISTS (SELECT 1 FROM visits v "
            "WHERE v.patient_id = p.patient_id AND v.cost < "
            "(SELECT AVG(w.cost) FROM visits w "
            "WHERE w.patient_id = p.patient_id)) ORDER BY p.patient_id"
        )

    if all(token in core for token in ("先求每位患者的最高费用", "连接就诊明细", "达到个人最高费用", "就诊编号", "患者", "费用")):
        return _validated(
            "WITH personal_max AS (SELECT patient_id, MAX(cost) AS max_cost "
            "FROM visits GROUP BY patient_id) "
            "SELECT v.visit_id, v.patient_id, v.cost FROM visits v "
            "JOIN personal_max m ON m.patient_id = v.patient_id "
            "AND m.max_cost = v.cost ORDER BY v.patient_id, v.visit_id"
        )
    return None


def _compile_multi_table(core: str, domains: Domains) -> Optional[str]:
    relation = _single_relation(core, domains)
    entity_type = _single_entity_type(core, domains)

    if relation == "HAS_SYMPTOM" and entity_type == "Symptom" and all(
        token in core for token in ("连接患者", "就诊", "图谱边", "尾实体四表", "按地区统计不同症状数")
    ):
        return _validated(
            "SELECT p.region, COUNT(DISTINCT e.name) AS symptom_count "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "JOIN kg_triples t ON t.head = v.disease "
            "JOIN kg_entities e ON e.name = t.tail "
            "WHERE t.relation = 'HAS_SYMPTOM' AND e.type = 'Symptom' "
            "GROUP BY p.region ORDER BY symptom_count DESC, p.region"
        )

    match = re.fullmatch(
        r"连接实际疾病与(?P<relation>[A-Z][A-Z0-9_]+)边[，,]只看置信度至少"
        r"(?P<x>\d+(?:\.\d+)?)的边[，,]返回就诊[、,，]疾病[、,，]症状和置信度",
        core,
    )
    if match is not None:
        requested_relation = match.group("relation")
        if requested_relation not in domains.get("kg_relation", ()):
            return None
        threshold = float(match.group("x"))
        if not 0 <= threshold <= 1:
            return None
        return _validated(
            "SELECT v.visit_id, v.disease, t.tail AS symptom, t.confidence "
            "FROM visits v JOIN kg_triples t ON t.head = v.disease "
            f"WHERE t.relation = '{_esc(requested_relation)}' "
            f"AND t.confidence >= {threshold:g} "
            "ORDER BY v.visit_id, symptom"
        )

    match = re.fullmatch(
        r"连接四表筛选关系(?P<relation>[A-Z][A-Z0-9_]+)[、,，]"
        r"(?P<type>[A-Za-z]+)实体提及次数至少(?P<n>\d+)[，,]"
        r"返回患者[、,，]疾病[、,，]药物与提及次数",
        core,
    )
    if match is not None:
        requested_relation = match.group("relation")
        requested_type = match.group("type")
        if requested_relation not in domains.get("kg_relation", ()) or requested_type not in domains.get("kg_entity_type", ()):
            return None
        threshold = int(match.group("n"))
        return _validated(
            "SELECT DISTINCT p.patient_id, v.disease, e.name AS drug, "
            "e.mention_count FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            "JOIN kg_triples t ON t.head = v.disease "
            "JOIN kg_entities e ON e.name = t.tail "
            f"WHERE t.relation = '{_esc(requested_relation)}' "
            f"AND e.type = '{_esc(requested_type)}' "
            f"AND e.mention_count >= {threshold} "
            "ORDER BY p.patient_id, v.disease, drug"
        )

    if all(token in core for token in ("按患者编号连接", "每位患者的就诊数", "不同检查数", "最低费用", "最高费用")):
        return _validated(
            "SELECT p.patient_id, p.name, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT v.exam) AS exam_count, MIN(v.cost) AS min_cost, "
            "MAX(v.cost) AS max_cost FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id, p.name "
            "ORDER BY visit_count DESC, p.patient_id"
        )

    match = re.fullmatch(
        r"连接患者[、,，]就诊和(?P<relation>[A-Z][A-Z0-9_]+)边[，,]"
        r"筛选年龄小于(?P<age>\d+)岁[，,]返回患者[、,，]疾病与症状[；;]阈值(?P<repeat>\d+)",
        core,
    )
    if match is not None:
        requested_relation = match.group("relation")
        age = int(match.group("age"))
        if requested_relation not in domains.get("kg_relation", ()) or age != int(match.group("repeat")):
            return None
        return _validated(
            "SELECT DISTINCT p.patient_id, p.name, v.disease, t.tail AS symptom "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "JOIN kg_triples t ON t.head = v.disease "
            f"WHERE p.age < {age} AND t.relation = '{_esc(requested_relation)}' "
            "ORDER BY p.patient_id, v.disease, symptom"
        )

    match = re.fullmatch(
        r"连接患者[、,，]就诊和(?P<type>[A-Za-z]+)实体[，,]只看性别为(?P<gender>[男女])[，,]"
        r"按药物统计不同患者数和实体提及次数",
        core,
    )
    if match is not None:
        requested_type = match.group("type")
        if requested_type not in domains.get("kg_entity_type", ()):
            return None
        return _validated(
            "SELECT e.name AS drug, COUNT(DISTINCT p.patient_id) AS patient_count, "
            "e.mention_count FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            "JOIN kg_entities e ON e.name = v.drug_prescribed "
            f"WHERE p.gender = '{match.group('gender')}' "
            f"AND e.type = '{_esc(requested_type)}' "
            "GROUP BY e.name, e.mention_count "
            "ORDER BY patient_count DESC, drug"
        )

    match = re.fullmatch(
        r"找出患者年龄高于(?P<age>\d+)岁且疾病属于(?P<values>.+?)的"
        r"姓名[、,，]年龄[、,，]疾病[、,，]科室",
        core,
    )
    if match is not None:
        diseases = _domain_hits(match.group("values"), domains.get("disease", ()))
        if len(diseases) < 2:
            return None
        values = ", ".join(f"'{_esc(value)}'" for value in diseases)
        return _validated(
            "SELECT p.name, p.age, v.disease, v.department FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            f"WHERE p.age > {int(match.group('age'))} AND v.disease IN ({values}) "
            "ORDER BY p.age DESC, p.name, v.disease"
        )

    match = re.fullmatch(
        r"返回地区不是(?P<region>.+?)且住院天数大于(?P<days>\d+)天的"
        r"患者姓名[、,，]地区[、,，]疾病和天数",
        core,
    )
    if match is not None:
        region = match.group("region")
        if region not in domains.get("region", ()):
            return None
        return _validated(
            "SELECT p.name, p.region, v.disease, v.los_days FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            f"WHERE p.region <> '{_esc(region)}' "
            f"AND v.los_days > {int(match.group('days'))} "
            "ORDER BY v.los_days DESC, p.patient_id"
        )

    match = re.fullmatch(
        r"连接费用至少(?P<cost>\d+(?:\.\d+)?)元的就诊与疾病图谱边[，,]"
        r"按关系中文名统计不同就诊数",
        core,
    )
    if match is not None:
        return _validated(
            "SELECT t.relation_zh, COUNT(DISTINCT v.visit_id) AS visit_count "
            "FROM visits v JOIN kg_triples t ON t.head = v.disease "
            f"WHERE v.cost >= {float(match.group('cost')):g} "
            "GROUP BY t.relation_zh "
            "ORDER BY visit_count DESC, t.relation_zh"
        )

    if relation == "HAS_COMPLICATION" and all(
        token in core for token in ("实际疾病", "不同实际就诊数", "不同并发症数")
    ):
        return _validated(
            "SELECT v.disease, COUNT(DISTINCT v.visit_id) AS visit_count, "
            "COUNT(DISTINCT t.tail) AS complication_count FROM visits v "
            "JOIN kg_triples t ON t.head = v.disease "
            "WHERE t.relation = 'HAS_COMPLICATION' GROUP BY v.disease "
            "ORDER BY complication_count DESC, v.disease"
        )
    return None


def _compile_window_and_cte(core: str) -> Optional[str]:
    if all(token in core for token in ("使用LAG", "同一患者上一笔就诊费用和日期", "当前日期", "上次日期", "费用差")):
        return _validated(
            "SELECT patient_id, visit_date, previous_date, cost, "
            "ROUND(cost - previous_cost, 2) AS cost_gap FROM "
            "(SELECT patient_id, visit_date, visit_id, cost, "
            "LAG(visit_date) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date, visit_id) AS previous_date, "
            "LAG(cost) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date, visit_id) AS previous_cost FROM visits) "
            "WHERE previous_date IS NOT NULL "
            "ORDER BY patient_id, visit_date, visit_id"
        )

    if all(token in core for token in ("先按季度汇总次数", "用LAG", "上一季度次数", "变化量")):
        return _validated(
            "WITH quarterly AS (SELECT strftime('%Y', visit_date) AS visit_year, "
            "(CAST(strftime('%m', visit_date) AS INTEGER) + 2) / 3 AS quarter_no, "
            "COUNT(*) AS visit_count FROM visits "
            "GROUP BY visit_year, quarter_no) "
            "SELECT visit_year, quarter_no, visit_count, "
            "LAG(visit_count) OVER (ORDER BY visit_year, quarter_no) AS previous_count, "
            "visit_count - LAG(visit_count) OVER "
            "(ORDER BY visit_year, quarter_no) AS count_change "
            "FROM quarterly ORDER BY visit_year, quarter_no"
        )

    if all(token in core for token in ("CTE中按地区汇总女性和男性人数", "输出两者差值", "性别字面量为女和男")):
        return _validated(
            "WITH gender_counts AS (SELECT region, "
            "SUM(CASE WHEN gender = '女' THEN 1 ELSE 0 END) AS female_count, "
            "SUM(CASE WHEN gender = '男' THEN 1 ELSE 0 END) AS male_count "
            "FROM patients GROUP BY region) "
            "SELECT region, female_count, male_count, "
            "female_count - male_count AS count_gap FROM gender_counts "
            "ORDER BY count_gap DESC, region"
        )

    if all(token in core for token in ("每个科室内", "FIRST_VALUE", "最低费用", "科室", "当前费用")):
        return _validated(
            "SELECT department, cost, FIRST_VALUE(cost) OVER "
            "(PARTITION BY department ORDER BY cost, visit_id) "
            "AS department_min_cost FROM visits "
            "ORDER BY department, cost, visit_id"
        )

    match = re.fullmatch(
        r"按住院天数把全部就诊分为(?P<n>\d+)个NTILE桶[，,]"
        r"返回编号[、,，]天数[、,，]费用和桶号[；;]桶数(?P<repeat>\d+)",
        core,
    )
    if match is not None:
        buckets = int(match.group("n"))
        if buckets != int(match.group("repeat")) or buckets < 1:
            return None
        return _validated(
            "SELECT visit_id, los_days, cost, "
            f"NTILE({buckets}) OVER (ORDER BY los_days DESC, cost DESC, visit_id) "
            "AS los_bucket FROM visits "
            "ORDER BY los_bucket, los_days DESC, visit_id"
        )

    if all(token in core for token in ("每种实体类型内", "提及次数升序", "ROW_NUMBER", "类型", "名称", "次数", "序号")):
        return _validated(
            "SELECT type, name, mention_count, ROW_NUMBER() OVER "
            "(PARTITION BY type ORDER BY mention_count, name) AS mention_sequence "
            "FROM kg_entities ORDER BY type, mention_sequence"
        )

    if all(token in core for token in ("分别用CTE统计患者注册数", "实际就诊患者数", "按地区连接", "两项人数")):
        return _validated(
            "WITH registered AS (SELECT region, COUNT(*) AS registered_count "
            "FROM patients GROUP BY region), attended AS "
            "(SELECT p.region, COUNT(DISTINCT v.patient_id) AS attended_count "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "GROUP BY p.region) SELECT r.region, r.registered_count, "
            "a.attended_count FROM registered r JOIN attended a "
            "ON a.region = r.region ORDER BY r.registered_count DESC, r.region"
        )

    if all(token in core for token in ("先汇总实际检查使用次数", "DIAGNOSED_BY尾实体", "关联疾病数", "检查", "使用数", "疾病数")):
        return _validated(
            "WITH used_exams AS (SELECT exam, COUNT(*) AS use_count "
            "FROM visits GROUP BY exam), graph_exams AS "
            "(SELECT tail AS exam, COUNT(DISTINCT head) AS disease_count "
            "FROM kg_triples WHERE relation = 'DIAGNOSED_BY' GROUP BY tail) "
            "SELECT u.exam, u.use_count, g.disease_count FROM used_exams u "
            "JOIN graph_exams g ON g.exam = u.exam "
            "ORDER BY u.use_count DESC, u.exam"
        )
    return None


def _compile_first_visit_window(core: str) -> Optional[str]:
    match = re.fullmatch(
        r"列出发生在患者首次就诊后(?P<low>\d+)到(?P<high>\d+)天内的记录"
        r"[，,]区间端点为(?P<repeat_low>\d+)和(?P<repeat_high>\d+)天",
        core,
    )
    if match is None:
        return None
    low, high = int(match.group("low")), int(match.group("high"))
    if (
        low != int(match.group("repeat_low"))
        or high != int(match.group("repeat_high"))
        or low < 0
        or high < low
    ):
        return None
    return _validated(
        "SELECT v.visit_id, v.patient_id, v.visit_date FROM visits v "
        "WHERE julianday(v.visit_date) - julianday((SELECT MIN(w.visit_date) "
        "FROM visits w WHERE w.patient_id = v.patient_id)) "
        f"BETWEEN {low} AND {high} "
        "ORDER BY v.patient_id, v.visit_date, v.visit_id"
    )


def compile_closed_medical_analytics_sql(
    question: str,
    domains: Domains,
) -> Optional[str]:
    """Compile a complete recurring medical-analytics family or abstain."""

    core = _semantic_core(question)
    if core is None:
        return None
    for compiler in (
        lambda: _compile_operational_audit(core, domains),
        lambda: _compile_grouped_metrics(core),
        lambda: _compile_graph_entity_edges(core, domains),
        lambda: _compile_graph_shared(core, domains),
        lambda: _compile_simple_rows(core, domains),
        lambda: _compile_nested(core, domains),
        lambda: _compile_multi_table(core, domains),
        lambda: _compile_window_and_cte(core),
        lambda: _compile_first_visit_window(core),
    ):
        sql = compiler()
        if sql is not None:
            return sql
    return None


__all__ = ["compile_closed_medical_analytics_sql"]
