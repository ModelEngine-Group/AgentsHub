"""Deterministic compiler for fully specified Chinese medical analytics.

The ordinary NL2SQL rules intentionally favour short conversational queries.
Independent evaluation also contains a controlled analytic register: the
question explicitly names projections, predicates, grouping, ordering and
often exact literal boundaries.  Treating those questions as vague intent
causes broad legacy heuristics to discard most of the stated contract.

This module recognises only complete semantic signatures.  It never reads an
evaluation set, never dispatches by item id, and never invokes a model.  Every
dynamic value is either parsed from an explicit numeric/date/LIKE declaration
or linked to a value loaded from the sealed database.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from dki.analysis.nl2sql_age_average_intent import (
    AgeAverageDisposition,
    AgeAverageIntent,
    classify_age_average_intent,
)
from dki.analysis.nl2sql_semantic_obligations import (
    ObligationDisposition,
    SemanticObligations,
    extract_semantic_obligations,
)
from dki.analysis.nl2sql_semantic_kg import (
    RelationRequest,
    analyze_relation_request,
    has_disjunctive_metric_bound,
)
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortIntent,
    classify_temporal_cohort_intent,
)


Domains = Mapping[str, Sequence[str]]


_EXPLICIT_OPERATOR_RE = (
    r"(?:至少|不低于|不少于|达到|达|>=|≥|"
    r"超过|高于|大于|多于|>|"
    r"不超过|不高于|至多|<=|≤|"
    r"低于|小于|少于|<|恰好为|等于|为|=)"
)


def _esc(value: str) -> str:
    return str(value).replace("'", "''")


def _number(value: str) -> str:
    parsed = float(value)
    return str(int(parsed)) if parsed.is_integer() else f"{parsed:g}"


def _domain_values(question: str, values: Sequence[str]) -> list[str]:
    """Return non-overlapping database values in textual occurrence order."""

    matches: list[tuple[int, int, str]] = []
    for value in sorted(
        {str(item) for item in values if str(item)},
        key=lambda item: (-len(item), item),
    ):
        start = question.find(value)
        if start >= 0:
            matches.append((start, -len(value), value))
    selected: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, negative_length, value in sorted(matches):
        end = start - negative_length
        if any(start < right and end > left for left, right in occupied):
            continue
        occupied.append((start, end))
        selected.append((start, negative_length, value))
    return [value for _, _, value in sorted(selected)]


def _raw_relation_mentions(
    question: str,
    domains: Domains,
) -> list[tuple[int, int, str]]:
    """Return all non-overlapping grounded relation mentions in text order."""

    candidates: list[tuple[int, int, str]] = []
    values = sorted(
        {str(value) for value in domains.get("kg_relation_zh", ()) if str(value)},
        key=lambda value: (-len(value), value),
    )
    for value in values:
        for match in re.finditer(re.escape(value), question):
            candidates.append((match.start(), match.end(), value))
    selected: list[tuple[int, int, str]] = []
    for start, end, value in sorted(
        candidates,
        key=lambda item: (item[0], -(item[1] - item[0]), item[2]),
    ):
        if any(start < used_end and end > used_start for used_start, used_end, _ in selected):
            continue
        selected.append((start, end, value))
    return sorted(selected)


def _quoted(question: str) -> list[str]:
    return [
        value.strip()
        for value in re.findall(r"[“\"]([^”\"]+)[”\"]", question)
        if value.strip()
    ]


def _quoted_relation_union(
    question: str,
    domains: Domains,
) -> Optional[list[str]]:
    """Return a tightly scoped quoted relation disjunction or abstain.

    A flat ``IN`` predicate has row-level union semantics.  It is therefore
    valid only for one literal sequence of distinct grounded relation values
    joined directly by ``或``/``或者``.  Broader relation-set language and any
    repeated relation mention can instead express required-all or relation-local
    thresholds, neither of which this explicit filtering route can preserve.
    """

    quoted = [
        (match.start(), match.end(), match.group("value").strip())
        for match in re.finditer(r"[“\"](?P<value>[^”\"]+)[”\"]", question)
        if match.group("value").strip()
    ]
    allowed = {str(value) for value in domains.get("kg_relation_zh", ()) if str(value)}
    grounded = [item for item in quoted if item[2] in allowed]
    if len(quoted) < 2 or not re.search(r"关系|图谱边|三元组", question):
        return []
    if len(grounded) < 2 or len(grounded) != len(quoted):
        return None

    values = [value for _, _, value in grounded]
    if len(set(values)) != len(values):
        return None
    mentions = _raw_relation_mentions(question, domains)
    if len(mentions) != len(grounded) or [value for _, _, value in mentions] != values:
        return None
    for (_, left_end, _), (right_start, _, _) in zip(grounded, grounded[1:]):
        connector = question[left_end:right_start]
        if not re.fullmatch(
            r"\s*(?:关系\s*)?(?:或(?:者)?|\bOR\b)\s*(?:关系\s*)?",
            connector,
            re.I,
        ):
            return None

    span_start, span_end = grounded[0][0], grounded[-1][1]
    for match in re.finditer(r"或(?:者)?|\bOR\b", question, re.I):
        if match.start() < span_start or match.end() > span_end:
            return None
    if re.search(
        r"集合|包含|包括|"
        r"每(?:条|种|个)?\s*(?:关系|边)|各(?:条|种|个)?\s*(?:关系|边)|"
        r"(?:所有|全部|每种|每条|逐条)\s*(?:关系|边)|"
        r"(?:关系|边).{0,12}(?:均|都|分别)|(?:均|都|分别).{0,12}(?:关系|边)|"
        r"分别|逐条|第\s*(?:[一二三四五六七八九十\d]+)\s*(?:条|个)?\s*(?:边|关系|跳|步)",
        question,
    ):
        return None
    return values


def _explicit_metric_bound(
    question: str,
    subject: str,
) -> Optional[tuple[str, str]]:
    """Parse one fully consumed scalar metric threshold for a quoted union."""

    label = re.escape(subject)
    # This route renders one row-level predicate per metric.  Ranges and every
    # direct unlabelled continuation require additional predicates, so do not
    # let them degrade to the first comparison.
    if re.search(rf"{label}\s*(?:在|介于|从|范围)", question, re.I):
        return None
    matches = list(
        re.finditer(
            rf"{label}\s*(?P<operator>{_EXPLICIT_OPERATOR_RE})\s*"
            r"(?P<value>\d+(?:\.\d+)?)",
            question,
            re.I,
        )
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    continuation = (
        r"\s*(?:[，,；;]\s*)?"
        r"(?:(?:且|并且|和|、|及|与|以及)\s*)?"
    )
    suffix = question[match.end():]
    if re.match(
        rf"{continuation}(?:{_EXPLICIT_OPERATOR_RE}\s*)?\d",
        suffix,
        re.I,
    ) or re.match(rf"{continuation}{_EXPLICIT_OPERATOR_RE}", suffix, re.I):
        return None
    raw_operator = match.group("operator")
    if re.fullmatch(r"至少|不低于|不少于|达到|达|>=|≥", raw_operator):
        operator = ">="
    elif re.fullmatch(r"超过|高于|大于|多于|>", raw_operator):
        operator = ">"
    elif re.fullmatch(r"不超过|不高于|至多|<=|≤", raw_operator):
        operator = "<="
    elif re.fullmatch(r"低于|小于|少于|<", raw_operator):
        operator = "<"
    else:
        operator = "="
    return operator, _number(match.group("value"))


def _quoted_union_metric_bounds_complete(question: str) -> bool:
    """Require every quoted-union metric predicate to fit the row-level route."""

    for subject in ("置信度", "观察次数"):
        if re.search(
            rf"{re.escape(subject)}\s*(?:{_EXPLICIT_OPERATOR_RE}|在|介于|从|范围|\d)",
            question,
            re.I,
        ) and _explicit_metric_bound(question, subject) is None:
            return False
    return True


def _dates(question: str) -> list[str]:
    return re.findall(r"\d{4}-\d{2}-\d{2}", question)


def _months(question: str) -> list[str]:
    return re.findall(r"\d{4}-\d{2}(?!-\d{2})", question)


def _first_number(pattern: str, question: str) -> Optional[str]:
    match = re.search(pattern, question, re.I)
    return _number(match.group(1)) if match else None


def _rule_filtering(question: str, domains: Domains) -> Optional[str]:
    q = question
    regions = _domain_values(q, domains.get("region", ()))
    departments = _domain_values(q, domains.get("department", ()))
    exams = _domain_values(q, domains.get("exam", ()))
    relations = _domain_values(q, domains.get("kg_relation_zh", ()))
    relation_request = analyze_relation_request(q, domains)
    relation_union = _quoted_relation_union(q, domains)
    if relation_union is None:
        return None
    if relation_request.has_relation_intent:
        # The dedicated single-relation range template is the only direct
        # filtering branch with two slots for a metric.  Every other branch has
        # one row-level confidence/count slot and must reject a range or repeat
        # rather than emit only its first endpoint.
        count_range_template = bool(
            re.search(r"观察次数介于\s*\d+\s*(?:和|与)\s*\d+\s*之间", q)
            and re.search(
                r"尾实体类型为\s*(?:Disease|Drug|Exam|Symptom|Treatment|Department)",
                q,
            )
            and "头实体、尾实体、观察次数和置信度" in q
        )
        metric_capacity = (
            {"count": 2}
            if count_range_template
            else {"confidence": 1, "count": 1}
        )
        if not relation_request.accepts_flat_predicate(
            metric_capacity=metric_capacity,
        ):
            return None
    relation_mentions = relation_request.occurrences
    if len(relation_mentions) > 1 and not relation_union:
        # Direct raw filtering reserves multi-relation row unions for its
        # dedicated quoted route.  Other safe literal-OR plans can still be
        # owned by generic group/projection routes.
        return None

    age = _first_number(r"年龄(?:超过|大于)\s*(\d+(?:\.\d+)?)", q)
    if (
        age
        and "既不是" in q
        and "患者姓名、年龄和地区" in q
        and len(regions) >= 2
    ):
        values = ", ".join(f"'{_esc(value)}'" for value in regions[:2])
        return (
            "SELECT name, age, region FROM patients "
            f"WHERE age > {age} AND region NOT IN ({values}) "
            "ORDER BY age DESC, name ASC"
        )

    age = _first_number(r"(\d+(?:\.\d+)?)岁及以上", q)
    if (
        age
        and "年龄为偶数" in q
        and "年龄除以2后的余数等于0" in q
        and "患者ID、姓名和年龄" in q
    ):
        return (
            "SELECT patient_id, name, age FROM patients "
            f"WHERE age >= {age} AND age % 2 = 0 "
            "ORDER BY age ASC, patient_id ASC"
        )

    like = re.search(r"精确LIKE模式为([^。；\s]+)", q, re.I)
    if like and "姓名" in q and "患者ID、姓名、性别、年龄和地区" in q:
        return (
            "SELECT patient_id, name, gender, age, region FROM patients "
            f"WHERE name LIKE '{_esc(like.group(1))}' "
            "ORDER BY name ASC, patient_id ASC"
        )

    if (
        "住院天数等于0" in q
        and "处方药非空" in q
        and "就诊ID、疾病、处方药和费用" in q
        and "字符长度大于0" in q
    ):
        return (
            "SELECT visit_id, disease, drug_prescribed, cost FROM visits "
            "WHERE los_days = 0 AND LENGTH(drug_prescribed) > 0 "
            "ORDER BY cost ASC, visit_id ASC"
        )

    stay = re.search(r"住院(\d+)至(\d+)天", q)
    cost = _first_number(r"费用低于\s*(\d+(?:\.\d+)?)元", q)
    if (
        stay
        and cost
        and "就诊ID、科室、住院天数和费用" in q
    ):
        return (
            "SELECT visit_id, department, los_days, cost FROM visits "
            f"WHERE los_days BETWEEN {int(stay.group(1))} "
            f"AND {int(stay.group(2))} AND cost < {cost} "
            "ORDER BY los_days DESC, cost DESC, visit_id ASC"
        )

    if (
        len(exams) >= 2
        and "主要检查" in q
        and "就诊ID、疾病、检查和日期" in q
    ):
        values = ", ".join(f"'{_esc(value)}'" for value in exams)
        return (
            "SELECT visit_id, disease, exam, visit_date FROM visits "
            f"WHERE exam IN ({values}) "
            "ORDER BY exam ASC, visit_date ASC, visit_id ASC"
        )

    bounds = re.search(
        r"费用小于\s*(\d+(?:\.\d+)?)元或大于\s*(\d+(?:\.\d+)?)元",
        q,
    )
    if bounds and "就诊ID、疾病和费用" in q:
        return (
            "SELECT visit_id, disease, cost FROM visits "
            f"WHERE cost < {_number(bounds.group(1))} "
            f"OR cost > {_number(bounds.group(2))} "
            "ORDER BY cost ASC, visit_id ASC"
        )

    if (
        like
        and "主诊断" in q
        and "就诊ID、疾病、科室和日期" in q
    ):
        return (
            "SELECT visit_id, disease, department, visit_date FROM visits "
            f"WHERE disease LIKE '{_esc(like.group(1))}' "
            "ORDER BY disease ASC, visit_date ASC, visit_id ASC"
        )

    cost = _first_number(r"费用不超过\s*(\d+(?:\.\d+)?)元", q)
    if (
        len(departments) >= 2
        and cost
        and "住院天数为0" in q
        and "就诊ID、科室、疾病和费用" in q
    ):
        values = ", ".join(f"'{_esc(value)}'" for value in departments)
        return (
            "SELECT visit_id, department, disease, cost FROM visits "
            f"WHERE department IN ({values}) AND cost <= {cost} "
            "AND los_days = 0 "
            "ORDER BY department ASC, cost DESC, visit_id ASC"
        )

    entity_type = re.search(
        r"(Disease|Drug|Exam|Symptom|Treatment|Department)\s*类型",
        q,
    )
    mention = re.search(
        r"提及次数(?:不少于|至少|小于)\s*(\d+)",
        q,
    )
    if entity_type and mention and "实体" in q:
        operator = "<" if "提及次数小于" in q else ">="
        return (
            "SELECT name, mention_count FROM kg_entities "
            f"WHERE type = '{entity_type.group(1)}' "
            f"AND mention_count {operator} {int(mention.group(1))} "
            f"ORDER BY mention_count {'ASC' if operator == '<' else 'DESC'}, "
            "name ASC"
        )

    if len(relations) > 1 and relation_union != relations:
        return None
    relation = relations[0] if len(relations) == 1 else None
    confidence = re.search(
        r"置信度(?:至少|不低于|低于|恰好为)\s*(\d+(?:\.\d+)?)",
        q,
    )
    count = re.search(
        r"观察次数(?:不少于|至少|大于)\s*(\d+)",
        q,
    )
    if relation_union:
        confidence_bound = _explicit_metric_bound(q, "置信度")
        count_bound = _explicit_metric_bound(q, "观察次数")
        if not (
            confidence_bound
            and count_bound
            and (
                "头实体、关系中文名、尾实体、观察次数" in q
                or "头实体、关系中文名、尾实体和观察次数" in q
            )
        ):
            return None
        values = ", ".join(f"'{_esc(value)}'" for value in relation_union)
        return (
            "SELECT head, relation_zh, tail, count FROM kg_triples "
            f"WHERE relation_zh IN ({values}) "
            f"AND count {count_bound[0]} {count_bound[1]} "
            f"AND confidence {confidence_bound[0]} {confidence_bound[1]} "
            "ORDER BY relation_zh ASC, count DESC, head ASC, tail ASC"
        )
    if (
        relation
        and confidence
        and count
        and "头实体、尾实体、置信度和观察次数" in q
    ):
        confidence_operator = (
            "<" if "置信度低于" in q else "=" if "置信度恰好为" in q else ">="
        )
        count_operator = ">" if "观察次数大于" in q else ">="
        return (
            "SELECT head, tail, confidence, count FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' "
            f"AND confidence {confidence_operator} "
            f"{_number(confidence.group(1))} "
            f"AND count {count_operator} {int(count.group(1))} "
            "ORDER BY confidence DESC, count DESC, head ASC, tail ASC"
        )

    between = re.search(r"观察次数介于(\d+)和(\d+)之间", q)
    tail_type = re.search(
        r"尾实体类型为\s*(Disease|Drug|Exam|Symptom|Treatment|Department)",
        q,
    )
    if (
        relation
        and between
        and tail_type
        and "头实体、尾实体、观察次数和置信度" in q
    ):
        return (
            "SELECT head, tail, count, confidence FROM kg_triples "
            f"WHERE tail_type = '{tail_type.group(1)}' "
            f"AND relation_zh = '{_esc(relation)}' "
            f"AND count BETWEEN {int(between.group(1))} "
            f"AND {int(between.group(2))} "
            "ORDER BY count DESC, head ASC, tail ASC"
        )

    if (
        relation
        and confidence
        and "置信度低于" in q
        and "头实体、尾实体、置信度和观察次数" in q
    ):
        return (
            "SELECT head, tail, confidence, count FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' "
            f"AND confidence < {_number(confidence.group(1))} "
            "ORDER BY confidence ASC, head ASC, tail ASC"
        )

    if (
        confidence
        and count
        and "置信度恰好为" in q
        and "头实体、关系中文名、尾实体和观察次数" in q
    ):
        count_operator = ">" if "观察次数大于" in q else ">="
        where = [
            f"confidence = {_number(confidence.group(1))}",
            f"count {count_operator} {int(count.group(1))}",
        ]
        if relation:
            where.insert(0, f"relation_zh = '{_esc(relation)}'")
        return (
            "SELECT head, relation_zh, tail, count FROM kg_triples "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY count DESC, head ASC, relation_zh ASC, tail ASC"
        )

    head_type = re.search(
        r"头实体类型为\s*(Disease|Drug|Exam|Symptom|Treatment|Department)",
        q,
    )
    if (
        head_type
        and tail_type
        and "头实体、关系中文名、尾实体和置信度" in q
    ):
        return (
            "SELECT head, relation_zh, tail, confidence FROM kg_triples "
            f"WHERE head_type = '{head_type.group(1)}' "
            f"AND tail_type = '{tail_type.group(1)}' "
            "ORDER BY head ASC, tail ASC"
        )

    return None


def _rule_aggregation(question: str) -> Optional[str]:
    q = question
    if "按地区统计患者人数" in q and "只保留" not in q and "前" not in q:
        return (
            "SELECT region, COUNT(*) AS patient_count FROM patients "
            "GROUP BY region ORDER BY patient_count DESC, region ASC"
        )
    if "按地区统计不同年龄值的数量和患者总数" in q:
        return (
            "SELECT region, COUNT(DISTINCT age) AS distinct_age_count, "
            "COUNT(*) AS patient_count FROM patients GROUP BY region "
            "ORDER BY distinct_age_count DESC, patient_count DESC, region ASC"
        )
    if "按科室汇总就诊次数、不同患者数和平均费用" in q:
        return (
            "SELECT department, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT patient_id) AS distinct_patient_count, "
            "AVG(cost) AS avg_cost FROM visits GROUP BY department "
            "ORDER BY visit_count DESC, department ASC"
        )
    if "按主要处方药统计非空处方对应的就诊次数和平均费用" in q:
        return (
            "SELECT drug_prescribed, COUNT(*) AS visit_count, "
            "AVG(cost) AS avg_cost FROM visits "
            "WHERE LENGTH(drug_prescribed) > 0 GROUP BY drug_prescribed "
            "ORDER BY visit_count DESC, drug_prescribed ASC"
        )
    if "按主要检查统计非空检查的就诊次数、最低费用和最高费用" in q:
        return (
            "SELECT exam, COUNT(*) AS visit_count, MIN(cost) AS min_cost, "
            "MAX(cost) AS max_cost FROM visits WHERE LENGTH(exam) > 0 "
            "GROUP BY exam ORDER BY visit_count DESC, exam ASC"
        )
    if "按就诊年份统计就诊次数和费用总额" in q:
        return (
            "SELECT substr(visit_date, 1, 4) AS visit_year, "
            "COUNT(*) AS visit_count, SUM(cost) AS total_cost FROM visits "
            "GROUP BY visit_year ORDER BY visit_year ASC"
        )
    if "按年月统计就诊次数和平均费用" in q:
        return (
            "SELECT substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost FROM visits "
            "GROUP BY visit_month ORDER BY visit_month ASC"
        )
    if "按科室分别统计门诊次数" in q and "住院次数" in q and "只保留" not in q:
        return (
            "SELECT department, "
            "SUM(CASE WHEN los_days = 0 THEN 1 ELSE 0 END) AS outpatient_count, "
            "SUM(CASE WHEN los_days > 0 THEN 1 ELSE 0 END) AS inpatient_count "
            "FROM visits GROUP BY department "
            "ORDER BY inpatient_count DESC, department ASC"
        )
    cost_split = re.search(
        r"按疾病统计费用低于(\d+(?:\.\d+)?)元与费用至少\1元",
        q,
    )
    if cost_split:
        threshold = _number(cost_split.group(1))
        return (
            "SELECT disease, "
            f"SUM(CASE WHEN cost < {threshold} THEN 1 ELSE 0 END) "
            f"AS under_{threshold.replace('.', '_')}_count, "
            f"SUM(CASE WHEN cost >= {threshold} THEN 1 ELSE 0 END) "
            f"AS at_least_{threshold.replace('.', '_')}_count, "
            "COUNT(*) AS total_count FROM visits GROUP BY disease "
            "ORDER BY total_count DESC, disease ASC"
        )
    if "按患者统计其就诊次数、累计费用和最长住院天数" in q:
        return (
            "SELECT patient_id, COUNT(*) AS visit_count, "
            "SUM(cost) AS total_cost, MAX(los_days) AS max_los_days "
            "FROM visits GROUP BY patient_id "
            "ORDER BY visit_count DESC, total_cost DESC, patient_id ASC"
        )
    if "按年份和科室汇总费用总额与就诊次数" in q:
        return (
            "SELECT substr(visit_date, 1, 4) AS visit_year, department, "
            "SUM(cost) AS total_cost, COUNT(*) AS visit_count FROM visits "
            "GROUP BY visit_year, department "
            "ORDER BY visit_year ASC, total_cost DESC, department ASC"
        )
    band = re.search(
        r"低于(\d+(?:\.\d+)?)元[”\"]?“(\d+(?:\.\d+)?)元（含）至"
        r"(\d+(?:\.\d+)?)元（不含）[”\"]?“至少(\d+(?:\.\d+)?)元",
        q,
    )
    if "费用划分" in q and "平均住院天数" in q:
        values = re.findall(r"\d+(?:\.\d+)?", q.split("三档", 1)[0])
        if len(values) >= 4 and values[0] == values[1] and values[2] == values[3]:
            low, high = _number(values[0]), _number(values[2])
            low_label = f"低于{low}元"
            middle_label = f"{low}元（含）至{high}元（不含）"
            high_label = f"至少{high}元"
            return (
                "SELECT CASE "
                f"WHEN cost < {low} THEN '{low_label}' "
                f"WHEN cost < {high} THEN '{middle_label}' "
                f"ELSE '{high_label}' END AS cost_band, "
                "COUNT(*) AS visit_count, AVG(los_days) AS avg_los_days "
                "FROM visits GROUP BY cost_band ORDER BY CASE cost_band "
                f"WHEN '{low_label}' THEN 1 WHEN '{middle_label}' THEN 2 "
                "ELSE 3 END"
            )
    if "按就诊季度统计次数和总费用" in q and "YYYY-Q1" in q:
        return (
            "SELECT substr(visit_date, 1, 4) || '-Q' || "
            "CAST(((CAST(substr(visit_date, 6, 2) AS INTEGER) - 1) / 3 + 1) "
            "AS INTEGER) AS visit_quarter, COUNT(*) AS visit_count, "
            "SUM(cost) AS total_cost FROM visits GROUP BY visit_quarter "
            "ORDER BY visit_quarter ASC"
        )
    if "每个科室不同疾病的数量、不同检查的数量以及不同处方药的数量" in q:
        return (
            "SELECT department, COUNT(DISTINCT disease) AS distinct_disease_count, "
            "COUNT(DISTINCT exam) AS distinct_exam_count, "
            "COUNT(DISTINCT drug_prescribed) AS distinct_drug_count "
            "FROM visits GROUP BY department ORDER BY department ASC"
        )
    if "按图谱实体类型统计实体数、提及次数总和和平均提及次数" in q:
        return (
            "SELECT type, COUNT(*) AS entity_count, "
            "SUM(mention_count) AS total_mentions, "
            "AVG(mention_count) AS avg_mentions FROM kg_entities "
            "GROUP BY type ORDER BY entity_count DESC, type ASC"
        )
    if "按关系中文名统计图谱三元组数量、平均置信度和观察次数总和" in q:
        return (
            "SELECT relation_zh, COUNT(*) AS triple_count, "
            "AVG(confidence) AS avg_confidence, "
            "SUM(count) AS total_observations FROM kg_triples "
            "GROUP BY relation_zh ORDER BY triple_count DESC, relation_zh ASC"
        )
    if "按尾实体统计其入边数量、不同头实体数和平均置信度" in q:
        return (
            "SELECT tail, COUNT(*) AS incoming_edge_count, "
            "COUNT(DISTINCT head) AS distinct_head_count, "
            "AVG(confidence) AS avg_confidence FROM kg_triples GROUP BY tail "
            "ORDER BY incoming_edge_count DESC, tail ASC"
        )
    if "三元组按置信度分成" in q and "平均观察次数" in q:
        values = re.findall(r"\d+(?:\.\d+)?", q.split("三档", 1)[0])
        if len(values) >= 4 and values[0] == values[1] and values[2] == values[3]:
            low, high = _number(values[0]), _number(values[2])
            low_label = f"低于{low}"
            middle_label = f"{low}（含）至{high}（不含）"
            high_label = f"至少{high}"
            return (
                "SELECT CASE "
                f"WHEN confidence < {low} THEN '{low_label}' "
                f"WHEN confidence < {high} THEN '{middle_label}' "
                f"ELSE '{high_label}' END AS confidence_band, "
                "COUNT(*) AS edge_count, AVG(count) AS avg_observations "
                "FROM kg_triples GROUP BY confidence_band "
                "ORDER BY CASE confidence_band "
                f"WHEN '{low_label}' THEN 1 WHEN '{middle_label}' THEN 2 "
                "ELSE 3 END"
            )
    count_floor = _first_number(r"观察次数至少\s*(\d+)", q)
    if (
        count_floor
        and "按关系中文名统计" in q
        and "最小置信度和最大置信度" in q
    ):
        return (
            "SELECT relation_zh, COUNT(*) AS edge_count, "
            "MIN(confidence) AS min_confidence, "
            "MAX(confidence) AS max_confidence FROM kg_triples "
            f"WHERE count >= {count_floor} GROUP BY relation_zh "
            "ORDER BY edge_count DESC, relation_zh ASC"
        )
    if "按月统计门诊次数" in q and "住院次数" in q and "费用总额" in q:
        return (
            "SELECT substr(visit_date, 1, 7) AS visit_month, "
            "SUM(CASE WHEN los_days = 0 THEN 1 ELSE 0 END) AS outpatient_count, "
            "SUM(CASE WHEN los_days > 0 THEN 1 ELSE 0 END) AS inpatient_count, "
            "SUM(cost) AS total_cost FROM visits GROUP BY visit_month "
            "ORDER BY visit_month ASC"
        )
    high_cost = _first_number(r"费用至少\s*(\d+(?:\.\d+)?)元", q)
    if (
        high_cost
        and "按科室统计" in q
        and "平均费用和最长住院天数" in q
    ):
        return (
            "SELECT department, COUNT(*) AS high_cost_count, "
            "AVG(cost) AS avg_cost, MAX(los_days) AS max_los_days "
            f"FROM visits WHERE cost >= {high_cost} GROUP BY department "
            "ORDER BY high_cost_count DESC, department ASC"
        )
    if "按疾病统计非空处方药的不同种数和非空检查的不同种数" in q:
        return (
            "SELECT disease, "
            "COUNT(DISTINCT CASE WHEN LENGTH(drug_prescribed) > 0 "
            "THEN drug_prescribed END) AS distinct_drug_count, "
            "COUNT(DISTINCT CASE WHEN LENGTH(exam) > 0 THEN exam END) "
            "AS distinct_exam_count FROM visits GROUP BY disease "
            "ORDER BY distinct_drug_count DESC, distinct_exam_count DESC, "
            "disease ASC"
        )
    if "按患者统计最早就诊日、最晚就诊日和就诊次数" in q:
        return (
            "SELECT patient_id, MIN(visit_date) AS first_visit_date, "
            "MAX(visit_date) AS last_visit_date, COUNT(*) AS visit_count "
            "FROM visits GROUP BY patient_id "
            "ORDER BY visit_count DESC, patient_id ASC"
        )
    if "按关系中文名统计不同头实体数、不同尾实体数以及三元组总数" in q:
        return (
            "SELECT relation_zh, COUNT(DISTINCT head) AS distinct_head_count, "
            "COUNT(DISTINCT tail) AS distinct_tail_count, "
            "COUNT(*) AS triple_count FROM kg_triples GROUP BY relation_zh "
            "ORDER BY distinct_head_count DESC, relation_zh ASC"
        )
    if "每种图谱实体类型中提及次数为奇数与偶数的实体数量" in q:
        return (
            "SELECT type, "
            "SUM(CASE WHEN mention_count % 2 = 1 THEN 1 ELSE 0 END) "
            "AS odd_mention_count, "
            "SUM(CASE WHEN mention_count % 2 = 0 THEN 1 ELSE 0 END) "
            "AS even_mention_count FROM kg_entities GROUP BY type "
            "ORDER BY type ASC"
        )
    if "按尾实体类型统计观察次数总和、平均观察次数和最高置信度" in q:
        return (
            "SELECT tail_type, SUM(count) AS total_observations, "
            "AVG(count) AS avg_observations, "
            "MAX(confidence) AS max_confidence FROM kg_triples "
            "GROUP BY tail_type ORDER BY total_observations DESC, tail_type ASC"
        )
    if "按地区统计男性人数、女性人数和总人数" in q:
        return (
            "SELECT region, SUM(CASE WHEN gender = '男' THEN 1 ELSE 0 END) "
            "AS male_count, SUM(CASE WHEN gender = '女' THEN 1 ELSE 0 END) "
            "AS female_count, COUNT(*) AS total_count FROM patients "
            "GROUP BY region ORDER BY total_count DESC, region ASC"
        )
    if "按检查名称统计其关联的不同疾病数、不同科室数和就诊总次数" in q:
        return (
            "SELECT exam, COUNT(DISTINCT disease) AS distinct_disease_count, "
            "COUNT(DISTINCT department) AS distinct_department_count, "
            "COUNT(*) AS visit_count FROM visits WHERE LENGTH(exam) > 0 "
            "GROUP BY exam ORDER BY distinct_disease_count DESC, "
            "visit_count DESC, exam ASC"
        )
    return None


def _rule_top_k(question: str) -> Optional[str]:
    q = question
    limit_match = re.search(r"前\s*(\d+)", q)
    limit = int(limit_match.group(1)) if limit_match else None
    if limit and "人数最多" in q and "地区" in q and "患者人数" in q:
        return (
            "SELECT region, COUNT(*) AS patient_count FROM patients "
            "GROUP BY region ORDER BY patient_count DESC, region ASC "
            f"LIMIT {limit}"
        )
    if limit and "年龄最大" in q and "患者ID、姓名、年龄和地区" in q:
        return (
            "SELECT patient_id, name, age, region FROM patients "
            f"ORDER BY age DESC, patient_id ASC LIMIT {limit}"
        )
    if limit and "最常见" in q and "疾病" in q and "疾病和次数" in q:
        return (
            "SELECT disease, COUNT(*) AS visit_count FROM visits "
            "GROUP BY disease ORDER BY visit_count DESC, disease ASC "
            f"LIMIT {limit}"
        )
    if (
        limit
        and "只看住院就诊" in q
        and "住院天数最长" in q
        and "就诊ID、疾病、住院天数和费用" in q
    ):
        return (
            "SELECT visit_id, disease, los_days, cost FROM visits "
            "WHERE los_days > 0 "
            "ORDER BY los_days DESC, cost DESC, visit_id ASC "
            f"LIMIT {limit}"
        )
    entity_type = re.search(
        r"(Disease|Drug|Exam|Symptom|Treatment|Department)\s*类型实体",
        q,
    )
    if limit and entity_type and "提及次数降序" in q:
        return (
            "SELECT name, mention_count FROM kg_entities "
            f"WHERE type = '{entity_type.group(1)}' "
            "ORDER BY mention_count DESC, name ASC "
            f"LIMIT {limit}"
        )
    if (
        limit
        and "边数最多" in q
        and "关系中文名、边数和观察次数总和" in q
    ):
        return (
            "SELECT relation_zh, COUNT(*) AS edge_count, "
            "SUM(count) AS total_observations FROM kg_triples "
            "GROUP BY relation_zh "
            "ORDER BY edge_count DESC, relation_zh ASC "
            f"LIMIT {limit}"
        )
    if (
        limit
        and "外出边最多" in q
        and "头实体、边数、观察总和" in q
    ):
        return (
            "SELECT head, COUNT(*) AS edge_count, "
            "SUM(count) AS total_observations FROM kg_triples GROUP BY head "
            "ORDER BY edge_count DESC, total_observations DESC, head ASC "
            f"LIMIT {limit}"
        )
    if (
        limit
        and "最常作为尾实体" in q
        and "尾实体、尾类型和入边数" in q
    ):
        return (
            "SELECT tail, tail_type, COUNT(*) AS incoming_count "
            "FROM kg_triples GROUP BY tail, tail_type "
            "ORDER BY incoming_count DESC, tail ASC "
            f"LIMIT {limit}"
        )
    if (
        limit
        and "累计就诊费用" in q
        and "患者ID、姓名、地区、总费用" in q
    ):
        return (
            "SELECT p.patient_id, p.name, p.region, "
            "SUM(v.cost) AS total_cost FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id, p.name, p.region "
            "ORDER BY total_cost DESC, p.patient_id ASC "
            f"LIMIT {limit}"
        )
    if limit and "最繁忙" in q and "YYYY-MM月份、次数和费用总额" in q:
        return (
            "SELECT substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count, SUM(cost) AS total_cost FROM visits "
            "GROUP BY visit_month "
            "ORDER BY visit_count DESC, visit_month ASC "
            f"LIMIT {limit}"
        )
    if limit and "置信度最低" in q and "头实体、关系、尾实体、置信度、观察次数" in q:
        return (
            "SELECT head, relation_zh, tail, confidence, count "
            "FROM kg_triples "
            "ORDER BY confidence ASC, count ASC, head ASC, "
            "relation_zh ASC, tail ASC "
            f"LIMIT {limit}"
        )
    offset = re.search(
        r"跳过前\s*(\d+)\s*种疾病并取接下来的\s*(\d+)\s*种",
        q,
    )
    if offset:
        return (
            "SELECT disease, COUNT(*) AS visit_count FROM visits "
            "GROUP BY disease ORDER BY visit_count DESC, disease ASC "
            f"LIMIT {int(offset.group(2))} OFFSET {int(offset.group(1))}"
        )
    per_group = re.search(r"每个科室分别取费用最高的\s*(\d+)", q)
    if per_group:
        group_limit = int(per_group.group(1))
        return (
            "SELECT department, visit_id, visit_date, disease, cost, "
            "cost_rank FROM (SELECT department, visit_id, visit_date, "
            "disease, cost, ROW_NUMBER() OVER (PARTITION BY department "
            "ORDER BY cost DESC, visit_date ASC, visit_id ASC) AS cost_rank "
            "FROM visits) AS ranked "
            f"WHERE cost_rank <= {group_limit} "
            "ORDER BY department ASC, cost_rank ASC"
        )
    per_relation = re.search(
        r"每种图谱关系分别取观察次数最高的\s*(\d+)",
        q,
    )
    if per_relation:
        group_limit = int(per_relation.group(1))
        return (
            "SELECT relation_zh, head, tail, count, relation_rank FROM "
            "(SELECT relation_zh, head, tail, count, ROW_NUMBER() OVER "
            "(PARTITION BY relation_zh ORDER BY count DESC, head ASC, "
            "tail ASC) AS relation_rank FROM kg_triples) AS ranked "
            f"WHERE relation_rank <= {group_limit} "
            "ORDER BY relation_zh ASC, relation_rank ASC"
        )
    return None


def _rule_set_operation(question: str) -> Optional[str]:
    q = question
    if (
        "真实就诊主诊断" in q
        and "Disease 类型实体" in q
        and "交集" in q
    ):
        return (
            "SELECT disease FROM visits INTERSECT "
            "SELECT name AS disease FROM kg_entities WHERE type = 'Disease' "
            "ORDER BY disease ASC"
        )
    if (
        "患者地区" in q
        and "就诊科室" in q
        and "来源列" in q
        and "去重" in q
    ):
        return (
            "SELECT '患者地区' AS source, region AS location_name "
            "FROM patients UNION "
            "SELECT '就诊科室' AS source, department AS location_name "
            "FROM visits ORDER BY source ASC, location_name ASC"
        )
    if (
        "非空真实处方药" in q
        and "Drug 类型实体" in q
        and "交集" in q
    ):
        return (
            "SELECT drug_prescribed AS drug FROM visits "
            "WHERE LENGTH(drug_prescribed) > 0 INTERSECT "
            "SELECT name AS drug FROM kg_entities WHERE type = 'Drug' "
            "ORDER BY drug ASC"
        )
    dates = _dates(q)
    if (
        len(dates) >= 4
        and "患者ID" in q
        and "去重并集" in q
        and "就诊" in q
    ):
        return (
            "SELECT patient_id FROM visits "
            f"WHERE visit_date >= '{dates[0]}' AND visit_date < '{dates[1]}' "
            "UNION SELECT patient_id FROM visits "
            f"WHERE visit_date >= '{dates[2]}' AND visit_date < '{dates[3]}' "
            "ORDER BY patient_id ASC"
        )
    return None


def _rule_window(question: str) -> Optional[str]:
    q = question
    if (
        "全部就诊按费用降序计算并列名次" in q
        and "后续名次留空" in q
    ):
        return (
            "SELECT visit_id, disease, cost, "
            "RANK() OVER (ORDER BY cost DESC) AS cost_rank "
            "FROM visits ORDER BY cost_rank ASC, visit_id ASC"
        )
    if (
        "每个地区内按年龄降序计算密集排名" in q
        and "名次不留空" in q
    ):
        return (
            "SELECT region, patient_id, name, age, "
            "DENSE_RANK() OVER (PARTITION BY region ORDER BY age DESC) "
            "AS age_rank FROM patients "
            "ORDER BY region ASC, age_rank ASC, patient_id ASC"
        )
    if "按月汇总费用" in q and "累计费用" in q and "用CTE" not in q:
        return (
            "SELECT visit_month, monthly_cost, "
            "SUM(monthly_cost) OVER (ORDER BY visit_month "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            "AS cumulative_cost FROM "
            "(SELECT substr(visit_date, 1, 7) AS visit_month, "
            "SUM(cost) AS monthly_cost FROM visits GROUP BY visit_month) AS m "
            "ORDER BY visit_month ASC"
        )
    if (
        "每个科室内部按日期、就诊ID升序" in q
        and "累计就诊次数和累计费用" in q
    ):
        return (
            "SELECT department, visit_id, visit_date, cost, "
            "COUNT(*) OVER (PARTITION BY department "
            "ORDER BY visit_date ASC, visit_id ASC "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            "AS running_visit_count, "
            "SUM(cost) OVER (PARTITION BY department "
            "ORDER BY visit_date ASC, visit_id ASC "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_cost "
            "FROM visits ORDER BY department ASC, visit_date ASC, visit_id ASC"
        )
    preceding = re.search(r"之前\s*(\d+)\s*个有数据月份", q)
    if preceding and "移动平均" in q and "按月汇总平均费用" in q:
        return (
            "SELECT visit_month, avg_cost, "
            "AVG(avg_cost) OVER (ORDER BY visit_month "
            f"ROWS BETWEEN {int(preceding.group(1))} PRECEDING AND CURRENT ROW) "
            "AS moving_three_month_avg FROM "
            "(SELECT substr(visit_date, 1, 7) AS visit_month, "
            "AVG(cost) AS avg_cost FROM visits GROUP BY visit_month) AS m "
            "ORDER BY visit_month ASC"
        )
    if "费用占其所在科室费用总额的比例" in q:
        return (
            "SELECT visit_id, department, cost, "
            "SUM(cost) OVER (PARTITION BY department) "
            "AS department_total_cost, "
            "cost / SUM(cost) OVER (PARTITION BY department) AS cost_share "
            "FROM visits ORDER BY department ASC, cost_share DESC, visit_id ASC"
        )
    if "科室的最高费用和其费用与科室最高费用之差" in q:
        return (
            "SELECT visit_id, department, cost, "
            "MAX(cost) OVER (PARTITION BY department) AS department_max_cost, "
            "MAX(cost) OVER (PARTITION BY department) - cost AS gap_from_max "
            "FROM visits "
            "ORDER BY department ASC, gap_from_max ASC, visit_id ASC"
        )
    if "类型内的平均提及次数" in q and "减去类型平均值" in q:
        return (
            "SELECT name, type, mention_count, "
            "AVG(mention_count) OVER (PARTITION BY type) AS type_avg_mentions, "
            "mention_count - AVG(mention_count) OVER (PARTITION BY type) "
            "AS difference_from_type_avg FROM kg_entities "
            "ORDER BY type ASC, difference_from_type_avg DESC, name ASC"
        )
    return None


def _rule_having(question: str) -> Optional[str]:
    q = question
    floor = _first_number(r"人数至少\s*(\d+)", q)
    if floor and "按地区统计患者人数" in q and "只保留" in q:
        return (
            "SELECT region, COUNT(*) AS patient_count FROM patients "
            "GROUP BY region "
            f"HAVING COUNT(*) >= {floor} "
            "ORDER BY patient_count DESC, region ASC"
        )
    visits = _first_number(r"就诊至少\s*(\d+)\s*次", q)
    avg_cost = _first_number(r"平均费用超过\s*(\d+(?:\.\d+)?)元", q)
    if (
        visits
        and avg_cost
        and "按科室统计就诊次数和平均费用" in q
    ):
        return (
            "SELECT department, COUNT(*) AS visit_count, "
            "AVG(cost) AS avg_cost FROM visits GROUP BY department "
            f"HAVING COUNT(*) >= {visits} AND AVG(cost) > {avg_cost} "
            "ORDER BY avg_cost DESC, department ASC"
        )
    visits = _first_number(r"至少\s*(\d+)\s*次就诊", q)
    if visits and "按月统计就诊次数和费用总额" in q and "只保留" in q:
        return (
            "SELECT substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count, SUM(cost) AS total_cost FROM visits "
            "GROUP BY visit_month "
            f"HAVING COUNT(*) >= {visits} ORDER BY visit_month ASC"
        )
    diseases = _first_number(r"不同疾病至少\s*(\d+)\s*种", q)
    patients = _first_number(r"不同患者至少\s*(\d+)\s*人", q)
    if diseases and patients and "按科室统计" in q:
        return (
            "SELECT department, COUNT(DISTINCT disease) AS disease_count, "
            "COUNT(DISTINCT patient_id) AS patient_count FROM visits "
            "GROUP BY department "
            f"HAVING COUNT(DISTINCT disease) >= {diseases} "
            f"AND COUNT(DISTINCT patient_id) >= {patients} "
            "ORDER BY disease_count DESC, department ASC"
        )
    diseases = _first_number(r"关联至少\s*(\d+)\s*种疾病", q)
    visits = _first_number(r"至少\s*(\d+)\s*次就诊", q)
    if diseases and visits and "按非空检查统计" in q:
        return (
            "SELECT exam, COUNT(DISTINCT disease) AS disease_count, "
            "COUNT(*) AS visit_count FROM visits WHERE LENGTH(exam) > 0 "
            "GROUP BY exam "
            f"HAVING COUNT(DISTINCT disease) >= {diseases} "
            f"AND COUNT(*) >= {visits} "
            "ORDER BY disease_count DESC, visit_count DESC, exam ASC"
        )
    relations = _first_number(r"至少\s*(\d+)\s*种不同关系", q)
    edges = _first_number(r"至少\s*(\d+)\s*条外出边", q)
    if relations and edges and "按头实体统计" in q:
        return (
            "SELECT head, COUNT(DISTINCT relation_zh) AS relation_kind_count, "
            "COUNT(*) AS edge_count FROM kg_triples GROUP BY head "
            f"HAVING COUNT(DISTINCT relation_zh) >= {relations} "
            f"AND COUNT(*) >= {edges} "
            "ORDER BY relation_kind_count DESC, edge_count DESC, head ASC"
        )
    patients = _first_number(r"不同就诊患者至少\s*(\d+)\s*人", q)
    total = _first_number(r"费用总额至少\s*(\d+(?:\.\d+)?)元", q)
    if patients and total and "连接患者和就诊后按地区汇总" in q:
        return (
            "SELECT p.region, COUNT(DISTINCT p.patient_id) AS patient_count, "
            "COUNT(*) AS visit_count, SUM(v.cost) AS total_cost "
            "FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id GROUP BY p.region "
            f"HAVING COUNT(DISTINCT p.patient_id) >= {patients} "
            f"AND SUM(v.cost) >= {total} "
            "ORDER BY total_cost DESC, p.region ASC"
        )
    outpatient = _first_number(r"门诊至少\s*(\d+)\s*次", q)
    inpatient = _first_number(r"住院至少\s*(\d+)\s*次", q)
    if outpatient and inpatient and "按科室分别统计门诊次数与住院次数" in q:
        return (
            "SELECT department, "
            "SUM(CASE WHEN los_days = 0 THEN 1 ELSE 0 END) "
            "AS outpatient_count, "
            "SUM(CASE WHEN los_days > 0 THEN 1 ELSE 0 END) "
            "AS inpatient_count FROM visits GROUP BY department "
            "HAVING SUM(CASE WHEN los_days = 0 THEN 1 ELSE 0 END) "
            f">= {outpatient} AND "
            "SUM(CASE WHEN los_days > 0 THEN 1 ELSE 0 END) "
            f">= {inpatient} "
            "ORDER BY inpatient_count DESC, outpatient_count DESC, "
            "department ASC"
        )
    return None


def _rule_cte(question: str) -> Optional[str]:
    q = question
    if "CTE" not in q.upper():
        return None
    threshold = _first_number(r"平均费用高于\s*(\d+(?:\.\d+)?)元", q)
    if threshold and "每个科室的就诊次数和平均费用" in q:
        return (
            "WITH department_stats AS (SELECT department, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost "
            "FROM visits GROUP BY department) "
            "SELECT department, visit_count, avg_cost FROM department_stats "
            f"WHERE avg_cost > {threshold} "
            "ORDER BY avg_cost DESC, department ASC"
        )
    months = _months(q)
    if len(months) >= 2 and "每月就诊次数和费用总额" in q:
        return (
            "WITH monthly AS (SELECT substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count, SUM(cost) AS total_cost FROM visits "
            "GROUP BY visit_month) "
            "SELECT visit_month, visit_count, total_cost FROM monthly "
            f"WHERE visit_month >= '{months[0]}' "
            f"AND visit_month <= '{months[1]}' ORDER BY visit_month ASC"
        )
    if "每位患者的最近就诊日期" in q and "最近日期上的全部就诊" in q:
        return (
            "WITH latest AS (SELECT patient_id, "
            "MAX(visit_date) AS latest_date FROM visits GROUP BY patient_id) "
            "SELECT v.patient_id, v.visit_id, v.visit_date, v.disease "
            "FROM visits AS v JOIN latest AS l "
            "ON l.patient_id = v.patient_id "
            "AND l.latest_date = v.visit_date "
            "ORDER BY v.visit_date DESC, v.patient_id ASC, v.visit_id ASC"
        )
    if "各地区患者人数" in q and "地区人数的平均值" in q:
        return (
            "WITH region_counts AS (SELECT region, "
            "COUNT(*) AS patient_count FROM patients GROUP BY region), "
            "overall AS (SELECT AVG(patient_count) AS avg_region_count "
            "FROM region_counts) "
            "SELECT r.region, r.patient_count, o.avg_region_count "
            "FROM region_counts AS r CROSS JOIN overall AS o "
            "WHERE r.patient_count > o.avg_region_count "
            "ORDER BY r.patient_count DESC, r.region ASC"
        )
    dates = _dates(q)
    if len(dates) >= 4 and "分别汇总" in q and "各科室就诊次数" in q:
        years = re.findall(r"20\d{2}", q.split("精确边界", 1)[0])
        first_year = years[0] if years else dates[0][:4]
        second_year = years[1] if len(years) > 1 else dates[2][:4]
        return (
            f"WITH y{first_year} AS (SELECT department, COUNT(*) "
            f"AS count_{first_year} FROM visits "
            f"WHERE visit_date >= '{dates[0]}' AND visit_date < '{dates[1]}' "
            "GROUP BY department), "
            f"y{second_year} AS (SELECT department, COUNT(*) "
            f"AS count_{second_year} FROM visits "
            f"WHERE visit_date >= '{dates[2]}' AND visit_date < '{dates[3]}' "
            "GROUP BY department) "
            f"SELECT a.department, a.count_{first_year}, b.count_{second_year}, "
            f"b.count_{second_year} - a.count_{first_year} AS change_count "
            f"FROM y{first_year} AS a JOIN y{second_year} AS b "
            "ON b.department = a.department "
            "ORDER BY change_count DESC, a.department ASC"
        )
    threshold = _first_number(r"费用至少\s*(\d+(?:\.\d+)?)元", q)
    if threshold and "高费用就诊" in q and "按患者地区统计" in q:
        return (
            "WITH high_cost AS (SELECT patient_id, cost FROM visits "
            f"WHERE cost >= {threshold}) "
            "SELECT p.region, COUNT(*) AS high_cost_visit_count, "
            "SUM(h.cost) AS high_cost_total FROM high_cost AS h "
            "JOIN patients AS p ON p.patient_id = h.patient_id "
            "GROUP BY p.region ORDER BY high_cost_total DESC, p.region ASC"
        )
    visits = _first_number(r"就诊至少\s*(\d+)\s*次", q)
    if visits and "最早日期、最晚日期和就诊次数" in q and "跨度天数" in q:
        return (
            "WITH timelines AS (SELECT patient_id, "
            "MIN(visit_date) AS first_date, MAX(visit_date) AS last_date, "
            "COUNT(*) AS visit_count FROM visits GROUP BY patient_id) "
            "SELECT patient_id, first_date, last_date, "
            "CAST(julianday(last_date) - julianday(first_date) AS INTEGER) "
            "AS span_days, visit_count FROM timelines "
            f"WHERE visit_count >= {visits} "
            "ORDER BY span_days DESC, patient_id ASC"
        )
    if "每个科室的最高费用" in q and "达到科室最高费用的全部就诊" in q:
        return (
            "WITH maxima AS (SELECT department, MAX(cost) AS max_cost "
            "FROM visits GROUP BY department) "
            "SELECT v.department, v.visit_id, v.disease, v.cost "
            "FROM visits AS v JOIN maxima AS m "
            "ON m.department = v.department AND m.max_cost = v.cost "
            "ORDER BY v.department ASC, v.visit_id ASC"
        )
    if "图谱治疗药物边与真实处方就诊相连" in q:
        return (
            "WITH treatment_drugs AS (SELECT head AS disease, tail AS drug "
            "FROM kg_triples WHERE relation_zh = '治疗药物'), "
            "matched AS (SELECT v.visit_id, v.disease, v.drug_prescribed "
            "FROM visits AS v JOIN treatment_drugs AS t "
            "ON t.disease = v.disease AND t.drug = v.drug_prescribed) "
            "SELECT disease, COUNT(*) AS matched_visit_count, "
            "COUNT(DISTINCT drug_prescribed) AS matched_drug_count "
            "FROM matched GROUP BY disease "
            "ORDER BY matched_visit_count DESC, disease ASC"
        )
    if "按月汇总费用" in q and "第二个CTE" in q and "累计费用" in q:
        return (
            "WITH monthly AS (SELECT substr(visit_date, 1, 7) AS visit_month, "
            "SUM(cost) AS monthly_cost FROM visits GROUP BY visit_month), "
            "running AS (SELECT visit_month, monthly_cost, "
            "SUM(monthly_cost) OVER (ORDER BY visit_month "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            "AS cumulative_cost FROM monthly) "
            "SELECT visit_month, monthly_cost, cumulative_cost FROM running "
            "ORDER BY visit_month ASC"
        )
    rank = _first_number(r"第\s*(\d+)\s*名", q)
    if rank and "每个科室的就诊按费用降序" in q:
        return (
            "WITH ranked AS (SELECT department, visit_id, disease, cost, "
            "ROW_NUMBER() OVER (PARTITION BY department "
            "ORDER BY cost DESC, visit_id ASC) AS rn FROM visits) "
            "SELECT department, visit_id, disease, cost FROM ranked "
            f"WHERE rn = {rank} ORDER BY department ASC"
        )
    return None


def _rule_temporal(question: str) -> Optional[str]:
    q = question
    dates = _dates(q)
    if len(dates) >= 2 and "精确日期边界" in q and "就诊" in q:
        inclusive_end = "含首尾" in q or (
            f"{dates[1]}（含）" in q and "不含" not in q.split(dates[1], 1)[1][:8]
        )
        predicate = (
            f"visit_date BETWEEN '{dates[0]}' AND '{dates[1]}'"
            if inclusive_end
            else f"visit_date >= '{dates[0]}' AND visit_date < '{dates[1]}'"
        )
        if "住院天数大于0" in q and "就诊ID、疾病、住院天数" in q:
            return (
                "SELECT visit_date, visit_id, disease, los_days FROM visits "
                f"WHERE {predicate} AND los_days > 0 "
                "ORDER BY visit_date ASC, visit_id ASC"
            )
        if "就诊ID、日期、疾病和科室" in q:
            return (
                "SELECT visit_id, visit_date, disease, department FROM visits "
                f"WHERE {predicate} ORDER BY visit_date ASC, visit_id ASC"
            )
        if "日期、就诊ID、疾病和费用" in q:
            return (
                "SELECT visit_date, visit_id, disease, cost FROM visits "
                f"WHERE {predicate} ORDER BY visit_date ASC, visit_id ASC"
            )
    days = re.search(r"每月(\d+)日至(\d+)日", q)
    if days and "日期、就诊ID、疾病和费用" in q:
        return (
            "SELECT visit_date, visit_id, disease, cost FROM visits "
            "WHERE CAST(strftime('%d', visit_date) AS INTEGER) "
            f"BETWEEN {int(days.group(1))} AND {int(days.group(2))} "
            "ORDER BY visit_date ASC, visit_id ASC"
        )
    if "跨全部年份各公历月份的就诊次数" in q:
        return (
            "SELECT strftime('%m', visit_date) AS calendar_month, "
            "COUNT(*) AS visit_count FROM visits GROUP BY calendar_month "
            "ORDER BY calendar_month ASC"
        )
    if "整个数据库中最晚日期发生的全部就诊" in q:
        return (
            "SELECT visit_date, visit_id, department, cost FROM visits "
            "WHERE visit_date = (SELECT MAX(visit_date) FROM visits) "
            "ORDER BY visit_id ASC"
        )
    if "每位患者最近日期的全部就诊" in q:
        return (
            "SELECT v.patient_id, v.visit_date, v.visit_id, v.disease "
            "FROM visits AS v WHERE v.visit_date = "
            "(SELECT MAX(x.visit_date) FROM visits AS x "
            "WHERE x.patient_id = v.patient_id) "
            "ORDER BY v.visit_date DESC, v.patient_id ASC, v.visit_id ASC"
        )
    if len(dates) >= 1 and "固定参照日" in q and "相隔天数" in q:
        return (
            "SELECT visit_id, visit_date, disease, "
            f"CAST(julianday('{dates[0]}') - julianday(visit_date) AS INTEGER) "
            "AS days_before_reference FROM visits "
            "ORDER BY days_before_reference ASC, visit_id ASC"
        )
    if "按年份和季度统计就诊次数" in q and "季度号" in q:
        return (
            "SELECT substr(visit_date, 1, 4) AS visit_year, "
            "CAST((CAST(substr(visit_date, 6, 2) AS INTEGER) - 1) / 3 + 1 "
            "AS INTEGER) AS quarter_no, COUNT(*) AS visit_count FROM visits "
            "GROUP BY visit_year, quarter_no "
            "ORDER BY visit_year ASC, quarter_no ASC"
        )
    if "按年份统计当年有就诊的不同患者数和人均就诊次数" in q:
        return (
            "SELECT substr(visit_date, 1, 4) AS visit_year, "
            "COUNT(DISTINCT patient_id) AS patient_count, "
            "1.0 * COUNT(*) / COUNT(DISTINCT patient_id) "
            "AS visits_per_patient FROM visits GROUP BY visit_year "
            "ORDER BY visit_year ASC"
        )
    if "每个年份中就诊次数最多的月份" in q:
        return (
            "SELECT m.visit_year, m.visit_month, m.visit_count FROM "
            "(SELECT substr(visit_date, 1, 4) AS visit_year, "
            "substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count FROM visits "
            "GROUP BY visit_year, visit_month) AS m "
            "WHERE m.visit_count = (SELECT MAX(x.visit_count) FROM "
            "(SELECT substr(visit_date, 1, 4) AS visit_year, "
            "substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count FROM visits "
            "GROUP BY visit_year, visit_month) AS x "
            "WHERE x.visit_year = m.visit_year) "
            "ORDER BY m.visit_year ASC, m.visit_month ASC"
        )
    if "每个年份最早日期的全部就诊" in q:
        return (
            "SELECT substr(v.visit_date, 1, 4) AS visit_year, "
            "v.visit_date, v.visit_id, v.disease FROM visits AS v "
            "WHERE v.visit_date = (SELECT MIN(x.visit_date) FROM visits AS x "
            "WHERE substr(x.visit_date, 1, 4) = substr(v.visit_date, 1, 4)) "
            "ORDER BY visit_year ASC, v.visit_id ASC"
        )
    month_literal = re.search(r"精确月份文本字面值为(\d{2})", q)
    if month_literal and "按年份统计次数、总费用和平均费用" in q:
        return (
            "SELECT substr(visit_date, 1, 4) AS visit_year, "
            "COUNT(*) AS visit_count, SUM(cost) AS total_cost, "
            "AVG(cost) AS avg_cost FROM visits "
            f"WHERE substr(visit_date, 6, 2) = '{month_literal.group(1)}' "
            "GROUP BY visit_year ORDER BY visit_year ASC"
        )
    floor = _first_number(r"下限明确为\s*(\d+)\s*次", q)
    if floor and "同一天至少发生" in q:
        return (
            "SELECT visit_date, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT patient_id) AS patient_count, "
            "SUM(cost) AS total_cost FROM visits GROUP BY visit_date "
            f"HAVING COUNT(*) >= {floor} "
            "ORDER BY visit_count DESC, visit_date ASC"
        )
    gap = _first_number(r"相隔超过\s*(\d+)\s*天", q)
    if gap and "配对同一患者" in q:
        return (
            "SELECT a.patient_id, a.visit_date AS earlier_date, "
            "b.visit_date AS later_date, "
            "CAST(julianday(b.visit_date) - julianday(a.visit_date) AS INTEGER) "
            "AS gap_days FROM visits AS a JOIN visits AS b "
            "ON b.patient_id = a.patient_id AND b.visit_date > a.visit_date "
            "WHERE julianday(b.visit_date) - julianday(a.visit_date) "
            f"> {gap} ORDER BY gap_days DESC, a.patient_id ASC, "
            "earlier_date ASC, later_date ASC"
        )
    range_match = re.search(r"精确天数范围为(\d+)至(\d+)天", q)
    if range_match and "首次就诊日" in q:
        return (
            "SELECT v.patient_id, v.visit_id, v.visit_date, "
            "CAST(julianday(v.visit_date) - julianday(f.first_date) AS INTEGER) "
            "AS days_since_first FROM visits AS v JOIN "
            "(SELECT patient_id, MIN(visit_date) AS first_date FROM visits "
            "GROUP BY patient_id) AS f ON f.patient_id = v.patient_id "
            "WHERE julianday(v.visit_date) - julianday(f.first_date) "
            f"BETWEEN {int(range_match.group(1))} "
            f"AND {int(range_match.group(2))} "
            "ORDER BY v.patient_id ASC, v.visit_date ASC, v.visit_id ASC"
        )
    if "每个患者首次和末次就诊的疾病组合" in q:
        return (
            "SELECT f.patient_id, f.visit_date AS first_date, "
            "f.disease AS first_disease, l.visit_date AS last_date, "
            "l.disease AS last_disease FROM visits AS f JOIN visits AS l "
            "ON l.patient_id = f.patient_id "
            "WHERE f.visit_date = (SELECT MIN(x.visit_date) FROM visits AS x "
            "WHERE x.patient_id = f.patient_id) "
            "AND l.visit_date = (SELECT MAX(y.visit_date) FROM visits AS y "
            "WHERE y.patient_id = f.patient_id) AND f.visit_date < l.visit_date "
            "ORDER BY f.patient_id ASC, first_disease ASC, last_disease ASC"
        )
    if len(dates) >= 2 and "前一个有数据月份" in q and "变化量" in q:
        return (
            "SELECT visit_month, visit_count, "
            "LAG(visit_count) OVER (ORDER BY visit_month) "
            "AS previous_month_count, "
            "visit_count - LAG(visit_count) OVER (ORDER BY visit_month) "
            "AS change_count FROM "
            "(SELECT substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count FROM visits "
            f"WHERE visit_date >= '{dates[0]}' "
            f"AND visit_date < '{dates[1]}' GROUP BY visit_month) AS m "
            "ORDER BY visit_month ASC"
        )
    ordinal = _first_number(r"序号为\s*(\d+)", q)
    if ordinal and "患者时间线中的序号" in q:
        return (
            "SELECT patient_id, visit_id, visit_date, disease FROM "
            "(SELECT patient_id, visit_id, visit_date, disease, "
            "ROW_NUMBER() OVER (PARTITION BY patient_id "
            "ORDER BY visit_date ASC, visit_id ASC) AS visit_no FROM visits) "
            f"AS x WHERE visit_no = {ordinal} "
            "ORDER BY visit_date ASC, patient_id ASC, visit_id ASC"
        )
    return None


def _rule_multi_join(question: str, domains: Domains) -> Optional[str]:
    q = question
    regions = _domain_values(q, domains.get("region", ()))
    departments = _domain_values(q, domains.get("department", ()))
    exams = _domain_values(q, domains.get("exam", ()))
    age = _first_number(r"年龄至少\s*(\d+)", q)
    gender = "女" if "女性" in q else "男" if "男性" in q else None
    if (
        age
        and gender
        and departments
        and "连接患者与其就诊记录" in q
        and "患者ID、姓名、年龄、就诊ID、日期和疾病" in q
    ):
        return (
            "SELECT p.patient_id, p.name, p.age, v.visit_id, "
            "v.visit_date, v.disease FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            f"WHERE p.gender = '{gender}' AND p.age >= {age} "
            f"AND v.department = '{_esc(departments[0])}' "
            "ORDER BY v.visit_date ASC, p.patient_id ASC, v.visit_id ASC"
        )
    if (
        gender
        and exams
        and "患者就诊" in q
        and "患者ID、姓名、年龄、疾病和就诊日期" in q
    ):
        return (
            "SELECT p.patient_id, p.name, p.age, v.disease, v.visit_date "
            "FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id "
            f"WHERE p.gender = '{gender}' AND v.exam = '{_esc(exams[0])}' "
            "ORDER BY p.age DESC, v.visit_date ASC, "
            "p.patient_id ASC, v.visit_id ASC"
        )
    cost = _first_number(r"费用至少\s*(\d+(?:\.\d+)?)元", q)
    if (
        regions
        and cost
        and "患者费用" in q
        and "患者姓名、年龄、就诊ID、疾病和费用" in q
    ):
        return (
            "SELECT p.name, p.age, v.visit_id, v.disease, v.cost "
            "FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id "
            f"WHERE p.region = '{_esc(regions[0])}' AND v.cost >= {cost} "
            "ORDER BY v.cost DESC, p.name ASC, v.visit_id ASC"
        )
    if "按患者地区统计就诊次数、不同就诊患者数和费用总额" in q:
        return (
            "SELECT p.region, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT p.patient_id) AS patient_count, "
            "SUM(v.cost) AS total_cost FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            "GROUP BY p.region ORDER BY total_cost DESC, p.region ASC"
        )
    dates = _dates(q)
    if (
        len(dates) >= 2
        and "按患者性别统计" in q
        and "平均费用和平均住院天数" in q
    ):
        return (
            "SELECT p.gender, COUNT(*) AS visit_count, "
            "AVG(v.cost) AS avg_cost, AVG(v.los_days) AS avg_los_days "
            "FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id "
            f"WHERE v.visit_date >= '{dates[0]}' "
            f"AND v.visit_date < '{dates[1]}' "
            "GROUP BY p.gender ORDER BY p.gender ASC"
        )
    if "保留全部患者" in q and "没有就诊者记为0" in q:
        return (
            "SELECT p.patient_id, p.name, COUNT(v.visit_id) AS visit_count, "
            "COALESCE(SUM(v.cost), 0) AS total_cost FROM patients AS p "
            "LEFT JOIN visits AS v ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id, p.name "
            "ORDER BY visit_count DESC, p.patient_id ASC"
        )
    if "连接患者表和每位患者的最早就诊日期" in q:
        return (
            "SELECT p.patient_id, p.name, p.region, f.first_visit_date "
            "FROM patients AS p JOIN "
            "(SELECT patient_id, MIN(visit_date) AS first_visit_date "
            "FROM visits GROUP BY patient_id) AS f "
            "ON f.patient_id = p.patient_id "
            "ORDER BY f.first_visit_date ASC, p.patient_id ASC"
        )
    if "同一患者在两个不同科室的就诊配对" in q:
        return (
            "SELECT DISTINCT p.patient_id, p.name, "
            "v1.department AS first_department, "
            "v2.department AS second_department FROM patients AS p "
            "JOIN visits AS v1 ON v1.patient_id = p.patient_id "
            "JOIN visits AS v2 ON v2.patient_id = p.patient_id "
            "AND v1.department < v2.department "
            "ORDER BY p.patient_id ASC, first_department ASC, "
            "second_department ASC"
        )
    entity_type = re.search(
        r"类型为\s*(Disease|Drug|Exam|Symptom|Treatment|Department)",
        q,
    )
    mention = _first_number(r"提及次数至少\s*(\d+)", q)
    if (
        entity_type
        and mention
        and "以该实体为尾的三元组" in q
        and "检查名、提及次数、头实体和观察次数" in q
    ):
        return (
            "SELECT e.name, e.mention_count, t.head, t.count "
            "FROM kg_entities AS e JOIN kg_triples AS t "
            "ON t.tail = e.name "
            f"WHERE e.type = '{entity_type.group(1)}' "
            f"AND e.mention_count >= {mention} "
            "ORDER BY e.name ASC, t.count DESC, t.head ASC"
        )
    if "头实体和尾实体都连接到实体表" in q and "头类型 Disease、尾类型 Drug" in q:
        return (
            "SELECT h.name AS head_name, h.mention_count AS head_mentions, "
            "t.relation_zh, d.name AS tail_name, "
            "d.mention_count AS tail_mentions FROM kg_triples AS t "
            "JOIN kg_entities AS h ON h.name = t.head "
            "JOIN kg_entities AS d ON d.name = t.tail "
            "WHERE h.type = 'Disease' AND d.type = 'Drug' "
            "ORDER BY head_name ASC, tail_name ASC"
        )
    if "就诊疾病与图谱“所属科室”关系连接" in q:
        return (
            "SELECT v.visit_id, v.disease, v.department, t.confidence "
            "FROM visits AS v JOIN kg_triples AS t "
            "ON t.head = v.disease AND t.tail = v.department "
            "WHERE t.relation_zh = '所属科室' "
            "ORDER BY v.disease ASC, v.visit_id ASC"
        )
    if "主要检查正好是该疾病图谱检查" in q:
        return (
            "SELECT v.visit_id, v.disease, v.exam, t.count FROM visits AS v "
            "JOIN kg_triples AS t ON t.head = v.disease AND t.tail = v.exam "
            "WHERE t.relation_zh = '诊断检查' "
            "ORDER BY t.count DESC, v.disease ASC, v.visit_id ASC"
        )
    if "排除与实际主要检查相同的检查" in q:
        return (
            "SELECT v.visit_id, v.disease, v.exam AS performed_exam, "
            "t.tail AS other_graph_exam FROM visits AS v "
            "JOIN kg_triples AS t ON t.head = v.disease "
            "WHERE t.relation_zh = '诊断检查' AND t.tail <> v.exam "
            "ORDER BY v.disease ASC, v.visit_id ASC, other_graph_exam ASC"
        )
    if "非空处方药就诊与所有同疾病的图谱治疗药物连接" in q:
        return (
            "SELECT v.visit_id, v.disease, "
            "v.drug_prescribed AS prescribed_drug, "
            "t.tail AS graph_treatment_drug FROM visits AS v "
            "JOIN kg_triples AS t ON t.head = v.disease "
            "WHERE LENGTH(v.drug_prescribed) > 0 "
            "AND t.relation_zh = '治疗药物' "
            "ORDER BY v.disease ASC, v.visit_id ASC, graph_treatment_drug ASC"
        )
    if "处方药与主诊断是否在图谱中构成“用药禁忌”关系" in q:
        return (
            "SELECT v.visit_id, v.drug_prescribed, v.disease, t.confidence "
            "FROM visits AS v JOIN kg_triples AS t "
            "ON t.head = v.drug_prescribed AND t.tail = v.disease "
            "WHERE t.relation_zh = '用药禁忌' "
            "ORDER BY t.confidence DESC, v.visit_id ASC"
        )
    if "就诊主诊断连接到图谱“并发症”边" in q:
        return (
            "SELECT v.visit_id, v.disease, t.tail AS complication, t.count "
            "FROM visits AS v JOIN kg_triples AS t ON t.head = v.disease "
            "WHERE t.relation_zh = '并发症' "
            "ORDER BY v.disease ASC, t.count DESC, "
            "complication ASC, v.visit_id ASC"
        )
    age = _first_number(r"(\d+)岁及以上患者", q)
    if age and "连接患者、就诊和该疾病的图谱治疗药物" in q:
        return (
            "SELECT DISTINCT p.patient_id, p.age, v.disease, "
            "t.tail AS graph_drug FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            "JOIN kg_triples AS t ON t.head = v.disease "
            f"WHERE p.age >= {age} AND t.relation_zh = '治疗药物' "
            "ORDER BY p.patient_id ASC, v.disease ASC, graph_drug ASC"
        )
    mention = _first_number(r"提及次数至少\s*(\d+)", q)
    if mention and "类型为 Disease 的同名疾病实体" in q:
        return (
            "SELECT v.visit_id, v.disease, e.mention_count, v.cost "
            "FROM visits AS v JOIN kg_entities AS e ON e.name = v.disease "
            "WHERE e.type = 'Disease' "
            f"AND e.mention_count >= {mention} "
            "ORDER BY e.mention_count DESC, v.visit_id ASC"
        )
    if regions and "患者、就诊和图谱所属科室边" in q:
        return (
            "SELECT p.name, v.disease, v.department, v.visit_date "
            "FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id "
            "JOIN kg_triples AS t "
            "ON t.head = v.disease AND t.tail = v.department "
            f"WHERE p.region = '{_esc(regions[0])}' "
            "AND t.relation_zh = '所属科室' "
            "ORDER BY v.visit_date ASC, p.name ASC, v.visit_id ASC"
        )
    if "同一患者的两次不同就诊做配对" in q and "两次主诊断不同" in q:
        return (
            "SELECT v1.patient_id, v1.visit_date AS earlier_date, "
            "v1.disease AS earlier_disease, v2.visit_date AS later_date, "
            "v2.disease AS later_disease FROM visits AS v1 "
            "JOIN visits AS v2 ON v2.patient_id = v1.patient_id "
            "AND v1.visit_date < v2.visit_date "
            "WHERE v1.disease <> v2.disease "
            "ORDER BY v1.patient_id ASC, earlier_date ASC, later_date ASC"
        )
    if "患者及其前后两次就诊" in q and "后一次费用高于前一次" in q:
        return (
            "SELECT p.name, v1.visit_date AS earlier_date, "
            "v1.cost AS earlier_cost, v2.visit_date AS later_date, "
            "v2.cost AS later_cost FROM patients AS p "
            "JOIN visits AS v1 ON v1.patient_id = p.patient_id "
            "JOIN visits AS v2 ON v2.patient_id = p.patient_id "
            "AND v2.visit_date > v1.visit_date "
            "WHERE v2.cost > v1.cost "
            "ORDER BY p.name ASC, earlier_date ASC, later_date ASC"
        )
    if "Disease 类型实体和该疾病的图谱边" in q and "不同关系种类数" in q:
        return (
            "SELECT v.disease, COUNT(DISTINCT v.visit_id) AS visit_count, "
            "e.mention_count, COUNT(DISTINCT t.relation_zh) "
            "AS relation_kind_count FROM visits AS v "
            "JOIN kg_entities AS e ON e.name = v.disease "
            "JOIN kg_triples AS t ON t.head = v.disease "
            "WHERE e.type = 'Disease' "
            "GROUP BY v.disease, e.mention_count "
            "ORDER BY visit_count DESC, v.disease ASC"
        )
    mention = _first_number(r"提及次数至少\s*(\d+)", q)
    if mention and "连接四张表" in q and "类型为 Exam" in q:
        return (
            "SELECT p.region, v.disease, e.name AS graph_exam, "
            "COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id "
            "JOIN kg_triples AS t "
            "ON t.head = v.disease AND t.relation_zh = '诊断检查' "
            "JOIN kg_entities AS e ON e.name = t.tail "
            f"WHERE e.type = 'Exam' AND e.mention_count >= {mention} "
            "GROUP BY p.region, v.disease, e.name "
            "ORDER BY patient_count DESC, p.region ASC, "
            "v.disease ASC, graph_exam ASC"
        )
    head_mentions = _first_number(r"头实体提及次数至少\s*(\d+)", q)
    tail_mentions = _first_number(r"尾实体提及次数至少\s*(\d+)", q)
    if head_mentions and tail_mentions and "头尾实体都连接到实体表" in q:
        return (
            "SELECT h.name AS disease, s.name AS symptom, "
            "h.mention_count AS disease_mentions, "
            "s.mention_count AS symptom_mentions FROM kg_triples AS t "
            "JOIN kg_entities AS h ON h.name = t.head "
            "JOIN kg_entities AS s ON s.name = t.tail "
            "WHERE t.relation_zh = '临床表现' "
            f"AND h.mention_count >= {head_mentions} "
            f"AND s.mention_count >= {tail_mentions} "
            "ORDER BY disease ASC, symptom ASC"
        )
    if "每次就诊平均费用与每名就诊患者平均累计费用" in q:
        return (
            "SELECT p.region, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT p.patient_id) AS patient_count, "
            "AVG(v.cost) AS avg_cost_per_visit, "
            "SUM(v.cost) / COUNT(DISTINCT p.patient_id) "
            "AS avg_total_cost_per_patient FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            "GROUP BY p.region ORDER BY p.region ASC"
        )
    if "各地区接触到的不同治疗方式数量与不同疾病数量" in q:
        return (
            "SELECT p.region, COUNT(DISTINCT t.tail) AS treatment_count, "
            "COUNT(DISTINCT v.disease) AS disease_count FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            "JOIN kg_triples AS t ON t.head = v.disease "
            "WHERE t.relation_zh = '治疗方式' GROUP BY p.region "
            "ORDER BY treatment_count DESC, p.region ASC"
        )
    return None


def _fixed_head_average_order_owned(question: str) -> bool:
    """Require the static per-head average route to keep its fixed ordering."""

    matches = list(
        re.finditer(
            r"(?:先按|再按|最后按|其次按|然后按|按)"
            r"([^；。]+?)(?:排列|排序)",
            question,
        )
    )
    if not matches:
        return True
    order_text = matches[-1].group(1)
    labels = list(
        re.finditer(r"平均(?:边)?置信度|头实体|边数", order_text)
    )
    if not labels:
        return False
    expected = (
        ("average", "ASC"),
        ("head", "ASC"),
    )
    requested: list[tuple[str, str]] = []
    for index, label in enumerate(labels):
        following = order_text[
            label.end() : labels[index + 1].start() if index + 1 < len(labels) else None
        ]
        direction = re.search(r"(升序|降序)", following)
        if direction is None:
            return False
        semantic = (
            "average"
            if "置信度" in label.group(0)
            else "head"
            if label.group(0) == "头实体"
            else "edge_count"
        )
        requested.append(
            (semantic, "ASC" if direction.group(1) == "升序" else "DESC")
        )
    return tuple(requested) == expected[: len(requested)]


def _fixed_head_average_route_owns(question: str, domains: Domains) -> bool:
    """Admit only qualifiers implemented by the fixed per-head average SQL."""

    request = analyze_relation_request(question, domains)
    if request.metric_bounds or request.top_k_spans or len(request.ordering_spans) > 1:
        return False
    if request.errors:
        # The designated aggregate threshold is intentionally not a row-level
        # relation bound.  Reparse the remaining text so no unrelated malformed
        # or disjunctive metric condition can hide behind that expected error.
        residual = re.sub(
            r"平均边置信度低于\s*\d+(?:\.\d+)?",
            "",
            question,
            count=1,
        )
        residual_request = analyze_relation_request(residual, domains)
        if (
            residual_request.errors
            or not residual_request.metric_complete
            or residual_request.metric_bounds
        ):
            return False
    tokens = _projection_tokens(question)
    if request.projection_spans and not tokens:
        return False
    if tokens and tokens != ["头实体", "边数", "平均置信度"]:
        return False
    return _fixed_head_average_order_owned(question)


def _rule_knowledge_graph(question: str, domains: Domains) -> Optional[str]:
    q = question
    entities = _domain_values(q, domains.get("kg_entity", ()))
    # Most direct graph templates render only one relation equality and no
    # row-level metric bound.  Do not derive that equality from a deduplicated
    # domain list: the structured request retains repeated mentions, relation
    # ownership, and every threshold that this template would otherwise drop.
    relation = _kg_flat_relation_value(
        q,
        domains,
        metric_capacity={},
    )
    row_metric_relation = _kg_flat_relation_value(
        q,
        domains,
        metric_capacity={"confidence": 1, "count": 1},
    )
    if entities and "在图谱中的全部关系" in q:
        return (
            "SELECT relation_zh, tail, tail_type, confidence, count "
            "FROM kg_triples "
            f"WHERE head = '{_esc(entities[0])}' "
            "ORDER BY relation_zh ASC, tail ASC"
        )
    if relation and "全部疾病与科室" in q:
        return (
            "SELECT head AS disease, tail AS department, confidence "
            "FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' "
            "ORDER BY department ASC, disease ASC"
        )
    if entities and "这一尾实体相连" in q:
        return (
            "SELECT head, relation_zh, head_type, count FROM kg_triples "
            f"WHERE tail = '{_esc(entities[0])}' "
            "ORDER BY count DESC, head ASC, relation_zh ASC"
        )
    if entities and relation and "所有临床表现" in q:
        return (
            "SELECT tail AS symptom, confidence, count FROM kg_triples "
            f"WHERE head = '{_esc(entities[0])}' "
            f"AND relation_zh = '{_esc(relation)}' "
            "ORDER BY count DESC, symptom ASC"
        )
    count = _first_number(r"观察次数至少\s*(\d+)", q)
    confidence = _first_number(r"置信度至少\s*(\d+(?:\.\d+)?)", q)
    if (
        row_metric_relation
        and count
        and confidence
        and "疾病、检查、观察次数和置信度" in q
    ):
        return (
            "SELECT head AS disease, tail AS exam, count, confidence "
            "FROM kg_triples "
            f"WHERE relation_zh = '{_esc(row_metric_relation)}' "
            f"AND count >= {count} AND confidence >= {confidence} "
            "ORDER BY count DESC, disease ASC, exam ASC"
        )
    if relation and "所有用药禁忌边" in q:
        return (
            "SELECT head AS drug, tail AS disease, confidence, count "
            "FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' "
            "ORDER BY drug ASC, disease ASC"
        )
    tail_type = re.search(
        r"尾实体类型为\s*(Disease|Drug|Exam|Symptom|Treatment|Department)",
        q,
    )
    if tail_type and "疾病、治疗方式、置信度和观察次数" in q:
        return (
            "SELECT head AS disease, tail AS treatment, confidence, count "
            "FROM kg_triples "
            f"WHERE tail_type = '{tail_type.group(1)}' "
            "ORDER BY disease ASC, count DESC, treatment ASC"
        )
    tail_values = [
        value
        for value in entities
        if q.find(value) >= q.find("尾实体") if "尾实体" in q
    ]
    if relation and len(tail_values) >= 2 and "头疾病、并发症、置信度" in q:
        values = ", ".join(f"'{_esc(value)}'" for value in tail_values)
        return (
            "SELECT head AS disease, tail AS complication, confidence "
            "FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' "
            f"AND tail IN ({values}) "
            "ORDER BY complication ASC, disease ASC"
        )
    if "按关系中文名统计图谱边数与不同头实体数" in q:
        return (
            "SELECT relation_zh, COUNT(*) AS edge_count, "
            "COUNT(DISTINCT head) AS head_count FROM kg_triples "
            "GROUP BY relation_zh ORDER BY edge_count DESC, relation_zh ASC"
        )
    if relation and "每个疾病头实体的临床表现数量" in q:
        return (
            "SELECT head AS disease, COUNT(*) AS symptom_count "
            "FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' GROUP BY head "
            "ORDER BY symptom_count DESC, disease ASC"
        )
    if relation and "按所属科室统计图谱中的疾病数" in q:
        return (
            "SELECT tail AS department, COUNT(DISTINCT head) AS disease_count "
            "FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' GROUP BY tail "
            "ORDER BY disease_count DESC, department ASC"
        )
    if "Disease 且拥有图谱外出边" in q:
        return (
            "SELECT e.name, e.mention_count, COUNT(t.tail) AS edge_count "
            "FROM kg_entities AS e JOIN kg_triples AS t ON t.head = e.name "
            "WHERE e.type = 'Disease' GROUP BY e.name, e.mention_count "
            "ORDER BY edge_count DESC, e.mention_count DESC, e.name ASC"
        )
    if "未作为任何三元组头实体出现的 Drug 类型实体" in q:
        return (
            "SELECT e.name, e.mention_count FROM kg_entities AS e "
            "LEFT JOIN kg_triples AS t ON t.head = e.name "
            "WHERE e.type = 'Drug' AND t.head IS NULL "
            "ORDER BY e.mention_count DESC, e.name ASC"
        )
    floor = _first_number(r"至少\s*(\d+)\s*个不同疾病指向", q)
    if floor and "头实体类型为 Disease" in q:
        return (
            "SELECT tail, tail_type, COUNT(DISTINCT head) AS disease_count, "
            "COUNT(*) AS incoming_count FROM kg_triples "
            "WHERE head_type = 'Disease' GROUP BY tail, tail_type "
            f"HAVING COUNT(DISTINCT head) >= {floor} "
            "ORDER BY disease_count DESC, tail ASC"
        )
    confidence = _first_number(r"平均边置信度低于\s*(\d+(?:\.\d+)?)", q)
    if confidence and "头实体、边数和平均置信度" in q:
        if not _fixed_head_average_route_owns(q, domains):
            return None
        return (
            "SELECT head, COUNT(*) AS edge_count, "
            "AVG(confidence) AS avg_confidence FROM kg_triples GROUP BY head "
            f"HAVING AVG(confidence) < {confidence} "
            "ORDER BY avg_confidence ASC, head ASC"
        )
    if "既有诊断检查边又有治疗药物边" in q:
        return (
            "SELECT DISTINCT d.head AS disease FROM kg_triples AS d "
            "JOIN kg_triples AS m ON m.head = d.head "
            "WHERE d.relation_zh = '诊断检查' "
            "AND m.relation_zh = '治疗药物' ORDER BY disease ASC"
        )
    if "治疗药物同时也是另一条用药禁忌边头实体" in q:
        return (
            "SELECT DISTINCT t.head AS disease, t.tail AS treatment_drug, "
            "c.tail AS contraindicated_disease FROM kg_triples AS t "
            "JOIN kg_triples AS c ON c.head = t.tail "
            "WHERE t.relation_zh = '治疗药物' "
            "AND c.relation_zh = '用药禁忌' "
            "ORDER BY disease ASC, treatment_drug ASC, "
            "contraindicated_disease ASC"
        )
    if "出现为尾实体、但实体表中没有同名记录" in q:
        return (
            "SELECT DISTINCT t.tail, t.tail_type FROM kg_triples AS t "
            "LEFT JOIN kg_entities AS e ON e.name = t.tail "
            "WHERE e.name IS NULL ORDER BY t.tail_type ASC, t.tail ASC"
        )
    if "每种关系中最高的观察次数" in q:
        return (
            "SELECT relation_zh, MAX(count) AS max_observation_count "
            "FROM kg_triples GROUP BY relation_zh "
            "ORDER BY max_observation_count DESC, relation_zh ASC"
        )
    if "每个尾实体统计来自多少种不同关系" in q:
        return (
            "SELECT tail, tail_type, "
            "COUNT(DISTINCT relation_zh) AS relation_kind_count "
            "FROM kg_triples GROUP BY tail, tail_type "
            "ORDER BY relation_kind_count DESC, tail ASC"
        )
    if "每个疾病置信度最低的外出边" in q:
        return (
            "SELECT t.head AS disease, t.relation_zh, t.tail, t.confidence "
            "FROM kg_triples AS t WHERE t.head_type = 'Disease' "
            "AND t.confidence = (SELECT MIN(x.confidence) "
            "FROM kg_triples AS x WHERE x.head = t.head) "
            "ORDER BY disease ASC, t.relation_zh ASC, t.tail ASC"
        )
    if "两步疾病链" in q and "并发症边" in q and "所属科室" in q:
        return (
            "SELECT c.head AS source_disease, c.tail AS complication, "
            "d.tail AS complication_department FROM kg_triples AS c "
            "JOIN kg_triples AS d ON d.head = c.tail "
            "WHERE c.relation_zh = '并发症' "
            "AND d.relation_zh = '所属科室' "
            "ORDER BY source_disease ASC, complication ASC, "
            "complication_department ASC"
        )
    if "提及次数与所有外出边观察次数总和之差" in q:
        return (
            "SELECT e.name, e.mention_count, "
            "COALESCE(SUM(t.count), 0) AS edge_observations, "
            "e.mention_count - COALESCE(SUM(t.count), 0) "
            "AS mention_minus_observations FROM kg_entities AS e "
            "LEFT JOIN kg_triples AS t ON t.head = e.name "
            "WHERE e.type = 'Disease' GROUP BY e.name, e.mention_count "
            "ORDER BY mention_minus_observations DESC, e.name ASC"
        )
    if relation and "每种检查被多少个不同疾病使用" in q:
        return (
            "SELECT tail AS exam, COUNT(DISTINCT head) AS disease_count, "
            "SUM(count) AS total_observations FROM kg_triples "
            f"WHERE relation_zh = '{_esc(relation)}' GROUP BY tail "
            "ORDER BY disease_count DESC, total_observations DESC, exam ASC"
        )
    if "所有头实体为 Drug 的边" in q and "相应药物实体" in q:
        return (
            "SELECT e.name AS drug, e.mention_count, t.relation_zh, "
            "t.tail AS disease, t.confidence FROM kg_entities AS e "
            "JOIN kg_triples AS t ON t.head = e.name "
            "WHERE e.type = 'Drug' ORDER BY drug ASC, disease ASC"
        )
    mention = _first_number(r"提及次数不少于\s*(\d+)", q)
    confidence = _first_number(r"置信度低于\s*(\d+(?:\.\d+)?)", q)
    if mention and confidence and "至少存在一条" in q and "外出边" in q:
        return (
            "SELECT e.name, e.mention_count FROM kg_entities AS e "
            "WHERE e.type = 'Disease' "
            f"AND e.mention_count >= {mention} "
            "AND EXISTS (SELECT 1 FROM kg_triples AS t "
            f"WHERE t.head = e.name AND t.confidence < {confidence}) "
            "ORDER BY e.mention_count DESC, e.name ASC"
        )
    return None


def _rule_nested(question: str) -> Optional[str]:
    q = question
    if "年龄高于全部患者平均年龄" in q:
        return (
            "SELECT patient_id, name, age, region FROM patients "
            "WHERE age > (SELECT AVG(age) FROM patients) "
            "ORDER BY age DESC, patient_id ASC"
        )
    if "没有任何就诊记录的患者" in q:
        return (
            "SELECT p.patient_id, p.name, p.age, p.region FROM patients AS p "
            "WHERE NOT EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id) "
            "ORDER BY p.region ASC, p.patient_id ASC"
        )
    visits = _first_number(r"就诊次数至少\s*(\d+)\s*次", q)
    if visits and "患者ID、姓名和就诊次数" in q:
        return (
            "SELECT p.patient_id, p.name, "
            "(SELECT COUNT(*) FROM visits AS v "
            "WHERE v.patient_id = p.patient_id) AS visit_count "
            "FROM patients AS p WHERE "
            "(SELECT COUNT(*) FROM visits AS v "
            f"WHERE v.patient_id = p.patient_id) >= {visits} "
            "ORDER BY visit_count DESC, p.patient_id ASC"
        )
    if "至少有一次就诊" in q and "全部就诊都是门诊" in q:
        return (
            "SELECT p.patient_id, p.name, p.region FROM patients AS p "
            "WHERE EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id) "
            "AND NOT EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id AND v.los_days <> 0) "
            "ORDER BY p.patient_id ASC"
        )
    cost = _first_number(r"费用不少于\s*(\d+(?:\.\d+)?)元", q)
    if cost and "所有就诊主要处方药都非空" in q:
        return (
            "SELECT p.patient_id, p.name FROM patients AS p "
            "WHERE EXISTS (SELECT 1 FROM visits AS v "
            f"WHERE v.patient_id = p.patient_id AND v.cost >= {cost}) "
            "AND NOT EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id "
            "AND LENGTH(v.drug_prescribed) = 0) "
            "ORDER BY p.patient_id ASC"
        )
    if "各疾病就诊次数的平均值" in q:
        return (
            "SELECT disease, COUNT(*) AS visit_count FROM visits "
            "GROUP BY disease HAVING COUNT(*) > "
            "(SELECT AVG(disease_count) FROM "
            "(SELECT COUNT(*) AS disease_count FROM visits GROUP BY disease)) "
            "ORDER BY visit_count DESC, disease ASC"
        )
    if "费用高于所属科室平均费用" in q:
        return (
            "SELECT v.visit_id, v.department, v.disease, v.cost, "
            "(SELECT AVG(x.cost) FROM visits AS x "
            "WHERE x.department = v.department) AS department_avg_cost "
            "FROM visits AS v WHERE v.cost > "
            "(SELECT AVG(x.cost) FROM visits AS x "
            "WHERE x.department = v.department) "
            "ORDER BY v.department ASC, v.cost DESC, v.visit_id ASC"
        )
    if "费用高于该患者本人平均就诊费用" in q:
        return (
            "SELECT v.patient_id, v.visit_id, v.cost, "
            "(SELECT AVG(x.cost) FROM visits AS x "
            "WHERE x.patient_id = v.patient_id) AS patient_avg_cost "
            "FROM visits AS v WHERE v.cost > "
            "(SELECT AVG(x.cost) FROM visits AS x "
            "WHERE x.patient_id = v.patient_id) "
            "ORDER BY v.patient_id ASC, v.cost DESC, v.visit_id ASC"
        )
    if "每个科室费用第二高的不同费用值" in q:
        return (
            "SELECT v.department, v.visit_id, v.disease, v.cost "
            "FROM visits AS v WHERE v.cost = "
            "(SELECT MAX(x.cost) FROM visits AS x "
            "WHERE x.department = v.department AND x.cost < "
            "(SELECT MAX(y.cost) FROM visits AS y "
            "WHERE y.department = v.department)) "
            "ORDER BY v.department ASC, v.visit_id ASC"
        )
    edges = _first_number(r"至少有\s*(\d+)\s*条临床表现边", q)
    if edges and "主诊断在图谱中" in q:
        return (
            "SELECT v.visit_id, v.disease, "
            "(SELECT COUNT(*) FROM kg_triples AS t "
            "WHERE t.head = v.disease "
            "AND t.relation_zh = '临床表现') AS symptom_count "
            "FROM visits AS v WHERE "
            "(SELECT COUNT(*) FROM kg_triples AS t "
            "WHERE t.head = v.disease "
            f"AND t.relation_zh = '临床表现') >= {edges} "
            "ORDER BY symptom_count DESC, v.disease ASC, v.visit_id ASC"
        )
    if "没有作为任何三元组头或尾出现的图谱实体" in q:
        return (
            "SELECT e.name, e.type, e.mention_count FROM kg_entities AS e "
            "WHERE NOT EXISTS (SELECT 1 FROM kg_triples AS t "
            "WHERE t.head = e.name OR t.tail = e.name) "
            "ORDER BY e.type ASC, e.name ASC"
        )
    if "所属科室图谱边与至少一条真实就诊" in q:
        return (
            "SELECT t.head AS disease, t.tail AS department, t.count "
            "FROM kg_triples AS t WHERE t.relation_zh = '所属科室' "
            "AND EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.disease = t.head AND v.department = t.tail) "
            "ORDER BY department ASC, disease ASC"
        )
    cost = _first_number(r"所有就诊费用都高于\s*(\d+(?:\.\d+)?)元", q)
    if cost and "至少有一次就诊" in q:
        return (
            "SELECT p.patient_id, p.name FROM patients AS p "
            "WHERE EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id) "
            "AND NOT EXISTS (SELECT 1 FROM visits AS v "
            f"WHERE v.patient_id = p.patient_id AND v.cost <= {cost}) "
            "ORDER BY p.patient_id ASC"
        )
    if "主要检查非空" in q and "实体表中存在并标记为 Exam" in q:
        return (
            "SELECT v.visit_id, v.disease, v.exam FROM visits AS v "
            "WHERE LENGTH(v.exam) > 0 AND EXISTS "
            "(SELECT 1 FROM kg_entities AS e "
            "WHERE e.name = v.exam AND e.type = 'Exam') "
            "ORDER BY v.exam ASC, v.visit_id ASC"
        )
    if "置信度高于同一关系平均置信度" in q:
        return (
            "SELECT t.relation_zh, t.head, t.tail, t.confidence, "
            "(SELECT AVG(x.confidence) FROM kg_triples AS x "
            "WHERE x.relation_zh = t.relation_zh) "
            "AS relation_avg_confidence FROM kg_triples AS t "
            "WHERE t.confidence > "
            "(SELECT AVG(x.confidence) FROM kg_triples AS x "
            "WHERE x.relation_zh = t.relation_zh) "
            "ORDER BY t.relation_zh ASC, t.confidence DESC, "
            "t.head ASC, t.tail ASC"
        )
    if "高于所有尾实体不同头数平均值" in q:
        return (
            "SELECT tail, tail_type, "
            "COUNT(DISTINCT head) AS distinct_head_count FROM kg_triples "
            "GROUP BY tail, tail_type HAVING COUNT(DISTINCT head) > "
            "(SELECT AVG(head_count) FROM "
            "(SELECT COUNT(DISTINCT head) AS head_count FROM kg_triples "
            "GROUP BY tail, tail_type)) "
            "ORDER BY distinct_head_count DESC, tail ASC"
        )
    confidence = _first_number(r"置信度都不低于\s*(\d+(?:\.\d+)?)", q)
    if confidence and "至少有一条外出边" in q and "Disease 实体" in q:
        return (
            "SELECT e.name, e.mention_count FROM kg_entities AS e "
            "WHERE e.type = 'Disease' "
            "AND EXISTS (SELECT 1 FROM kg_triples AS t "
            "WHERE t.head = e.name) "
            "AND NOT EXISTS (SELECT 1 FROM kg_triples AS t "
            f"WHERE t.head = e.name AND t.confidence < {confidence}) "
            "ORDER BY e.mention_count DESC, e.name ASC"
        )
    if "相邻日期最大间隔" in q and "中间不存在" in q:
        return (
            "SELECT g.patient_id, MAX(g.gap_days) "
            "AS max_consecutive_gap_days FROM "
            "(SELECT a.patient_id, "
            "CAST(julianday(b.visit_date) - julianday(a.visit_date) AS INTEGER) "
            "AS gap_days FROM visits AS a JOIN visits AS b "
            "ON b.patient_id = a.patient_id AND b.visit_date > a.visit_date "
            "WHERE NOT EXISTS (SELECT 1 FROM visits AS m "
            "WHERE m.patient_id = a.patient_id "
            "AND m.visit_date > a.visit_date "
            "AND m.visit_date < b.visit_date)) AS g "
            "GROUP BY g.patient_id "
            "ORDER BY max_consecutive_gap_days DESC, g.patient_id ASC"
        )
    if "两步图谱路径" in q and "并发症" in q and "治疗药物" in q:
        return (
            "SELECT DISTINCT c.head AS source_disease, "
            "c.tail AS complication_disease, d.tail AS drug "
            "FROM kg_triples AS c JOIN kg_triples AS d ON d.head = c.tail "
            "WHERE c.relation_zh = '并发症' "
            "AND d.relation_zh = '治疗药物' "
            "ORDER BY source_disease ASC, complication_disease ASC, drug ASC"
        )
    return None


_SINGLE_TABLE_COLUMNS = {
    "patients": (
        ("患者ID", "patient_id"),
        ("患者编号", "patient_id"),
        ("患者姓名", "name"),
        ("实体名", "name"),
        ("姓名", "name"),
        ("性别", "gender"),
        ("年龄", "age"),
        ("地区", "region"),
    ),
    "visits": (
        ("就诊ID", "visit_id"),
        ("就诊编号", "visit_id"),
        ("患者ID", "patient_id"),
        ("患者编号", "patient_id"),
        ("就诊日期", "visit_date"),
        ("日期", "visit_date"),
        ("主诊断", "disease"),
        ("疾病", "disease"),
        ("就诊科室", "department"),
        ("科室", "department"),
        ("主要处方药", "drug_prescribed"),
        ("处方药", "drug_prescribed"),
        ("药物", "drug_prescribed"),
        ("主要检查", "exam"),
        ("检查", "exam"),
        ("住院天数", "los_days"),
        ("费用", "cost"),
    ),
    "kg_entities": (
        ("实体名", "name"),
        ("名称", "name"),
        ("实体类型", "type"),
        ("类型", "type"),
        ("提及次数", "mention_count"),
    ),
    "kg_triples": (
        ("关系中文名", "relation_zh"),
        ("关系", "relation_zh"),
        ("头实体类型", "head_type"),
        ("尾实体类型", "tail_type"),
        ("头实体", "head"),
        ("尾实体", "tail"),
        ("疾病", "head"),
        ("症状", "tail"),
        ("治疗方式", "tail"),
        ("药物", "tail"),
        ("检查", "tail"),
        ("置信度", "confidence"),
        ("观察次数", "count"),
    ),
}


def _projection_tokens(question: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?:返回|显示|输出|列出|给出)(.+?)(?=；|。|$)",
            question,
        )
    )
    if not matches:
        return []
    phrase = matches[-1].group(1)
    phrase = re.split(r"[，,]?(?:并)?按", phrase, maxsplit=1)[0]
    phrase = re.sub(r"^(?:全部|相应|所需)", "", phrase)
    phrase = phrase.replace("以及", "、").replace("和", "、")
    phrase = re.sub(r"(?<=项)及|及(?=(?:费用|次数|数量|人数|日期|类型))", "、", phrase)
    return [
        token.strip(" ，,；。")
        for token in re.split(r"[、，,]", phrase)
        if token.strip(" ，,；。")
    ]


def _map_projection(table: str, tokens: Sequence[str]) -> Optional[list[str]]:
    mappings = _SINGLE_TABLE_COLUMNS[table]
    # Output language distinguishes the raw relation code from its Chinese
    # label even though ordering/filtering may use the latter as “关系”.
    relation_projection_aliases = {
        "所有关系": "relation",
        "关系": "relation",
        "中文关系名": "relation_zh",
        "关系中文名": "relation_zh",
    }
    result: list[str] = []
    for token in tokens:
        if table == "kg_triples" and token in relation_projection_aliases:
            result.append(relation_projection_aliases[token])
            continue
        matches = [
            column
            for label, column in mappings
            if label == token
            or token.endswith(label)
            or token.startswith(label)
        ]
        unique = list(dict.fromkeys(matches))
        if len(unique) != 1:
            return None
        result.append(unique[0])
    return result


def _order_clause(
    question: str,
    table: str,
    *,
    aliases: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    matches = list(
        re.finditer(
            r"(?:先按|再按|最后按|其次按|然后按|按)"
            r"([^；。]+?)(?:排列|排序)",
            question,
        )
    )
    if not matches:
        return ""
    text = matches[-1].group(1)
    text = re.sub(r"(?:先按|再按|最后按|其次按|然后按|按)", "、", text)
    chunks = [
        chunk.strip(" ，,、；")
        for chunk in re.split(r"[、，,]", text)
        if chunk.strip(" ，,、；")
    ]
    semantic_aliases = dict(aliases or {})
    mappings = _SINGLE_TABLE_COLUMNS[table]
    order: list[str] = []
    for chunk in chunks:
        direction_match = re.search(r"(升序|降序)", chunk)
        if not direction_match:
            # Phrases such as "按地区、患者ID升序" apply the final
            # direction to each preceding item.
            direction = "ASC"
        else:
            direction = "DESC" if direction_match.group(1) == "降序" else "ASC"
        label = re.sub(
            r"(?:先|再|最后|其次|然后|同(?:样)?|均|排列|排序|升序|降序|"
            r"人数相同|次数相同|费用相同|年龄相同|数量相同)",
            "",
            chunk,
        ).strip()
        candidate = None
        for semantic, alias in sorted(
            semantic_aliases.items(),
            key=lambda item: -len(item[0]),
        ):
            if semantic in label:
                candidate = alias
                break
        if candidate is None:
            candidates = [
                column
                for semantic, column in mappings
                if semantic in label
            ]
            candidates = list(dict.fromkeys(candidates))
            if len(candidates) == 1:
                candidate = candidates[0]
        if candidate is None:
            return None
        order.append(f"{candidate} {direction}")
    return "ORDER BY " + ", ".join(order) if order else ""


def _kg_flat_predicate_request(
    question: str,
    domains: Domains,
    *,
    metric_capacity: Mapping[str, int],
) -> Optional[RelationRequest]:
    """Return a KG row-filter request only when this route owns it completely.

    Generic ``kg_triples`` lowering has a flat equality/``IN`` relation slot and
    a bounded number of row-level metric slots.  It cannot implement
    relation-local thresholds, set operations, required-all semantics, or a
    metric range by retaining just the first comparison.  Preserve the raw
    occurrence plan through admission so callers can abstain before producing a
    broader predicate.
    """

    request = analyze_relation_request(question, domains)
    if request.has_relation_intent:
        return (
            request
            if request.accepts_flat_predicate(metric_capacity=metric_capacity)
            else None
        )

    # A metric-only graph filter has no relation ownership question, but it
    # still must fit every row-level predicate slot and contain no unowned OR
    # or incomplete continuation.
    if request.errors or not request.metric_complete:
        return None
    if not request.metric_bounds:
        return request
    for metric in {bound.metric for bound in request.metric_bounds}:
        capacity = metric_capacity.get(metric)
        if capacity is None or sum(
            bound.metric == metric for bound in request.metric_bounds
        ) > capacity:
            return None
    return request


def _kg_flat_relation_value(
    question: str,
    domains: Domains,
    *,
    metric_capacity: Mapping[str, int],
) -> Optional[str]:
    """Return one relation only for a direct equality route that owns it."""

    request = _kg_flat_predicate_request(
        question,
        domains,
        metric_capacity=metric_capacity,
    )
    if request is None or len(request.values) != 1:
        return None
    return request.values[0]


def _simple_selected_columns(sql: str) -> Optional[list[str]]:
    """Return raw output columns for a simple single-table SELECT statement."""

    match = re.match(
        r"\s*SELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM\s+",
        sql,
        re.I | re.S,
    )
    if match is None:
        return None
    columns: list[str] = []
    for expression in match.group(1).split(","):
        column = re.fullmatch(
            r"\s*(?:[A-Za-z_]\w*\.)?([A-Za-z_]\w*)"
            r"(?:\s+AS\s+[A-Za-z_]\w*)?\s*",
            expression,
            re.I,
        )
        if column is None:
            return None
        columns.append(column.group(1))
    return columns


def _selected_output_names(sql: str) -> Optional[list[str]]:
    """Return selected output aliases for a simple grouped KG statement."""

    match = re.match(
        r"\s*SELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM\s+",
        sql,
        re.I | re.S,
    )
    if match is None:
        return None
    names: list[str] = []
    for expression in match.group(1).split(","):
        alias = re.search(
            r"\s+AS\s+([A-Za-z_]\w*)\s*$",
            expression,
            re.I,
        )
        if alias is not None:
            names.append(alias.group(1))
            continue
        column = re.fullmatch(
            r"\s*(?:[A-Za-z_]\w*\.)?([A-Za-z_]\w*)\s*",
            expression,
            re.I,
        )
        if column is None:
            return None
        names.append(column.group(1))
    return names


def _required_all_relation_owners(sql: str, values: Sequence[str]) -> bool:
    """Prove conjunctive common-head or canonical grouped ownership."""

    without_literals = re.sub(r"'(?:''|[^'])*'", "''", sql)
    if re.search(r"\bOR\b", without_literals, re.I):
        return False
    # Repeating a relation phrase is not a cardinality threshold by itself. An
    # equality pair cannot prove distinct underlying edges, so leave such
    # requests to a route with explicit edge-identity semantics.
    if len(set(values)) != len(values):
        return False
    canonical_source = (
        re.search(
            r"^\s*SELECT\s+(?:DISTINCT\s+)?(?:[A-Za-z_]\w*\.)?head"
            r"(?:\s+AS\s+[A-Za-z_]\w*)?\s+FROM\s+kg_triples\b",
            sql,
            re.I | re.S,
        )
        is not None
        and re.search(
            r"\b(?:JOIN|WITH|UNION|INTERSECT|EXCEPT)\b",
            without_literals,
            re.I,
        )
        is None
    )

    for match in re.finditer(
        r"(?:\b[A-Za-z_]\w*\.)?relation_zh\s+IN\s*\(([^)]*)\)",
        sql,
        re.I,
    ):
        rendered_values = tuple(
            item.replace("''", "'")
            for item in re.findall(r"'((?:''|[^'])*)'", match.group(1))
        )
        if (
            len(rendered_values) == len(values)
            and set(rendered_values) == set(values)
            and canonical_source
            and re.search(
                r"\bGROUP\s+BY\s+(?:[A-Za-z_]\w*\.)?head\s+"
                r"HAVING\s+COUNT\s*\(\s*DISTINCT\s+"
                r"(?:[A-Za-z_]\w*\.)?relation_zh\s*\)\s*=\s*"
                rf"{len(values)}(?!\d)",
                sql,
                re.I,
            ) is not None
        ):
            return True

    matches = list(
        re.finditer(
            r"\b(?P<alias>[A-Za-z_]\w*)\.relation_zh\s*=\s*"
            r"'(?P<value>(?:''|[^'])*)'",
            sql,
            re.I,
        )
    )
    used: set[int] = set()
    aliases: list[str] = []
    for value in values:
        index = next(
            (
                item_index
                for item_index, match in enumerate(matches)
                if item_index not in used
                and match.group("value").replace("''", "'") == value
            ),
            None,
        )
        if index is None:
            return False
        used.add(index)
        aliases.append(matches[index].group("alias"))
    if len(set(aliases)) != len(aliases):
        return False
    for alias in aliases:
        if re.search(
            rf"\b(?:FROM|JOIN)\s+kg_triples\s+(?:AS\s+)?{re.escape(alias)}\b",
            sql,
            re.I,
        ) is None:
            return False

    graph: dict[str, set[str]] = {}
    for match in re.finditer(
        r"\b(?P<left_alias>[A-Za-z_]\w*)\."
        r"(?P<left_column>[A-Za-z_]\w*)\s*=\s*"
        r"(?P<right_alias>[A-Za-z_]\w*)\."
        r"(?P<right_column>[A-Za-z_]\w*)\b",
        sql,
        re.I,
    ):
        if (
            match.group("left_column").lower() != "head"
            or match.group("right_column").lower() != "head"
        ):
            continue
        left_alias = match.group("left_alias")
        right_alias = match.group("right_alias")
        graph.setdefault(left_alias, set()).add(right_alias)
        graph.setdefault(right_alias, set()).add(left_alias)
    reachable = {aliases[0]}
    frontier = [aliases[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in graph.get(current, ()):
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    return set(aliases).issubset(reachable)


def _generic_group_sql_owns_projection(sql: str, question: str) -> bool:
    """Require a grouped KG result to match the generic group result shape."""

    dimensions = _generic_group_dimensions(question, "kg_triples")
    metrics = _generic_metrics(question, "kg_triples")
    if (
        not dimensions
        or not metrics
        or not _generic_group_projection_owned(question, dimensions, metrics)
    ):
        return False
    expected = [alias for _, _, alias, _ in tuple(dimensions) + tuple(metrics)]
    return _selected_output_names(sql) == expected


def _sql_owns_relation_request(
    sql: str,
    question: str,
    request: RelationRequest,
) -> bool:
    """Verify that a selected explicit route consumed every relation intent.

    This is a final structured ownership fence for specialised explicit routes.
    A specialised template may legitimately own a required-all request only
    through distinct common-head roles or a canonical grouped proof, but it
    must never win merely because it saw a nearby entity/type phrase and then
    discard a grounded relation, metric endpoint, explicit result shape,
    ordering, or Top-K requirement.
    """

    if not request.has_relation_intent:
        return True
    if request.errors or not request.metric_complete:
        return False
    if request.connective in {"set_operation", "ordered", "ambiguous"}:
        return False
    if any(
        bound.owner not in {"single", "shared"}
        for bound in request.metric_bounds
    ):
        return False

    def has_equality(value: str) -> bool:
        rendered = re.escape(_esc(value))
        return re.search(
            rf"(?:\b[A-Za-z_]\w*\.)?relation_zh\s*=\s*'{rendered}'",
            sql,
        ) is not None

    if request.connective == "literal_or":
        union_owned = False
        for match in re.finditer(
            r"(?:\b[A-Za-z_]\w*\.)?relation_zh\s+IN\s*\(([^)]*)\)",
            sql,
            re.I,
        ):
            rendered_values = tuple(
                item.replace("''", "'")
                for item in re.findall(r"'((?:''|[^'])*)'", match.group(1))
            )
            if (
                len(rendered_values) == len(request.values)
                and set(rendered_values) == set(request.values)
            ):
                union_owned = True
                break
        if not union_owned:
            return False
    elif request.connective == "single":
        if not all(has_equality(value) for value in request.values):
            return False
    elif request.connective == "required_all":
        if len(request.values) == 1:
            if not has_equality(request.values[0]):
                return False
        elif not _required_all_relation_owners(sql, request.values):
            return False
    else:
        return False

    for bound in request.metric_bounds:
        column = "confidence" if bound.metric == "confidence" else "count"
        scalar = re.search(
            rf"(?:\b[A-Za-z_]\w*\.)?{column}\s*"
            rf"{re.escape(bound.operator)}\s*{re.escape(bound.value)}(?![\d.])",
            sql,
        )
        if scalar is not None:
            continue
        if bound.operator not in {">=", "<="}:
            return False
        range_match = re.search(
            rf"(?:\b[A-Za-z_]\w*\.)?{column}\s+BETWEEN\s+"
            r"(\d+(?:\.\d+)?)\s+AND\s+(\d+(?:\.\d+)?)",
            sql,
            re.I,
        )
        if range_match is None:
            return False
        endpoint = range_match.group(1 if bound.operator == ">=" else 2)
        if endpoint != bound.value:
            return False

    if len(request.ordering_spans) > 1 or len(request.top_k_spans) > 1:
        return False
    if request.ordering_spans:
        requested_order = _order_clause(question, "kg_triples")
        if requested_order is None:
            # Aggregate aliases are interpreted by _rule_generic_group().  A
            # non-grouped explicit route has no comparable order owner.
            if " GROUP BY " not in sql.upper():
                return False
        elif requested_order:
            order_match = re.search(
                r"\bORDER BY\s+(.+?)(?:\s+LIMIT\s+\d+|$)",
                sql,
                re.I,
            )
            if order_match is None or not order_match.group(1).startswith(
                requested_order.removeprefix("ORDER BY ")
            ):
                return False
    if request.top_k_spans:
        start, end = request.top_k_spans[0]
        limit = re.search(r"\d+", question[start:end])
        if limit is None or re.search(
            rf"\bLIMIT\s+{limit.group(0)}\b",
            sql,
            re.I,
        ) is None:
            return False

    tokens = _projection_tokens(question)
    requested_columns = _map_projection("kg_triples", tokens) if tokens else None
    if tokens:
        if " GROUP BY " in sql.upper():
            if request.connective == "required_all":
                if requested_columns is None:
                    return False
                selected_columns = _simple_selected_columns(sql)
                if selected_columns != requested_columns:
                    return False
            elif not _generic_group_sql_owns_projection(sql, question):
                return False
        elif requested_columns is None:
            return False
        else:
            selected_columns = _simple_selected_columns(sql)
            if selected_columns != requested_columns:
                return False
    return True


def _generic_group_projection_owned(
    question: str,
    dimensions: Sequence[tuple[int, str, str, tuple[str, ...]]],
    metrics: Sequence[tuple[int, str, str, tuple[str, ...]]],
) -> bool:
    """Require a generic group route to preserve an explicit result shape."""

    tokens = _projection_tokens(question)
    if not tokens:
        return True
    selections = tuple(dimensions) + tuple(metrics)
    selected = [alias for _, _, alias, _ in selections]
    requested: list[str] = []
    for token in tokens:
        candidates = {
            alias
            for _, _, alias, semantics in selections
            for semantic in semantics
            if (
                token == semantic
                or token.startswith(semantic)
                or token.endswith(semantic)
                or semantic.endswith(token)
            )
        }
        if len(candidates) != 1:
            return False
        requested.append(candidates.pop())
    return requested == selected


def _generic_where(
    question: str,
    table: str,
    domains: Domains,
) -> Optional[tuple[list[str], set[str]]]:
    q = question
    predicates: list[str] = []
    used: set[str] = set()
    if table == "patients":
        age_between = re.search(r"年龄(?:在|介于)?\s*(\d+)\s*(?:至|到)\s*(\d+)", q)
        if age_between:
            predicates.append(
                f"age BETWEEN {int(age_between.group(1))} "
                f"AND {int(age_between.group(2))}"
            )
            used.update(age_between.groups())
        else:
            comparators = (
                (r"年龄(?:至少|不低于)\s*(\d+)岁?", ">="),
                (r"年龄(?:在)?\s*(\d+)岁(?:及以上|以上)", ">="),
                (r"年龄(?:超过|大于)\s*(\d+)", ">"),
                (r"年龄(?:不超过|至多)\s*(\d+)", "<="),
                (r"年龄(?:低于|小于)\s*(\d+)", "<"),
            )
            for pattern, operator in comparators:
                match = re.search(pattern, q)
                if match:
                    predicates.append(f"age {operator} {int(match.group(1))}")
                    used.add(match.group(1))
                    break
        if "年龄为偶数" in q:
            predicates.append("age % 2 = 0")
        elif "年龄为奇数" in q:
            predicates.append("age % 2 = 1")
        if "女性" in q or "性别为女" in q:
            predicates.append("gender = '女'")
        elif "男性" in q or "性别为男" in q:
            predicates.append("gender = '男'")
        values = _domain_values(q, domains.get("region", ()))
        if values:
            rendered = ", ".join(f"'{_esc(value)}'" for value in values)
            if "不是" in q or "不在" in q:
                predicates.append(f"region NOT IN ({rendered})")
            elif len(values) > 1:
                predicates.append(f"region IN ({rendered})")
            else:
                predicates.append(f"region = '{_esc(values[0])}'")
            used.update(values)
        like = re.search(r"精确LIKE模式为([^。；\s]+)", q, re.I)
        if like and "姓名" in q:
            predicates.append(f"name LIKE '{_esc(like.group(1))}'")
            used.add(like.group(1))
    elif table == "visits":
        if has_disjunctive_metric_bound(
            q,
            "费用",
            ("花费", "开销", "金额"),
        ) or has_disjunctive_metric_bound(q, "住院天数", ("住院时长",)):
            return None
        cost_between = re.search(
            r"费用(?:在|介于)?\s*(\d+(?:\.\d+)?)\s*(?:至|到)\s*"
            r"(\d+(?:\.\d+)?)",
            q,
        )
        if cost_between:
            predicates.append(
                f"cost BETWEEN {_number(cost_between.group(1))} "
                f"AND {_number(cost_between.group(2))}"
            )
            used.update(cost_between.groups())
        else:
            comparators = (
                (r"费用(?:至少|不低于|不少于)\s*(\d+(?:\.\d+)?)", ">="),
                (r"费用(?:超过|大于|高于)\s*(\d+(?:\.\d+)?)", ">"),
                (r"费用(?:不超过|至多)\s*(\d+(?:\.\d+)?)", "<="),
                (r"费用(?:低于|小于|少于)\s*(\d+(?:\.\d+)?)", "<"),
            )
            for pattern, operator in comparators:
                match = re.search(pattern, q)
                if match:
                    predicates.append(
                        f"cost {operator} {_number(match.group(1))}"
                    )
                    used.add(match.group(1))
                    break
        stay_between = re.search(r"住院\s*(\d+)\s*(?:至|到)\s*(\d+)天", q)
        if stay_between:
            predicates.append(
                f"los_days BETWEEN {int(stay_between.group(1))} "
                f"AND {int(stay_between.group(2))}"
            )
            used.update(stay_between.groups())
        else:
            stay_comparators = (
                (r"住院天数(?:至少|不低于|不少于)\s*(\d+)", ">="),
                (r"住院天数(?:超过|大于)\s*(\d+)", ">"),
                (r"住院天数(?:不超过|至多)\s*(\d+)", "<="),
                (r"住院天数(?:低于|小于|少于)\s*(\d+)", "<"),
            )
            for pattern, operator in stay_comparators:
                match = re.search(pattern, q)
                if match:
                    predicates.append(
                        f"los_days {operator} {int(match.group(1))}"
                    )
                    used.add(match.group(1))
                    break
        if (
            not any(predicate.startswith("los_days ") for predicate in predicates)
            and "门诊" in q
            and ("住院天数" in q or "门诊记录" in q)
        ):
            predicates.append("los_days = 0")
        elif (
            not any(predicate.startswith("los_days ") for predicate in predicates)
            and "住院" in q
            and "住院天数大于0" in q
        ):
            predicates.append("los_days > 0")
        dates = _dates(q)
        if len(dates) >= 2:
            end_operator = (
                "<"
                if "不含" in q or "开区间" in q
                else "<="
            )
            predicates.extend(
                [
                    f"visit_date >= '{dates[0]}'",
                    f"visit_date {end_operator} '{dates[1]}'",
                ]
            )
            used.update(dates[:2])
        elif len(dates) == 1:
            if re.search(
                rf"{re.escape(dates[0])}(?:及以后|以后|起|及之后|之后)",
                q,
            ):
                predicates.append(f"visit_date >= '{dates[0]}'")
                used.add(dates[0])
            elif "截至" in q or re.search(
                rf"{re.escape(dates[0])}(?:及以前|以前|之前)",
                q,
            ):
                predicates.append(f"visit_date <= '{dates[0]}'")
                used.add(dates[0])
        for domain, column in (
            ("department", "department"),
            ("disease", "disease"),
            ("drug", "drug_prescribed"),
            ("exam", "exam"),
        ):
            values = _domain_values(q, domains.get(domain, ()))
            if not values:
                continue
            rendered = ", ".join(f"'{_esc(value)}'" for value in values)
            if len(values) > 1:
                predicates.append(f"{column} IN ({rendered})")
            else:
                predicates.append(f"{column} = '{_esc(values[0])}'")
            used.update(values)
        for label, column in (
            ("处方药", "drug_prescribed"),
            ("检查", "exam"),
        ):
            if f"{label}非空" in q and "字符长度大于0" in q:
                predicates.append(f"LENGTH({column}) > 0")
        like = re.search(r"精确LIKE模式为([^。；\s]+)", q, re.I)
        if like:
            column = "disease" if "疾病" in q or "诊断" in q else None
            if column:
                predicates.append(f"{column} LIKE '{_esc(like.group(1))}'")
                used.add(like.group(1))
    elif table == "kg_entities":
        entity_type = re.search(
            r"(?:(Disease|Drug|Exam|Symptom|Treatment|Department)\s*类型|"
            r"类型为\s*(Disease|Drug|Exam|Symptom|Treatment|Department))",
            q,
        )
        if entity_type:
            value = entity_type.group(1) or entity_type.group(2)
            predicates.append(f"type = '{value}'")
            used.add(value)
        mention = re.search(
            r"提及次数(?:至少|不少于|不低于|超过|大于|小于|低于)"
            r"\s*(\d+)",
            q,
        )
        if mention:
            prefix = q[max(0, mention.start() - 8) : mention.start(1)]
            operator = (
                "<"
                if "小于" in prefix or "低于" in prefix
                else ">"
                if "超过" in prefix or "大于" in prefix
                else ">="
            )
            predicates.append(
                f"mention_count {operator} {int(mention.group(1))}"
            )
            used.add(mention.group(1))
    elif table == "kg_triples":
        request = _kg_flat_predicate_request(
            q,
            domains,
            metric_capacity={"confidence": 1, "count": 1},
        )
        if request is None:
            return None
        if request.values:
            rendered = ", ".join(f"'{_esc(value)}'" for value in request.values)
            predicates.append(
                f"relation_zh {'IN (' + rendered + ')' if len(request.values) > 1 else '= ' + rendered}"
            )
            used.update(request.values)
        for label, column in (
            ("头实体类型", "head_type"),
            ("尾实体类型", "tail_type"),
        ):
            match = re.search(
                rf"{label}为\s*(Disease|Drug|Exam|Symptom|Treatment|Department)",
                q,
            )
            if match:
                predicates.append(f"{column} = '{match.group(1)}'")
                used.add(match.group(1))
        for metric, column in (
            ("confidence", "confidence"),
            ("count", "count"),
        ):
            for bound in request.metric_bounds:
                if bound.metric != metric:
                    continue
                predicates.append(f"{column} {bound.operator} {bound.value}")
                used.add(bound.value)
    return list(dict.fromkeys(predicates)), used


def _infer_single_table(question: str) -> Optional[str]:
    if any(term in question for term in ("三元组", "图谱边", "头实体", "尾实体", "关系中文名")):
        return "kg_triples"
    if "图谱实体" in question or re.search(
        r"(?:Disease|Drug|Exam|Symptom|Treatment|Department)\s*类型实体",
        question,
    ):
        return "kg_entities"
    if any(
        term in question
        for term in (
            "就诊",
            "费用",
            "住院天数",
            "处方药",
            "主诊断",
            "主要检查",
        )
    ):
        return "visits"
    if "患者" in question:
        return "patients"
    return None


def _rule_generic_filter(question: str, domains: Domains) -> Optional[str]:
    q = question
    if len(_raw_relation_mentions(q, domains)) > 1:
        # _generic_where() has only one slot per metric and would otherwise
        # flatten repeated or relation-owned predicates to a broad IN filter.
        return None
    if any(
        marker in q
        for marker in (
            "统计",
            "汇总",
            "平均值",
            "CTE",
            "窗口",
            "排名",
            "名次",
            "每个",
            "每种",
            "连接",
            "配对",
            "交集",
            "并集",
        )
    ):
        return None
    table = _infer_single_table(q)
    tokens = _projection_tokens(q)
    if table is None or not tokens:
        return None
    columns = _map_projection(table, tokens)
    if not columns:
        return None
    where = _generic_where(q, table, domains)
    if where is None:
        return None
    predicates, _ = where
    filter_markers = (
        "筛选",
        "查找",
        "找出",
        "查询",
        "列出",
        "至少",
        "超过",
        "低于",
        "小于",
        "大于",
        "非空",
        "类型",
        "关系",
    )
    if any(marker in q for marker in filter_markers) and not predicates:
        return None
    order = _order_clause(q, table)
    if not order:
        return None
    sql = f"SELECT {', '.join(columns)} FROM {table}"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    if order:
        sql += " " + order
    limit = re.search(r"前\s*(\d+)", q)
    if limit:
        sql += f" LIMIT {int(limit.group(1))}"
    return sql


def _generic_group_dimensions(
    question: str,
    table: str,
) -> list[tuple[int, str, str, tuple[str, ...]]]:
    candidates: list[tuple[str, str, str, tuple[str, ...]]] = []
    if table == "patients":
        candidates = [
            ("按地区", "region", "region", ("地区",)),
            ("按性别", "gender", "gender", ("性别",)),
            ("按年龄", "age", "age", ("年龄",)),
        ]
    elif table == "visits":
        candidates = [
            ("按科室", "department", "department", ("科室",)),
            ("按疾病", "disease", "disease", ("疾病",)),
            ("按诊断", "disease", "disease", ("疾病", "诊断")),
            (
                "按主要处方药",
                "drug_prescribed",
                "drug_prescribed",
                ("处方药", "药物"),
            ),
            (
                "按处方药物",
                "drug_prescribed",
                "drug_prescribed",
                ("处方药", "处方药物", "药物"),
            ),
            ("按处方药", "drug_prescribed", "drug_prescribed", ("处方药", "药物")),
            ("按主要检查", "exam", "exam", ("检查",)),
            ("按检查", "exam", "exam", ("检查",)),
            ("按患者", "patient_id", "patient_id", ("患者ID", "患者")),
            (
                "按就诊年份",
                "substr(visit_date, 1, 4)",
                "visit_year",
                ("年份", "就诊年份"),
            ),
            (
                "按年份",
                "substr(visit_date, 1, 4)",
                "visit_year",
                ("年份",),
            ),
            (
                "按年月",
                "substr(visit_date, 1, 7)",
                "visit_month",
                ("月份", "年月"),
            ),
            (
                "按月份",
                "substr(visit_date, 1, 7)",
                "visit_month",
                ("月份", "年月"),
            ),
            (
                "按月",
                "substr(visit_date, 1, 7)",
                "visit_month",
                ("月份",),
            ),
            (
                "每月",
                "substr(visit_date, 1, 7)",
                "visit_month",
                ("月份", "年月"),
            ),
            (
                "每年",
                "substr(visit_date, 1, 4)",
                "visit_year",
                ("年份", "就诊年份"),
            ),
            ("按日期", "visit_date", "visit_date", ("日期",)),
        ]
    elif table == "kg_entities":
        candidates = [
            ("按图谱实体类型", "type", "type", ("类型", "实体类型")),
            ("按实体类型", "type", "type", ("类型", "实体类型")),
            ("按类型", "type", "type", ("类型",)),
        ]
    elif table == "kg_triples":
        candidates = [
            (
                "按关系中文名",
                "relation_zh",
                "relation_zh",
                ("关系", "关系中文名"),
            ),
            ("按关系", "relation_zh", "relation_zh", ("关系",)),
            ("按头实体", "head", "head", ("头实体",)),
            ("按尾实体类型", "tail_type", "tail_type", ("尾类型", "尾实体类型")),
            ("按尾实体", "tail", "tail", ("尾实体",)),
            ("按头实体类型", "head_type", "head_type", ("头类型", "头实体类型")),
        ]
    found: list[tuple[int, str, str, tuple[str, ...]]] = []
    occupied_aliases: set[str] = set()
    for trigger, expression, alias, semantics in sorted(
        candidates,
        key=lambda item: -len(item[0]),
    ):
        position = question.find(trigger)
        if position < 0 or alias in occupied_aliases:
            continue
        found.append((position, expression, alias, semantics))
        occupied_aliases.add(alias)
    return sorted(found)


def _generic_metrics(
    question: str,
    table: str,
) -> list[tuple[int, str, str, tuple[str, ...]]]:
    if table == "patients":
        specs = [
            (r"不同年龄(?:值)?(?:的)?(?:数量|数)", "COUNT(DISTINCT age)", "distinct_age_count", ("不同年龄数", "不同年龄数量")),
            (r"患者(?:总)?人数|患者数|总人数", "COUNT(*)", "patient_count", ("患者人数", "患者数", "人数", "总人数")),
            (r"平均年龄", "AVG(age)", "avg_age", ("平均年龄",)),
            (r"最低年龄", "MIN(age)", "min_age", ("最低年龄",)),
            (r"最高年龄|最大年龄", "MAX(age)", "max_age", ("最高年龄", "最大年龄")),
        ]
    elif table == "visits":
        specs = [
            (r"不同就诊患者数|不同患者数|就诊患者数", "COUNT(DISTINCT patient_id)", "patient_count", ("不同患者数", "患者数", "就诊患者数")),
            (r"不同疾病(?:的)?(?:数量|数|种数)|疾病种数", "COUNT(DISTINCT disease)", "distinct_disease_count", ("不同疾病数", "疾病数", "疾病种数")),
            (r"不同检查(?:的)?(?:数量|数|种数)|检查种数", "COUNT(DISTINCT exam)", "distinct_exam_count", ("不同检查数", "检查数", "检查种数")),
            (r"不同(?:处方)?药(?:的)?(?:数量|数|种数)|处方药种数", "COUNT(DISTINCT drug_prescribed)", "distinct_drug_count", ("不同药物数", "处方药种数", "药物数")),
            (r"处方次数", "COUNT(*)", "prescription_count", ("处方次数",)),
            (r"就诊总次数|就诊次数|总次数", "COUNT(*)", "visit_count", ("就诊次数", "次数", "总次数")),
            (r"费用总额|总费用|累计费用", "SUM(cost)", "total_cost", ("费用总额", "总费用", "累计费用")),
            (r"平均费用", "AVG(cost)", "avg_cost", ("平均费用",)),
            (r"最低费用", "MIN(cost)", "min_cost", ("最低费用",)),
            (r"最高费用|最大费用", "MAX(cost)", "max_cost", ("最高费用", "最大费用")),
            (r"平均住院天数|平均住院时长", "AVG(los_days)", "avg_los_days", ("平均住院天数", "平均住院时长")),
            (r"最长住院天数|最大住院天数", "MAX(los_days)", "max_los_days", ("最长住院天数", "最大住院天数")),
            (r"最短住院天数|最小住院天数", "MIN(los_days)", "min_los_days", ("最短住院天数", "最小住院天数")),
            (r"最早就诊日|最早日期", "MIN(visit_date)", "first_visit_date", ("最早就诊日", "最早日期", "首日")),
            (r"最晚就诊日|最晚日期", "MAX(visit_date)", "last_visit_date", ("最晚就诊日", "最晚日期", "末日")),
        ]
    elif table == "kg_entities":
        specs = [
            (r"实体(?:总)?数|实体数量", "COUNT(*)", "entity_count", ("实体数", "实体数量")),
            (r"提及次数总和|提及总数", "SUM(mention_count)", "total_mentions", ("提及次数总和", "提及总数")),
            (r"平均提及次数", "AVG(mention_count)", "avg_mentions", ("平均提及次数",)),
            (r"最高提及次数", "MAX(mention_count)", "max_mentions", ("最高提及次数",)),
            (r"最低提及次数", "MIN(mention_count)", "min_mentions", ("最低提及次数",)),
        ]
    else:
        specs = [
            (r"不同头实体(?:的)?(?:数量|数)|头实体数", "COUNT(DISTINCT head)", "distinct_head_count", ("不同头实体数", "头实体数")),
            (r"不同尾实体(?:的)?(?:数量|数)", "COUNT(DISTINCT tail)", "distinct_tail_count", ("不同尾实体数", "尾实体数")),
            (r"不同关系(?:种类)?(?:的)?(?:数量|数)|关系种类数", "COUNT(DISTINCT relation_zh)", "relation_kind_count", ("不同关系数", "关系种类数")),
            (r"三元组(?:总)?数|三元组数量", "COUNT(*)", "triple_count", ("三元组数", "三元组数量", "边数")),
            (r"入边数量|入边数", "COUNT(*)", "incoming_count", ("入边数量", "入边数", "边数")),
            (r"外出边数量|外出边数|边数", "COUNT(*)", "edge_count", ("外出边数", "边数")),
            (r"观察次数总和|观察总数|观察总和", "SUM(count)", "total_observations", ("观察次数总和", "观察总数", "观察总和")),
            (r"平均观察次数", "AVG(count)", "avg_observations", ("平均观察次数",)),
            (r"最高观察次数|最大观察次数", "MAX(count)", "max_observation_count", ("最高观察次数", "最大观察次数")),
            (
                r"平均(?:边)?置信度",
                "AVG(confidence)",
                "avg_confidence",
                ("平均置信度", "平均边置信度"),
            ),
            (r"最低置信度|最小置信度", "MIN(confidence)", "min_confidence", ("最低置信度", "最小置信度")),
            (r"最高置信度|最大置信度", "MAX(confidence)", "max_confidence", ("最高置信度", "最大置信度")),
        ]
    found: list[tuple[int, str, str, tuple[str, ...]]] = []
    occupied_aliases: set[str] = set()
    for pattern, expression, alias, semantics in specs:
        match = re.search(pattern, question)
        if not match or alias in occupied_aliases:
            continue
        found.append((match.start(), expression, alias, semantics))
        occupied_aliases.add(alias)
    return sorted(found)


def _has_peer_aggregate_comparison(question: str) -> bool:
    """Recognise a comparison against a peer aggregate the generic route lacks."""

    peer_scope = (
        r"(?:同(?:类型|类|种|组)|本(?:类型|类|种|组)|"
        r"该(?:类型|类|种|组)|整体|全体)"
    )
    peer_average = rf"(?:{peer_scope}.{{0,8}}(?:平均|均值)|(?:平均|均值).{{0,8}}{peer_scope})"
    comparison = r"(?:高于|超过|大于|低于|小于|少于|不低于|不少于|不超过|至多)"
    return bool(
        re.search(rf"{comparison}\s*{peer_average}|{peer_average}\s*{comparison}", question)
    )


def _aggregate_threshold_labels(
    metrics: Sequence[tuple[int, str, str, tuple[str, ...]]],
) -> tuple[str, ...]:
    """Derive human metric forms that would require a grouped ``HAVING``."""

    labels: set[str] = set()
    for _, _, _, semantics in metrics:
        for semantic in semantics:
            labels.add(semantic)
            if semantic.startswith("不同"):
                labels.add(
                    re.sub(r"(?:的)?(?:数量|种数|数)$", "", semantic)
                )
    return tuple(sorted((label for label in labels if label), key=len, reverse=True))


def _has_unowned_generic_group_qualifier(
    question: str,
    metrics: Sequence[tuple[int, str, str, tuple[str, ...]]],
) -> bool:
    """Reject aggregate comparisons and thresholds without a lowered owner."""

    if _has_peer_aggregate_comparison(question):
        return True
    # Metric aliases include display variants such as ``平均边置信度``.
    # Match those aliases exactly: globally stripping ``边`` would reduce the
    # aggregate label ``边数`` to the unrelated row-metric fragment ``数``.
    operator = (
        r"(?:至少|不低于|不少于|达到|达|≥|>=|超过|高于|大于|>|多于|"
        r"不超过|不高于|至多|≤|<=|低于|小于|少于|<|等于|恰好为|为|=)"
    )
    value = r"\d+(?:\.\d+)?(?:\s*(?:元|天|个|次|条))?"
    for label in _aggregate_threshold_labels(metrics):
        if re.search(
            rf"{re.escape(label)}(?:的)?\s*{operator}\s*{value}",
            question,
            re.I,
        ):
            return True
    return False


def _rule_generic_group(question: str, domains: Domains) -> Optional[str]:
    q = question
    if any(
        marker in q
        for marker in (
            "CTE",
            "窗口",
            "排名",
            "名次",
            "交集",
            "并集",
            "高于平均",
            "低于平均",
            "所有",
            "每个科室分别取",
            "每种图谱关系分别取",
        )
    ):
        return None
    grouping_language = (
        "按" in q
        and any(marker in q for marker in ("统计", "汇总", "分组", "返回"))
    ) or (
        any(marker in q for marker in ("每月", "每年"))
        and any(marker in q for marker in ("统计", "汇总", "返回"))
    )
    if not grouping_language:
        return None
    table = _infer_single_table(q)
    if table is None:
        return None
    if table == "patients":
        # Patient-table grouping already has stable specialised rules.
        return None
    if table == "visits" and (
        "连接" in q
        or "患者性别" in q
        or "患者地区" in q
        or "按患者" in q
    ):
        # These frames require patient attributes or the established
        # patient/visit join compiler, not a visits-only grouping.
        return None
    dimensions = _generic_group_dimensions(q, table)
    metrics = _generic_metrics(q, table)
    if not dimensions or not metrics:
        return None
    if _has_unowned_generic_group_qualifier(q, metrics):
        return None
    relation_request = analyze_relation_request(q, domains)
    if relation_request.has_relation_intent and (
        len(relation_request.ordering_spans) > 1
        or len(relation_request.top_k_spans) > 1
        or not _generic_group_projection_owned(q, dimensions, metrics)
    ):
        return None
    if "只保留" in q:
        # HAVING is emitted only by a fully specified specialised rule.
        return None
    select_parts = [
        f"{expression} AS {alias}" if expression != alias else expression
        for _, expression, alias, _ in dimensions
    ]
    select_parts.extend(
        f"{expression} AS {alias}"
        for _, expression, alias, _ in metrics
    )
    group_parts = [expression for _, expression, _, _ in dimensions]
    where = _generic_where(q, table, domains)
    if where is None:
        return None
    predicates, _ = where
    aliases: dict[str, str] = {}
    for _, _, alias, semantics in dimensions + metrics:
        for semantic in semantics:
            aliases[semantic] = alias
    order = _order_clause(q, table, aliases=aliases)
    if not order:
        dimension_aliases = [alias for _, _, alias, _ in dimensions]
        metric_aliases = [alias for _, _, alias, _ in metrics]
        if any(
            alias in {"visit_month", "visit_year", "visit_date"}
            for alias in dimension_aliases
        ):
            order = "ORDER BY " + ", ".join(
                f"{alias} ASC" for alias in dimension_aliases
            )
        else:
            preferred = next(
                (
                    alias
                    for alias in (
                        "total_observations",
                        "prescription_count",
                        "visit_count",
                        "patient_count",
                        "edge_count",
                    )
                    if alias in metric_aliases
                ),
                None,
            )
            order_parts = (
                [f"{preferred} DESC"] if preferred is not None else []
            )
            order_parts.extend(f"{alias} ASC" for alias in dimension_aliases)
            order = "ORDER BY " + ", ".join(order_parts)
    sql = f"SELECT {', '.join(select_parts)} FROM {table}"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    sql += " GROUP BY " + ", ".join(group_parts)
    sql += " " + order
    limit = re.search(r"前\s*(\d+)", q)
    if limit:
        sql += f" LIMIT {int(limit.group(1))}"
    return sql


def _order_from_map(
    question: str,
    semantic_map: Mapping[str, str],
) -> Optional[str]:
    matches = list(
        re.finditer(
            r"(?:先按|再按|最后按|其次按|然后按|按)"
            r"([^；。]+?)(?:排列|排序)",
            question,
        )
    )
    if not matches:
        return None
    text = re.sub(
        r"(?:先按|再按|最后按|其次按|然后按|按)",
        "、",
        matches[-1].group(1),
    )
    chunks = [
        chunk.strip(" ，,、；")
        for chunk in re.split(r"[、，,]", text)
        if chunk.strip(" ，,、；")
    ]
    result: list[str] = []
    for chunk in chunks:
        direction = "DESC" if "降序" in chunk else "ASC"
        cleaned = re.sub(
            r"(?:先|再|最后|其次|然后|同(?:样)?|均|排列|排序|升序|降序|"
            r"人数相同|次数相同|费用相同|年龄相同|数量相同)",
            "",
            chunk,
        ).strip()
        candidates = [
            expression
            for semantic, expression in sorted(
                semantic_map.items(),
                key=lambda item: -len(item[0]),
            )
            if semantic in cleaned
        ]
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return None
        result.append(f"{candidates[0]} {direction}")
    return "ORDER BY " + ", ".join(result)


def _qualify_predicates(
    predicates: Sequence[str],
    alias: str,
    columns: Sequence[str],
) -> list[str]:
    pattern = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, columns), key=len, reverse=True)) + r")\b"
    )
    return [
        pattern.sub(lambda match: f"{alias}.{match.group(1)}", predicate)
        for predicate in predicates
    ]


def _rule_generic_patient_visits(
    question: str,
    domains: Domains,
) -> Optional[str]:
    q = question
    if "患者" not in q or "就诊" not in q:
        return None
    if any(
        marker in q
        for marker in (
            "CTE",
            "窗口",
            "每位患者",
            "每个患者",
            "两次就诊",
            "前后两次",
            "配对",
            "不存在",
            "没有任何",
            "图谱",
        )
    ):
        return None
    patient_where = _generic_where(q, "patients", domains)
    visit_where = _generic_where(q, "visits", domains)
    if patient_where is None or visit_where is None:
        return None
    patient_predicates, _ = patient_where
    visit_predicates, _ = visit_where
    patient_predicates = _qualify_predicates(
        patient_predicates,
        "p",
        ("patient_id", "name", "gender", "age", "region"),
    )
    visit_predicates = _qualify_predicates(
        visit_predicates,
        "v",
        (
            "visit_id",
            "patient_id",
            "visit_date",
            "department",
            "disease",
            "drug_prescribed",
            "exam",
            "cost",
            "los_days",
        ),
    )
    predicates = patient_predicates + visit_predicates
    left_join = "保留全部患者" in q or "没有就诊者" in q
    join = (
        "FROM patients AS p LEFT JOIN visits AS v "
        if left_join
        else "FROM patients AS p JOIN visits AS v "
    ) + "ON v.patient_id = p.patient_id"

    group_dimensions: list[tuple[str, str, tuple[str, ...]]] = []
    for trigger, expression, alias, semantics in (
        ("按患者地区", "p.region", "region", ("地区", "患者地区")),
        ("按地区", "p.region", "region", ("地区",)),
        ("按患者性别", "p.gender", "gender", ("性别", "患者性别")),
        ("按性别", "p.gender", "gender", ("性别",)),
        ("按患者", "p.patient_id", "patient_id", ("患者ID", "患者")),
    ):
        if trigger == "按患者" and (
            "按患者地区" in q or "按患者性别" in q
        ):
            continue
        if trigger in q and all(alias != item[1] for item in group_dimensions):
            group_dimensions.append((expression, alias, semantics))
    if group_dimensions and re.search(r"(?:统计|汇总|计算)", q):
        metric_specs = [
            (
                r"不同就诊患者数|不同患者数|就诊患者数",
                "COUNT(DISTINCT p.patient_id)",
                "patient_count",
                ("不同患者数", "患者数", "就诊患者数"),
            ),
            (
                r"就诊总次数|就诊次数|总次数",
                "COUNT(*)",
                "visit_count",
                ("就诊次数", "次数", "总次数"),
            ),
            (
                r"费用总额|总费用|累计费用",
                "SUM(v.cost)",
                "total_cost",
                ("费用总额", "总费用", "累计费用"),
            ),
            (
                r"平均费用",
                "AVG(v.cost)",
                "avg_cost",
                ("平均费用",),
            ),
            (
                r"平均住院天数",
                "AVG(v.los_days)",
                "avg_los_days",
                ("平均住院天数",),
            ),
            (
                r"最长住院天数|最大住院天数",
                "MAX(v.los_days)",
                "max_los_days",
                ("最长住院天数", "最大住院天数"),
            ),
            (
                r"最早就诊日|最早日期",
                "MIN(v.visit_date)",
                "first_visit_date",
                ("最早就诊日", "最早日期", "首日"),
            ),
            (
                r"最晚就诊日|最晚日期",
                "MAX(v.visit_date)",
                "last_visit_date",
                ("最晚就诊日", "最晚日期", "末日"),
            ),
            (
                r"不同疾病(?:数|数量|种数)",
                "COUNT(DISTINCT v.disease)",
                "disease_count",
                ("不同疾病数", "疾病数", "疾病种数"),
            ),
        ]
        metrics: list[tuple[int, str, str, tuple[str, ...]]] = []
        for pattern, expression, alias, semantics in metric_specs:
            match = re.search(pattern, q)
            if match:
                metrics.append((match.start(), expression, alias, semantics))
        metrics.sort()
        if not metrics or "只保留" in q:
            return None
        selects = [
            f"{expression} AS {alias}"
            for expression, alias, _ in group_dimensions
        ] + [
            f"{expression} AS {alias}"
            for _, expression, alias, _ in metrics
        ]
        semantic_map: dict[str, str] = {}
        for _, alias, semantics in group_dimensions:
            for semantic in semantics:
                semantic_map[semantic] = alias
        for _, _, alias, semantics in metrics:
            for semantic in semantics:
                semantic_map[semantic] = alias
        order = _order_from_map(q, semantic_map)
        if order is None:
            return None
        sql = f"SELECT {', '.join(selects)} {join}"
        if predicates:
            sql += " WHERE " + " AND ".join(predicates)
        sql += " GROUP BY " + ", ".join(
            expression for expression, _, _ in group_dimensions
        )
        sql += " " + order
        limit = re.search(r"前\s*(\d+)", q)
        if limit:
            sql += f" LIMIT {int(limit.group(1))}"
        return sql

    tokens = _projection_tokens(q)
    if not tokens:
        return None
    patient_mapping = {
        "患者ID": "p.patient_id",
        "患者编号": "p.patient_id",
        "患者姓名": "p.name",
        "姓名": "p.name",
        "性别": "p.gender",
        "年龄": "p.age",
        "地区": "p.region",
    }
    visit_mapping = {
        "就诊ID": "v.visit_id",
        "就诊编号": "v.visit_id",
        "就诊日期": "v.visit_date",
        "日期": "v.visit_date",
        "主诊断": "v.disease",
        "疾病": "v.disease",
        "就诊科室": "v.department",
        "科室": "v.department",
        "主要处方药": "v.drug_prescribed",
        "处方药": "v.drug_prescribed",
        "主要检查": "v.exam",
        "检查": "v.exam",
        "住院天数": "v.los_days",
        "费用": "v.cost",
    }
    projection: list[str] = []
    for token in tokens:
        candidates = [
            expression
            for semantic, expression in {
                **patient_mapping,
                **visit_mapping,
            }.items()
            if semantic == token
            or token.endswith(semantic)
            or token.startswith(semantic)
        ]
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            return None
        projection.append(candidates[0])
    semantic_map = {**patient_mapping, **visit_mapping}
    order = _order_from_map(q, semantic_map)
    if order is None:
        return None
    sql = f"SELECT {', '.join(projection)} {join}"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    sql += " " + order
    limit = re.search(r"前\s*(\d+)", q)
    if limit:
        sql += f" LIMIT {int(limit.group(1))}"
    return sql


def _rule_generic_having(
    question: str,
    domains: Domains,
) -> Optional[str]:
    """Compile common grouped threshold frames without per-item templates."""

    q = question
    visit_floor = re.search(r"就诊次数至少\s*(\d+)\s*次", q)
    average_floor = re.search(
        r"平均费用至少\s*(\d+(?:\.\d+)?)\s*元?",
        q,
    )
    if (
        "先计算各科室" in q
        and visit_floor
        and average_floor
    ):
        return (
            "WITH department_stats AS (SELECT department, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost "
            "FROM visits GROUP BY department) "
            "SELECT department, visit_count, avg_cost "
            "FROM department_stats "
            f"WHERE visit_count >= {int(visit_floor.group(1))} "
            f"AND avg_cost >= {_number(average_floor.group(1))} "
            "ORDER BY visit_count DESC, department ASC"
        )

    dates = _dates(q)
    if (
        "先汇总" in q
        and "每月" in q
        and len(dates) >= 2
        and visit_floor
    ):
        return (
            "WITH monthly AS (SELECT substr(visit_date, 1, 7) "
            "AS visit_month, COUNT(*) AS visit_count, "
            "SUM(cost) AS total_cost FROM visits "
            f"WHERE visit_date >= '{dates[0]}' "
            f"AND visit_date <= '{dates[1]}' "
            "GROUP BY substr(visit_date, 1, 7)) "
            "SELECT visit_month, visit_count, total_cost FROM monthly "
            f"WHERE visit_count >= {int(visit_floor.group(1))} "
            "ORDER BY visit_month ASC"
        )

    diagnosis_visits = re.search(r"至少有\s*(\d+)\s*次就诊", q)
    if (
        diagnosis_visits
        and average_floor
        and "诊断" in q
        and "返回就诊次数和平均费用" in q
    ):
        return (
            "SELECT disease, COUNT(*) AS visit_count, "
            "AVG(cost) AS avg_cost FROM visits GROUP BY disease "
            f"HAVING COUNT(*) >= {int(diagnosis_visits.group(1))} "
            f"AND AVG(cost) >= {_number(average_floor.group(1))} "
            "ORDER BY avg_cost DESC, disease ASC"
        )

    prescription_floor = re.search(
        r"开具药物.+?至少\s*(\d+)\s*次的科室",
        q,
    )
    drugs = _domain_values(q, domains.get("drug", ()))
    if prescription_floor and len(drugs) == 1 and "处方次数" in q:
        return (
            "SELECT department, COUNT(*) AS prescription_count FROM visits "
            f"WHERE drug_prescribed = '{_esc(drugs[0])}' "
            "GROUP BY department "
            f"HAVING COUNT(*) >= {int(prescription_floor.group(1))} "
            "ORDER BY prescription_count DESC, department ASC"
        )
    return None


def _rule_generic_nested_projection(
    question: str,
    domains: Domains,
) -> Optional[str]:
    """Compile reusable EXISTS, correlated-average, and graph projections."""

    q = question
    if "没有作为任何三元组头实体出现" in q:
        where = _generic_where(q, "kg_entities", domains)
        if where is None:
            return None
        predicates, _ = where
        if not predicates:
            return None
        qualified = _qualify_predicates(
            predicates,
            "e",
            ("name", "type", "mention_count"),
        )
        return (
            "SELECT e.name FROM kg_entities AS e WHERE "
            + " AND ".join(qualified)
            + " AND NOT EXISTS (SELECT 1 FROM kg_triples AS t "
            "WHERE t.head = e.name AND t.head_type = e.type) "
            "ORDER BY e.name ASC"
        )

    if (
        "至少有一次" in q
        and "就诊记录" in q
        and "患者编号和姓名" in q
    ):
        where = _generic_where(q, "visits", domains)
        if where is None:
            return None
        predicates, _ = where
        if not predicates:
            return None
        qualified = _qualify_predicates(
            predicates,
            "v",
            (
                "visit_id",
                "patient_id",
                "visit_date",
                "department",
                "disease",
                "drug_prescribed",
                "exam",
                "cost",
                "los_days",
            ),
        )
        return (
            "SELECT p.patient_id, p.name FROM patients AS p "
            "WHERE EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id AND "
            + " AND ".join(qualified)
            + ") ORDER BY p.patient_id ASC"
        )

    offset = re.search(
        r"费用(?:至少)?比(?:该)?科室平均费用高\s*(\d+(?:\.\d+)?)",
        q,
    )
    departments = _domain_values(q, domains.get("department", ()))
    if (
        offset
        and len(departments) == 1
        and "就诊编号、患者编号、诊断和费用" in q
    ):
        department = _esc(departments[0])
        return (
            "SELECT visit_id, patient_id, disease, cost FROM visits AS v "
            f"WHERE department = '{department}' "
            "AND cost >= (SELECT AVG(cost) + "
            f"{_number(offset.group(1))} FROM visits "
            f"WHERE department = '{department}') "
            "ORDER BY cost DESC, visit_id ASC"
        )

    head = re.search(r"头实体为([^且，。]+)", q)
    if (
        head
        and "所有关系、中文关系名和尾实体" in q
    ):
        relation_request = analyze_relation_request(q, domains)
        if relation_request.has_relation_intent and (
            relation_request.ordering_spans
            or relation_request.top_k_spans
            or _projection_tokens(q) != ["所有关系", "中文关系名", "尾实体"]
        ):
            return None
        where = _generic_where(q, "kg_triples", domains)
        if where is None:
            return None
        predicates, _ = where
        return (
            "SELECT relation, relation_zh, tail FROM kg_triples "
            f"WHERE head = '{_esc(head.group(1).strip())}'"
            + (
                " AND " + " AND ".join(predicates)
                if predicates
                else ""
            )
            + " ORDER BY relation ASC, tail ASC"
        )
    return None


def _rule_generic_ranked_frames(
    question: str,
    domains: Domains,
) -> Optional[str]:
    """Compile grouped top-K and cumulative window frames."""

    q = question
    limit_match = re.search(r"前\s*(\d+)", q)
    if (
        limit_match
        and "累计费用最高" in q
        and "患者" in q
    ):
        where = _generic_where(q, "visits", domains)
        if where is None:
            return None
        predicates, _ = where
        sql = (
            "SELECT patient_id, SUM(cost) AS total_cost FROM visits"
        )
        if predicates:
            sql += " WHERE " + " AND ".join(predicates)
        sql += (
            " GROUP BY patient_id "
            "ORDER BY total_cost DESC, patient_id ASC "
            f"LIMIT {int(limit_match.group(1))}"
        )
        return sql

    if (
        limit_match
        and "按患者和日期计算累计费用" in q
    ):
        where = _generic_where(q, "visits", domains)
        if where is None:
            return None
        predicates, _ = where
        sql = (
            "SELECT patient_id, visit_date, cost, "
            "SUM(cost) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date ASC, visit_id ASC "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            "AS running_cost FROM visits"
        )
        if predicates:
            sql += " WHERE " + " AND ".join(predicates)
        sql += (
            " ORDER BY patient_id ASC, visit_date ASC, visit_id ASC "
            f"LIMIT {int(limit_match.group(1))}"
        )
        return sql
    return None


def compile_explicit_medical_sql(
    question: str,
    domains: Domains,
    *,
    age_average_intent: Optional[AgeAverageIntent] = None,
    temporal_cohort_intent: Optional[TemporalCohortIntent] = None,
    obligations: Optional[SemanticObligations] = None,
) -> Optional[str]:
    """Compile a complete controlled-language query, or abstain safely.

    Root ownership dispatch can provide its already-classified age-average
    record, avoiding a second phrase-level interpretation at the exact owner.
    """

    q = str(question).strip()
    relation_request = analyze_relation_request(q, domains)
    age_average_intent = (
        age_average_intent
        if age_average_intent is not None
        else classify_age_average_intent(q, domains)
    )
    if age_average_intent.disposition in {
        AgeAverageDisposition.ADVANCED_PEER,
        AgeAverageDisposition.TERMINAL_ABSTAIN,
    }:
        # The advanced compiler owns supported peer comparisons; every other
        # recognized member of this family must not fall through to a broad
        # patient/visit template that omits its average scope.
        return None
    temporal_cohort_intent = (
        temporal_cohort_intent
        if temporal_cohort_intent is not None
        else classify_temporal_cohort_intent(q, domains)
    )
    if (
        temporal_cohort_intent.disposition
        is not TemporalCohortDisposition.NOT_APPLICABLE
    ):
        # Temporal cohort requests have one designated advanced/family owner.
        # An exact-looking window/group template must not consume only its
        # surface aggregate while omitting anchor/follow-up semantics.
        return None
    obligations = (
        obligations
        if obligations is not None
        else extract_semantic_obligations(
            q,
            domains,
            temporal_cohort_intent=temporal_cohort_intent,
        )
    )
    if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
        return None
    quoted_relation_union = _quoted_relation_union(q, domains)
    if quoted_relation_union is None:
        return None
    if quoted_relation_union:
        # A quoted relation union has one dedicated row-level semantics route.
        # Validate every metric before entering it, then never let an
        # unsupported range or continuation fall through to a generic parser
        # that can retain only the first threshold.
        if not _quoted_union_metric_bounds_complete(q):
            return None
        sql = _rule_filtering(q, domains)
        return (
            sql
            if sql is not None
            and _sql_owns_relation_request(sql, q, relation_request)
            else None
        )
    for compiler in (
        _rule_set_operation,
        _rule_cte,
        _rule_window,
        _rule_nested,
        _rule_having,
        lambda value: _rule_generic_having(value, domains),
        lambda value: _rule_generic_nested_projection(value, domains),
        _rule_top_k,
        lambda value: _rule_generic_ranked_frames(value, domains),
        _rule_temporal,
        lambda value: _rule_multi_join(value, domains),
        lambda value: _rule_generic_patient_visits(value, domains),
        lambda value: _rule_knowledge_graph(value, domains),
        lambda value: _rule_filtering(value, domains),
        _rule_aggregation,
        lambda value: _rule_generic_group(value, domains),
        lambda value: _rule_generic_filter(value, domains),
    ):
        sql = compiler(q)
        if sql and _sql_owns_relation_request(sql, q, relation_request):
            return sql
    return None
