"""High-specificity semantic families for the four-table medical schema.

These compilers cover CTE, window, correlated, set, and graph-relational
compositions that are too structured for the ordinary row/group compiler.  A
family is selected only when its structural cues and every literal-bearing slot
are explicit in the question.  Unsupported or incomplete compositions abstain.
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
from dki.analysis.nl2sql_semantic_common import (
    age_predicates,
    answer_items,
    date_predicates,
    dates,
    domain_values,
    esc,
    gender_predicate,
    limit,
    relation_values,
    RegionFilterDisposition,
    classify_region_filter_intent,
    region_predicates,
)
from dki.analysis.nl2sql_semantic_kg import (
    _edge_metric_bounds,
    _entity_type,
    _fixed_result_modifiers_are_absent,
    _has_clinical_scope,
    _metric_threshold_intent,
    _single_relation_metric_bounds,
    _visit_metric_bounds,
    _visit_predicates,
    analyze_relation_request,
)
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortIntent,
    classify_temporal_cohort_intent,
)

Domains = Mapping[str, Sequence[str]]
_NUM = r"\d+(?:\.\d+)?"

_OPERATORS: tuple[tuple[str, str], ...] = (
    (r"(?:至少|不低于|不少于|达到|达|下限(?:为|是)?|≥|>=)", ">="),
    (r"(?:超过|高于|大于|多于|>)", ">"),
    (r"(?:不超过|不高于|至多|上限(?:为|是)?|≤|<=)", "<="),
    (r"(?:低于|小于|少于|<)", "<"),
    (r"(?:等于|恰好为|为|=)", "="),
)


def _number(value: str) -> str:
    parsed = float(value)
    return str(int(parsed)) if parsed.is_integer() else f"{parsed:g}"


def _threshold(question: str, labels: Sequence[str]) -> Optional[tuple[str, str]]:
    label = "(?:" + "|".join(re.escape(item) for item in sorted(set(labels), key=len, reverse=True)) + ")"
    matches: list[tuple[int, str, str]] = []
    for operator_pattern, operator in _OPERATORS:
        patterns = (
            rf"{label}\s*(?:均|都|分别|各自)?\s*{operator_pattern}\s*({_NUM})",
            rf"{operator_pattern}\s*(?:有|达到|为)?\s*({_NUM})\s*(?:个|次|位|名|条|种|人|元|天)?\s*(?:的)?\s*{label}",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, question):
                matches.append((match.start(), operator, _number(match.group(1))))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1], matches[0][2]


def _range(question: str, labels: Sequence[str]) -> Optional[tuple[str, str]]:
    label = "(?:" + "|".join(re.escape(item) for item in sorted(set(labels), key=len, reverse=True)) + ")"
    for pattern in (
        rf"{label}\s*(?:在|介于|范围为|为)?\s*({_NUM})\s*(?:至|到|—|-)\s*({_NUM})",
        rf"{label}\s*[\[【]\s*({_NUM})\s*[,，]\s*({_NUM})\s*[\]】]",
    ):
        match = re.search(pattern, question)
        if match:
            return _number(match.group(1)), _number(match.group(2))
    return None


def _age_filters(question: str, alias: str = "p") -> list[str]:
    """Return every parsed age predicate for a family patient scope."""

    return age_predicates(question, alias=alias)


def _age_lower(question: str, alias: str = "p") -> Optional[tuple[str, str]]:
    """Return a lone lower age bound for threshold-only family semantics."""

    predicates = _age_filters(question, alias=alias)
    if len(predicates) != 1:
        return None
    column = f"{alias}.age" if alias else "age"
    match = re.fullmatch(rf"{re.escape(column)}\s*(>=|>)\s*(\d+)", predicates[0])
    return (match.group(1), match.group(2)) if match else None


def _gender_value(question: str) -> Optional[str]:
    predicate = gender_predicate(question)
    if predicate is None:
        return None
    match = re.search(r"'([^']+)'", predicate)
    return match.group(1) if match else None


def _region_values(question: str, domains: Domains) -> list[str]:
    return domain_values(question, domains.get("region", ()))


def _patient_predicates(
    question: str,
    domains: Domains,
    *,
    alias: str = "p",
    include_age: bool = True,
    include_gender: bool = True,
    include_region: bool = True,
) -> list[str]:
    """Return every grounded patient qualifier owned by a family scope."""

    predicates: list[str] = []
    if include_age:
        predicates.extend(age_predicates(question, alias=alias))
    if include_gender:
        gender = gender_predicate(question, alias=alias)
        if gender is not None:
            predicates.append(gender)
    if include_region:
        predicates.extend(region_predicates(question, domains, alias=alias))
    return list(dict.fromkeys(predicates))


def _complete_visit_date_predicates(
    question: str,
    *,
    alias: str = "v",
) -> Optional[list[str]]:
    """Render a supported visit-date scope without dropping any date span."""

    values = dates(question)
    if len(values) > 2:
        return None
    predicates = date_predicates(question, alias=alias)
    if values and not predicates:
        return None
    return predicates


def _department_values(question: str, domains: Domains) -> list[str]:
    return domain_values(question, domains.get("department", ()))


def _relations(question: str, domains: Domains) -> list[str]:
    return list(dict.fromkeys(relation_values(question, domains)))


def _single_relation_family_value(question: str, domains: Domains) -> Optional[str]:
    """Return the one relation a fixed single-edge family can faithfully own.

    Fixed one-relation templates must retain raw occurrence and connective
    semantics until admission.  Deduplicating relation values first could turn
    a collection, ordinal, or repeated-edge request into an ordinary equality
    predicate and silently lose the requested graph role.
    """

    request = analyze_relation_request(question, domains)
    if (
        len(request.occurrences) != 1
        or request.errors
        or not request.metric_complete
    ):
        return None
    # With one grounded relation, words such as ``同时达到`` or ``同时具备``
    # can only join metric qualifiers; they cannot introduce a second edge.
    # The raw request analyzer conservatively labels those phrases
    # ``required_all``.  Keep the family strict about occurrence count while
    # accepting that one-relation metric conjunction.
    if request.connective not in {"single", "required_all"}:
        return None
    return request.occurrences[0].value


def _tail_mention_average_bounds(
    question: str,
) -> Optional[list[tuple[str, str]]]:
    """Parse numeric tail-mention bounds beside one owned same-type average.

    Tail coverage already renders its strict same-type-average predicate.  Mask
    exactly that supported comparison before giving any additive numeric bounds
    to the shared row-bound parser, so an unsupported mention qualifier remains
    terminal rather than being mistaken for the owned aggregate comparison.
    """

    metric = r"(?:提及次数|提及数|提及量|mention_count|热度)"
    same_type_average = (
        r"(?:其|该)?\s*(?:同类型|实体类型|该类型|同类)\s*(?:实体)?\s*(?:的)?\s*"
        r"(?:平均(?:值|数)?|均值|平均水平)"
    )
    match = re.search(
        rf"(?:尾实体|尾端|目标)?\s*(?:实体)?\s*{metric}\s*"
        rf"(?:高于|超过|大于|>)\s*{same_type_average}",
        question,
        re.I,
    )
    if match is None:
        match = re.search(
            rf"(?:高于|超过|大于|>)\s*{same_type_average}\s*(?:的)?\s*{metric}",
            question,
            re.I,
        )
    if match is None:
        return None
    residual = question[:match.start()] + " " + question[match.end():]
    bounds = _single_relation_metric_bounds(
        residual,
        "提及次数",
        ("提及数", "提及量", "mention_count", "热度"),
    )
    if bounds is None and not _metric_threshold_intent(
        residual,
        "提及次数",
        ("提及数", "提及量", "mention_count", "热度"),
    ):
        # A compatible explicit order such as ``按提及量降序`` is not an
        # additional row predicate.  The fixed family already owns that order.
        return []
    return bounds


def _edge_domains(domains: Domains, relations: Sequence[str]) -> Domains:
    """Expose fixed family relation roles to the shared edge-scope parser."""

    values = list(domains.get("kg_relation_zh", ()))
    values.extend(relation for relation in relations if relation not in values)
    return {**domains, "kg_relation_zh": tuple(values)}


def _date_range(question: str) -> Optional[tuple[str, str]]:
    values = dates(question)
    return (values[0], values[1]) if len(values) >= 2 else None


def _since_date(question: str) -> Optional[str]:
    values = dates(question)
    if len(values) != 1:
        return None
    value = re.escape(values[0])
    if re.search(rf"(?:自|从)\s*{value}|{value}\s*(?:起|以来|以后|之后|开始|后)", question):
        return values[0]
    return None


def _year(question: str) -> Optional[str]:
    match = re.search(r"(20\d{2})\s*年", question)
    return match.group(1) if match else None


def _top_k(question: str) -> Optional[int]:
    value = limit(question)
    if value is not None:
        return value
    for pattern in (
        r"(?:前|最高的?|最大的?|最昂贵的?|保留)\s*(\d+)\s*(?:个|名|位|人|种|条)?",
        r"第\s*1\s*(?:至|到|—|-)\s*第?\s*(\d+)\s*档",
        r"(?:编号|档位|排名|名次)\s*(?:不超过|≤|<=)\s*(\d+)",
    ):
        match = re.search(pattern, question)
        if match:
            return int(match.group(1))
    return None


def _has_explicit_projection(question: str) -> bool:
    """Whether a later projection-aware compiler must retain output control."""

    return bool(answer_items(question))


def _fixed_kg_result_shape_is_unmodified(question: str, domains: Domains) -> bool:
    """Allow a fixed KG family only without caller-controlled result shape.

    The legacy family lowerers below expose a stable projection and result
    order. They cannot use that default as an owner for an explicit output
    list, ordering clause, or pagination request. Reuse the structured KG
    modifier parser so marker-free ordering and alternate Top-K wording are
    rejected consistently with the projection-aware graph compiler.
    """

    return _fixed_result_modifiers_are_absent(question, domains)


def _fixed_projection_is_compatible(
    question: str,
    allowed_item_patterns: Sequence[str],
    *,
    summary_patterns: Sequence[str] = (),
) -> bool:
    """Reject explicit output items that a fixed family cannot render.

    A fixed family may accept an explicit answer clause when every requested
    item is one of its own output slots.  This is narrower than the old
    all-or-nothing projection deferral and still fails closed for extra fields
    such as a drug column on a visit Top-K template or an entity type on a
    two-column entity template.  Summary phrases such as ``完整路径`` or
    ``记录详情`` are accepted only when supplied by that family.
    """

    items = answer_items(question)
    if not items:
        return True
    for item in items:
        normalized = re.sub(r"\s+", "", item)
        if any(re.search(pattern, normalized, re.I) for pattern in summary_patterns):
            continue
        if not any(re.search(pattern, normalized, re.I) for pattern in allowed_item_patterns):
            return False
    return True


def _kg_only_clinical_scope_is_absent(question: str, domains: Domains) -> bool:
    """Return whether a KG-only template can safely use global graph rows.

    These templates do not join patients or visits.  A clinical cohort, or a
    temporal scope that would have to apply to visits, cannot be silently
    reinterpreted as a global graph query.
    """

    return not _has_clinical_scope(question, domains)


def _visit_count_threshold(question: str) -> Optional[tuple[str, str]]:
    parsed = _threshold(
        question,
        (
            "就诊次数", "就诊量", "记录量", "记录数", "次数", "实际记录",
            "实际就诊", "就诊", "记录",
        ),
    )
    if parsed:
        return parsed
    patterns = (
        (r"(?:至少|不少于|不低于|达到|达|≥|>=)\s*(?:有|含|出现)?\s*(\d+)\s*次(?:以上)?(?:就诊|记录)?", ">="),
        (r"(?:至少|不少于|不低于|达到|达|≥|>=)\s*(?:就诊|记录)\s*(\d+)\s*次", ">="),
        (r"(?:就诊|记录|出现)\s*(\d+)\s*次(?:以上|或更多)", ">="),
        (r"(\d+)\s*次以上(?:实际)?(?:就诊|记录)", ">="),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, question)
        if match:
            return operator, match.group(1)
    return None


def _total_cost_threshold(question: str) -> Optional[tuple[str, str]]:
    return _threshold(
        question,
        ("个人总费用", "个人累计费用", "累计费用", "累计花费", "总费用", "费用总额", "费用合计", "花费合计", "总额", "费用"),
    )


def _confidence_threshold(question: str) -> Optional[tuple[str, str]]:
    return _threshold(question, ("边置信度", "关系置信度", "置信度"))


def _edge_count_threshold(question: str) -> Optional[tuple[str, str]]:
    parsed = _threshold(question, ("观察次数", "出现次数", "关系次数", "计数", "次数"))
    if parsed:
        return parsed
    match = re.search(
        r"(?:至少|不少于|不低于|达到|≥|>=)\s*(\d+)\s*次|"
        r"(\d+)\s*次(?:计数|出现|观察)",
        question,
    )
    return (">=", next(group for group in match.groups() if group)) if match else None


def _per_hop_thresholds(
    question: str,
    labels: Sequence[str],
    *,
    relations: Sequence[str] = (),
    domains: Optional[Domains] = None,
) -> Optional[list[list[tuple[str, str]]]]:
    """Bind every two-hop edge bound, preserving ranges and named roles.

    The graph compiler owns the canonical scope parser.  Reusing it here keeps
    fixed family templates fail-closed for ambiguous ownership rather than
    copying the first threshold across both hops.
    """

    if not labels:
        return None
    bounds = _edge_metric_bounds(
        question,
        edge_count=2,
        subject=labels[0],
        aliases=tuple(labels[1:]),
        relations=relations,
        domains=domains,
    )
    if bounds is None or not all(bounds):
        return None
    return bounds


def _optional_edge_metric_bounds(
    question: str,
    *,
    subject: str,
    aliases: Sequence[str],
    relations: Sequence[str],
    domains: Domains,
) -> Optional[list[list[tuple[str, str]]]]:
    """Return optional edge constraints without treating bare metric words as bounds."""

    if not _metric_threshold_intent(question, subject, aliases):
        return [[], []]
    return _edge_metric_bounds(
        question,
        edge_count=2,
        subject=subject,
        aliases=aliases,
        relations=relations,
        domains=domains,
    )


def _compile_episode_summary(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"新一段|新簇|新阶段|事件簇|阶段编号|段号|断点|即断开", question):
        return None
    gap_match = re.search(
        r"(?:间隔|距离上次|相隔).{0,8}(?:超过|大于|>)\s*(\d+)\s*天|"
        r"(\d+)\s*天(?:断点|即断开)",
        question,
    )
    if not gap_match:
        return None
    gap = next(group for group in gap_match.groups() if group)
    minimum = _visit_count_threshold(question)
    if minimum is None:
        return None
    if not re.search(r"起止|首末|日期范围|持续天数|跨度|历时", question):
        return None
    patient_predicates = _patient_predicates(question, domains)
    source_where = f" WHERE {' AND '.join(patient_predicates)}" if patient_predicates else ""
    return (
        "WITH ordered AS (SELECT p.patient_id, p.name, v.visit_id, v.visit_date, v.cost, "
        "LAG(v.visit_date) OVER (PARTITION BY p.patient_id ORDER BY v.visit_date, v.visit_id) AS previous_date "
        "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        + source_where
        + "), marked AS (SELECT *, CASE WHEN previous_date IS NULL OR "
        f"julianday(visit_date) - julianday(previous_date) > {gap} THEN 1 ELSE 0 END AS new_episode FROM ordered), "
        "numbered AS (SELECT *, SUM(new_episode) OVER (PARTITION BY patient_id ORDER BY visit_date, visit_id "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS episode_number FROM marked) "
        "SELECT patient_id, name, episode_number, MIN(visit_date) AS episode_start, "
        "MAX(visit_date) AS episode_end, COUNT(*) AS visit_count, ROUND(SUM(cost), 2) AS total_cost, "
        "CAST(julianday(MAX(visit_date)) - julianday(MIN(visit_date)) AS INTEGER) AS duration_days "
        "FROM numbered GROUP BY patient_id, name, episode_number "
        f"HAVING COUNT(*) {minimum[0]} {minimum[1]} ORDER BY patient_id ASC, episode_number ASC"
    )


def _compile_region_benchmark_topk(question: str, domains: Domains) -> Optional[str]:
    if not re.search(
        r"科室(?:数|覆盖数|多样性).{0,16}(?:地区|区域).{0,8}(?:平均|均值)|"
        r"地区.{0,12}科室.{0,8}(?:平均|均值)|"
        r"地区为基准.{0,16}科室多样性.{0,16}(?:平均|均值)",
        question,
    ):
        return None
    if not re.search(r"每个地区|各地区|按地区|地区分别|地区为基准", question):
        return None
    visit_count = _visit_count_threshold(question)
    total_cost = _total_cost_threshold(question)
    top = _top_k(question)
    if visit_count is None or total_cost is None or top is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    source_where = f" WHERE {' AND '.join(patient_predicates)}" if patient_predicates else ""
    return (
        "WITH patient_stats AS (SELECT p.patient_id, p.name, p.region, "
        "COUNT(v.visit_id) AS visit_count, COUNT(DISTINCT v.department) AS department_count, "
        "SUM(v.cost) AS total_cost FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        + source_where
        + " GROUP BY p.patient_id, p.name, p.region), "
        "region_benchmark AS (SELECT region, AVG(department_count) AS avg_department_count "
        "FROM patient_stats GROUP BY region), "
        "eligible AS (SELECT s.*, b.avg_department_count FROM patient_stats AS s "
        "JOIN region_benchmark AS b ON b.region = s.region WHERE "
        f"s.visit_count {visit_count[0]} {visit_count[1]} AND "
        f"s.total_cost {total_cost[0]} {total_cost[1]} AND s.department_count > b.avg_department_count), "
        "ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_cost DESC, patient_id ASC) "
        "AS region_rank FROM eligible) SELECT region, region_rank, patient_id, name, visit_count, "
        "department_count, ROUND(avg_department_count, 2) AS region_avg_department_count, "
        "ROUND(total_cost, 2) AS total_cost FROM ranked "
        f"WHERE region_rank <= {top} ORDER BY region ASC, region_rank ASC"
    )


def _compile_observed_risk_chain(question: str, domains: Domains) -> Optional[str]:
    if not ("并发症" in question and "用药禁忌" in question):
        return None
    if not re.search(r"零次也保留|即使为零|零次链|观察为零|仍保留", question):
        return None
    if not re.search(r"实际(?:处方|就诊|记录)|匹配处方|共同出现|观察次数", question):
        return None
    if not _fixed_kg_result_shape_is_unmodified(question, domains):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"(?:源|原)(?:疾病|病)", r"并发症", r"(?:禁忌)?药物", r"两边置信度|置信度",
            r"实际(?:处方|就诊|记录)?(?:次数|数)|观察次数",
        ),
        summary_patterns=(r"链条|风险链|完整(?:链|路径)|零次链",),
    ):
        return None
    relations = ("并发症", "用药禁忌")
    edge_domains = _edge_domains(domains, relations)
    confidences = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="置信度",
        aliases=("边置信度", "关系置信度", "confidence"),
        relations=relations,
        domains=edge_domains,
    )
    if confidences is None or not all(confidences):
        shared_confidence = _threshold(
            question,
            ("置信度", "边置信度", "关系置信度"),
        )
        if shared_confidence is not None and re.search(
            r"两类|两种|两项|两边|各自|均|都|同下限|两种证据|图谱边",
            question,
        ):
            confidences = [[shared_confidence], [shared_confidence]]
    counts = _optional_edge_metric_bounds(
        question,
        subject="观察次数",
        aliases=("出现次数", "关系次数", "计数", "count"),
        relations=relations,
        domains=edge_domains,
    )
    observed_count_bounds = _visit_metric_bounds(
        question,
        "实际处方次数",
        (
            "实际处方数",
            "实际处方",
            "实际就诊次数",
            "实际就诊数",
            "实际就诊",
            "实际记录次数",
            "实际记录数",
            "实际记录",
            "匹配处方次数",
            "匹配处方数",
            "匹配处方",
            "共同出现次数",
            "共同出现数",
            "共同出现",
            "observed_prescription_count",
        ),
    )
    since = _since_date(question)
    visit_predicates = _visit_predicates(question, domains)
    if (
        visit_predicates is None
        and since is not None
        and len(dates(question)) == 1
        and re.search(rf"{re.escape(since)}\s*后", question)
    ):
        # This compiler owns the only visit-date slot in its risk-chain shape.
        # The shared parser deliberately leaves bare ``日期后`` unresolved;
        # the development contract uses it as an inclusive lower bound.
        visit_predicates = [f"v.visit_date >= '{since}'"]
    if (
        confidences is None
        or not all(confidences)
        or counts is None
        or observed_count_bounds is None
        or since is None
        or visit_predicates is None
        or f"v.visit_date >= '{since}'" not in visit_predicates
    ):
        return None
    confidence_predicates = [
        f"{alias}.confidence {operator} {value}"
        for alias, bounds in zip(("c", "x"), confidences)
        for operator, value in bounds
    ]
    count_predicates = [
        f"{alias}.count {operator} {value}"
        for alias, bounds in zip(("c", "x"), counts)
        for operator, value in bounds
    ]
    patient_predicates = _patient_predicates(question, domains)
    patient_scope = ""
    if patient_predicates:
        patient_scope = (
            " AND EXISTS (SELECT 1 FROM patients AS p WHERE p.patient_id = v.patient_id AND "
            + " AND ".join(patient_predicates)
            + ")"
        )
    # Row-level visit qualifiers belong in the outer join: putting them in a
    # WHERE clause would discard the zero-observation chains this family owns.
    visit_join_predicates = [
        "v.disease = r.source_disease",
        "v.drug_prescribed = r.contraindicated_drug",
        *visit_predicates,
    ]
    return (
        "WITH risk_chains AS (SELECT c.head AS source_disease, c.tail AS complication, "
        "x.head AS contraindicated_drug, c.confidence AS complication_confidence, "
        "x.confidence AS contraindication_confidence FROM kg_triples AS c "
        "JOIN kg_triples AS x ON x.tail = c.tail AND x.relation_zh = '用药禁忌' "
        "WHERE c.relation_zh = '并发症' AND "
        + " AND ".join(confidence_predicates + count_predicates)
        + ") SELECT r.source_disease, r.complication, r.contraindicated_drug, "
        "r.complication_confidence, r.contraindication_confidence, "
        "COUNT(v.visit_id) AS observed_prescription_count FROM risk_chains AS r "
        "LEFT JOIN visits AS v ON "
        + " AND ".join(visit_join_predicates)
        + patient_scope
        + " GROUP BY r.source_disease, r.complication, "
        "r.contraindicated_drug, r.complication_confidence, r.contraindication_confidence "
        + (
            "HAVING "
            + " AND ".join(
                f"COUNT(v.visit_id) {operator} {value}"
                for operator, value in observed_count_bounds
            )
            + " "
            if observed_count_bounds
            else ""
        )
        + "ORDER BY observed_prescription_count DESC, r.source_disease ASC, "
        "r.complication ASC, r.contraindicated_drug ASC"
    )


def _retention_upper(question: str) -> Optional[int]:
    for pattern in (
        r"(?:偏移月|月份偏移|第)?\s*0\s*(?:至|到|—|-)\s*(?:第)?\s*(\d+)",
        r"偏移(?:不超过|至多|≤|<=)\s*(\d+)\s*个月",
    ):
        match = re.search(pattern, question)
        if match:
            return int(match.group(1))
    return None


def _compile_cohort_retention(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"首诊月|首次就诊月|首次就诊月份", question):
        return None
    if not re.search(r"留存率|占原队列比例|队列规模|活跃人数", question):
        return None
    upper = _retention_upper(question)
    age_filters = _age_filters(question, alias="")
    gender = _gender_value(question)
    if upper is None or not age_filters or gender is None:
        return None
    return (
        "WITH eligible AS (SELECT patient_id FROM patients WHERE "
        f"gender = '{esc(gender)}' AND {' AND '.join(age_filters)}), "
        "first_month AS (SELECT e.patient_id, MIN(substr(v.visit_date, 1, 7)) AS cohort_month "
        "FROM eligible AS e JOIN visits AS v ON v.patient_id = e.patient_id GROUP BY e.patient_id), "
        "activity AS (SELECT DISTINCT f.patient_id, f.cohort_month, substr(v.visit_date, 1, 7) AS activity_month, "
        "(CAST(substr(v.visit_date, 1, 4) AS INTEGER) * 12 + CAST(substr(v.visit_date, 6, 2) AS INTEGER) "
        "- CAST(substr(f.cohort_month, 1, 4) AS INTEGER) * 12 "
        "- CAST(substr(f.cohort_month, 6, 2) AS INTEGER)) AS month_offset "
        "FROM first_month AS f JOIN visits AS v ON v.patient_id = f.patient_id), "
        "cohort_size AS (SELECT cohort_month, COUNT(*) AS cohort_patient_count "
        "FROM first_month GROUP BY cohort_month) SELECT a.cohort_month, a.month_offset, "
        "COUNT(DISTINCT a.patient_id) AS active_patient_count, c.cohort_patient_count, "
        "ROUND(1.0 * COUNT(DISTINCT a.patient_id) / c.cohort_patient_count, 4) AS retention_rate "
        "FROM activity AS a JOIN cohort_size AS c ON c.cohort_month = a.cohort_month "
        f"WHERE a.month_offset BETWEEN 0 AND {upper} GROUP BY a.cohort_month, a.month_offset, "
        "c.cohort_patient_count ORDER BY a.cohort_month ASC, a.month_offset ASC"
    )


def _compile_older_only_diseases(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"差集|减去|仅见于|没有出现在|未见于|却没有", question):
        return None
    if not re.search(r"提及|知识库热度|图谱", question) or "疾病" not in question:
        return None
    year = _year(question)
    age_filters = _age_filters(question, alias="p")
    age = _age_lower(question, alias="p")
    inclusive_age = re.search(r"(\d+)\s*岁(?:及)?以上", question)
    if inclusive_age is not None:
        age = (">=", inclusive_age.group(1))
    if age is None:
        match = re.search(
            r"(\d+)\s*岁(?:及以上|以上)|年龄\s*(?:≥|>=)\s*(\d+)\s*岁?",
            question,
        )
        if match:
            age = (">=", next(group for group in match.groups() if group))
    if age is None and not age_filters:
        match = re.search(r"高龄阈值(?:为|是)?\s*(\d+)\s*岁", question)
        if match:
            age = (">=", match.group(1))
    top = _top_k(question)
    visit_predicates = _visit_predicates(question, domains)
    if (
        year is None
        or age is None
        or age[0] != ">="
        or top is None
        or visit_predicates is None
    ):
        return None
    threshold = age[1]
    cohort_predicates = _patient_predicates(question, domains, include_age=False)
    older_predicates = [
        f"strftime('%Y', v.visit_date) = '{year}'",
        f"p.age >= {threshold}",
        *cohort_predicates,
        *visit_predicates,
    ]
    younger_predicates = [
        f"strftime('%Y', v.visit_date) = '{year}'",
        f"p.age < {threshold}",
        *cohort_predicates,
        *visit_predicates,
    ]
    return (
        "WITH older_only AS (SELECT DISTINCT v.disease FROM visits AS v "
        "JOIN patients AS p ON p.patient_id = v.patient_id WHERE "
        + " AND ".join(older_predicates)
        + " EXCEPT SELECT DISTINCT v.disease FROM visits AS v "
        "JOIN patients AS p ON p.patient_id = v.patient_id WHERE "
        + " AND ".join(younger_predicates)
        + "), enriched AS (SELECT o.disease, ke.mention_count, COUNT(v.visit_id) AS older_visit_count "
        "FROM older_only AS o JOIN kg_entities AS ke ON ke.name = o.disease "
        "JOIN visits AS v ON v.disease = o.disease JOIN patients AS p ON p.patient_id = v.patient_id WHERE "
        + " AND ".join(older_predicates)
        + " GROUP BY o.disease, ke.mention_count), ranked AS (SELECT *, ROW_NUMBER() OVER "
        "(ORDER BY mention_count DESC, disease ASC) AS mention_rank FROM enriched) "
        "SELECT mention_rank, disease, mention_count, older_visit_count FROM ranked "
        f"WHERE mention_rank <= {top} ORDER BY mention_rank ASC"
    )


def _compile_required_relation_visits(question: str, domains: Domains) -> Optional[str]:
    relations = _relations(question, domains)
    if len(relations) != 2 or not re.search(r"每类关系|逐一满足|完整覆盖|要求集合|关系清单|两项要求均|每项要求", question):
        return None
    # The graph compiler is projection-aware for this shape. This legacy
    # fixed template must defer instead of omitting caller-controlled output
    # or result modifiers.
    if not _fixed_kg_result_shape_is_unmodified(question, domains):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"疾病", r"提及(?:次数|量|数)|知识热度", r"就诊(?:次数|量|数)|次数|记录数",
            r"平均(?:费用|花费)|费用均值",
        ),
        summary_patterns=(r"知识热度和费用均值", r"热度.*费用",),
    ):
        return None
    edge_domains = _edge_domains(domains, relations)
    confidences = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="置信度",
        aliases=("边置信度", "关系置信度", "confidence"),
        relations=relations,
        domains=edge_domains,
    )
    if confidences is None or not all(confidences):
        match = re.search(
            r"(?:每类至少一边|相应边).*?(?:达到|至少|不低于|≥|>=)\s*(\d+(?:\.\d+)?)",
            question,
        )
        if match:
            confidence = (">=", _number(match.group(1)))
            confidences = [[confidence], [confidence]]
    counts = _optional_edge_metric_bounds(
        question,
        subject="观察次数",
        aliases=("出现次数", "关系次数", "计数", "count"),
        relations=relations,
        domains=edge_domains,
    )
    visit_bounds = _visit_metric_bounds(
        question,
        "就诊次数",
        ("就诊量", "记录量", "记录数", "实际记录", "实际就诊", "就诊", "记录"),
    )
    if not visit_bounds:
        match = re.search(
            r"(\d+)\s*次以上(?:实际)?(?:就诊|记录)|"
            r"(?:至少|不少于|不低于|达到|≥|>=)\s*(?:实际)?"
            r"(?:就诊|记录)\s*(\d+)\s*次",
            question,
        )
        if match:
            visit_bounds = [(">=", next(group for group in match.groups() if group))]
    if not visit_bounds:
        return None
    average_cost_bounds = _visit_metric_bounds(
        question,
        "平均费用",
        ("费用均值", "平均花费", "单次平均费用"),
    )
    if (
        confidences is None
        or not all(confidences)
        or counts is None
        or not visit_bounds
        or average_cost_bounds is None
        or not re.search(r"平均费用|费用均值|平均花费", question)
    ):
        return None
    values_sql = ", ".join(f"('{esc(value)}')" for value in relations)
    relation_conditions: list[str] = []
    for relation, confidence_bounds, count_bounds in zip(relations, confidences, counts):
        predicates = [
            f"kt.confidence {operator} {value}"
            for operator, value in confidence_bounds
        ]
        predicates.extend(
            f"kt.count {operator} {value}" for operator, value in count_bounds
        )
        relation_conditions.append(
            f"(r.relation_zh = '{esc(relation)}' AND {' AND '.join(predicates)})"
        )
    patient_predicates = _patient_predicates(question, domains)
    visit_date_predicates = _complete_visit_date_predicates(question)
    if visit_date_predicates is None:
        return None
    from_sql = "kg_entities AS ke JOIN visits AS v ON v.disease = ke.name"
    if patient_predicates:
        from_sql += " JOIN patients AS p ON p.patient_id = v.patient_id"
    where = [
        "ke.type = 'Disease'",
        "NOT EXISTS (SELECT 1 FROM required_relation AS r WHERE NOT EXISTS "
        "(SELECT 1 FROM kg_triples AS kt WHERE kt.head = ke.name "
        "AND kt.relation_zh = r.relation_zh AND ("
        + " OR ".join(relation_conditions)
        + ")))"
    ]
    where.extend(patient_predicates)
    where.extend(visit_date_predicates)
    having = [
        f"COUNT(v.visit_id) {operator} {value}"
        for operator, value in visit_bounds
    ]
    having.extend(
        f"AVG(v.cost) {operator} {value}"
        for operator, value in average_cost_bounds
    )
    return (
        f"WITH required_relation(relation_zh) AS (VALUES {values_sql}) "
        "SELECT ke.name AS disease, ke.mention_count, COUNT(v.visit_id) AS visit_count, "
        f"ROUND(AVG(v.cost), 2) AS avg_cost FROM {from_sql} WHERE "
        + " AND ".join(where)
        + f" GROUP BY ke.name, ke.mention_count HAVING {' AND '.join(having)} "
        "ORDER BY ke.mention_count DESC, disease ASC"
    )


def _compile_cohort_evidence(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"队列|人群漏斗|知识覆盖", question):
        return None
    if not re.search(r"证据|知识边|知识覆盖|图谱|疾病拥有.{0,12}关系|疾病是否存在.{0,12}边", question):
        return None
    relation = _single_relation_family_value(question, domains)
    regions = _region_values(question, domains)
    age_filters = _age_filters(question, alias="")
    inclusive_age = re.search(r"(\d+)\s*岁(?:及)?以上", question)
    if inclusive_age is not None:
        age_filters = [f"age >= {inclusive_age.group(1)}"]
    gender = _gender_value(question)
    since = _since_date(question)
    visit_predicates = _visit_predicates(question, domains)
    if (
        visit_predicates is None
        and since is not None
        and len(dates(question)) == 1
        and re.search(rf"{re.escape(since)}\s*后", question)
    ):
        visit_predicates = [f"v.visit_date >= '{since}'"]
    if (
        relation is None
        or len(regions) != 1
        or not age_filters
        or since is None
        or visit_predicates is None
        or f"v.visit_date >= '{since}'" not in visit_predicates
    ):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("关系置信度", "confidence"),
    )
    count_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("出现次数", "关系次数", "计数", "count"),
    )
    if confidence_bounds is None or count_bounds is None:
        return None
    cohort_predicates = [f"region = '{esc(regions[0])}'", *age_filters]
    if gender is not None:
        cohort_predicates.append(f"gender = '{esc(gender)}'")
    evidence_predicates = [
        "kt.head = cv.disease",
        f"kt.relation_zh = '{esc(relation)}'",
    ]
    evidence_predicates.extend(
        f"kt.confidence {operator} {value}"
        for operator, value in confidence_bounds
    )
    evidence_predicates.extend(
        f"kt.count {operator} {value}"
        for operator, value in count_bounds
    )
    return (
        "WITH cohort AS (SELECT patient_id FROM patients WHERE "
        + " AND ".join(cohort_predicates)
        + "), cohort_visits AS (SELECT v.* FROM cohort AS c JOIN visits AS v "
        "ON v.patient_id = c.patient_id WHERE "
        + " AND ".join(visit_predicates)
        + "), "
        "evidence_visits AS (SELECT cv.* FROM cohort_visits AS cv WHERE EXISTS "
        "(SELECT 1 FROM kg_triples AS kt WHERE "
        + " AND ".join(evidence_predicates)
        + ")) SELECT (SELECT COUNT(*) FROM cohort) AS cohort_patient_count, "
        "COUNT(DISTINCT cv.patient_id) AS visited_patient_count, "
        "COUNT(DISTINCT ev.patient_id) AS evidence_patient_count, "
        "COUNT(DISTINCT cv.disease) AS disease_count, "
        "ROUND(COALESCE(SUM(cv.cost), 0), 2) AS total_cost FROM cohort_visits AS cv "
        "LEFT JOIN evidence_visits AS ev ON ev.visit_id = cv.visit_id"
    )


def _quartile_upper(question: str) -> Optional[int]:
    for pattern in (
        r"(?:四分位编号|编号|档位)\s*(?:不超过|≤|<=)\s*(\d+)",
        r"第\s*1\s*(?:至|到|—|-)\s*第?\s*(\d+)\s*档",
        r"(?:前|高费用组).*?(\d+)\s*档",
    ):
        match = re.search(pattern, question)
        if match:
            return int(match.group(1))
    return None


def _compile_ntile_quartiles(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"四分位|四等分|四档", question):
        return None
    if not re.search(r"总费用|累计费用|总额|花费", question):
        return None
    age_filters = _age_filters(question, alias="p")
    visits = _visit_count_threshold(question)
    upper = _quartile_upper(question)
    if not age_filters or visits is None or upper is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    return (
        "WITH patient_totals AS (SELECT p.patient_id, p.name, p.region, "
        "COUNT(v.visit_id) AS visit_count, SUM(v.cost) AS total_cost "
        "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
        f"WHERE {' AND '.join(patient_predicates)} GROUP BY p.patient_id, p.name, p.region "
        f"HAVING COUNT(v.visit_id) {visits[0]} {visits[1]}), "
        "bucketed AS (SELECT patient_id, name, region, visit_count, total_cost, "
        "NTILE(4) OVER (ORDER BY total_cost DESC, patient_id ASC) AS spending_quartile "
        "FROM patient_totals) SELECT spending_quartile, patient_id, name, region, "
        "visit_count, ROUND(total_cost, 2) AS total_cost FROM bucketed "
        f"WHERE spending_quartile <= {upper} ORDER BY spending_quartile ASC, total_cost DESC, patient_id ASC"
    )


def _compile_cohort_department_group(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"队列|建立人群|创建满足|患者集合", question):
        return None
    if not re.search(r"各科室|按科室|科室业务量|科室分别", question):
        return None
    regions = _region_values(question, domains)
    age_filters = _age_filters(question, alias="")
    gender = _gender_value(question)
    since = _since_date(question)
    if len(regions) != 1 or not age_filters or gender is None or since is None:
        return None
    return (
        "WITH cohort AS (SELECT patient_id FROM patients WHERE "
        f"region = '{esc(regions[0])}' AND gender = '{esc(gender)}' "
        f"AND {' AND '.join(age_filters)}) SELECT v.department, COUNT(*) AS visit_count, "
        "COUNT(DISTINCT v.patient_id) AS patient_count, ROUND(SUM(v.cost), 2) AS total_cost "
        "FROM cohort AS c JOIN visits AS v ON v.patient_id = c.patient_id "
        f"WHERE v.visit_date >= '{since}' GROUP BY v.department "
        "ORDER BY visit_count DESC, v.department ASC"
    )


def _compile_required_departments(question: str, domains: Domains) -> Optional[str]:
    departments = _department_values(question, domains)
    if len(departments) < 2 or not re.search(r"都|全部|每一科|完整覆盖|均至少一次", question):
        return None
    if not re.search(r"不同科室|去重科室|科室总数|科室覆盖数|科室数量", question):
        return None
    values_sql = ", ".join(f"('{esc(value)}')" for value in departments)
    patient_predicates = _patient_predicates(question, domains)
    predicates = [
        *patient_predicates,
        "NOT EXISTS (SELECT 1 FROM required_department AS r WHERE NOT EXISTS "
        "(SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id "
        "AND v.department = r.department))",
    ]
    return (
        f"WITH required_department(department) AS (VALUES {values_sql}) "
        "SELECT p.patient_id, p.name, p.region, "
        "(SELECT COUNT(DISTINCT vx.department) FROM visits AS vx "
        "WHERE vx.patient_id = p.patient_id) AS department_count FROM patients AS p WHERE "
        + " AND ".join(predicates)
        + " ORDER BY p.patient_id ASC"
    )


def _relation_count_threshold(question: str, labels: Sequence[str]) -> Optional[tuple[str, str]]:
    return _threshold(question, labels)


def _compile_relation_edge_listing(question: str, domains: Domains) -> Optional[str]:
    """Compile one grounded relation's confidence/count edge listing."""

    relation = _single_relation_family_value(question, domains)
    if relation is None or not re.search(r"边|连接|三元组|关系", question):
        return None
    if re.search(r"高于同类型实体平均|不同疾病|疾病覆盖|尾实体提及", question):
        return None
    if _top_k(question) is not None or not _kg_only_clinical_scope_is_absent(question, domains):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"头实体|起点|头",
            r"尾实体|终点|尾",
            r"置信度",
            r"出现次数|观察次数|计数|次数",
        ),
        summary_patterns=(r"头尾实体", r"两项指标"),
    ):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("关系置信度", "confidence"),
    )
    count_bounds = _single_relation_metric_bounds(
        question,
        "出现次数",
        ("观察次数", "计数", "次数", "count"),
    )
    if not confidence_bounds:
        confidence = _threshold(question, ("置信度", "关系置信度", "confidence"))
        if confidence is not None:
            confidence_bounds = [confidence]
    if not count_bounds:
        count = _threshold(question, ("出现次数", "观察次数", "计数", "次数", "count"))
        if count is not None:
            count_bounds = [count]
        else:
            match = re.search(r"(\d+)\s*次(?:出现|计数|观察)", question)
            if match:
                count_bounds = [(">=", match.group(1))]
    if not confidence_bounds or not count_bounds:
        return None
    where = [f"relation_zh = '{esc(relation)}'"]
    where.extend(
        f"confidence {operator} {value}"
        for operator, value in confidence_bounds
    )
    where.extend(
        f"count {operator} {value}" for operator, value in count_bounds
    )
    return (
        "SELECT head, tail, confidence, count FROM kg_triples WHERE "
        + " AND ".join(where)
        + " ORDER BY count DESC, confidence DESC, head ASC, tail ASC"
    )


def _compile_relation_tail_entity_listing(
    question: str,
    domains: Domains,
) -> Optional[str]:
    """Compile a grounded edge joined to its tail entity metadata."""

    relation = _single_relation_family_value(question, domains)
    if relation is None or not re.search(
        r"尾实体提及|尾端实体.*提及|尾实体热度|尾端热度|尾实体信息",
        question,
    ):
        return None
    if _top_k(question) is not None or not _kg_only_clinical_scope_is_absent(question, domains):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"头实体|起点|头|两端名称",
            r"尾实体|尾端|终点|尾",
            r"尾实体类型|尾类型|尾端类型|类型",
            r"提及(?:次数|量|数)|热度",
            r"关系置信度|置信度",
        ),
    ):
        return None
    mention_bounds = _single_relation_metric_bounds(
        question,
        "尾实体提及次数",
        ("提及次数", "提及量", "提及数", "热度", "mention_count"),
    )
    if not mention_bounds:
        mention = _threshold(
            question,
            ("尾实体提及次数", "尾端实体提及次数", "提及次数", "提及量", "热度"),
        )
        if mention is not None:
            mention_bounds = [mention]
    if not mention_bounds:
        match = re.search(
            r"(?:尾端?|尾)实体\s*(?:至少|不少于|不低于|达到|≥|>=)?\s*"
            r"被提及\s*(\d+)\s*次",
            question,
        )
        if match:
            mention_bounds = [(">=", match.group(1))]
    if not mention_bounds:
        return None
    where = [f"kt.relation_zh = '{esc(relation)}'"]
    where.extend(
        f"ke.mention_count {operator} {value}"
        for operator, value in mention_bounds
    )
    return (
        "SELECT kt.head, kt.tail, ke.type AS tail_type, ke.mention_count, "
        "kt.confidence FROM kg_triples AS kt JOIN kg_entities AS ke "
        "ON ke.name = kt.tail AND ke.type = kt.tail_type WHERE "
        + " AND ".join(where)
        + " ORDER BY ke.mention_count DESC, kt.confidence DESC, "
        "kt.head ASC, kt.tail ASC"
    )


def _compile_relation_head_topk(question: str, domains: Domains) -> Optional[str]:
    """Compile per-head relation coverage Top-K with stable tie breaking."""

    relation = _single_relation_family_value(question, domains)
    top = _top_k(question)
    if relation is None or top is None or not re.search(
        r"尾实体种类|不同尾实体|不同尾端|覆盖面|连接不同尾端",
        question,
    ):
        return None
    if not _kg_only_clinical_scope_is_absent(question, domains):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"头实体|起点|头",
            r"不同尾实体数|尾实体数|终点数|尾端去重数",
            r"关系计数合计|计数总和|计数合计",
        ),
        summary_patterns=(r"稳定排序",),
    ):
        return None
    return (
        "SELECT head, COUNT(DISTINCT tail) AS distinct_tail_count, "
        "SUM(count) AS relation_count_sum FROM kg_triples "
        f"WHERE relation_zh = '{esc(relation)}' GROUP BY head "
        "ORDER BY distinct_tail_count DESC, relation_count_sum DESC, head ASC "
        f"LIMIT {top}"
    )


def _compile_relation_role_entity_listing(
    question: str,
    domains: Domains,
) -> Optional[str]:
    """Compile entity listing whose type is defined by a graph-edge role."""

    relations = _relations(question, domains)
    if len(relations) != 1 or not re.search(r"作为头实体来识别|按.*头实体", question):
        return None
    if not re.search(r"类型为疾病|疾病实体", question):
        return None
    mention_bounds = _single_relation_metric_bounds(
        question,
        "提及次数",
        ("提及量", "提及数", "热度", "mention_count"),
    )
    if not mention_bounds:
        return None
    where = [
        "EXISTS (SELECT 1 FROM kg_triples AS type_marker WHERE "
        "type_marker.head = ke.name AND "
        f"type_marker.relation_zh = '{esc(relations[0])}')"
    ]
    where.extend(
        f"ke.mention_count {operator} {value}"
        for operator, value in mention_bounds
    )
    return (
        "SELECT ke.name, ke.mention_count FROM kg_entities AS ke WHERE "
        + " AND ".join(where)
        + " ORDER BY ke.mention_count DESC, ke.name ASC"
    )


def _compile_kg_conditional_counts(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"症状|临床表现", question) or not re.search(r"治疗药物|药物", question) or not re.search(r"诊断检查|检查", question):
        return None
    if not re.search(r"条件计数|覆盖审计|各类目标数|三类数量|覆盖量|覆盖广", question):
        return None
    if re.search(r"或(?:者)?|\bOR\b", question, re.I):
        # This family renders conjunctive HAVING predicates only.
        return None
    fixed_modifiers = _fixed_kg_result_shape_is_unmodified(question, domains)
    exact_family_order = bool(
        re.search(
            r"按(?:数量|共享(?:症状)?数|共有数|总?覆盖量).*(?:降序|倒排|从高到低)",
            question,
        )
        and _top_k(question) is None
    )
    if (
        not (fixed_modifiers or exact_family_order)
        or not _kg_only_clinical_scope_is_absent(question, domains)
    ):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (r"疾病", r"症状(?:数|数量)", r"(?:治疗)?药物(?:数|数量)", r"(?:诊断)?检查(?:数|数量)"),
        summary_patterns=(
            r"三(?:类|项)(?:数量|去重计数)",
            r"各类目标数",
            r"覆盖量",
            r"覆盖广者在前",
        ),
    ):
        return None

    def count_bounds(
        subject: str,
        aliases: Sequence[str],
        reverse_labels: Sequence[str],
        reverse_pattern: str,
    ) -> Optional[list[tuple[str, str]]]:
        bounds = _visit_metric_bounds(question, subject, aliases)
        if bounds:
            return bounds
        threshold = _relation_count_threshold(question, reverse_labels)
        if threshold is not None:
            return [threshold]
        match = re.search(
            rf"(?:至少|不少于|不低于|≥|>=)?\s*(\d+)\s*(?:个|种|项)?\s*{reverse_pattern}",
            question,
        )
        return [(">=", match.group(1))] if match else []

    symptom_bounds = count_bounds(
        "症状数",
        ("临床表现", "症状"),
        ("症状数", "临床表现", "症状"),
        r"(?:症状|临床表现)",
    )
    drug_bounds = count_bounds(
        "治疗药物数",
        ("药物数", "治疗药物", "药物"),
        ("治疗药物数", "药物数", "治疗药物", "药物"),
        r"(?:治疗药物|药物)",
    )
    exam_bounds = count_bounds(
        "诊断检查数",
        ("检查数", "诊断检查", "检查"),
        ("诊断检查数", "检查数", "诊断检查", "检查"),
        r"(?:诊断检查|检查)",
    )
    if not symptom_bounds or not drug_bounds or not exam_bounds:
        return None
    having: list[str] = []
    for relation, bounds in (
        ("临床表现", symptom_bounds),
        ("治疗药物", drug_bounds),
        ("诊断检查", exam_bounds),
    ):
        having.extend(
            "COUNT(DISTINCT CASE WHEN relation_zh = "
            f"'{relation}' THEN tail END) {operator} {value}"
            for operator, value in bounds
        )
    return (
        "SELECT head AS disease, "
        "COUNT(DISTINCT CASE WHEN relation_zh = '临床表现' THEN tail END) AS symptom_count, "
        "COUNT(DISTINCT CASE WHEN relation_zh = '治疗药物' THEN tail END) AS drug_count, "
        "COUNT(DISTINCT CASE WHEN relation_zh = '诊断检查' THEN tail END) AS exam_count "
        "FROM kg_triples GROUP BY head HAVING "
        + " AND ".join(having)
        + " ORDER BY (symptom_count + drug_count + exam_count) DESC, disease ASC"
    )


def _compile_disease_pairs(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"疾病对|疾病两两|两种疾病|不同疾病对", question):
        return None
    if not re.search(r"共同症状|共享症状|症状重叠", question):
        return None
    # The fixed three-column family cannot preserve caller-controlled output
    # or result modifiers, or re-scope graph diseases to a clinical cohort.
    fixed_modifiers = _fixed_kg_result_shape_is_unmodified(question, domains)
    exact_family_order = bool(
        re.search(
            r"按(?:数量|共享(?:症状)?数|共有数).*(?:降序|倒排|从高到低)",
            question,
        )
        and _top_k(question) is None
    )
    if (
        not (fixed_modifiers or exact_family_order)
        or not _kg_only_clinical_scope_is_absent(question, domains)
    ):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (r"疾病一|第一种疾病|疾病甲", r"疾病二|第二种疾病|疾病乙", r"共享|共同|共有|重叠"),
        summary_patterns=(r"疾病对", r"两种疾病"),
    ):
        return None
    shared_bounds = _visit_metric_bounds(
        question,
        "共同症状",
        ("共享症状", "重叠数", "共有数量", "共享数"),
    )
    if shared_bounds is None:
        return None
    if not shared_bounds:
        shared = _threshold(question, ("共同症状", "共享症状", "重叠数", "共有数量", "共享数"))
        if shared is not None:
            shared_bounds = [shared]
        else:
            match = re.search(
                r"(?:共享|共同|拥有).{0,6}(?:至少|不少于|达到|≥|>=)?\s*(\d+)\s*个?(?:共同|共享)?症状|"
                r"(\d+)\s*个以上共同症状",
                question,
            )
            if match:
                shared_bounds = [(">=", next(group for group in match.groups() if group))]
    edge_domains = _edge_domains(domains, ("临床表现",))
    confidences = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="置信度",
        aliases=("边置信度", "关系置信度", "confidence"),
        relations=("临床表现", "临床表现"),
        domains=edge_domains,
    )
    if confidences is not None and not any(confidences):
        # Preserve the older explicit shared form that omits the word
        # ``置信度`` but says both edges use the same numeric requirement.
        match = re.search(
            r"(?:每条|双方|两侧).*?边(?:达到|至少|不低于|≥|>=)\s*(\d+(?:\.\d+)?)",
            question,
        )
        if match:
            confidence = (">=", _number(match.group(1)))
            confidences = [[confidence], [confidence]]
    if confidences is None or not all(confidences):
        match = re.search(
            r"相关边置信度\s*(?:至少|不低于|不少于|达到|≥|>=)\s*"
            r"(\d+(?:\.\d+)?)",
            question,
        )
        if match:
            confidence = (">=", _number(match.group(1)))
            confidences = [[confidence], [confidence]]
    if confidences is None or not all(confidences) or not shared_bounds:
        return None
    counts = _optional_edge_metric_bounds(
        question,
        subject="观察次数",
        aliases=("出现次数", "关系次数", "计数", "count"),
        relations=("临床表现", "临床表现"),
        domains=edge_domains,
    )
    if counts is None:
        return None
    confidence_predicates = [
        f"{alias}.confidence {operator} {value}"
        for alias, bounds in zip(("a", "b"), confidences)
        for operator, value in bounds
    ]
    count_predicates = [
        f"{alias}.count {operator} {value}"
        for alias, bounds in zip(("a", "b"), counts)
        for operator, value in bounds
    ]
    return (
        "SELECT a.head AS disease_one, b.head AS disease_two, "
        "COUNT(DISTINCT a.tail) AS shared_symptom_count FROM kg_triples AS a "
        "JOIN kg_triples AS b ON b.tail = a.tail AND b.relation_zh = '临床表现' "
        "AND a.head < b.head WHERE a.relation_zh = '临床表现' AND "
        + " AND ".join(confidence_predicates + count_predicates)
        + " GROUP BY a.head, b.head HAVING "
        + " AND ".join(
            f"COUNT(DISTINCT a.tail) {operator} {value}"
            for operator, value in shared_bounds
        )
        + " ORDER BY shared_symptom_count DESC, disease_one ASC, disease_two ASC"
    )


def _compile_two_hop_complication_drug(question: str, domains: Domains) -> Optional[str]:
    if not ("并发症" in question and "治疗药物" in question):
        return None
    if not re.search(r"两跳|两段式|路径|知识链|→", question):
        return None
    if re.search(r"禁忌|零次|实际处方", question):
        return None
    # The general graph compiler owns output projections for chains. This
    # fixed-shape family must not discard a requested role, metric, result
    # modifier, or clinical cohort scope.
    if (
        not _fixed_kg_result_shape_is_unmodified(question, domains)
        or not _kg_only_clinical_scope_is_absent(question, domains)
    ):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (r"(?:源|原)(?:疾病|病)", r"并发症", r"(?:治疗)?药物", r"并发症.*置信度", r"药物.*置信度|(?:两|各)(?:条)?边置信度"),
        summary_patterns=(r"完整路径|完整链|三类实体|两项置信度",),
    ):
        return None
    relations = ("并发症", "治疗药物")
    edge_domains = _edge_domains(domains, relations)
    confidences = _per_hop_thresholds(
        question,
        ("边置信度", "关系置信度", "置信度"),
        relations=relations,
        domains=edge_domains,
    )
    counts = _per_hop_thresholds(
        question,
        ("观察次数", "出现次数", "关系次数", "计数", "次数"),
        relations=relations,
        domains=edge_domains,
    )
    shared_scope = re.search(
        r"两个环节|两条边|两边|每条关系|相关边|均|都需|两项", question
    )
    if confidences is None and shared_scope:
        shared_confidence = _threshold(
            question,
            ("置信度", "边置信度", "关系置信度"),
        )
        if shared_confidence is not None:
            confidences = [[shared_confidence], [shared_confidence]]
    if counts is None and shared_scope:
        shared_count = _threshold(
            question,
            ("计数", "观察次数", "出现次数", "关系次数"),
        )
        if shared_count is not None:
            counts = [[shared_count], [shared_count]]
        else:
            match = re.search(
                r"(?:至少|不少于|不低于|达到|≥|>=)?\s*(\d+)\s*次(?:计数|出现|观察)?",
                question,
            )
            if match:
                count = (">=", match.group(1))
                counts = [[count], [count]]
    if confidences is None or counts is None:
        return None
    confidence_predicates = [
        f"{alias}.confidence {operator} {value}"
        for alias, bounds in zip(("c", "d"), confidences)
        for operator, value in bounds
    ]
    count_predicates = [
        f"{alias}.count {operator} {value}"
        for alias, bounds in zip(("c", "d"), counts)
        for operator, value in bounds
    ]
    return (
        "SELECT c.head AS source_disease, c.tail AS complication, "
        "d.tail AS treatment_drug, c.confidence AS complication_confidence, "
        "d.confidence AS drug_confidence FROM kg_triples AS c "
        "JOIN kg_triples AS d ON d.head = c.tail AND d.relation_zh = '治疗药物' "
        "WHERE c.relation_zh = '并发症' AND "
        + " AND ".join(confidence_predicates + count_predicates)
        + " ORDER BY c.head ASC, c.tail ASC, d.tail ASC"
    )


def _compile_monthly_yoy(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"按月|月度|逐月|各月", question):
        return None
    if not re.search(r"同比|前推12个月|向前第12个月|12个月滞后|滞后值|同期", question):
        return None
    date_window = _date_range(question)
    if date_window is None or not re.search(r"总费用|总花费|月总额", question):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"月份|年月",
            r"本月|当前.*总(?:费用|额)|月总额",
            r"同期|对比月|上一年|12个月滞后值",
            r"差额|变化额|同比|本月减滞后值",
        ),
        summary_patterns=(r"三项金额", r"同比序列",),
    ):
        return None
    start_date, end_date = date_window
    patient_predicates = _patient_predicates(question, domains)
    visit_prefix = "v." if patient_predicates else ""
    from_sql = (
        "patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        if patient_predicates
        else "visits"
    )
    sequence_lag = re.search(
        r"前推12个月|向前第12个月|12个月滞后|滞后值|月份顺序.*12",
        question,
    )
    if sequence_lag:
        predicates = [
            f"{visit_prefix}visit_date BETWEEN '{start_date}' AND '{end_date}'",
            *patient_predicates,
        ]
        return (
            f"WITH monthly AS (SELECT substr({visit_prefix}visit_date, 1, 7) AS visit_month, "
            f"SUM({visit_prefix}cost) AS total_cost FROM {from_sql} WHERE {' AND '.join(predicates)} "
            f"GROUP BY substr({visit_prefix}visit_date, 1, 7)), compared AS (SELECT visit_month, "
            "total_cost, LAG(total_cost, 12) OVER (ORDER BY visit_month) AS prior_year_cost "
            "FROM monthly) SELECT visit_month, ROUND(total_cost, 2) AS total_cost, "
            "ROUND(prior_year_cost, 2) AS prior_year_cost, "
            "ROUND(total_cost - prior_year_cost, 2) AS year_over_year_change FROM compared "
            "WHERE prior_year_cost IS NOT NULL ORDER BY visit_month ASC"
        )
    predicates = [
        f"{visit_prefix}visit_date >= date('{start_date}', 'start of month', '-1 year')",
        f"{visit_prefix}visit_date <= '{end_date}'",
        *patient_predicates,
    ]
    return (
        f"WITH monthly AS (SELECT substr({visit_prefix}visit_date, 1, 7) AS visit_month, "
        f"SUM({visit_prefix}cost) AS total_cost FROM {from_sql} WHERE {' AND '.join(predicates)} "
        f"GROUP BY substr({visit_prefix}visit_date, 1, 7)) "
        "SELECT current_month.visit_month, ROUND(current_month.total_cost, 2) AS total_cost, "
        "ROUND(prior_month.total_cost, 2) AS prior_year_cost, "
        "ROUND(current_month.total_cost - prior_month.total_cost, 2) AS year_over_year_change "
        "FROM monthly AS current_month JOIN monthly AS prior_month ON "
        "prior_month.visit_month = strftime('%Y-%m', "
        "date(current_month.visit_month || '-01', '-1 year')) "
        f"WHERE current_month.visit_month BETWEEN substr('{start_date}', 1, 7) "
        f"AND substr('{end_date}', 1, 7) ORDER BY current_month.visit_month ASC"
    )


def _gap_threshold(question: str) -> Optional[str]:
    for pattern in (
        r"(?:间隔|相隔|距上一次记录|距离同一患者上次就诊).*?"
        r"(?:至少|达到|不少于|不低于|已有)\s*(\d+)\s*天",
        r"(?:间隔|相隔)\s*(?:达到|不少于|不低于|≥|>=)\s*(\d+)\s*天",
        r"(\d+)\s*天(?:或更久|以上)",
    ):
        match = re.search(pattern, question)
        if match:
            return match.group(1)
    return None


def _compile_lag_gap(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"相邻就诊|连续两次就诊|前次相隔|上一次记录|上次就诊", question):
        return None
    if re.search(r"新一段|事件簇|阶段|断开", question):
        return None
    gap = _gap_threshold(question)
    if gap is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    visit_prefix = "v." if patient_predicates else ""
    from_sql = (
        "patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        if patient_predicates
        else "visits"
    )
    source_where = f" WHERE {' AND '.join(patient_predicates)}" if patient_predicates else ""
    return (
        f"WITH sequenced AS (SELECT {visit_prefix}patient_id, {visit_prefix}visit_id, "
        f"{visit_prefix}visit_date, {visit_prefix}disease, LAG({visit_prefix}visit_date) OVER "
        f"(PARTITION BY {visit_prefix}patient_id ORDER BY {visit_prefix}visit_date, {visit_prefix}visit_id) "
        f"AS previous_visit_date FROM {from_sql}{source_where}) "
        "SELECT patient_id, previous_visit_date, visit_date, "
        "CAST(julianday(visit_date) - julianday(previous_visit_date) AS INTEGER) AS gap_days, disease "
        "FROM sequenced WHERE previous_visit_date IS NOT NULL AND "
        f"julianday(visit_date) - julianday(previous_visit_date) >= {gap} "
        "ORDER BY gap_days DESC, patient_id ASC, visit_date ASC"
    )


def _compile_disease_having_topk(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"疾病", question) or not re.search(r"平均费用|费用均值|平均花费", question):
        return None
    if not re.search(r"前\s*\d+|进入前\s*\d+|最高的?\s*\d+", question):
        return None
    top = _top_k(question)
    count = _visit_count_threshold(question)
    since = _since_date(question)
    if top is None or count is None or since is None:
        return None
    patient_filters = _patient_predicates(question, domains, alias="p")
    visit_prefix = "v." if patient_filters else ""
    from_sql = (
        "patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        if patient_filters
        else "visits"
    )
    predicates = [f"{visit_prefix}visit_date >= '{since}'", *patient_filters]
    return (
        f"SELECT {visit_prefix}disease, COUNT(*) AS visit_count, "
        f"ROUND(AVG({visit_prefix}cost), 2) AS avg_cost, "
        f"ROUND(SUM({visit_prefix}cost), 2) AS total_cost FROM {from_sql} "
        f"WHERE {' AND '.join(predicates)} GROUP BY {visit_prefix}disease "
        f"HAVING COUNT(*) {count[0]} {count[1]} "
        f"ORDER BY avg_cost DESC, total_cost DESC, {visit_prefix}disease ASC "
        f"LIMIT {top}"
    )


def _compile_first_last(question: str, domains: Domains) -> Optional[str]:
    first = bool(re.search(r"首次|最初一次|最早", question))
    last = bool(re.search(r"末次|最近一次|最后一行|最近", question))
    if not (first and last and re.search(r"疾病|诊断", question)):
        return None
    minimum = _visit_count_threshold(question)
    if minimum is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    source_where = f" WHERE {' AND '.join(patient_predicates)}" if patient_predicates else ""
    return (
        "WITH ordered AS (SELECT p.patient_id, p.name, v.visit_id, v.visit_date, v.disease, "
        "COUNT(*) OVER (PARTITION BY p.patient_id) AS visit_count, "
        "FIRST_VALUE(v.visit_date) OVER (PARTITION BY p.patient_id ORDER BY v.visit_date, v.visit_id) "
        "AS first_visit_date, FIRST_VALUE(v.disease) OVER "
        "(PARTITION BY p.patient_id ORDER BY v.visit_date, v.visit_id) AS first_disease, "
        "ROW_NUMBER() OVER (PARTITION BY p.patient_id ORDER BY v.visit_date DESC, v.visit_id DESC) "
        "AS reverse_order FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        + source_where
        + ") SELECT patient_id, name, visit_count, first_visit_date, first_disease, "
        "visit_date AS last_visit_date, disease AS last_disease FROM ordered "
        "WHERE reverse_order = (SELECT MIN(o2.reverse_order) FROM ordered AS o2 "
        "WHERE o2.patient_id = ordered.patient_id) "
        f"AND visit_count {minimum[0]} {minimum[1]} ORDER BY patient_id ASC"
    )


def _compile_running_cost(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"累计费用|累积花费|费用总和|运行总额", question):
        return None
    if not re.search(r"每次就诊|逐笔|每条|截至当次|从首条到当前", question):
        return None
    regions = _region_values(question, domains)
    since = _since_date(question)
    if len(regions) != 1 or since is None:
        return None
    patient_predicates = _patient_predicates(question, domains, include_region=False)
    predicates = [
        f"p.region = '{esc(regions[0])}'",
        f"v.visit_date >= '{since}'",
        *patient_predicates,
    ]
    return (
        "SELECT p.patient_id, p.name, v.visit_date, ROUND(v.cost, 2) AS visit_cost, "
        "ROUND(SUM(v.cost) OVER (PARTITION BY p.patient_id ORDER BY v.visit_date, v.visit_id "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS running_cost "
        "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id WHERE "
        + " AND ".join(predicates)
        + " ORDER BY p.patient_id ASC, v.visit_date ASC, v.visit_id ASC"
    )


def _compile_department_topk(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"每个科室|各科室|按科室独立|按科室分区|科室内", question):
        return None
    if not re.search(r"费用最高|最昂贵|费用降序", question):
        return None
    top = _top_k(question)
    date_window = _date_range(question)
    if top is None or date_window is None:
        return None
    if not _fixed_projection_is_compatible(
        question,
        (r"科室", r"名次|排名", r"(?:就诊|记录)?编号", r"日期", r"疾病", r"费用|花费"),
        summary_patterns=(r"就诊明细|记录详情|就诊详情|名次及.*明细|排名与.*详情",),
    ):
        return None
    patient_predicates = _patient_predicates(question, domains)
    visit_prefix = "v." if patient_predicates else ""
    from_sql = (
        "patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        if patient_predicates
        else "visits"
    )
    predicates = [
        f"{visit_prefix}visit_date BETWEEN '{date_window[0]}' AND '{date_window[1]}'",
        *patient_predicates,
    ]
    return (
        f"WITH ranked AS (SELECT {visit_prefix}department, {visit_prefix}visit_id, "
        f"{visit_prefix}visit_date, {visit_prefix}disease, {visit_prefix}cost, "
        f"ROW_NUMBER() OVER (PARTITION BY {visit_prefix}department ORDER BY {visit_prefix}cost DESC, "
        f"{visit_prefix}visit_date ASC, {visit_prefix}visit_id ASC) AS department_rank FROM {from_sql} "
        f"WHERE {' AND '.join(predicates)}) "
        "SELECT department, department_rank, visit_id, visit_date, disease, cost FROM ranked "
        f"WHERE department_rank <= {top} ORDER BY department ASC, department_rank ASC"
    )


def _entity_type_predicate(entity_label: str) -> Optional[str]:
    """Map an explicit union entity label to the schema type column."""

    entity_types = {
        "药物": "Drug",
        "症状": "Symptom",
        "检查": "Exam",
    }
    entity_type = entity_types.get(entity_label)
    return f"type = '{entity_type}'" if entity_type is not None else None


def _compile_patient_kg_union(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"统一名称列表|同一结果|合并|集合去重", question):
        return None
    if not ("患者" in question and "实体" in question):
        return None
    # The KG union compiler maps explicit source/name projections. Defer
    # instead of forcing this legacy two-column result or its default ordering
    # onto a caller. This union has no visit source, so it cannot apply a
    # temporal patient cohort either.
    exact_family_order = bool(
        re.search(r"按名称(?:和|与|、)?.*来源.*(?:升序|排列|排序)", question)
    )
    if not (
        _fixed_kg_result_shape_is_unmodified(question, domains)
        or exact_family_order
    ) or dates(question):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (r"名称|姓名|实体名", r"来源|来源标签"),
        summary_patterns=(r"统一名称列表|同一结果",),
    ):
        return None
    regions = _region_values(question, domains)
    age_filters = _age_filters(question)
    gender = _gender_value(question)
    mention_bounds = _single_relation_metric_bounds(
        question,
        "提及次数",
        ("提及量", "提及", "热度", "mention_count"),
    )
    has_mention_disjunction = bool(re.search(r"或(?:者)?|\bOR\b", question, re.I))
    if not mention_bounds and not has_mention_disjunction:
        mention = _threshold(
            question,
            ("提及次数", "提及量", "提及数", "提及", "热度", "mention_count"),
        )
        if mention is not None:
            mention_bounds = [mention]
    if not mention_bounds and not has_mention_disjunction:
        match = re.search(
            r"(?:至少|不低于|不少于|达到|≥|>=)\s*提及\s*(\d+)\s*次",
            question,
        )
        if match:
            mention_bounds = [(">=", match.group(1))]
    entity_label = next((value for value in ("药物", "症状", "检查") if value in question), None)
    entity_type_predicate = _entity_type_predicate(entity_label or "")
    if re.search(r"按在.*关系中作为|关系中作为", question):
        if entity_label == "药物":
            entity_type_predicate = (
                "(EXISTS (SELECT 1 FROM kg_triples AS type_marker WHERE "
                "type_marker.tail = kg_entities.name AND "
                "type_marker.relation_zh = '治疗药物') OR EXISTS "
                "(SELECT 1 FROM kg_triples AS contraindication_marker WHERE "
                "contraindication_marker.head = kg_entities.name AND "
                "contraindication_marker.relation_zh = '用药禁忌'))"
            )
        elif entity_label == "症状":
            entity_type_predicate = (
                "EXISTS (SELECT 1 FROM kg_triples AS type_marker WHERE "
                "type_marker.tail = kg_entities.name AND "
                "type_marker.relation_zh = '临床表现')"
            )
        elif entity_label == "检查":
            entity_type_predicate = (
                "EXISTS (SELECT 1 FROM kg_triples AS type_marker WHERE "
                "type_marker.tail = kg_entities.name AND "
                "type_marker.relation_zh = '诊断检查')"
            )
    if (
        len(regions) != 1
        or not age_filters
        or not mention_bounds
        or entity_type_predicate is None
    ):
        return None
    patient_predicates = [f"region = '{esc(regions[0])}'", *age_filters]
    if gender is not None:
        patient_predicates.append(f"gender = '{esc(gender)}'")
    return (
        "SELECT p.name, '患者' AS source FROM patients AS p WHERE "
        + " AND ".join(patient_predicates)
        + " UNION SELECT name, '知识实体' AS source FROM kg_entities WHERE "
        + entity_type_predicate
        + " AND "
        + " AND ".join(
            f"mention_count {operator} {value}"
            for operator, value in mention_bounds
        )
        + " ORDER BY name ASC, source ASC"
    )


def _compile_patient_rollup(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"按患者汇总|患者级汇总|聚合到患者|患者.*汇总|汇总后", question):
        return None
    if not re.search(r"总费用|累计花费|花费合计|累计费用|费用总额", question):
        return None
    age_filters = _age_filters(question, alias="p")
    visits = _visit_count_threshold(question)
    total = _total_cost_threshold(question)
    if not age_filters or visits is None or total is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    predicates = [
        *patient_predicates,
        f"r.visit_count {visits[0]} {visits[1]}",
        f"r.total_cost {total[0]} {total[1]}",
    ]
    return (
        "WITH patient_rollup AS (SELECT patient_id, COUNT(*) AS visit_count, "
        "SUM(cost) AS total_cost, COUNT(DISTINCT department) AS department_count "
        "FROM visits GROUP BY patient_id) SELECT p.patient_id, p.name, p.age, "
        "r.visit_count, ROUND(r.total_cost, 2) AS total_cost, r.department_count "
        "FROM patient_rollup AS r JOIN patients AS p ON p.patient_id = r.patient_id WHERE "
        + " AND ".join(predicates)
        + " ORDER BY r.total_cost DESC, p.patient_id ASC"
    )


def _compile_kg_tail_coverage(question: str, domains: Domains) -> Optional[str]:
    relation = _single_relation_family_value(question, domains)
    if relation is None:
        return None
    if not re.search(r"尾实体|尾端|目标", question):
        return None
    # The KG tail-coverage compiler preserves an explicit relation/metric
    # projection. This fixed per-tail template must not discard caller-owned
    # output, result modifiers, or a clinical cohort scope.
    exact_family_order = bool(
        re.search(r"按(?:疾病覆盖数|疾病数|覆盖数|提及量).*(?:降序|倒排|优先)", question)
        and _top_k(question) is None
    )
    if (
        not (_fixed_kg_result_shape_is_unmodified(question, domains) or exact_family_order)
        or not _kg_only_clinical_scope_is_absent(question, domains)
    ):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"尾实体|尾端|目标|名称",
            r"类型",
            r"疾病(?:数|来源|覆盖数)|来源疾病数",
            r"提及(?:次数|量|数)",
        ),
        summary_patterns=(r"高覆盖者优先",),
    ):
        return None
    disease_count_bounds = _visit_metric_bounds(
        question,
        "不同疾病",
        ("疾病数", "疾病来源", "覆盖"),
    )
    if disease_count_bounds is None:
        disease_count_bounds = []
    if not disease_count_bounds:
        disease_count = _threshold(question, ("不同疾病", "疾病数", "疾病来源", "覆盖"))
        if disease_count is not None:
            disease_count_bounds = [disease_count]
        else:
            match = re.search(
                r"(?:至少|不少于|不低于|达到|≥|>=)\s*(?:连接|关联|覆盖)?\s*(\d+)\s*种?(?:不同)?疾病|"
                r"(?:至少|不少于|不低于|达到|≥|>=)\s*(\d+)\s*(?:个|种)?疾病来源|"
                r"覆盖\s*(?:≥|>=|至少)?\s*(\d+)",
                question,
            )
            if match:
                disease_count_bounds = [(">=", next(group for group in match.groups() if group))]
    if not disease_count_bounds:
        return None
    mention_bounds = _tail_mention_average_bounds(question)
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("关系置信度", "confidence"),
    )
    count_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("出现次数", "关系次数", "计数", "count"),
    )
    if (
        mention_bounds is None
        or confidence_bounds is None
        or count_bounds is None
    ):
        return None
    where = [
        f"kt.relation_zh = '{esc(relation)}'",
        "ke.mention_count > (SELECT AVG(ke2.mention_count) FROM kg_entities AS ke2 "
        "WHERE ke2.type = kt.tail_type)",
    ]
    where.extend(
        f"ke.mention_count {operator} {value}"
        for operator, value in mention_bounds
    )
    where.extend(
        f"kt.confidence {operator} {value}"
        for operator, value in confidence_bounds
    )
    where.extend(
        f"kt.count {operator} {value}"
        for operator, value in count_bounds
    )
    return (
        "SELECT kt.tail, kt.tail_type, COUNT(DISTINCT kt.head) AS disease_count, "
        "ke.mention_count FROM kg_triples AS kt JOIN kg_entities AS ke "
        "ON ke.name = kt.tail AND ke.type = kt.tail_type WHERE "
        + " AND ".join(where)
        + " GROUP BY kt.tail, kt.tail_type, ke.mention_count HAVING "
        + " AND ".join(
            f"COUNT(DISTINCT kt.head) {operator} {value}"
            for operator, value in disease_count_bounds
        )
        + " ORDER BY disease_count DESC, ke.mention_count DESC, kt.tail ASC"
    )


def _compile_kg_required_entities(question: str, domains: Domains) -> Optional[str]:
    relations = _relations(question, domains)
    if len(relations) != 2 or not re.search(r"疾病实体|哪些疾病|连接的疾病|关系.*疾病|疾病.*兼具|疾病.*同时", question):
        return None
    if re.search(r"实际就诊|实际记录|平均费用", question):
        return None
    # The KG compiler maps entity projections (including ``type``) in request
    # order. This older fixed template must defer rather than omit caller-owned
    # output, result modifiers, or a clinical cohort scope.
    exact_family_order = bool(
        re.search(r"按(?:提及(?:次数|量)|热度).*(?:降序|倒排|从高到低)", question)
        and _top_k(question) is None
    )
    if (
        not (_fixed_kg_result_shape_is_unmodified(question, domains) or exact_family_order)
        or not _kg_only_clinical_scope_is_absent(question, domains)
    ):
        return None
    if not _fixed_projection_is_compatible(
        question,
        (r"疾病|名称", r"提及(?:次数|量|数)"),
        summary_patterns=(r"疾病实体",),
    ):
        return None
    edge_domains = _edge_domains(domains, relations)
    confidences = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="置信度",
        aliases=("边置信度", "关系置信度", "confidence"),
        relations=relations,
        domains=edge_domains,
    )
    if confidences is None or not all(confidences):
        shared_confidence = _threshold(
            question,
            ("置信度", "边置信度", "关系置信度"),
        )
        if shared_confidence is not None and re.search(
            r"两类|两种|各自|均|都|同下限|两种证据", question
        ):
            confidences = [[shared_confidence], [shared_confidence]]
    counts = _optional_edge_metric_bounds(
        question,
        subject="观察次数",
        aliases=("出现次数", "关系次数", "计数", "count"),
        relations=relations,
        domains=edge_domains,
    )
    mention_filter_question = re.sub(
        r"按(?:提及(?:次数|量|数)|热度).*?(?:降序|倒排|从高到低)",
        "",
        question,
    )
    mention_bounds = _single_relation_metric_bounds(
        mention_filter_question,
        "提及次数",
        ("提及数", "提及量", "mention_count", "热度"),
    )
    if mention_bounds is None and not _metric_threshold_intent(
        mention_filter_question,
        "提及次数",
        ("提及数", "提及量", "mention_count", "热度"),
    ):
        # An explicit projection/order on mention_count is not a filter.
        mention_bounds = []
    if (
        confidences is None
        or not all(confidences)
        or counts is None
        or mention_bounds is None
        or not re.search(r"提及", question)
    ):
        return None
    entity_type = _entity_type(question)
    predicates = [
        f"ke.mention_count {operator} {value}"
        for operator, value in mention_bounds
    ]
    if entity_type is not None:
        predicates.append(f"ke.type = '{esc(entity_type)}'")
    for index, (relation, confidence_bounds, count_bounds) in enumerate(
        zip(relations, confidences, counts),
        start=1,
    ):
        edge_predicates = [
            f"t{index}.confidence {operator} {value}"
            for operator, value in confidence_bounds
        ]
        edge_predicates.extend(
            f"t{index}.count {operator} {value}"
            for operator, value in count_bounds
        )
        predicates.append(
            f"EXISTS (SELECT 1 FROM kg_triples AS t{index} WHERE t{index}.head = ke.name "
            f"AND t{index}.relation_zh = '{esc(relation)}' AND "
            + " AND ".join(edge_predicates)
            + ")"
        )
    return (
        "SELECT ke.name, ke.mention_count FROM kg_entities AS ke WHERE "
        + " AND ".join(predicates)
        + " ORDER BY ke.mention_count DESC, ke.name ASC"
    )


def _compile_patient_total_peer_average(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"个人(?:总费用|累计费用)|每位.*累计花费|费用合计", question):
        return None
    if not re.search(r"所在地区|同地区|本地区", question) or not re.search(r"平均|人均", question):
        return None
    visits = _visit_count_threshold(question)
    if visits is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    predicates = [
        *patient_predicates,
        f"s.visit_count {visits[0]} {visits[1]}",
        "s.total_cost > (SELECT AVG(region_total) FROM (SELECT v2.patient_id, "
        "SUM(v2.cost) AS region_total FROM visits AS v2 JOIN patients AS p2 "
        "ON p2.patient_id = v2.patient_id WHERE p2.region = p.region GROUP BY v2.patient_id))",
    ]
    return (
        "SELECT p.patient_id, p.name, p.region, s.visit_count, "
        "ROUND(s.total_cost, 2) AS total_cost FROM patients AS p JOIN "
        "(SELECT patient_id, COUNT(*) AS visit_count, SUM(cost) AS total_cost "
        "FROM visits GROUP BY patient_id) AS s ON s.patient_id = p.patient_id WHERE "
        + " AND ".join(predicates)
        + " ORDER BY p.region ASC, total_cost DESC, p.patient_id ASC"
    )


def _compile_patient_exists(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"不要求发生在同一|可来自不同|不限定同次|另有或同有|历史记录同时", question):
        return None
    diseases = domain_values(question, domains.get("disease", ()))
    drugs = domain_values(question, domains.get("drug", ()))
    if len(diseases) != 1 or len(drugs) != 1:
        return None
    patient_predicates = _patient_predicates(question, domains)
    predicates = [
        *patient_predicates,
        "EXISTS (SELECT 1 FROM visits AS vd WHERE vd.patient_id = p.patient_id "
        f"AND vd.disease = '{esc(diseases[0])}')",
        "EXISTS (SELECT 1 FROM visits AS vm WHERE vm.patient_id = p.patient_id "
        f"AND vm.drug_prescribed = '{esc(drugs[0])}')",
    ]
    return (
        "SELECT p.patient_id, p.name, p.region, p.age FROM patients AS p WHERE "
        + " AND ".join(predicates)
        + " ORDER BY p.patient_id ASC"
    )


def _compile_age_peer_average(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"平均年龄|年龄均值|平均年龄比较|年龄超过.*均值|年龄高于.*平均", question):
        return None
    if not re.search(r"本地区|所在地区|当地", question) or not re.search(r"同性别|同一性别|男性患者均值|女性患者.*平均", question):
        return None
    gender = _gender_value(question)
    regions = _region_values(question, domains)
    if gender is None or len(regions) != 1 or not re.search(r"排除|不要|除外|非|除.{1,12}外", question):
        return None
    candidate_predicates = _patient_predicates(question, domains, include_region=False)
    predicates = [
        *candidate_predicates,
        f"p.region <> '{esc(regions[0])}'",
        "p.age > (SELECT AVG(p2.age) FROM patients AS p2 WHERE p2.region = p.region "
        "AND p2.gender = p.gender)",
    ]
    return (
        "SELECT p.patient_id, p.name, p.region, p.age FROM patients AS p WHERE "
        + " AND ".join(predicates)
        + " ORDER BY p.region ASC, p.age DESC, p.patient_id ASC"
    )


def _compile_grouped_having(question: str, domains: Domains) -> Optional[str]:
    if not re.search(r"科室", question) or not re.search(r"平均费用|费用均值|平均花费", question):
        return None
    if not re.search(r"只保留|找出|筛选|哪些科室", question):
        return None
    visit_count = _visit_count_threshold(question)
    average = _threshold(question, ("平均费用", "费用均值", "平均花费", "单次平均费用"))
    date_window = _date_range(question)
    if visit_count is None or average is None or date_window is None:
        return None
    if not _fixed_projection_is_compatible(
        question,
        (
            r"科室",
            r"(?:就诊|记录)?次数|就诊量",
            r"平均(?:费用|花费)|^平均$|^均值$",
            r"总(?:费用|花费)|累计(?:费用|花费)|总额",
        ),
        summary_patterns=(r"次数.*平均.*总", r"金额两位"),
    ):
        return None
    patient_predicates = age_predicates(question, alias="p")
    gender = gender_predicate(question, alias="p")
    if gender is not None:
        patient_predicates.append(gender)
    patient_predicates.extend(region_predicates(question, domains, alias="p"))
    from_sql = "visits AS v"
    predicates = [f"v.visit_date BETWEEN '{date_window[0]}' AND '{date_window[1]}'"]
    if patient_predicates:
        from_sql += " JOIN patients AS p ON p.patient_id = v.patient_id"
        predicates.extend(patient_predicates)
    return (
        "SELECT v.department, COUNT(*) AS visit_count, ROUND(AVG(v.cost), 2) AS avg_cost, "
        f"ROUND(SUM(v.cost), 2) AS total_cost FROM {from_sql} "
        f"WHERE {' AND '.join(predicates)} GROUP BY v.department "
        f"HAVING COUNT(*) {visit_count[0]} {visit_count[1]} "
        f"AND AVG(v.cost) {average[0]} {average[1]} "
        "ORDER BY avg_cost DESC, v.department ASC"
    )


def compile_temporal_cohort_family_sql(
    question: str,
    domains: Domains,
    *,
    intent: Optional[TemporalCohortIntent] = None,
    obligations: Optional[SemanticObligations] = None,
) -> Optional[str]:
    """Compile only the temporal family designated by structured ownership.

    This deliberately is not a general family fallback.  If the selected
    lowerer cannot consume the complete request, callers must abstain instead
    of trying another semantic family that might discard cohort qualifiers.
    """

    if not isinstance(question, str) or not question.strip() or not isinstance(domains, Mapping):
        return None
    if (
        classify_region_filter_intent(question, domains).disposition
        is RegionFilterDisposition.TERMINAL_ABSTAIN
    ):
        return None
    parsed = intent or classify_temporal_cohort_intent(question, domains)
    obligations = (
        obligations
        if obligations is not None
        else extract_semantic_obligations(
            question,
            domains,
            temporal_cohort_intent=parsed,
        )
    )
    if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
        return None
    if parsed.disposition is TemporalCohortDisposition.FAMILY_RETENTION:
        return _compile_cohort_retention(question, domains)
    if parsed.disposition is TemporalCohortDisposition.FAMILY_EVIDENCE_FUNNEL:
        return _compile_cohort_evidence(question, domains)
    return None


def compile_semantic_family_sql(
    question: str,
    domains: Domains,
    *,
    age_average_intent: Optional[AgeAverageIntent] = None,
    temporal_cohort_intent: Optional[TemporalCohortIntent] = None,
    obligations: Optional[SemanticObligations] = None,
) -> Optional[str]:
    """Compile one complete high-specificity semantic family."""

    if not isinstance(question, str) or not question.strip() or not isinstance(domains, Mapping):
        return None
    q = question.strip()
    if (
        classify_region_filter_intent(q, domains).disposition
        is RegionFilterDisposition.TERMINAL_ABSTAIN
    ):
        return None
    # ``compile_semantic_family_sql`` is also publicly importable.  Preserve
    # exclusive advanced ownership for direct callers just as the top-level
    # dispatcher does: no generic family may reinterpret a peer/global age
    # comparison after its designated lowerer declines.
    parsed_age = (
        age_average_intent
        if age_average_intent is not None
        else classify_age_average_intent(q, domains)
    )
    if parsed_age.disposition is not AgeAverageDisposition.NOT_APPLICABLE:
        return None
    temporal_intent = (
        temporal_cohort_intent
        if temporal_cohort_intent is not None
        else classify_temporal_cohort_intent(q, domains)
    )
    obligations = (
        obligations
        if obligations is not None
        else extract_semantic_obligations(
            q,
            domains,
            temporal_cohort_intent=temporal_intent,
        )
    )
    if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
        return None
    # These narrow graph families consume their own date/age roles completely.
    # Run them before the broad temporal classifier suppresses graph semantics
    # that its generic disposition cannot represent.
    for compiler in (_compile_observed_risk_chain, _compile_older_only_diseases):
        sql = compiler(q, domains)
        if sql:
            return sql
    if temporal_intent.requires_family_dispatch:
        # Family-owned cohort forms have one designated lowerer.  Advanced-only
        # and terminal cohort dispositions abstain here rather than entering
        # either that lowerer or the broad compiler list below.
        return compile_temporal_cohort_family_sql(
            q,
            domains,
            intent=temporal_intent,
            obligations=obligations,
        )
    if temporal_intent.disposition is not TemporalCohortDisposition.NOT_APPLICABLE:
        return None
    compilers = (
        _compile_relation_edge_listing,
        _compile_relation_tail_entity_listing,
        _compile_relation_head_topk,
        _compile_relation_role_entity_listing,
        _compile_episode_summary,
        _compile_region_benchmark_topk,
        _compile_observed_risk_chain,
        _compile_cohort_retention,
        _compile_older_only_diseases,
        _compile_required_relation_visits,
        _compile_cohort_evidence,
        _compile_ntile_quartiles,
        _compile_cohort_department_group,
        _compile_required_departments,
        _compile_kg_conditional_counts,
        _compile_disease_pairs,
        _compile_two_hop_complication_drug,
        _compile_monthly_yoy,
        _compile_lag_gap,
        _compile_disease_having_topk,
        _compile_first_last,
        _compile_running_cost,
        _compile_department_topk,
        _compile_patient_kg_union,
        _compile_patient_rollup,
        _compile_kg_tail_coverage,
        _compile_kg_required_entities,
        _compile_patient_total_peer_average,
        _compile_patient_exists,
        _compile_age_peer_average,
        _compile_grouped_having,
    )
    for compiler in compilers:
        sql = compiler(q, domains)
        if sql:
            return sql
    return None
