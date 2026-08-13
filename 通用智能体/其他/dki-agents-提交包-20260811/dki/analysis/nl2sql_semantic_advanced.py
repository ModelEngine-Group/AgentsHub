"""Schema-bound advanced NL2SQL compilers.

This module is intentionally a small, conservative semantic layer.  It does
not guess a query shape from a noun such as ``patient`` or ``cost``: every
compiler below requires the structural words, grounded values, and numeric or
temporal slots that make its SQL interpretation unambiguous.  The database
schema is the four-table medical schema used by the rule engine::

    patients(patient_id, name, gender, age, region)
    visits(visit_id, patient_id, visit_date, department, disease,
           drug_prescribed, exam, cost, los_days)

Only SELECT statements are emitted.  Values that become SQL literals come
from the active ``domains`` mapping; dates and numeric bounds are parsed from
the question and rendered as literals after validation.
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
from dki.analysis.nl2sql_compile_result import CompileDisposition, CompileResult
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortIntent,
    TemporalCohortKind,
    classify_temporal_cohort_intent,
)
from dki.analysis.nl2sql_semantic_common import (
    age_predicates,
    answer_items,
    comparison,
    date_predicates,
    dates,
    domain_values,
    esc,
    gender_predicate,
    grounded_predicates,
    integer,
    limit,
    number,
    RegionFilterDisposition,
    classify_region_filter_intent,
    region_predicates,
)


_NUM = r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十])"
_PATIENT = r"(?:患者|病人|病患|病号|参保人|人群)"
_RANGE_UNIT = r"(?:个|次|位|名|条|种|人|元|天)"
_RANGE_SEPARATOR = r"(?:至|到|—|-|~|～|\.\.)"

_OPERATOR_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:至少|不低于|不少于|达到|达|≥|>=)", ">="),
    (r"(?:超过|高于|大于|多于|>)", ">"),
    (r"(?:不超过|不高于|至多|≤|<=)", "<="),
    (r"(?:低于|小于|少于|<)", "<"),
    (r"(?:等于|恰好为|为|=)", "="),
)

_VISIT_DOMAIN_COLUMNS = (
    ("department", "department"),
    ("disease", "disease"),
    ("drug", "drug_prescribed"),
    ("exam", "exam"),
)

_NUMERIC_VISIT_OR_AGGREGATE_LABELS = (
    "就诊次数",
    "就诊量",
    "访问次数",
    "访问量",
    "记录数",
    "记录量",
    "总费用",
    "费用总额",
    "累计费用",
    "合计费用",
    "平均费用",
    "费用均值",
    "费用",
    "花费",
    "金额",
    "住院天数",
    "住院时长",
)
_REGION_TOPK_VISIT_COUNT_LABELS = (
    "就诊次数",
    "就诊量",
    "访问次数",
    "访问量",
    "记录数",
    "记录量",
)
_TOTAL_COST_BOUND_LABELS = (
    "总费用",
    "费用总额",
    "累计费用",
    "合计费用",
    "总花费",
    "累计花费",
    "患者总额",
    "患者累计费用",
)
_RUNNING_COST_BOUND_LABELS = ("累计费用", "运行总费用", "累计花费")


# ---------------------------------------------------------------------------
# Small parsing/rendering helpers


def _has(question: str, pattern: str) -> bool:
    return re.search(pattern, question, flags=re.IGNORECASE) is not None


def _literal(token: str) -> Optional[str]:
    """Normalize an Arabic or supported Chinese integer token."""

    try:
        parsed = integer(token)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        return str(parsed)
    try:
        return number(token)
    except (TypeError, ValueError):
        return None


def _threshold(question: str, labels: Sequence[str]) -> Optional[tuple[str, str]]:
    """Parse a comparison whose subject is one of ``labels``.

    ``common.comparison`` is tried first so the shared slot semantics remain
    the source of truth.  The second pass accepts the common Chinese form
    ``至少有 3 次就诊`` where the number precedes the semantic label.
    """

    for label in labels:
        parsed = comparison(question, label)
        if parsed is not None:
            return parsed

    label_re = "(?:" + "|".join(re.escape(item) for item in sorted(labels, key=len, reverse=True)) + ")"
    for operator_re, operator in _OPERATOR_PATTERNS:
        for pattern in (
            rf"{label_re}\s*{operator_re}\s*({_NUM})",
            # A positive reverse comparator must not start at the second
            # character of a negative phrase such as ``不超过 5 次就诊``.
            rf"(?<!不){operator_re}\s*(?:有|达到|为)?\s*({_NUM})\s*(?:个|次|位|名|条|种|人|元|天)?\s*(?:的)?\s*{label_re}",
        ):
            match = re.search(pattern, question)
            if match:
                value = _literal(match.group(1))
                if value is not None:
                    return operator, value
    return None


def _count_threshold(
    question: str,
    labels: Sequence[str],
    *,
    units: str = r"(?:个|次|位|名|条|人)?",
) -> Optional[tuple[str, str]]:
    parsed = _threshold(question, labels)
    if parsed is not None:
        return parsed
    # Covers "至少有 3 次就诊" and "超过 2 个科室" without making a
    # generic number into a threshold.
    for operator_re, operator in _OPERATOR_PATTERNS:
        match = re.search(
            rf"(?<!不){operator_re}\s*(?:有|达到|为)?\s*({_NUM})\s*{units}\s*(?:的)?"
            rf"(?:{'|'.join(re.escape(item) for item in labels)})",
            question,
        )
        if match:
            value = _literal(match.group(1))
            if value is not None:
                return operator, value
    return None


def _thresholds(question: str, labels: Sequence[str]) -> list[tuple[str, str]]:
    """Parse every unambiguous bound for a single aggregate metric."""

    label_re = "(?:" + "|".join(
        re.escape(item) for item in sorted(set(labels), key=len, reverse=True)
    ) + ")"
    matches: list[tuple[int, str, str]] = []
    seen: set[tuple[int, int, str, str]] = set()
    operator_union = "(?:" + "|".join(pattern for pattern, _ in _OPERATOR_PATTERNS) + ")"
    range_prefix = r"(?:在|介于|范围(?:为|是)?|从|为|是|=)?"

    def starts_range_after(end: int) -> bool:
        return re.match(
            rf"\s*{_RANGE_UNIT}?\s*{_RANGE_SEPARATOR}",
            question[end:],
        ) is not None

    def normalize_operator(raw_operator: str) -> Optional[str]:
        for pattern, operator in _OPERATOR_PATTERNS:
            if re.fullmatch(pattern, raw_operator, re.I):
                return operator
        return None

    def add(start: int, end: int, operator: str, raw_value: str) -> None:
        value = _literal(raw_value)
        if value is None:
            return
        key = (start, end, operator, value)
        if key not in seen:
            matches.append((start, operator, value))
            seen.add(key)

    for operator_re, operator in _OPERATOR_PATTERNS:
        labelled = re.compile(
            rf"{label_re}\s*(?P<operator>{operator_re})\s*(?P<value>{_NUM})",
            re.I,
        )
        for match in labelled.finditer(question):
            # A scalar prefix such as ``就诊次数为 2`` cannot be retained
            # when it begins an inclusive range like ``为 2 至 5 次``.
            if starts_range_after(match.end()):
                continue
            add(match.start(), match.end(), operator, match.group("value"))
            cursor = match.end()
            while True:
                following = re.match(
                    rf"\s*(?:且|并且|及)\s*(?P<operator>{operator_union})\s*"
                    rf"(?P<value>{_NUM})",
                    question[cursor:],
                    re.I,
                )
                if following is None:
                    break
                following_operator = normalize_operator(following.group("operator"))
                if following_operator is None:
                    break
                add(
                    cursor + following.start("operator"),
                    cursor + following.end("value"),
                    following_operator,
                    following.group("value"),
                )
                cursor += following.end()
        reverse = re.compile(
            rf"(?P<operator>(?<!不){operator_re})\s*(?:有|达到|为)?\s*"
            rf"(?P<value>{_NUM})\s*(?:个|次|位|名|条|种|人|元|天)?\s*(?:的)?\s*{label_re}",
            re.I,
        )
        for match in reverse.finditer(question):
            add(match.start(), match.end(), operator, match.group("value"))
            # Preserve an adjacent upper/lower bound in natural wording such
            # as "至少有 2 次就诊且不超过 5 次".  The metric is named only
            # once, so a second label is not required after the connector.
            cursor = match.end()
            while True:
                following = re.match(
                    rf"\s*(?:且|并且|及)\s*(?P<operator>{operator_union})\s*"
                    rf"(?:有|达到|为)?\s*(?P<value>{_NUM})"
                    r"(?:\s*(?:个|次|位|名|条|种|人|元|天))?",
                    question[cursor:],
                    re.I,
                )
                if following is None:
                    break
                following_operator = normalize_operator(following.group("operator"))
                if following_operator is None:
                    break
                add(
                    cursor + following.start("operator"),
                    cursor + following.end("value"),
                    following_operator,
                    following.group("value"),
                )
                cursor += following.end()

    range_patterns = (
        rf"{label_re}\s*{range_prefix}\s*(?P<low>{_NUM})\s*{_RANGE_UNIT}?\s*"
        rf"{_RANGE_SEPARATOR}\s*(?P<high>{_NUM})\s*{_RANGE_UNIT}?(?:之间)?",
        rf"{label_re}\s*{range_prefix}\s*(?P<low>{_NUM})\s*{_RANGE_UNIT}?\s*和\s*"
        rf"(?P<high>{_NUM})\s*{_RANGE_UNIT}?(?:之间)?",
    )
    for pattern in range_patterns:
        for match in re.finditer(pattern, question, re.I):
            low = _literal(match.group("low"))
            high = _literal(match.group("high"))
            if low is None or high is None:
                continue
            try:
                if float(low) > float(high):
                    continue
            except ValueError:
                continue
            add(match.start("low"), match.end("low"), ">=", match.group("low"))
            add(match.start("high"), match.end("high"), "<=", match.group("high"))
    return [(operator, value) for _, operator, value in sorted(matches)]


def _threshold_intent(question: str, labels: Sequence[str]) -> bool:
    """Detect a requested aggregate bound even when it is malformed."""

    label_re = "(?:" + "|".join(
        re.escape(item) for item in sorted(set(labels), key=len, reverse=True)
    ) + ")"
    operator_re = "(?:" + "|".join(pattern for pattern, _ in _OPERATOR_PATTERNS) + ")"
    non_equality_operator_re = "(?:" + "|".join(
        pattern for pattern, operator in _OPERATOR_PATTERNS if operator != "="
    ) + ")"
    range_attempt = (
        rf"{label_re}\s*(?:(?:在|介于|范围(?:为|是)?|从|为|是|=)\s*)?"
        rf"{_NUM}\s*{_RANGE_UNIT}?\s*{_RANGE_SEPARATOR}"
    )
    return bool(
        re.search(
            rf"{label_re}\s*(?:{operator_re}|在|介于|范围|从)",
            question,
            re.I,
        )
        or re.search(range_attempt, question, re.I)
        or re.search(
            rf"(?<!不){operator_re}\s*(?:有|达到|为)?\s*{_NUM}\s*"
            rf"(?:个|次|位|名|条|种|人|元|天)?\s*(?:的)?\s*{label_re}",
            question,
            re.I,
        )
        # Equality phrases such as ``分为四分位`` are structural grouping
        # cues, not reverse numeric bounds.  Non-equality forms can safely
        # retain a short descriptive gap so ``至少 100 元的单次费用`` is still
        # recognized and cannot be silently dropped.
        or re.search(
            rf"(?<!不){non_equality_operator_re}\s*(?:有|达到|为)?\s*{_NUM}"
            rf"[^。；;]{{0,32}}{label_re}",
            question,
            re.I,
        )
    )


def _complete_metric_thresholds(
    question: str,
    labels: Sequence[str],
) -> Optional[list[tuple[str, str]]]:
    """Return all renderable bounds or abstain from incomplete metric intent.

    The advanced lowerers use derived totals and window values at different SQL
    stages.  This shared coverage check ensures an expressed metric bound is
    either lowered at its declared stage or prevents a broader query from being
    emitted; callers do not add their own ad-hoc qualifier recognition.
    """

    bounds = _thresholds(question, labels)
    if _threshold_intent(question, labels) and not bounds:
        return None
    return bounds


def _has_unowned_numeric_metric_threshold(
    question: str,
    covered_labels: Sequence[str],
) -> bool:
    """Return whether another numeric metric request would be discarded.

    A derived-metric lowerer may own one named threshold (for example patient
    total cost or a running total), but must not reinterpret a bound for visit
    count, average cost, or source-row cost.  Mask complete owned bound clauses
    before checking the shared numeric vocabulary so the generic ``费用`` token
    inside ``累计费用至少1000元`` does not create a false uncovered qualifier.
    """

    label_re = "(?:" + "|".join(
        re.escape(label) for label in sorted(set(covered_labels), key=len, reverse=True)
    ) + ")"
    operator_re = "(?:" + "|".join(pattern for pattern, _ in _OPERATOR_PATTERNS) + ")"
    range_prefix = r"(?:在|介于|范围(?:为|是)?|从|为|是|=)?"
    patterns = (
        rf"{label_re}\s*{range_prefix}\s*{_NUM}\s*{_RANGE_UNIT}?\s*"
        rf"{_RANGE_SEPARATOR}\s*{_NUM}\s*{_RANGE_UNIT}?(?:之间)?",
        rf"{label_re}\s*{operator_re}\s*{_NUM}\s*(?:个|次|位|名|条|种|人|元|天)?",
        rf"{operator_re}\s*(?:有|达到|为)?\s*{_NUM}\s*"
        rf"(?:个|次|位|名|条|种|人|元|天)?\s*(?:的)?\s*{label_re}",
    )
    masked = question
    for pattern in patterns:
        masked = re.sub(pattern, lambda match: " " * len(match.group(0)), masked, flags=re.I)
    uncovered_labels = tuple(
        label for label in _NUMERIC_VISIT_OR_AGGREGATE_LABELS
        if label not in covered_labels
    )
    return _threshold_intent(masked, uncovered_labels)


def _limit(question: str) -> Optional[int]:
    parsed = limit(question)
    if parsed is not None:
        return parsed
    for pattern in (
        r"(?:Top|top)\s*-?\s*(?:K\s*[=:：]\s*)?(\d+)",
        r"K\s*[=:：]\s*(\d+)",
        r"(?:前|取|保留)\s*(\d+)\s*(?:个|位|名|条|次|人)?",
    ):
        match = re.search(pattern, question)
        if match:
            try:
                value = int(match.group(1))
            except ValueError:
                return None
            return value if value > 0 else None
    return None


def _date_filters(question: str, alias: str) -> list[str]:
    """Render explicit date bounds, using the shared date slot parser."""

    values = dates(question)
    if len(values) >= 2:
        prefix = f"{alias}." if alias else ""
        return [
            f"{prefix}visit_date >= '{values[0]}'",
            f"{prefix}visit_date <= '{values[1]}'",
        ]
    return date_predicates(question, alias=alias)


def _where(predicates: Sequence[str]) -> str:
    return " WHERE " + " AND ".join(predicates) if predicates else ""


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _explicit_age_predicates(question: str, *, alias: str = "p") -> list[str]:
    """Use the shared age parser only when an age slot is actually stated.

    ``age_predicates`` intentionally accepts phrases such as ``至少 40``;
    callers must guard it so a cost or count threshold is not mistaken for an
    age bound.
    """

    if not _has(question, r"年龄|岁"):
        return []
    return age_predicates(question, alias=alias)


def _patient_filters(question: str, domains: Mapping[str, Sequence[str]], *, alias: str = "p") -> list[str]:
    """Return explicit patient predicates without inventing a region value."""

    predicates: list[str] = []
    predicates.extend(_explicit_age_predicates(question, alias=alias))
    gender = gender_predicate(question, alias=alias)
    if gender is not None:
        predicates.append(gender)
    predicates.extend(region_predicates(question, domains, alias=alias))
    return _dedupe(predicates)


def _visit_filters(
    question: str,
    domains: Mapping[str, Sequence[str]],
    *,
    alias: str = "v",
) -> list[str]:
    """Collect date and grounded visit predicates.

    ``grounded_predicates`` deliberately handles domain matching and escaping;
    this wrapper only adds temporal slots and removes duplicate predicates.
    """

    predicates = _date_filters(question, alias)
    predicates.extend(grounded_predicates(question, domains, alias=alias))
    return _dedupe(predicates)


def _has_numeric_visit_or_aggregate_constraint(question: str) -> bool:
    """Return whether a request contains an unsupported numeric visit scope."""

    return _threshold_intent(question, _NUMERIC_VISIT_OR_AGGREGATE_LABELS)


def _has_unsupported_non_date_filters(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Return whether a date-only analytical shape would drop a qualifier.

    Monthly and cohort compilers have deliberately narrow contracts: their
    temporal predicate has a documented placement, while applying a patient,
    visit-domain, or numeric visit/aggregate qualifier can change either the
    cohort anchor or later-event population.  Do not guess that placement.
    """

    return bool(
        _patient_filters(question, domains)
        or grounded_predicates(question, domains)
        or _has_numeric_visit_or_aggregate_constraint(question)
    )


def _value_occurrences(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> list[tuple[int, int, str, str]]:
    """Find non-overlapping grounded visit values in question order."""

    candidates: list[tuple[int, int, str, str]] = []
    for key, column in _VISIT_DOMAIN_COLUMNS:
        values = sorted(
            {str(value) for value in domains.get(key, ()) if str(value)},
            key=lambda value: (-len(value), value),
        )
        for value in values:
            for match in re.finditer(re.escape(value), question):
                candidates.append((match.start(), match.end(), column, value))

    selected: list[tuple[int, int, str, str]] = []
    for candidate in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]), item[2], item[3])):
        if any(candidate[0] < used[1] and candidate[1] > used[0] for used in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item[0])


def _projection(
    question: str,
    patterns: Sequence[tuple[str, str]],
) -> Optional[list[str]]:
    """Parse every item in an explicit answer clause.

    Returning ``None`` for an unrecognised item is important: silently
    omitting a requested column is worse than abstaining for a compiler used
    as a rule-engine component.
    """

    items = answer_items(question)
    if not items:
        return None
    result: list[str] = []
    for raw_item in items:
        item = raw_item.strip()
        # Ranking phrases often decorate the actual requested entity, but a
        # standalone "排名" is itself a requested rank column.
        if not re.fullmatch(r"排名|名次|序号", item):
            item = re.sub(r"^(?:排名|名次|前\s*\d+\s*(?:个|位|名|条)?(?:的)?)[，,、 ]*", "", item)
        matched: Optional[str] = None
        for key, pattern in patterns:
            if re.search(pattern, item, flags=re.IGNORECASE):
                matched = key
                break
        if matched is None:
            return None
        if matched not in result:
            result.append(matched)
    return result or None


_PATIENT_PROJECTION = (
    ("patient_id", r"患者\s*(?:编号|ID)|病人\s*(?:编号|ID)|患者ID|编号|\bID\b"),
    ("name", r"姓名|名字|名称"),
    ("age", r"年龄"),
    ("gender", r"性别"),
    ("region", r"地区|区域|城市"),
)

_PATIENT_METRIC_PROJECTION = (
    ("visit_count", r"就诊(?:次数|量)|访问次数|记录(?:数|量)|频次"),
    ("total_cost", r"(?:总|累计|合计|总计)\s*(?:费用|花费|成本)|(?:费用|花费|成本)\s*(?:总额|累计)"),
    ("department_count", r"(?:不同|涉及|覆盖|就诊)?\s*科室\s*(?:数|数量)"),
    ("avg_total_cost", r"(?:地区|区域|同区).*平均.*(?:累计|总).*费用|平均.*(?:累计|总).*费用"),
)

_VISIT_PROJECTION = (
    ("visit_id", r"就诊\s*(?:编号|ID)|记录\s*(?:编号|ID)"),
    ("patient_id", r"患者\s*(?:编号|ID)|病人\s*(?:编号|ID)"),
    ("visit_date", r"就诊日期|就诊时间|日期|时间"),
    ("department", r"就诊科室|科室"),
    ("disease", r"疾病|诊断"),
    ("drug", r"处方药物|处方药|药物"),
    ("exam", r"检查"),
    ("los_days", r"住院天数|住院时长"),
    # Specific derived metrics precede the generic cost token.
    ("running_cost", r"累计费用|运行总费用|累计花费"),
    ("cost", r"费用|花费|金额|成本"),
    ("gap_days", r"间隔天数|相隔天数|间隔日数|gap"),
    ("previous_visit_date", r"前一次日期|上一次日期|此前日期|前次日期"),
    ("current_visit_date", r"当前日期|后一次日期|本次日期|后次日期"),
    ("episode_id", r"就诊事件|事件编号|episode"),
    ("visit_rank", r"排名|名次|序号"),
)

_GROUP_PROJECTION = (
    ("department", r"科室"),
    ("disease", r"疾病|诊断"),
    ("drug", r"药物|处方药"),
    ("exam", r"检查"),
    ("region", r"地区|区域|城市"),
    ("visit_month", r"月份|月度|就诊月|月份"),
    ("visit_count", r"就诊(?:次数|量)|记录(?:数|量)|频次"),
    ("patient_count", r"患者数|患者人数|不同患者"),
    ("avg_cost", r"平均(?:费用|花费|成本)|(?:费用|花费|成本)均值"),
    ("total_cost", r"(?:总|累计|合计|总计)(?:费用|花费|成本)|(?:费用|花费|成本)总额"),
)


def _group_dimension(question: str) -> Optional[tuple[str, str]]:
    """Return a single explicit visit grouping dimension."""

    candidates = (
        ("visit_month", r"(?:按|按照|以|按每个)?\s*(?:月|月份|就诊月)|每月|月度|逐月"),
        ("department", r"(?:按|按照|以|各|每个|每一)\s*(?:个)?\s*科室|科室\s*(?:分组|统计|汇总|分布)"),
        ("disease", r"(?:按|按照|以|各|每个|每一)\s*(?:种|个)?\s*疾病|疾病\s*(?:分组|统计|汇总|分布)"),
        ("drug", r"(?:按|按照|以|各|每个|每一)\s*(?:种|个)?\s*(?:处方)?药物|药物\s*(?:分组|统计|汇总|分布)"),
        ("exam", r"(?:按|按照|以|各|每个|每一)\s*(?:种|个)?\s*检查|检查\s*(?:分组|统计|汇总|分布)"),
        ("region", r"(?:按|按照|以|各|每个|每一)\s*(?:个)?\s*(?:地区|区域|城市)|(?:地区|区域|城市)\s*(?:分组|统计|汇总|分布)"),
    )
    found: list[tuple[int, str, str]] = []
    for key, pattern in candidates:
        match = re.search(pattern, question)
        if match:
            found.append((match.start(), key, pattern))
    keys = {key for _, key, _ in found}
    if len(keys) != 1:
        return None
    key = next(iter(keys))
    columns = {
        "visit_month": "substr(v.visit_date, 1, 7)",
        "department": "v.department",
        "disease": "v.disease",
        "drug": "v.drug_prescribed",
        "exam": "v.exam",
        "region": "p.region",
    }
    return key, columns[key]


def _implicit_grouped_having_shape(
    question: str,
) -> Optional[tuple[str, str, str]]:
    """Recognize a bounded ``metric threshold -> groups`` development shape."""

    suffix = r"(?:有)?(?:哪些|哪一些|是什么|有哪几种|有哪几个)"
    punctuation = r"[。！？?!]?"
    cases = (
        (
            "disease",
            "v.disease",
            "visit_count",
            rf"(?:就诊次数|就诊量|访问次数|记录数)"
            rf"(?:超过|高于|大于|多于|不少于|至少|不低于)"
            rf"\s*{_NUM}\s*(?:次|条|个)?的(?:疾病|诊断){suffix}{punctuation}",
        ),
        (
            "department",
            "v.department",
            "patient_count",
            rf"(?:患者数|患者人数|不同患者数)"
            rf"(?:超过|高于|大于|多于|不少于|至少|不低于)"
            rf"\s*{_NUM}\s*(?:人|名|个)?的科室{suffix}{punctuation}",
        ),
        (
            "department",
            "v.department",
            "avg_cost",
            rf"(?:平均费用|费用均值|费用平均值|平均花费|平均成本)"
            rf"(?:超过|高于|大于|不少于|至少|不低于)"
            rf"\s*{_NUM}\s*元?的科室(?:及其|和)"
            rf"(?:平均费用|费用均值|费用平均值|平均花费|平均成本)"
            rf"{punctuation}",
        ),
    )
    for group_key, group_column, metric, pattern in cases:
        if re.fullmatch(pattern, question):
            return group_key, group_column, metric
    return None


def _metric_expression(metric: str) -> Optional[tuple[str, str]]:
    return {
        "visit_count": ("COUNT(*)", "visit_count"),
        "patient_count": ("COUNT(DISTINCT v.patient_id)", "patient_count"),
        "avg_cost": ("AVG(v.cost)", "avg_cost"),
        "total_cost": ("SUM(v.cost)", "total_cost"),
    }.get(metric)


def _group_thresholds(
    question: str,
) -> Optional[dict[str, list[tuple[str, str]]]]:
    """Return every aggregate bound, or defer an unrenderable request."""

    specifications = {
        "visit_count": ("就诊次数", "就诊量", "记录数", "访问次数", "频次"),
        "patient_count": ("患者数", "患者人数", "不同患者数"),
        "avg_cost": ("平均费用", "费用均值", "费用平均值", "平均花费", "平均成本"),
        "total_cost": ("总费用", "费用总额", "累计费用", "合计费用", "总花费"),
    }
    result: dict[str, list[tuple[str, str]]] = {}
    for metric, labels in specifications.items():
        bounds = _thresholds(question, labels)
        if bounds:
            result[metric] = bounds
        elif _threshold_intent(question, labels):
            # A threshold that cannot be associated with an aggregate exactly
            # must not be silently omitted from HAVING.
            return None
    return result


def _ordered_metrics(question: str, selected: Sequence[str], thresholds: Mapping[str, object]) -> list[str]:
    """Use requested metric order, then ensure thresholded metrics are visible."""

    result = list(selected)
    for metric in thresholds:
        if metric not in result:
            return []
    return result


def _order_direction(question: str, *, default: str = "DESC") -> str:
    if _has(question, r"升序|从低到高|由小到大|最早"):
        return "ASC"
    if _has(question, r"降序|从高到低|由大到小|最高|最多|最贵"):
        return "DESC"
    return default


# ---------------------------------------------------------------------------
# Individual advanced compilers


def _compile_grouped_having(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    implicit_shape = _implicit_grouped_having_shape(question)
    if (
        implicit_shape is None
        and not _has(question, r"就诊|访问|记录")
    ) or _has(question, r"患者滚动|patient_rollup|累计费用高于同地区"):
        return None
    grouping = (
        (implicit_shape[0], implicit_shape[1])
        if implicit_shape is not None
        else _group_dimension(question)
    )
    if grouping is None or (
        implicit_shape is None
        and not _has(question, r"(?:筛选|找出|返回|列出|统计|汇总)")
    ):
        return None
    group_key, group_column = grouping
    thresholds = _group_thresholds(question)
    if thresholds is None or not thresholds:
        return None
    projection = (
        [implicit_shape[0], implicit_shape[2]]
        if implicit_shape is not None
        else _projection(question, _GROUP_PROJECTION)
    )
    if projection is None:
        return None
    selected_metrics = [item for item in projection if item in {"visit_count", "patient_count", "avg_cost", "total_cost"}]
    selected_group = [item for item in projection if item == group_key]
    if not selected_group or not selected_metrics:
        return None
    metrics = _ordered_metrics(question, selected_metrics, thresholds)
    if not metrics:
        return None

    select_parts = [group_column]
    having_parts: list[str] = []
    order_alias: Optional[str] = None
    for metric in metrics:
        expression = _metric_expression(metric)
        if expression is None:
            return None
        sql_expression, alias = expression
        selected_expression = (
            f"ROUND({sql_expression}, 2)"
            if implicit_shape is not None and metric == "avg_cost"
            else sql_expression
        )
        select_parts.append(f"{selected_expression} AS {alias}")
        for operator, value in thresholds.get(metric, []):
            having_parts.append(f"{sql_expression} {operator} {value}")
        if order_alias is None and metric in {"visit_count", "patient_count", "avg_cost", "total_cost"}:
            order_alias = alias

    predicates = _visit_filters(question, domains)
    patient_predicates = _patient_filters(question, domains, alias="p")
    if group_key == "region" or patient_predicates:
        # Region groups and patient eligibility predicates both require the
        # patient table.  Keeping every parsed qualifier prevents a grouped
        # aggregate from silently broadening its population.
        from_sql = "visits AS v JOIN patients AS p ON p.patient_id = v.patient_id"
        predicates = _dedupe(predicates + patient_predicates)
    else:
        from_sql = "visits AS v"
    group_by = group_column
    order_alias = order_alias or metrics[0]
    direction = _order_direction(question)
    order_clause = f" ORDER BY {order_alias} {direction}"
    if implicit_shape is None:
        order_clause += f", {group_key} ASC"
    return (
        f"SELECT {', '.join(select_parts)} FROM {from_sql}"
        f"{_where(predicates)} GROUP BY {group_by}"
        f" HAVING {' AND '.join(having_parts)}"
        f"{order_clause}"
    )


def _is_department_gender_visit_count_request(question: str) -> bool:
    return re.fullmatch(
        r"(?:统计)?每个科室每种(?:患者)?性别的"
        r"(?:就诊次数|就诊量|访问次数|记录数)[。！？?!]?",
        question,
    ) is not None


def is_fallback_fenced_group_request(question: str) -> bool:
    """Return whether a compact group request has one exclusive lowerer."""

    return (
        isinstance(question, str)
        and (
            _implicit_grouped_having_shape(question) is not None
            or _is_department_gender_visit_count_request(question)
        )
    )


def _compile_department_gender_visit_count(
    question: str,
    _domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    """Lower the exact two-dimensional department-by-gender count shape."""

    if not _is_department_gender_visit_count_request(question):
        return None
    return (
        "SELECT v.department, p.gender, COUNT(*) AS visit_count "
        "FROM visits AS v JOIN patients AS p "
        "ON v.patient_id = p.patient_id "
        "GROUP BY v.department, p.gender "
        "ORDER BY v.department ASC, p.gender ASC"
    )


def _rollup_projection(question: str) -> Optional[list[str]]:
    return _projection(question, _PATIENT_PROJECTION + _PATIENT_METRIC_PROJECTION)


def _compile_patient_rollup(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _has(question, r"患者|病人") or not _has(question, r"(?:筛选|过滤|找出|返回|列出)"):
        return None
    if not _has(question, r"patient_rollup|患者汇总|汇总每位患者|每位患者汇总"):
        # The structural combination is distinctive enough to accept natural
        # wording without requiring a literal CTE token.
        if not (
            _has(question, r"就诊次数|就诊量")
            and _has(question, r"总费用|累计费用|费用总额")
            and _has(question, r"科室数|科室数量|覆盖科室|不同科室")
            and _has(question, r"年龄")
        ):
            return None
    ages = _explicit_age_predicates(question, alias="p")
    visit_count_bounds = _thresholds(
        question,
        ("就诊次数", "就诊量", "访问次数", "记录数"),
    )
    total_cost_bounds = _thresholds(
        question,
        ("总费用", "费用总额", "累计费用", "合计费用"),
    )
    department_count_bounds = _thresholds(
        question,
        ("科室数", "科室数量", "不同科室数", "覆盖科室数", "涉及科室数"),
    )
    if not ages or not visit_count_bounds or not total_cost_bounds or not department_count_bounds:
        return None
    projection = _rollup_projection(question)
    if projection is None or "patient_id" not in projection:
        return None
    required = {"age", "visit_count", "total_cost", "department_count"}
    if not required.issubset(projection):
        return None

    base_patient_columns = ["p.patient_id", "p.name", "p.age", "p.gender", "p.region"]
    base_metrics = [
        "COUNT(v.visit_id) AS visit_count",
        "SUM(v.cost) AS total_cost",
        "COUNT(DISTINCT v.department) AS department_count",
    ]
    outer_columns = [
        {
            "patient_id": "patient_id",
            "name": "name",
            "age": "age",
            "gender": "gender",
            "region": "region",
            "visit_count": "visit_count",
            "total_cost": "total_cost",
            "department_count": "department_count",
        }[item]
        for item in projection
    ]
    predicates = [predicate.replace("p.", "") for predicate in ages]
    predicates.extend(
        f"visit_count {operator} {value}"
        for operator, value in visit_count_bounds
    )
    predicates.extend(
        f"total_cost {operator} {value}"
        for operator, value in total_cost_bounds
    )
    predicates.extend(
        f"department_count {operator} {value}"
        for operator, value in department_count_bounds
    )
    # Explicit demographic filters apply to the rollup output.  Do not use a
    # region value unless the shared domain matcher found it in the question.
    for predicate in _patient_filters(question, domains, alias="p"):
        if predicate not in ages:
            predicates.append(predicate.replace("p.", ""))
    predicates = _dedupe(predicates)
    # Visit qualifiers define the source rows for every derived metric.  They
    # cannot be moved to the outer CTE query without changing aggregate values.
    visit_filters = _visit_filters(question, domains, alias="v")
    return (
        "WITH patient_rollup AS ("
        f"SELECT {', '.join(base_patient_columns + base_metrics)} "
        "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        f"{_where(visit_filters)} "
        "GROUP BY p.patient_id, p.name, p.age, p.gender, p.region) "
        f"SELECT {', '.join(outer_columns)} FROM patient_rollup"
        f"{_where(predicates)} ORDER BY patient_id ASC"
    )


def _age_average_intent(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> AgeAverageIntent:
    """Return the exclusive ownership record for a patient age-average query."""

    return classify_age_average_intent(question, domains)


def _temporal_cohort_intent(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> TemporalCohortIntent:
    """Return the exclusive owner for a temporal cohort request."""

    return classify_temporal_cohort_intent(question, domains)


def _has_unsupported_age_peer_average_visit_constraints(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Keep the historical visit-scope guard available to local callers."""

    return _age_average_intent(question, domains).has_visit_qualifier


def _compile_age_peer_average(
    question: str,
    domains: Mapping[str, Sequence[str]],
    *,
    intent: Optional[AgeAverageIntent] = None,
) -> Optional[str]:
    """Lower only a fully covered correlated peer-age comparison."""

    parsed = intent or _age_average_intent(question, domains)
    if parsed.disposition is not AgeAverageDisposition.ADVANCED_PEER:
        return None
    if not parsed.projection or not {"patient_id", "age"}.issubset(parsed.projection):
        return None

    peer_predicates: list[str] = []
    if parsed.peer_by_region:
        peer_predicates.append("peer.region = p.region")
    if parsed.peer_by_gender:
        peer_predicates.append("peer.gender = p.gender")
    if parsed.excludes_outer_patient_from_peer:
        peer_predicates.append("peer.patient_id <> p.patient_id")
    if not peer_predicates:
        return None

    # Direct demographic qualifiers narrow the returned patients while the
    # correlated predicates define each patient's comparison cohort.  In
    # particular, an explicit positive region such as Shenzhen must remain an
    # outer restriction as well as retaining ``peer.region = p.region``.  The
    # intent owns direct ``女病人``/``男病人`` spelling too, which the legacy
    # primitive gender slot did not recognize.
    outer_predicates = _patient_filters(question, domains, alias="p")
    if parsed.outer_gender is not None:
        gender = f"p.gender = '{parsed.outer_gender}'"
        if gender not in outer_predicates:
            outer_predicates.append(gender)
    outer_predicates = _dedupe(outer_predicates)
    outer_suffix = " AND " + " AND ".join(outer_predicates) if outer_predicates else ""
    return (
        f"SELECT {', '.join('p.' + item for item in parsed.projection)} FROM patients AS p "
        f"WHERE p.age > (SELECT AVG(peer.age) FROM patients AS peer WHERE {' AND '.join(peer_predicates)})"
        f"{outer_suffix} ORDER BY p.age DESC, p.patient_id ASC"
    )


def _patient_exists_conditions(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> list[tuple[str, str]]:
    """Return distinct independent-EXISTS visit conditions in mention order."""

    conditions: list[tuple[str, str]] = []
    for _, _, column, value in _value_occurrences(question, domains):
        condition = (column, value)
        if condition not in conditions:
            conditions.append(condition)
    return conditions


def _is_patient_exists_request(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Recognize the narrow independent-EXISTS signature before fallback."""

    return bool(
        _has(question, _PATIENT)
        and _has(question, r"(?:既.*又|同时|分别|且|并且|以及|都|至少一次)")
        and not _has(question, r"同一次|同一条|同一记录|同次就诊")
        and len(_patient_exists_conditions(question, domains)) >= 2
    )


def _has_unsupported_patient_exists_visit_constraints(question: str) -> bool:
    """Identify visit constraints outside independent-EXISTS semantics."""

    return bool(
        _date_filters(question, "")
        or _has_numeric_visit_or_aggregate_constraint(question)
    )


def _compile_patient_exists(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _is_patient_exists_request(question, domains):
        return None
    # A date or numeric predicate could constrain every EXISTS subquery, one
    # particular condition, or a same-row event.  This compiler has no such
    # contract, so it must not emit a broader independent-EXISTS query.
    if _has_unsupported_patient_exists_visit_constraints(question):
        return None
    unique_conditions = _patient_exists_conditions(question, domains)
    projection = _projection(question, _PATIENT_PROJECTION)
    if projection is None or "patient_id" not in projection:
        return None

    exists_parts = []
    for index, (column, value) in enumerate(unique_conditions, start=1):
        exists_parts.append(
            f"EXISTS (SELECT 1 FROM visits AS v{index} "
            f"WHERE v{index}.patient_id = p.patient_id "
            f"AND v{index}.{column} = '{esc(value)}')"
        )
    patient_predicates = _patient_filters(question, domains, alias="p")
    return (
        f"SELECT {', '.join('p.' + item for item in projection)} FROM patients AS p "
        f"WHERE {' AND '.join(exists_parts + patient_predicates)} "
        "ORDER BY p.patient_id ASC"
    )


def _mask_qualifier_spans(
    question: str,
    spans: Sequence[tuple[int, int]],
) -> str:
    """Hide consumed qualifier spans while preserving source positions."""

    characters = list(question)
    for start, end in spans:
        for index in range(max(start, 0), min(end, len(characters))):
            characters[index] = " "
    return "".join(characters)


def _patient_total_peer_average_spans(question: str) -> list[tuple[int, int]]:
    """Return the complete total-cost peer-average clauses owned by this shape."""

    total_cost = "(?:" + "|".join(
        re.escape(label) for label in sorted(_TOTAL_COST_BOUND_LABELS, key=len, reverse=True)
    ) + ")"
    comparator = r"(?:高于|超过|大于)"
    peer_scope = r"同(?:一|个)?(?:地区|区域|城市)"
    peer_average = (
        rf"{peer_scope}\s*(?:(?:所有|全部|其他|其它)?\s*{_PATIENT})?\s*"
        rf"(?:的)?\s*(?:平均|均值)\s*(?:的)?\s*{total_cost}"
    )
    spans: set[tuple[int, int]] = set()
    for pattern in (
        rf"{total_cost}\s*{comparator}\s*{peer_average}",
        rf"{comparator}\s*{peer_average}",
    ):
        spans.update((match.start(), match.end()) for match in re.finditer(pattern, question, re.I))
    return sorted(spans)


def _is_patient_total_peer_average_request(question: str) -> bool:
    """Recognize a same-region comparison whose subject is patient total cost."""

    # A total-cost word merely requested in the output must not turn an age
    # peer comparison into a cost comparison.  The full comparison clause is
    # later marked as consumed before scalar metric-bound coverage is checked.
    return bool(
        _has(question, _PATIENT)
        and _patient_total_peer_average_spans(question)
    )


def _compile_patient_total_peer_average(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _is_patient_total_peer_average_request(question):
        return None
    projection = _projection(
        question,
        _PATIENT_PROJECTION + (("total_cost", _PATIENT_METRIC_PROJECTION[1][1]), ("avg_total_cost", _PATIENT_METRIC_PROJECTION[3][1])),
    )
    if projection is None or "patient_id" not in projection or "total_cost" not in projection:
        return None
    # The same-region average comparison is this compiler's primary predicate,
    # not an incomplete scalar total-cost bound.  Mark its complete span as
    # consumed before assessing the remaining numeric qualifiers.
    scalar_qualifier_question = _mask_qualifier_spans(
        question,
        _patient_total_peer_average_spans(question),
    )
    total_cost_bounds = _complete_metric_thresholds(
        scalar_qualifier_question,
        _TOTAL_COST_BOUND_LABELS,
    )
    if total_cost_bounds is None:
        return None
    if _has_unowned_numeric_metric_threshold(
        scalar_qualifier_question,
        _TOTAL_COST_BOUND_LABELS,
    ):
        return None
    # Visit constraints define each patient's accumulated total.  Demographic
    # constraints narrow returned patients below, preserving the regional peer
    # baseline rather than silently discarding the requested patient cohort.
    visit_predicates = _visit_filters(question, domains, alias="v")
    base_where = _where(visit_predicates)
    patient_predicates = _patient_filters(question, domains, alias="p")
    base_columns = ["p.patient_id", "p.name", "p.region"]
    outer_columns: list[str] = []
    for item in projection:
        mapping = {
            "patient_id": "pt.patient_id",
            "name": "pt.name",
            "age": "pt.age",
            "gender": "pt.gender",
            "region": "pt.region",
            "total_cost": "pt.total_cost",
            "avg_total_cost": "ra.avg_total_cost",
        }
        if item not in mapping:
            return None
        outer_columns.append(mapping[item])
    # The outer cohort predicates reference CTE columns.  Project age/gender
    # whenever they are requested or needed to render an explicit predicate.
    for column, label in (("p.age", "age"), ("p.gender", "gender")):
        if (
            label in projection
            or any(predicate.startswith(f"p.{label}") for predicate in patient_predicates)
        ) and column not in base_columns:
            base_columns.append(column)
    group_columns = [column for column in base_columns]
    outer_patient_predicates = [
        predicate.replace("p.", "pt.") for predicate in patient_predicates
    ]
    outer_where = _where(
        [
            "pt.total_cost > ra.avg_total_cost",
            *(
                f"pt.total_cost {operator} {value}"
                for operator, value in total_cost_bounds
            ),
            *outer_patient_predicates,
        ]
    )
    return (
        "WITH patient_totals AS ("
        f"SELECT {', '.join(base_columns)}, SUM(v.cost) AS total_cost "
        "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        f"{base_where} GROUP BY {', '.join(group_columns)}), "
        "region_averages AS ("
        "SELECT region, AVG(total_cost) AS avg_total_cost "
        "FROM patient_totals GROUP BY region) "
        f"SELECT {', '.join(outer_columns)} FROM patient_totals AS pt "
        "JOIN region_averages AS ra ON ra.region = pt.region"
        f"{outer_where} ORDER BY pt.total_cost DESC, pt.patient_id ASC"
    )


def _visit_projection_with_defaults(question: str, *, required: Sequence[str]) -> Optional[list[str]]:
    projection = _projection(question, _VISIT_PROJECTION)
    if projection is None:
        return None
    if not set(required).issubset(projection):
        return None
    return projection


def _rank_metric(question: str) -> Optional[tuple[str, str, str]]:
    if _has(question, r"费用|花费|金额") and _has(question, r"(?:最高|最贵|降序|从高到低|按.*费用)"):
        return "cost", "v.cost", "DESC"
    if _has(question, r"住院天数|住院时长") and _has(question, r"(?:最长|最高|降序|从高到低|按.*住院)"):
        return "los_days", "v.los_days", "DESC"
    if _has(question, r"(?:日期|时间)") and _has(question, r"(?:最早|最近|升序|降序|按.*日期)"):
        direction = "ASC" if _has(question, r"最早|升序|从早到晚") else "DESC"
        return "visit_date", "v.visit_date", direction
    return None


def _visit_select_expression(item: str, alias: str = "v") -> Optional[str]:
    mapping = {
        "visit_id": f"{alias}.visit_id",
        "patient_id": f"{alias}.patient_id",
        "visit_date": f"{alias}.visit_date",
        "department": f"{alias}.department",
        "disease": f"{alias}.disease",
        "drug": f"{alias}.drug_prescribed",
        "exam": f"{alias}.exam",
        "los_days": f"{alias}.los_days",
        "cost": f"{alias}.cost",
        "running_cost": "running_cost",
        "gap_days": "gap_days",
        "previous_visit_date": "previous_visit_date",
        "current_visit_date": "current_visit_date",
        "episode_id": "episode_id",
        "visit_rank": "visit_rank",
    }
    return mapping.get(item)


def _compile_department_topk(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _has(question, r"(?:每个|各个|各|每一)\s*科室") or _limit(question) is None:
        return None
    if not _has(question, r"(?:前|Top|top)\s*[-]?\s*(?:\d+|K)"):
        return None
    ranking = _rank_metric(question)
    if ranking is None:
        return None
    k = _limit(question)
    if k is None:
        return None
    projection = _visit_projection_with_defaults(question, required=("department", "visit_id", "visit_rank"))
    if projection is None:
        return None
    derived_items = {
        "visit_id",
        "patient_id",
        "visit_date",
        "department",
        "disease",
        "drug",
        "exam",
        "los_days",
        "cost",
        "visit_rank",
    }
    # The ranked subquery contains raw visit columns plus its rank.  Window and
    # episode fields belong to other derived shapes and must not be projected
    # here as though they existed on ``visits``.
    if any(item not in derived_items for item in projection):
        return None
    inner_items = list(dict.fromkeys(item for item in projection if item != "visit_rank"))
    if ranking[0] not in inner_items:
        inner_items.append(ranking[0])
    inner_select: list[str] = []
    for item in inner_items:
        expression = _visit_select_expression(item)
        if expression is None:
            return None
        # ``drug`` is the semantic output name, whereas the source schema
        # calls the column ``drug_prescribed``.  Alias it inside the derived
        # table so the outer projection always references a real column.
        if item == "drug":
            expression += " AS drug"
        inner_select.append(expression)
    outer_select = list(projection)
    visit_filters = _visit_filters(question, domains)
    patient_filters = _patient_filters(question, domains, alias="p")
    if patient_filters:
        from_sql = "visits AS v JOIN patients AS p ON p.patient_id = v.patient_id"
        filters = _dedupe(visit_filters + patient_filters)
    else:
        from_sql = "visits AS v"
        filters = visit_filters
    window_order = f"{ranking[1]} {ranking[2]}, v.visit_date ASC, v.visit_id ASC"
    return (
        "SELECT "
        + ", ".join(outer_select)
        + " FROM (SELECT "
        + ", ".join(inner_select)
        + ", ROW_NUMBER() OVER (PARTITION BY v.department ORDER BY "
        + window_order
        + ") AS visit_rank FROM "
        + from_sql
        + _where(filters)
        + ") AS ranked_visits WHERE visit_rank <= "
        + str(k)
        + " ORDER BY department ASC, visit_rank ASC, visit_id ASC"
    )


def _is_running_cost_request(question: str) -> bool:
    """Recognize a date-scoped per-patient running-total request."""

    return bool(
        _has(question, r"(?:累计费用|运行总费用|累计花费)")
        and _has(question, r"(?:按|按照|每(?:个|位)?)\s*患者|患者.*(?:日期|时间)|PARTITION")
        and _has(
            question,
            r"(?:按|按照|依据|根据)\s*(?:日期|时间)|"
            r"日期.*(?:升序|顺序)|时间.*(?:升序|顺序)|"
            r"按患者\s*(?:和|与|及)\s*(?:日期|时间)",
        )
        and bool(_date_filters(question, "v"))
    )


def _compile_running_cost(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _is_running_cost_request(question):
        return None
    projection = _visit_projection_with_defaults(question, required=("patient_id", "visit_date", "cost", "running_cost"))
    if projection is None:
        return None
    running_cost_bounds = _complete_metric_thresholds(question, _RUNNING_COST_BOUND_LABELS)
    if running_cost_bounds is None:
        return None
    if _has_unowned_numeric_metric_threshold(question, _RUNNING_COST_BOUND_LABELS):
        return None
    supported_projection = {
        "visit_id",
        "patient_id",
        "visit_date",
        "department",
        "disease",
        "drug",
        "exam",
        "los_days",
        "cost",
        "running_cost",
    }
    # The running-total shape has only source visit columns plus its derived
    # window value.  Do not expose unrelated derived columns as though the CTE
    # produced them.
    if any(item not in supported_projection for item in projection):
        return None
    inner_items = list(projection)
    if running_cost_bounds and "visit_id" not in inner_items:
        # The deterministic outer ordering needs the source visit identifier
        # even when it was not requested in the result projection.
        inner_items.append("visit_id")
    select_parts: list[str] = []
    for item in inner_items:
        expression = _visit_select_expression(item)
        if expression is None:
            return None
        if item == "running_cost":
            expression = (
                "SUM(v.cost) OVER (PARTITION BY v.patient_id ORDER BY v.visit_date ASC, v.visit_id ASC "
                "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_cost"
            )
        select_parts.append(expression)
    visit_filters = _visit_filters(question, domains, alias="v")
    patient_filters = _patient_filters(question, domains, alias="p")
    if patient_filters:
        from_sql = "visits AS v JOIN patients AS p ON p.patient_id = v.patient_id"
        filters = _dedupe(visit_filters + patient_filters)
    else:
        from_sql = "visits AS v"
        filters = visit_filters
    if running_cost_bounds:
        outer_mapping = {
            "visit_id": "rv.visit_id",
            "patient_id": "rv.patient_id",
            "visit_date": "rv.visit_date",
            "department": "rv.department",
            "disease": "rv.disease",
            "drug": "rv.drug_prescribed",
            "exam": "rv.exam",
            "los_days": "rv.los_days",
            "cost": "rv.cost",
            "running_cost": "rv.running_cost",
        }
        outer_select: list[str] = []
        for item in projection:
            expression = outer_mapping.get(item)
            if expression is None:
                return None
            outer_select.append(expression)
        sql = (
            "WITH running_visits AS (SELECT "
            + ", ".join(select_parts)
            + " FROM "
            + from_sql
            + _where(filters)
            + ") SELECT "
            + ", ".join(outer_select)
            + " FROM running_visits AS rv"
            + _where(
                [
                    f"rv.running_cost {operator} {value}"
                    for operator, value in running_cost_bounds
                ]
            )
            + " ORDER BY rv.patient_id ASC, rv.visit_date ASC, rv.visit_id ASC"
        )
    else:
        sql = (
            f"SELECT {', '.join(select_parts)} FROM {from_sql}{_where(filters)} "
            "ORDER BY v.patient_id ASC, v.visit_date ASC, v.visit_id ASC"
        )
    explicit_limit = _limit(question)
    if explicit_limit is not None and _has(question, r"前|返回|保留|取"):
        sql += f" LIMIT {explicit_limit}"
    return sql


def _gap_threshold(question: str) -> Optional[tuple[str, str]]:
    for operator_re, operator in _OPERATOR_PATTERNS:
        patterns = (
            rf"(?:相隔|间隔|相差|差距)\s*{operator_re}\s*({_NUM})\s*天",
            rf"{operator_re}\s*({_NUM})\s*天\s*(?:的)?(?:相邻)?(?:就诊)?(?:间隔|相隔|差距)?",
            rf"(?:连续|相邻).{{0,8}}(?:就诊|访问).{{0,8}}(?:间隔|相隔)\s*{operator_re}\s*({_NUM})\s*天",
        )
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                value = _literal(match.group(1))
                if value is not None:
                    return operator, value
    return None


def _compile_lag_gap(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if _has(question, r"episode|事件分段|划分为新|新的就诊事件|分段"):
        return None
    if not _has(question, r"患者|病人") or not _has(question, r"(?:相邻|连续|前一次|上一次|两次就诊)"):
        return None
    if not _has(question, r"(?:间隔|相隔|相差|gap)"):
        return None
    gap = _gap_threshold(question)
    if gap is None:
        return None
    projection = _projection(
        question,
        _VISIT_PROJECTION,
    )
    if projection is None or not {"patient_id", "gap_days"}.issubset(projection):
        return None
    if "前后日期" in question:
        projection = [item for item in projection if item != "previous_visit_date"]
        if "visit_date" not in projection:
            return None
        projection.insert(projection.index("visit_date"), "previous_visit_date")
    inner = ["v.patient_id", "v.visit_id", "v.visit_date"]
    inner += ["v." + item for item in ("department", "disease", "cost") if item in projection]
    visit_filters = _visit_filters(question, domains, alias="v")
    patient_filters = _patient_filters(question, domains, alias="p")
    if patient_filters:
        from_sql = "visits AS v JOIN patients AS p ON p.patient_id = v.patient_id"
        filters = _dedupe(visit_filters + patient_filters)
    else:
        from_sql = "visits AS v"
        filters = visit_filters
    condition = f"julianday(visit_date) - julianday(previous_visit_date) {gap[0]} {gap[1]}"
    outer: list[str] = []
    for item in projection:
        mapping = {
            "patient_id": "patient_id",
            "visit_id": "visit_id",
            "visit_date": "visit_date",
            "department": "department",
            "disease": "disease",
            "cost": "cost",
            "previous_visit_date": "previous_visit_date",
            "current_visit_date": "visit_date AS current_visit_date",
            "gap_days": "CAST(julianday(visit_date) - julianday(previous_visit_date) AS INTEGER) AS gap_days",
        }
        if item not in mapping:
            return None
        outer.append(mapping[item])
    return (
        "WITH ordered_visits AS (SELECT "
        + ", ".join(inner)
        + ", LAG(v.visit_date) OVER (PARTITION BY v.patient_id ORDER BY v.visit_date ASC, v.visit_id ASC) AS previous_visit_date "
        + "FROM "
        + from_sql
        + _where(filters)
        + ") SELECT "
        + ", ".join(outer)
        + " FROM ordered_visits WHERE previous_visit_date IS NOT NULL AND "
        + condition
        + (
            " ORDER BY gap_days DESC, patient_id ASC, visit_date ASC"
            if re.search(r"按(?:间隔|相隔).*(?:降序|倒排|从高到低)", question)
            else " ORDER BY patient_id ASC, visit_date ASC, visit_id ASC"
        )
    )


def _monthly_metrics(question: str) -> list[str]:
    result: list[str] = []
    if _has(question, r"就诊(?:次数|量)|记录(?:数|量)|访问次数"):
        result.append("visit_count")
    if _has(question, r"(?:总|累计|合计)\s*费用|费用总额"):
        result.append("total_cost")
    if _has(question, r"平均费用|费用均值"):
        result.append("avg_cost")
    return result


def _compile_monthly_yoy(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _has(question, r"(?:按月|每月|月度|月份|逐月)") or not _has(question, r"(?:同比|上年同期|12\s*个月前|12\s*月前|LAG)"):
        return None
    if not _has(question, r"(?:变化|变动|增长|减少|差值|同比)"):
        return None
    if _has_unsupported_non_date_filters(question, domains):
        return None
    metrics = _monthly_metrics(question)
    if not metrics:
        return None
    date_window = dates(question)
    if len(date_window) < 2:
        return None
    start_date, end_date = date_window[:2]
    projection = _projection(
        question,
        _GROUP_PROJECTION
        + (
            ("previous_year_count", r"上年同期|去年同期|前一年|12\s*个月前"),
            ("yoy_change", r"同比变化|同比变动|同比差值|变化量|变化值|增长量|增减"),
        ),
    )
    if projection is None or "visit_month" not in projection or "yoy_change" not in projection:
        return None
    for metric in metrics:
        if metric not in projection:
            return None
    metric_sql = {
        "visit_count": ("COUNT(*)", "visit_count"),
        "total_cost": ("SUM(v.cost)", "total_cost"),
        "avg_cost": ("AVG(v.cost)", "avg_cost"),
    }
    monthly_select = ["substr(v.visit_date, 1, 7) AS visit_month"]
    for metric in metrics:
        expression, alias = metric_sql[metric]
        monthly_select.append(f"{expression} AS {alias}")
    outer: list[str] = []
    for item in projection:
        if item == "visit_month":
            outer.append("current_month.visit_month")
        elif item in metrics:
            outer.append(f"current_month.{item}")
        elif item == "previous_year_count":
            outer.append(f"prior_month.{metrics[0]} AS previous_{metrics[0]}")
        elif item == "yoy_change":
            outer.append(
                f"current_month.{metrics[0]} - prior_month.{metrics[0]} AS yoy_change"
            )
        else:
            return None
    # Aggregate a calendar-year baseline before filtering the requested output
    # months.  A self-join on the previous calendar month remains correct when
    # the source has gaps, unlike ``LAG(..., 12)`` over observed rows.
    baseline_filters = [
        f"v.visit_date >= date('{start_date}', 'start of month', '-1 year')",
        f"v.visit_date <= '{end_date}'",
    ]
    return (
        "WITH monthly AS (SELECT "
        + ", ".join(monthly_select)
        + " FROM visits AS v"
        + _where(baseline_filters)
        + " GROUP BY substr(v.visit_date, 1, 7)) SELECT "
        + ", ".join(outer)
        + " FROM monthly AS current_month JOIN monthly AS prior_month ON "
        + "prior_month.visit_month = strftime('%Y-%m', "
        + "date(current_month.visit_month || '-01', '-1 year'))"
        + f" WHERE current_month.visit_month BETWEEN substr('{start_date}', 1, 7) "
        + f"AND substr('{end_date}', 1, 7) ORDER BY current_month.visit_month ASC"
    )


def _visit_count_bounds(question: str) -> list[tuple[str, str]]:
    return _thresholds(question, ("就诊次数", "就诊量", "次就诊", "次访问"))


def _compile_first_last(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _has(question, _PATIENT) or not _has(question, r"(?:首次|第一次|最早).*(?:末次|最后|最近)|(?:末次|最后|最近).*(?:首次|第一次|最早)"):
        return None
    if not _has(question, r"(?:疾病|诊断|科室|日期|时间|费用)"):
        return None
    visit_count_bounds = _visit_count_bounds(question)
    if not visit_count_bounds:
        return None
    projection = _projection(
        question,
        _PATIENT_PROJECTION
        + (
            ("visit_count", r"就诊(?:次数|量)|记录数"),
            ("first_visit_date", r"首次(?:就诊)?日期|最早(?:就诊)?日期|首次日期"),
            ("first_department", r"首次.*科室|最早.*科室"),
            ("first_disease", r"首次.*疾病|最早.*疾病|首次.*诊断"),
            ("first_cost", r"首次.*(?:费用|花费)"),
            ("last_visit_date", r"末次(?:就诊)?日期|最后(?:就诊)?日期|最近(?:一次)?日期|末次日期"),
            ("last_department", r"末次.*科室|最后.*科室|最近.*科室"),
            ("last_disease", r"末次.*疾病|最后.*疾病|最近.*疾病|末次.*诊断"),
            ("last_cost", r"末次.*(?:费用|花费)|最后.*(?:费用|花费)"),
        ),
    )
    if projection is None or "patient_id" not in projection or not any(item.startswith("first_") for item in projection) or not any(item.startswith("last_") for item in projection):
        return None
    base_patient = ["p.patient_id", "p.name", "p.age", "p.gender", "p.region"]
    base_group = ["p.patient_id", "p.name", "p.age", "p.gender", "p.region"]
    selected_output: list[str] = []
    for item in projection:
        mapping = {
            "patient_id": "pvc.patient_id",
            "name": "pvc.name",
            "age": "pvc.age",
            "gender": "pvc.gender",
            "region": "pvc.region",
            "visit_count": "pvc.visit_count",
            "first_visit_date": "f.visit_date AS first_visit_date",
            "first_department": "f.department AS first_department",
            "first_disease": "f.disease AS first_disease",
            "first_cost": "f.cost AS first_cost",
            "last_visit_date": "l.visit_date AS last_visit_date",
            "last_department": "l.department AS last_department",
            "last_disease": "l.disease AS last_disease",
            "last_cost": "l.cost AS last_cost",
        }
        if item not in mapping:
            return None
        selected_output.append(mapping[item])
    visit_filters = _visit_filters(question, domains, alias="v")
    patient_filters = _patient_filters(question, domains, alias="p")
    # The minimum count must describe the same population as first/last rows.
    # Patient eligibility belongs to the patient CTE; the final join then
    # retains all qualifying visits for exactly that cohort.
    return (
        "WITH patient_visit_counts AS (SELECT "
        + ", ".join(base_patient)
        + ", COUNT(v.visit_id) AS visit_count FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        + _where(_dedupe(visit_filters + patient_filters))
        + " GROUP BY "
        + ", ".join(base_group)
        + "), ranked_visits AS (SELECT v.*, ROW_NUMBER() OVER (PARTITION BY v.patient_id ORDER BY v.visit_date ASC, v.visit_id ASC) AS first_rank, ROW_NUMBER() OVER (PARTITION BY v.patient_id ORDER BY v.visit_date DESC, v.visit_id DESC) AS last_rank FROM visits AS v"
        + _where(visit_filters)
        + ") SELECT "
        + ", ".join(selected_output)
        + " FROM patient_visit_counts AS pvc JOIN ranked_visits AS f ON f.patient_id = pvc.patient_id AND f.first_rank = 1 JOIN ranked_visits AS l ON l.patient_id = pvc.patient_id AND l.last_rank = 1"
        + _where(
            [
                f"pvc.visit_count {operator} {value}"
                for operator, value in visit_count_bounds
            ]
        )
        + " ORDER BY pvc.patient_id ASC"
    )


def _is_required_departments_request(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    return bool(
        _has(question, _PATIENT)
        and _has(question, r"(?:全部|所有|每个|每一|都|均|所列|列出的)")
        and _has(question, r"(?:就诊|去过|到过|覆盖)")
        and len(domain_values(question, domains.get("department", ()))) >= 2
        and _has(question, r"(?:都|全部|所有|每个|列出的|所列|要求|覆盖)")
    )


def _has_unsupported_required_department_visit_constraints(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Identify visit constraints outside relational-division semantics."""

    return bool(
        dates(question)
        or _has_numeric_visit_or_aggregate_constraint(question)
        or any(
            domain_values(question, domains.get(key, ()))
            for key in ("disease", "drug", "exam")
        )
    )


def _has_negative_required_departments_intent(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Recognize negative universal department requests we cannot prove."""

    if not (
        _has(question, _PATIENT)
        and _has(question, r"(?:就诊|去过|到过|覆盖)")
        and len(domain_values(question, domains.get("department", ()))) >= 2
    ):
        return False
    universal = r"(?:都|全部|所有|每个|每一|均|同时|分别|各自|单独|既.{0,12}又)"
    visit = r"(?:就诊|去过|到过|覆盖)"
    lexical_negative = r"(?:从来没有|从未|未曾|不曾|没有|没(?:有)?|并非|不是)"
    # Bare ``不`` and ``未`` are negative only when immediately scoped to a
    # coverage phrase.  Treating either as a global cue would misread valid
    # patient filters such as ``年龄不低于 40 岁`` as negated division.
    direct_negative = rf"(?:不|未)\s*{universal}\s*{visit}|{universal}\s*(?:不|未)\s*{visit}"
    # Also cover action-before-universal wording such as ``未覆盖所有内科和
    # 外科``.  The narrow action adjacency avoids misreading a positive lower
    # age bound like ``年龄不低于 40 岁`` as a negative division request.
    action_negative = rf"(?:{lexical_negative}|不|未)"
    action_before_universal = rf"{action_negative}\s*{visit}.{{0,16}}{universal}"
    return bool(
        re.search(
            rf"{lexical_negative}.{{0,12}}{universal}.{{0,16}}{visit}|"
            rf"{universal}.{{0,12}}{lexical_negative}.{{0,12}}{visit}|"
            rf"{direct_negative}|{action_before_universal}",
            question,
        )
    )


def _compile_required_departments(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _is_required_departments_request(question, domains):
        return None
    if _has_negative_required_departments_intent(question, domains):
        return None
    departments = domain_values(question, domains.get("department", ()))
    # Department membership is this compiler's relational-division contract.
    # A date or another grounded visit dimension would need an explicit rule
    # for whether it constrains every required department or merely any visit.
    # Abstain rather than silently discard it.
    if _has_unsupported_required_department_visit_constraints(question, domains):
        return None
    projection = _projection(question, _PATIENT_PROJECTION)
    if projection is None or "patient_id" not in projection:
        return None
    selected = ["p." + item for item in projection]
    values_sql = ", ".join(f"('{esc(value)}')" for value in departments)
    patient_filters = _patient_filters(question, domains, alias="p")
    division = (
        "NOT EXISTS (SELECT 1 FROM required_departments AS r WHERE NOT EXISTS "
        "(SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id "
        "AND v.department = r.department))"
    )
    return (
        "WITH required_departments(department) AS (VALUES "
        + values_sql
        + ") SELECT "
        + ", ".join(selected)
        + " FROM patients AS p WHERE "
        + " AND ".join(patient_filters + [division])
        + " ORDER BY p.patient_id ASC"
    )


def _is_ntile_quartiles_request(question: str) -> bool:
    """Recognize a patient-total quartile request before SQL lowering."""

    return bool(
        _has(question, _PATIENT)
        and _has(question, r"(?:四分位|四等分|quartile|NTILE)")
        and (any(label in question for label in _TOTAL_COST_BOUND_LABELS) or _has(question, r"消费|花费"))
    )


def _compile_ntile_quartiles(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _is_ntile_quartiles_request(question):
        return None
    projection = _projection(
        question,
        _PATIENT_PROJECTION
        + (
            ("total_cost", _PATIENT_METRIC_PROJECTION[1][1]),
            ("spending_quartile", r"四分位|分位组|quartile|消费组"),
        ),
    )
    if projection is None or not {"patient_id", "total_cost", "spending_quartile"}.issubset(projection):
        return None
    total_cost_bounds = _complete_metric_thresholds(question, _TOTAL_COST_BOUND_LABELS)
    if total_cost_bounds is None:
        return None
    if _has_unowned_numeric_metric_threshold(question, _TOTAL_COST_BOUND_LABELS):
        return None
    direction = "DESC" if _has(question, r"从高到低|高消费优先|费用最高|降序") else "ASC"
    patient_columns = ["p.patient_id", "p.name", "p.age", "p.gender", "p.region"]
    output: list[str] = []
    for item in projection:
        mapping = {
            "patient_id": "patient_id",
            "name": "name",
            "age": "age",
            "gender": "gender",
            "region": "region",
            "total_cost": "total_cost",
            "spending_quartile": "NTILE(4) OVER (ORDER BY total_cost " + direction + ", patient_id ASC) AS spending_quartile",
        }
        if item not in mapping:
            return None
        output.append(mapping[item])
    visit_filters = _visit_filters(question, domains, alias="v")
    patient_filters = _patient_filters(question, domains, alias="p")
    filters = _dedupe(visit_filters + patient_filters)
    ranked_source = "patient_totals"
    eligibility_cte = ""
    if total_cost_bounds:
        ranked_source = "eligible_patient_totals"
        eligibility_cte = (
            ", eligible_patient_totals AS (SELECT * FROM patient_totals"
            + _where(
                [
                    f"total_cost {operator} {value}"
                    for operator, value in total_cost_bounds
                ]
            )
            + ")"
        )
    return (
        "WITH patient_totals AS (SELECT "
        + ", ".join(patient_columns)
        + ", SUM(v.cost) AS total_cost FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        + _where(filters)
        + " GROUP BY p.patient_id, p.name, p.age, p.gender, p.region)"
        + eligibility_cte
        + " SELECT "
        + ", ".join(output)
        + " FROM "
        + ranked_source
        + " ORDER BY spending_quartile ASC, patient_id ASC"
    )


def _compile_episode_segmentation(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _has(question, _PATIENT) or not _has(question, r"(?:episode|事件|分段|分组|划分)"):
        return None
    if not _has(question, r"(?:间隔|相隔|相差|gap)"):
        return None
    gap = _gap_threshold(question)
    if gap is None:
        return None
    projection = _projection(question, _VISIT_PROJECTION)
    if projection is None or not {"patient_id", "visit_date", "episode_id"}.issubset(projection):
        return None
    select_map = {
        "patient_id": "patient_id",
        "visit_id": "visit_id",
        "visit_date": "visit_date",
        "department": "department",
        "disease": "disease",
        "drug": "drug_prescribed",
        "exam": "exam",
        "cost": "cost",
        "episode_id": "episode_id",
        "previous_visit_date": "previous_visit_date",
    }
    selected: list[str] = []
    for item in projection:
        if item not in select_map:
            return None
        selected.append(select_map[item])
    visit_filters = _visit_filters(question, domains, alias="v")
    patient_filters = _patient_filters(question, domains, alias="p")
    if patient_filters:
        from_sql = "visits AS v JOIN patients AS p ON p.patient_id = v.patient_id"
        filters = _dedupe(visit_filters + patient_filters)
    else:
        from_sql = "visits AS v"
        filters = visit_filters
    gap_expression = "julianday(visit_date) - julianday(previous_visit_date)"
    new_episode = f"CASE WHEN previous_visit_date IS NULL OR {gap_expression} {gap[0]} {gap[1]} THEN 1 ELSE 0 END AS new_episode"
    return (
        "WITH ordered_visits AS (SELECT v.patient_id, v.visit_id, v.visit_date, v.department, v.disease, v.drug_prescribed, v.exam, v.cost, LAG(v.visit_date) OVER (PARTITION BY v.patient_id ORDER BY v.visit_date ASC, v.visit_id ASC) AS previous_visit_date FROM "
        + from_sql
        + _where(filters)
        + "), marked_visits AS (SELECT ov.*, "
        + new_episode
        + " FROM ordered_visits AS ov), segmented_visits AS (SELECT mv.*, SUM(mv.new_episode) OVER (PARTITION BY mv.patient_id ORDER BY mv.visit_date ASC, mv.visit_id ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS episode_id FROM marked_visits AS mv) SELECT "
        + ", ".join(selected)
        + " FROM segmented_visits ORDER BY patient_id ASC, visit_date ASC, visit_id ASC"
    )


def _has_unrenderable_region_topk_visit_count(question: str) -> bool:
    """Detect a visit-count predicate that cannot be rendered completely."""

    label_re = "(?:" + "|".join(
        re.escape(item)
        for item in sorted(_REGION_TOPK_VISIT_COUNT_LABELS, key=len, reverse=True)
    ) + ")"
    range_start = re.compile(
        rf"{label_re}\s*(?:(?:在|介于|范围(?:为|是)?|从|为|是|=)\s*)?"
        rf"{_NUM}\s*{_RANGE_UNIT}?\s*{_RANGE_SEPARATOR}",
        re.I,
    )
    for match in range_start.finditer(question):
        # Check each attempted interval independently.  A separate valid
        # threshold must not make a malformed interval safe to discard.
        if re.match(rf"\s*{_NUM}\s*{_RANGE_UNIT}?(?:之间)?", question[match.end():]) is None:
            return True
    return bool(
        _threshold_intent(question, _REGION_TOPK_VISIT_COUNT_LABELS)
        and not _thresholds(question, _REGION_TOPK_VISIT_COUNT_LABELS)
    )


def _is_region_topk_request(question: str) -> bool:
    """Recognize the structural per-region patient Top-K signature."""

    return bool(
        _has(question, r"(?:每个|各个|各|每一)\s*(?:地区|区域|城市)")
        and _limit(question) is not None
        and _has(question, r"(?:前|Top|top)\s*[-]?\s*(?:\d+|K)")
        and _has(question, r"(?:累计费用|总费用|费用总额|患者总额)")
    )


def _eligibility_predicates(question: str) -> tuple[list[str], bool]:
    predicates: list[str] = []
    # Eligibility predicates run against the ``patient_totals`` CTE, whose
    # projected age column is named simply ``age`` rather than ``age.age``.
    ages = _explicit_age_predicates(question, alias="")
    predicates.extend(ages)
    visit_count_bounds = _thresholds(question, _REGION_TOPK_VISIT_COUNT_LABELS)
    predicates.extend(
        f"visit_count {operator} {value}"
        for operator, value in visit_count_bounds
    )
    total_cost_bounds = _thresholds(
        question,
        ("总费用", "累计费用", "费用总额", "患者累计费用"),
    )
    predicates.extend(
        f"total_cost {operator} {value}"
        for operator, value in total_cost_bounds
    )
    department_count_bounds = _thresholds(
        question,
        ("科室数", "科室数量", "不同科室数"),
    )
    predicates.extend(
        f"department_count {operator} {value}"
        for operator, value in department_count_bounds
    )
    return _dedupe(predicates), bool(predicates)


def _compile_region_topk(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _is_region_topk_request(question):
        return None
    if _has_unrenderable_region_topk_visit_count(question):
        return None
    k = _limit(question)
    if k is None:
        return None
    eligibility, has_eligibility = _eligibility_predicates(question)
    if not has_eligibility:
        return None
    projection = _projection(
        question,
        _PATIENT_PROJECTION
        + (
            ("visit_count", _PATIENT_METRIC_PROJECTION[0][1]),
            ("total_cost", _PATIENT_METRIC_PROJECTION[1][1]),
            ("department_count", _PATIENT_METRIC_PROJECTION[2][1]),
            ("region_rank", r"排名|名次"),
        ),
    )
    if projection is None or not {"region", "patient_id", "total_cost", "region_rank"}.issubset(projection):
        return None
    base_columns = ["p.patient_id", "p.name", "p.age", "p.gender", "p.region"]
    metrics = [
        "COUNT(v.visit_id) AS visit_count",
        "SUM(v.cost) AS total_cost",
        "COUNT(DISTINCT v.department) AS department_count",
    ]
    output_map = {
        "patient_id": "patient_id",
        "name": "name",
        "age": "age",
        "gender": "gender",
        "region": "region",
        "visit_count": "visit_count",
        "total_cost": "total_cost",
        "department_count": "department_count",
        "region_rank": "region_rank",
    }
    selected = [output_map[item] for item in projection if item in output_map]
    if len(selected) != len(projection):
        return None
    # Eligibility and ranking operate on the explicitly requested cohort.  A
    # positive region restriction is compatible with a per-region window and
    # must remain in the CTE rather than being silently stripped.
    visit_filters = _visit_filters(question, domains, alias="v")
    patient_filters = _patient_filters(question, domains, alias="p")
    base_where = _where(_dedupe(visit_filters + patient_filters))
    eligible_where = _where(eligibility)
    return (
        "WITH patient_totals AS (SELECT "
        + ", ".join(base_columns + metrics)
        + " FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
        + base_where
        + " GROUP BY p.patient_id, p.name, p.age, p.gender, p.region), eligible_patients AS (SELECT * FROM patient_totals"
        + eligible_where
        + "), ranked_patients AS (SELECT eligible_patients.*, ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_cost DESC, patient_id ASC) AS region_rank FROM eligible_patients) SELECT "
        + ", ".join(selected)
        + " FROM ranked_patients WHERE region_rank <= "
        + str(k)
        + " ORDER BY region ASC, region_rank ASC, patient_id ASC"
    )


# ---------------------------------------------------------------------------
# Cohort compilers


def _cohort_anchor(question: str) -> bool:
    """Require an explicit first-visit month, not a bare cohort noun."""

    return _has(question, r"(?:首诊|首次就诊|第一次就诊)\s*(?:月|月份)")


def _cohort_intervals(question: str) -> list[tuple[int, str, str]]:
    """Return explicit positive day/month intervals as (value, unit, text)."""

    found: list[tuple[int, str, str, int]] = []
    patterns = (
        (r"(?<!\d)(\d+|[一二两三四五六七八九十])\s*(天|日)(?:内|后|时|留存)?", "day"),
        (r"第\s*(\d+|[一二两三四五六七八九十])\s*(?:个)?月", "month"),
        (r"(?<!\d)(\d+|[一二两三四五六七八九十])\s*个月(?:内|后|留存)?", "month"),
    )
    for pattern, unit in patterns:
        for match in re.finditer(pattern, question):
            value = integer(match.group(1))
            if value is None or value <= 0:
                continue
            found.append((value, unit, match.group(0), match.start()))
    unique: list[tuple[int, str, str]] = []
    for value, unit, text, _ in sorted(found, key=lambda item: item[3]):
        token = (value, unit)
        if not any(existing[0] == token[0] and existing[1] == token[1] for existing in unique):
            unique.append((value, unit, text))
    return unique


def _cohort_date_where(question: str, alias: str = "v") -> str:
    filters = _date_filters(question, alias)
    return _where(filters)


def _compile_cohort_funnel(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _has(question, r"漏斗|funnel") or not _cohort_anchor(question):
        return None
    if _has_unsupported_non_date_filters(question, domains):
        return None
    intervals = _cohort_intervals(question)
    if len(intervals) < 2:
        return None
    if not _has(question, r"(?:再次|复诊|复访|随访|就诊|阶段|人数|统计|返回|计算)"):
        return None
    stage_aliases = ["cohort_patients"] + [f"within_{value}_{'days' if unit == 'day' else 'months'}_patients" for value, unit, _ in intervals]
    stage_expressions = ["COUNT(DISTINCT c.patient_id) AS cohort_patients"]
    for (value, unit, _), alias in zip(intervals, stage_aliases[1:]):
        modifier = f"+{value} {'day' if unit == 'day' else 'month'}"
        stage_expressions.append(
            f"COUNT(DISTINCT CASE WHEN v.visit_date > c.first_visit_date AND v.visit_date <= date(c.first_visit_date, '{modifier}') THEN c.patient_id END) AS {alias}"
        )
    date_where = _cohort_date_where(question, "v")
    return (
        "WITH first_visits AS (SELECT v.patient_id, MIN(v.visit_date) AS first_visit_date FROM visits AS v"
        + date_where
        + " GROUP BY v.patient_id), cohorts AS (SELECT patient_id, first_visit_date, substr(first_visit_date, 1, 7) AS cohort_month FROM first_visits), cohort_funnel AS (SELECT c.cohort_month, "
        + ", ".join(stage_expressions)
        + " FROM cohorts AS c LEFT JOIN visits AS v ON v.patient_id = c.patient_id GROUP BY c.cohort_month) SELECT cohort_month, "
        + ", ".join(stage_aliases)
        + " FROM cohort_funnel ORDER BY cohort_month ASC"
    )


def _compile_cohort_retention(question: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _has(question, r"留存|retention") or not _cohort_anchor(question):
        return None
    if _has_unsupported_non_date_filters(question, domains):
        return None
    intervals = _cohort_intervals(question)
    if len(intervals) < 2:
        return None
    if not _has(question, r"(?:月|天).*(?:留存|retention)|留存.*(?:月|天)"):
        return None
    rate = _has(question, r"留存率|比例|百分比|百分率")
    columns: list[str] = []
    expressions: list[str] = []
    for value, unit, _ in intervals:
        if unit == "month":
            start_modifier = f"+{value} month"
            end_modifier = f"+{value + 1} month"
            alias = f"month_{value}_retention"
        else:
            start_modifier = f"+{value} day"
            end_modifier = f"+{value + 1} day"
            alias = f"day_{value}_retention"
        condition = (
            f"v.visit_date >= date(c.first_visit_date, '{start_modifier}') AND "
            f"v.visit_date < date(c.first_visit_date, '{end_modifier}')"
        )
        if rate:
            expressions.append(
                f"100.0 * COUNT(DISTINCT CASE WHEN {condition} THEN c.patient_id END) / COUNT(DISTINCT c.patient_id) AS {alias}_rate"
            )
            columns.append(f"{alias}_rate")
        else:
            expressions.append(
                f"COUNT(DISTINCT CASE WHEN {condition} THEN c.patient_id END) AS {alias}_patients"
            )
            columns.append(f"{alias}_patients")
    date_where = _cohort_date_where(question, "v")
    return (
        "WITH first_visits AS (SELECT v.patient_id, MIN(v.visit_date) AS first_visit_date FROM visits AS v"
        + date_where
        + " GROUP BY v.patient_id), cohorts AS (SELECT patient_id, first_visit_date, substr(first_visit_date, 1, 7) AS cohort_month FROM first_visits), cohort_retention AS (SELECT c.cohort_month, COUNT(DISTINCT c.patient_id) AS cohort_size, "
        + ", ".join(expressions)
        + " FROM cohorts AS c LEFT JOIN visits AS v ON v.patient_id = c.patient_id GROUP BY c.cohort_month) SELECT cohort_month, cohort_size, "
        + ", ".join(columns)
        + " FROM cohort_retention ORDER BY cohort_month ASC"
    )


def _is_date_only_analytical_request(question: str) -> bool:
    """Recognize the monthly analytical signature with documented date scope.

    Temporal cohort ownership is handled by ``TemporalCohortIntent`` before
    this guard.  Keeping bare cohort nouns out of this generic predicate lets
    their designated semantic-family routes remain reachable.
    """

    return bool(
        _has(question, r"(?:按月|每月|月度|月份|逐月)")
        and _has(question, r"(?:同比|上年同期|12\s*个月前|12\s*月前|LAG)")
        and _has(question, r"(?:变化|变动|增长|减少|差值|同比)")
    )


def _requires_terminal_advanced_abstention(
    question: str,
    domains: Mapping[str, Sequence[str]],
    *,
    age_average_intent: Optional[AgeAverageIntent] = None,
) -> bool:
    """Prevent fallback compilers from reinterpreting known abstentions."""

    if _has_negative_required_departments_intent(question, domains):
        return True
    if (
        _is_required_departments_request(question, domains)
        and _has_unsupported_required_department_visit_constraints(question, domains)
    ):
        return True
    if (
        _is_patient_exists_request(question, domains)
        and _has_unsupported_patient_exists_visit_constraints(question)
    ):
        return True
    peer_intent = age_average_intent or _age_average_intent(question, domains)
    if peer_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN:
        return True
    if (
        _is_region_topk_request(question)
        and _has_unrenderable_region_topk_visit_count(question)
    ):
        return True
    return bool(
        _is_date_only_analytical_request(question)
        and _has_unsupported_non_date_filters(question, domains)
    )


# ---------------------------------------------------------------------------
# Public dispatcher


def compile_advanced_medical_result(
    question: str,
    domains: Mapping[str, Sequence[str]],
    *,
    age_average_intent: Optional[AgeAverageIntent] = None,
    temporal_cohort_intent: Optional[TemporalCohortIntent] = None,
    obligations: Optional["SemanticObligations"] = None,
) -> CompileResult:
    """Compile an advanced query while preserving exclusive abstentions.

    The public ``Optional[str]`` API intentionally cannot distinguish an
    unrelated question from a recognized request that this layer owns but
    cannot lower.  Internal dispatch must retain that distinction so a later
    compiler cannot silently discard an owned qualifier.  Callers that already
    classified an exclusive intent pass the same immutable record here rather
    than asking this module to classify the question again.
    """

    if not isinstance(question, str) or not question.strip() or not isinstance(domains, Mapping):
        return CompileResult(CompileDisposition.NOT_APPLICABLE)
    q = question.strip()
    if (
        classify_region_filter_intent(q, domains).disposition
        is RegionFilterDisposition.TERMINAL_ABSTAIN
    ):
        return CompileResult(CompileDisposition.BLOCKED)
    parsed_age = (
        age_average_intent
        if age_average_intent is not None
        else _age_average_intent(q, domains)
    )
    # A recognized age-to-average comparison has an exclusive owner.  In
    # particular, a correlated peer comparison must not reach a generic CTE or
    # legacy family compiler if its full qualifier set cannot be lowered.
    if parsed_age.disposition is AgeAverageDisposition.ADVANCED_PEER:
        sql = _compile_age_peer_average(q, domains, intent=parsed_age)
        return (
            CompileResult(
                CompileDisposition.SQL,
                sql,
                contract_self_validated=True,
            )
            if sql
            else CompileResult(CompileDisposition.BLOCKED)
        )
    if parsed_age.disposition is not AgeAverageDisposition.NOT_APPLICABLE:
        # The explicit global form is owned by the explicit compiler; a direct
        # advanced caller must not reinterpret it as an unrelated query.
        return CompileResult(CompileDisposition.BLOCKED)

    parsed_temporal = (
        temporal_cohort_intent
        if temporal_cohort_intent is not None
        else _temporal_cohort_intent(q, domains)
    )
    # A recognized cohort request also has one exclusive owner.  Family-owned
    # forms go through a narrow dispatcher so a failed retention/evidence
    # lowerer cannot fall through to a different family, KG, or row compiler.
    if parsed_temporal.requires_family_dispatch:
        from dki.analysis.nl2sql_semantic_families import (
            compile_temporal_cohort_family_sql,
        )

        sql = compile_temporal_cohort_family_sql(
            q,
            domains,
            intent=parsed_temporal,
            obligations=obligations,
        )
        return (
            CompileResult(
                CompileDisposition.SQL,
                sql,
                contract_self_validated=True,
            )
            if sql
            else CompileResult(CompileDisposition.BLOCKED)
        )
    if parsed_temporal.disposition is TemporalCohortDisposition.ADVANCED_DATE_ONLY:
        if parsed_temporal.kind is TemporalCohortKind.FUNNEL:
            sql = _compile_cohort_funnel(q, domains)
        elif parsed_temporal.kind is TemporalCohortKind.RETENTION:
            sql = _compile_cohort_retention(q, domains)
        else:
            sql = None
        return (
            CompileResult(
                CompileDisposition.SQL,
                sql,
                contract_self_validated=True,
            )
            if sql
            else CompileResult(CompileDisposition.BLOCKED)
        )
    if parsed_temporal.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN:
        return CompileResult(CompileDisposition.BLOCKED)
    if parsed_temporal.disposition is not TemporalCohortDisposition.NOT_APPLICABLE:
        # A new shared exclusive disposition cannot safely reach broad advanced
        # compilers until this dispatcher has a designated lowerer for it.
        return CompileResult(CompileDisposition.BLOCKED)

    # Individual compilers use ``None`` for both an unrelated question and an
    # intentional abstention.  These distinctive unsupported shapes must end
    # dispatch here so neither another advanced compiler nor the legacy family
    # fallback can reinterpret them while dropping a qualifier.
    if _requires_terminal_advanced_abstention(
        q,
        domains,
        age_average_intent=parsed_age,
    ):
        return CompileResult(CompileDisposition.BLOCKED)
    # These compact public-development shapes have exact dedicated lowerers.
    # Once recognized, a rejected candidate must not fall through to a broad
    # simple rule that can silently drop the group, threshold, or gender slot.
    if _implicit_grouped_having_shape(q) is not None:
        sql = _compile_grouped_having(q, domains)
        return (
                CompileResult(
                    CompileDisposition.SQL,
                    sql,
                    fallback_fenced=True,
                    contract_self_validated=True,
                )
            if sql
            else CompileResult(CompileDisposition.BLOCKED)
        )
    if _is_department_gender_visit_count_request(q):
        sql = _compile_department_gender_visit_count(q, domains)
        return (
                CompileResult(
                    CompileDisposition.SQL,
                    sql,
                    fallback_fenced=True,
                    contract_self_validated=True,
                )
            if sql
            else CompileResult(CompileDisposition.BLOCKED)
        )
    # These shapes own their aggregate or window threshold stage.  Once one is
    # recognized, a declined lowerer is terminal: a later generic compiler
    # cannot reinterpret the question while losing its derived-metric bound.
    from dki.analysis.nl2sql_semantic_families import (
        _compile_ntile_quartiles as _compile_ntile_quartiles_family,
        _compile_patient_total_peer_average as _compile_patient_total_peer_average_family,
        _compile_running_cost as _compile_running_cost_family,
    )

    exclusive_metric_compilers = (
        (
            _is_patient_total_peer_average_request,
            _compile_patient_total_peer_average,
            _compile_patient_total_peer_average_family,
        ),
        (_is_running_cost_request, _compile_running_cost, _compile_running_cost_family),
        (_is_ntile_quartiles_request, _compile_ntile_quartiles, _compile_ntile_quartiles_family),
    )
    for recognizes, compiler, compatible_family in exclusive_metric_compilers:
        if recognizes(q):
            sql = compiler(q, domains) or compatible_family(q, domains)
            return (
                CompileResult(
                    CompileDisposition.SQL,
                    sql,
                    contract_self_validated=True,
                )
                if sql
                else CompileResult(CompileDisposition.BLOCKED)
            )
    compilers = (
        _compile_cohort_funnel,
        _compile_cohort_retention,
        _compile_required_departments,
        _compile_patient_rollup,
        _compile_patient_exists,
        _compile_episode_segmentation,
        _compile_lag_gap,
        _compile_department_topk,
        _compile_monthly_yoy,
        _compile_first_last,
        _compile_region_topk,
        _compile_grouped_having,
    )
    # These compilers retain the parsed output projection.  Let them run before
    # broad legacy families, which remain a fallback for disjoint signatures or
    # families that can faithfully render the complete answer shape.
    for compiler in compilers:
        sql = compiler(q, domains)
        if sql:
            return CompileResult(
                CompileDisposition.SQL,
                sql,
                contract_self_validated=True,
            )
    from dki.analysis.nl2sql_semantic_families import compile_semantic_family_sql

    sql = compile_semantic_family_sql(
        q,
        domains,
        age_average_intent=parsed_age,
        temporal_cohort_intent=parsed_temporal,
        obligations=obligations,
    )
    return (
        CompileResult(
            CompileDisposition.SQL,
            sql,
            contract_self_validated=True,
        )
        if sql
        else CompileResult(CompileDisposition.NOT_APPLICABLE)
    )


def compile_advanced_medical_sql(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    """Compile one complete advanced medical query or abstain safely."""

    return compile_advanced_medical_result(question, domains).sql
