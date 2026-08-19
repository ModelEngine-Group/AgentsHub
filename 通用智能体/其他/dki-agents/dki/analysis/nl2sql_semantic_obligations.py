"""Immutable row-qualifier obligations for the ordinary NL2SQL path.

``SemanticContract`` proves the requested answer shape.  This module proves a
different property: every grounded row qualifier in a question is attached to
the correct SQLite source, column, operator, polarity, and (for derived
queries) a base-row predicate stage.

The implementation is intentionally bounded to the four public project tables
and the source-proven qualifier gaps.  It is not a second general SQL parser.
When a qualifier cannot be assigned safely, extraction reports a terminal
abstention and candidate validation fails closed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

from dki.analysis.nl2sql_semantic_common import (
    RegionFilterDisposition,
    RegionPolarity,
    age_predicates,
    classify_region_filter_intent,
    domain_values,
    esc,
    gender_predicate,
)
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortIntent,
)

Domains = Mapping[str, Sequence[str]]


class ObligationDisposition(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    RENDERABLE = "renderable"
    TERMINAL_ABSTAIN = "terminal_abstain"


class SourceRole(str, Enum):
    PATIENT = "patients"
    VISIT = "visits"


class PredicatePhase(str, Enum):
    BASE_ROWS = "base_rows"
    COHORT_ANCHOR = "cohort_anchor"
    COHORT_FOLLOWUP = "cohort_followup"
    HAVING = "having"
    CONDITIONAL_AGGREGATE = "conditional_aggregate"


class PredicateOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    ALL_OF = "all_of"
    ABSENT = "absent"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    BETWEEN = "between"


class DateRole(str, Enum):
    EXACT = "exact"
    RANGE_START = "range_start"
    RANGE_END = "range_end"
    SINCE = "since"
    UNTIL = "until"
    COHORT_ANCHOR = "cohort_anchor"
    COHORT_FOLLOWUP = "cohort_followup"


@dataclass(frozen=True)
class PredicateObligation:
    concept: str
    source: SourceRole
    column: str
    operator: PredicateOperator
    values: tuple[object, ...]
    phase: PredicatePhase
    span: tuple[int, int]
    scope_id: Optional[str] = None


@dataclass(frozen=True)
class DateObligation:
    value: str
    role: DateRole
    operator: PredicateOperator
    phase: PredicatePhase
    span: tuple[int, int]


@dataclass(frozen=True)
class SemanticObligations:
    disposition: ObligationDisposition
    predicates: tuple[PredicateObligation, ...] = ()
    dates: tuple[DateObligation, ...] = ()
    unsupported_reasons: tuple[str, ...] = ()

    @property
    def requires_patient_source(self) -> bool:
        return any(item.source is SourceRole.PATIENT for item in self.predicates)

    @property
    def requires_visit_source(self) -> bool:
        return bool(self.dates) or any(
            item.source is SourceRole.VISIT for item in self.predicates
        )

    @property
    def requires_patient_visit_join(self) -> bool:
        return self.requires_patient_source and self.requires_visit_source

    @property
    def has_terminal_issue(self) -> bool:
        return self.disposition is ObligationDisposition.TERMINAL_ABSTAIN


@dataclass(frozen=True)
class ObligationValidationResult:
    accepted: bool
    reasons: tuple[str, ...] = ()


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PATIENT_COLUMNS = frozenset(("patient_id", "name", "age", "gender", "region"))
_VISIT_COLUMNS = frozenset(
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
    )
)
_TABLE_COLUMNS = {
    SourceRole.PATIENT.value: _PATIENT_COLUMNS,
    SourceRole.VISIT.value: _VISIT_COLUMNS,
}


def _is_graph_only_domain_request(question: str) -> bool:
    """Whether clinical domain literals name KG entities, not visit filters.

    Compact legacy questions such as ``高血压有哪些症状`` use a disease value
    as the graph head.  Treating the same token as ``visits.disease`` creates a
    false cross-source obligation and rejects the valid graph query.  Explicit
    row/cohort language keeps ordinary visit-domain extraction enabled for
    graph-plus-clinical compositions.
    """

    graph_cue = re.search(
        r"症状|临床表现|并发症|并发|禁忌|禁用|不能用|"
        r"用什么药|治疗药物|吃什么药|常用药|"
        r"治疗方式|怎么治疗|如何治疗|治疗手段|"
        r"如何确诊|怎么确诊|诊断依据|"
        r"需要(?:做|进行)?(?:什么|哪些)检查|做什么检查|"
        r"属于哪个科室|挂什么科|挂哪个科|看什么科|所属科室|"
        r"知识图谱|图谱|三元组|关系(?:边|链)?",
        question,
    )
    if graph_cue is None:
        return False
    row_or_cohort_cue = re.search(
        r"就诊|就医|住院|门诊|访问|病历|记录|"
        r"费用|花费|金额|成本|日期|时间|年龄|性别|地区|城市|"
        r"按(?:科室|疾病|月份|地区)|统计|汇总|覆盖率",
        question,
    )
    return row_or_cohort_cue is None


def _dedupe_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason)))


def _domain_occurrences(
    question: str,
    values: Sequence[str],
) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    for value in sorted(
        {str(item) for item in values if str(item)},
        key=lambda item: (-len(item), item),
    ):
        for match in re.finditer(re.escape(value), question):
            matches.append((match.start(), match.end(), value))
    selected: list[tuple[int, int, str]] = []
    for start, end, value in sorted(
        matches,
        key=lambda item: (item[0], -(item[1] - item[0]), item[2]),
    ):
        if any(start < used_end and end > used_start for used_start, used_end, _ in selected):
            continue
        selected.append((start, end, value))
    return sorted(selected)


def _local_context(question: str, start: int, end: int, width: int = 20) -> str:
    return question[max(0, start - width):min(len(question), end + width)]


def _has_role_context(
    question: str,
    start: int,
    end: int,
    *,
    role_pattern: str,
) -> bool:
    before = question[max(0, start - 48):start]
    after = question[end:min(len(question), end + 48)]
    return bool(
        re.search(
            rf"(?:{role_pattern})\s*(?:的)?\s*"
            rf"(?:为|是|=|：|:|with)?\s*$",
            before,
            re.I,
        )
        or re.match(
            rf"\s*(?:的|这位|这名|该名|这次|该次|this\s+)?\s*"
            rf"(?:{role_pattern})",
            after,
            re.I,
        )
    )


def _patient_id_source(
    question: str,
    start: int,
    end: int,
) -> SourceRole:
    """Choose the physical owner for an explicitly grounded patient ID.

    Explicit ``patient id``/``subject id`` and every Chinese identity form
    denote the patient entity.  A bare English ``patient P...`` in a
    visit-count question is intentionally visit-owned because the public
    schema can answer it directly from ``visits.patient_id``.  Questions that
    request a patient attribute still require ``patients.patient_id``.
    """

    context = question[max(0, start - 48):min(len(question), end + 48)]
    if re.search(
        r"(?:patient|subject)\s+(?:identifier|id|number)\s*"
        r"(?:is|=|:)?\s*$",
        question[max(0, start - 48):start],
        re.I,
    ):
        return SourceRole.PATIENT
    if re.search(r"患者|病人|病患|病号|参保人", context):
        return SourceRole.PATIENT
    if re.search(
        r"\b(?:how\s+old|age|gender|sex|name|region)\b",
        question,
        re.I,
    ):
        return SourceRole.PATIENT
    if re.search(
        r"\b(?:count|number\s+of\s+times|times|received?|prescri(?:bed|ption)|"
        r"visit|encounter|admission|hospitali[sz]ed|exam|procedure|drug|"
        r"disease|diagnosis)\b",
        question,
        re.I,
    ):
        return SourceRole.VISIT
    return SourceRole.PATIENT


def _negative_context(question: str, start: int, end: int) -> bool:
    before = re.split(r"[，,；;。！？?\n]", question[:start])[-1][-20:]
    after = re.split(r"[，,；;。！？?\n]", question[end:], maxsplit=1)[0][:12]
    return bool(
        re.search(r"(?:排除|不含|不包括|不要|不是|不在|不为|除外|非)\s*$", before)
        or re.match(r"\s*(?:以外|之外|除外)", after)
    )


def _absence_scope_id(
    question: str,
    start: int,
    end: int,
) -> Optional[str]:
    """Return the nearest explicit anti-existence cue owning this value."""

    after = question[end:min(len(question), end + 20)]
    trailing_cue = re.match(
        r"\s*(?:没有|没|未)(?:出现|发生|就诊|去过|使用|服用|开过)",
        after,
    )
    if trailing_cue is not None:
        return f"absence:{end + trailing_cue.start()}"

    clause_start = max(
        (
            question.rfind(separator, 0, start)
            for separator in ("，", ",", "；", ";", "。", "！", "？", "!", "?", "\n")
        ),
        default=-1,
    ) + 1
    before = question[clause_start:start]
    cue_pattern = re.compile(
        r"从来没有|从未|未曾|没有任何|没有在|没在|未在|"
        r"没有被|未被|没有使用过|没用过|未使用过|未使用|"
        r"没去过|未去过"
    )
    cues = list(cue_pattern.finditer(before))
    if not cues:
        return None
    cue = cues[-1]
    between = before[cue.end():]
    # A second explicit negative clause owns later values independently.
    if re.search(r"(?:且|并且|以及|同时).{0,8}(?:从未|未曾|没有|没|未)", between):
        return None
    if len(between) > 32:
        return None
    action = bool(
        re.search(
            r"就诊|看(?:过)?病|就医|去过|使用|服用|开过|被开过|出现|发生",
            cue.group(0) + between + after,
        )
    )
    return f"absence:{clause_start + cue.start()}" if action else None


def _mixed_or_polarity(
    question: str,
    occurrences: Sequence[tuple[int, int, str]],
) -> bool:
    if len(occurrences) < 2:
        return False
    polarities = [
        bool(
            _negative_context(question, start, end)
            or _absence_scope_id(question, start, end)
        )
        for start, end, _ in occurrences
    ]
    for index in range(len(occurrences) - 1):
        between = question[occurrences[index][1]:occurrences[index + 1][0]]
        if re.search(r"(?:或|或者)", between) and polarities[index] != polarities[index + 1]:
            return True
    return False


def _grouped_domain_obligations(
    question: str,
    *,
    concept: str,
    source: SourceRole,
    column: str,
    values: Sequence[str],
) -> tuple[list[PredicateObligation], list[str]]:
    occurrences = _domain_occurrences(question, values)
    if not occurrences:
        return [], []
    if _mixed_or_polarity(question, occurrences):
        return [], [f"mixed polarity for {concept}"]

    positive: list[str] = []
    negative: list[str] = []
    absent: dict[str, list[str]] = {}
    for start, end, value in occurrences:
        absence_scope = _absence_scope_id(question, start, end)
        if absence_scope is not None:
            target = absent.setdefault(absence_scope, [])
        else:
            target = negative if _negative_context(question, start, end) else positive
        if value not in target:
            target.append(value)

    span = (occurrences[0][0], occurrences[-1][1])
    obligations: list[PredicateObligation] = []
    if positive:
        required_all = bool(
            len(positive) > 1
            and re.search(
                r"既.{0,80}又|同时.{0,48}(?:且|并且|与|和)|分别",
                question,
            )
        )
        obligations.append(
            PredicateObligation(
                concept=concept,
                source=source,
                column=column,
                operator=(
                    PredicateOperator.EQ
                    if len(positive) == 1
                    else (
                        PredicateOperator.ALL_OF
                        if required_all
                        else PredicateOperator.IN
                    )
                ),
                values=tuple(positive),
                phase=PredicatePhase.BASE_ROWS,
                span=span,
            )
        )
    if negative:
        obligations.append(
            PredicateObligation(
                concept=concept,
                source=source,
                column=column,
                operator=(
                    PredicateOperator.NE
                    if len(negative) == 1
                    else PredicateOperator.NOT_IN
                ),
                values=tuple(negative),
                phase=PredicatePhase.BASE_ROWS,
                span=span,
            )
        )
    for scope_id, values_for_scope in absent.items():
        obligations.append(
            PredicateObligation(
                concept=concept,
                source=source,
                column=column,
                operator=PredicateOperator.ABSENT,
                values=tuple(values_for_scope),
                phase=PredicatePhase.BASE_ROWS,
                span=span,
                scope_id=scope_id,
            )
        )
    return obligations, []


def _identity_obligations(
    question: str,
    domains: Domains,
) -> tuple[list[PredicateObligation], list[str]]:
    obligations: list[PredicateObligation] = []
    reasons: list[str] = []
    identity_specs = (
        (
            "patient_id",
            SourceRole.PATIENT,
            "patient_id",
            r"(?:患者|病人|病患|病号|参保人)(?:\s*的)?"
            r"(?:\s*(?:编号|编码|ID|id|号))?|"
            r"(?:patient|subject)(?:\s+(?:identifier|id|number))?",
        ),
        (
            "visit_id",
            SourceRole.VISIT,
            "visit_id",
            r"(?:医疗)?(?:就诊|就医|住院|门诊|记录)(?:\s*的)?"
            r"(?:\s*(?:编号|编码|ID|id|号))?|"
            r"(?:visit|encounter|admission)(?:\s+(?:identifier|id|number))?|"
            r"(?:编号|编码)",
        ),
    )
    for concept, source, column, role_pattern in identity_specs:
        occurrences = _domain_occurrences(question, domains.get(concept, ()))
        contextual = [
            item
            for item in occurrences
            if _has_role_context(
                question,
                item[0],
                item[1],
                role_pattern=role_pattern,
            )
        ]
        # Numeric active-domain values are intentionally ignored without an
        # explicit role label.  They may be Top-K, age, gap, or metric bounds.
        if not contextual:
            continue
        by_source: dict[SourceRole, list[tuple[int, int, str]]] = {}
        for item in contextual:
            actual_source = (
                _patient_id_source(question, item[0], item[1])
                if concept == "patient_id"
                else source
            )
            by_source.setdefault(actual_source, []).append(item)
        for actual_source, source_items in by_source.items():
            values = tuple(dict.fromkeys(item[2] for item in source_items))
            obligations.append(
                PredicateObligation(
                    concept=concept,
                    source=actual_source,
                    column=column,
                    operator=(
                        PredicateOperator.EQ
                        if len(values) == 1
                        else PredicateOperator.IN
                    ),
                    values=values,
                    phase=PredicatePhase.BASE_ROWS,
                    span=(source_items[0][0], source_items[-1][1]),
                )
            )

    name_occurrences = _domain_occurrences(
        question,
        domains.get("patient_name", ()),
    )
    contextual_names = [
        item
        for item in name_occurrences
        if _has_role_context(
            question,
            item[0],
            item[1],
            role_pattern=r"(?:患者|病人|病患|病号|参保人)(?:\s*的)?"
            r"(?:\s*(?:姓名|名字|名称))?|"
            r"(?:姓名|名字|名称)|"
            r"(?:patient|subject)(?:\s+name)?",
        )
    ]
    if contextual_names:
        values = tuple(dict.fromkeys(item[2] for item in contextual_names))
        obligations.append(
            PredicateObligation(
                concept="patient_name",
                source=SourceRole.PATIENT,
                column="name",
                operator=PredicateOperator.EQ if len(values) == 1 else PredicateOperator.IN,
                values=values,
                phase=PredicatePhase.BASE_ROWS,
                span=(contextual_names[0][0], contextual_names[-1][1]),
            )
        )
    return obligations, reasons


def _age_obligations(question: str) -> list[PredicateObligation]:
    if not re.search(
        r"年龄|年纪|\d+\s*岁|\bage(?:d)?\b|\byears?\s+of\s+age\b|"
        r"\byounger\s+than\b",
        question,
        re.I,
    ):
        return []
    rendered = age_predicates(question, alias="")
    obligations: list[PredicateObligation] = []
    for predicate in rendered:
        between = re.fullmatch(r"age\s+BETWEEN\s+(\d+)\s+AND\s+(\d+)", predicate, re.I)
        if between:
            obligations.append(
                PredicateObligation(
                    concept="age",
                    source=SourceRole.PATIENT,
                    column="age",
                    operator=PredicateOperator.BETWEEN,
                    values=(int(between.group(1)), int(between.group(2))),
                    phase=PredicatePhase.BASE_ROWS,
                    span=(0, len(question)),
                )
            )
            continue
        comparison = re.fullmatch(r"age\s*(>=|>|<=|<|=)\s*(\d+)", predicate)
        if comparison:
            operator = {
                ">=": PredicateOperator.GE,
                ">": PredicateOperator.GT,
                "<=": PredicateOperator.LE,
                "<": PredicateOperator.LT,
                "=": PredicateOperator.EQ,
            }[comparison.group(1)]
            obligations.append(
                PredicateObligation(
                    concept="age",
                    source=SourceRole.PATIENT,
                    column="age",
                    operator=operator,
                    values=(int(comparison.group(2)),),
                    phase=PredicatePhase.BASE_ROWS,
                    span=(0, len(question)),
                )
            )

    english_patterns = (
        (
            PredicateOperator.LT,
            r"(?:\baged?\s+)?(?:below|less\s+than|younger\s+than)\s*"
            r"(\d+)\s*(?:years?(?:\s+of\s+age)?)?",
        ),
        (
            PredicateOperator.GT,
            r"(?:\baged?\s+)?(?:above|more\s+than|older\s+than)\s*"
            r"(\d+)\s*(?:years?(?:\s+of\s+age)?)?",
        ),
        (
            PredicateOperator.GE,
            r"(\d+)\s*(?:years?(?:\s+of\s+age)?)?\s*"
            r"(?:or|and)\s+(?:above|older)",
        ),
        (
            PredicateOperator.LE,
            r"(\d+)\s*(?:years?(?:\s+of\s+age)?)?\s*"
            r"(?:or|and)\s+(?:below|younger)",
        ),
    )
    existing = {
        (item.operator, item.values)
        for item in obligations
        if item.concept == "age"
    }
    for operator, pattern in english_patterns:
        for match in re.finditer(pattern, question, re.I):
            key = (operator, (int(match.group(1)),))
            if key in existing:
                continue
            existing.add(key)
            obligations.append(
                PredicateObligation(
                    concept="age",
                    source=SourceRole.PATIENT,
                    column="age",
                    operator=operator,
                    values=key[1],
                    phase=PredicatePhase.BASE_ROWS,
                    span=(match.start(), match.end()),
                )
            )
    return obligations


def _explicit_gender_value(question: str) -> Optional[str]:
    """Return a grounded database gender value from Chinese or English text."""

    female = bool(
        re.search(
            r"(?:女性|女)\s*(?:患者|病人|patients?)|"
            r"(?:患者|病人|patients?)\s*(?:为|是|are)?\s*(?:女性|女)|"
            r"\b(?:female|woman|women)\s+(?:patients?|subjects?)\b",
            question,
            re.I,
        )
    )
    male = bool(
        re.search(
            r"(?:男性|男)\s*(?:患者|病人|patients?)|"
            r"(?:患者|病人|patients?)\s*(?:为|是|are)?\s*(?:男性|男)|"
            r"\bmale\s+(?:patients?|subjects?)\b|\bmen\s+(?:patients?|subjects?)\b",
            question,
            re.I,
        )
    )
    if female == male:
        return None
    return "女" if female else "男"


def _demographic_obligations(
    question: str,
    domains: Domains,
) -> tuple[list[PredicateObligation], list[str]]:
    obligations = _age_obligations(question)
    reasons: list[str] = []
    gender = gender_predicate(question, alias="")
    gender_value: Optional[str] = None
    if gender:
        match = re.search(r"'([男女])'", gender)
        if match:
            gender_value = match.group(1)
    if gender_value is None:
        gender_value = _explicit_gender_value(question)
    if gender_value is not None:
        obligations.append(
            PredicateObligation(
                concept="gender",
                source=SourceRole.PATIENT,
                column="gender",
                operator=PredicateOperator.EQ,
                values=(gender_value,),
                phase=PredicatePhase.BASE_ROWS,
                span=(0, len(question)),
            )
        )

    region_intent = classify_region_filter_intent(question, domains)
    if region_intent.disposition is RegionFilterDisposition.TERMINAL_ABSTAIN:
        reasons.append("mixed or unsupported region polarity")
    elif region_intent.disposition is RegionFilterDisposition.RENDERABLE:
        occurrences = _domain_occurrences(question, domains.get("region", ()))
        span = (
            (occurrences[0][0], occurrences[-1][1])
            if occurrences
            else (0, len(question))
        )
        if region_intent.positive_values:
            obligations.append(
                PredicateObligation(
                    concept="region",
                    source=SourceRole.PATIENT,
                    column="region",
                    operator=(
                        PredicateOperator.EQ
                        if len(region_intent.positive_values) == 1
                        else PredicateOperator.IN
                    ),
                    values=tuple(region_intent.positive_values),
                    phase=PredicatePhase.BASE_ROWS,
                    span=span,
                )
            )
        if region_intent.negative_values:
            obligations.append(
                PredicateObligation(
                    concept="region",
                    source=SourceRole.PATIENT,
                    column="region",
                    operator=(
                        PredicateOperator.NE
                        if len(region_intent.negative_values) == 1
                        else PredicateOperator.NOT_IN
                    ),
                    values=tuple(region_intent.negative_values),
                    phase=PredicatePhase.BASE_ROWS,
                    span=span,
                )
            )
    return obligations, reasons


def _row_metric_obligations(question: str) -> list[PredicateObligation]:
    obligations: list[PredicateObligation] = []
    inclusive_cost_range = re.search(
        r"(?:费用|医疗费|花费|成本)\s*(?:在|从)\s*"
        r"(\d+(?:\.\d+)?)\s*元?\s*(?:到|至)\s*"
        r"(\d+(?:\.\d+)?)\s*元?\s*(?:之间)?\s*"
        r"(?:[（(]\s*)?(?:含边界|含首尾)(?:\s*[）)])?",
        question,
    )
    if inclusive_cost_range is not None:
        low_raw, high_raw = inclusive_cost_range.groups()
        low: object = float(low_raw) if "." in low_raw else int(low_raw)
        high: object = float(high_raw) if "." in high_raw else int(high_raw)
        obligations.append(
            PredicateObligation(
                concept="cost",
                source=SourceRole.VISIT,
                column="cost",
                operator=PredicateOperator.BETWEEN,
                values=(low, high),
                phase=PredicatePhase.BASE_ROWS,
                span=(inclusive_cost_range.start(), inclusive_cost_range.end()),
            )
        )

    operator_patterns = (
        (r"(?:至少|不低于|不少于|达到|达|≥|>=)", PredicateOperator.GE),
        (r"(?:不超过|不高于|至多|不多于|≤|<=)", PredicateOperator.LE),
        (r"(?:超过|高于|大于|多于|>)", PredicateOperator.GT),
        (r"(?:低于|小于|少于|<)", PredicateOperator.LT),
        (r"(?:等于|恰好为|为|=)", PredicateOperator.EQ),
    )
    metric_specs = (
        (
            "cost",
            "cost",
            r"(?:单次|每次|每笔|单笔|一次就诊|该次就诊|本次就诊)"
            r".{0,8}(?:费用|医疗费|花费|成本)|"
            r"(?:费用|医疗费|花费|成本).{0,8}"
            r"(?:单次|每次|每笔|单笔|一次就诊|该次就诊|本次就诊)",
        ),
        (
            "los_days",
            "los_days",
            r"(?:住院天数|住院时长|住院时间)",
        ),
    )
    for concept, column, label_pattern in metric_specs:
        if any(item.concept == concept for item in obligations):
            continue
        label = None
        for candidate in re.finditer(label_pattern, question):
            if concept == "los_days":
                before = question[max(0, candidate.start() - 12):candidate.start()]
                after = question[candidate.end():min(len(question), candidate.end() + 12)]
                derived_prefix = re.search(
                    r"(?:最长|最大|平均|均值|总|累计|合计|总和|求和)\s*的?\s*$",
                    before,
                )
                derived_suffix = re.match(
                    r"\s*(?:的\s*)?"
                    r"(?:最长|最大|平均|均值|总计|累计|合计|总和|求和)(?:值|数)?",
                    after,
                )
                if derived_prefix is not None or derived_suffix is not None:
                    continue
            label = candidate
            break
        if label is None and concept == "cost":
            # A plain "费用至少1000元" is a visit-row predicate unless the
            # measure is explicitly aggregate (平均/总/累计/人均费用).  Keep
            # aggregate thresholds for their HAVING owner instead of
            # misclassifying them as base-row obligations.
            for candidate in re.finditer(r"(?:费用|医疗费|花费|成本)", question):
                before = question[max(0, candidate.start() - 6):candidate.start()]
                if re.search(r"(?:平均|均值|总|累计|合计|人均)\s*$", before):
                    continue
                label = candidate
                break
        if not label:
            continue
        tail = question[label.end():min(len(question), label.end() + 32)]
        matched: Optional[tuple[PredicateOperator, str, tuple[int, int]]] = None
        for operator_pattern, operator in operator_patterns:
            after = re.search(
                rf".{{0,8}}?{operator_pattern}"
                r"\s*(\d+(?:\.\d+)?)",
                tail,
            )
            if after:
                if (
                    concept == "cost"
                    and operator is PredicateOperator.EQ
                    and re.search(
                        r"(?:编号|序号|排名|名次|行号)",
                        tail[:after.start(1)],
                    )
                ):
                    continue
                start = label.start()
                end = label.end() + after.end()
                matched = (operator, after.group(1), (start, end))
                break
        if matched is None:
            continue
        operator, raw_value, span = matched
        value: object = float(raw_value) if "." in raw_value else int(raw_value)
        obligations.append(
            PredicateObligation(
                concept=concept,
                source=SourceRole.VISIT,
                column=column,
                operator=operator,
                values=(value,),
                phase=PredicatePhase.BASE_ROWS,
                span=span,
            )
        )

    english_los_patterns = (
        (
            PredicateOperator.GT,
            r"(?:admitted|hospitali[sz]ed|stayed|length\s+of\s+stay|"
            r"duration\s+of\s+stay).{0,24}?"
            r"(?:for\s+)?more\s+than\s*(\d+(?:\.\d+)?)\s*days?",
        ),
        (
            PredicateOperator.GE,
            r"(?:admitted|hospitali[sz]ed|stayed|length\s+of\s+stay|"
            r"duration\s+of\s+stay).{0,24}?"
            r"(?:for\s+)?(?:at\s+least|no\s+less\s+than)\s*"
            r"(\d+(?:\.\d+)?)\s*days?",
        ),
        (
            PredicateOperator.EQ,
            r"(?:admitted|hospitali[sz]ed|stayed).{0,24}?"
            r"(?:for\s+)?(\d+(?:\.\d+)?)\s*days?",
        ),
    )
    if not any(item.concept == "los_days" for item in obligations):
        for operator, pattern in english_los_patterns:
            match = re.search(pattern, question, re.I)
            if match is None:
                continue
            raw_value = match.group(1)
            value: object = (
                float(raw_value) if "." in raw_value else int(raw_value)
            )
            obligations.append(
                PredicateObligation(
                    concept="los_days",
                    source=SourceRole.VISIT,
                    column="los_days",
                    operator=operator,
                    values=(value,),
                    phase=PredicatePhase.BASE_ROWS,
                    span=(match.start(), match.end()),
                )
            )
            break

    if not any(item.concept == "cost" for item in obligations):
        amount = re.search(
            r"(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下)",
            question,
        )
        if amount is not None and re.search(
            r"小金额|结算|费用|医疗费|花费|成本|金额",
            question,
        ):
            raw_value = amount.group(1)
            value = float(raw_value) if "." in raw_value else int(raw_value)
            obligations.append(
                PredicateObligation(
                    concept="cost",
                    source=SourceRole.VISIT,
                    column="cost",
                    operator=PredicateOperator.LE,
                    values=(value,),
                    phase=PredicatePhase.BASE_ROWS,
                    span=(amount.start(), amount.end()),
                )
            )
    return obligations


def _date_obligations(
    question: str,
    *,
    temporal_cohort_intent: Optional[TemporalCohortIntent],
) -> tuple[list[DateObligation], list[str]]:
    matches = list(_DATE_RE.finditer(question))
    if not matches:
        half_year = re.search(r"(20\d{2})年\s*(上半年|下半年)", question)
        if half_year:
            year = half_year.group(1)
            if half_year.group(2) == "上半年":
                start, end = f"{year}-01-01", f"{year}-06-30"
            else:
                start, end = f"{year}-07-01", f"{year}-12-31"
            span = (half_year.start(), half_year.end())
            return [
                DateObligation(
                    value=start,
                    role=DateRole.RANGE_START,
                    operator=PredicateOperator.GE,
                    phase=PredicatePhase.BASE_ROWS,
                    span=span,
                ),
                DateObligation(
                    value=end,
                    role=DateRole.RANGE_END,
                    operator=PredicateOperator.LE,
                    phase=PredicatePhase.BASE_ROWS,
                    span=span,
                ),
            ], []
        return [], []
    if len(matches) > 2:
        # A designated temporal owner may eventually allocate anchor/follow-up
        # roles explicitly.  Until that lowerer provides the allocation, never
        # let a generic parser truncate the question to its first two dates.
        owner = (
            temporal_cohort_intent is not None
            and temporal_cohort_intent.disposition
            is not TemporalCohortDisposition.NOT_APPLICABLE
        )
        reason = (
            "temporal owner has unassigned date literals"
            if owner
            else "unassigned date literals"
        )
        return [], [reason]

    if len(matches) == 2:
        left, right = matches
        between = question[left.end():right.start()]
        prefix = question[max(0, left.start() - 12):left.start()]
        proved_range = bool(
            re.search(r"(?:到|至|—|–|~|～|-)", between)
            or (
                re.search(r"(?:从|自)\s*$", prefix)
                and re.search(r"(?:到|至)", between)
            )
            or re.search(r"(?:和|与).{0,4}(?:之间|范围)", between)
        )
        if not proved_range:
            return [], ["two date literals have no proven shared range"]
        return [
            DateObligation(
                value=left.group(0),
                role=DateRole.RANGE_START,
                operator=PredicateOperator.GE,
                phase=PredicatePhase.BASE_ROWS,
                span=(left.start(), left.end()),
            ),
            DateObligation(
                value=right.group(0),
                role=DateRole.RANGE_END,
                operator=PredicateOperator.LE,
                phase=PredicatePhase.BASE_ROWS,
                span=(right.start(), right.end()),
            ),
        ], []

    match = matches[0]
    value = re.escape(match.group(0))
    before = question[max(0, match.start() - 16):match.start()]
    after = question[match.end():min(len(question), match.end() + 16)]
    if re.search(r"(?:自|从)\s*$", before) or re.match(
        r"\s*(?:起|以来|及以后|及之后|以后|之后)", after
    ):
        role = DateRole.SINCE
        operator = PredicateOperator.GE
    elif re.search(r"(?:截至|截止到|到)\s*$", before) or re.match(
        r"\s*(?:以前|之前|及以前|为止)", after
    ):
        role = DateRole.UNTIL
        operator = PredicateOperator.LE
    elif re.search(
        rf"(?:日期|就诊日|就诊日期)(?:为|是|=|：|:)?\s*{value}|"
        rf"(?:在|于)\s*{value}|{value}\s*(?:当天|当日)",
        question,
    ):
        role = DateRole.EXACT
        operator = PredicateOperator.EQ
    else:
        # An ISO date in a compact benchmark-style question is still an
        # explicit literal.  Treat it as an exact row qualifier rather than
        # silently ignoring it.
        role = DateRole.EXACT
        operator = PredicateOperator.EQ
    return [
        DateObligation(
            value=match.group(0),
            role=role,
            operator=operator,
            phase=PredicatePhase.BASE_ROWS,
            span=(match.start(), match.end()),
        )
    ], []


def extract_semantic_obligations(
    question: object,
    domains: Optional[Domains],
    *,
    temporal_cohort_intent: Optional[TemporalCohortIntent] = None,
) -> SemanticObligations:
    """Extract one immutable qualifier record for the public rule chain."""

    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(domains, Mapping)
    ):
        return SemanticObligations(ObligationDisposition.NOT_APPLICABLE)

    q = question.strip()
    predicates: list[PredicateObligation] = []
    reasons: list[str] = []

    identity, identity_reasons = _identity_obligations(q, domains)
    predicates.extend(identity)
    reasons.extend(identity_reasons)

    demographics, demographic_reasons = _demographic_obligations(q, domains)
    predicates.extend(demographics)
    reasons.extend(demographic_reasons)

    if not _is_graph_only_domain_request(q):
        for concept, source, column, key in (
            ("department", SourceRole.VISIT, "department", "department"),
            ("disease", SourceRole.VISIT, "disease", "disease"),
            ("drug", SourceRole.VISIT, "drug_prescribed", "drug"),
            ("exam", SourceRole.VISIT, "exam", "exam"),
        ):
            items, item_reasons = _grouped_domain_obligations(
                q,
                concept=concept,
                source=source,
                column=column,
                values=domains.get(key, ()),
            )
            predicates.extend(items)
            reasons.extend(item_reasons)

    absence_scopes = {
        item.scope_id
        for item in predicates
        if item.operator is PredicateOperator.ABSENT and item.scope_id is not None
    }
    if len(absence_scopes) > 1:
        reasons.append("independent anti-existence scopes are unsupported")

    predicates.extend(_row_metric_obligations(q))
    # In "X 占 ... 的比例/百分比", X belongs only to the numerator.
    # Perform this after every predicate extractor, otherwise later row metrics
    # such as "住院天数为0" would incorrectly remain base WHERE filters.
    ratio_index = q.find("占")
    if ratio_index >= 0 and re.search(
        r"占比|百分比|比例|占(?:全部|所有|总体|总)",
        q,
    ):
        conditional_concepts = {
            "gender",
            "age",
            "region",
            "department",
            "disease",
            "drug",
            "exam",
            "cost",
            "los_days",
        }
        predicates = [
            replace(item, phase=PredicatePhase.CONDITIONAL_AGGREGATE)
            if (
                item.phase is PredicatePhase.BASE_ROWS
                and item.concept in conditional_concepts
                and item.span[0] < ratio_index
            )
            else item
            for item in predicates
        ]
    date_items, date_reasons = _date_obligations(
        q,
        temporal_cohort_intent=temporal_cohort_intent,
    )
    reasons.extend(date_reasons)

    # Preserve deterministic identity if two parsers describe the same slot.
    predicates = list(dict.fromkeys(predicates))
    unsupported = _dedupe_reasons(reasons)
    if unsupported:
        return SemanticObligations(
            ObligationDisposition.TERMINAL_ABSTAIN,
            predicates=tuple(predicates),
            dates=tuple(date_items),
            unsupported_reasons=unsupported,
        )
    if predicates or date_items:
        return SemanticObligations(
            ObligationDisposition.RENDERABLE,
            predicates=tuple(predicates),
            dates=tuple(date_items),
        )
    return SemanticObligations(ObligationDisposition.NOT_APPLICABLE)


def _sql_literal(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    return f"'{esc(value)}'"


def render_predicate_obligation(
    obligation: PredicateObligation,
    *,
    alias: str,
) -> str:
    column = f"{alias}.{obligation.column}" if alias else obligation.column
    values = obligation.values
    if obligation.operator is PredicateOperator.BETWEEN:
        return (
            f"{column} BETWEEN {_sql_literal(values[0])} "
            f"AND {_sql_literal(values[1])}"
        )
    if obligation.operator in {PredicateOperator.IN, PredicateOperator.NOT_IN}:
        rendered = ", ".join(_sql_literal(value) for value in values)
        keyword = "IN" if obligation.operator is PredicateOperator.IN else "NOT IN"
        return f"{column} {keyword} ({rendered})"
    if obligation.operator in {
        PredicateOperator.ALL_OF,
        PredicateOperator.ABSENT,
    }:
        raise ValueError(f"{obligation.operator.value} requires a set-aware lowerer")
    operator = {
        PredicateOperator.EQ: "=",
        PredicateOperator.NE: "!=",
        PredicateOperator.GT: ">",
        PredicateOperator.GE: ">=",
        PredicateOperator.LT: "<",
        PredicateOperator.LE: "<=",
    }[obligation.operator]
    return f"{column}{operator}{_sql_literal(values[0])}"


def render_date_obligation(
    obligation: DateObligation,
    *,
    alias: str,
) -> str:
    column = f"{alias}.visit_date" if alias else "visit_date"
    operator = {
        PredicateOperator.EQ: "=",
        PredicateOperator.GT: ">",
        PredicateOperator.GE: ">=",
        PredicateOperator.LT: "<",
        PredicateOperator.LE: "<=",
    }[obligation.operator]
    return f"{column}{operator}'{esc(obligation.value)}'"


def render_source_predicates(
    obligations: SemanticObligations,
    source: SourceRole,
    *,
    alias: str,
    phase: PredicatePhase = PredicatePhase.BASE_ROWS,
) -> list[str]:
    if obligations.has_terminal_issue:
        return []
    rendered = [
        render_predicate_obligation(item, alias=alias)
        for item in obligations.predicates
        if (
            item.source is source
            and item.phase is phase
            and item.operator not in {
                PredicateOperator.ALL_OF,
                PredicateOperator.ABSENT,
            }
        )
    ]
    if source is SourceRole.VISIT:
        rendered.extend(
            render_date_obligation(item, alias=alias)
            for item in obligations.dates
            if item.phase is phase
        )
    return list(dict.fromkeys(rendered))


def _nearest(node: Any, classes: tuple[type[Any], ...]) -> Optional[Any]:
    current = node
    while current is not None:
        if isinstance(current, classes):
            return current
        current = getattr(current, "parent", None)
    return None


def _nearest_select(node: Any, exp: Any) -> Optional[Any]:
    return _nearest(node, (exp.Select,))


def _direct_table_aliases(select: Any, exp: Any) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in select.find_all(exp.Table):
        if _nearest_select(table, exp) is not select:
            continue
        name = str(getattr(table, "name", "")).casefold()
        alias = str(getattr(table, "alias_or_name", "")).casefold()
        if name and alias:
            aliases[alias] = name
            aliases.setdefault(name, name)
    return aliases


def _visible_qualified_source(select: Any, alias: str, exp: Any) -> Optional[str]:
    """Resolve a qualified alias through correlated outer SELECT scopes."""

    current = select
    while current is not None:
        aliases = _direct_table_aliases(current, exp)
        if alias in aliases:
            return aliases[alias]
        parent = getattr(current, "parent", None)
        current = _nearest(parent, (exp.Select,)) if parent is not None else None
    return None


def _column_source(
    column: Any,
    *,
    expected_source: SourceRole,
    exp: Any,
) -> bool:
    if not isinstance(column, exp.Column):
        return False
    if str(column.name).casefold() not in _TABLE_COLUMNS[expected_source.value]:
        return False
    select = _nearest_select(column, exp)
    if select is None:
        return False
    aliases = _direct_table_aliases(select, exp)
    table = str(getattr(column, "table", "")).casefold()
    if table:
        return _visible_qualified_source(select, table, exp) == expected_source.value

    candidates = {
        source
        for source in aliases.values()
        if str(column.name).casefold() in _TABLE_COLUMNS.get(source, ())
    }
    return candidates == {expected_source.value}


def _literal_value(node: Any, exp: Any) -> Optional[str]:
    current = node
    while isinstance(current, (exp.Paren, exp.Cast)):
        current = current.this
    if isinstance(current, exp.Literal):
        return str(current.this)
    if isinstance(current, exp.Neg) and isinstance(current.this, exp.Literal):
        return f"-{current.this.this}"
    return None


def _normalise_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _predicate_container(node: Any, exp: Any) -> Optional[Any]:
    return _nearest(node, (exp.Where, exp.Join, exp.Having))


def _column_in_expression(
    node: Any,
    *,
    source: SourceRole,
    column: str,
    exp: Any,
) -> Optional[Any]:
    if node is None:
        return None
    for candidate in node.find_all(exp.Column):
        if (
            str(candidate.name).casefold() == column.casefold()
            and _column_source(candidate, expected_source=source, exp=exp)
        ):
            return candidate
    if isinstance(node, exp.Column) and (
        str(node.name).casefold() == column.casefold()
        and _column_source(node, expected_source=source, exp=exp)
    ):
        return node
    return None


def _direct_obligation_column(
    node: Any,
    *,
    source: SourceRole,
    column: str,
    exp: Any,
) -> Optional[Any]:
    """Return a directly compared source column, never a nested transform.

    Searching an arbitrary operand subtree is not a semantic proof: expressions
    such as ``age + 100 > 18`` or ``CAST(age AS INTEGER) > 18`` contain the
    requested column while comparing a different value.  Parentheses are the
    only transparent wrapper accepted here.
    """

    current = node
    while isinstance(current, exp.Paren):
        current = current.this
    if not isinstance(current, exp.Column):
        return None
    if str(current.name).casefold() != column.casefold():
        return None
    if not _column_source(current, expected_source=source, exp=exp):
        return None
    return current


def _operator_for_binary(node: Any, exp: Any) -> Optional[PredicateOperator]:
    mapping = (
        (exp.EQ, PredicateOperator.EQ),
        (exp.NEQ, PredicateOperator.NE),
        (exp.GT, PredicateOperator.GT),
        (exp.GTE, PredicateOperator.GE),
        (exp.LT, PredicateOperator.LT),
        (exp.LTE, PredicateOperator.LE),
    )
    for expression_type, operator in mapping:
        if isinstance(node, expression_type):
            return operator
    return None


def _reverse_operator(operator: PredicateOperator) -> PredicateOperator:
    return {
        PredicateOperator.GT: PredicateOperator.LT,
        PredicateOperator.GE: PredicateOperator.LE,
        PredicateOperator.LT: PredicateOperator.GT,
        PredicateOperator.LE: PredicateOperator.GE,
    }.get(operator, operator)


def _container_is_valid(
    node: Any,
    *,
    phase: PredicatePhase,
    exp: Any,
    allow_anti_scope: bool = False,
) -> bool:
    if phase is PredicatePhase.CONDITIONAL_AGGREGATE:
        # The qualifier must be a CASE/IF condition consumed by an aggregate.
        # A WHERE/HAVING/JOIN predicate changes the denominator and is not an
        # equivalent lowering of a conditional numerator.
        saw_condition = False
        saw_aggregate = False
        aggregate_type = getattr(exp, "AggFunc", ())
        current = node
        while current is not None and not isinstance(current, exp.Select):
            if isinstance(current, (exp.Where, exp.Having, exp.Join)):
                return False
            if isinstance(current, (exp.Case, exp.If)):
                saw_condition = True
            if aggregate_type and isinstance(current, aggregate_type):
                saw_aggregate = True
            current = getattr(current, "parent", None)
        return saw_condition and saw_aggregate

    container = _predicate_container(node, exp)
    if container is None:
        return False
    current = getattr(node, "parent", None)
    while current is not None and current is not container:
        # An obligation is active only as a positive conjunctive atom.  Merely
        # occurring below another comparison/boolean operator is insufficient:
        # ``(age > 18) = 0`` reverses the requested predicate, while an OR branch
        # can be bypassed by its sibling.
        if not isinstance(current, (exp.And, exp.Paren)):
            return False
        current = getattr(current, "parent", None)
    if phase is PredicatePhase.HAVING:
        return isinstance(container, exp.Having)
    # Base-row, cohort-anchor, and follow-up qualifiers may be rendered in a
    # source WHERE or a filtering INNER JOIN.  A HAVING predicate is too late,
    # and an outer-join ON condition does not filter preserved-side rows.
    if isinstance(container, exp.Join):
        side = str(container.args.get("side") or "").casefold()
        kind = str(container.args.get("kind") or "").casefold()
        if side in {"right", "full"} or kind not in {"", "inner"}:
            return False
        if side == "left":
            # A right-side qualifier in LEFT JOIN ... ON is semantically
            # active when that nullable side is the explicitly grouped count
            # source (zero-match groups may be preserved).  It is not active
            # for a plain preserved-side patient list.
            select = _nearest_select(container, exp)
            source = container.this
            right_alias = str(
                getattr(source, "alias_or_name", "")
                or getattr(source, "name", "")
            ).casefold()
            group = select.args.get("group") if select is not None else None
            counted_right_source = bool(
                select is not None
                and group is not None
                and right_alias
                and any(
                    _nearest_select(count, exp) is select
                    and any(
                        isinstance(column, exp.Column)
                        and str(column.table).casefold() == right_alias
                        for column in count.find_all(exp.Column)
                    )
                    for count in select.find_all(exp.Count)
                )
            )
            if not counted_right_source:
                return False
    elif not isinstance(container, exp.Where):
        return False
    if allow_anti_scope:
        return True
    current = getattr(container, "parent", None)
    while current is not None:
        if isinstance(current, exp.Not) and isinstance(current.this, exp.Exists):
            return False
        if isinstance(current, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            current = getattr(current, "parent", None)
            continue
        current = getattr(current, "parent", None)
    return True


def _matches_binary_obligation(
    node: Any,
    *,
    source: SourceRole,
    column: str,
    operator: PredicateOperator,
    values: tuple[object, ...],
    phase: PredicatePhase,
    exp: Any,
    allow_anti_scope: bool = False,
) -> bool:
    actual_operator = _operator_for_binary(node, exp)
    if actual_operator is None:
        return False
    left_column = _direct_obligation_column(
        node.args.get("this"),
        source=source,
        column=column,
        exp=exp,
    )
    right_column = _direct_obligation_column(
        node.args.get("expression"),
        source=source,
        column=column,
        exp=exp,
    )
    if bool(left_column) == bool(right_column):
        return False
    if left_column is not None:
        literal = _literal_value(node.args.get("expression"), exp)
    else:
        actual_operator = _reverse_operator(actual_operator)
        literal = _literal_value(node.args.get("this"), exp)
    if literal is None:
        return False
    if actual_operator is not operator:
        return False
    return (
        literal == _normalise_value(values[0])
        and _container_is_valid(
            node,
            phase=phase,
            exp=exp,
            allow_anti_scope=allow_anti_scope,
        )
    )


def _matches_between_obligation(
    node: Any,
    *,
    source: SourceRole,
    column: str,
    values: tuple[object, ...],
    phase: PredicatePhase,
    exp: Any,
) -> bool:
    if not isinstance(node, exp.Between):
        return False
    if _direct_obligation_column(
        node.args.get("this"),
        source=source,
        column=column,
        exp=exp,
    ) is None:
        return False
    low = _literal_value(node.args.get("low"), exp)
    high = _literal_value(node.args.get("high"), exp)
    return (
        low == _normalise_value(values[0])
        and high == _normalise_value(values[1])
        and _container_is_valid(node, phase=phase, exp=exp)
    )


def _matches_in_obligation(
    node: Any,
    *,
    source: SourceRole,
    column: str,
    operator: PredicateOperator,
    values: tuple[object, ...],
    phase: PredicatePhase,
    exp: Any,
    allow_anti_scope: bool = False,
) -> bool:
    negative = isinstance(node, exp.Not) and isinstance(node.this, exp.In)
    target = node.this if negative else node
    if not isinstance(target, exp.In):
        return False
    expected_negative = operator is PredicateOperator.NOT_IN
    if negative != expected_negative:
        return False
    if _direct_obligation_column(
        target.args.get("this"),
        source=source,
        column=column,
        exp=exp,
    ) is None:
        return False
    actual_values = tuple(
        value
        for item in target.expressions
        if (value := _literal_value(item, exp)) is not None
    )
    expected_values = tuple(_normalise_value(value) for value in values)
    return (
        len(actual_values) == len(target.expressions)
        and set(actual_values) == set(expected_values)
        and _container_is_valid(
            node,
            phase=phase,
            exp=exp,
            allow_anti_scope=allow_anti_scope,
        )
    )


def _intersect_leaves(node: Any, exp: Any) -> list[Any]:
    wrappers = tuple(
        kind
        for name in ("Subquery", "Paren")
        if (kind := getattr(exp, name, None)) is not None
    )
    while wrappers and isinstance(node, wrappers):
        node = node.this
    if isinstance(node, exp.Intersect):
        return [
            *_intersect_leaves(node.args.get("this"), exp),
            *_intersect_leaves(node.args.get("expression"), exp),
        ]
    return [node]


def _union_leaves(node: Any, exp: Any) -> list[Any]:
    wrappers = tuple(
        kind
        for name in ("Subquery", "Paren")
        if (kind := getattr(exp, name, None)) is not None
    )
    while wrappers and isinstance(node, wrappers):
        node = node.this
    if isinstance(node, exp.Union) and node.args.get("distinct") is not False:
        return [
            *_union_leaves(node.args.get("this"), exp),
            *_union_leaves(node.args.get("expression"), exp),
        ]
    return [node]


def _matches_any_of_union(
    tree: Any,
    obligation: PredicateObligation,
    exp: Any,
) -> bool:
    """Prove an IN obligation lowered as homogeneous distinct UNION leaves."""

    expected = {_normalise_value(value) for value in obligation.values}
    if len(expected) < 2:
        return False
    for union in tree.find_all(exp.Union):
        if union.args.get("distinct") is False:
            continue
        leaves = _union_leaves(union, exp)
        covered: set[str] = set()
        valid = len(leaves) >= 2
        for leaf in leaves:
            leaf_values = {
                value
                for value in expected
                if any(
                    _matches_binary_obligation(
                        node,
                        source=obligation.source,
                        column=obligation.column,
                        operator=PredicateOperator.EQ,
                        values=(value,),
                        phase=obligation.phase,
                        exp=exp,
                    )
                    for node in leaf.walk()
                )
            }
            if len(leaf_values) != 1:
                valid = False
                break
            covered.update(leaf_values)
        if valid and covered == expected:
            return True
    return False


def _matches_all_of_obligation(
    tree: Any,
    obligation: PredicateObligation,
    exp: Any,
) -> bool:
    intersections = [
        node for node in tree.walk() if isinstance(node, exp.Intersect)
    ]
    expected = tuple(_normalise_value(value) for value in obligation.values)
    for intersection in intersections:
        leaves = _intersect_leaves(intersection, exp)
        covered: set[str] = set()
        matched_leaves = 0
        for leaf in leaves:
            leaf_values = {
                value
                for value in expected
                if any(
                    _matches_binary_obligation(
                        node,
                        source=obligation.source,
                        column=obligation.column,
                        operator=PredicateOperator.EQ,
                        values=(value,),
                        phase=obligation.phase,
                        exp=exp,
                    )
                    for node in leaf.walk()
                )
            }
            if leaf_values:
                matched_leaves += 1
                covered.update(leaf_values)
        if covered == set(expected) and matched_leaves >= len(expected):
            return True

    exists_matches: dict[str, int] = {}
    for exists in tree.find_all(exp.Exists):
        current = getattr(exists, "parent", None)
        under_or = False
        while current is not None and not isinstance(current, exp.Where):
            if isinstance(current, exp.Or):
                under_or = True
                break
            current = getattr(current, "parent", None)
        if under_or or not _has_patient_visit_join(exists, exp):
            continue
        matched_values = {
            value
            for value in expected
            if any(
                _matches_binary_obligation(
                    node,
                    source=obligation.source,
                    column=obligation.column,
                    operator=PredicateOperator.EQ,
                    values=(value,),
                    phase=obligation.phase,
                    exp=exp,
                )
                for node in exists.walk()
            )
        }
        if len(matched_values) != 1:
            continue
        value = next(iter(matched_values))
        exists_matches[value] = exists_matches.get(value, 0) + 1
    if set(exists_matches) == set(expected):
        return True
    return False


def _matches_absent_values_in_scope(
    scope: Any,
    obligation: PredicateObligation,
    exp: Any,
    *,
    allow_anti_scope: bool,
) -> bool:
    if len(obligation.values) == 1:
        return any(
            _matches_binary_obligation(
                node,
                source=obligation.source,
                column=obligation.column,
                operator=PredicateOperator.EQ,
                values=obligation.values,
                phase=obligation.phase,
                exp=exp,
                allow_anti_scope=allow_anti_scope,
            )
            for node in scope.walk()
        )
    return any(
        _matches_in_obligation(
            node,
            source=obligation.source,
            column=obligation.column,
            operator=PredicateOperator.IN,
            values=obligation.values,
            phase=obligation.phase,
            exp=exp,
            allow_anti_scope=allow_anti_scope,
        )
        for node in scope.find_all(exp.In)
    )


def _not_exists_is_conjunctive(node: Any, exp: Any) -> bool:
    current = getattr(node, "parent", None)
    while current is not None and not isinstance(current, exp.Where):
        if isinstance(current, exp.Or):
            return False
        current = getattr(current, "parent", None)
    return isinstance(current, exp.Where)


def _matches_absence_scope(
    tree: Any,
    obligations: Sequence[PredicateObligation],
    exp: Any,
) -> bool:
    """Prove one shared anti-existence clause in one NOT EXISTS/EXCEPT scope."""

    for node in tree.find_all(exp.Not):
        exists = node.this
        if (
            not isinstance(exists, exp.Exists)
            or not _not_exists_is_conjunctive(node, exp)
            or not _has_patient_visit_join(
                exists,
                exp,
                allow_anti_scope=True,
            )
        ):
            continue
        if all(
            _matches_absent_values_in_scope(
                exists,
                obligation,
                exp,
                allow_anti_scope=True,
            )
            for obligation in obligations
        ):
            return True

    except_nodes = [tree] if isinstance(tree, exp.Except) else []
    except_nodes.extend(tree.find_all(exp.Except))
    for difference in except_nodes:
        if difference.args.get("distinct") is False:
            continue
        right = difference.args.get("expression")
        wrappers = tuple(
            kind
            for name in ("Subquery", "Paren")
            if (kind := getattr(exp, name, None)) is not None
        )
        while wrappers and isinstance(right, wrappers):
            right = right.this
        if not isinstance(right, exp.Select):
            continue
        if all(
            _matches_absent_values_in_scope(
                right,
                obligation,
                exp,
                allow_anti_scope=False,
            )
            for obligation in obligations
        ):
            return True
    return False


def _matches_obligation(
    tree: Any,
    obligation: PredicateObligation,
    exp: Any,
) -> bool:
    if obligation.operator is PredicateOperator.ABSENT:
        return _matches_absence_scope(tree, (obligation,), exp)
    if obligation.operator is PredicateOperator.ALL_OF:
        return _matches_all_of_obligation(tree, obligation, exp)
    if obligation.operator is PredicateOperator.BETWEEN:
        return any(
            _matches_between_obligation(
                node,
                source=obligation.source,
                column=obligation.column,
                values=obligation.values,
                phase=obligation.phase,
                exp=exp,
            )
            for node in tree.find_all(exp.Between)
        )
    if obligation.operator in {PredicateOperator.IN, PredicateOperator.NOT_IN}:
        nodes: list[Any] = list(tree.find_all(exp.In))
        nodes.extend(tree.find_all(exp.Not))
        direct_match = any(
            _matches_in_obligation(
                node,
                source=obligation.source,
                column=obligation.column,
                operator=obligation.operator,
                values=obligation.values,
                phase=obligation.phase,
                exp=exp,
            )
            for node in nodes
        )
        if direct_match:
            return True
        if obligation.operator is PredicateOperator.IN:
            return _matches_any_of_union(tree, obligation, exp)
        return False
    return any(
        _matches_binary_obligation(
            node,
            source=obligation.source,
            column=obligation.column,
            operator=obligation.operator,
            values=obligation.values,
            phase=obligation.phase,
            exp=exp,
        )
        for node in tree.walk()
    )


def _matches_date_obligation(
    tree: Any,
    obligation: DateObligation,
    exp: Any,
) -> bool:
    predicate = PredicateObligation(
        concept="visit_date",
        source=SourceRole.VISIT,
        column="visit_date",
        operator=obligation.operator,
        values=(obligation.value,),
        phase=obligation.phase,
        span=obligation.span,
    )
    if _matches_obligation(tree, predicate, exp):
        return True
    # A two-date range is commonly rendered as one BETWEEN predicate.  Prove
    # each endpoint against its correct role instead of requiring two nodes.
    for node in tree.find_all(exp.Between):
        if _direct_obligation_column(
            node.args.get("this"),
            source=SourceRole.VISIT,
            column="visit_date",
            exp=exp,
        ) is None:
            continue
        low = _literal_value(node.args.get("low"), exp)
        high = _literal_value(node.args.get("high"), exp)
        if (
            obligation.role is DateRole.RANGE_START
            and low == obligation.value
            and _container_is_valid(node, phase=obligation.phase, exp=exp)
        ):
            return True
        if (
            obligation.role is DateRole.RANGE_END
            and high == obligation.value
            and _container_is_valid(node, phase=obligation.phase, exp=exp)
        ):
            return True
    return False


def _has_patient_visit_join(
    tree: Any,
    exp: Any,
    *,
    allow_anti_scope: bool = False,
) -> bool:
    for node in tree.find_all(exp.EQ):
        left = node.args.get("this")
        right = node.args.get("expression")
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        if str(left.name).casefold() != "patient_id" or str(right.name).casefold() != "patient_id":
            continue
        left_patient = _column_source(left, expected_source=SourceRole.PATIENT, exp=exp)
        left_visit = _column_source(left, expected_source=SourceRole.VISIT, exp=exp)
        right_patient = _column_source(right, expected_source=SourceRole.PATIENT, exp=exp)
        right_visit = _column_source(right, expected_source=SourceRole.VISIT, exp=exp)
        if (left_patient and right_visit) or (left_visit and right_patient):
            if _container_is_valid(
                node,
                phase=PredicatePhase.BASE_ROWS,
                exp=exp,
                allow_anti_scope=allow_anti_scope,
            ):
                return True
    return False


def _direct_base_column_identity(
    node: Any,
    exp: Any,
) -> Optional[tuple[SourceRole, str]]:
    for source, columns in (
        (SourceRole.PATIENT, _PATIENT_COLUMNS),
        (SourceRole.VISIT, _VISIT_COLUMNS),
    ):
        for column in columns:
            if _direct_obligation_column(
                node,
                source=source,
                column=column,
                exp=exp,
            ) is not None:
                return source, column
    return None


def _obligation_accepts_binary_atom(
    node: Any,
    obligation: PredicateObligation,
    exp: Any,
) -> bool:
    operators: tuple[PredicateOperator, ...]
    values: tuple[object, ...]
    allow_anti_scope = False
    if obligation.operator is PredicateOperator.ABSENT:
        operators = (PredicateOperator.EQ,)
        values = obligation.values
        allow_anti_scope = True
    elif obligation.operator in {
        PredicateOperator.ALL_OF,
        PredicateOperator.IN,
    }:
        operators = (PredicateOperator.EQ,)
        values = obligation.values
    elif obligation.operator in {
        PredicateOperator.EQ,
        PredicateOperator.NE,
        PredicateOperator.GT,
        PredicateOperator.GE,
        PredicateOperator.LT,
        PredicateOperator.LE,
    }:
        operators = (obligation.operator,)
        values = obligation.values[:1]
    else:
        return False
    return any(
        _matches_binary_obligation(
            node,
            source=obligation.source,
            column=obligation.column,
            operator=operator,
            values=(value,),
            phase=obligation.phase,
            exp=exp,
            allow_anti_scope=allow_anti_scope,
        )
        for operator in operators
        for value in values
    )


def _date_obligation_accepts_binary_atom(
    node: Any,
    obligation: DateObligation,
    exp: Any,
) -> bool:
    return _matches_binary_obligation(
        node,
        source=SourceRole.VISIT,
        column="visit_date",
        operator=obligation.operator,
        values=(obligation.value,),
        phase=obligation.phase,
        exp=exp,
    )


def _unconsumed_filter_reasons(
    tree: Any,
    obligations: SemanticObligations,
    exp: Any,
) -> tuple[str, ...]:
    """Reject literal base-row filters that no extracted obligation owns.

    The positive obligation checks prove required predicates are present.  This
    complementary closure prevents a candidate from silently narrowing the
    answer with an extra patient/visit literal.  Deliberately modeled structural
    predicates (rank aliases, aggregate comparisons, correlations, calendar
    functions) are not direct base-column/literal atoms and remain outside this
    narrow audit.
    """

    reasons: list[str] = []
    binary_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
    for node in tree.walk():
        if not isinstance(node, binary_types):
            continue
        if _predicate_container(node, exp) is None:
            continue

        left = node.args.get("this")
        right = node.args.get("expression")
        left_literal = _literal_value(left, exp)
        right_literal = _literal_value(right, exp)
        if left_literal is not None and right_literal is not None:
            reasons.append("unconsumed constant predicate")
            continue

        left_identity = _direct_base_column_identity(left, exp)
        right_identity = _direct_base_column_identity(right, exp)
        if bool(left_identity) == bool(right_identity):
            continue
        literal = right_literal if left_identity is not None else left_literal
        identity = left_identity or right_identity
        if identity is None or literal is None:
            continue
        source, column = identity
        if (
            source is SourceRole.VISIT
            and column in {"drug_prescribed", "exam"}
            and isinstance(node, exp.NEQ)
            and literal == ""
        ):
            # These two sparse text dimensions use the empty string as the
            # schema-level "not recorded" sentinel.  Distinct/grouped listing
            # compilers may exclude that sentinel without inventing a clinical
            # qualifier.
            continue

        if any(
            _obligation_accepts_binary_atom(node, obligation, exp)
            for obligation in obligations.predicates
        ) or any(
            _date_obligation_accepts_binary_atom(node, obligation, exp)
            for obligation in obligations.dates
        ):
            continue
        reasons.append(
            f"unconsumed {source.value}.{column} literal predicate"
        )

    for node in tree.find_all(exp.Between):
        if _predicate_container(node, exp) is None:
            continue
        identity = _direct_base_column_identity(node.args.get("this"), exp)
        low = _literal_value(node.args.get("low"), exp)
        high = _literal_value(node.args.get("high"), exp)
        if identity is None or low is None or high is None:
            continue
        consumed = any(
            obligation.operator is PredicateOperator.BETWEEN
            and _matches_between_obligation(
                node,
                source=obligation.source,
                column=obligation.column,
                values=obligation.values,
                phase=obligation.phase,
                exp=exp,
            )
            for obligation in obligations.predicates
        )
        if not consumed and identity == (SourceRole.VISIT, "visit_date"):
            bound_values = {
                obligation.value
                for obligation in obligations.dates
                if obligation.phase is PredicatePhase.BASE_ROWS
            }
            consumed = (
                low in bound_values
                and high in bound_values
                and _container_is_valid(
                    node,
                    phase=PredicatePhase.BASE_ROWS,
                    exp=exp,
                )
            )
        if not consumed:
            source, column = identity
            reasons.append(
                f"unconsumed {source.value}.{column} BETWEEN predicate"
            )

    for node in tree.find_all(exp.In):
        wrapper = (
            node.parent
            if isinstance(getattr(node, "parent", None), exp.Not)
            and node.parent.this is node
            else node
        )
        if _predicate_container(wrapper, exp) is None:
            continue
        identity = _direct_base_column_identity(node.args.get("this"), exp)
        values = tuple(_literal_value(item, exp) for item in node.expressions)
        if identity is None or not values or any(value is None for value in values):
            continue
        consumed = any(
            obligation.operator in {
                PredicateOperator.IN,
                PredicateOperator.NOT_IN,
            }
            and _matches_in_obligation(
                wrapper,
                source=obligation.source,
                column=obligation.column,
                operator=obligation.operator,
                values=obligation.values,
                phase=obligation.phase,
                exp=exp,
            )
            for obligation in obligations.predicates
        )
        if not consumed:
            source, column = identity
            reasons.append(
                f"unconsumed {source.value}.{column} IN predicate"
            )
    return _dedupe_reasons(reasons)


def validate_candidate_obligations(
    obligations: SemanticObligations,
    sql: str,
) -> ObligationValidationResult:
    """Prove all extracted row qualifiers against a SQLite SQL AST."""

    if obligations.has_terminal_issue:
        return ObligationValidationResult(
            False,
            obligations.unsupported_reasons or ("terminal semantic obligation",),
        )
    if not isinstance(sql, str) or not sql.strip():
        return ObligationValidationResult(False, ("empty SQL candidate",))

    try:
        from sqlglot import exp, parse

        statements = parse(sql, read="sqlite")
    except Exception:
        return ObligationValidationResult(False, ("invalid SQL AST",))
    if len(statements) != 1:
        return ObligationValidationResult(False, ("candidate is not one statement",))
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return ObligationValidationResult(False, ("candidate is not a read query",))

    reasons: list[str] = []
    absence_groups: dict[str, list[PredicateObligation]] = {}
    for obligation in obligations.predicates:
        if obligation.operator is PredicateOperator.ABSENT:
            scope_id = obligation.scope_id or f"absence:{obligation.span[0]}"
            absence_groups.setdefault(scope_id, []).append(obligation)
            continue
        if not _matches_obligation(tree, obligation, exp):
            reasons.append(f"missing {obligation.concept} qualifier")
    for group in absence_groups.values():
        if not _matches_absence_scope(tree, group, exp):
            concepts = "+".join(item.concept for item in group)
            reasons.append(f"missing anti-existence qualifier: {concepts}")
    for obligation in obligations.dates:
        if not _matches_date_obligation(tree, obligation, exp):
            reasons.append("unconsumed date literal")
    if obligations.requires_patient_visit_join and not _has_patient_visit_join(tree, exp):
        reasons.append("missing patient visit join")
    reasons.extend(_unconsumed_filter_reasons(tree, obligations, exp))
    reasons = list(_dedupe_reasons(reasons))
    return ObligationValidationResult(not reasons, tuple(reasons))
