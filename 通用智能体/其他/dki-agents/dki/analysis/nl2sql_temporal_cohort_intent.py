"""Structured ownership for temporal cohort, funnel, and retention requests.

This leaf deliberately decides ownership without producing SQL.  Its callers
must treat every disposition other than ``NOT_APPLICABLE`` as exclusive: a
failed designated owner may not let an unrelated compiler reinterpret only a
subset of the temporal cohort request.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dki.analysis.nl2sql_semantic_common import age_predicates, answer_items


Domains = Mapping[str, Sequence[str]]


class TemporalCohortDisposition(str, Enum):
    """The sole owner of a recognized temporal cohort request."""

    NOT_APPLICABLE = "not_applicable"
    ADVANCED_DATE_ONLY = "advanced_date_only"
    FAMILY_RETENTION = "family_retention"
    FAMILY_EVIDENCE_FUNNEL = "family_evidence_funnel"
    TERMINAL_ABSTAIN = "terminal_abstain"


class TemporalCohortKind(str, Enum):
    """The dedicated advanced structure, when one is present."""

    NONE = "none"
    FUNNEL = "funnel"
    RETENTION = "retention"
    REENGAGEMENT = "reengagement"


class TemporalCohortReason(str, Enum):
    """Why a recognized temporal cohort request cannot use its owner."""

    CONFLICTING_KIND = "conflicting_kind"
    MISSING_FIRST_VISIT_ANCHOR = "missing_first_visit_anchor"
    MISSING_INTERVALS = "missing_intervals"
    UNSUPPORTED_NON_DATE_QUALIFIER = "unsupported_non_date_qualifier"
    UNSUPPORTED_PROJECTION = "unsupported_projection"
    UNSUPPORTED_GROUPING = "unsupported_grouping"
    UNSUPPORTED_NEGATED_QUALIFIER = "unsupported_negated_qualifier"
    UNSUPPORTED_STAGE = "unsupported_stage"
    UNSUPPORTED_REENGAGEMENT = "unsupported_reengagement"
    UNSUPPORTED_FAMILY_RETENTION = "unsupported_family_retention"
    UNSUPPORTED_FAMILY_EVIDENCE = "unsupported_family_evidence"


@dataclass(frozen=True)
class TemporalCohortIntent:
    """Structured ownership facts for a temporal cohort request.

    ``FAMILY_*`` dispositions designate exactly one narrow semantic-family
    lowerer.  A caller must invoke only that lowerer and treat a ``None``
    result as terminal; broad family, KG, and simple fallbacks are unsafe.
    """

    disposition: TemporalCohortDisposition
    kind: TemporalCohortKind = TemporalCohortKind.NONE
    has_first_visit_anchor: bool = False
    has_first_visit_month: bool = False
    has_reengagement: bool = False
    has_relative_reengagement: bool = False
    has_funnel: bool = False
    has_retention: bool = False
    intervals: tuple[tuple[int, str], ...] = ()
    has_non_date_qualifier: bool = False
    has_unrenderable_projection: bool = False
    has_unsupported_grouping: bool = False
    has_negated_qualifier: bool = False
    has_complete_family_retention: bool = False
    has_complete_family_evidence: bool = False
    reasons: tuple[TemporalCohortReason, ...] = ()

    @property
    def is_exclusive(self) -> bool:
        """Return whether another generic compiler must not consume it."""

        return self.disposition is not TemporalCohortDisposition.NOT_APPLICABLE

    @property
    def requires_family_dispatch(self) -> bool:
        """Return whether one designated family lowerer owns this request."""

        return self.disposition in {
            TemporalCohortDisposition.FAMILY_RETENTION,
            TemporalCohortDisposition.FAMILY_EVIDENCE_FUNNEL,
        }


_FIRST_VISIT_ANCHOR = r"(?:首诊|首次就诊|第一次就诊)"
_FIRST_VISIT_MONTH = rf"{_FIRST_VISIT_ANCHOR}\s*(?:月|月份)"
_REENGAGEMENT = (
    r"(?:再次就诊|二次就诊|再(?:次)?就医|复诊|再诊|复访|随访|"
    r"回访|回诊|随诊)"
)
_FUNNEL = r"(?:漏斗|funnel)"
_RETENTION = r"(?:留存|retention)"
_COHORT_SIGNAL = r"(?:队列|cohort|人群漏斗|知识覆盖)"
_EVIDENCE = r"(?:证据|知识边|知识覆盖|图谱|疾病拥有.{0,12}关系|疾病是否存在.{0,12}边)"
_DATE = r"\d{4}-\d{2}-\d{2}"
_OPERATOR = r"(?:至少|不低于|不少于|达到|达|≥|>=|超过|高于|大于|多于|>|不超过|不高于|至多|≤|<=|低于|小于|少于|<|等于|恰好为|为|=)"
_NUMERIC_VISIT_OR_AGGREGATE = (
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


def _append_reason(
    reasons: list[TemporalCohortReason],
    reason: TemporalCohortReason,
) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _domain_values(question: str, values: Sequence[object]) -> tuple[str, ...]:
    """Return grounded values in question order without compiler imports."""

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
        if any(
            start < used_end and end > used_start
            for used_start, used_end, _ in selected
        ):
            continue
        selected.append((start, end, value))
    return tuple(value for _, _, value in sorted(selected))


def _intervals(question: str) -> tuple[tuple[int, str], ...]:
    """Parse the explicit positive day/month windows used by advanced SQL."""

    found: list[tuple[int, str, int]] = []
    patterns = (
        (r"(?<!\d)(\d+|[一二两三四五六七八九十])\s*(天|日)(?:内|后|时|留存)?", "day"),
        (r"第\s*(\d+|[一二两三四五六七八九十])\s*(?:个)?月", "month"),
        (r"(?<!\d)(\d+|[一二两三四五六七八九十])\s*个月(?:内|后|留存)?", "month"),
    )
    numerals = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    for pattern, unit in patterns:
        for match in re.finditer(pattern, question):
            token = match.group(1)
            value = int(token) if token.isdigit() else numerals.get(token)
            if value is not None and value > 0:
                found.append((value, unit, match.start()))
    result: list[tuple[int, str]] = []
    for value, unit, _ in sorted(found, key=lambda item: item[2]):
        item = (value, unit)
        if item not in result:
            result.append(item)
    return tuple(result)


def _has_relative_reengagement(question: str) -> bool:
    """Recognize an anchored follow-up interval, not an ordinary repeat visit."""

    return re.search(
        rf"{_FIRST_VISIT_ANCHOR}[^，,；。？?\n]{{0,24}}(?:后|之后|以来|起)"
        rf"[^，,；。？?\n]{{0,32}}{_REENGAGEMENT}",
        question,
        re.I,
    ) is not None


def _has_age_mention(question: str) -> bool:
    return re.search(r"(?:年龄|年纪|(?<!\d)\d+\s*岁)", question) is not None


def _has_renderable_cohort_age(question: str) -> bool:
    """Require an age predicate family cohort SQL retains in full."""

    predicates = age_predicates(question, alias="p")
    return bool(
        len(predicates) == 1
        and re.fullmatch(
            r"p\.age\s*(?:(?:>=|>)\s*\d+|BETWEEN\s+\d+\s+AND\s+\d+)",
            predicates[0],
        )
        is not None
    )


def _has_unrenderable_projection(question: str) -> bool:
    """Reject requested columns a fixed cohort template would silently omit."""

    items = answer_items(question)
    return bool(
        items
        and any(
            re.fullmatch(r"(?:结果|查询结果|统计结果)", item, re.I) is None
            for item in items
        )
    )


def _has_unsupported_grouping(question: str) -> bool:
    """Recognize grouping dimensions absent from the fixed cohort templates."""

    dimension = r"(?:地区|区域|城市|性别|年龄|科室|疾病|患者|病人|病患|病号)"
    return re.search(
        rf"(?:(?:按|依|按照|分)\s*{dimension}.{{0,16}}(?:分组|分别|统计|汇总)|"
        rf"(?:每个|各)\s*{dimension})",
        question,
    ) is not None


def _has_negated_value(question: str, values: Sequence[str]) -> bool:
    """Return whether a grounded value occurs in an unrenderable negative scope."""

    negative = r"(?:排除|不含|不包括|不要|非|不是|不在|没有|无|不具有|未具有|不存在)"
    for value in values:
        escaped = re.escape(value)
        if re.search(
            rf"(?:{negative}).{{0,16}}{escaped}|{escaped}.{{0,8}}(?:以外|之外|除外)",
            question,
        ):
            return True
    return False


def _genders(question: str) -> tuple[str, ...]:
    """Return direct genders the designated family renderer can express."""

    values: list[str] = []
    if re.search(r"女性|女患者|性别(?:为|是)?\s*女", question):
        values.append("女")
    if re.search(r"男性|男患者|性别(?:为|是)?\s*男", question):
        values.append("男")
    return tuple(values)


def _has_gender_mention(question: str) -> bool:
    return re.search(
        r"(?:女性|男性|女(?:性)?(?:患者|病人|病患|病号)|男(?:性)?(?:患者|病人|病患|病号)|性别)",
        question,
    ) is not None


def _has_unrenderable_gender(question: str, genders: Sequence[str]) -> bool:
    gender = r"(?:女性|男性|女(?:性)?(?:患者|病人|病患|病号)|男(?:性)?(?:患者|病人|病患|病号)|性别(?:为|是)?\s*[男女])"
    return bool(
        len(genders) != 1
        or re.search(rf"(?:不是|并非|非|不为|不属于|排除|不含|不要)\s*{gender}", question)
        or re.search(rf"除(?:了)?\s*{gender}\s*(?:以)?外", question)
        or re.search(r"性别\s*不(?:为|是)\s*[男女]", question)
    )


def _has_numeric_visit_or_aggregate_constraint(question: str) -> bool:
    labels = "|".join(
        re.escape(value)
        for value in sorted(_NUMERIC_VISIT_OR_AGGREGATE, key=len, reverse=True)
    )
    return bool(
        re.search(rf"(?:{labels})\s*(?:{_OPERATOR}|在|介于|范围|从)", question)
        or re.search(rf"(?:{_OPERATOR})\s*(?:有|达到|为)?\s*\d+(?:\.\d+)?[^。；;]{{0,24}}(?:{labels})", question)
    )


def _has_unrenderable_visit_qualifier(question: str) -> bool:
    """Catch ungrounded visit-scoped terms before a cohort template omits them."""

    return re.search(
        r"(?:患(?:有)?\s*[一-鿿A-Za-z0-9]{1,16}(?:病|症|癌)|"
        r"诊断(?:为|是)\s*[^，,；。？?\n]+|疾病(?:为|是)\s*[^，,；。？?\n]+|"
        r"(?:服用|使用|开具|处方|用药)\s*[一-鿿A-Za-z0-9]+|"
        r"(?:检查|检验|化验)\s*(?:为|是|显示|提示)|"
        r"(?:在|去|曾去|就诊于)\s*[一-鿿A-Za-z0-9]{1,12}(?:科|科室))",
        question,
    ) is not None


def _has_identity_filter(question: str) -> bool:
    return re.search(
        r"(?:患者|病人)?(?:姓名|名字|编号|ID)\s*(?:为|是|等于|叫|名为)\s*[^，,；。？?\n]+",
        question,
        re.I,
    ) is not None


def _has_unsupported_order(question: str) -> bool:
    return re.search(
        r"(?:按|依|按照|排序|排列).{0,32}(?:升序|降序|从高到低|从低到高|由大到小|由小到大)|"
        r"(?:前\s*(?:\d+|[一二两三四五六七八九十]+)|\btop\s*\d+|排名|名次)",
        question,
        re.I,
    ) is not None


def _retention_offset_upper(question: str) -> Optional[int]:
    for pattern in (
        r"(?:偏移月|月份偏移|第)?\s*0\s*(?:至|到|—|-)\s*(?:第)?\s*(\d+)",
        r"偏移(?:不超过|至多|≤|<=)\s*(\d+)\s*个月",
    ):
        match = re.search(pattern, question)
        if match:
            return int(match.group(1))
    return None


def _has_family_retention_measure(question: str) -> bool:
    """Match the only retention result shapes the fixed family renderer emits."""

    return re.search(r"留存率|占原队列比例|队列规模|活跃人数", question) is not None


def _family_retention_projection_is_supported(question: str) -> bool:
    """Accept only the five columns emitted by the retention family."""

    items = answer_items(question)
    if not items:
        return True
    patterns = (
        r"队列月|首诊月|cohort_month",
        r"(?:月份)?偏移|month_offset",
        r"人数|活跃人数|active_patient_count",
        r"队列规模|队列人数|cohort_patient_count",
        r"留存率|retention_rate",
    )
    return all(
        any(re.fullmatch(pattern, item, re.I) for pattern in patterns)
        for item in items
    )


def _has_family_retention_anchor(question: str) -> bool:
    """Keep family designation aligned with its lowerer's accepted anchor."""

    return re.search(r"首诊月|首次就诊月|首次就诊月份", question) is not None


def _has_single_since_date(question: str) -> bool:
    values = re.findall(_DATE, question)
    if len(values) != 1:
        return False
    value = re.escape(values[0])
    return re.search(rf"(?:自|从)\s*{value}|{value}\s*(?:起|以来|以后|之后|开始|后)", question) is not None


def _has_relation_metric_bound(question: str, labels: str) -> bool:
    return re.search(rf"(?:{labels})\s*(?:{_OPERATOR})\s*\d+(?:\.\d+)?", question, re.I) is not None


def _optional_relation_metric_is_complete(question: str, labels: str) -> bool:
    """Accept an omitted edge metric, but never a mentioned unparsed bound."""

    return (
        re.search(rf"(?:{labels})", question, re.I) is None
        or _has_relation_metric_bound(question, labels)
    )


def classify_temporal_cohort_intent(
    question: object,
    domains: Optional[Domains] = None,
) -> TemporalCohortIntent:
    """Classify temporal cohort ownership without selecting SQL.

    Bare cohort nouns and ordinary repeat-visit wording intentionally remain
    ``NOT_APPLICABLE``.  A first-visit monthly anchor plus funnel/retention,
    or a first-visit anchor paired with reengagement wording, is required
    before this leaf claims exclusive temporal ownership.
    """

    if not isinstance(question, str) or not question.strip():
        return TemporalCohortIntent(TemporalCohortDisposition.NOT_APPLICABLE)

    q = question.strip()
    active_domains = domains or {}
    has_first_visit_anchor = re.search(_FIRST_VISIT_ANCHOR, q, re.I) is not None
    has_first_visit_month = re.search(_FIRST_VISIT_MONTH, q, re.I) is not None
    has_reengagement = re.search(_REENGAGEMENT, q, re.I) is not None
    has_relative_reengagement = _has_relative_reengagement(q)
    has_funnel = re.search(_FUNNEL, q, re.I) is not None
    has_retention = re.search(_RETENTION, q, re.I) is not None
    has_evidence = re.search(_EVIDENCE, q, re.I) is not None
    has_cohort_signal = re.search(_COHORT_SIGNAL, q, re.I) is not None
    intervals = _intervals(q)
    regions = _domain_values(q, active_domains.get("region", ()))
    visit_values = tuple(
        value
        for key in ("department", "disease", "drug", "exam")
        for value in _domain_values(q, active_domains.get(key, ()))
    )
    relations = _domain_values(q, active_domains.get("kg_relation_zh", ()))
    has_age = _has_age_mention(q)
    has_renderable_age = _has_renderable_cohort_age(q)
    genders = _genders(q)
    has_gender = _has_gender_mention(q)
    has_unrenderable_gender = has_gender and _has_unrenderable_gender(q, genders)
    has_unrenderable_region = not regions and re.search(
        r"[一-鿿]{2,8}(?:地区|区域|城市)", q
    ) is not None
    has_unrenderable_visit = _has_unrenderable_visit_qualifier(q)
    has_numeric_visit = _has_numeric_visit_or_aggregate_constraint(q)
    has_identity = _has_identity_filter(q)
    has_order = _has_unsupported_order(q)
    has_unrenderable_projection = _has_unrenderable_projection(q)
    has_unsupported_grouping = _has_unsupported_grouping(q)
    has_negated_qualifier = bool(
        _has_negated_value(q, regions) or _has_negated_value(q, relations)
    )
    has_non_date_qualifier = bool(
        has_age
        or has_gender
        or regions
        or has_unrenderable_region
        or has_unrenderable_visit
        or visit_values
        or relations
        or has_evidence
        or has_numeric_visit
        or has_identity
        or has_order
    )
    common = {
        "has_first_visit_anchor": has_first_visit_anchor,
        "has_first_visit_month": has_first_visit_month,
        "has_reengagement": has_reengagement,
        "has_relative_reengagement": has_relative_reengagement,
        "has_funnel": has_funnel,
        "has_retention": has_retention,
        "intervals": intervals,
        "has_non_date_qualifier": has_non_date_qualifier,
        "has_unrenderable_projection": has_unrenderable_projection,
        "has_unsupported_grouping": has_unsupported_grouping,
        "has_negated_qualifier": has_negated_qualifier,
    }
    reasons: list[TemporalCohortReason] = []
    if has_unrenderable_projection:
        _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_PROJECTION)
    if has_unsupported_grouping:
        _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_GROUPING)
    if has_negated_qualifier:
        _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_NEGATED_QUALIFIER)

    # Family evidence only needs special ownership when it overlaps a funnel
    # phrase.  Evidence requests without that overlap stay available to the
    # existing family route rather than being claimed by temporal dispatch.
    evidence_funnel = has_funnel and has_evidence and has_cohort_signal
    family_evidence_complete = bool(
        evidence_funnel
        and len(regions) == 1
        and has_renderable_age
        and not has_unrenderable_gender
        and len(relations) == 1
        and _has_single_since_date(q)
        and _optional_relation_metric_is_complete(q, r"(?:关系)?置信度|confidence")
        and _optional_relation_metric_is_complete(
            q,
            r"(?:观察次数|出现次数|关系次数|计数|count)",
        )
        and not visit_values
        and not has_unrenderable_visit
        and not has_numeric_visit
        and not has_identity
        and not has_order
        and not has_unrenderable_region
        and not has_unrenderable_projection
        and not has_unsupported_grouping
        and not has_negated_qualifier
    )
    if evidence_funnel:
        if family_evidence_complete:
            return TemporalCohortIntent(
                TemporalCohortDisposition.FAMILY_EVIDENCE_FUNNEL,
                kind=TemporalCohortKind.FUNNEL,
                has_complete_family_evidence=True,
                **common,
            )
        _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_FAMILY_EVIDENCE)
        return TemporalCohortIntent(
            TemporalCohortDisposition.TERMINAL_ABSTAIN,
            kind=TemporalCohortKind.FUNNEL,
            reasons=tuple(reasons),
            **common,
        )

    # A first-visit anchor paired with reengagement is a cohort structure even
    # without the words ``漏斗`` or ``留存``.  No designated lowerer currently
    # preserves its year/month aggregation and follow-up-window semantics, so
    # seal it rather than letting a row-level repeat-visit compiler reinterpret
    # only the trailing reengagement phrase.  Ordinary repeat visits have no
    # first-visit anchor and intentionally remain not applicable.
    if not (has_funnel or has_retention):
        if has_first_visit_anchor and has_reengagement:
            _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_REENGAGEMENT)
            return TemporalCohortIntent(
                TemporalCohortDisposition.TERMINAL_ABSTAIN,
                kind=TemporalCohortKind.REENGAGEMENT,
                reasons=tuple(reasons),
                **common,
            )
        return TemporalCohortIntent(TemporalCohortDisposition.NOT_APPLICABLE, **common)
    if not has_first_visit_month:
        if len(intervals) >= 2 or re.search(r"(?:再次|复诊|复访|随访|阶段|留存)", q):
            _append_reason(reasons, TemporalCohortReason.MISSING_FIRST_VISIT_ANCHOR)
            return TemporalCohortIntent(
                TemporalCohortDisposition.TERMINAL_ABSTAIN,
                reasons=tuple(reasons),
                **common,
            )
        return TemporalCohortIntent(TemporalCohortDisposition.NOT_APPLICABLE, **common)

    if has_funnel and has_retention:
        _append_reason(reasons, TemporalCohortReason.CONFLICTING_KIND)
        return TemporalCohortIntent(
            TemporalCohortDisposition.TERMINAL_ABSTAIN,
            reasons=tuple(reasons),
            **common,
        )

    if has_retention:
        family_retention_complete = bool(
            _has_family_retention_anchor(q)
            and _has_family_retention_measure(q)
            and _retention_offset_upper(q) is not None
            and has_renderable_age
            and len(genders) == 1
            and not has_unrenderable_gender
            and not regions
            and not has_unrenderable_region
            and not visit_values
            and not has_unrenderable_visit
            and not relations
            and not has_evidence
            and not has_numeric_visit
            and not has_identity
            and not has_order
            and (
                not has_unrenderable_projection
                or _family_retention_projection_is_supported(q)
            )
            and not has_unsupported_grouping
            and not has_negated_qualifier
            and not re.search(_DATE, q)
        )
        if family_retention_complete:
            return TemporalCohortIntent(
                TemporalCohortDisposition.FAMILY_RETENTION,
                kind=TemporalCohortKind.RETENTION,
                has_complete_family_retention=True,
                **common,
            )
        if (
            len(intervals) >= 2
            and not has_non_date_qualifier
            and not has_unrenderable_projection
            and not has_unsupported_grouping
            and not has_negated_qualifier
            and re.search(r"(?:月|天).*(?:留存|retention)|(?:留存|retention).*(?:月|天)", q, re.I)
        ):
            return TemporalCohortIntent(
                TemporalCohortDisposition.ADVANCED_DATE_ONLY,
                kind=TemporalCohortKind.RETENTION,
                **common,
            )
        if _retention_offset_upper(q) is not None:
            _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_FAMILY_RETENTION)
        if len(intervals) < 2:
            _append_reason(reasons, TemporalCohortReason.MISSING_INTERVALS)
        if has_non_date_qualifier:
            _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_NON_DATE_QUALIFIER)
        if not reasons:
            _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_STAGE)
        return TemporalCohortIntent(
            TemporalCohortDisposition.TERMINAL_ABSTAIN,
            kind=TemporalCohortKind.RETENTION,
            reasons=tuple(reasons),
            **common,
        )

    # The only remaining temporal kind is a first-visit funnel.
    if (
        len(intervals) >= 2
        and not has_non_date_qualifier
        and not has_unrenderable_projection
        and not has_unsupported_grouping
        and not has_negated_qualifier
        and re.search(r"(?:再次|复诊|复访|随访|就诊|阶段|人数|统计|返回|计算)", q, re.I)
    ):
        return TemporalCohortIntent(
            TemporalCohortDisposition.ADVANCED_DATE_ONLY,
            kind=TemporalCohortKind.FUNNEL,
            **common,
        )
    if len(intervals) < 2:
        _append_reason(reasons, TemporalCohortReason.MISSING_INTERVALS)
    if has_non_date_qualifier:
        _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_NON_DATE_QUALIFIER)
    if not reasons:
        _append_reason(reasons, TemporalCohortReason.UNSUPPORTED_STAGE)
    return TemporalCohortIntent(
        TemporalCohortDisposition.TERMINAL_ABSTAIN,
        kind=TemporalCohortKind.FUNNEL,
        reasons=tuple(reasons),
        **common,
    )
