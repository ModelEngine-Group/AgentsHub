"""Ownership records for patient age-to-average comparisons.

This module deliberately records a narrow semantic family without rendering
SQL.  The explicit compiler owns one fixed global-average template, while the
advanced compiler owns correlated same-region/same-gender comparisons.  Keeping
that distinction in a dependency-light module lets both callers defer instead
of partially lowering a comparison that belongs to the other route.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional


Domains = Mapping[str, Sequence[str]]


class AgeAverageDisposition(str, Enum):
    """Exclusive owner for a recognized patient age-average comparison."""

    NOT_APPLICABLE = "not_applicable"
    EXPLICIT_GLOBAL = "explicit_global"
    ADVANCED_PEER = "advanced_peer"
    TERMINAL_ABSTAIN = "terminal_abstain"


class AgeAverageOrder(str, Enum):
    """Whether an explicit ordering is compatible with peer-age lowering."""

    NONE = "none"
    DEFAULT = "default"
    UNSUPPORTED = "unsupported"


class AgeAverageReason(str, Enum):
    """Recognized qualifier that prevents a safe owner from lowering SQL."""

    CONFLICTING_SCOPE = "conflicting_scope"
    DISJUNCTIVE_PEER_SCOPE = "disjunctive_peer_scope"
    UNSUPPORTED_COMPARATOR = "unsupported_comparator"
    VISIT_QUALIFIER = "visit_qualifier"
    UNSUPPORTED_ORDER = "unsupported_order"
    UNSUPPORTED_AGE_BOUND = "unsupported_age_bound"
    UNSUPPORTED_REGION_FILTER = "unsupported_region_filter"
    UNSUPPORTED_GENDER_FILTER = "unsupported_gender_filter"
    UNSUPPORTED_PEER_DEMOGRAPHIC_SCOPE = "unsupported_peer_demographic_scope"
    UNSUPPORTED_DERIVED_PEER_PROJECTION = "unsupported_derived_peer_projection"
    NONCANONICAL_GLOBAL_SURFACE = "noncanonical_global_surface"
    UNSUPPORTED_IDENTITY_FILTER = "unsupported_identity_filter"
    UNSUPPORTED_GROUPING = "unsupported_grouping"
    MISSING_PROJECTION = "missing_projection"
    UNSUPPORTED_PROJECTION = "unsupported_projection"
    UNSUPPORTED_AGGREGATE_PROJECTION = "unsupported_aggregate_projection"
    UNSUPPORTED_DISTINCT_PROJECTION = "unsupported_distinct_projection"


@dataclass(frozen=True)
class AgeAverageIntent:
    """Structured record of a narrow patient age-average request.

    ``projection`` uses schema-level patient names so the advanced compiler can
    render it directly.  ``outer_gender`` is populated only for one positive
    equality filter; ``outer_genders`` preserves every direct gender mention so
    negation or disjunction cannot be silently narrowed.  ``reasons`` retains
    every qualifier that made a recognized request unlowerable, rather than
    reducing it to an unrelated broad patient query.
    """

    disposition: AgeAverageDisposition
    peer_by_region: bool = False
    peer_by_gender: bool = False
    has_fixed_peer_gender_scope: bool = False
    has_fixed_peer_qualifier_scope: bool = False
    excludes_outer_patient_from_peer: bool = False
    has_derived_peer_average_projection: bool = False
    has_legacy_region_peer_scope: bool = False
    has_disjunctive_peer_scope: bool = False
    global_population: bool = False
    outer_gender: Optional[str] = None
    outer_genders: tuple[str, ...] = ()
    has_unrenderable_gender_filter: bool = False
    has_unrenderable_identity_filter: bool = False
    has_unrenderable_grouping: bool = False
    outer_regions: tuple[str, ...] = ()
    has_unrenderable_region_filter: bool = False
    has_outer_age_bound: bool = False
    has_unrenderable_age_bound: bool = False
    projection: tuple[str, ...] = ()
    has_projection: bool = False
    has_unrecognized_projection: bool = False
    has_aggregate_projection: bool = False
    has_distinct_projection: bool = False
    order: AgeAverageOrder = AgeAverageOrder.NONE
    has_date_qualifier: bool = False
    has_grounded_visit_value: bool = False
    has_explicit_visit_language: bool = False
    has_numeric_visit_constraint: bool = False
    reasons: tuple[AgeAverageReason, ...] = ()

    @property
    def has_visit_qualifier(self) -> bool:
        """Return whether a visit-owned qualifier lacks a peer-age owner."""

        return bool(
            self.has_date_qualifier
            or self.has_grounded_visit_value
            or self.has_explicit_visit_language
            or self.has_numeric_visit_constraint
        )


_PATIENT = r"(?:患者|病人|病患|病号|参保人|人群)"
_AVERAGE_AGE = r"(?:平均年龄|年龄平均|年龄均值)"
_POSITIVE_COMPARATOR = r"(?:高于|超过|大于|高过)"
_NEGATIVE_COMPARATOR = r"(?:不高于|不超过|低于|小于|少于)"
_REGION_PEER = r"同(?:一|个)?(?:地区|区域|城市|地域)"
_LEGACY_REGION_PEER = r"(?:本地区|所在地区|当地)"
_GENDER_PEER = r"(?:同性别|相同性别|同一性别)"
_PEER_GENDER = r"(?:女性|男性|女(?:性)?(?:患者|病人|病患|病号)|男(?:性)?(?:患者|病人|病患|病号))"
_FIXED_PEER_GENDER = rf"{_REGION_PEER}\s*(?:的)?\s*{_PEER_GENDER}\s*(?:的)?\s*{_AVERAGE_AGE}"
_FIXED_PEER_PATIENT_SCOPE = (
    rf"(?:{_REGION_PEER}|{_GENDER_PEER})\s*(?:的)?\s*"
    rf"(?P<scope>[^，,；。？?\n]{{1,24}}?)(?:的)?{_PATIENT}\s*(?:的)?\s*{_AVERAGE_AGE}"
)
_OTHER_PATIENT_PEER = (
    rf"(?:{_REGION_PEER}|{_GENDER_PEER})[^，,；。？?\n]{{0,12}}"
    rf"(?:其他|其它)(?:患者|病人|病患|病号)[^，,；。？?\n]{{0,12}}{_AVERAGE_AGE}"
)
_GLOBAL_POPULATION = r"(?:全部|全体|所有)(?:的)?患者"
_GLOBAL_ANCHOR = r"年龄\s*(?:高于|超过|大于|高过)\s*全部患者\s*(?:的)?\s*" + _AVERAGE_AGE
_DATE_QUALIFIER = r"(?:\d{4}-\d{2}-\d{2}|\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号]?)?)?)"
_VISIT_LANGUAGE = (
    r"就诊|门(?:急)?诊|急诊|住院|入院|出院|复诊|复访|随访|回访|回诊|随诊|"
    r"看(?:过)?病|就医|来院|诊疗|患(?!者)(?:有|上)?|确诊|诊断|"
    r"疾病|病种|使用(?:过)?|服用(?:过)?|开过|处方|用药|"
    r"做过|进行了?|接受了?|检查|检验|化验"
)
_DISEASE_QUALIFIED_PATIENT = r"[一-鿿]{1,8}(?:病|癌|炎|症|瘤)(?:患者|病人|病患|病号)"
_VISIT_METRIC = (
    r"(?:就诊次数|就诊量|访问次数|访问量|记录数|记录量|"
    r"总费用|费用总额|累计费用|合计费用|平均费用|费用均值|"
    r"费用|花费|金额|住院天数|住院时长)"
)
_COMPARATOR = (
    r"(?:至少|不低于|不少于|达到|达|>=|≥|超过|高于|大于|多于|>|"
    r"不超过|不高于|至多|<=|≤|低于|小于|少于|<|等于|恰好为|为|=)"
)


def _domain_values(question: str, values: Sequence[str]) -> list[str]:
    """Return schema-grounded values in textual order without dependencies."""

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
    return [value for _, _, value in sorted(selected)]


def _answer_items(question: str) -> list[str]:
    """Return an explicit final output list, excluding an ordering suffix."""

    marker = re.compile(
        r"(?:并)?(?:列出|列举|给出|返回|显示|输出|展示|提供|附上?|"
        r"需包含|需要包含|包含|包括|需带|带出|只需)"
    )
    clause = ""
    for match in reversed(list(marker.finditer(question))):
        candidate = re.split(r"[；。？?\n]", question[match.end():], maxsplit=1)[0]
        candidate = candidate.strip(" ：:，,")
        if candidate:
            clause = candidate
            break
    if not clause:
        return []
    order_marker = re.search(
        r"(?:，|,)?\s*(?:(?:(?:并|再|然后|最后|随后|其次)\s*)?"
        r"(?:先)?(?:按|依|按照|依照|依据|采用).*$)",
        clause,
    )
    if order_marker:
        clause = clause[:order_marker.start()]
    clause = clause.replace("以及", "、").replace("及其", "、")
    clause = clause.replace("并附", "、").replace("并给出", "、")
    clause = re.sub(r"(?<=[一-鿿])(和|与|及)(?=[一-鿿])", "、", clause)
    return [
        item.strip(" ，,、；:：")
        for item in re.split(r"[、，,]", clause)
        if item.strip(" ，,、；:：")
    ]


def _has_distinct_projection(question: str) -> bool:
    """Return whether the requested output needs row deduplication."""

    marker = re.compile(
        r"(?:并)?(?:列出|列举|给出|返回|显示|输出|展示|提供|附上?|"
        r"需包含|需要包含|包含|包括|需带|带出|只需)"
    )
    for match in reversed(list(marker.finditer(question))):
        clause = re.split(r"[；。？?\n]", question[match.end():], maxsplit=1)[0]
        if clause.strip(" ：:，,"):
            return re.search(r"(?:去重|不重复|唯一|不同(?:的)?|distinct)", clause, re.I) is not None
    return False


def _append_reason(
    reasons: list[AgeAverageReason],
    reason: AgeAverageReason,
) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _outer_genders(question: str) -> tuple[str, ...]:
    """Return every directly requested gender, excluding peer-scope wording."""

    genders: list[str] = []
    if re.search(r"女性|女(?:性)?(?:患者|病人|病患|病号)|性别(?:为|是)?\s*女", question):
        genders.append("女")
    if re.search(r"男性|男(?:性)?(?:患者|病人|病患|病号)|性别(?:为|是)?\s*男", question):
        genders.append("男")
    # ``性别为女或男`` and ``男女患者`` have two eligibility values even
    # though their second value need not repeat the ``性别`` marker.
    if re.search(r"性别(?:为|是)?\s*女\s*(?:或|或者|/|、|和|与)\s*男|男女(?:患者|病人|病患|病号)?", question):
        if "女" not in genders:
            genders.append("女")
        if "男" not in genders:
            genders.append("男")
    if re.search(r"性别(?:为|是)?\s*男\s*(?:或|或者|/|、|和|与)\s*女", question):
        if "男" not in genders:
            genders.append("男")
        if "女" not in genders:
            genders.append("女")
    return tuple(genders)


def _has_unrenderable_gender_filter(
    question: str,
    genders: Sequence[str],
) -> bool:
    """Identify gender semantics the equality-only outer renderer cannot own."""

    if len(genders) > 1:
        return True
    gender = r"(?:女性|男性|女(?:性)?(?:患者|病人|病患|病号)|男(?:性)?(?:患者|病人|病患|病号)|性别(?:为|是)?\s*[男女])"
    return bool(
        re.search(rf"(?:不是|并非|非|不为|不属于|排除|不含|不要)\s*{gender}", question)
        or re.search(rf"除(?:了)?\s*{gender}\s*(?:以)?外", question)
        or re.search(r"性别\s*不(?:为|是)\s*[男女]", question)
    )


def _eligibility_clause(question: str) -> str:
    """Return the portion that can constrain returned patients, not output."""

    return re.split(
        r"(?:列出|列举|给出|返回|显示|输出|展示|提供|需包含|需要包含|包含|包括)",
        question,
        maxsplit=1,
    )[0]


def _has_canonical_global_eligibility(question: str) -> bool:
    """Require the fixed global template's complete patient eligibility form."""

    eligibility = re.sub(r"\s+", "", _eligibility_clause(question))
    eligibility = eligibility.strip("，,；。？?：:")
    return re.fullmatch(
        r"(?:请)?(?:找出|查询|列出)?年龄(?:高于|超过|大于|高过)全部患者(?:的)?"
        r"(?:平均年龄|年龄平均|年龄均值)(?:的)?患者",
        eligibility,
    ) is not None


def _has_unrenderable_region_filter(
    question: str,
    outer_regions: Sequence[str],
) -> bool:
    """Detect a direct location phrase the outer renderer cannot preserve.

    Peer language is a correlation key rather than a direct outer location, but
    filter modifiers can appear after an output list.  Removing only the peer
    phrase keeps suffix/exclusion forms in that trailing position from being
    silently lowered as a different projection or population filter.
    """

    scope = re.sub(_REGION_PEER, "", question)
    eligibility = re.sub(_REGION_PEER, "", _eligibility_clause(question))
    if outer_regions:
        # The peer-age lowerer owns only its explicitly modelled region scope.
        # Keep suffix/exclusion forms such as ``北京地区以外`` exclusive and
        # terminal rather than letting the generic outer-region renderer choose
        # a possibly unintended population.
        for region in outer_regions:
            surface = rf"{re.escape(region)}(?:地区|区域|城市)?"
            if re.search(
                rf"(?:不为|不属于|不含|不包括|未在|没有)\s*{surface}|"
                rf"{surface}\s*(?:以外|之外|外)|"
                rf"除(?:了)?\s*{surface}\s*(?:以|之)?外",
                scope,
            ):
                return True
        return False
    # An ungrounded location such as ``广州地区`` cannot be rendered either.
    # Keep ordinary ``返回…地区`` output wording out of this heuristic.
    return re.search(r"[一-鿿]{2,8}(?:地区|区域|城市)", eligibility) is not None


def _has_unrenderable_identity_filter(question: str) -> bool:
    """Recognize direct patient identity filters absent from peer lowering.

    Predicate wording can follow an output phrase.  It must still seal the
    row-level peer route rather than be misread as an output name and omitted.
    """

    return bool(
        re.search(
            r"(?:患者|病人)?(?:姓名|名字|编号|ID)\s*(?:为|是|等于|叫|名为)\s*[^，,；。？?\n]+",
            question,
            re.I,
        )
        or re.search(r"(?:名为|叫)\s*[一-鿿]{1,12}(?:的)?(?:患者|病人)", question)
    )


def _has_unrenderable_grouping(question: str) -> bool:
    """Recognize grouping anywhere the row-level peer renderer cannot own.

    A grouping modifier may follow the requested output list (for example,
    ``返回患者编号和年龄，按地区分组``).  Scanning only eligibility would let
    that modifier become an ignored trailing clause and emit ungrouped rows.
    """

    dimension = r"(?:地区|区域|城市|性别|年龄)"
    return re.search(
        rf"(?:(?:按|依|按照|分)\s*{dimension}\s*(?:分别|分组|统计|汇总)|"
        rf"(?:每个|各)\s*{dimension})",
        question,
    ) is not None


def _has_outer_age_bound(question: str) -> bool:
    """Recognize a numeric age eligibility condition apart from the average."""

    age_number = r"(?:\d+|[一二两三四五六七八九十百]+)\s*岁"
    return bool(
        re.search(age_number, question)
        or re.search(
            r"(?:年龄|年纪)\s*(?:至少|不低于|不少于|达到|达|"
            r"超过|高于|大于|不超过|不高于|至多|低于|小于|少于|"
            r"等于|恰好为|为|>=|≥|<=|≤|>|<|=)\s*\d+",
            question,
        )
    )


def _has_renderable_outer_age_bound(question: str) -> bool:
    """Match the numeric age-bound forms retained by the shared renderer."""

    numeric = r"\d+"
    patterns = (
        rf"(?:年龄|年纪)(?:在|介于)?\s*{numeric}\s*岁?\s*(?:至|到|和|—|-)\s*{numeric}\s*岁?",
        rf"{numeric}\s*岁?(?:至|到|—|-)\s*{numeric}\s*岁",
        rf"满\s*{numeric}\s*岁?且(?:不超过|至多)\s*{numeric}\s*岁?",
        rf"年龄不少于\s*{numeric}\s*岁?且(?:不超过|至多)\s*{numeric}\s*岁?",
        rf"(?:年龄|年纪)(?:至少|不低于|不少于|达到|达|≥|>=)\s*{numeric}\s*岁?",
        rf"(?:至少|不低于|不少于|达到|达|≥|>=)\s*{numeric}\s*岁",
        rf"(?<!未)满\s*{numeric}\s*岁?",
        rf"(?:年龄|年纪)(?:超过|大于|高于|>)\s*{numeric}\s*岁?",
        rf"(?:超过|大于|高于|>)\s*{numeric}\s*岁",
        rf"(?:年龄|年纪)(?:不超过|至多|不高于|≤|<=|低于|小于|<)\s*{numeric}\s*岁?",
        rf"{numeric}\s*岁(?:及以上|以上|之上)",
        rf"未满\s*{numeric}\s*岁?",
    )
    return any(re.search(pattern, question) for pattern in patterns)


def _projection(question: str) -> tuple[tuple[str, ...], bool, bool, bool]:
    """Return requested patient projections and unsupported output modifiers."""

    items = _answer_items(question)
    if not items:
        return (), False, False, False
    projection: list[str] = []
    unknown = False
    aggregate = False
    for item in items:
        if re.search(
            r"平均|均值|总(?:数|费用|金额)?|累计|合计|计数|数量|"
            r"最大|最高|最小|最低|排名|比例|百分比",
            item,
        ):
            aggregate = True
            mapped = None
        elif re.search(r"就诊|记录", item):
            mapped = None
        elif re.search(
            r"患者\s*(?:编号|ID)|病人\s*(?:编号|ID)|患者ID|病人ID|编号|\bID\b",
            item,
            re.I,
        ):
            mapped = "patient_id"
        elif re.search(r"姓名|名字|名称", item):
            mapped = "name"
        elif re.search(r"年龄", item):
            mapped = "age"
        elif re.search(r"性别", item):
            mapped = "gender"
        elif re.search(r"地区|区域|城市", item):
            mapped = "region"
        else:
            mapped = None
        if mapped is None:
            unknown = True
            continue
        if mapped not in projection:
            projection.append(mapped)
    return tuple(projection), True, unknown, aggregate


def _has_derived_peer_average_projection(question: str) -> bool:
    """Detect a requested correlated aggregate not represented by ``p.age``."""

    peer_scope = (
        r"(?:同(?:一|个)?(?:地区|区域|城市|地域)|本地区|所在地区|当地|"
        r"同性别|相同性别|同一性别)"
    )
    return any(
        re.search(
            rf"{peer_scope}[^，,；。？?\n]{{0,16}}(?:平均年龄|年龄平均|年龄均值)",
            item,
        ) is not None
        for item in _answer_items(question)
    )


def _order(question: str) -> AgeAverageOrder:
    """Accept only the renderer's age-descending deterministic order."""

    if re.search(
        r"(?:前\s*(?:\d+|[一二两三四五六七八九十百]+)|"
        r"\btop\s*(?:\d+|[一二两三四五六七八九十百]+)|排名|名次|"
        r"第\s*(?:\d+|[一二两三四五六七八九十百]+)\s*页|"
        r"(?:取|保留|选出|筛选|仅(?:返回|保留)|只(?:返回|保留))\s*(?:前\s*)?"
        r"(?:\d+|[一二两三四五六七八九十百]+)\s*(?:个|位|名|条|人))",
        question,
        re.I,
    ):
        return AgeAverageOrder.UNSUPPORTED
    matches = list(
        re.finditer(
            r"(?:先按|再按|最后按|其次按|然后按|按照|按|依|依据)\s*"
            r"(?P<clause>[^；。？?\n]+?)(?:排列|排序|(?=$))",
            question,
        )
    )
    if not matches:
        if re.search(r"(?:升序|降序|从高到低|由大到小|从低到高|由小到大)", question):
            return AgeAverageOrder.UNSUPPORTED
        return AgeAverageOrder.NONE

    # Every stated ordering term must be covered.  Joining separate ``先按`` /
    # ``再按`` clauses makes an unsupported earlier term visible instead of
    # accepting only the final default-looking clause.
    clause = "、".join(match.group("clause") for match in matches)
    if not re.search(r"年龄\s*(?:降序|从高到低|由大到小)", clause):
        return AgeAverageOrder.UNSUPPORTED
    residual = re.sub(r"年龄\s*(?:降序|从高到低|由大到小)", "", clause)
    residual = re.sub(
        r"(?:患者\s*(?:编号|ID)|病人\s*(?:编号|ID)|患者ID|病人ID|编号|\bID\b)"
        r"\s*(?:升序|从低到高|由小到大)",
        "",
        residual,
        flags=re.I,
    )
    residual = re.sub(
        r"(?:年龄相同(?:时)?|同龄(?:者)?|优先|再|然后|以及|和|与|及|、|，|,|\s)",
        "",
        residual,
    )
    return AgeAverageOrder.DEFAULT if not residual else AgeAverageOrder.UNSUPPORTED


def _has_disjunctive_peer_scope(question: str) -> bool:
    """Detect a peer cohort union that correlation-by-AND cannot preserve."""

    connector = r"(?:或|或者|/|\bOR\b)"
    return bool(
        re.search(rf"{_REGION_PEER}\s*{connector}\s*{_GENDER_PEER}", question, re.I)
        or re.search(rf"{_GENDER_PEER}\s*{connector}\s*{_REGION_PEER}", question, re.I)
    )


def _has_fixed_peer_gender_scope(question: str) -> bool:
    """Detect a fixed peer demographic, distinct from a returned-patient filter."""

    return re.search(_FIXED_PEER_GENDER, question) is not None


def _has_fixed_peer_qualifier_scope(question: str) -> bool:
    """Detect peer-only demographics the correlation renderer cannot express."""

    supported_scope_words = {
        "其他",
        "其它",
        "同性别",
        "相同性别",
        "同一性别",
    }
    for match in re.finditer(_FIXED_PEER_PATIENT_SCOPE, question):
        scope = re.sub(r"\s+", "", match.group("scope")).strip("的")
        if scope and scope not in supported_scope_words:
            return True
    return False


def _excludes_outer_patient_from_peer(question: str) -> bool:
    """Return whether the comparison explicitly names other peer patients."""

    return re.search(_OTHER_PATIENT_PEER, question) is not None


def _relative_comparator(question: str) -> Optional[bool]:
    """Return whether an age comparison to an average is positive or negative."""

    # The left operand must be a patient's age, not the aggregate phrase
    # ``平均年龄高于 40``.  Do not scan across a clause boundary into a later
    # output request that merely mentions average age.
    direct = re.search(
        rf"(?<!平均)年龄\s*(?P<operator>{_POSITIVE_COMPARATOR}|{_NEGATIVE_COMPARATOR})"
        rf"[^，,；。？?\n]{{0,32}}?{_AVERAGE_AGE}",
        question,
    )
    if direct:
        return bool(re.fullmatch(_POSITIVE_COMPARATOR, direct.group("operator")))
    inverse = re.search(
        rf"(?<!平均)年龄[^，,；。？?\n]{{0,24}}?(?:比|较)"
        rf"[^，,；。？?\n]{{0,24}}?{_AVERAGE_AGE}"
        rf"[^，,；。？?\n]{{0,12}}?(?P<operator>高|大|低|小)",
        question,
    )
    if inverse:
        return inverse.group("operator") in {"高", "大"}
    return None


def _has_grounded_visit_value(question: str, domains: Domains) -> bool:
    return any(
        _domain_values(question, domains.get(key, ()))
        for key in ("department", "disease", "drug", "exam")
    )


def _has_numeric_visit_constraint(question: str) -> bool:
    return bool(
        re.search(rf"{_VISIT_METRIC}\s*(?:{_COMPARATOR}|在|介于|范围|从)", question)
        or re.search(rf"{_COMPARATOR}\s*\d+(?:\.\d+)?[^。；;]{{0,20}}{_VISIT_METRIC}", question)
    )


def classify_age_average_intent(
    question: object,
    domains: Optional[Domains] = None,
) -> AgeAverageIntent:
    """Classify a patient age-to-average request without choosing SQL.

    A disposition other than :attr:`AgeAverageDisposition.NOT_APPLICABLE` is
    exclusive: a caller must not let a different generic compiler consume only
    its patient or visit filters after the designated owner abstains.
    """

    if not isinstance(question, str) or not question.strip():
        return AgeAverageIntent(AgeAverageDisposition.NOT_APPLICABLE)
    q = question.strip()
    compact = re.sub(r"\s+", "", q)
    peer_by_region = re.search(_REGION_PEER, q) is not None
    peer_by_gender = re.search(_GENDER_PEER, q) is not None
    has_fixed_peer_gender_scope = _has_fixed_peer_gender_scope(q)
    has_fixed_peer_qualifier_scope = _has_fixed_peer_qualifier_scope(q)
    excludes_outer_patient_from_peer = _excludes_outer_patient_from_peer(q)
    has_legacy_region_peer_scope = re.search(_LEGACY_REGION_PEER, q) is not None
    has_disjunctive_peer_scope = _has_disjunctive_peer_scope(q)
    global_population = re.search(_GLOBAL_POPULATION, q) is not None
    (
        projection,
        has_projection,
        has_unrecognized_projection,
        has_aggregate_projection,
    ) = _projection(q)
    has_derived_peer_average_projection = _has_derived_peer_average_projection(q)
    has_distinct_projection = _has_distinct_projection(q)
    order = _order(q)
    has_date_qualifier = re.search(_DATE_QUALIFIER, q) is not None
    has_grounded_visit_value = _has_grounded_visit_value(q, domains or {})
    has_explicit_visit_language = bool(
        re.search(_VISIT_LANGUAGE, q)
        or re.search(_DISEASE_QUALIFIED_PATIENT, q)
    )
    has_numeric_visit_constraint = _has_numeric_visit_constraint(q)
    outer_genders = _outer_genders(q)
    has_unrenderable_gender_filter = _has_unrenderable_gender_filter(q, outer_genders)
    outer_gender = (
        outer_genders[0]
        if len(outer_genders) == 1 and not has_unrenderable_gender_filter
        else None
    )
    has_unrenderable_identity_filter = _has_unrenderable_identity_filter(q)
    has_unrenderable_grouping = _has_unrenderable_grouping(q)
    outer_regions = tuple(_domain_values(q, (domains or {}).get("region", ())))
    has_unrenderable_region_filter = _has_unrenderable_region_filter(q, outer_regions)
    has_outer_age_bound = _has_outer_age_bound(q)
    has_unrenderable_age_bound = bool(
        has_outer_age_bound and not _has_renderable_outer_age_bound(q)
    )
    reasons: list[AgeAverageReason] = []
    has_age_average = re.search(_AVERAGE_AGE, q) is not None
    relative = _relative_comparator(q)
    has_patient = re.search(_PATIENT, q) is not None
    exact_global = re.search(_GLOBAL_ANCHOR, compact) is not None
    visit_qualifier = bool(
        has_date_qualifier
        or has_grounded_visit_value
        or has_explicit_visit_language
        or has_numeric_visit_constraint
    )

    common = {
        "peer_by_region": peer_by_region,
        "peer_by_gender": peer_by_gender,
        "has_fixed_peer_gender_scope": has_fixed_peer_gender_scope,
        "has_fixed_peer_qualifier_scope": has_fixed_peer_qualifier_scope,
        "excludes_outer_patient_from_peer": excludes_outer_patient_from_peer,
        "has_derived_peer_average_projection": has_derived_peer_average_projection,
        "has_legacy_region_peer_scope": has_legacy_region_peer_scope,
        "has_disjunctive_peer_scope": has_disjunctive_peer_scope,
        "global_population": global_population,
        "outer_gender": outer_gender,
        "outer_genders": outer_genders,
        "has_unrenderable_gender_filter": has_unrenderable_gender_filter,
        "has_unrenderable_identity_filter": has_unrenderable_identity_filter,
        "has_unrenderable_grouping": has_unrenderable_grouping,
        "outer_regions": outer_regions,
        "has_unrenderable_region_filter": has_unrenderable_region_filter,
        "has_outer_age_bound": has_outer_age_bound,
        "has_unrenderable_age_bound": has_unrenderable_age_bound,
        "projection": projection,
        "has_projection": has_projection,
        "has_unrecognized_projection": has_unrecognized_projection,
        "has_aggregate_projection": has_aggregate_projection,
        "has_distinct_projection": has_distinct_projection,
        "order": order,
        "has_date_qualifier": has_date_qualifier,
        "has_grounded_visit_value": has_grounded_visit_value,
        "has_explicit_visit_language": has_explicit_visit_language,
        "has_numeric_visit_constraint": has_numeric_visit_constraint,
    }

    if exact_global:
        # The legacy explicit owner emits one fixed four-column, age-descending
        # template.  Treating a matching substring as sufficient would let it
        # silently discard an otherwise valid patient, visit, output, or ranking
        # qualifier.  Only the exact documented surface belongs to that owner.
        if not _has_canonical_global_eligibility(q):
            _append_reason(reasons, AgeAverageReason.NONCANONICAL_GLOBAL_SURFACE)
        if (
            peer_by_region
            or peer_by_gender
            or has_legacy_region_peer_scope
            or has_disjunctive_peer_scope
        ):
            _append_reason(reasons, AgeAverageReason.CONFLICTING_SCOPE)
        if outer_regions or has_unrenderable_region_filter:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_REGION_FILTER)
        if outer_genders or has_unrenderable_gender_filter:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_GENDER_FILTER)
        if has_outer_age_bound or has_unrenderable_age_bound:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_AGE_BOUND)
        if visit_qualifier:
            _append_reason(reasons, AgeAverageReason.VISIT_QUALIFIER)
        if has_unrenderable_identity_filter:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_IDENTITY_FILTER)
        if has_unrenderable_grouping:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_GROUPING)
        if order is not AgeAverageOrder.DEFAULT:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_ORDER)
        if not has_projection:
            _append_reason(reasons, AgeAverageReason.MISSING_PROJECTION)
        elif (
            projection != ("patient_id", "name", "age", "region")
            or has_unrecognized_projection
        ):
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_PROJECTION)
        if has_aggregate_projection:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_AGGREGATE_PROJECTION)
        if has_distinct_projection:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_DISTINCT_PROJECTION)
        disposition = (
            AgeAverageDisposition.EXPLICIT_GLOBAL
            if not reasons
            else AgeAverageDisposition.TERMINAL_ABSTAIN
        )
        return AgeAverageIntent(disposition, reasons=tuple(reasons), **common)

    # The legacy family deliberately owns its ``本地区`` / ``所在地区`` /
    # ``当地`` grammar.  A co-occurring modern gender marker must not make this
    # narrow intent consume only the gender half of that older cohort contract.
    if has_legacy_region_peer_scope and not peer_by_region and not global_population:
        return AgeAverageIntent(AgeAverageDisposition.NOT_APPLICABLE, **common)

    # Do not claim ordinary grouped average-age requests.  A peer/global scope
    # or a direct age-to-average comparison is required before this family owns
    # the wording.
    if not has_age_average or (not has_patient and not (peer_by_region or peer_by_gender)):
        return AgeAverageIntent(AgeAverageDisposition.NOT_APPLICABLE, **common)

    if peer_by_region or peer_by_gender:
        if has_disjunctive_peer_scope:
            _append_reason(reasons, AgeAverageReason.DISJUNCTIVE_PEER_SCOPE)
        if has_fixed_peer_gender_scope or has_fixed_peer_qualifier_scope:
            _append_reason(
                reasons,
                AgeAverageReason.UNSUPPORTED_PEER_DEMOGRAPHIC_SCOPE,
            )
        if has_derived_peer_average_projection:
            _append_reason(
                reasons,
                AgeAverageReason.UNSUPPORTED_DERIVED_PEER_PROJECTION,
            )
        if relative is not True:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_COMPARATOR)
        if visit_qualifier:
            _append_reason(reasons, AgeAverageReason.VISIT_QUALIFIER)
        if has_unrenderable_region_filter:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_REGION_FILTER)
        if has_unrenderable_gender_filter:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_GENDER_FILTER)
        if has_unrenderable_identity_filter:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_IDENTITY_FILTER)
        if has_unrenderable_grouping:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_GROUPING)
        if has_unrenderable_age_bound:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_AGE_BOUND)
        if order is AgeAverageOrder.UNSUPPORTED:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_ORDER)
        if not has_projection:
            _append_reason(reasons, AgeAverageReason.MISSING_PROJECTION)
        elif has_unrecognized_projection or not {"patient_id", "age"}.issubset(projection):
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_PROJECTION)
        if has_aggregate_projection:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_AGGREGATE_PROJECTION)
        if has_distinct_projection:
            _append_reason(reasons, AgeAverageReason.UNSUPPORTED_DISTINCT_PROJECTION)
        disposition = (
            AgeAverageDisposition.ADVANCED_PEER
            if not reasons
            else AgeAverageDisposition.TERMINAL_ABSTAIN
        )
        return AgeAverageIntent(disposition, reasons=tuple(reasons), **common)

    # A global-scope phrase or otherwise explicit patient age-to-average
    # comparison outside the fixed global template has no faithful owner.
    if global_population or relative is not None:
        _append_reason(reasons, AgeAverageReason.UNSUPPORTED_COMPARATOR)
        return AgeAverageIntent(
            AgeAverageDisposition.TERMINAL_ABSTAIN,
            reasons=tuple(reasons),
            **common,
        )
    return AgeAverageIntent(AgeAverageDisposition.NOT_APPLICABLE, **common)
