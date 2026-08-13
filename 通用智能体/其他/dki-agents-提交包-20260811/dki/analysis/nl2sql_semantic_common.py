"""Shared slots for schema-bound Chinese medical NL2SQL compilers.

The helpers in this module parse primitive values only: database-grounded
literals, numeric thresholds, dates, age/gender predicates and Top-K bounds.
They deliberately do not decide a query shape.  Shape selection remains in the
simple, advanced and graph compilers so an unsupported composition can abstain
instead of silently dropping a qualifier.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional

Domains = Mapping[str, Sequence[str]]


class RegionFilterDisposition(str, Enum):
    """Whether every grounded region reference has one safe interpretation."""

    NOT_APPLICABLE = "not_applicable"
    RENDERABLE = "renderable"
    TERMINAL_ABSTAIN = "terminal_abstain"


class RegionPolarity(str, Enum):
    """The eligibility polarity attached to one grounded region value."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass(frozen=True)
class RegionFilterIntent:
    """Structured ownership record for grounded patient-region filters.

    A region reference is negative only when an exclusion operator is attached
    to that reference or its contiguous enumeration.  This prevents unrelated
    wording such as ``不要求发生在同一次就诊`` from inverting an earlier
    positive region constraint.  Disjunctive mixed polarity cannot be rendered
    as a conjunction and therefore remains terminal instead of being broadened.
    """

    disposition: RegionFilterDisposition
    positive_values: tuple[str, ...] = ()
    negative_values: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        """Return whether callers must abstain rather than omit a region slot."""

        return self.disposition is RegionFilterDisposition.TERMINAL_ABSTAIN

_CN_INTS = {
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


def esc(value: object) -> str:
    return str(value).replace("'", "''")


def number(value: str) -> str:
    parsed = float(value)
    return str(int(parsed)) if parsed.is_integer() else f"{parsed:g}"


def _domain_value_occurrences(
    question: str,
    values: Sequence[str],
) -> list[tuple[int, int, str]]:
    """Return non-overlapping grounded-value spans in textual order."""

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


def domain_values(question: str, values: Sequence[str]) -> list[str]:
    """Return grounded values in occurrence order, preferring longer matches."""

    return [value for _, _, value in _domain_value_occurrences(question, values)]


def quoted(question: str) -> list[str]:
    return [
        value.strip()
        for value in re.findall(r"[“\"]([^”\"]+)[”\"]", question)
        if value.strip()
    ]


def dates(question: str) -> list[str]:
    return re.findall(r"\d{4}-\d{2}-\d{2}", question)


def integer(value: str) -> Optional[int]:
    cleaned = value.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return _CN_INTS.get(cleaned)


def limit(question: str) -> Optional[int]:
    for pattern in (
        r"(?:哪些|哪几种)\s*(\d+)\s*种.*?(?:最多|最频繁|最高)",
        r"前\s*(\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次|种|人)?",
        r"(?:最高|最低|最多|最少|最大|最小|最贵|最昂贵|最便宜|最长|最短)的?\s*"
        r"(\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次|种|人)?",
        r"(?:保留|取|返回)\s*(\d+|[一二两三四五六七八九十])\s*"
        r"(?:个|名|条|次|种|人)",
    ):
        match = re.search(pattern, question)
        if match:
            return integer(match.group(1))
    return None


def first_numeric(question: str, patterns: Sequence[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return number(match.group(1))
    return None


def answer_clause(question: str) -> str:
    """Return the final explicit output clause, excluding ordering language."""

    marker = re.compile(
        r"(?:并)?(?:列出|列举|给出|返回|显示|输出|展示|提供|附上?|"
        r"需包含|需要包含|包含|包括|需带|带出|只需)"
    )
    matches = list(marker.finditer(question))
    if not matches:
        return ""
    clause = ""
    order_only = re.compile(
        r"(?:大组|高频组合|时间顺序|较早(?:就诊|记录)?(?:优先|在前)?|"
        r"年长者(?:优先|在前)?|最贵的在前|便宜的在前|顺序输出)"
    )
    for match in reversed(matches):
        candidate = question[match.end():]
        candidate = re.split(r"[；。？?\n]", candidate, maxsplit=1)[0].strip(" ：:，,")
        if candidate and not order_only.fullmatch(candidate):
            clause = candidate
            break
    if not clause:
        return ""
    order_marker = re.search(
        r"(?:，|,)?\s*(?:"
        r"(?:(?:并|再|然后|最后|随后|其次)\s*)?(?:先)?(?:按|依|按照|依照|依据|采用).*$|"
        r"年长者(?:优先|在前).*$|最贵的在前.*$|便宜的在前.*$|"
        r"时间较早.*$|时间顺序.*$|较早(?:就诊|记录).*$|费用相同.*$|"
        r"先显示大组.*$|高频组合.*$|(?:并)?稳定排序.*$)",
        clause,
    )
    if order_marker:
        clause = clause[:order_marker.start()]
    presentation_marker = re.search(
        r"(?:，|,)?\s*(?:"
        r"涉及.*?(?:小数|金额|比例).*$|"
        r"(?:全部|两项|三项)?(?:金额|费用|均值|数值结果).*(?:保留|取|四舍五入).*$|"
        r"(?:全部|两项|三项)?金额(?:结果)?(?:保留|取|四舍五入)?"
        r"(?:到)?(?:小数点后)?(?:两|二|2|四|4)位.*$|"
        r"小数结果.*(?:保留|四舍五入).*$)",
        clause,
    )
    if presentation_marker:
        clause = clause[:presentation_marker.start()]
    return clause.strip(" ：:，,")


def answer_items(question: str) -> list[str]:
    """Split an explicit answer clause into ordered semantic phrases."""

    clause = answer_clause(question)
    if not clause:
        return []
    clause = clause.replace("以及", "、").replace("及其", "、")
    clause = clause.replace("并附", "、").replace("并给出", "、")
    clause = re.sub(
        r"(?<=[一-鿿])(?:和|与|(?<![提涉])及)(?=[一-鿿])",
        "、",
        clause,
    )
    return [
        item.strip(" ，,、；:：")
        for item in re.split(r"[、，,]", clause)
        if item.strip(" ，,、；:：")
    ]


def comparison(
    question: str,
    subject: str,
    *,
    aliases: Sequence[str] = (),
) -> Optional[tuple[str, str]]:
    """Parse one explicit scalar comparison adjacent to a semantic subject."""

    names = (subject, *aliases)
    label = "(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + ")"
    operators = (
        (r"(?:至少|不低于|不少于|达到|达|≥|>=)", ">="),
        (r"(?:超过|高于|大于|>|多于)", ">"),
        (r"(?:不超过|不高于|至多|≤|<=)", "<="),
        (r"(?:低于|小于|少于|<)", "<"),
        (r"(?:等于|恰好为|为|=)", "="),
    )
    for operator_pattern, operator in operators:
        match = re.search(
            rf"{label}\s*{operator_pattern}\s*(\d+(?:\.\d+)?)",
            question,
        )
        if match:
            return operator, number(match.group(1))
    return None


def age_predicates(question: str, alias: str = "") -> list[str]:
    """Parse explicit age ranges and one-sided age bounds."""

    column = f"{alias}.age" if alias else "age"
    between_patterns = (
        r"年龄(?:在|介于)?\s*(\d+)\s*岁?\s*(?:至|到|和|—|-)\s*(\d+)\s*岁?(?:之间)?",
        r"(\d+)\s*岁?(?:至|到|—|-)\s*(\d+)\s*岁",
        r"满\s*(\d+)\s*岁?且(?:不超过|至多)\s*(\d+)\s*岁?",
        r"年龄不少于\s*(\d+)\s*岁?且(?:不超过|至多)\s*(\d+)\s*岁?",
        r"(?=(?:年龄|年纪)|(?:至少|不低于|不少于|达到|达|≥|>=)\s*\d+\s*岁)"
        r"(?:年龄|年纪)(?:至少|不低于|不少于|达到|达|≥|>=)\s*(\d+)\s*岁?"
        r"\s*且(?:年龄|年纪)?(?:不超过|至多|不高于|≤|<=)\s*(\d+)\s*岁?",
        r"(?:至少|不低于|不少于|达到|达|≥|>=)\s*(\d+)\s*岁"
        r"\s*且(?:年龄|年纪)?(?:不超过|至多|不高于|≤|<=)\s*(\d+)\s*岁?",
    )
    for pattern in between_patterns:
        match = re.search(pattern, question)
        if match:
            return [
                f"{column} BETWEEN {int(match.group(1))} AND {int(match.group(2))}"
            ]

    predicates: list[str] = []
    lower_patterns = (
        (r"(?:年龄|年纪)(?:至少|不低于|不少于|不小于|达到|达|≥|>=)\s*(\d+)\s*岁?", ">="),
        (r"(?:至少|不低于|不少于|不小于|达到|达|≥|>=)\s*(\d+)\s*岁", ">="),
        (r"(?<!未)满\s*(\d+)\s*岁?", ">="),
        (r"(\d+)\s*岁及以上", ">="),
        (r"(?:年龄|年纪)(?:超过|大于|高于|>)\s*(\d+)\s*岁?", ">"),
        (r"(?:(?<!不)超过|(?<!不)大于|高于|>)\s*(\d+)\s*岁", ">"),
        (r"(\d+)\s*岁(?:以上|之上)", ">"),
    )
    upper_patterns = (
        (r"(?:年龄|年纪)(?:不超过|不大于|至多|不高于|≤|<=)\s*(\d+)\s*岁?", "<="),
        (r"(?:年龄|年纪)(?:低于|小于|<)\s*(\d+)\s*岁?", "<"),
        (r"未满\s*(\d+)\s*岁?", "<"),
        (r"不足\s*(\d+)\s*岁?", "<"),
        (r"(?:年龄|年纪)不足\s*(\d+)\s*岁?", "<"),
    )
    for patterns in (lower_patterns, upper_patterns):
        for pattern, operator in patterns:
            match = re.search(pattern, question)
            if match:
                predicate = f"{column} {operator} {int(match.group(1))}"
                if predicate not in predicates:
                    predicates.append(predicate)
                break
    return predicates


def gender_predicate(question: str, alias: str = "") -> Optional[str]:
    column = f"{alias}.gender" if alias else "gender"
    if re.search(r"女性|女患者|性别(?:为|是)?女", question):
        return f"{column} = '女'"
    if re.search(r"男性|男患者|性别(?:为|是)?男", question):
        return f"{column} = '男'"
    return None


_REGION_CLAUSE_BOUNDARY = r"[，,；;。！？?\n]"
_REGION_NEGATIVE_PREFIX = re.compile(
    r"(?:"
    r"(?:排除|不含|不包括|不要|不是|不在|不为|不属于|除外|非)\s*"
    r"(?:(?:居住地|所在地|所在)?(?:地区|区域|城市))?\s*(?:为|是)?"
    r"|(?:居住地|所在地|所在地区|地区|区域|城市)\s*"
    r"(?:不在|不是|不为|不属于|非)"
    r")\s*$"
)


def _region_negative_sources(question: str, start: int, end: int) -> tuple[str, ...]:
    """Return exclusion operators attached to one region-value occurrence."""

    before = re.split(_REGION_CLAUSE_BOUNDARY, question[:start])[-1][-24:]
    after = re.split(_REGION_CLAUSE_BOUNDARY, question[end:], maxsplit=1)[0][:24]
    sources: list[str] = []
    if _REGION_NEGATIVE_PREFIX.search(before):
        sources.append("prefix")
    if (
        re.match(r"\s*(?:(?:地区|区域|城市)\s*)?(?:以外|之外|除外)", after)
        or re.match(r"\s*(?:地区|区域|城市)\s*外", after)
        or re.match(r"\s*外(?=的|[，,；;。！？?\s]|$)", after)
    ):
        sources.append("suffix")
    return tuple(sources)


def _region_list_connector(text: str) -> Optional[bool]:
    """Return whether adjacent region mentions are joined by an ``or`` list."""

    normalized = re.sub(r"\s+", "", text)
    normalized = re.sub(r"^(?:地区|区域|城市)", "", normalized)
    normalized = re.sub(r"(?:地区|区域|城市)$", "", normalized)
    if normalized in {"、", "和", "与", "及", "以及", "/"}:
        return False
    if normalized in {"或", "或者"}:
        return True
    return None


def classify_region_filter_intent(
    question: object,
    domains: Optional[Domains] = None,
) -> RegionFilterIntent:
    """Classify grounded region values before a query family renders them.

    The parser keeps each value's local polarity instead of scanning the whole
    question for a negative word.  Contiguous conjunction lists inherit an
    attached exclusion operator, while an ``or`` expression that mixes a
    positive region with an exclusion is terminal because a flat ``WHERE``
    conjunction would change its meaning.
    """

    if not isinstance(question, str) or not question.strip() or not isinstance(domains, Mapping):
        return RegionFilterIntent(RegionFilterDisposition.NOT_APPLICABLE)
    occurrences = _domain_value_occurrences(question, domains.get("region", ()))
    if not occurrences:
        return RegionFilterIntent(RegionFilterDisposition.NOT_APPLICABLE)

    references = [
        (start, end, value, _region_negative_sources(question, start, end))
        for start, end, value in occurrences
    ]
    polarities: list[Optional[RegionPolarity]] = [None] * len(references)
    terminal = False
    index = 0
    while index < len(references):
        last = index
        connectors: list[bool] = []
        while last + 1 < len(references):
            connector = _region_list_connector(
                question[references[last][1]:references[last + 1][0]]
            )
            if connector is None:
                break
            connectors.append(connector)
            last += 1

        group = references[index:last + 1]
        has_negative = any(sources for _, _, _, sources in group)
        has_or = any(connectors)
        if has_or:
            prefix_indices = [
                position
                for position, (_, _, _, sources) in enumerate(group)
                if "prefix" in sources
            ]
            suffix_present = any("suffix" in sources for _, _, _, sources in group)
            if suffix_present or (prefix_indices and prefix_indices[0] != 0):
                terminal = True
            elif prefix_indices:
                for position in range(index, last + 1):
                    polarities[position] = RegionPolarity.EXCLUDE
            else:
                for position in range(index, last + 1):
                    polarities[position] = RegionPolarity.INCLUDE
        else:
            polarity = RegionPolarity.EXCLUDE if has_negative else RegionPolarity.INCLUDE
            for position in range(index, last + 1):
                polarities[position] = polarity
        index = last + 1

    for position in range(len(references) - 1):
        between = question[references[position][1]:references[position + 1][0]]
        if (
            not re.search(_REGION_CLAUSE_BOUNDARY, between)
            and re.search(r"(?:或|或者)", between)
            and polarities[position] is not polarities[position + 1]
        ):
            terminal = True

    positive_values: list[str] = []
    negative_values: list[str] = []
    for (_, _, value, _), polarity in zip(references, polarities):
        if polarity is RegionPolarity.EXCLUDE:
            if value not in negative_values:
                negative_values.append(value)
        elif value not in positive_values:
            positive_values.append(value)

    if terminal or set(positive_values).intersection(negative_values):
        return RegionFilterIntent(RegionFilterDisposition.TERMINAL_ABSTAIN)
    return RegionFilterIntent(
        RegionFilterDisposition.RENDERABLE,
        positive_values=tuple(positive_values),
        negative_values=tuple(negative_values),
    )


def region_predicates(question: str, domains: Domains, alias: str = "") -> list[str]:
    """Render each complete grounded region eligibility constraint.

    Query-family dispatchers must inspect ``classify_region_filter_intent``
    first when they own terminal abstention.  This compatibility helper returns
    no partial region predicate for an unrenderable intent.
    """

    intent = classify_region_filter_intent(question, domains)
    if intent.disposition is not RegionFilterDisposition.RENDERABLE:
        return []
    column = f"{alias}.region" if alias else "region"
    predicates: list[str] = []
    if intent.positive_values:
        rendered = ", ".join(f"'{esc(value)}'" for value in intent.positive_values)
        if len(intent.positive_values) == 1:
            predicates.append(f"{column} = {rendered}")
        else:
            predicates.append(f"{column} IN ({rendered})")
    if intent.negative_values:
        rendered = ", ".join(f"'{esc(value)}'" for value in intent.negative_values)
        predicates.append(f"{column} NOT IN ({rendered})")
    return predicates


def date_predicates(question: str, alias: str = "") -> list[str]:
    column = f"{alias}.visit_date" if alias else "visit_date"
    values = dates(question)
    if len(values) >= 2:
        return [f"{column} BETWEEN '{values[0]}' AND '{values[1]}'"]
    if len(values) == 1:
        value = re.escape(values[0])
        if re.search(rf"(?:自|从)\s*{value}|{value}\s*及?(?:起|以来|以后|之后)", question):
            return [f"{column} >= '{values[0]}'"]
        if "截至" in question:
            return [f"{column} <= '{values[0]}'"]
        if re.search(rf"{value}\s*(?:以前|之前|及以前)", question):
            return [f"{column} < '{values[0]}'"]
    return []


def grounded_predicates(
    question: str,
    domains: Domains,
    *,
    alias: str = "",
) -> list[str]:
    prefix = f"{alias}." if alias else ""
    predicates: list[str] = []
    for key, column in (
        ("department", "department"),
        ("disease", "disease"),
        ("drug", "drug_prescribed"),
        ("exam", "exam"),
    ):
        values = domain_values(question, domains.get(key, ()))
        if not values:
            continue
        rendered = ", ".join(f"'{esc(value)}'" for value in values)
        if len(values) == 1:
            predicates.append(f"{prefix}{column} = {rendered}")
        else:
            predicates.append(f"{prefix}{column} IN ({rendered})")
    return predicates


def relation_values(question: str, domains: Domains) -> list[str]:
    values = domain_values(question, domains.get("kg_relation_zh", ()))
    if values:
        return values
    allowed = set(map(str, domains.get("kg_relation_zh", ())))
    return [value for value in quoted(question) if value in allowed]
