"""Schema-bound compilers for reusable medical knowledge-graph queries.

The compiler intentionally recognises semantic signatures rather than complete
question strings.  Relation names are accepted only when they are grounded in
``domains['kg_relation_zh']``; entity types and column names are limited to the
analysis schema.  An incomplete signature returns ``None`` instead of silently
omitting a direction, threshold, or requested projection.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from dki.analysis.nl2sql_compile_result import CompileDisposition
from dki.analysis.nl2sql_semantic_common import (
    RegionFilterDisposition,
    age_predicates,
    answer_items,
    classify_region_filter_intent,
    date_predicates,
    dates,
    esc,
    gender_predicate,
    grounded_predicates,
    integer,
    limit,
    region_predicates,
    relation_values,
)


_ENTITY_TYPES = {
    "Disease": "Disease",
    "Drug": "Drug",
    "Exam": "Exam",
    "Symptom": "Symptom",
    "Treatment": "Treatment",
    "Department": "Department",
    "疾病": "Disease",
    "药物": "Drug",
    "检查": "Exam",
    "症状": "Symptom",
    "治疗方式": "Treatment",
    "科室": "Department",
}


_OPERATOR_RE = (
    r"(?:至少|不低于|不少于|达到|达|≥|>=|"
    r"超过|高于|大于|>|多于|"
    r"不超过|不高于|至多|≤|<=|"
    r"低于|小于|少于|不足|<|"
    r"等于|恰好为|为|=)"
)

# Delimiters that end a caller-visible output or result-modifier clause.
_RESULT_CLAUSE_DELIMITERS = r"，,；;。.!！？?\n"


@dataclass(frozen=True)
class RelationOccurrence:
    """One grounded relation occurrence, retained even when values repeat."""

    value: str
    start: int
    end: int
    quoted: bool
    ordinal: int


@dataclass(frozen=True)
class RelationMetricBound:
    """One numeric relation metric comparison with its source ownership."""

    metric: str
    operator: str
    value: str
    start: int
    end: int
    owner: str
    owner_ordinal: Optional[int]


@dataclass(frozen=True)
class RelationRequest:
    """Structured relation request used before a row-level predicate is lowered.

    This intentionally records source spans rather than only deduplicated
    domain values.  A later route can therefore prove that it owns every
    relation occurrence, metric bound, projection, order clause, and Top-K
    clause before it emits a flat ``relation_zh`` predicate.
    """

    occurrences: tuple[RelationOccurrence, ...]
    connective: str
    relation_span: Optional[tuple[int, int]]
    metric_bounds: tuple[RelationMetricBound, ...]
    metric_complete: bool
    projection_spans: tuple[tuple[int, int], ...]
    ordering_spans: tuple[tuple[int, int], ...]
    top_k_spans: tuple[tuple[int, int], ...]
    errors: tuple[str, ...]

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.occurrences)

    @property
    def has_relation_intent(self) -> bool:
        return bool(self.occurrences)

    def accepts_flat_predicate(
        self,
        *,
        metric_capacity: Optional[Mapping[str, int]] = None,
        require_quoted_union: bool = False,
    ) -> bool:
        """Return whether one equality/IN predicate can consume this request.

        ``metric_capacity`` names how many bounds a candidate route can render
        for each row metric.  A missing capacity means that the route does not
        own that metric at all; ``None`` means it owns every parsed bound.
        """

        if not self.occurrences or self.errors or not self.metric_complete:
            return False
        if self.connective not in {"single", "literal_or"}:
            return False
        if self.connective == "literal_or" and (
            len(set(self.values)) != len(self.values)
            or any(not item.quoted for item in self.occurrences)
            and any(item.quoted for item in self.occurrences)
        ):
            # A partially quoted list has no unambiguous literal-list boundary.
            return False
        if require_quoted_union and (
            self.connective != "literal_or"
            or any(not item.quoted for item in self.occurrences)
        ):
            return False
        if any(bound.owner not in {"single", "shared"} for bound in self.metric_bounds):
            return False
        if metric_capacity is None:
            return True
        for metric in {bound.metric for bound in self.metric_bounds}:
            capacity = metric_capacity.get(metric)
            if capacity is None or sum(
                bound.metric == metric for bound in self.metric_bounds
            ) > capacity:
                return False
        return True


def relation_fallback_disposition(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> CompileDisposition:
    """Classify whether a declined structured relation request may fall back.

    Callers must try :func:`compile_kg_medical_sql` first.  A relation request
    with graph metrics, multiple relation roles, or parser errors cannot be
    safely broadened by a less specialised row or legacy route after that
    compiler abstains.  Plain, unqualified single-relation questions remain
    available to compatible legacy owners.
    """

    request = analyze_relation_request(question, domains)
    if not request.has_relation_intent:
        return CompileDisposition.NOT_APPLICABLE
    if (
        request.errors
        or not request.metric_complete
        or request.connective != "single"
        or request.metric_bounds
    ):
        return CompileDisposition.BLOCKED
    return CompileDisposition.NOT_APPLICABLE


def _unique(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item)
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _relation_mentions(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> list[tuple[int, int, str]]:
    """Return non-overlapping grounded relation mentions in text order."""

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


def _relations(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> list[str]:
    """Ground relation values and remove repeated mentions deterministically."""

    # relation_values also supports a quoted value when a caller supplies a
    # value domain but the normal occurrence matcher did not see it as a bare
    # token.  The occurrence list supplies stable ordering for repeated values.
    grounded = _unique(relation_values(question, domains))
    mentions = _unique([value for _, _, value in _relation_mentions(question, domains)])
    if mentions:
        return [value for value in mentions if value in grounded or not grounded]
    return grounded


def _number(value: str) -> str:
    parsed = float(value)
    return str(int(parsed)) if parsed.is_integer() else f"{parsed:g}"


def _metric_label(subject: str, aliases: Sequence[str] = ()) -> str:
    """Build a metric matcher without matching ``count`` inside ``visit_count``."""

    patterns: list[str] = []
    for name in sorted((subject, *aliases), key=len, reverse=True):
        escaped = re.escape(name)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            patterns.append(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        else:
            patterns.append(escaped)
    return "(?:" + "|".join(patterns) + ")"


def _aggregate_metric_modifier(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> bool:
    """Reject aggregate metric labels in row-level predicate compilers."""

    label = _metric_label(subject, aliases)
    modifier = (
        r"(?:平均(?:值|数)?(?:之?后(?:的)?)?|均值|总(?:和|计|量|数|额)?|累计|累积|合计|"
        r"最大|最小|AVG|AVERAGE|MEAN|SUM|TOTAL|MAX|MIN)"
    )
    threshold_start = rf"\s*(?:{_OPERATOR_RE}|在|介于|从|\d)"
    return bool(
        re.search(
            rf"(?:关系|边)?\s*{modifier}\s*(?:关系|边)?\s*{label}"
            rf"{threshold_start}|{label}\s*{modifier}{threshold_start}",
            question,
            re.I,
        )
    )


def _comparison_matches(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> list[tuple[str, str, int, int]]:
    """Parse every explicit numeric bound for one schema metric.

    A repeated upper or lower bound often omits the metric label (for example,
    ``置信度至少0.7且不超过0.9``).  The continuation is safe to retain because
    its operator is explicit and it immediately follows a labelled bound.  A
    bare second value remains intentionally unparsed unless ``分别`` makes its
    per-edge ownership explicit.
    """

    def normalize(raw_operator: str) -> str:
        if re.fullmatch(r"至少|不低于|不少于|达到|达|≥|>=", raw_operator):
            return ">="
        if re.fullmatch(r"超过|高于|大于|>|多于", raw_operator):
            return ">"
        if re.fullmatch(r"不超过|不高于|至多|≤|<=", raw_operator):
            return "<="
        if re.fullmatch(r"低于|小于|少于|不足|<", raw_operator):
            return "<"
        return "="

    label = _metric_label(subject, aliases)
    pattern = rf"{label}\s*(?:(?P<qualifier>均|都|分别|各)\s*)?"
    pattern += rf"(?P<operator>{_OPERATOR_RE})\s*"
    pattern += r"(?P<value>\d+(?:\.\d+)?)"
    if _aggregate_metric_modifier(question, subject, aliases):
        return []

    result: list[tuple[str, str, int, int]] = []
    seen_spans: set[tuple[int, int]] = set()

    def add(raw_operator: str, raw_value: str, start: int, end: int) -> None:
        try:
            value = _number(raw_value)
        except (TypeError, ValueError):
            return
        span = (start, end)
        if span not in seen_spans:
            result.append((normalize(raw_operator), value, start, end))
            seen_spans.add(span)

    def consume_conjunctive_bounds(cursor: int) -> None:
        """Retain every explicit unlabelled bound following one metric clause."""

        while True:
            following = re.match(
                rf"\s*(?:[，,；;]\s*)?"
                rf"(?:(?:且|并且|和|、|及|与|以及)\s*)?"
                rf"(?P<operator>{_OPERATOR_RE})\s*"
                r"(?P<value>\d+(?:\.\d+)?)",
                question[cursor:],
                re.I,
            )
            if following is None:
                return
            start = cursor + following.start("operator")
            end = cursor + following.end("value")
            add(following.group("operator"), following.group("value"), start, end)
            cursor += following.end()

    for match in re.finditer(pattern, question, re.I):
        add(
            match.group("operator"),
            match.group("value"),
            match.start(),
            match.end(),
        )
        # “分别至少0.7和0.8” contains the second bound without repeating
        # the metric label.  It is still an explicit per-edge threshold.
        if match.group("qualifier") == "分别":
            following = question[match.end():]
            second = re.match(r"\s*(?:和|、|及)\s*(\d+(?:\.\d+)?)", following)
            if second:
                second_start = match.end() + second.start(1)
                second_end = match.end() + second.end(1)
                add(match.group("operator"), second.group(1), second_start, second_end)
        consume_conjunctive_bounds(match.end())

    reverse_pattern = (
        rf"(?P<operator>{_OPERATOR_RE})\s*"
        rf"(?:有|含|包含|达到|为|覆盖|连接|关联)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:个|次|位|名|条|种|人|元|天)?\s*(?:的)?\s*"
        rf"{label}"
    )
    for match in re.finditer(reverse_pattern, question, re.I):
        add(
            match.group("operator"),
            match.group("value"),
            match.start(),
            match.end(),
        )

    range_patterns = (
        rf"{label}\s*(?:均|都|各)?\s*(?:在|介于|范围(?:为|是)?|从)\s*"
        r"(?P<low>\d+(?:\.\d+)?)\s*(?:至|到|—|-)\s*(?P<high>\d+(?:\.\d+)?)(?:之间)?",
        rf"{label}\s*(?:均|都|各)?\s*(?:在|介于)\s*"
        r"(?P<low>\d+(?:\.\d+)?)\s*(?:和|与)\s*(?P<high>\d+(?:\.\d+)?)(?:之间)?",
    )
    for range_pattern in range_patterns:
        for match in re.finditer(range_pattern, question, re.I):
            try:
                low = float(match.group("low"))
                high = float(match.group("high"))
            except (TypeError, ValueError):
                continue
            if low > high:
                continue
            add(">=", match.group("low"), match.start("low"), match.end("low"))
            # The range end includes “之间”, so follow-up completeness checks
            # begin at the actual end of the range rather than its upper value.
            add("<=", match.group("high"), match.start("high"), match.end())
            consume_conjunctive_bounds(match.end())
    return sorted(result, key=lambda item: item[2])


def _or_matches(question: str) -> list[re.Match[str]]:
    """Find Chinese, ASCII, or full-width OR tokens without word leakage."""

    return list(
        re.finditer(
            r"或(?:者)?|(?<![A-Za-zＡ-Ｚａ-ｚ])[OＯ][RＲ](?![A-Za-zＡ-Ｚａ-ｚ])",
            question,
            re.I,
        )
    )


def has_disjunctive_metric_bound(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> bool:
    """Detect a scalar metric OR that an AND-only route cannot lower safely.

    A route that later conjuncts row predicates can preserve a range or a
    conjunction, but not ``metric >= low OR <= high`` or a cross-metric
    alternative.  The second branch commonly omits the repeated metric label,
    so inspect source spans around every recognised OR token rather than
    looking only for two fully-labelled comparisons.
    """

    label = _metric_label(subject, aliases)
    value = r"\d+(?:\.\d+)?(?:\s*(?:元|天|个|次|条))?"
    comparison = rf"(?:{_OPERATOR_RE})\s*{value}"
    for token in _or_matches(question):
        before = question[max(0, token.start() - 96) : token.start()]
        after = question[token.end() : token.end() + 96]
        before_own = re.search(
            rf"{label}\s*{comparison}\s*(?:[，,；;]\s*)?$",
            before,
            re.I,
        )
        after_own = re.match(
            rf"\s*(?:[，,；;]\s*)?(?:{label}\s*)?{comparison}",
            after,
            re.I,
        )
        before_bound = re.search(
            rf"{comparison}\s*(?:[，,；;]\s*)?$",
            before,
            re.I,
        )
        after_bound = re.match(
            rf"\s*(?:[，,；;]\s*)?(?:[^\d，,；;\n]{{0,24}}?)?{comparison}",
            after,
            re.I,
        )
        # A cross-metric alternative (for example, ``费用至少100 OR
        # 住院天数至少3``) is no safer than a repeated-metric alternative:
        # the row-level routes below can only conjunct their predicates.
        if (before_own and after_bound) or (after_own and before_bound):
            return True
    return False


def _quoted_spans(question: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in re.finditer(r"[“\"](?:[^”\"]+)[”\"]", question)
    ]


def _relation_occurrences(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> tuple[RelationOccurrence, ...]:
    """Ground every relation occurrence and retain quotation provenance."""

    quote_spans = _quoted_spans(question)
    return tuple(
        RelationOccurrence(
            value=value,
            start=start,
            end=end,
            quoted=any(left <= start and end <= right for left, right in quote_spans),
            ordinal=ordinal,
        )
        for ordinal, (start, end, value) in enumerate(
            _relation_mentions(question, domains)
        )
    )


def _normalize_operator(raw_operator: str) -> str:
    if re.fullmatch(r"至少|不低于|不少于|达到|达|≥|>=", raw_operator):
        return ">="
    if re.fullmatch(r"超过|高于|大于|>|多于", raw_operator):
        return ">"
    if re.fullmatch(r"不超过|不高于|至多|≤|<=", raw_operator):
        return "<="
    if re.fullmatch(r"低于|小于|少于|不足|<", raw_operator):
        return "<"
    return "="


def _metric_bound_source_spans(
    question: str,
    subject: str,
    aliases: Sequence[str],
) -> tuple[list[tuple[str, str, int, int]], list[tuple[int, int]], bool]:
    """Parse one metric completely, including ranges and bare continuations."""

    label = _metric_label(subject, aliases)
    intent = list(
        re.finditer(
            rf"{label}\s*(?:(?:阈值(?:为|是)?|必须|需|需要|应|须|"
            rf"限制(?:为|是)?|要求(?:为|是)?)\s*)?(?:均|都|分别|各)?\s*"
            rf"(?:{_OPERATOR_RE}|\d|在|介于|从|范围)",
            question,
            re.I,
        )
    )
    if not intent:
        return [], [], True
    if _aggregate_metric_modifier(question, subject, aliases):
        return [], [(match.start(), match.end()) for match in intent], False

    bounds = _comparison_matches(question, subject, aliases)
    if not bounds:
        return [], [(match.start(), match.end()) for match in intent], False

    # A label must begin a parsed scalar or range clause.  This detects a
    # dangling operator/range marker rather than letting a later value disguise
    # it as a usable threshold.
    source_starts: list[int] = []
    scalar_pattern = re.compile(
        rf"{label}\s*(?:均|都|分别|各)?\s*(?:{_OPERATOR_RE})\s*"
        r"\d+(?:\.\d+)?",
        re.I,
    )
    ordered_scalar_pattern = re.compile(
        rf"{label}\s*分别\s*(?:{_OPERATOR_RE})\s*\d+(?:\.\d+)?\s*"
        r"(?:和|、|及)\s*\d+(?:\.\d+)?",
        re.I,
    )
    range_pattern = re.compile(
        rf"{label}\s*(?:均|都|分别|各)?\s*(?:在|介于|范围(?:为|是)?|从)\s*"
        r"\d+(?:\.\d+)?\s*(?:至|到|和|与|—|-)\s*\d+(?:\.\d+)?(?:之间)?",
        re.I,
    )
    source_spans = [
        (match.start(), match.end())
        for pattern in (scalar_pattern, ordered_scalar_pattern, range_pattern)
        for match in pattern.finditer(question)
    ]
    source_starts.extend(start for start, _ in source_spans)
    complete = all(any(start <= match.start() < end for start, end in source_spans) for match in intent)

    # _comparison_matches() records the comparison value span.  Retain those
    # exact spans as the evidence that an operator and value came from one
    # parsed source expression.  A bare value or unfinished operator after an
    # otherwise valid expression is terminal instead of being dropped.
    for _, _, _, end in bounds:
        if any(start <= end < finish for start, finish in source_spans):
            # The lower endpoint of a parsed range is followed by its upper
            # endpoint.  It is not an independent bare continuation.
            continue
        following = question[end:]
        if re.match(
            r"\s*(?:[，,；;]\s*)?(?:(?:且|并且|和|、|及|与|以及)\s*)?"
            r"\d+(?:\.\d+)?",
            following,
            re.I,
        ):
            complete = False
        if re.match(
            rf"\s*(?:[，,；;]\s*)?(?:(?:且|并且|和|、|及|与|以及)\s*)?"
            rf"(?:{_OPERATOR_RE})(?!\s*\d)",
            following,
            re.I,
        ):
            complete = False
    return bounds, source_spans, complete


def _relation_metric_owner(
    question: str,
    occurrences: Sequence[RelationOccurrence],
    connective: str,
    relation_span: Optional[tuple[int, int]],
    start: int,
) -> tuple[str, Optional[int]]:
    """Assign a metric span to one relation owner, or leave it unresolved."""

    if not occurrences:
        return "unresolved", None
    if len(occurrences) == 1:
        return "single", occurrences[0].ordinal
    # In an expression such as ``两条边的置信度分别至少0.7和0.8`` or
    # ``两条边置信度均至少0.8``, the ownership marker follows the metric
    # label rather than preceding it.
    if re.match(
        rf"[^，,；;\n]{{0,16}}(?:均|都|分别|各)\s*(?:{_OPERATOR_RE})",
        question[start:],
        re.I,
    ):
        return "per_relation", None
    scope = question[max(0, start - 24) : start]
    if re.search(
        r"分别|逐条|每(?:条|种|个)?\s*(?:关系|边)|各(?:条|种|个)?\s*(?:关系|边)|"
        r"第\s*(?:[一二三四五六七八九十\d]+)\s*(?:条|个)?\s*(?:边|关系|跳|步)|"
        r"前者|后者",
        scope,
    ):
        return "per_relation", None
    if connective == "literal_or":
        if relation_span is not None and (start < relation_span[0] or start >= relation_span[1]):
            return "shared", None
        return "unresolved", None
    preceding = [item for item in occurrences if item.end <= start]
    if preceding:
        owner = preceding[-1]
        between = question[owner.end:start]
        if start - owner.end <= 64 and not re.search(r"[，,。；;]", between):
            return "relation_local", owner.ordinal
    if re.search(r"均|都|每条|每种|各条|各边|共同", scope):
        return "per_relation", None
    return "unresolved", None


def _relation_intent_spans(question: str) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """Record output, ordering, and Top-K spans for route ownership checks."""

    projections = tuple(
        (match.start(), match.end())
        for match in re.finditer(
            rf"(?:返回|显示|输出|列出|给出)(?:[^{_RESULT_CLAUSE_DELIMITERS}]+)",
            question,
        )
    )
    direction = (
        r"升序|降序|正序|倒序|ASC|DESC|"
        r"从高到低|由高到低|从低到高|由低到高|"
        r"从大到小|由大到小|从小到大|由小到大|递增|递减"
    )
    ordering_matches: list[tuple[int, int]] = []
    for pattern in (
        rf"(?:先按|再按|最后按|其次按|然后按|按|依|按照|依照|依据)"
        rf"[^{_RESULT_CLAUSE_DELIMITERS}]{{0,80}}?(?:{direction}|排列|排序)",
        # Natural language often names the result key directly, for example
        # ``目标科室从高到低`` or ``目标科室排序``. It is still an explicit
        # ordering request even without an introductory ``按`` marker.
        rf"[^{_RESULT_CLAUSE_DELIMITERS}]{{0,80}}?(?:{direction}|排列|排序)"
        rf"(?=\s*(?:[{_RESULT_CLAUSE_DELIMITERS}]|$))",
    ):
        ordering_matches.extend(
            (match.start(), match.end())
            for match in re.finditer(pattern, question, re.I)
        )
    selected_ordering: list[tuple[int, int]] = []
    for start, end in sorted(
        ordering_matches,
        key=lambda item: (item[0], -(item[1] - item[0])),
    ):
        if any(
            start < used_end and end > used_start
            for used_start, used_end in selected_ordering
        ):
            continue
        selected_ordering.append((start, end))
    ordering = tuple(sorted(selected_ordering))
    top_k = tuple(
        (match.start(), match.end())
        for match in re.finditer(
            r"(?:前|TOP)\s*(?:\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次|种|人)?",
            question,
            re.I,
        )
    )
    return projections, ordering, top_k


def analyze_relation_request(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> RelationRequest:
    """Build a source-span relation plan before a flat relation filter is used.

    The plan deliberately distinguishes a relation *occurrence* from a relation
    value, keeps connective/quantifier semantics, and requires every recognised
    metric threshold to be parsed.  Callers select a route only after checking
    :meth:`RelationRequest.accepts_flat_predicate` with their actual capacity.
    """

    occurrences = _relation_occurrences(question, domains)
    values = tuple(item.value for item in occurrences)
    errors: list[str] = []
    relation_span = (
        (occurrences[0].start, occurrences[-1].end) if occurrences else None
    )
    set_operation = bool(
        re.search(r"交集|并集|差集|集合|任一|任意|两种|多种", question)
    )
    ordered = bool(
        re.search(
            r"分别|逐条|第\s*(?:[一二三四五六七八九十\d]+)\s*(?:条|个)?\s*"
            r"(?:边|关系|跳|步)|第(?:一|二|三|四|五|六|七|八|九|十|\d)+(?:个|条)|"
            r"前者|后者",
            question,
        )
    )
    required_all = bool(
        re.search(
            r"同时|既有|又有|既存在|又存在|关系集合|关系清单|要求关系|必需关系|"
            r"全部关系|每种关系都|(?:所有|全部|每种|每条|逐条)\s*(?:关系|边)|"
            r"均具有|均存在|具有|具备|包含|包括",
            question,
        )
    )
    connector_pattern = (
        r"\s*[”\"]?\s*(?:关系\s*)?(?:或(?:者)?|"
        r"(?<![A-Za-zＡ-Ｚａ-ｚ])[OＯ][RＲ](?![A-Za-zＡ-Ｚａ-ｚ]))"
        r"\s*(?:关系\s*)?[“\"]?\s*"
    )
    literal_or = bool(occurrences) and len(occurrences) >= 2 and all(
        re.fullmatch(connector_pattern, question[left.end:right.start], re.I)
        for left, right in zip(occurrences, occurrences[1:])
    )
    or_matches = _or_matches(question)
    if literal_or and relation_span is not None and any(
        match.start() < relation_span[0] or match.end() > relation_span[1]
        for match in or_matches
    ):
        literal_or = False
    if not occurrences:
        connective = "none"
    # A cardinality phrase such as ``两种关系`` does not turn an explicit
    # simultaneous-required request into a set operation.  Preserve the
    # required-all owner whenever it names multiple distinct relations; retain
    # collection-first requests with one relation as terminal set operations.
    elif set_operation and not (required_all and len(set(values)) >= 2):
        connective = "set_operation"
    elif ordered:
        connective = "ordered"
    elif required_all:
        connective = "required_all"
    elif len(occurrences) == 1:
        connective = "single"
    elif literal_or and len(set(values)) == len(values):
        connective = "literal_or"
    else:
        connective = "ambiguous"

    if len(occurrences) >= 2 and any(item.quoted for item in occurrences) and any(
        not item.quoted for item in occurrences
    ):
        errors.append("partially quoted relation list")
    if connective == "ambiguous":
        errors.append("unowned relation connective")
    if connective == "literal_or" and len(set(values)) != len(values):
        errors.append("repeated relation value in OR list")
    if or_matches and connective != "literal_or":
        errors.append("unowned disjunction")

    metric_bounds: list[RelationMetricBound] = []
    metric_complete = True
    for metric, subject, aliases in (
        ("confidence", "置信度", ("关系置信度", "边置信度", "confidence")),
        ("count", "观察次数", ("出现次数", "观察计数", "观察量", "边次数", "count")),
    ):
        parsed_bounds, _, complete = _metric_bound_source_spans(
            question,
            subject,
            aliases,
        )
        metric_complete = metric_complete and complete
        for operator, value, start, end in parsed_bounds:
            owner, owner_ordinal = _relation_metric_owner(
                question,
                occurrences,
                connective,
                relation_span,
                start,
            )
            metric_bounds.append(
                RelationMetricBound(
                    metric=metric,
                    operator=operator,
                    value=value,
                    start=start,
                    end=end,
                    owner=owner,
                    owner_ordinal=owner_ordinal,
                )
            )
    if not metric_complete:
        errors.append("unconsumed metric threshold")

    projections, ordering, top_k = _relation_intent_spans(question)
    return RelationRequest(
        occurrences=occurrences,
        connective=connective,
        relation_span=relation_span,
        metric_bounds=tuple(sorted(metric_bounds, key=lambda item: item.start)),
        metric_complete=metric_complete,
        projection_spans=projections,
        ordering_spans=ordering,
        top_k_spans=top_k,
        errors=tuple(dict.fromkeys(errors)),
    )


def _metric_comparison(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> Optional[tuple[str, str]]:
    matches = _comparison_matches(question, subject, aliases)
    if not matches:
        return None
    return matches[0][0], matches[0][1]


def _ordered_thresholds(
    question: str,
    subject: str,
    count: int,
    aliases: Sequence[str] = (),
) -> Optional[list[tuple[str, str]]]:
    """Return one threshold per edge, allowing one explicitly shared bound."""

    matches = _comparison_matches(question, subject, aliases)
    if len(matches) == count:
        return [(operator, value) for operator, value, _, _ in matches]
    if len(matches) == 1 and re.search(
        r"均|都|每条|两条|两边|各边|共同",
        question,
    ):
        operator, value, _, _ = matches[0]
        return [(operator, value)] * count
    return None


def _relation_predicate(
    relations: Sequence[str],
    *,
    alias: str = "",
) -> Optional[str]:
    values = _unique(relations)
    if not values:
        return None
    column = f"{alias}.relation_zh" if alias else "relation_zh"
    if len(values) == 1:
        return f"{column} = '{esc(values[0])}'"
    rendered = ", ".join(f"'{esc(value)}'" for value in values)
    return f"{column} IN ({rendered})"


def _relation_set_allowed(question: str, relations: Sequence[str]) -> bool:
    """Allow a nominal relation only when collection semantics are absent."""

    if re.search(
        r"交集|并集|差集|集合|包含|包括|同时|既有|又有|任一|任意|"
        r"两种|多种|每(?:条|种|个)?\s*(?:关系|边)|各(?:条|种|个)?\s*(?:关系|边)|"
        r"(?:所有|全部|每种|每条|逐条)\s*(?:关系|边)|分别|逐条|"
        r"第\s*(?:[一二三四五六七八九十\d]+)\s*(?:条|个)?\s*(?:边|关系|跳|步)|"
        r"前者|后者",
        question,
    ):
        return False
    return len(relations) == 1 or bool(_or_matches(question))


def _flat_relation_union_span(
    question: str,
    relations: Sequence[str],
    domains: Mapping[str, Sequence[str]],
) -> Optional[tuple[int, int]]:
    """Return a structurally owned literal relation OR span, if any.

    This is deliberately a consumer of :func:`analyze_relation_request`, not a
    second phrase-based parser.  It admits ``IN`` only when every raw relation
    occurrence and every parsed row metric has the one shared flat-union owner.
    """

    request = analyze_relation_request(question, domains)
    if (
        len(relations) < 2
        or tuple(relations) != request.values
        or not request.accepts_flat_predicate()
        or request.connective != "literal_or"
    ):
        return None
    return request.relation_span


def _projection_items(question: str) -> list[str]:
    """Return explicit projection items, repairing ``提及次数`` tokenisation."""

    raw = [
        item.strip(" ：:，,、；")
        for item in answer_items(question)
        if item.strip(" ：:，,、；")
    ]
    # Some explicit RETURN clauses lead with row filters and use the final
    # possessive ``的`` to introduce the real answer list.  Reconstruct that
    # boundary before mapping individual schema fields so a type, relation, or
    # confidence predicate cannot impersonate a projection.
    for index, item in enumerate(raw):
        if "的" not in item:
            continue
        prefix, suffix = item.rsplit("的", 1)
        qualifier_prefix = "、".join((*raw[:index], prefix))
        if (
            suffix
            and re.search(
                r"(?:为|等于|恰好为|至少|不低于|不少于|不超过|不高于|"
                r"大于|小于|高于|低于|>=|<=|>|<)",
                qualifier_prefix,
                re.I,
            )
        ):
            raw = [suffix, *raw[index + 1:]]
        break
    repaired: list[str] = []
    index = 0
    while index < len(raw):
        item = raw[index]
        if item.endswith("提") and index + 1 < len(raw) and raw[index + 1].startswith("次数"):
            repaired.append(item + "及" + raw[index + 1])
            index += 2
            continue
        # ``排名前3`` is a pagination modifier that answer_clause cannot
        # distinguish from a final requested column.  It is not a projection.
        if re.fullmatch(r"(?:排名|排行)?\s*(?:前|top)\s*\d+\s*(?:名|个|条|位)?", item, re.I):
            index += 1
            continue
        repaired.append(item)
        index += 1
    return repaired


def _projection_item_semantic(
    item: str,
    aliases: Mapping[str, Sequence[str]],
) -> Optional[str]:
    """Map one answer item, preferring its most specific schema label."""

    candidates: list[tuple[int, int, str]] = []
    for priority, (semantic, patterns) in enumerate(aliases.items()):
        for pattern in patterns:
            match = re.search(pattern, item, re.I)
            if match:
                candidates.append((len(match.group(0)), -priority, semantic))
    return max(candidates)[2] if candidates else None


def _projection_semantics(
    question: str,
    aliases: Mapping[str, Sequence[str]],
) -> list[str]:
    """Map an explicit projection to schema semantic names in request order.

    Projection labels overlap naturally: ``不同尾实体数`` contains ``尾实体``
    and ``观察次数总和`` contains ``观察次数``.  Prefer the longest matched
    label so a metric is not reduced to a raw edge column.
    """

    result: list[str] = []
    for item in _projection_items(question):
        semantic = _projection_item_semantic(item, aliases)
        if semantic is not None and semantic not in result:
            result.append(semantic)
    return result


def _has_projection(
    question: str,
    required: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
) -> Optional[list[str]]:
    items = _projection_items(question)
    semantics = _projection_semantics(question, aliases)
    # A generic projection-aware compiler must not silently discard a requested
    # answer item it cannot map.  Duplicate semantics are also rejected because
    # this compiler cannot faithfully preserve the requested result shape.
    if (
        not items
        or len(semantics) != len(items)
        or any(name not in semantics for name in required)
    ):
        return None
    return semantics


_EDGE_PROJECTION = {
    "head": (r"头实体", r"(?<!尾)头(?:节点|名称)?", r"source[_ ]?head"),
    "tail": (r"尾实体(?!数|数量|个数|种类)", r"尾(?:节点|名称)?(?!数|数量|个数|种类)", r"target[_ ]?tail"),
    "relation_zh": (r"关系中文名", r"中文关系名", r"关系名"),
    "confidence": (r"置信度", r"confidence"),
    "count": (r"观察次数(?!总和|总数|合计|总量)", r"出现次数", r"观察计数", r"观察量", r"边次数", r"\bcount\b"),
}

_ENTITY_PROJECTION = {
    "name": (r"实体名称", r"实体名", r"(?<!患者)名称", r"\bname\b"),
    "type": (r"实体类型", r"类型", r"\btype\b"),
    "mention_count": (r"提及次数", r"提及数", r"mention_count"),
    "head": (r"头实体", r"头节点"),
    "tail": (r"尾实体(?!数|数量|个数|种类)", r"尾节点(?!数|数量|个数|种类)"),
    "relation_zh": (r"关系中文名", r"中文关系名", r"关系名"),
    "confidence": (r"置信度", r"confidence"),
    "count": (r"观察次数(?!总和|总数|合计|总量)", r"观察计数", r"count"),
}


# Prefer the entity-table column for an explicitly requested tail name or type.
# The generic aliases otherwise correctly retain ``tail`` as the triple column.
_TAIL_JOIN_PROJECTION = {
    **_ENTITY_PROJECTION,
    "name": (r"尾实体(?:名称|名)", *_ENTITY_PROJECTION["name"]),
    "type": (r"尾实体类型", *_ENTITY_PROJECTION["type"]),
}


_GENERIC_METRIC_PROJECTION = {
    "head": (r"头实体", r"头节点", r"source[_ ]?head"),
    "tail": (r"尾实体(?!数|数量|个数|种类)", r"尾节点(?!数|数量|个数|种类)", r"target[_ ]?tail"),
    "relation_zh": (r"关系中文名", r"中文关系名", r"关系名"),
    "confidence": (r"置信度", r"confidence"),
    "count": (r"观察次数(?!总和|总数|合计|总量)", r"观察计数", r"观察量", r"\bcount\b"),
    "distinct_tail_count": (
        r"不同尾实体(?:数|数量|个数)",
        r"尾实体(?:种类)?数",
        r"distinct[_ ]?tail[_ ]?count",
        r"尾覆盖",
    ),
    "distinct_head_count": (
        r"不同头实体(?:数|数量|个数)",
        r"头实体(?:种类)?数",
        r"distinct[_ ]?head[_ ]?count",
    ),
    "total_observations": (
        r"观察次数(?:总和|总数|合计|总量)",
        r"观察总和",
        r"总观察次数",
        r"total[_ ]?observations?",
    ),
    "tail_coverage": (r"尾实体覆盖(?:数|量)?", r"tail[_ ]?coverage"),
    "visit_count": (r"就诊(?:次数|数|量|总数)", r"访问次数", r"visit[_ ]?count"),
    "patient_count": (r"不同患者(?:数|人数|数量)", r"患者(?:数|人数)", r"patient[_ ]?count"),
    "disease_count": (r"不同疾病(?:数|数量)", r"疾病(?:数|数量)", r"disease[_ ]?count"),
    "total_visit_count": (r"总就诊(?:次数|数|量)", r"total[_ ]?visit[_ ]?count"),
    "covered_visit_count": (r"覆盖就诊(?:次数|数|量)", r"图谱覆盖(?:就诊|记录)(?:数|次数)?", r"covered[_ ]?visit"),
    "coverage": (r"覆盖率", r"coverage"),
    "older_patient_count": (r"老年(?:患者)?(?:数|人数|数量)", r"older[_ ]?patient"),
    "relation_kind_count": (r"不同关系(?:种类)?(?:数|数量)", r"关系种类数", r"relation[_ ]?kind[_ ]?count"),
}


def _entity_type(question: str) -> Optional[str]:
    """Read an explicitly typed schema entity, never infer from a noun alone."""

    all_types = "|".join(re.escape(value) for value in _ENTITY_TYPES)
    for match in re.finditer(
        rf"(?:类型(?:为|是|等于)?\s*|type\s*(?:=|is)?\s*)({all_types})",
        question,
        re.I,
    ):
        return _ENTITY_TYPES.get(match.group(1), _ENTITY_TYPES.get(match.group(1).title()))
    for match in re.finditer(
        rf"({all_types})\s*类型(?:实体)?",
        question,
        re.I,
    ):
        return _ENTITY_TYPES.get(match.group(1), _ENTITY_TYPES.get(match.group(1).title()))
    for match in re.finditer(
        rf"({all_types})\s*(?:图谱)?实体",
        question,
        re.I,
    ):
        return _ENTITY_TYPES.get(match.group(1), _ENTITY_TYPES.get(match.group(1).title()))
    # Chinese schema nouns are accepted only when followed by an explicit
    # entity marker; this prevents a relation such as “治疗药物” becoming a
    # type filter accidentally.
    for chinese, normalized in (
        ("疾病", "Disease"),
        ("药物", "Drug"),
        ("检查", "Exam"),
        ("症状", "Symptom"),
        ("治疗方式", "Treatment"),
        ("科室", "Department"),
    ):
        if re.search(rf"{chinese}(?:类型)?实体", question):
            return normalized
    return None


def _position(question: str) -> Optional[str]:
    """Return an explicitly requested edge position: ``head`` or ``tail``."""

    head = bool(
        re.search(
            r"作为头实体|以(?:该)?实体为头|头实体(?:的)?(?:关系|边)|"
            r"头部(?:关系|边)|出边|外出边|头位置|head\s*(?:side|position)",
            question,
            re.I,
        )
    )
    tail = bool(
        re.search(
            r"作为尾实体|以(?:该)?实体为尾|尾实体(?:的)?(?:关系|边|连接|关联|联结)|"
            r"尾部(?:关系|边)|入边|进入边|尾位置|tail\s*(?:side|position)",
            question,
            re.I,
        )
    )
    if head and not tail:
        return "head"
    if tail and not head:
        return "tail"
    # A lone explicit position noun is sufficient for a position-bound join;
    # requiring both nouns would make “关系的尾实体” unnecessarily abstain.
    # When both occur, the request is ambiguous and remains fail-closed.
    if not head and not tail:
        has_head_noun = bool(re.search(r"头实体|头节点|head", question, re.I))
        has_tail_noun = bool(re.search(r"尾实体|尾节点|tail", question, re.I))
        if has_head_noun and not has_tail_noun:
            return "head"
        if has_tail_noun and not has_head_noun:
            return "tail"
    return None


def _chain_direction(question: str) -> bool:
    """Require text that states the first edge's head-to-tail direction."""

    return bool(
        re.search(
            r"头实体.{0,20}尾实体|尾实体.{0,20}下一跳(?:头|起点)|"
            r"从头.{0,20}到尾|出边|外出边|沿着|链路方向|"
            r"first\s+hop.{0,20}(?:head|tail)",
            question,
            re.I,
        )
    )


def _top_k(question: str) -> Optional[int]:
    value = limit(question)
    if value is not None:
        return value
    match = re.search(
        r"\bTOP\s*(\d+)\b|前\s*(\d+)|(?:最大|最多|最高).{0,6}?(\d+)\s*(?:个|名|位)",
        question,
        re.I,
    )
    if match:
        return int(next(group for group in match.groups() if group is not None))
    return None


def _flat_relation_route_request(
    question: str,
    domains: Mapping[str, Sequence[str]],
    *,
    allow_top_k: bool,
    require_relation: bool = True,
) -> Optional[RelationRequest]:
    """Admit a fixed-order row route only after every relation span is owned.

    These routes lower one row filter (or a proven literal ``IN`` filter).  They
    must inspect raw occurrences before any caller removes duplicates through
    :func:`_relations`: collection/ordinal wording can otherwise turn a
    first-edge or multi-hop request into an unrestricted flat result.
    """

    request = analyze_relation_request(question, domains)
    if request.has_relation_intent:
        if not request.accepts_flat_predicate():
            return None
    elif require_relation or request.errors or not request.metric_complete:
        return None

    # The route-specific compilers below expose one stable ordering.  None of
    # them can faithfully lower a caller-provided ORDER BY expression.
    if request.ordering_spans:
        return None

    top = _top_k(question)
    if len(request.top_k_spans) > 1:
        return None
    if request.top_k_spans:
        if not allow_top_k or top is None:
            return None
    elif top is not None and not allow_top_k:
        return None
    return request


def _fixed_result_modifiers_are_absent(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Return whether a fixed-shape route can consume result modifiers.

    The specialised aggregate and graph routes below hard-code a stable result
    order and do not implement paging.  Their default ordering is not an owner
    for a caller-supplied ordering or Top-K request, even when it happens to
    look similar.  Use the same span-based request analysis as the row routes
    so every recognised modifier causes terminal abstention.
    """

    request = analyze_relation_request(question, domains)
    return (
        not request.ordering_spans
        and not request.top_k_spans
        and _top_k(question) is None
    )


def _has_ordinal_or_chain_relation_semantics(question: str) -> bool:
    """Return whether relation order denotes edge roles rather than a set."""

    return bool(
        re.search(
            r"两跳|两步|二跳|两层|关系链|风险链|路径|"
            r"第\s*(?:[一二三四五六七八九十\d]+)\s*(?:条|个)?\s*"
            r"(?:边|关系|跳|步)|"
            r"(?:首|第一|第二|下一|后续)\s*(?:跳|步)|前者|后者|"
            r"first\s+hop|second\s+hop|next\s+hop|(?:risk[- ]?)?chain|path",
            question,
            re.I,
        )
    )


def _required_all_relation_route_request(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[RelationRequest]:
    """Admit an EXISTS/HAVING required-all route with complete ownership.

    Required-all routes can bind one declared relation to each EXISTS arm (or
    to each term in a canonical ``HAVING`` proof).  A later repeated mention is
    allowed only when it uniquely owns a relation-local metric qualifier; a
    duplicate declaration cannot be treated as an independently identified
    edge.  Their output ordering is fixed and they do not implement paging, so
    those modifiers remain terminal rather than being replaced by a default
    result shape.
    """

    request = analyze_relation_request(question, domains)
    if (
        request.connective not in {"required_all", "ordered"}
        or len(_unique(request.values)) < 2
        or request.errors
        or not request.metric_complete
        or len(request.projection_spans) > 1
        or request.ordering_spans
        or request.top_k_spans
        or _top_k(question) is not None
    ):
        return None
    # ``分别`` can own per-relation threshold values, whereas an ordinal or
    # chain assigns edge roles that an unordered EXISTS/HAVING proof cannot
    # preserve.  Do not let the presence of ``分别`` mask that distinction.
    if request.connective == "ordered" and (
        "分别" not in question or _has_ordinal_or_chain_relation_semantics(question)
    ):
        return None
    if any(
        bound.owner not in {"relation_local", "per_relation"}
        for bound in request.metric_bounds
    ):
        return None
    qualifier_occurrences = {
        bound.owner_ordinal
        for bound in request.metric_bounds
        if bound.owner == "relation_local" and bound.owner_ordinal is not None
    }
    seen: set[str] = set()
    for occurrence in request.occurrences:
        if occurrence.value in seen and occurrence.ordinal not in qualifier_occurrences:
            return None
        seen.add(occurrence.value)
    return request


def _having_threshold(question: str, phrases: Sequence[str]) -> Optional[tuple[str, int]]:
    # Grouped family routes lower one conjunctive HAVING predicate.  A
    # disjunction cannot be represented by retaining only the first endpoint,
    # including a full-width OR token, so it must remain terminal here.
    if _or_matches(question):
        return None
    ordered_phrases = sorted({str(phrase) for phrase in phrases}, key=len, reverse=True)
    label = "(?:" + "|".join(re.escape(phrase) for phrase in ordered_phrases) + ")"
    match = re.search(
        rf"{label}\s*(?:至少|不低于|不少于|达到|达|≥|>=|超过|多于|大于|>|"
        rf"至多|不超过|≤|<=|低于|小于|<)\s*(\d+)",
        question,
    )
    if not match:
        match = re.search(
            rf"(?:至少|不低于|不少于|达到|达|≥|>=|超过|多于|大于|>)\s*(\d+)\s*"
            rf"(?:个|种)?\s*{label}",
            question,
        )
    if not match:
        return None
    text = match.group(0)
    operator = ">=" if re.search(r"至少|不低于|不少于|达到|达|≥|>=", text) else ">"
    if re.search(r"至多|不超过|≤|<=|低于|小于|<", text):
        operator = "<=" if re.search(r"至多|不超过|≤|<=", text) else "<"
    number_match = re.search(r"\d+", text)
    if number_match is None:
        return None
    return operator, int(number_match.group(0))


def _mention_range(question: str) -> Optional[tuple[int, int]]:
    patterns = (
        r"提及次数(?:在|介于|为)?\s*(\d+)\s*(?:至|到|—|-)\s*(\d+)",
        r"提及次数(?:在|介于)?\s*(\d+)\s*和\s*(\d+)(?:之间)?",
        r"提及次数从\s*(\d+)\s*到\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            first, second = int(match.group(1)), int(match.group(2))
            if first <= second:
                return first, second
            return None
    return None


def _patient_predicates(question: str, domains: Mapping[str, Sequence[str]]) -> list[str]:
    predicates = age_predicates(question, alias="p")
    gender = gender_predicate(question, alias="p")
    if gender:
        predicates.append(gender)
    predicates.extend(region_predicates(question, domains, alias="p"))
    return predicates


def _has_clinical_scope(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Whether a request needs a patient or visit-capable graph route.

    KG-only routes cannot reinterpret a patient cohort, visit record, or date
    scope as a global graph predicate.  Dispatch such requests only to routes
    that join the clinical tables and explicitly consume their qualifiers.
    """

    return bool(
        _patient_predicates(question, domains)
        or grounded_predicates(question, domains, alias="v")
        or dates(question)
        or re.search(
            r"患者|病人|就诊|病历|处方|住院|诊疗|主诊断|实际记录|"
            r"费用|花费|成本",
            question,
        )
    )


def _visit_predicates(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[list[str]]:
    predicates = grounded_predicates(question, domains, alias="v")
    date_values = dates(question)
    if len(date_values) > 2:
        return None
    if len(date_values) == 2:
        predicates.extend(
            [
                f"v.visit_date >= '{date_values[0]}'",
                f"v.visit_date <= '{date_values[1]}'",
            ]
        )
    elif date_values:
        rendered_dates = date_predicates(question, alias="v")
        if not rendered_dates:
            return None
        predicates.extend(rendered_dates)
    cost_bounds = _visit_metric_bounds(question, "费用", aliases=("cost",))
    stay_bounds = _visit_metric_bounds(
        question,
        "住院天数",
        aliases=("住院时长", "los_days"),
    )
    if cost_bounds is None or stay_bounds is None:
        return None
    predicates.extend(f"v.cost {operator} {value}" for operator, value in cost_bounds)
    predicates.extend(
        f"v.los_days {operator} {value}" for operator, value in stay_bounds
    )
    return list(dict.fromkeys(predicates))


def _group_dimension(question: str) -> Optional[tuple[str, str]]:
    if re.search(r"按(?:患者)?地区|按区域|各地区|每个地区|分地区", question):
        return "region", "p.region"
    if re.search(r"按(?:就诊)?科室|各科室|每个科室|分科室", question):
        return "department", "v.department"
    if re.search(r"按(?:主)?疾病|按诊断|各疾病|每种疾病|分疾病", question):
        return "disease", "v.disease"
    if re.search(r"按性别|各性别|分性别", question):
        return "gender", "p.gender"
    return None


def _group_projection_name(question: str, dimension: str) -> bool:
    patterns = {
        "region": r"地区|区域",
        "department": r"科室",
        "disease": r"疾病|诊断",
        "gender": r"性别",
    }
    items = _projection_items(question)
    return any(re.search(patterns[dimension], item) for item in items)


def _metric_semantics(question: str) -> list[str]:
    return _projection_semantics(question, _GENERIC_METRIC_PROJECTION)


def _relation_threshold_predicate(
    alias: str,
    relation: str,
    confidence: Optional[tuple[str, str]],
    count: Optional[tuple[str, str]],
) -> list[str]:
    predicates = [f"{alias}.relation_zh = '{esc(relation)}'"]
    if confidence:
        predicates.append(f"{alias}.confidence {confidence[0]} {confidence[1]}")
    if count:
        predicates.append(f"{alias}.count {count[0]} {count[1]}")
    return predicates


def _metric_mentioned(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> bool:
    label = _metric_label(subject, aliases)
    return bool(re.search(label, question, re.I))


def _metric_threshold_intent(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> bool:
    """Recognise a metric-bound request, including unsupported forms.

    A compiler must distinguish a requested output column from a malformed
    predicate.  This narrow detector only treats a metric as a bound when an
    operator, number, or range marker is adjacent to the metric label.
    """

    label = _metric_label(subject, aliases)
    return bool(
        re.search(
            rf"{label}\s*(?:(?:阈值(?:为|是)?|必须|需|需要|应|须|"
            rf"限制(?:为|是)?|要求(?:为|是)?)\s*)?(?:均|都|分别|各)?\s*"
            rf"(?:{_OPERATOR_RE}|\d|在|介于|从)",
            question,
            re.I,
        )
        or re.search(
            rf"(?:{_OPERATOR_RE})\s*\d+(?:\.\d+)?(?:的)?\s*{label}",
            question,
            re.I,
        )
    )


def _has_shared_relation_scope(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> bool:
    """Require explicit wording before copying one bound to every edge."""

    label = _metric_label(subject, aliases)
    scope = (
        r"(?:均|都|每条(?:边|关系)?|每种关系|各(?:条|边|关系)|"
        r"所有(?:边|关系)|全部(?:边|关系))"
    )
    bound_start = rf"(?:{_OPERATOR_RE}|在|介于|从)"
    return bool(
        re.search(rf"{label}\s*{scope}\s*{bound_start}", question, re.I)
        or re.search(rf"{scope}.{{0,12}}{label}\s*{bound_start}", question, re.I)
    )


def _has_ordered_relation_scope(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> bool:
    """Recognise the explicit ``分别`` form for relation-order bounds."""

    label = _metric_label(subject, aliases)
    return bool(re.search(rf"{label}\s*分别\s*(?:{_OPERATOR_RE})", question, re.I))


def _unlabeled_bound_follows(question: str, end: int) -> bool:
    """Reject a bare follow-up value that has no metric or comparison operator."""

    return bool(
        re.match(
            r"\s*(?:[，,；;]\s*)?(?:且|并且|和|、|及)\s*\d+(?:\.\d+)?",
            question[end:],
            re.I,
        )
    )


def _incomplete_explicit_bound_follows(question: str, end: int) -> bool:
    """Reject an unfinished explicit continuation instead of dropping it."""

    following = re.match(
        rf"\s*(?:[，,；;]\s*)?(?:且|并且|和|、|及)\s*(?:{_OPERATOR_RE})",
        question[end:],
        re.I,
    )
    if following is None:
        return False
    return re.match(r"\s*\d+(?:\.\d+)?", question[end + following.end():]) is None


def _visit_metric_bounds(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
) -> Optional[list[tuple[str, str]]]:
    """Return every faithful conjunctive visit-metric bound or abstain."""

    if _aggregate_metric_modifier(question, subject, aliases):
        return None
    matches = _comparison_matches(question, subject, aliases)
    if not matches:
        return None if _metric_threshold_intent(question, subject, aliases) else []
    if _or_matches(question):
        # Visit predicates never own a relation-list OR.  Searching the whole
        # request prevents a distant disjunction from being rewritten as AND.
        return None
    if any(
        _unlabeled_bound_follows(question, end)
        or _incomplete_explicit_bound_follows(question, end)
        for _, _, _, end in matches
    ):
        return None
    return [(operator, value) for operator, value, _, _ in matches]


def _single_relation_metric_bounds(
    question: str,
    subject: str,
    aliases: Sequence[str] = (),
    *,
    allowed_or_span: Optional[tuple[int, int]] = None,
) -> Optional[list[tuple[str, str]]]:
    """Parse all unambiguous bounds for one grounded triple relation.

    ``allowed_or_span`` is reserved for a separately validated literal relation
    list in a flat-``IN`` route.  Every other disjunction remains unsafe because
    its metric ownership cannot be rendered as conjunctive row predicates.
    """

    if _aggregate_metric_modifier(question, subject, aliases):
        return None
    matches = _comparison_matches(question, subject, aliases)
    if not matches:
        return None if _metric_threshold_intent(question, subject, aliases) else []
    for match in _or_matches(question):
        if (
            allowed_or_span is None
            or match.start() < allowed_or_span[0]
            or match.end() > allowed_or_span[1]
        ):
            return None
    if _has_ordered_relation_scope(question, subject, aliases):
        # ``分别`` requires multiple identified edges and must never be
        # collapsed onto the single edge handled by a tail join.
        return None
    if any(
        _unlabeled_bound_follows(question, end)
        or _incomplete_explicit_bound_follows(question, end)
        for _, _, _, end in matches
    ):
        return None
    return [(operator, value) for operator, value, _, _ in matches]


def _local_relation_metric_bounds(
    question: str,
    relations: Sequence[str],
    matches: Sequence[tuple[str, str, int, int]],
    domains: Mapping[str, Sequence[str]],
) -> Optional[list[list[tuple[str, str]]]]:
    """Map named bounds to the relation immediately preceding each bound.

    More than one condition may belong to one relation (such as a lower and an
    upper confidence limit).  Repeated relation values cannot be identified by
    name alone, so that shape deliberately defers to ordinal hop wording.
    """

    if len(set(relations)) != len(relations):
        return None
    mentions = _relation_mentions(question, domains)
    bound_by_relation: dict[str, list[tuple[str, str]]] = {}
    for operator, value, start, _ in matches:
        preceding = [mention for mention in mentions if mention[1] <= start]
        if not preceding:
            return None
        _, relation_end, relation = preceding[-1]
        # Multiple predicates (for example confidence and count lower/upper
        # limits) may follow one named relation.  They remain local only while
        # no clause separator or another relation can have ended that scope.
        between = question[relation_end:start]
        if start - relation_end > 64 or re.search(r"[，,。；;]", between):
            return None
        if relation not in relations:
            return None
        bound_by_relation.setdefault(relation, []).append((operator, value))
    if set(bound_by_relation) != set(relations):
        return None
    return [bound_by_relation[relation] for relation in relations]


def _ordinal_edge_index(question: str, start: int, edge_count: int) -> Optional[int]:
    """Find the explicit hop/edge ordinal immediately governing a bound."""

    ordinal_patterns: tuple[tuple[str, int], ...] = (
        (r"第\s*(?:一|1|首)\s*(?:条|个)?\s*(?:边|关系|跳|步)", 0),
        (r"第\s*(?:二|2)\s*(?:条|个)?\s*(?:边|关系|跳|步)", 1),
    )
    candidates: list[tuple[int, int]] = []
    for pattern, index in ordinal_patterns:
        if index >= edge_count:
            continue
        for match in re.finditer(pattern, question, re.I):
            # A metric synonym such as ``边置信度`` shares the final ``边``
            # character with ``第一条边``.  Treat that one-character overlap
            # as an immediately governing ordinal, but never search forward.
            if match.end() - 1 <= start and start - match.end() <= 18:
                candidates.append((match.end(), index))
    return max(candidates)[1] if candidates else None


def _edge_metric_bounds(
    question: str,
    *,
    edge_count: int,
    subject: str,
    aliases: Sequence[str] = (),
    relations: Sequence[str] = (),
    domains: Optional[Mapping[str, Sequence[str]]] = None,
) -> Optional[list[list[tuple[str, str]]]]:
    """Bind all edge-metric bounds without weakening their stated scope.

    Shared bounds apply to each edge only with an explicit shared marker.
    Distinct values need either an ordinal edge label, an ordered ``分别`` form,
    or a locally named grounded relation.  Any other ownership is ambiguous.
    """

    if edge_count <= 0 or _aggregate_metric_modifier(question, subject, aliases):
        return None
    matches = _comparison_matches(question, subject, aliases)
    if not matches:
        if _metric_mentioned(question, subject, aliases):
            return None
        return [[] for _ in range(edge_count)]
    if _or_matches(question):
        return None
    if any(
        _incomplete_explicit_bound_follows(question, end)
        for _, _, _, end in matches
    ):
        return None
    if not _has_ordered_relation_scope(question, subject, aliases) and any(
        _unlabeled_bound_follows(question, end) for _, _, _, end in matches
    ):
        return None
    bounds = [(operator, value) for operator, value, _, _ in matches]
    if _has_shared_relation_scope(question, subject, aliases):
        return [list(bounds) for _ in range(edge_count)]

    ordinal_groups: list[list[tuple[str, str]]] = [[] for _ in range(edge_count)]
    ordinal_found = False
    for operator, value, start, _ in matches:
        index = _ordinal_edge_index(question, start, edge_count)
        if index is None:
            ordinal_groups = []
            break
        ordinal_found = True
        ordinal_groups[index].append((operator, value))
    if ordinal_found and ordinal_groups and all(ordinal_groups):
        return ordinal_groups

    if (
        _has_ordered_relation_scope(question, subject, aliases)
        and len(matches) == edge_count
    ):
        return [[(operator, value)] for operator, value, _, _ in matches]

    if relations and domains is not None:
        named = _local_relation_metric_bounds(question, relations, matches, domains)
        if named is not None:
            return named
    return None


def _required_relation_metric_bounds(
    question: str,
    relations: Sequence[str],
    subject: str,
    aliases: Sequence[str],
    domains: Mapping[str, Sequence[str]],
) -> Optional[list[list[tuple[str, str]]]]:
    """Bind every requested edge condition to its required relation."""

    return _edge_metric_bounds(
        question,
        edge_count=len(relations),
        subject=subject,
        aliases=aliases,
        relations=relations,
        domains=domains,
    )


def _compile_relation_edge_filter(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    request = _flat_relation_route_request(
        question,
        domains,
        allow_top_k=True,
    )
    if request is None:
        return None
    relations = list(request.values)
    if not relations or not re.search(r"三元组|图谱边|关系", question):
        return None
    union_span = (
        request.relation_span if request.connective == "literal_or" else None
    )
    top = _top_k(question)
    if top is None and re.search(
        r"(?:排名|排行)?\s*(?:前|TOP)\s*(?:[一二三四五六七八九十几若干]|$)",
        question,
        re.I,
    ):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("confidence",),
        allowed_or_span=union_span,
    )
    observation_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("观察计数", "观察量", "count"),
        allowed_or_span=union_span,
    )
    if not confidence_bounds or not observation_bounds:
        return None
    projection = _has_projection(
        question,
        ("head", "tail", "confidence", "count"),
        _EDGE_PROJECTION,
    )
    if projection is None:
        return None
    relation_predicate = _relation_predicate(relations)
    if relation_predicate is None:
        return None
    where = [relation_predicate]
    where.extend(
        f"confidence {operator} {value}" for operator, value in confidence_bounds
    )
    where.extend(
        f"count {operator} {value}" for operator, value in observation_bounds
    )
    select = ", ".join(projection)
    confidence_direction = "ASC" if confidence_bounds[0][0] in {"<", "<="} else "DESC"
    count_direction = "ASC" if observation_bounds[0][0] in {"<", "<="} else "DESC"
    order = (
        f"confidence {confidence_direction}, count {count_direction}, "
        "head ASC, tail ASC"
    )
    sql = f"SELECT {select} FROM kg_triples WHERE {' AND '.join(where)} ORDER BY {order}"
    return f"{sql} LIMIT {top}" if top is not None else sql


def _compile_tail_entity_join(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not re.search(r"尾实体", question) or not re.search(
        r"连接|关联|联结|join|实体表",
        question,
        re.I,
    ):
        return None
    request = _flat_relation_route_request(
        question,
        domains,
        allow_top_k=False,
        require_relation=False,
    )
    if request is None:
        return None
    relations = list(request.values)
    union_span = (
        request.relation_span if request.connective == "literal_or" else None
    )
    mention_bounds = _single_relation_metric_bounds(
        question,
        "提及次数",
        ("提及数", "mention_count"),
        allowed_or_span=union_span,
    )
    if not mention_bounds or _position(question) != "tail":
        return None
    if not re.search(r"三元组|图谱边|kg_triples|kg_entities", question, re.I):
        return None
    projection = _has_projection(
        question,
        ("mention_count",),
        _TAIL_JOIN_PROJECTION,
    )
    if projection is None or not any(
        item in projection for item in ("name", "head", "tail", "relation_zh", "confidence", "count")
    ):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("关系置信度", "confidence"),
        allowed_or_span=union_span,
    )
    count_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("出现次数", "观察计数", "观察量", "边次数", "count"),
        allowed_or_span=union_span,
    )
    if confidence_bounds is None or count_bounds is None:
        return None
    entity_type = _entity_type(question)
    select_map = {
        "name": "e.name",
        "type": "e.type",
        "mention_count": "e.mention_count",
        "head": "t.head",
        "tail": "t.tail",
        "relation_zh": "t.relation_zh",
        "confidence": "t.confidence",
        "count": "t.count",
    }
    select = ", ".join(select_map[item] for item in projection)
    where = [
        f"e.mention_count {operator} {value}"
        for operator, value in mention_bounds
    ]
    relation_predicate = _relation_predicate(relations, alias="t")
    if relation_predicate:
        where.append(relation_predicate)
    where.extend(f"t.confidence {operator} {value}" for operator, value in confidence_bounds)
    where.extend(f"t.count {operator} {value}" for operator, value in count_bounds)
    if entity_type:
        where.append(f"e.type = '{entity_type}'")
    mention_direction = "ASC" if mention_bounds[0][0] in {"<", "<="} else "DESC"
    return (
        f"SELECT {select} FROM kg_triples AS t JOIN kg_entities AS e "
        "ON e.name = t.tail WHERE "
        f"{' AND '.join(where)} ORDER BY e.mention_count {mention_direction}, "
        "e.name ASC, t.head ASC, t.tail ASC"
    )


def _compile_per_head_relation_topk(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not (
        re.search(r"不同尾实体|尾实体(?:种类)?数", question)
        and re.search(r"观察次数(?:总和|总数|合计)|观察总和|总观察次数", question)
        and re.search(r"每个头实体|按头实体|按头|各头实体", question)
    ):
        return None
    request = _flat_relation_route_request(
        question,
        domains,
        allow_top_k=True,
    )
    if request is None:
        return None
    relations = list(request.values)
    if not relations:
        return None
    union_span = (
        request.relation_span if request.connective == "literal_or" else None
    )
    top = _top_k(question)
    if top is None:
        return None
    projection = _has_projection(
        question,
        ("head", "distinct_tail_count", "total_observations"),
        _GENERIC_METRIC_PROJECTION,
    )
    if projection is None or any(
        item not in {"head", "distinct_tail_count", "total_observations"}
        for item in projection
    ):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("confidence",),
        allowed_or_span=union_span,
    )
    observation_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("观察计数", "观察量", "count"),
        allowed_or_span=union_span,
    )
    if confidence_bounds is None or observation_bounds is None:
        return None
    relation_predicate = _relation_predicate(relations)
    if relation_predicate is None:
        return None
    where = [relation_predicate]
    where.extend(
        f"confidence {operator} {value}" for operator, value in confidence_bounds
    )
    where.extend(
        f"count {operator} {value}" for operator, value in observation_bounds
    )
    select_map = {
        "head": "head",
        "distinct_tail_count": "COUNT(DISTINCT tail) AS distinct_tail_count",
        "total_observations": "SUM(count) AS total_observations",
    }
    select = ", ".join(select_map[item] for item in projection if item in select_map)
    if not select:
        return None
    return (
        f"SELECT {select} FROM kg_triples WHERE {' AND '.join(where)} "
        "GROUP BY head ORDER BY distinct_tail_count DESC, "
        f"total_observations DESC, head ASC LIMIT {top}"
    )


def _compile_entity_required_relations(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    request = _required_all_relation_route_request(question, domains)
    if request is None:
        return None
    relations = _unique(request.values)
    if not re.search(
        r"同时|既有|又有|两种关系|关系集合|要求关系|必需关系|均具有|具有|具备|包含",
        question,
    ):
        return None
    if not re.search(r"实体|知识图谱", question) or re.search(r"就诊|患者队列", question):
        return None
    position = _position(question)
    if position is None:
        return None
    entity_type = _entity_type(question)
    if entity_type is None:
        return None
    confidence_bounds = _required_relation_metric_bounds(
        question,
        relations,
        "置信度",
        ("confidence",),
        domains,
    )
    count_bounds = _required_relation_metric_bounds(
        question,
        relations,
        "观察次数",
        ("出现次数", "观察计数", "观察量", "边次数", "count"),
        domains,
    )
    if (
        confidence_bounds is None
        or count_bounds is None
        or not all(confidence_bounds)
    ):
        return None
    projection = _has_projection(
        question,
        ("name",),
        _ENTITY_PROJECTION,
    )
    if projection is None:
        return None
    select_map = {
        "name": "e.name",
        "type": "e.type",
        "mention_count": "e.mention_count",
    }
    if any(item not in select_map for item in projection):
        return None
    select = ", ".join(select_map[item] for item in projection)
    exists: list[str] = []
    for index, (relation, relation_confidence, relation_counts) in enumerate(
        zip(relations, confidence_bounds, count_bounds),
        start=1,
    ):
        predicates = [
            f"t{index}.{position} = e.name",
            f"t{index}.relation_zh = '{esc(relation)}'",
        ]
        predicates.extend(
            f"t{index}.confidence {operator} {value}"
            for operator, value in relation_confidence
        )
        predicates.extend(
            f"t{index}.count {operator} {value}"
            for operator, value in relation_counts
        )
        exists.append(
            f"EXISTS (SELECT 1 FROM kg_triples AS t{index} WHERE "
            + " AND ".join(predicates)
            + ")"
        )
    order = "e.name ASC"
    if "mention_count" in projection:
        order = "e.mention_count DESC, e.name ASC"
    return (
        f"SELECT {select} FROM kg_entities AS e WHERE e.type = '{entity_type}' "
        f"AND {' AND '.join(exists)} ORDER BY {order}"
    )


def _compile_tail_coverage(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not (
        re.search(r"同(?:一)?类型平均|同类型的平均|type\s*average", question, re.I)
        and re.search(r"按关系|每种关系|各关系|关系覆盖", question)
        and re.search(r"尾实体", question)
    ):
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    having = _having_threshold(
        question,
        ("不同头实体", "头实体数", "头实体数量", "覆盖头实体"),
    )
    if having is None:
        return None
    projection = _has_projection(
        question,
        ("relation_zh", "distinct_tail_count", "distinct_head_count"),
        _GENERIC_METRIC_PROJECTION,
    )
    if projection is None:
        return None
    relations = _relations(question, domains)
    if len(relations) > 1 or (relations and not _relation_set_allowed(question, relations)):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("关系置信度", "confidence"),
    )
    count_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("出现次数", "观察计数", "观察量", "边次数", "count"),
    )
    if confidence_bounds is None or count_bounds is None:
        return None
    where = [
        "e.mention_count > (SELECT AVG(e2.mention_count) FROM kg_entities AS e2 "
        "WHERE e2.type = e.type)"
    ]
    if relations:
        where.append(f"t.relation_zh = '{esc(relations[0])}'")
    where.extend(
        f"t.confidence {operator} {value}"
        for operator, value in confidence_bounds
    )
    where.extend(
        f"t.count {operator} {value}"
        for operator, value in count_bounds
    )
    entity_type = _entity_type(question)
    if entity_type:
        where.append(f"e.type = '{entity_type}'")
    select_map = {
        "relation_zh": "t.relation_zh",
        "distinct_tail_count": "COUNT(DISTINCT t.tail) AS distinct_tail_count",
        "tail_coverage": "COUNT(DISTINCT t.tail) AS distinct_tail_count",
        "distinct_head_count": "COUNT(DISTINCT t.head) AS distinct_head_count",
    }
    if any(item not in select_map for item in projection):
        return None
    select = ", ".join(select_map[item] for item in projection)
    group = "GROUP BY t.relation_zh"
    op, value = having
    return (
        f"SELECT {select} FROM kg_triples AS t JOIN kg_entities AS e "
        f"ON e.name = t.tail WHERE {' AND '.join(where)} {group} "
        f"HAVING COUNT(DISTINCT t.head) {op} {value} "
        "ORDER BY distinct_tail_count DESC, distinct_head_count DESC, "
        "t.relation_zh ASC"
    )


def _chain_projection(question: str) -> Optional[list[str]]:
    items = _projection_items(question)
    if not items:
        return None
    semantics: list[str] = []
    for item in items:
        if re.search(r"源(?:疾病|头实体)?|起点|source", item, re.I):
            candidate = "source"
        elif re.search(r"中间|第一跳尾|并发症|中间实体|intermediate", item, re.I):
            candidate = "middle"
        elif re.search(r"目标|终点|第二跳尾|最终尾|科室|target", item, re.I):
            candidate = "target"
        elif re.search(r"头实体", item):
            candidate = "source" if "source" not in semantics else "middle"
        elif re.search(r"尾实体", item):
            candidate = "middle" if "middle" not in semantics else "target"
        else:
            continue
        if candidate not in semantics:
            semantics.append(candidate)
    if (
        len(semantics) != len(items)
        or not all(name in semantics for name in ("source", "middle", "target"))
    ):
        return None
    return semantics


def _chain_relations(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[list[str]]:
    """Preserve the two hop roles even when their relation values repeat."""

    mentions = [value for _, _, value in _relation_mentions(question, domains)]
    if len(mentions) == 2:
        return mentions
    relations = _relations(question, domains)
    return relations if len(relations) == 2 else None


def _chain_thresholds(
    question: str,
    relations: Sequence[str],
    domains: Mapping[str, Sequence[str]],
) -> Optional[list[list[tuple[str, str]]]]:
    bounds = _edge_metric_bounds(
        question,
        edge_count=len(relations),
        subject="置信度",
        aliases=("confidence",),
        relations=relations,
        domains=domains,
    )
    if bounds is None or not all(bounds):
        return None
    return bounds


def _compile_two_hop_chain(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not re.search(r"两跳|两步|二跳|两层|关系链|路径", question):
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    relations = _chain_relations(question, domains)
    if relations is None or len(relations) != 2:
        return None
    if not _chain_direction(question):
        return None
    projection = _chain_projection(question)
    if projection is None:
        return None
    confidence_bounds = _chain_thresholds(question, relations, domains)
    count_bounds = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="观察次数",
        aliases=("观察计数", "观察量", "count"),
        relations=relations,
        domains=domains,
    )
    if confidence_bounds is None or count_bounds is None:
        return None
    select_map = {
        "source": "t1.head AS source_head",
        "middle": "t1.tail AS intermediate",
        "target": "t2.tail AS target_tail",
    }
    # Use domain language in aliases when the question requests it; otherwise
    # stable graph-role aliases make the result reusable by callers.
    if re.search(r"源疾病", question):
        select_map["source"] = "t1.head AS source_disease"
    if re.search(r"并发症", question):
        select_map["middle"] = "t1.tail AS complication"
    if re.search(r"科室", question):
        select_map["target"] = "t2.tail AS department"
    select = ", ".join(select_map[item] for item in projection)
    where = [
        f"t1.relation_zh = '{esc(relations[0])}'",
        f"t2.relation_zh = '{esc(relations[1])}'",
    ]
    for alias, confidence, counts in zip(
        ("t1", "t2"),
        confidence_bounds,
        count_bounds,
    ):
        where.extend(
            f"{alias}.confidence {operator} {value}"
            for operator, value in confidence
        )
        where.extend(
            f"{alias}.count {operator} {value}"
            for operator, value in counts
        )
    order = "t1.head ASC, t1.tail ASC, t2.tail ASC"
    return (
        f"SELECT {select} FROM kg_triples AS t1 JOIN kg_triples AS t2 "
        f"ON t2.head = t1.tail WHERE {' AND '.join(where)} "
        f"ORDER BY {order}"
    )


def _pair_projection(question: str) -> Optional[list[str]]:
    items = _projection_items(question)
    if not items:
        return None
    semantics: list[str] = []
    for item in items:
        if re.search(r"疾病对|第一个疾病|疾病(?:一|甲|1|A)|disease[_ ]?a", item, re.I):
            candidate = "disease_a"
        elif re.search(r"第二个疾病|疾病(?:二|乙|2|B)|disease[_ ]?b", item, re.I):
            candidate = "disease_b"
        elif re.search(r"共享|共同|相同的?尾实体(?:数|数量|个数)?", item, re.I):
            candidate = "shared_tail_count"
        else:
            continue
        if candidate not in semantics:
            semantics.append(candidate)
    if (
        len(semantics) != len(items)
        or not all(name in semantics for name in ("disease_a", "disease_b", "shared_tail_count"))
    ):
        return None
    return semantics


def _compile_disease_pairs(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not (
        re.search(r"疾病", question)
        and re.search(r"共享|共同|相同", question)
        and re.search(r"尾实体", question)
        and re.search(r"配对|成对|疾病对", question)
    ):
        return None
    if not re.search(r"头实体|出边|指向", question):
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    projection = _pair_projection(question)
    if projection is None:
        return None
    having = _having_threshold(
        question,
        ("共享尾实体", "共同尾实体", "共享尾实体数", "共同尾实体数"),
    )
    if having is None:
        return None
    confidence_bounds = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="置信度",
        aliases=("confidence",),
    )
    count_bounds = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="观察次数",
        aliases=("观察计数", "观察量", "count"),
    )
    if confidence_bounds is None or count_bounds is None or not all(confidence_bounds):
        return None
    relations = _relations(question, domains)
    if len(relations) > 1 or (relations and not _relation_set_allowed(question, relations)):
        return None
    where = ["a.head_type = 'Disease'", "b.head_type = 'Disease'"]
    if relations:
        relation = esc(relations[0])
        where.extend([f"a.relation_zh = '{relation}'", f"b.relation_zh = '{relation}'"])
    for alias, confidence, counts in zip(
        ("a", "b"),
        confidence_bounds,
        count_bounds,
    ):
        where.extend(
            f"{alias}.confidence {operator} {value}"
            for operator, value in confidence
        )
        where.extend(
            f"{alias}.count {operator} {value}"
            for operator, value in counts
        )
    select_map = {
        "disease_a": "a.head AS disease_a",
        "disease_b": "b.head AS disease_b",
        "shared_tail_count": "COUNT(DISTINCT a.tail) AS shared_tail_count",
    }
    select = ", ".join(select_map[item] for item in projection)
    operator, threshold = having
    return (
        f"SELECT {select} FROM kg_triples AS a JOIN kg_triples AS b "
        "ON a.tail = b.tail AND a.head < b.head "
        f"WHERE {' AND '.join(where)} GROUP BY a.head, b.head "
        f"HAVING COUNT(DISTINCT a.tail) {operator} {threshold} "
        "ORDER BY shared_tail_count DESC, disease_a ASC, disease_b ASC"
    )


def _compile_required_relations_visits(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    request = _required_all_relation_route_request(question, domains)
    if request is None:
        return None
    relations = _unique(request.values)
    if not re.search(
        r"同时|既有|又有|关系集合|要求关系|必需关系|全部关系|每种关系都|具有|具备|包含",
        question,
    ):
        return None
    if _or_matches(question):
        # This CTE proves that every named relation exists.  An explicit
        # relation disjunction must not be inverted into required-all.
        return None
    if not re.search(r"就诊|患者", question) or re.search(r"两跳|两步|链|路径", question):
        return None
    if not re.search(r"头实体|出边|疾病.*关系|主诊断", question):
        return None
    group = _group_dimension(question)
    if group is None or not _group_projection_name(question, group[0]):
        return None
    group_aliases = {
        "region": (r"地区", r"区域"),
        "department": (r"科室",),
        "disease": (r"疾病", r"诊断"),
        "gender": (r"性别",),
    }
    projection_items = _projection_items(question)
    projection = _projection_semantics(
        question,
        {group[0]: group_aliases[group[0]], **_GENERIC_METRIC_PROJECTION},
    )
    allowed_metrics = {"visit_count", "patient_count", "disease_count"}
    allowed_projection = allowed_metrics | {group[0]}
    if (
        not projection_items
        or len(projection) != len(projection_items)
        or group[0] not in projection
        or any(item not in allowed_projection for item in projection)
    ):
        return None
    requested_metrics = [item for item in projection if item in allowed_metrics]
    if not requested_metrics:
        return None
    confidence_bounds = _required_relation_metric_bounds(
        question,
        relations,
        "置信度",
        ("关系置信度", "confidence"),
        domains,
    )
    count_bounds = _required_relation_metric_bounds(
        question,
        relations,
        "观察次数",
        ("出现次数", "观察计数", "观察量", "边次数", "count"),
        domains,
    )
    if confidence_bounds is None or count_bounds is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    visit_predicates = _visit_predicates(question, domains)
    if visit_predicates is None:
        return None
    # A disease may have additional relation kinds; the CTE checks that every
    # required kind is present without excluding those additional kinds.
    relation_values_sql = ", ".join(f"'{esc(value)}'" for value in relations)
    if any(confidence_bounds) or any(count_bounds):
        relation_conditions: list[str] = []
        for relation, confidence, counts in zip(relations, confidence_bounds, count_bounds):
            predicates = [f"relation_zh = '{esc(relation)}'"]
            predicates.extend(
                f"confidence {operator} {value}" for operator, value in confidence
            )
            predicates.extend(
                f"count {operator} {value}" for operator, value in counts
            )
            relation_conditions.append("(" + " AND ".join(predicates) + ")")
        qualified_where = " OR ".join(relation_conditions)
    else:
        qualified_where = f"relation_zh IN ({relation_values_sql})"
    qualified = (
        "qualified_heads AS (SELECT head FROM kg_triples "
        f"WHERE {qualified_where} GROUP BY head "
        f"HAVING COUNT(DISTINCT relation_zh) = {len(relations)})"
    )
    metric_sql = {
        "visit_count": "COUNT(*) AS visit_count",
        "patient_count": "COUNT(DISTINCT p.patient_id) AS patient_count",
        "disease_count": "COUNT(DISTINCT v.disease) AS disease_count",
    }
    select_map = {group[0]: f"{group[1]} AS {group[0]}", **metric_sql}
    select_parts = [select_map[item] for item in projection]
    predicates = patient_predicates + visit_predicates
    sql = (
        f"WITH {qualified} SELECT {', '.join(select_parts)} "
        "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
        "JOIN qualified_heads AS q ON q.head = v.disease"
    )
    if predicates:
        sql += " WHERE " + " AND ".join(dict.fromkeys(predicates))
    sql += f" GROUP BY {group[1]}"
    order_parts = [f"{metric} DESC" for metric in requested_metrics]
    order_parts.append(f"{group[0]} ASC")
    return sql + " ORDER BY " + ", ".join(order_parts)


def _coverage_projection(question: str) -> Optional[list[str]]:
    aliases = {
        "region": (r"地区", r"区域"),
        "department": (r"科室",),
        "disease": (r"疾病", r"诊断"),
        "total_visit_count": _GENERIC_METRIC_PROJECTION["total_visit_count"],
        "covered_visit_count": _GENERIC_METRIC_PROJECTION["covered_visit_count"],
        "coverage": _GENERIC_METRIC_PROJECTION["coverage"],
    }
    return _has_projection(
        question,
        ("total_visit_count", "covered_visit_count", "coverage"),
        aliases,
    )


def _compile_kg_to_visits_coverage(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not (
        re.search(r"覆盖率|覆盖情况|图谱覆盖|KG.{0,8}覆盖|知识图谱.{0,8}覆盖", question, re.I)
        and re.search(r"就诊|访问|患者", question)
    ):
        return None
    group = _group_dimension(question)
    projection = _coverage_projection(question)
    if group is None or projection is None:
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    relations = _relations(question, domains)
    if len(relations) > 1 or (relations and not _relation_set_allowed(question, relations)):
        return None
    if not re.search(r"头实体|主诊断|疾病.*图谱|图谱.*疾病", question):
        return None
    visit_predicates = _visit_predicates(question, domains)
    if visit_predicates is None:
        return None
    where = _patient_predicates(question, domains) + visit_predicates
    on_parts = ["t.head = v.disease"]
    if relations:
        on_parts.append(f"t.relation_zh = '{esc(relations[0])}'")
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("confidence",),
    )
    observation_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("观察计数", "观察量", "count"),
    )
    if confidence_bounds is None or observation_bounds is None:
        return None
    on_parts.extend(
        f"t.confidence {operator} {value}"
        for operator, value in confidence_bounds
    )
    on_parts.extend(
        f"t.count {operator} {value}"
        for operator, value in observation_bounds
    )
    select_map = {
        "region": "p.region AS region",
        "department": "v.department AS department",
        "disease": "v.disease AS disease",
        "total_visit_count": "COUNT(DISTINCT v.visit_id) AS total_visit_count",
        "covered_visit_count": (
            "COUNT(DISTINCT CASE WHEN t.head IS NOT NULL THEN v.visit_id END) "
            "AS covered_visit_count"
        ),
        "coverage": (
            "CAST(COUNT(DISTINCT CASE WHEN t.head IS NOT NULL THEN v.visit_id END) AS REAL) "
            "/ NULLIF(COUNT(DISTINCT v.visit_id), 0) AS coverage"
        ),
    }
    select = ", ".join(select_map[item] for item in projection if item in select_map)
    if not select:
        return None
    sql = (
        f"SELECT {select} FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id LEFT JOIN kg_triples AS t ON "
        f"{' AND '.join(on_parts)}"
    )
    if where:
        sql += " WHERE " + " AND ".join(dict.fromkeys(where))
    sql += f" GROUP BY {group[1]} ORDER BY coverage DESC, {group[0]} ASC"
    return sql


def _risk_projection(question: str) -> Optional[list[str]]:
    aliases = {
        "visit_id": (r"就诊(?:记录)?(?:编号|ID)", r"visit[_ ]?id"),
        "disease": (r"疾病", r"诊断", r"disease"),
        "risk_entity": (r"风险实体", r"中间风险", r"第一跳尾", r"中间实体"),
        "risk_target": (r"风险结果", r"下游实体", r"终点实体", r"第二跳尾", r"目标实体"),
    }
    return _has_projection(
        question,
        ("visit_id", "disease", "risk_entity", "risk_target"),
        aliases,
    )


def _compile_risk_chain(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not re.search(r"风险链|风险路径|风险传播|risk[- ]?chain", question, re.I):
        return None
    if not re.search(r"就诊|访问", question):
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    relations = _chain_relations(question, domains)
    if relations is None or len(relations) != 2 or not _chain_direction(question):
        return None
    projection = _risk_projection(question)
    if projection is None:
        return None
    confidence_bounds = _chain_thresholds(question, relations, domains)
    count_bounds = _edge_metric_bounds(
        question,
        edge_count=2,
        subject="观察次数",
        aliases=("观察计数", "观察量", "count"),
        relations=relations,
        domains=domains,
    )
    if confidence_bounds is None or count_bounds is None:
        return None
    select_map = {
        "visit_id": "v.visit_id",
        "disease": "v.disease",
        "risk_entity": "t1.tail AS risk_entity",
        "risk_target": "t2.tail AS risk_target",
    }
    select = ", ".join(select_map[item] for item in projection)
    where = [
        f"t1.relation_zh = '{esc(relations[0])}'",
        f"t2.relation_zh = '{esc(relations[1])}'",
    ]
    for alias, confidence, counts in zip(
        ("t1", "t2"),
        confidence_bounds,
        count_bounds,
    ):
        where.extend(
            f"{alias}.confidence {operator} {value}"
            for operator, value in confidence
        )
        where.extend(
            f"{alias}.count {operator} {value}"
            for operator, value in counts
        )
    patient_predicates = _patient_predicates(question, domains)
    visit_predicates = _visit_predicates(question, domains)
    if visit_predicates is None:
        return None
    where = patient_predicates + visit_predicates + where
    from_sql = "visits AS v"
    if patient_predicates:
        from_sql += " JOIN patients AS p ON p.patient_id = v.patient_id"
    return (
        f"SELECT DISTINCT {select} FROM {from_sql} JOIN kg_triples AS t1 "
        "ON t1.head = v.disease JOIN kg_triples AS t2 ON t2.head = t1.tail "
        f"WHERE {' AND '.join(where)} ORDER BY v.disease ASC, v.visit_id ASC, "
        "risk_entity ASC, risk_target ASC"
    )


def _older_projection(question: str) -> Optional[list[str]]:
    aliases = {
        "disease": (r"疾病", r"诊断", r"disease"),
        "older_patient_count": _GENERIC_METRIC_PROJECTION["older_patient_count"],
        "visit_count": _GENERIC_METRIC_PROJECTION["visit_count"],
        "mention_count": (r"提及次数", r"图谱提及", r"mention_count"),
        "relation_kind_count": _GENERIC_METRIC_PROJECTION["relation_kind_count"],
    }
    return _has_projection(
        question,
        ("disease",),
        aliases,
    )


def _compile_older_disease_enrichment(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not (
        re.search(r"老年|年龄至少|年龄不低于|older", question, re.I)
        and re.search(r"富集|增强|丰富|图谱.*疾病|疾病.*图谱", question)
        and re.search(r"就诊|患者", question)
        and re.search(r"头实体|作为头|主诊断", question)
    ):
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    age = age_predicates(question, alias="p")
    lower_age = [predicate for predicate in age if re.search(r"\.age\s*>=", predicate)]
    if len(lower_age) != 1:
        return None
    entity_type = _entity_type(question)
    if entity_type not in {None, "Disease"}:
        return None
    projection = _older_projection(question)
    if projection is None or not any(
        item in projection for item in ("older_patient_count", "visit_count", "mention_count", "relation_kind_count")
    ):
        return None
    relations = _relations(question, domains)
    if len(relations) > 1 or (relations and not _relation_set_allowed(question, relations)):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("关系置信度", "confidence"),
    )
    count_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("出现次数", "观察计数", "观察量", "边次数", "count"),
    )
    if confidence_bounds is None or count_bounds is None:
        return None
    select_map = {
        "disease": "v.disease",
        "older_patient_count": "COUNT(DISTINCT p.patient_id) AS older_patient_count",
        "visit_count": "COUNT(DISTINCT v.visit_id) AS visit_count",
        "mention_count": "e.mention_count",
        "relation_kind_count": "COUNT(DISTINCT t.relation_zh) AS relation_kind_count",
    }
    select = ", ".join(select_map[item] for item in projection)
    on_parts = ["t.head = v.disease"]
    if relations:
        on_parts.append(f"t.relation_zh = '{esc(relations[0])}'")
    on_parts.extend(
        f"t.confidence {operator} {value}"
        for operator, value in confidence_bounds
    )
    on_parts.extend(
        f"t.count {operator} {value}"
        for operator, value in count_bounds
    )
    visit_predicates = _visit_predicates(question, domains)
    if visit_predicates is None:
        return None
    predicates = _patient_predicates(question, domains) + visit_predicates
    predicates = list(dict.fromkeys(predicates))
    sql = (
        f"SELECT {select} FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id JOIN kg_entities AS e "
        "ON e.name = v.disease AND e.type = 'Disease' LEFT JOIN kg_triples AS t ON "
        f"{' AND '.join(on_parts)} WHERE {' AND '.join(predicates)}"
    )
    group = "v.disease, e.mention_count"
    if "relation_kind_count" not in projection:
        # A left join is still harmless, but retaining a compact grouped query
        # makes the requested enrichment columns explicit.
        group = "v.disease, e.mention_count"
    sql += f" GROUP BY {group} ORDER BY "
    if "older_patient_count" in projection:
        sql += "older_patient_count DESC, v.disease ASC"
    elif "visit_count" in projection:
        sql += "visit_count DESC, v.disease ASC"
    else:
        sql += "v.disease ASC"
    return sql


def _compile_entity_type_recognition(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    mention_range = _mention_range(question)
    if not (
        re.search(r"实体", question)
        and re.search(r"提及次数", question)
        and mention_range is not None
    ):
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    request = _flat_relation_route_request(
        question,
        domains,
        allow_top_k=False,
    )
    if request is None:
        return None
    relations = list(request.values)
    position = _position(question)
    entity_type = _entity_type(question)
    if len(relations) != 1 or position is None or entity_type is None:
        return None
    projection = _has_projection(
        question,
        ("name", "mention_count"),
        _ENTITY_PROJECTION,
    )
    if projection is None:
        return None
    if any(item not in {"name", "type", "mention_count", "relation_zh", "confidence", "count"} for item in projection):
        return None
    mention_bounds = _single_relation_metric_bounds(
        question,
        "提及次数",
        ("提及数", "mention_count"),
    )
    if not mention_bounds:
        return None
    low, high = mention_range
    select_map = {
        "name": "e.name",
        "type": "e.type",
        "mention_count": "e.mention_count",
        "relation_zh": "t.relation_zh",
        "confidence": "t.confidence",
        "count": "t.count",
    }
    select = ", ".join(select_map[item] for item in projection)
    where = [
        f"e.type = '{entity_type}'",
        f"t.relation_zh = '{esc(relations[0])}'",
        f"e.mention_count BETWEEN {low} AND {high}",
    ]
    range_bounds = {(">=", str(low)), ("<=", str(high))}
    where.extend(
        f"e.mention_count {operator} {value}"
        for operator, value in mention_bounds
        if (operator, value) not in range_bounds
    )
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("confidence",),
    )
    observation_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("观察计数", "观察量", "count"),
    )
    if confidence_bounds is None or observation_bounds is None:
        return None
    where.extend(
        f"t.confidence {operator} {value}"
        for operator, value in confidence_bounds
    )
    where.extend(
        f"t.count {operator} {value}"
        for operator, value in observation_bounds
    )
    return (
        f"SELECT DISTINCT {select} FROM kg_entities AS e JOIN kg_triples AS t "
        f"ON t.{position} = e.name WHERE {' AND '.join(where)} "
        "ORDER BY e.mention_count DESC, e.name ASC"
    )


def _union_labels(question: str) -> tuple[str, str]:
    patient = re.search(
        r"患者(?:来源|标签|标记)(?:为|是)?\s*[“\"]?([\w一-鿿-]+)",
        question,
        re.I,
    )
    graph = re.search(
        r"(?:图谱|知识图谱|实体)(?:来源|标签|标记)(?:为|是)?\s*[“\"]?([\w一-鿿-]+)",
        question,
        re.I,
    )
    return (patient.group(1) if patient else "patient", graph.group(1) if graph else "kg_entity")


def _compile_patient_kg_union(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not (
        re.search(r"UNION|并集|并入|合并|统一列表|两类来源", question, re.I)
        and re.search(r"患者", question)
        and re.search(r"图谱|知识图谱|实体表", question)
    ):
        return None
    if not _fixed_result_modifiers_are_absent(question, domains):
        return None
    projection = _has_projection(
        question,
        ("name", "source"),
        {
            "name": (r"患者姓名", r"实体名称", r"实体名", r"名称", r"统一名称", r"\bname\b"),
            "source": (r"来源", r"来源列", r"source"),
        },
    )
    if projection is None:
        return None
    patient_predicates = _patient_predicates(question, domains)
    if not patient_predicates or not re.search(r"队列|筛选|年龄|性别|地区|区域", question):
        return None
    visit_predicates = _visit_predicates(question, domains)
    if visit_predicates is None:
        return None
    mention_bounds = _single_relation_metric_bounds(
        question,
        "提及次数",
        ("提及数", "mention_count"),
    )
    if mention_bounds is None:
        return None
    entity_type = _entity_type(question)
    if entity_type is None:
        return None
    if any(item not in {"name", "source"} for item in projection):
        return None
    patient_label, graph_label = _union_labels(question)
    select_map = {
        "name": "p.name AS name",
        "source": f"'{esc(patient_label)}' AS source",
    }
    left = ", ".join(select_map[item] for item in projection)
    graph_map = {
        "name": "e.name AS name",
        "source": f"'{esc(graph_label)}' AS source",
    }
    right = ", ".join(graph_map[item] for item in projection)
    patient_where = list(patient_predicates)
    if visit_predicates:
        patient_where.append(
            "EXISTS (SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id AND "
            + " AND ".join(visit_predicates)
            + ")"
        )
    graph_where = [f"e.type = '{entity_type}'"]
    graph_where.extend(
        f"e.mention_count {operator} {value}"
        for operator, value in mention_bounds
    )
    left_sql = "SELECT " + left + " FROM patients AS p WHERE " + " AND ".join(patient_where)
    right_sql = f"SELECT {right} FROM kg_entities AS e WHERE {' AND '.join(graph_where)}"
    operator = "UNION ALL" if re.search(r"不去重|保留重复|UNION\s+ALL", question, re.I) else "UNION"
    return f"{left_sql} {operator} {right_sql} ORDER BY name ASC, source ASC"


def _semantic_bound(
    question: str,
    labels: Sequence[str],
) -> Optional[tuple[str, str]]:
    label = "(?:" + "|".join(re.escape(value) for value in sorted(labels, key=len, reverse=True)) + ")"
    patterns = (
        (rf"{label}\s*(?:下限(?:为|是)?|至少|不低于|不少于|达到|达|≥|>=)\s*(\d+(?:\.\d+)?)", ">="),
        (rf"{label}\s*(?:超过|高于|大于|>)\s*(\d+(?:\.\d+)?)", ">"),
        (rf"{label}\s*(?:上限(?:为|是)?|不超过|不高于|至多|≤|<=)\s*(\d+(?:\.\d+)?)", "<="),
        (rf"(?:至少|不低于|不少于|达到|达|≥|>=)\s*(\d+(?:\.\d+)?)\s*{label}", ">="),
        (rf"(\d+(?:\.\d+)?)\s*{label}", ">="),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, question)
        if match:
            return operator, _number(match.group(1))
    return None


def _edge_count_bound(question: str) -> Optional[tuple[str, str]]:
    parsed = _semantic_bound(
        question,
        ("出现次数", "观察次数", "观察计数", "计数", "次数"),
    )
    if parsed:
        return parsed
    for pattern in (
        r"(?:至少|不低于|不少于|达到|达|≥|>=)\s*(\d+(?:\.\d+)?)\s*次(?:出现|观察|计数)?",
        r"(\d+(?:\.\d+)?)\s*次(?:出现|观察|计数)",
    ):
        match = re.search(pattern, question)
        if match:
            return ">=", _number(match.group(1))
    return None


def _compile_grounded_edge_filter(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    request = _flat_relation_route_request(
        question,
        domains,
        allow_top_k=True,
    )
    if request is None:
        return None
    relations = list(request.values)
    # A bare relation mention also occurs in aggregate relation analyses.  This
    # raw-edge compiler requires an explicit edge/triple/connection request so
    # it cannot replace an aggregate projection with raw head/tail columns.
    explicit_edge_projection = bool(
        re.search(r"中文关系", question)
        and re.search(r"头实体", question)
        and re.search(r"尾实体", question)
    )
    if len(relations) != 1 or not (
        re.search(r"边|三元组|连接", question)
        or explicit_edge_projection
    ):
        return None
    if not re.search(r"观察次数|出现次数|计数|\d+\s*次出现", question):
        return None
    # The generic edge compiler already preserves the stable all-four-column
    # shape.  This specialised path exists only for an exact partial projection.
    if _has_projection(
        question,
        ("head", "tail", "confidence", "count"),
        _EDGE_PROJECTION,
    ):
        return None
    confidence_bounds = _single_relation_metric_bounds(
        question,
        "置信度",
        ("关系置信度", "confidence"),
    )
    observation_bounds = _single_relation_metric_bounds(
        question,
        "观察次数",
        ("出现次数", "观察计数", "观察量", "边次数", "count"),
    )
    if (
        confidence_bounds is None
        or observation_bounds is None
        or not (confidence_bounds or observation_bounds)
    ):
        return None
    if not re.search(r"头实体|头尾实体|起点|两端名称", question):
        return None
    if not re.search(r"尾实体|头尾实体|终点|两端名称", question):
        return None
    projection_items = _projection_items(question)
    projection = _projection_semantics(question, _EDGE_PROJECTION)
    if (
        not projection_items
        or len(projection) != len(projection_items)
        or any(item not in _EDGE_PROJECTION for item in projection)
    ):
        return None
    where = [f"relation_zh = '{esc(relations[0])}'"]
    if re.search(r"头实体类型", question):
        head_type = next(
            (
                str(value)
                for value in domains.get("kg_head_type", ())
                if value
                and re.search(
                    rf"头实体类型(?:为|是|等于|=|：|:)?\s*{re.escape(str(value))}",
                    question,
                    re.I,
                )
            ),
            None,
        )
        if head_type is None:
            return None
        where.append(f"head_type = '{esc(head_type)}'")
    where.extend(
        f"confidence {operator} {value}" for operator, value in confidence_bounds
    )
    where.extend(
        f"count {operator} {value}" for operator, value in observation_bounds
    )
    top = _top_k(question)
    if top is None and re.search(
        r"(?:排名|排行)?\s*(?:前|TOP)\s*(?:[一二三四五六七八九十几若干]|$)",
        question,
        re.I,
    ):
        return None
    sql = (
        f"SELECT {', '.join(projection)} FROM kg_triples "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY count DESC, confidence DESC, head ASC, tail ASC"
    )
    return f"{sql} LIMIT {top}" if top is not None else sql


def _compile_grounded_tail_join(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    request = _flat_relation_route_request(
        question,
        domains,
        allow_top_k=False,
    )
    if request is None:
        return None
    relations = list(request.values)
    if len(relations) != 1 or not re.search(r"尾实体|尾端实体|尾端|目标", question):
        return None
    mention_bounds = _single_relation_metric_bounds(
        question,
        "提及次数",
        ("尾实体提及次数", "提及数", "提及量", "mention_count", "热度"),
    )
    if not mention_bounds or not re.search(r"提及|热度", question):
        return None
    if not re.search(r"头实体|(?:输出|返回|列出|需返回).{0,8}头|两端名称", question):
        return None
    projection_items = _projection_items(question)
    projection = _projection_semantics(question, _TAIL_JOIN_PROJECTION)
    select_map = {
        "name": "e.name",
        "type": "e.type",
        "mention_count": "e.mention_count",
        "head": "t.head",
        "tail": "t.tail",
        "relation_zh": "t.relation_zh",
        "confidence": "t.confidence",
        "count": "t.count",
    }
    if (
        not projection_items
        or len(projection) != len(projection_items)
        or any(item not in select_map for item in projection)
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
        ("出现次数", "观察计数", "观察量", "边次数", "count"),
    )
    # This specialised signature formerly required the word “置信度” while
    # dropping its condition.  Retain the narrow signature, but require an
    # executable confidence bound rather than accepting a bare word.
    if confidence_bounds is None or not confidence_bounds or count_bounds is None:
        return None
    where = [f"t.relation_zh = '{esc(relations[0])}'"]
    where.extend(
        f"e.mention_count {operator} {value}"
        for operator, value in mention_bounds
    )
    where.extend(f"t.confidence {operator} {value}" for operator, value in confidence_bounds)
    where.extend(f"t.count {operator} {value}" for operator, value in count_bounds)
    select = ", ".join(select_map[item] for item in projection)
    mention_direction = "ASC" if mention_bounds[0][0] in {"<", "<="} else "DESC"
    return (
        f"SELECT {select} FROM kg_triples AS t JOIN kg_entities AS e "
        "ON e.name = t.tail WHERE "
        f"{' AND '.join(where)} ORDER BY e.mention_count {mention_direction}, "
        "e.name ASC, t.head ASC, t.tail ASC"
    )


def _compile_grounded_per_head_topk(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    request = _flat_relation_route_request(
        question,
        domains,
        allow_top_k=True,
    )
    if request is None:
        return None
    relations = list(request.values)
    top = _top_k(question)
    if len(relations) != 1 or top is None:
        return None
    if not re.search(r"不同尾|尾实体种类|去重终点|不同尾端|覆盖面", question):
        return None
    if not re.search(r"计数(?:总和|合计)|出现次数总和|总次数|关系计数", question):
        return None
    if not re.search(r"头实体|起点", question):
        return None
    if _projection_items(question):
        return None
    if (
        _metric_threshold_intent(question, "置信度", ("confidence",))
        or _metric_threshold_intent(
            question,
            "观察次数",
            ("出现次数", "观察计数", "观察量", "count"),
        )
    ):
        return None
    return (
        "SELECT head, COUNT(DISTINCT tail) AS distinct_tail_count, "
        "SUM(count) AS relation_count_sum FROM kg_triples "
        f"WHERE relation_zh = '{esc(relations[0])}' GROUP BY head "
        "ORDER BY distinct_tail_count DESC, relation_count_sum DESC, head ASC "
        f"LIMIT {top}"
    )


def compile_kg_medical_sql(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    """Compile a complete schema-bound KG/medical query or abstain."""

    q = str(question).strip()
    if not q or not isinstance(domains, Mapping):
        return None
    if (
        classify_region_filter_intent(q, domains).disposition
        is RegionFilterDisposition.TERMINAL_ABSTAIN
    ):
        return None
    if _has_clinical_scope(q, domains):
        # A clinical scope is not a global KG filter.  Only these routes join
        # patients or visits and can therefore prove ownership of that scope.
        compilers = (
            _compile_risk_chain,
            _compile_older_disease_enrichment,
            _compile_patient_kg_union,
            _compile_required_relations_visits,
            _compile_kg_to_visits_coverage,
        )
    else:
        compilers = (
            _compile_grounded_tail_join,
            _compile_grounded_per_head_topk,
            _compile_grounded_edge_filter,
            _compile_risk_chain,
            _compile_older_disease_enrichment,
            _compile_patient_kg_union,
            _compile_required_relations_visits,
            _compile_kg_to_visits_coverage,
            _compile_disease_pairs,
            _compile_two_hop_chain,
            _compile_tail_coverage,
            _compile_entity_type_recognition,
            _compile_entity_required_relations,
            _compile_tail_entity_join,
            _compile_per_head_relation_topk,
            _compile_relation_edge_filter,
        )
    for compiler in compilers:
        sql = compiler(q, domains)
        if sql:
            return sql
    return None
