"""Deterministic compositional compiler for natural Chinese medical queries.

This module deliberately composes a small, schema-bound semantic plan:

* requested projections;
* database-grounded filters;
* aggregation/grouping operators;
* extrema, Top-K, set, temporal and anti-join operators.

It does not dispatch on complete question strings or evaluation item ids.
Every emitted identifier is a constant from the four-table schema and every
string literal must be linked to a value loaded from the current database.
Unsupported or ambiguous combinations abstain with ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from dki.analysis.nl2sql_semantic_kg import analyze_relation_request


Domains = Mapping[str, Sequence[str]]

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

_PATIENT_PATTERN = r"(?:患者|病人|病患|病号|参保人)"
_INFIX_DISJUNCTION_PATTERN = r"(?:或(?:者|是)?|任一|任意(?:一个)?|至少一个|要么|\bOR\b)"
_POSTFIX_DISJUNCTION_PATTERN = r"二者之一"
_DISJUNCTION_PATTERN = rf"(?:{_INFIX_DISJUNCTION_PATTERN}|{_POSTFIX_DISJUNCTION_PATTERN})"
_SHARED_SET_CONNECTOR = r"(?:且|并且|以及|同时)"
_NEGATIVE_VISIT_PATTERN = (
    r"(?:从来没有|从未|未曾|没有任何|没有在|没在|没有被|未被|"
    r"没有使用过|没用过|未使用过|未使用|没去过|未去过|没有)"
)

# These controlled slots are deliberately narrower than the general domain
# matcher.  They let boolean compilers notice an explicit value which did not
# ground to the supplied schema domain instead of silently omitting it.
_EXPLICIT_VISIT_VALUE_SLOTS: tuple[tuple[str, str], ...] = (
    (
        "department",
        r"(?:要么\s*在|在|于|到|去过|去|、|，|,|或(?:者|是)?|\bOR\b|和|与|又|要么)\s*"
        r"(?:去过|到过|在)?\s*"
        r"(?P<value>[^\s、，,。；;？！?!或和与又且并但却要]{1,16}(?:内科|外科|科))"
        r"(?=\s*(?:就诊|看(?:过)?病|就医|来院|使用|服用|开药|"
        r"任一|任意|至少|二者|要么|\bOR\b|[、，,。；;？！?!或和与又且并但却]|$))",
    ),
    (
        "disease",
        r"(?:患(?!者)(?:有|上)?|确诊(?:为)?|诊断(?:为)?)\s*"
        r"(?P<value>[^\s、，,。；;？！?!或和与又且并但却要的]{1,24})"
        r"(?=\s*(?:就诊|看(?:过)?病|使用|服用|开药|的|"
        r"[、，,。；;？！?!或和与又且并但却]|$))",
    ),
    # A set continuation is explicit only when a disease verb already anchors
    # the first operand.  This avoids classifying an ordinary department
    # phrase such as ``在内科或外科的患者`` as a disease slot.
    (
        "disease",
        r"(?:患(?!者)(?:有|上)?|确诊(?:为)?|诊断(?:为)?)\s*"
        r"[^\s、，,。；;？！?!或和与又且并但却要的]{1,24}\s*"
        r"(?:或(?:者|是)?|\bOR\b|要么|和|与|又|、|，|,)\s*"
        r"(?P<value>[^\s、，,。；;？！?!或和与又且并但却要的]{1,24})"
        r"(?=\s*(?:的)?(?:患者|病人|病患|病号|参保人|就诊|看(?:过)?病|$))",
    ),
    (
        "drug_prescribed",
        r"(?:使用(?:过)?|服用(?:过)?|开过|被开过)\s*"
        r"(?P<value>[^\s、，,。；;？！?!或和与又且并但却要的]{1,24})"
        r"(?=\s*(?:就诊|看(?:过)?病|的|[、，,。；;？！?!或和与又且并但却]|$))",
    ),
    (
        "drug_prescribed",
        r"(?:使用(?:过)?|服用(?:过)?|开过|被开过)\s*"
        r"[^\s、，,。；;？！?!或和与又且并但却要的]{1,24}\s*"
        r"(?:或(?:者|是)?|\bOR\b|要么|和|与|又|、|，|,)\s*"
        r"(?P<value>[^\s、，,。；;？！?!或和与又且并但却要的]{1,24})"
        r"(?=\s*(?:的)?(?:患者|病人|病患|病号|参保人|就诊|看(?:过)?病|$))",
    ),
    (
        "exam",
        r"(?:做过|进行了?|接受了?)\s*"
        r"(?P<value>[^\s、，,。；;？！?!或和与又且并但却要的]{1,24})"
        r"(?=\s*(?:检查|的|[、，,。；;？！?!或和与又且并但却]|$))",
    ),
    (
        "exam",
        r"(?:做过|进行了?|接受了?)\s*"
        r"[^\s、，,。；;？！?!或和与又且并但却要的]{1,24}\s*"
        r"(?:或(?:者|是)?|\bOR\b|要么|和|与|又|、|，|,)\s*"
        r"(?P<value>[^\s、，,。；;？！?!或和与又且并但却要的]{1,24})"
        r"(?=\s*(?:的)?(?:患者|病人|病患|病号|参保人|就诊|看(?:过)?病|$))",
    ),
)


def _esc(value: str) -> str:
    return str(value).replace("'", "''")


def _domain_values(question: str, values: Sequence[str]) -> list[str]:
    """Return non-overlapping grounded values in textual occurrence order."""

    candidates: list[tuple[int, int, str]] = []
    for value in sorted(
        {str(item) for item in values if str(item)},
        key=lambda item: (-len(item), item),
    ):
        for match in re.finditer(re.escape(value), question):
            candidates.append((match.start(), match.end(), value))
    selected: list[tuple[int, int, str]] = []
    for start, end, value in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < used_end and end > used_start for used_start, used_end, _ in selected):
            continue
        selected.append((start, end, value))
    return [value for _, _, value in sorted(selected)]


def _integer(value: str) -> Optional[int]:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return _CN_INTS.get(value)


def _limit(question: str, *, default: Optional[int] = None) -> Optional[int]:
    for pattern in (
        r"前\s*(\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次)?",
        r"(?:最高|最低|最多|最少)的?\s*(\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次)",
        r"(?:最贵|最便宜)的?\s*(\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次)",
        r"(?:列出|返回)\s*(\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次)",
    ):
        match = re.search(pattern, question)
        if match:
            return _integer(match.group(1))
    return default


def _patient_projection(question: str) -> list[tuple[int, str]]:
    candidates = (
        (rf"{_PATIENT_PATTERN}(?:编号|ID)", "p.patient_id"),
        (r"姓名|名字|叫什么", "p.name"),
        (r"性别", "p.gender"),
        (r"年龄", "p.age"),
        (r"地区|区域|城市", "p.region"),
    )
    found: list[tuple[int, str]] = []
    for pattern, column in candidates:
        match = re.search(pattern, question, re.I)
        if match:
            found.append((match.start(), column))
    return _dedupe_projection(found)


def _visit_projection(question: str) -> list[tuple[int, str]]:
    candidates = (
        (r"(?:就诊|记录)(?:的)?(?:编号|ID)", "visit_id"),
        (rf"{_PATIENT_PATTERN}(?:编号|ID)", "patient_id"),
        (r"就诊日期|就诊时间|日期|时间", "visit_date"),
        (r"就诊科室|科室", "department"),
        (r"疾病|诊断", "disease"),
        (r"处方药物|处方药|药物", "drug_prescribed"),
        (r"检查", "exam"),
        (r"住院天数|住院时长", "los_days"),
        (r"费用|金额", "cost"),
    )
    found: list[tuple[int, str]] = []
    for pattern, column in candidates:
        for match in re.finditer(pattern, question, re.I):
            # "患者编号" contains the generic word "编号"; the visit-id
            # pattern above intentionally requires an explicit visit/record.
            found.append((match.start(), column))
            break
    return _dedupe_projection(found)


def _dedupe_projection(items: Sequence[tuple[int, str]]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    seen: set[str] = set()
    for position, column in sorted(items):
        if column in seen:
            continue
        seen.add(column)
        result.append((position, column))
    return result


def _requested_clause(question: str) -> str:
    """Prefer the explicit answer clause over filter/grouping mentions."""

    matches = list(
        re.finditer(
            r"(?:列出|给出|返回|显示|输出)(.+?)(?:[。？?]|$)",
            question,
        )
    )
    clause = matches[-1].group(1) if matches else question
    # Natural Top-K questions often put the ordering metric before the actual
    # projection: "列出费用最高的3次就诊的编号和费用".  Projection order starts
    # after the final visit-record possessive, not at the ranking modifier.
    tail = re.search(r"(?:次|条)?就诊(?:记录)?的(.+)$", clause)
    return f"就诊的{tail.group(1)}" if tail else clause


def _patient_columns(question: str, *, default: Sequence[str]) -> list[str]:
    projection = _patient_projection(_requested_clause(question))
    return [column for _, column in projection] or list(default)


def _visit_columns(question: str, *, default: Sequence[str]) -> list[str]:
    projection = _visit_projection(_requested_clause(question))
    return [column for _, column in projection] or list(default)


def _value_dimension_occurrences(
    question: str,
    domains: Domains,
) -> list[tuple[int, int, str, str]]:
    """Find non-overlapping grounded visit values with their source spans."""

    candidates: list[tuple[int, int, str, str]] = []
    for key, column in (
        ("department", "department"),
        ("disease", "disease"),
        ("drug", "drug_prescribed"),
        ("exam", "exam"),
    ):
        for value in sorted(
            {str(item) for item in domains.get(key, ()) if str(item)},
            key=lambda item: (-len(item), item),
        ):
            candidates.extend(
                (match.start(), match.end(), column, value)
                for match in re.finditer(re.escape(value), question)
            )
    selected: list[tuple[int, int, str, str]] = []
    for candidate in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]), item[2])):
        start, end, _, _ = candidate
        if any(start < used_end and end > used_start for used_start, used_end, _, _ in selected):
            continue
        selected.append(candidate)
    return sorted(selected)


def _value_dimensions(
    question: str,
    domains: Domains,
) -> list[tuple[str, str]]:
    """Find grounded visit dimensions, preserving their textual order."""

    return [
        (column, value)
        for _, _, column, value in _value_dimension_occurrences(question, domains)
    ]


def _explicit_visit_value_slots(question: str) -> list[tuple[int, int, str, str]]:
    """Return controlled explicit visit-value slots with their source spans."""

    slots: list[tuple[int, int, str, str]] = []
    for column, pattern in _EXPLICIT_VISIT_VALUE_SLOTS:
        for match in re.finditer(pattern, question, re.I):
            value = match.group("value").strip()
            if value:
                slots.append((match.start("value"), match.end("value"), column, value))
    return sorted(set(slots))


def _has_unmatched_explicit_visit_value(question: str, domains: Domains) -> bool:
    """Return whether a controlled visit-value slot failed exact grounding."""

    domain_by_column = {
        "department": domains.get("department", ()),
        "disease": domains.get("disease", ()),
        "drug_prescribed": domains.get("drug", ()),
        "exam": domains.get("exam", ()),
    }
    for _, _, column, value in _explicit_visit_value_slots(question):
        if value not in {str(item) for item in domain_by_column[column]}:
            return True
    return False


def _has_cross_dimension_or(question: str, domains: Domains) -> bool:
    """Return whether an OR joins grounded values from different columns."""

    values = _value_dimension_occurrences(question, domains)
    for (_, left_end, left_column, _), (right_start, _, right_column, _) in zip(
        values,
        values[1:],
    ):
        if left_column != right_column and re.search(
            _INFIX_DISJUNCTION_PATTERN,
            question[left_end:right_start],
            re.I,
        ):
            return True
    # ``二者之一`` is postfix and applies to the two most recent values before
    # it.  Looking at all following text would incorrectly attach that marker
    # to a later shared qualifier in ``A 和 B 二者之一且患 C``.
    for marker in re.finditer(_POSTFIX_DISJUNCTION_PATTERN, question, re.I):
        preceding = [value for value in values if value[1] <= marker.start()]
        if len(preceding) >= 2 and preceding[-2][2] != preceding[-1][2]:
            return True
    return False


def _where_for_visit_values(
    question: str,
    domains: Domains,
    *,
    alias: str = "",
) -> list[str]:
    prefix = f"{alias}." if alias else ""
    predicates = [
        f"{prefix}{column} = '{_esc(value)}'"
        for column, value in _value_dimensions(question, domains)
    ]
    return list(dict.fromkeys(predicates))


def _set_operator(question: str) -> Optional[str]:
    """Return the exact supported set operator for a controlled request."""

    if (
        ("但" in question or "却" in question)
        and re.search(_NEGATIVE_VISIT_PATTERN, question)
    ):
        return "EXCEPT"
    # An explicit disjunction is stronger than ``同时`` when the latter joins
    # a proven globally shared qualifier: ``在 A 或 B 就诊同时患 C`` is a
    # union of the two department branches, each constrained by C.
    if re.search(_DISJUNCTION_PATTERN, question, re.I):
        return "UNION"
    if re.search(r"既.+又|同时|两者都|都去过|都就诊", question):
        return "INTERSECT"
    return None


def _is_set_operation_intent(question: str, domains: Domains) -> bool:
    """Recognize an explicit patient visit-set shape independent of projection."""

    if not re.search(_PATIENT_PATTERN, question) or _set_operator(question) is None:
        return False
    values = _value_dimension_occurrences(question, domains)
    slots = _explicit_visit_value_slots(question)
    # A grounded value plus a controlled ungrounded slot is still an explicit
    # two-operand set shape.  Combining the evidence prevents broad fallback
    # from dropping the ungrounded operand merely because it cannot appear in
    # the domain-occurrence list.
    return len({*values, *slots}) >= 2


def _is_set_operation_request(question: str, domains: Domains) -> bool:
    """Recognize a supported patient-ID set-operation request."""

    return bool(
        _is_set_operation_intent(question, domains)
        and re.search(r"编号|ID", question, re.I)
    )


def _has_global_shared_set_scope(
    question: str,
    operands: Sequence[tuple[int, int, str, str]],
    shared: Sequence[tuple[int, int, str, str]],
) -> bool:
    """Prove that every shared predicate scopes over both set operands."""

    first_operand_start = min(start for start, _, _, _ in operands)
    last_operand_end = max(end for _, end, _, _ in operands)
    if all(end <= first_operand_start for _, end, _, _ in shared):
        boundary = question[
            max(end for _, end, _, _ in shared) : first_operand_start
        ]
    elif all(start >= last_operand_end for start, _, _, _ in shared):
        boundary = question[
            last_operand_end : min(start for start, _, _, _ in shared)
        ]
    else:
        return False
    return re.search(_SHARED_SET_CONNECTOR, boundary) is not None


def _set_operation_parts(
    question: str,
    domains: Domains,
) -> Optional[tuple[str, str, str, str, list[tuple[str, str]]]]:
    """Return a two-operand set plan with proven global shared predicates."""

    if not _is_set_operation_request(question, domains):
        return None
    if (
        _has_unmatched_explicit_visit_value(question, domains)
        or _has_cross_dimension_or(question, domains)
    ):
        return None
    operator = _set_operator(question)
    if operator is None:
        return None
    occurrences = _value_dimension_occurrences(question, domains)
    by_column: dict[str, list[tuple[int, int, str, str]]] = {}
    for occurrence in occurrences:
        by_column.setdefault(occurrence[2], []).append(occurrence)
    operands = [
        (column, column_occurrences)
        for column, column_occurrences in by_column.items()
        if len({value for _, _, _, value in column_occurrences}) >= 2
    ]
    # A set operator has exactly two values of one dimension.  A third value,
    # a repeated operand, or a second repeated dimension cannot be assigned to
    # a branch safely.
    if len(operands) != 1:
        return None
    column, operand_occurrences = operands[0]
    if (
        len(operand_occurrences) != 2
        or len({value for _, _, _, value in operand_occurrences}) != 2
    ):
        return None
    shared = [
        occurrence
        for shared_column, column_occurrences in by_column.items()
        if shared_column != column
        for occurrence in column_occurrences
    ]
    if any(
        len(column_occurrences) != 1
        for shared_column, column_occurrences in by_column.items()
        if shared_column != column
    ):
        return None
    if shared and not _has_global_shared_set_scope(
        question,
        operand_occurrences,
        shared,
    ):
        return None
    first, second = (value for _, _, _, value in operand_occurrences)
    return operator, column, first, second, [
        (shared_column, value)
        for _, _, shared_column, value in shared
    ]


def _compile_set_operation(question: str, domains: Domains) -> Optional[str]:
    parts = _set_operation_parts(question, domains)
    if parts is None:
        return None
    operator, column, first, second, shared = parts

    def branch(value: str) -> str:
        predicates = [f"{column} = '{_esc(value)}'"]
        predicates.extend(
            f"{shared_column} = '{_esc(shared_value)}'"
            for shared_column, shared_value in shared
        )
        return "SELECT patient_id FROM visits WHERE " + " AND ".join(predicates)

    return f"{branch(first)} {operator} {branch(second)} ORDER BY patient_id ASC"


def _is_anti_join_request(question: str) -> bool:
    """Recognize the controlled negative patient-visit signature."""

    return bool(
        re.search(_PATIENT_PATTERN, question)
        and re.search(_NEGATIVE_VISIT_PATTERN, question)
    )


def _has_independent_negative_visit_scopes(question: str) -> bool:
    """Return whether multiple negatives would need separate anti-joins."""

    return len(re.findall(_NEGATIVE_VISIT_PATTERN, question)) > 1


def _has_conflicting_visit_values(values: Sequence[tuple[str, str]]) -> bool:
    """Return whether one visit column is assigned distinct values together."""

    values_by_column: dict[str, set[str]] = {}
    for column, value in values:
        values_by_column.setdefault(column, set()).add(value)
    return any(len(column_values) > 1 for column_values in values_by_column.values())


def _compile_anti_join(question: str, domains: Domains) -> Optional[str]:
    if not _is_anti_join_request(question):
        return None
    columns = _patient_columns(question, default=("p.name",))
    select = ", ".join(columns)
    values = _value_dimensions(question, domains)
    if (
        _has_unmatched_explicit_visit_value(question, domains)
        or _has_cross_dimension_or(question, domains)
        or _has_independent_negative_visit_scopes(question)
    ):
        return None
    # Several distinct values from one column need an explicit OR/IN scope;
    # conjoining them within one visit row would be contradictory.
    if _has_conflicting_visit_values(values):
        return None
    if values:
        conditions = ["v.patient_id = p.patient_id"]
        conditions.extend(
            f"v.{column} = '{_esc(value)}'"
            for column, value in dict.fromkeys(values)
        )
    elif re.search(r"就诊|看过病|看病|就医|到院", question):
        conditions = ["v.patient_id = p.patient_id"]
    else:
        return None
    return (
        f"SELECT {select} FROM patients AS p WHERE NOT EXISTS "
        "(SELECT v.patient_id AS patient_id FROM visits AS v WHERE "
        + " AND ".join(conditions)
        + f") ORDER BY {columns[0]} ASC"
    )


def _having_threshold(question: str) -> Optional[tuple[str, int]]:
    patterns = (
        (r"超过\s*(\d+|[一二两三四五六七八九十])\s*(?:次|回)(?:的)?就诊", ">"),
        (r"就诊(?:次数|数|量)\s*(?:超过|多于)\s*(\d+|[一二两三四五六七八九十])\s*(?:次|回)", ">"),
        (r"就诊(?:超过|多于)\s*(\d+|[一二两三四五六七八九十])\s*(?:次|回)", ">"),
        (r"(?:至少|不低于|不少于)\s*(\d+|[一二两三四五六七八九十])\s*(?:次|回)(?:的)?就诊", ">="),
        (r"就诊(?:次数|数|量)\s*(?:至少|不低于|不少于)\s*(\d+|[一二两三四五六七八九十])\s*(?:次|回)", ">="),
        (r"就诊(?:至少|不低于|不少于)\s*(\d+|[一二两三四五六七八九十])\s*(?:次|回)", ">="),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, question)
        if match and (value := _integer(match.group(1))) is not None:
            return operator, value
    return None


def _compile_patient_having(question: str, domains: Domains) -> Optional[str]:
    if not re.search(_PATIENT_PATTERN, question) or "就诊" not in question:
        return None
    threshold = _having_threshold(question)
    if threshold is None:
        return None
    columns = _patient_columns(question, default=("p.patient_id",))
    operator, value = threshold
    predicates = _where_for_visit_values(question, domains, alias="v")
    conditions = ["v.patient_id = p.patient_id", *predicates]
    return (
        f"SELECT {', '.join(columns)} FROM patients AS p WHERE "
        "(SELECT COUNT(*) AS visit_count FROM visits AS v WHERE "
        + " AND ".join(conditions)
        + f") {operator} {value} ORDER BY {columns[0]} ASC"
    )


def _compile_per_patient_temporal(question: str, domains: Domains) -> Optional[str]:
    if not re.search(
        rf"(?:每(?:位|个)|各(?:个)?){_PATIENT_PATTERN}",
        question,
    ) or "就诊" not in question:
        return None
    if re.search(r"最近|最后|末次|最晚", question):
        aggregate = "MAX"
    elif re.search(r"首次|第一次|最早", question):
        aggregate = "MIN"
    else:
        return None
    columns = _visit_columns(
        question,
        default=("patient_id", "visit_date"),
    )
    rendered = ", ".join(f"v.{column}" for column in columns)
    outer_predicates = _where_for_visit_values(question, domains, alias="v")
    inner_predicates = _where_for_visit_values(question, domains, alias="v2")
    inner_suffix = " AND " + " AND ".join(inner_predicates) if inner_predicates else ""
    outer_suffix = " AND " + " AND ".join(outer_predicates) if outer_predicates else ""
    return (
        f"SELECT {rendered} FROM visits AS v "
        f"WHERE v.visit_date = (SELECT {aggregate}(v2.visit_date) "
        "FROM visits AS v2 WHERE v2.patient_id = v.patient_id"
        + inner_suffix
        + ")"
        + outer_suffix
        + " ORDER BY v.patient_id ASC, v.visit_id ASC"
    )


def _extreme_direction(question: str, semantic: str) -> Optional[str]:
    if semantic == "cost":
        if re.search(r"费用.{0,5}(?:最高|最大)|最高.{0,8}费用", question):
            return "DESC"
        if re.search(r"费用.{0,5}(?:最低|最小)|最低.{0,8}费用", question):
            return "ASC"
    if semantic == "visit_date":
        if re.search(r"最近|最后|末次|最晚", question):
            return "DESC"
        if re.search(r"首次|第一次|最早", question):
            return "ASC"
    return None


def _compile_extreme_row(question: str, domains: Domains) -> Optional[str]:
    if "就诊" not in question:
        return None
    for semantic, order_column, tie_direction in (
        ("cost", "cost", "ASC"),
        ("visit_date", "visit_date", "DESC"),
    ):
        direction = _extreme_direction(question, semantic)
        if direction is None:
            continue
        patient_columns = _patient_columns(_requested_clause(question), default=())
        visit_columns = _visit_columns(_requested_clause(question), default=())
        predicates = _where_for_visit_values(question, domains, alias="v" if patient_columns else "")
        if patient_columns:
            select = ", ".join(patient_columns)
            sql = (
                f"SELECT {select} FROM patients AS p JOIN visits AS v "
                "ON v.patient_id = p.patient_id"
            )
            if predicates:
                sql += " WHERE " + " AND ".join(predicates)
            sql += (
                f" ORDER BY v.{order_column} {direction}, "
                f"v.visit_id {tie_direction} LIMIT 1"
            )
            return sql
        if visit_columns and (
            len(visit_columns) > 1 or visit_columns[0] != order_column
        ):
            sql = f"SELECT {', '.join(visit_columns)} FROM visits"
            if predicates:
                sql += " WHERE " + " AND ".join(predicates)
            sql += (
                f" ORDER BY {order_column} {direction}, "
                f"visit_id {tie_direction} LIMIT 1"
            )
            return sql
    return None


def _compile_top_rows(question: str, domains: Domains) -> Optional[str]:
    if "就诊" not in question or not re.search(
        r"费用.{0,8}(?:最高|最低)|(?:最高|最低).{0,8}费用|最贵|最便宜",
        question,
    ):
        return None
    limit = _limit(question)
    if limit is None:
        return None
    columns = _visit_columns(_requested_clause(question), default=("visit_id", "cost"))
    direction = "ASC" if re.search(r"最低|最少|最便宜", question) else "DESC"
    predicates = _where_for_visit_values(question, domains)
    sql = f"SELECT {', '.join(columns)} FROM visits"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return sql + f" ORDER BY cost {direction}, visit_id ASC LIMIT {limit}"


def _compile_patient_cost_top(question: str, domains: Domains) -> Optional[str]:
    if not (
        re.search(_PATIENT_PATTERN, question)
        and re.search(r"累计费用|总费用|总花费|总支出|花费总额", question)
        and re.search(
            rf"汇总|按{_PATIENT_PATTERN}|每位{_PATIENT_PATTERN}|"
            rf"前[一二两三四五六七八九十\d]+位{_PATIENT_PATTERN}",
            question,
        )
    ):
        return None
    limit = _limit(question)
    if limit is None:
        return None
    columns = _patient_columns(_requested_clause(question), default=("p.patient_id",))
    include_total_cost = bool(
        re.search(
            r"(?:列出|给出|返回|显示|输出).{0,64}"
            r"(?:姓名|名字|患者).{0,16}(?:和|及|以及|、)"
            r"(?:累计费用|总费用|总花费|总支出|花费总额)|"
            r"(?:及其|以及其)(?:累计费用|总费用|总花费|总支出|花费总额)",
            question,
        )
    )
    predicates = _where_for_visit_values(question, domains, alias="v")
    # Visit domains constrain the rows defining each patient's total, before
    # the grouped Top-K ranking is evaluated.
    select_items = [*columns]
    if include_total_cost:
        select_items.append("SUM(v.cost) AS total_cost")
    sql = (
        f"SELECT {', '.join(select_items)} "
        "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id"
    )
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return (
        sql
        + f" GROUP BY p.patient_id, {', '.join(columns)}"
        + " ORDER BY "
        + ("total_cost" if include_total_cost else "SUM(v.cost)")
        + f" DESC, {columns[0]} ASC LIMIT {limit}"
    )


def _group_dimension(question: str) -> Optional[tuple[str, str]]:
    if "科室" in question:
        return "department", "department"
    if "疾病" in question or "诊断" in question:
        return "disease", "disease"
    if (
        "处方药" in question
        or "药物" in question
        or "开药" in question
        or re.search(r"哪(?:一)?种药", question)
    ):
        return "drug_prescribed", "drug_prescribed"
    if "检查" in question:
        return "exam", "exam"
    return None


def _compile_group(question: str, domains: Domains) -> Optional[str]:
    # Date-bounded and explicitly controlled analytical questions remain with
    # the established compiler, which already consumes their full contract.
    if re.search(r"\d{4}-\d{2}-\d{2}", question):
        return None
    dimension = _group_dimension(question)
    if dimension is None:
        return None
    grouped = re.search(
        r"每(?:个|种|类)|各(?:个|种|类)?|哪(?:一)?个|哪(?:一)?种|"
        r"按.+(?:汇总|统计|分组)|(?:找出|筛选).+(?:最多|最少|最高|最低)",
        question,
    )
    count_language = re.search(
        r"多少次|次数|数量|人数|患者数|被开出|就诊量|接诊量|开药",
        question,
    )
    if not grouped or not count_language:
        return None
    expression, alias = dimension
    if re.search(r"不同.{0,4}患者|患者人数|患者数", question):
        metric = "COUNT(DISTINCT patient_id)"
        metric_alias = "patient_count"
    elif expression == "drug_prescribed" and re.search(r"开出|处方|开药", question):
        metric = "COUNT(*)"
        metric_alias = "prescription_count"
    else:
        metric = "COUNT(*)"
        metric_alias = "visit_count"
    predicates = _where_for_visit_values(question, domains)
    # The value of the grouping dimension itself is not a filter.
    predicates = [
        predicate
        for predicate in predicates
        if not predicate.startswith(f"{expression} = ")
    ]
    if expression in {"drug_prescribed", "exam"}:
        predicates.insert(0, f"{expression} != ''")
    sql = (
        f"SELECT {expression}, {metric} AS {metric_alias} FROM visits"
    )
    if predicates:
        sql += " WHERE " + " AND ".join(dict.fromkeys(predicates))
    sql += f" GROUP BY {expression}"
    if re.search(r"最多|最高|最大", question):
        sql += f" ORDER BY {metric_alias} DESC, {alias} ASC LIMIT {_limit(question, default=1)}"
    elif re.search(r"最少|最低|最小", question):
        sql += f" ORDER BY {metric_alias} ASC, {alias} ASC LIMIT {_limit(question, default=1)}"
    else:
        sql += f" ORDER BY {alias} ASC"
    return sql


def _compile_scalar(question: str, domains: Domains) -> Optional[str]:
    predicates = _where_for_visit_values(question, domains)
    if re.search(
        r"(?:多少|几)种不同的?(?:疾病|诊断)|"
        r"不同的?(?:疾病|诊断).{0,8}(?:数量|数|多少种|几种)",
        question,
    ):
        expression, alias = "COUNT(DISTINCT disease)", "disease_count"
    elif re.search(r"多少种不同的?(?:处方)?药物|不同的?(?:处方)?药物.{0,4}(?:数量|数)", question):
        expression, alias = "COUNT(DISTINCT drug_prescribed)", "drug_count"
        predicates.insert(0, "drug_prescribed != ''")
    elif re.search(r"多少种不同的?检查|不同的?检查.{0,4}(?:数量|数)", question):
        expression, alias = "COUNT(DISTINCT exam)", "exam_count"
        predicates.insert(0, "exam != ''")
    elif re.search(
        r"(?:多少|几)(?:位|个|名)?不同的?(?:就诊)?患者|"
        r"不同患者.{0,4}(?:人数|数量|多少|几名)",
        question,
    ):
        expression, alias = "COUNT(DISTINCT patient_id)", "patient_count"
    elif re.search(
        r"平均费用|费用.{0,3}平均|平均.{0,6}(?:花费|多少钱)|"
        r"(?:花费|金额).{0,3}平均",
        question,
    ):
        # The established disease-cost rule preserves the historical
        # ROUND(..., 2) result contract used by the public development set.
        # Do not replace it with an unrounded aggregate.
        if _domain_values(question, domains.get("disease", ())):
            return None
        expression, alias = "AVG(cost)", "avg_cost"
    elif re.search(r"费用.{0,5}(?:最高值|最大值)|(?:最高|最大).{0,8}费用.{0,3}(?:值|是多少)", question):
        expression, alias = "MAX(cost)", "max_cost"
    elif re.search(r"费用.{0,5}(?:最低值|最小值)|(?:最低|最小).{0,8}费用.{0,3}(?:值|是多少)", question):
        expression, alias = "MIN(cost)", "min_cost"
    elif re.search(r"费用(?:总额|总和)|总费用", question):
        expression, alias = "SUM(cost)", "total_cost"
    else:
        return None
    sql = f"SELECT {expression} AS {alias} FROM visits"
    if predicates:
        sql += " WHERE " + " AND ".join(dict.fromkeys(predicates))
    return sql


def _is_distinct_list_request(question: str) -> bool:
    """Recognize the controlled distinct visit-value list signature."""

    return bool(
        re.search(r"列出|给出|返回|有哪些", question)
        and re.search(r"不同|去重|所有种类", question)
    )


def _compile_distinct_list(question: str, domains: Domains) -> Optional[str]:
    if not _is_distinct_list_request(question):
        return None
    if (
        _has_unmatched_explicit_visit_value(question, domains)
        or _has_cross_dimension_or(question, domains)
    ):
        return None
    if re.search(r"处方药物|处方药|药物", question):
        column, nonempty = "drug_prescribed", True
    elif re.search(r"疾病|诊断", question):
        column, nonempty = "disease", False
    elif "检查" in question:
        column, nonempty = "exam", True
    elif "科室" in question:
        column, nonempty = "department", False
    else:
        return None
    predicates = _where_for_visit_values(question, domains)
    if nonempty:
        predicates.append(f"{column} != ''")
    sql = f"SELECT DISTINCT {column} FROM visits"
    if predicates:
        sql += " WHERE " + " AND ".join(dict.fromkeys(predicates))
    return sql + f" ORDER BY {column} ASC"


def _compile_ordered_projection(question: str, domains: Domains) -> Optional[str]:
    if "就诊" not in question or not re.search(r"列出|给出|返回", question):
        return None
    if re.search(r"费用(?:从高到低|降序|由高到低)", question):
        direction = "DESC"
    elif re.search(r"费用(?:从低到高|升序|由低到高)", question):
        direction = "ASC"
    else:
        return None
    columns = _visit_columns(_requested_clause(question), default=())
    if not columns:
        return None
    predicates = _where_for_visit_values(question, domains)
    sql = f"SELECT {', '.join(columns)} FROM visits"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return sql + f" ORDER BY cost {direction}, visit_id ASC"


def _compile_patient_list(question: str, domains: Domains) -> Optional[str]:
    if not re.search(_PATIENT_PATTERN, question) or not re.search(
        r"列出|给出|哪些|姓名|名字",
        question,
    ):
        return None
    if not re.search(r"就诊|去过|来院|患过|诊断|看病|看过病|就医", question):
        return None
    if _has_cross_dimension_or(question, domains):
        return None
    columns = _patient_columns(question, default=("p.name",))
    predicates = _where_for_visit_values(question, domains, alias="v")
    sql = (
        f"SELECT DISTINCT {', '.join(columns)} FROM patients AS p "
        "JOIN visits AS v ON v.patient_id = p.patient_id"
    )
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return sql + f" ORDER BY {columns[0]} ASC"


def _requires_terminal_compositional_abstention(
    question: str,
    domains: Domains,
) -> bool:
    """Stop later broad routes from dropping an unsupported boolean scope."""

    if _has_cross_dimension_or(question, domains):
        return True
    if _is_set_operation_intent(question, domains):
        # This compiler only has a proven plan for patient-ID set operations.
        # In particular, do not let a patient-name request fall through to an
        # AND-based visit-list query.
        if (
            not re.search(r"编号|ID", question, re.I)
            or _set_operation_parts(question, domains) is None
        ):
            return True
        # A fully grounded supported set operation owns the boolean scope.
        # Do not reinterpret its negative EXCEPT operand as an independent
        # anti-join request below.
        return False
    if _is_anti_join_request(question):
        values = _value_dimensions(question, domains)
        if (
            _has_unmatched_explicit_visit_value(question, domains)
            or _has_conflicting_visit_values(values)
            or _has_independent_negative_visit_scopes(question)
        ):
            return True
    if (
        _is_distinct_list_request(question)
        and _has_unmatched_explicit_visit_value(question, domains)
    ):
        return True
    return False


def compile_compositional_medical_sql(
    question: str,
    domains: Domains,
) -> Optional[str]:
    """Compile a schema-bound natural-language plan, or abstain safely."""

    q = str(question).strip()
    if not q:
        return None
    # Grounded graph-relation values are not visit predicates.  Reserving them
    # for a KG owner keeps a row grouping from silently dropping the relation.
    if analyze_relation_request(q, domains).has_relation_intent:
        return None
    # A compiler's ``None`` can mean either an unrelated signature or a local
    # abstention.  Preserve known unsupported boolean scopes before broad
    # patient-list fallback can reinterpret their grounded values as AND.
    if _requires_terminal_compositional_abstention(q, domains):
        return None
    # Leave explicit SQL terminology and graph semantics to their specialised
    # protocol compilers.  This layer is for ordinary natural questions.
    if re.search(
        r"\b(?:CTE|LIKE|ROW_NUMBER|DENSE_RANK)\b|"
        r"窗口|图谱|三元组|头实体|尾实体|置信度|精确模式",
        q,
        re.I,
    ):
        return None
    # The established controlled-language compiler already handles numerical
    # predicates, multi-clause analytical contracts, graph terminology and
    # date-bounded questions.  Let it consume those complete contracts rather
    # than allowing this intentionally smaller natural-language layer to drop
    # a qualifier.  A bare Top-K cardinality is the sole numeric operator this
    # layer owns.
    if re.search(r"\d{4}-\d{2}-\d{2}", q):
        return None
    if re.search(
        r"年龄|性别|住院天数|住院时长|临床表现|关系中文名|"
        r"先汇总|再返回|只保留|筛选|排名|名次|相同条件|"
        r"等于该|(?:高于|低于|超过|大于|小于)该|"
        r"高于平均|低于平均|超过平均|平均每|至少比|"
        r"高于所有|低于所有|高于全体|低于全体|"
        r"前\s*\d+.{0,12}(?:中|里|内).{0,12}(?:最|第一)|"
        r"关联的?(?:症状|并发症|禁忌|治疗)|"
        r"费用(?:至少|不低于|不少于|超过|大于|低于|小于)\s*\d|"
        r"累计费用(?:至少|不低于|不少于|超过|大于|低于|小于)\s*\d|"
        r"按患者性别|按患者地区|(?:按|进行).{0,8}分组",
        q,
    ):
        return None
    if re.search(
        r"女性|男性|"
        r"^按(?:科室|疾病|诊断|检查|药物).*(?:统计|汇总)|"
        r"(?:各|每).{0,8}(?:分布|使用次数|平均费用)|"
        r"平均费用.{0,8}(?:最高|最低).{0,8}(?:科室|疾病|诊断)|"
        r"^平均费用(?:是)?多少[？?]?$",
        q,
    ):
        return None
    if "返回既在科室" in q:
        return None
    for compiler in (
        _compile_set_operation,
        _compile_anti_join,
        _compile_patient_having,
        _compile_per_patient_temporal,
        # Aggregate patient-cost Top-K must run before visit-row Top-K: the
        # latter also recognizes the word ``就诊`` but would rank individual
        # rows instead of each patient's requested accumulated spending.
        _compile_patient_cost_top,
        _compile_top_rows,
        _compile_extreme_row,
        _compile_group,
        _compile_scalar,
        _compile_distinct_list,
        _compile_ordered_projection,
        _compile_patient_list,
    ):
        sql = compiler(q, domains)
        if sql:
            return sql
    return None
