"""Semantic contracts for a read-only SQLite NL2SQL candidate gate.

The extractor is deliberately conservative: it records only high-signal
question language, while the validator inspects the candidate's outer SELECT
AST.  Validation never executes, renders, or mutates the candidate SQL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata
from typing import Any, Iterable, Optional, Sequence


class ContractStatus(str, Enum):
    """How much of a question's output shape was grounded semantically."""

    NONE = "none"
    PARTIAL = "partial"
    COMPLETE = "complete"


class RowSetMode(str, Enum):
    """Whether the requested output preserves rows or explicitly deduplicates."""

    PRESERVE = "preserve"
    DISTINCT = "distinct"


@dataclass(frozen=True)
class SemanticContract:
    """Ordered, schema-neutral semantic requirements for one candidate.

    Columns use their raw SQLite names (for example ``patient_id`` or
    ``relation_zh``).  Aggregates are canonical strings such as ``AVG(cost)``
    and ``COUNT(DISTINCT patient_id)``.  Ordering entries are canonical key
    strings with optional ``ASC``/``DESC`` and ``NULLS FIRST``/``NULLS LAST``
    suffixes.
    """

    projections: tuple[str, ...] = ()
    aggregates: tuple[str, ...] = ()
    grouping: tuple[str, ...] = ()
    ordering: tuple[str, ...] = ()
    limit: Optional[int] = None
    confidence: float = 0.0
    status: ContractStatus = ContractStatus.NONE
    # Keep this after every legacy positional field.  Cross-category output
    # order is populated only when extraction can ground every output slot;
    # older/manual contracts retain projections-then-aggregates behavior.
    output_slots: tuple[str, ...] = ()
    # Boolean set semantics are separate from their SQL lowering.  The value
    # is one of any_of/all_of/difference; simple IN/EXISTS lowerings remain
    # valid, while a compound root must use the matching set operator.
    set_semantics: Optional[str] = None
    # SELECT-level deduplication changes result cardinality and must be bound to
    # the question rather than accepted as a harmless rendering detail.
    row_set_mode: RowSetMode = RowSetMode.PRESERVE

    def __post_init__(self) -> None:
        for field_name in (
            "projections",
            "aggregates",
            "output_slots",
            "grouping",
            "ordering",
        ):
            value = getattr(self, field_name)
            if value is None:
                value = ()
            elif isinstance(value, str):
                value = (value,)
            else:
                value = tuple(str(item) for item in value)
            object.__setattr__(self, field_name, value)
        if self.limit is not None:
            object.__setattr__(self, "limit", int(self.limit))
        if self.set_semantics is not None:
            set_semantics = str(self.set_semantics).strip().casefold()
            if set_semantics not in {"any_of", "all_of", "difference"}:
                raise ValueError(f"unsupported set semantics: {self.set_semantics}")
            object.__setattr__(self, "set_semantics", set_semantics)
        if not isinstance(self.row_set_mode, RowSetMode):
            object.__setattr__(
                self,
                "row_set_mode",
                RowSetMode(str(self.row_set_mode).strip().casefold()),
            )
        object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.status, ContractStatus):
            object.__setattr__(self, "status", ContractStatus(self.status))

    @property
    def ordered_outputs(self) -> tuple[str, ...]:
        """Return output slots in their requested cross-category order.

        Older callers construct contracts from separate projection and aggregate
        lists, which historically meant projections preceded aggregates.  Keep
        that safe compatibility fallback when the extractor could not ground a
        complete interleaved answer list.
        """

        return self.output_slots or (*self.projections, *self.aggregates)

    @property
    def ordered_projections(self) -> tuple[str, ...]:
        """Compatibility name emphasizing that projection order is semantic."""

        return self.projections

    @property
    def ordered_aggregates(self) -> tuple[str, ...]:
        """Compatibility name emphasizing that aggregate order is semantic."""

        return self.aggregates


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of candidate validation; reasons are stable, human-readable text."""

    accepted: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.reasons, str):
            object.__setattr__(self, "reasons", (self.reasons,))
        else:
            object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    value: str


_OUTPUT_VERBS = (
    "列出",
    "给出",
    "返回",
    "显示",
    "输出",
    "展示",
    "list",
    "give",
    "return",
    "show",
    "display",
)

# The aliases intentionally describe the four SQLite tables used by this
# project.  Longer aliases win, so "患者姓名" is not split into "姓名".
_PROJECTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "patient_id",
        (
            "患者编号",
            "患者ID",
            "患者 id",
            "病人编号",
            "病人ID",
            "病人 id",
            "患者标识",
            "患者号",
            "编号",
            "patient_id",
            "patient id",
        ),
    ),
    (
        "name",
        (
            "患者姓名",
            "病人姓名",
            "患者名字",
            "病人名字",
            "实体名称",
            "实体名",
            "姓名",
            "名字",
            "name",
        ),
    ),
    ("gender", ("患者性别", "性别", "gender", "sex")),
    ("age", ("患者年龄", "年龄分组", "年龄段", "年龄", "岁数", "age")),
    ("region", ("患者地区", "地区", "城市", "region")),
    (
        "visit_id",
        (
            "就诊编号",
            "就诊ID",
            "就诊 id",
            "访问编号",
            "访问ID",
            "记录编号",
            "visit_id",
            "visit id",
        ),
    ),
    (
        "visit_date",
        ("就诊日期", "访问日期", "日期", "visit_date", "visit date"),
    ),
    ("department", ("就诊科室", "科室", "部门", "department")),
    ("disease", ("主诊断", "诊断", "疾病", "disease")),
    (
        "drug_prescribed",
        ("处方药物", "开具药物", "处方", "用药", "药物", "drug_prescribed", "drug"),
    ),
    ("exam", ("检查项目", "检查", "exam")),
    ("cost", ("就诊费用", "费用", "花费", "开销", "金额", "cost")),
    (
        "los_days",
        ("住院天数", "住院时长", "住院日", "los_days", "length of stay"),
    ),
    ("type", ("实体类型", "类型", "type")),
    ("mention_count", ("提及次数", "提及数", "mention_count")),
    ("head", ("头实体", "头节点", "head")),
    ("head_type", ("头实体类型", "头节点类型", "head_type")),
    ("relation_zh", ("中文关系", "关系中文名", "relation_zh")),
    # ``关系`` is the project's canonical English/code relation column.  Keep
    # the bare mention here (rather than under relation_zh) so local explicit
    # Chinese mentions remain relation_zh even when another clause filters an
    # uppercase relation code.
    ("relation", ("关系英文名", "英文关系", "关系", "relation")),
    ("tail", ("尾实体", "尾节点", "tail")),
    ("tail_type", ("尾实体类型", "尾节点类型", "tail_type")),
    ("confidence", ("置信度", "confidence")),
    ("count", ("证据次数", "观察次数", "证据数", "观察数", "count")),
)

_MEASURE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("visit_date", ("就诊日期", "访问日期", "visit_date", "visit date")),
    ("los_days", ("住院天数", "住院时长", "住院日", "los_days", "length of stay")),
    ("mention_count", ("提及次数", "提及数", "mention_count")),
    ("confidence", ("置信度", "confidence")),
    ("relation_zh", ("中文关系", "关系中文名", "relation_zh")),
    ("cost", ("就诊费用", "费用", "花费", "开销", "金额", "cost")),
    ("age", ("年龄", "岁数", "age")),
    ("count", ("证据次数", "观察次数", "证据数", "观察数", "count")),
)

_GROUP_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("department", ("科室", "部门", "department")),
    ("disease", ("疾病", "诊断", "disease")),
    (
        "drug_prescribed",
        ("处方药物", "处方药", "药物", "药", "drug_prescribed", "drug"),
    ),
    ("exam", ("检查项目", "检查", "exam")),
    ("region", ("地区", "城市", "region")),
    ("gender", ("性别", "gender", "sex")),
    ("age", ("年龄段", "年龄分组", "age group")),
    ("month", ("月份", "月度", "按月", "每月", "month")),
    ("year", ("年份", "年度", "按年", "每年", "year")),
    ("visit_date", ("就诊日期", "日期", "visit_date")),
    ("head_type", ("头实体类型", "head_type")),
    ("tail_type", ("尾实体类型", "tail_type")),
    (
        "relation_zh",
        ("关系中文名", "中文关系", "relation_zh"),
    ),
    (
        "relation",
        ("关系英文名", "英文关系", "关系", "relation"),
    ),
    ("head", ("头实体", "头节点", "head")),
    ("tail", ("尾实体", "尾节点", "tail")),
    ("type", ("实体类型", "type")),
)

# COUNT(DISTINCT ...) is inferred only when an explicit deduplication marker is
# attached to one of these schema-backed nouns.  Keeping this list separate
# from the broad projection aliases prevents a bare ``不同`` elsewhere in the
# question from being mistaken for either an aggregate or SELECT DISTINCT.
_DISTINCT_COUNT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("patient_id", ("患者编号", "病人编号", "患者", "病人")),
    ("visit_id", ("就诊编号", "访问编号", "就诊记录")),
    ("department", ("就诊科室", "科室", "部门")),
    ("disease", ("主诊断", "诊断", "疾病")),
    ("drug_prescribed", ("处方药物", "处方药", "药物")),
    ("exam", ("检查项目", "检查")),
    ("region", ("患者地区", "地区", "城市")),
    ("gender", ("患者性别", "性别")),
    ("type", ("实体类型",)),
    ("head_type", ("头实体类型", "头节点类型")),
    ("tail_type", ("尾实体类型", "尾节点类型")),
    ("head", ("头实体", "头节点")),
    ("tail", ("尾实体", "尾节点")),
    ("relation", ("关系英文名", "英文关系", "关系")),
    ("relation_zh", ("关系中文名", "中文关系")),
)

_FUNCTION_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AVG", ("平均值", "平均", "均值", "人均", "average", "avg", "mean")),
    ("SUM", ("总和", "总额", "合计", "累计", "总计", "总", "total", "sum")),
    ("MAX", ("最大值", "最高", "最大", "最多", "max", "maximum")),
    ("MIN", ("最小值", "最低", "最小", "最少", "min", "minimum")),
)

_DIRECTION_WORDS = (
    ("DESC", ("从高到低", "由高到低", "降序", "倒序", "descending", "desc")),
    ("ASC", ("从低到高", "由低到高", "升序", "正序", "ascending", "asc")),
)

_CONNECTOR_RE = re.compile(r"[、，,；;：:\s和与及以及并附其各个每位种名条项行的]")


def _normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(question)).strip())


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    if re.fullmatch(r"[A-Za-z0-9_ ]+", alias):
        return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    return escaped


def _all_alias_matches(
    text: str,
    definitions: Sequence[tuple[str, Sequence[str]]],
) -> list[_Span]:
    matches: list[_Span] = []
    for canonical, aliases in definitions:
        for alias in sorted(aliases, key=len, reverse=True):
            if not alias:
                continue
            pattern = re.compile(_alias_pattern(alias), re.IGNORECASE)
            matches.extend(
                _Span(match.start(), match.end(), canonical)
                for match in pattern.finditer(text)
            )
    return _non_overlapping(matches)


def _non_overlapping(matches: Iterable[_Span]) -> list[_Span]:
    selected: list[_Span] = []
    for item in sorted(matches, key=lambda span: (span.start, -(span.end - span.start), span.value)):
        if any(item.start < used.end and item.end > used.start for used in selected):
            continue
        selected.append(item)
    return sorted(selected, key=lambda span: (span.start, span.end))


def _cn_int(value: str) -> Optional[int]:
    value = value.strip()
    if value.isdigit():
        return int(value)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "零": 0}
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if value in digits:
        return digits[value]
    return None


def _has_controlled_row_filter(text: str) -> bool:
    """Recognize only an explicit comparison or a closed date range."""

    comparison = re.search(
        r"(?:为|等于|恰好为|至少|不低于|不少于|不超过|不高于|"
        r"高于|低于|大于|小于|>=|<=|=|>|<)\s*"
        r"(?:[-+]?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_\-]*|"
        r"[^\s，,;；。]{1,24})",
        text,
        re.IGNORECASE,
    )
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    date_range = len(dates) >= 2 and (
        re.search(
            r"(?:到|至|—|–|~|～|-).{0,16}(?:内|之间|范围)",
            text,
        )
        or re.search(
            r"(?:从|自)\s*\d{4}-\d{2}-\d{2}\s*(?:到|至)\s*"
            r"\d{4}-\d{2}-\d{2}\s*"
            r"(?:[（(](?:含首尾|含边界)[）)]|(?:含首尾|含边界))",
            text,
        )
    )
    return bool(comparison or date_range)


def _complete_field_list(text: str) -> Optional[str]:
    """Return a fully grounded list containing at least two schema fields."""

    cleaned = re.sub(
        r"\s*(?:字段)?(?:清单|列表|明细)(?:列表|清单)?\s*$",
        "",
        text.strip(" ：:,，"),
    ).strip(" ：:,，")
    spans = _all_alias_matches(cleaned, _PROJECTION_ALIASES)
    if len({span.value for span in spans}) < 2:
        return None
    if _meaningful_residual(cleaned, spans):
        return None
    return cleaned


def _extract_filtered_field_list(question: str) -> Optional[str]:
    """Treat tightly controlled 统计/汇总 field lists as row outputs.

    The verbs remain aggregate/grouping language everywhere else.  This branch
    is deliberately gated by a real filter and a complete multi-field list so
    a phrase such as ``汇总费用`` cannot silently become a row contract.
    """

    if not re.search(r"(?:统计|汇总)", question):
        return None

    for separator in reversed(list(re.finditer(r"的", question))):
        prefix = question[:separator.start()]
        suffix = re.split(r"[。！？?!；;\n]", question[separator.end():], maxsplit=1)[0]
        if not _has_controlled_row_filter(prefix):
            continue
        fields = _complete_field_list(suffix)
        if fields is not None:
            return fields

    marker = re.search(
        r"(?:字段)?(?:清单|列表|明细)(?:列表|清单)?\s*[：:]\s*"
        r"(?P<fields>[^。！？?!；;\n]+)",
        question,
    )
    if marker and _has_controlled_row_filter(question[:marker.start()]):
        return _complete_field_list(marker.group("fields"))
    return None


def _extract_output_clause(question: str) -> tuple[str, bool]:
    pattern = re.compile(
        r"(?:" + "|".join(map(re.escape, _OUTPUT_VERBS)) + r")\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(question))
    if not matches:
        filtered_fields = _extract_filtered_field_list(question)
        if filtered_fields is not None:
            return filtered_fields, True
        if _has_partitioned_row_selection(question):
            for separator in reversed(list(re.finditer(r"的", question))):
                suffix = re.split(
                    r"[。！？?!；;\n]",
                    question[separator.end():],
                    maxsplit=1,
                )[0]
                fields = _complete_field_list(suffix)
                if fields is not None:
                    return fields, True
        if re.search(
            r"(?:关系\s*(?:CODE|编码|代码)|relation\s*(?:code|id))",
            question,
            re.IGNORECASE,
        ):
            for separator in reversed(list(re.finditer(r"的", question))):
                suffix = re.split(
                    r"[。！？?!；;\n]",
                    question[separator.end():],
                    maxsplit=1,
                )[0].strip(" ：:,，")
                if re.fullmatch(
                    r"(?:头|尾)实体(?:的)?(?:名称|名字|实体名)?\s*"
                    r"(?:、|和|与|及|以及)\s*(?:其\s*)?"
                    r"(?:(?:头|尾)实体(?:的)?)?类型",
                    suffix,
                ):
                    return suffix, True
        # Compact controlled-language queries can put a complete answer list
        # after the final predicate without a separate RETURN verb:
        # "查询科室为...且费用高于...的就诊编号、患者编号、诊断和费用".
        # Treat only a query-led, comparison-qualified suffix whose residual is
        # fully explained by known projection aliases as an output clause.  The
        # guard keeps filter nouns (notably "平均费用") out of the select-list
        # contract without guessing an answer shape for ordinary noun phrases.
        if re.match(r"\s*(?:查询|查找|检索|筛选|找出)", question):
            for separator in reversed(list(re.finditer(r"的", question))):
                prefix = question[:separator.start()]
                suffix = question[separator.end():]
                suffix = re.split(r"[。！？?!；;\n]", suffix, maxsplit=1)[0]
                projection_spans = _all_alias_matches(
                    suffix,
                    _PROJECTION_ALIASES,
                )
                if not projection_spans:
                    continue
                if not re.search(
                    r"(?:为|等于|恰好为|至少|不低于|不少于|不超过|不高于|"
                    r"大于|小于|高于|低于|>=|<=|>|<)",
                    prefix,
                    re.IGNORECASE,
                ):
                    continue
                if _meaningful_residual(suffix, projection_spans):
                    continue
                return suffix.strip(" ：:，,"), True
        name_question = re.search(
            r"(?:患者|病人)?(?:叫|是)?(?:什么|啥)(?:名字|姓名)|"
            r"(?:患者|病人)(?:的)?(?:名字|姓名)(?:是)?(?:什么|啥)",
            question,
        )
        if name_question:
            return "患者名字", True
        interrogative = re.search(
            r"的(?P<answer>[^。！？?!；;\n]{1,64}?)"
            r"(?:分别)?(?:是|为)?(?:什么|哪些|多少)[？?]?$",
            question,
        )
        interrogative_aliases = (
            _all_alias_matches(
                interrogative.group("answer"),
                _PROJECTION_ALIASES,
            )
            if interrogative
            else []
        )
        if interrogative and len(interrogative_aliases) >= 2:
            return interrogative.group("answer").strip(" ：:，,"), True
        return "", False
    tail = question[matches[-1].end():]
    tail = re.split(r"[。！？?!；;\n]", tail, maxsplit=1)[0]
    # Ordering is a separate contract slot.  Do not let its column nouns turn
    # into projections in the final answer clause.
    order_marker = re.search(
        r"(?:[，,]\s*|\s+)(?:(?:并|再|然后|最后)\s*)?(?:按|按照|依据|依照)\s*",
        tail,
    )
    if order_marker:
        tail = tail[:order_marker.start()]
    # Public questions often put filters between RETURN and the actual answer
    # list: "返回头实体类型为...且置信度至少...的头实体、尾实体和观察次数".
    # Do not reinterpret those grounded predicates (or their literal values)
    # as projected columns.  Only strip a prefix with an explicit comparison,
    # and only when the suffix itself contains a recognizable answer slot.
    for separator in reversed(list(re.finditer(r"的", tail))):
        prefix = tail[:separator.start()]
        suffix = tail[separator.end():]
        comparison_prefix = re.search(
            r"(?:为|等于|恰好为|至少|不低于|不少于|不超过|不高于|"
            r"大于|小于|高于|低于|>=|<=|>|<)",
            prefix,
            re.IGNORECASE,
        )
        required_set_prefix = re.search(
            r"既.{0,64}又|同时.{0,32}(?:且|并且)|"
            r"(?:在|于).{0,32}(?:就诊过|出现过)",
            prefix,
        )
        date_filter_prefix = bool(
            len(re.findall(r"\d{4}-\d{2}-\d{2}", prefix)) >= 2
            and re.search(r"(?:到|至|—|–|~|～|-).{0,16}(?:内|之间|范围)", prefix)
        )
        ranking_prefix = re.search(
            r"(?:最高|最低|最大|最小|最多|最少).{0,12}"
            r"(?:\d+|[一二两三四五六七八九十]+)\s*"
            r"(?:个|位|名|条|次|种|项|行).{0,8}(?:就诊|记录)?",
            prefix,
        )
        if not (
            comparison_prefix
            or required_set_prefix
            or ranking_prefix
            or date_filter_prefix
        ):
            continue
        if _all_alias_matches(suffix, _PROJECTION_ALIASES) or _extract_natural_aggregates(
            suffix
        ):
            tail = suffix
            break
    return tail.strip(" ：:，,"), True


def _parse_direct_aggregate(text: str) -> list[_Span]:
    pattern = re.compile(
        r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*"
        r"(?:(DISTINCT)\s+)?(\*|[A-Za-z_][A-Za-z0-9_.]*)\s*\)",
        re.IGNORECASE,
    )
    spans: list[_Span] = []
    for match in pattern.finditer(text):
        function = match.group(1).upper()
        distinct = bool(match.group(2))
        argument = match.group(3)
        argument = "*" if argument == "*" else _column_text(argument)
        if function == "COUNT" and distinct:
            value = f"COUNT(DISTINCT {argument})"
        else:
            value = f"{function}({argument})"
        spans.append(_Span(match.start(), match.end(), value))
    return spans


def _looks_like_ranking_word(text: str, end: int) -> bool:
    after = text[end:]
    return bool(
        re.match(
            r"\s*(?:的\s*)?(?:前\s*(?:\d+|[一二两三四五六七八九十]+)|"
            r"(?:\d+|[一二两三四五六七八九十]+)\s*"
            r"(?:个|位|名|条|次|种|项|行)|"
            r"top[-\s]*\d+|排名|第\s*\d+|个|种|名|"
            r"(?:那|该)?(?:次|条)?(?:就诊|记录))",
            after,
            re.IGNORECASE,
        )
    )


def _function_for_word(word: str) -> Optional[str]:
    lowered = word.casefold()
    for function, words in _FUNCTION_WORDS:
        if any(lowered == candidate.casefold() for candidate in words):
            return function
    return None


def _aggregate_words_are_adjacent(text: str, left_end: int, right_start: int) -> bool:
    """Allow only function-phrase glue between a function and its measure."""

    if right_start < left_end:
        return False
    return re.fullmatch(r"\s*(?:的|每|每个|每项)?\s*", text[left_end:right_start]) is not None


def _english_count_entity_matches(
    text: str,
    entity_pattern: str,
) -> Iterable[re.Match[str]]:
    """Yield local ``number/count of ... entities`` phrases.

    Public English questions may put a short filter noun phrase before the
    counted entity (for example ``number of 高血压 primary disease patients``).
    The phrase must not, however, cross another requested measure or a relative
    clause: ``number of days for which patient ...`` asks for ``los_days`` and
    is not a patient count.
    """

    pattern = re.compile(
        rf"\b(?:number|count)\s+of\b"
        rf"(?P<middle>[^,.;:?!\n]{{0,64}}?)"
        rf"\b(?:{entity_pattern})\b",
        re.IGNORECASE,
    )
    clause_breaker = re.compile(
        r"\b(?:times?|days?|weeks?|months?|years?|"
        r"for\s+which|who|whose|where|when)\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        if clause_breaker.search(match.group("middle")):
            continue
        yield match


def _extreme_phrase_selects_row(text: str, end: int) -> bool:
    """Distinguish a scalar extreme from attributes of the extreme row."""

    return bool(
        re.match(
            r"\s*(?:的\s*)?(?:对应的?\s*)?(?:那|该)?"
            r"(?:次|条|笔)?(?:就诊|记录|三元组|患者|病人)",
            text[end:],
        )
    )


def _extract_natural_aggregates(text: str) -> list[_Span]:
    spans: list[_Span] = []

    # ``类型转换次数`` is a row count only for the fully named KG transition
    # tuple.  Keeping the proof local prevents generic "transition" language,
    # distinct transition kinds, or the physical observation-count column from
    # being silently rewritten to COUNT(*).
    for match in re.finditer(r"类型转换次数", text):
        if re.search(
            r"(?:不同|去重|不重复|唯一)\s*(?:的\s*)?$",
            text[max(0, match.start() - 12):match.start()],
        ):
            continue
        graph_start = text.rfind("知识图谱", 0, match.start())
        if graph_start < 0 or match.start() - graph_start > 128:
            continue
        context = text[graph_start:match.start()]
        if re.search(r"[。！？!?；;]", context):
            continue
        dimensions = {
            span.value for span in _all_alias_matches(context, _GROUP_PATTERNS)
        }
        if {"head_type", "relation", "tail_type"}.issubset(dimensions):
            spans.append(_Span(match.start(), match.end(), "COUNT(*)"))

    # These two controlled phrases place the SUM word inside the natural
    # measure name, so the ordinary adjacent ``function + measure`` scanner
    # cannot recover them.  They denote kg_entities.mention_count, never a
    # count of rows.
    for match in re.finditer(
        r"(?:提及总次数|总提及次数|提及次数(?:总和|合计|累计))",
        text,
        re.IGNORECASE,
    ):
        spans.append(_Span(match.start(), match.end(), "SUM(mention_count)"))
    for match in re.finditer(r"证据(?:次数|数)?(?:合计|总和|累计)", text):
        spans.append(_Span(match.start(), match.end(), "SUM(count)"))

    # Controlled scalar extrema whose modifier is part of the measure phrase.
    # Do not claim an aggregate when the phrase instead selects a row and asks
    # for that row's attributes (for example "各科室费用最高的就诊编号").
    for match in re.finditer(r"(?:最高|最大)(?:的)?单次(?:就诊)?费用", text):
        if not _extreme_phrase_selects_row(text, match.end()):
            spans.append(_Span(match.start(), match.end(), "MAX(cost)"))
    for match in re.finditer(r"(?:最长(?:的)?住院天数|住院天数最长)", text):
        if not _extreme_phrase_selects_row(text, match.end()):
            spans.append(_Span(match.start(), match.end(), "MAX(los_days)"))

    # Bind natural distinct counts to an explicit, whitelisted counted noun.
    # A request such as "列出不同头实体" therefore remains SELECT DISTINCT,
    # while "不同头实体数" is COUNT(DISTINCT head).
    for canonical, aliases in _DISTINCT_COUNT_ALIASES:
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = re.compile(
                rf"(?:不同|去重|不重复|唯一)\s*{_alias_pattern(alias)}\s*"
                r"(?:总数|数量|个数|种数|数)",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                spans.append(
                    _Span(
                        match.start(),
                        match.end(),
                        f"COUNT(DISTINCT {canonical})",
                    )
                )

    # Scalar extrema may insert a classifier between the function word and
    # measure ("最低的一笔就诊费用是多少").  This asks for the value, not
    # for attributes of the selected visit row.
    for match in re.finditer(
        r"(?:最高|最大)(?:的)?(?:一笔|一项|一次)?(?:就诊)?费用"
        r"(?:的)?(?:最高值|最大值|值)?(?:是|为)?(?:多少|什么)",
        text,
    ):
        spans.append(_Span(match.start(), match.end(), "MAX(cost)"))
    for match in re.finditer(
        r"(?:最低|最小)(?:的)?(?:一笔|一项|一次)?(?:就诊)?费用"
        r"(?:的)?(?:最低值|最小值|值)?(?:是|为)?(?:多少|什么)",
        text,
    ):
        spans.append(_Span(match.start(), match.end(), "MIN(cost)"))

    for match in re.finditer(r"(?:首次|最早)(?:一次)?就诊日期", text):
        spans.append(_Span(match.start(), match.end(), "MIN(visit_date)"))
    for match in re.finditer(r"(?:最近一次|最新|最后一次)就诊日期", text):
        spans.append(_Span(match.start(), match.end(), "MAX(visit_date)"))

    distinct_count = re.compile(
        r"(?:不同|去重|不重复|唯一)?\s*(?:患者|病人)\s*"
        r"(?:总数|数量|人数|数)",
        re.IGNORECASE,
    )
    for match in distinct_count.finditer(text):
        spans.append(_Span(match.start(), match.end(), "COUNT(DISTINCT patient_id)"))

    count_all = re.compile(
        r"(?:就诊|接诊|访问|记录|病例|三元组|实体|关系|开药|开出|处方|"
        r"药物使用|检查(?:项目)?(?:的)?使用)"
        r"(?:的)?\s*"
        r"(?:总)?\s*(?:次数|数量|数|量|条数)",
        re.IGNORECASE,
    )
    for match in count_all.finditer(text):
        spans.append(_Span(match.start(), match.end(), "COUNT(*)"))

    # A bare "多少患者/几位患者" is the same distinct-patient measure.
    for match in re.finditer(r"(?:多少|几)\s*(?:名|位)?\s*(?:患者|病人)", text):
        spans.append(_Span(match.start(), match.end(), "COUNT(DISTINCT patient_id)"))
    for match in _english_count_entity_matches(text, r"patients?"):
        spans.append(_Span(match.start(), match.end(), "COUNT(DISTINCT patient_id)"))
    for match in re.finditer(r"(?:多少|几)\s*(?:次|条)\s*(?:就诊|记录|三元组)?", text):
        spans.append(_Span(match.start(), match.end(), "COUNT(*)"))
    for match in _english_count_entity_matches(text, r"times?|visits?|records?"):
        spans.append(_Span(match.start(), match.end(), "COUNT(*)"))

    measure_matches = _all_alias_matches(text, _MEASURE_ALIASES)
    for function, words in _FUNCTION_WORDS:
        for word in sorted(words, key=len, reverse=True):
            if not word:
                continue
            function_pattern = re.compile(_alias_pattern(word), re.IGNORECASE)
            for function_match in function_pattern.finditer(text):
                if (
                    function == "SUM"
                    and word == "总"
                    and function_match.start() > 0
                    and text[function_match.start() - 1:function_match.start()] == "汇"
                ):
                    # "汇总" is an operation verb; its final character is not an
                    # independently stated SUM function word.
                    continue
                # Function word followed by a measure: "平均费用".
                measure = next(
                    (
                        item
                        for item in measure_matches
                        if item.start >= function_match.end()
                        and _aggregate_words_are_adjacent(text, function_match.end(), item.start)
                    ),
                    None,
                )
                if measure is not None:
                    end = measure.end
                    if function in {"MAX", "MIN"} and _looks_like_ranking_word(text, end):
                        continue
                    spans.append(_Span(function_match.start(), end, f"{function}({measure.value})"))

                # And a measure followed by a function word: "费用最高".
                before_measure = [
                    item for item in measure_matches if item.end <= function_match.start()
                ]
                measure = next(
                    (
                        item
                        for item in reversed(before_measure)
                        if _aggregate_words_are_adjacent(text, item.end, function_match.start())
                    ),
                    None,
                )
                if measure is not None:
                    if function in {"MAX", "MIN"} and _looks_like_ranking_word(text, function_match.end()):
                        continue
                    spans.append(
                        _Span(measure.start, function_match.end(), f"{function}({measure.value})")
                    )

    return spans


def _extract_aggregates(text: str, *, grouped: bool = False) -> list[_Span]:
    spans = _parse_direct_aggregate(text) + _extract_natural_aggregates(text)
    selected = _non_overlapping(spans)
    window_analysis = bool(
        re.search(r"连续排名|排序编号|累计分布", text, re.IGNORECASE)
    )
    if (
        not selected
        and grouped
        and not window_analysis
        and "类型转换" not in text
        and re.search(r"统计|计数|数量|分布|趋势|排名|多少", text, re.IGNORECASE)
    ):
        # The project query language uses grouped "统计各X" as COUNT(*).
        selected = [_Span(len(text), len(text), "COUNT(*)")]
    return selected


def _meaningful_residual(text: str, spans: Sequence[_Span]) -> bool:
    if not text:
        return False
    covered = [False] * len(text)
    for span in spans:
        for index in range(max(0, span.start), min(len(text), span.end)):
            covered[index] = True
    residual = "".join(char for index, char in enumerate(text) if not covered[index])
    residual = _CONNECTOR_RE.sub("", residual)
    residual = re.sub(r"(?:患者|病人|就诊|访问|知识图谱|字段|信息|结果|数据)", "", residual)
    residual = re.sub(r"\d+", "", residual)
    return bool(residual.strip())


def _controlled_multidimensional_grouping(question: str) -> list[str]:
    """Extract only explicit, fully enumerated multi-column group clauses."""

    result: list[str] = []
    if re.search(
        r"头\s*尾(?:实体)?类型组合|头尾(?:实体)?类型组合|"
        r"头(?:实体)?类型\s*(?:、|和|与|及)\s*尾(?:实体)?类型组合",
        question,
    ) and re.search(
        r"(?:统计|汇总|计数|数量|次数|三元组数|每种组合|各组合)",
        question,
    ):
        return ["head_type", "tail_type"]

    patterns = (
        re.compile(
            r"(?:按|按照|依据|依照)\s*"
            r"(?P<dimensions>[^。；;,，]{1,72}?)\s*"
            r"(?:进行)?(?:分组(?:统计|汇总|计数)?|"
            r"(?:统计|汇总|计数)(?:分组)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:统计|汇总|计数)\s*"
            r"(?P<dimensions>[^。；;,，]{1,48}?)\s*组合(?:的)?"
            r"(?:就诊|记录|三元组|实体|患者)?(?:次数|数量|数|量)",
            re.IGNORECASE,
        ),
    )
    for match in (item for pattern in patterns for item in pattern.finditer(question)):
        dimensions = match.group("dimensions").strip()
        if re.fullmatch(
            r"(?:头(?:实体)?\s*[、和与及]?\s*尾(?:实体)?|"
            r"头尾(?:实体)?)类型(?:组合|维度)?",
            dimensions,
        ):
            return ["head_type", "tail_type"]

        spans = _all_alias_matches(dimensions, _GROUP_PATTERNS)
        values: list[str] = []
        for span in spans:
            if span.value not in values:
                values.append(span.value)
        if len(values) < 2:
            continue

        covered = [False] * len(dimensions)
        for span in spans:
            for index in range(span.start, span.end):
                covered[index] = True
        residual = "".join(
            char for index, char in enumerate(dimensions) if not covered[index]
        )
        residual = re.sub(r"[、，,\s和与及以及]", "", residual)
        residual = re.sub(r"(?:患者|病人|就诊|实体|字段|维度|组合)", "", residual)
        if residual:
            continue
        result = values
        break
    return result


def _row_selection_follows_dimension(text: str) -> bool:
    """Whether text immediately selects a row inside the preceding partition."""

    lead = r"\s*(?:分区)?\s*(?:中|内|里|下|的)?\s*(?:分区)?\s*(?:取|选取|选择|保留|返回)?\s*"
    row_target = r"(?:就诊|记录|三元组|数据行|行)"
    temporal = (
        r"(?:第\s*(?:1|一)\s*(?:条|次)?|第一\s*(?:条|次)|"
        r"首条|首次|最早(?:一)?(?:条|次)?|"
        r"最近(?:一)?(?:条|次)?|最新(?:一)?(?:条|次)?|"
        r"最后(?:一)?(?:条|次)?|末次|最晚(?:一)?(?:条|次)?)"
    )
    metric = (
        r"(?:单次)?(?:费用|花费|金额|住院天数|住院时长|"
        r"置信度|证据次数|观察次数|提及次数|就诊日期)"
    )
    extreme = r"(?:最高|最低|最大|最小|最多|最少|最长|最短|最早|最晚|最新)"
    return bool(
        re.match(rf"{lead}第\s*(?:1|一)\s*行", text)
        or re.match(rf"{lead}{temporal}.{{0,10}}{row_target}", text)
        or re.match(rf"{lead}{metric}.{{0,6}}{extreme}.{{0,8}}的?\s*{row_target}", text)
        or re.match(rf"{lead}{extreme}.{{0,4}}{metric}.{{0,8}}的?\s*{row_target}", text)
        or re.match(
            r"\s*(?:中|内|里|下|的)?\s*(?:找出|筛选|选出).{0,40}?"
            r"(?:第\s*(?:1|一)\s*(?:条|次|行)?|最近|最新|最早|最后|末次|"
            r"(?:等于|达到).{0,16}(?:最高|最低|最大|最小|最长|最短))",
            text,
        )
        or re.match(
            r"\s*的?\s*(?:就诊|记录).{0,32}?按.{0,24}?"
            r"(?:升序|降序|倒序|正序).{0,12}?编号.{0,32}?"
            r"(?:列出|筛选|保留).{0,12}?编号\s*(?:为|=)\s*(?:1|一)",
            text,
        )
        or re.match(
            r"\s*(?:中|内|里|下|的)?.{0,72}?"
            r"(?:等于|达到).{0,24}?"
            r"(?:最高|最低|最大|最小|最长|最短).{0,24}?的?\s*"
            r"(?:就诊编号|记录编号|就诊|记录)",
            text,
        )
    )


def _is_partitioned_row_selection(question: str, start: int, end: int) -> bool:
    before = question[max(0, start - 18):start]
    partition_cue = re.search(
        r"(?:各(?:个|种|类|位)?|每(?:个|种|类|位)?|"
        r"按|按照|依据|依照)\s*$",
        before,
    )
    return bool(partition_cue and _row_selection_follows_dimension(question[end:]))


def _has_partitioned_row_selection(question: str) -> bool:
    for _, aliases in _GROUP_PATTERNS:
        for alias in sorted(aliases, key=len, reverse=True):
            for match in re.finditer(_alias_pattern(alias), question, re.IGNORECASE):
                if _is_partitioned_row_selection(question, match.start(), match.end()):
                    return True
    for match in re.finditer(
        r"(?:各(?:位|个)?|每(?:位|个)?|按|按照|依据|依照)\s*"
        r"(?:患者|病人)",
        question,
    ):
        if _row_selection_follows_dimension(question[match.end():]):
            return True
    return False


def _has_complete_cost_band_grouping(question: str) -> bool:
    """Recognize only the complete two-band cost grouping instruction."""

    boundary = re.search(
        r"以\s*费用\s*\d+(?:\.\d+)?\s*(?:元)?\s*为界",
        question,
    )
    separator = r"(?:、|和|与|及|/)"
    lower_then_higher = re.search(
        rf"分成\s*(?:"
        rf"较低费用档\s*{separator}\s*较高费用档|"
        rf"较低\s*{separator}\s*较高\s*(?:两个)?费用档"
        rf")",
        question,
    )
    higher_then_lower = re.search(
        rf"分成\s*(?:"
        rf"较高费用档\s*{separator}\s*较低费用档|"
        rf"较高\s*{separator}\s*较低\s*(?:两个)?费用档"
        rf")",
        question,
    )
    operation = re.search(r"各档\s*(?:分别\s*)?(?:统计|汇总|比较)", question)
    bands = lower_then_higher or higher_then_lower
    if boundary is None or bands is None or operation is None:
        return False
    if not (boundary.end() <= bands.start() and bands.end() <= operation.start()):
        return False
    return re.search(
        r"[。！？!?；;]",
        question[boundary.start():operation.end()],
    ) is None


def _extract_grouping(question: str) -> list[str]:
    q = question
    if _has_complete_cost_band_grouping(q):
        return ["cost"]
    controlled_dimensions = _controlled_multidimensional_grouping(q)
    if controlled_dimensions:
        return controlled_dimensions
    if _has_partitioned_row_selection(q):
        # The dimension identifies a window/correlated partition, not the
        # cardinality of the outer result.  Inner mechanics are validated by
        # AST lineage and must never be laundered into an outer GROUP BY.
        return []
    candidates: list[tuple[int, int, str]] = []
    for canonical, aliases in _GROUP_PATTERNS:
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = re.compile(_alias_pattern(alias), re.IGNORECASE)
            for match in pattern.finditer(q):
                before = q[max(0, match.start() - 10):match.start()]
                after = q[match.end():match.end() + 48]
                is_each = bool(
                    re.search(
                        r"(?:各(?:个|种|类)?|每(?:个|种|类|位)?|分别)\s*$",
                        before,
                    )
                )
                is_which = bool(
                    re.search(r"哪(?:一)?(?:个|种|类)?\s*$", before)
                )
                is_by = bool(re.search(r"(?:按|按照|依据|依照)[^。；,，]{0,8}$", before))
                is_clause = bool(re.search(r"(?:分组|统计|分布|计数|返回)", after))
                is_metric_question = bool(
                    re.search(
                        r"(?:人数|患者数|次数|数量|数|量|就诊量|接诊量|"
                        r"开药次数|使用次数|多少|几次)",
                        before + after,
                    )
                )
                is_order_only = bool(
                    re.match(r"\s*(?:升序|降序|正序|倒序|排序|排列|优先|在前|在后)", after)
                )
                is_superlative_result = bool(
                    re.search(
                        r"(?:最多|最少|最高|最低|最大|最小)(?:的)?\s*$",
                        before,
                    )
                )
                is_topk_result = bool(
                    re.search(
                        r"(?:最多|最少|最高|最低|最大|最小)?(?:的)?\s*"
                        r"前\s*(?:\d+|[一二两三四五六七八九十百千万]+)\s*"
                        r"(?:个|位|名|条|次|种|项|行|家)?(?:的)?\s*$",
                        before,
                    )
                )
                is_explicit_temporal_group = bool(
                    canonical in {"month", "year"}
                    and alias in {"按月", "每月", "按年", "每年"}
                    and not (
                        alias in {"按年", "每年"}
                        and q[match.end():].startswith("龄")
                    )
                    and is_clause
                )
                before_wide = q[max(0, match.start() - 48):match.start()]
                is_having_target = bool(
                    re.search(
                        r"(?:就诊|记录|开具|开药|处方|使用)?"
                        r"(?:次数|数量|数|人数)|费用总额|总费用|平均费用|平均住院天数",
                        before_wide,
                    )
                    and re.search(
                        r"(?:至少|不低于|不少于|超过|高于|大于|不超过|至多|"
                        r"低于|小于|少于|等于|恰好为|≥|≤|>=|<=|>|<|=)"
                        r"\s*\d+(?:\.\d+)?\s*(?:次|条|人|位|名|元|天)?\s*的?\s*$",
                        before_wide,
                    )
                )
                is_partitioned_row_selection = _is_partitioned_row_selection(
                    q,
                    match.start(),
                    match.end(),
                )
                if is_partitioned_row_selection:
                    continue
                if (
                    is_each
                    or (is_which and is_metric_question)
                    or is_superlative_result
                    or is_topk_result
                    or is_explicit_temporal_group
                    or is_having_target
                    or (is_by and not is_order_only)
                ):
                    candidates.append((match.start(), match.end(), canonical))
    # Explicit patient grouping is useful but should not treat every mention of
    # 患者 as a dimension.  Only accept it under 每位/每个患者 phrasing.
    for match in re.finditer(r"(?:每位|每个|各(?:位|个)?)\s*患者", q):
        after = q[match.end():match.end() + 10]
        if _row_selection_follows_dimension(q[match.end():]) or re.match(
            r"\s*(?:最近|最新|最后|末次|最晚|首次|第一次|最早)\s*一?次?",
            after,
        ):
            # "每位患者最近一次..." is a per-patient row-selection request,
            # not a SQL GROUP BY contract.  Its patient key is still inferred
            # as an output below.
            continue
        candidates.append((match.start(), match.end(), "patient_id"))
    for match in re.finditer(
        r"(?:按|按照|依据|依照)\s*(?:每位|每个|各(?:位|个)?)?\s*"
        r"(?:患者|病人).{0,8}(?:汇总|统计|分组)",
        q,
    ):
        if re.search(
            r"(?:患者|病人)\s*(?:性别|地区|区域|城市|年龄|年龄段)",
            match.group(0),
        ):
            # "按患者性别分组" groups by gender, not by patient entity.
            continue
        candidates.append((match.start(), match.end(), "patient_id"))
    selected: list[tuple[int, int, str]] = []
    for start, end, value in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < used_end and end > used_start for used_start, used_end, _ in selected):
            continue
        selected.append((start, end, value))
    result: list[str] = []
    for _, _, value in sorted(selected):
        if value not in result:
            result.append(value)
    return result


def _split_order_keys(text: str) -> list[str]:
    parts = re.split(r"[、，,]|\s+(?:和|与|及)\s*|(?:和|与|及)", text)
    return [part.strip(" ：:()（）") for part in parts if part.strip(" ：:()（）")]


def _semantic_key_from_text(text: str, aggregates: Sequence[str]) -> Optional[str]:
    cleaned = text.strip(" ：:()（）")
    cleaned = re.sub(r"^(?:若|如果|相同|同样|并列|再|然后|其次)\s*", "", cleaned)
    cleaned = re.sub(r"^(?:按|按照|依据|依照)\s*", "", cleaned)
    cleaned = re.sub(r"(?:排序|排列)$", "", cleaned).strip()

    direct = _extract_aggregates(cleaned)
    if direct:
        return direct[0].value

    # If the question says only "费用" while one aggregate measure is already
    # explicit, order the requested measure rather than the raw column.
    raw_measure = next(
        (item.value for item in _all_alias_matches(cleaned, _MEASURE_ALIASES)),
        None,
    )
    if raw_measure is not None:
        matching = [item for item in aggregates if f"({raw_measure})" in item]
        if len(matching) == 1:
            return matching[0]

    if len(aggregates) == 1 and re.search(
        r"次数|数量|数|量|条数|人数|患者数", cleaned
    ):
        return aggregates[0]

    columns = _all_alias_matches(cleaned, _PROJECTION_ALIASES)
    if columns:
        return columns[0].value
    if re.search(r"患者数|人数|多少患者|患者数量", cleaned):
        return "COUNT(DISTINCT patient_id)"
    if re.search(r"就诊量|就诊次数|记录数|三元组数|实体数|数量", cleaned):
        return "COUNT(*)"
    return None


def _normalise_direction(word: str) -> str:
    lowered = word.casefold()
    for direction, words in _DIRECTION_WORDS:
        if any(lowered == candidate.casefold() for candidate in words):
            return direction
    return "ASC"


def _temporal_group_key_from_text(
    text: str,
    grouping: Sequence[str],
) -> Optional[str]:
    """Resolve only an already-proven month/year grouping key."""

    for span in _all_alias_matches(text, _GROUP_PATTERNS):
        if span.value in {"month", "year"} and span.value in grouping:
            return span.value
    return None


def _has_temporal_inner_row_selection_or_rank(question: str) -> bool:
    """Recognize row selection/ranking scoped inside a year or month."""

    period = r"(?:每(?:个)?(?:年份?|月份?|年度|月度)|各(?:年份?|月份?|年度|月度))"
    for match in re.finditer(rf"{period}(?=\s*(?:中|内|里|下|的))", question):
        tail = question[match.end():]
        if _row_selection_follows_dimension(tail):
            return True
        clause = re.split(r"[。！？?!；;]", tail, maxsplit=1)[0]
        if re.match(
            r"\s*(?:中|内|里|下|的)\s*.{0,72}?"
            r"(?:"
            r"(?:取|选取|选择|保留|返回|筛选|选出).{0,12}?"
            r"(?:前\s*(?:\d+|[一二两三四五六七八九十]+)|"
            r"第\s*(?:1|一))\s*(?:个|位|名|条|次|项|行)?|"
            r"连续排名|排名|排序编号|名次|行号|ROW_NUMBER"
            r")",
            clause,
            re.IGNORECASE,
        ):
            return True
    return False


def _extract_ordering(
    question: str,
    aggregates: Sequence[str],
    grouping: Sequence[str],
    limit: Optional[int],
) -> tuple[list[str], bool]:
    found: list[tuple[int, str]] = []
    explicit = False
    partitioned_row_selection = _has_partitioned_row_selection(question)
    temporal_count_inner_selection = (
        tuple(aggregates) == ("COUNT(*)",)
        and _has_temporal_inner_row_selection_or_rank(question)
    )
    if partitioned_row_selection or temporal_count_inner_selection:
        # Ordering used to choose row 1 inside each partition is not an outer
        # result-order promise.  The contract intentionally leaves that inner
        # mechanism to the AST lineage validator.
        return [], False
    direction_words = "|".join(
        re.escape(word)
        for _, words in _DIRECTION_WORDS
        for word in sorted(words, key=len, reverse=True)
    )
    directional = re.compile(
        rf"(?P<key>[^。；;，,、]+?)\s*(?P<direction>{direction_words})"
        rf"(?=\s*(?:排序|排列|[。；;，,、]|$))",
        re.IGNORECASE,
    )
    for match in directional.finditer(question):
        key = _semantic_key_from_text(match.group("key"), aggregates)
        if key is None:
            key = _temporal_group_key_from_text(match.group("key"), grouping)
        if key is None:
            continue
        explicit = True
        found.append(
            (
                match.start(),
                f"{key} {_normalise_direction(match.group('direction'))}",
            )
        )

    natural_age_desc = re.search(
        r"(?:先列|优先列出|先显示)?年龄(?:较大|更大|大)者(?:优先|在前)?",
        question,
    )
    if natural_age_desc:
        explicit = True
        found.append((natural_age_desc.start(), "age DESC"))
    same_age_ties = re.search(
        r"(?:同龄|年龄相同)时?.*?姓名.*?(?:编号|患者编号|ID).*(?:排序|升序|排列)",
        question,
        re.I,
    )
    if same_age_ties:
        explicit = True
        found.append((same_age_ties.start(), "name ASC"))
        found.append((same_age_ties.start() + 1, "patient_id ASC"))

    # "按X排序" without a direction is an explicit ascending order.
    sorting = re.compile(r"(?:按|按照|依据|依照)\s*(?P<key>[^。；;，,]+?)\s*(?:排序|排列)")
    for match in sorting.finditer(question):
        # The direction-aware scanner above already owns clauses such as
        # "按费用降序排序".  Without this guard the undirected rule sees
        # "费用降序" as its key and appends a contradictory ``cost ASC``.
        if re.search(
            rf"(?:{direction_words})",
            match.group("key"),
            re.IGNORECASE,
        ):
            continue
        key = _semantic_key_from_text(match.group("key"), aggregates)
        if key is None:
            key = _temporal_group_key_from_text(match.group("key"), grouping)
        if key is not None:
            explicit = True
            found.append((match.start(), f"{key} ASC"))

    # Ranking language carries an order direction even when it does not say
    # "按...降序", for example "平均费用最高的前5种诊断".
    ranking = re.compile(
        r"(?P<key>[^。；;，,]{1,32}?)\s*"
        r"(?P<direction>最高|最大|最多|最低|最小|最少)\s*"
        r"(?:的\s*)?(?=(?:前\s*(?:\d+|[一二两三四五六七八九十]+)|"
        r"(?:\d+|[一二两三四五六七八九十]+)\s*"
        r"(?:个|位|名|条|次|种|项|行)|"
        r"top[-\s]*\d+|排名|第\s*\d+|"
        r"(?:那|该)?(?:次|条)?(?:就诊|记录)))",
        re.IGNORECASE,
    )
    for match in ranking.finditer(question):
        if partitioned_row_selection:
            # The extreme belongs inside each partition's row-selection
            # mechanism (window/correlated subquery), not the outer result.
            continue
        key = _semantic_key_from_text(match.group("key"), aggregates)
        if key is None and aggregates:
            key = aggregates[0]
        if key is None:
            continue
        explicit = True
        direction = "ASC" if match.group("direction") in {"最低", "最小", "最少"} else "DESC"
        found.append((match.start(), f"{key} {direction}"))

    # A global temporal superlative selects one visit row even though the
    # question usually names only the row's projected patient attributes.
    # Keep per-patient temporal requests out of this branch: they are
    # correlated row selections, not one global Top-1 ordering.
    per_patient_temporal = re.search(
        r"(?:每位|每个|各(?:位|个)?)\s*(?:患者|病人).{0,12}"
        r"(?:最近|最新|最后|末次|最晚|首次|第一次|最早)",
        question,
    )
    if not per_patient_temporal and not partitioned_row_selection:
        recent = re.search(
            r"(?:最近(?:一)?次|最新(?:一)?次|最后(?:一)?次|末次|最晚)"
            r".{0,6}(?:就诊|记录)|"
            r"最后一条.{0,6}(?:就诊|记录)",
            question,
        )
        earliest = re.search(
            r"(?:首次|第一次|最早(?:一)?次).{0,6}(?:就诊|记录)",
            question,
        )
        temporal_match = recent or earliest
        if temporal_match is not None:
            explicit = True
            direction = "DESC" if recent is not None else "ASC"
            found.append((temporal_match.start(), f"visit_date {direction}"))

    if tuple(aggregates) == ("COUNT(*)",):
        temporal_count_phrases = (
            ("year", re.search(r"每年内\s*按\s*次数\s*降序", question)),
            (
                "month",
                re.search(r"按\s*月份\s*及\s*次数\s*降序\s*排列", question),
            ),
        )
        for dimension, match in temporal_count_phrases:
            if (
                match is not None
                and dimension in grouping
                and not any(
                    value.rsplit(" ", 1)[0] == dimension
                    for _, value in found
                )
            ):
                # The dimension establishes the outer chronological sequence;
                # only the explicitly named count key is descending inside it.
                found.append((match.start() - 1, f"{dimension} ASC"))
                explicit = True

    if limit is not None and not found and aggregates:
        # "前K个科室/疾病" is a Top-K request; use the stated measure as the
        # ranking key.  This is only inferred when an aggregate is present.
        direction = (
            "ASC"
            if re.search(r"最少|最低|最小", question)
            else "DESC"
        )
        found.append(
            (
                question.find("前") if "前" in question else 0,
                f"{aggregates[0]} {direction}",
            )
        )
        explicit = True

    result: list[str] = []
    for _, value in sorted(found, key=lambda item: item[0]):
        if value not in result:
            result.append(value)
    return result, explicit


def _extract_limit(question: str) -> Optional[int]:
    if _has_partitioned_row_selection(question):
        # A per-partition row number is not a global SQL LIMIT.
        return None
    number_token = r"(?:\d+|[一二两三四五六七八九十百千万]+)"
    # Formatting precision is not result cardinality.  Mask only tightly bound
    # precision phrases so a real request in the same question (for example
    # "返回3条，费用保留2位小数") remains visible to the LIMIT patterns.
    precision_patterns = (
        rf"(?:保留|四舍五入(?:到|至)?|精确到|舍入到)\s*"
        rf"(?:小数点后\s*)?{number_token}\s*位"
        r"(?=\s*(?:小数|精度|[()（）,，;；。.]|$))",
        rf"(?:小数|精度)[^。；;，,]{{0,8}}"
        rf"(?:保留|四舍五入(?:到|至)?|精确到|舍入到)?\s*"
        rf"{number_token}\s*位",
    )
    limit_source = question
    for precision_pattern in precision_patterns:
        limit_source = re.sub(
            precision_pattern,
            lambda match: " " * (match.end() - match.start()),
            limit_source,
            flags=re.IGNORECASE,
        )

    patterns = (
        r"\btop[-\s]*(\d+)\b",
        r"前\s*(\d+|[一二两三四五六七八九十百千万]+)\s*(?:个|位|名|条|次|种|项|行)?",
        r"排名\s*前\s*(\d+|[一二两三四五六七八九十百千万]+)",
        r"(?:最高|最低|最多|最少|最贵|最便宜)的?\s*"
        r"(\d+|[一二两三四五六七八九十百千万]+)\s*"
        r"(?:个|位|名|条|次|种|项|行)",
        r"(?:取|保留|返回)\s*(\d+|[一二两三四五六七八九十百千万]+)\s*"
        r"(?:个|位|名|条|次|种|项|行)",
    )
    for pattern in patterns:
        match = re.search(pattern, limit_source, re.IGNORECASE)
        if match:
            value = _cn_int(match.group(1))
            if value is not None:
                return value
    if re.search(
        r"(?:最多|最少|最高|最低|最大|最小)(?:的)?\s*"
        r"(?:科室|部门|疾病|诊断|地区|城市|性别|患者|实体|关系)",
        question,
    ):
        return 1
    if re.search(
        r"(?:最高|最低|最大|最小)(?:的)?\s*"
        r"(?:那|该)?(?:次|条)?(?:就诊|记录)",
        question,
    ):
        return 1
    if re.search(
        r"哪(?:一)?(?:个|种|类)?\s*"
        r"(?:科室|部门|疾病|诊断|药物|处方药|药|检查)"
        r".{0,48}(?:最多|最少|最高|最低|最大|最小)",
        question,
    ):
        return 1
    if re.search(
        r"(?:最多|最少|最高|最低|最大|最小).{0,24}"
        r"哪(?:一)?(?:个|种|类)?\s*"
        r"(?:科室|部门|疾病|诊断|药物|处方药|药|检查)",
        question,
    ):
        return 1
    if not re.search(
        r"(?:每位|每个|各(?:位|个)?)\s*(?:患者|病人).{0,12}"
        r"(?:最近|最新|最后|末次|最晚|首次|第一次|最早)",
        question,
    ) and re.search(
        r"(?:最近(?:一)?次|最新(?:一)?次|最后(?:一)?次|末次|最晚|"
        r"首次|第一次|最早(?:一)?次).{0,6}(?:就诊|记录)|"
        r"最后一条.{0,6}(?:就诊|记录)",
        question,
    ):
        return 1
    return None


def _extract_set_semantics(question: str) -> Optional[str]:
    """Extract high-signal row or graph boolean-set language."""

    patient_set = bool(
        re.search(r"(?:患者|病人|病患|病号|参保人)", question)
        and re.search(
            r"就诊|看(?:过)?病|就医|去过|来院|使用|服用|开过",
            question,
        )
    )
    graph_set = bool(
        re.search(r"疾病|实体|头实体|尾实体|症状", question)
        and re.search(
            r"并发症|用药禁忌|禁忌|临床表现|症状|治疗|关系|三元组",
            question,
        )
    )
    dimension_set = bool(
        re.search(r"疾病|诊断|科室|部门|药物|检查", question)
        and re.search(r"出现|发生|存在|来自|属于", question)
    )
    if not (patient_set or graph_set or dimension_set):
        return None
    if re.search(
        r"(?:但|却).{0,24}(?:从未|未曾|没有|没在|没去过|未去过|"
        r"未使用|没用过|没出现|未出现|没有出现)",
        question,
    ):
        return "difference"
    if re.search(
        r"既.{1,48}又|同时.{1,48}(?:和|与)|两者都|二者都|都去过|都就诊",
        question,
    ):
        return "all_of"
    if re.search(
        r"或(?:者|是)?|任一|任意(?:一个)?|至少一个|二者之一|要么|\bOR\b",
        question,
        re.IGNORECASE,
    ):
        return "any_of"
    return None


def _is_cross_table_object_set_enumeration(
    question: str,
    projections: Sequence[str],
    aggregates: Sequence[str],
    grouping: Sequence[str],
    *,
    limit: Optional[int],
    set_semantics: Optional[str],
) -> bool:
    """Recognize one narrow family of joined object-tuple list requests.

    A join can multiply otherwise identical entity/relation tuples.  The
    controlled questions owned here ask for the object set, so their outer
    SELECT must deduplicate.  Mere multi-table language is not enough: require
    an explicit connection clause, at least three output columns including two
    graph-object roles, and no record-grain, analytical, paging, or compound
    semantics.  These exclusions keep DISTINCT from becoming a generic repair
    for joins whose multiplicity is part of the requested result.
    """

    if aggregates or grouping or limit is not None or set_semantics is not None:
        return False

    join_marker = re.search(r"(?:连接|联结)", question)
    if join_marker is None:
        return False
    output_marker = re.search(
        r"(?:[，,]\s*)?(?:(?:并|再|然后)\s*)?"
        r"(?:列出|给出|返回|显示|输出|展示)",
        question[join_marker.end():],
        re.IGNORECASE,
    )
    if output_marker is None:
        return False
    source_clause = question[
        join_marker.end():join_marker.end() + output_marker.start()
    ].strip(" ，,")
    source_atom = (
        r"(?:患者(?:表)?|病人(?:表)?|就诊(?:记录|表)?|访问(?:记录|表)?|"
        r"(?:知识)?图谱三元组|三元组(?:表)?|图谱实体|实体表|实体|(?:知识)?图谱)"
    )
    if re.search(
        rf"{source_atom}.{{0,32}}(?:、|和|与|及).{{0,32}}{source_atom}",
        source_clause,
    ) is None:
        return False

    canonical_projections = tuple(
        _canonical_column_value(value) for value in projections
    )
    graph_objects = {"head", "tail", "relation", "relation_zh"}
    if (
        len(canonical_projections) < 3
        or len(graph_objects.intersection(canonical_projections)) < 2
        or "visit_id" in canonical_projections
    ):
        return False

    if re.search(r"每条|每次|逐条|明细|保留\s*重复|按\s*记录", question):
        return False
    if re.search(
        r"就诊编号|访问编号|记录编号|\bvisit_id\b",
        question,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"连续排名|排序编号|累计分布|窗口|"
        r"\b(?:ROW_NUMBER|RANK|DENSE_RANK|OVER)\b",
        question,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"统计|汇总|聚合|计数|总数|数量|平均|均值|合计|总和|"
        r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(",
        question,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"偏移|跳过\s*(?:\d+|[一二两三四五六七八九十百千万]+)\s*条|"
        r"从第\s*(?:\d+|[一二两三四五六七八九十百千万]+)\s*条"
        r"(?:之后|开始)|\bOFFSET\b",
        question,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"并集|交集|差集|集合运算|合并(?:两|多个|这些|查询|结果)|"
        r"\b(?:UNION|INTERSECT|EXCEPT)\b",
        question,
        re.IGNORECASE,
    ):
        return False
    return True


def _extract_row_set_mode(
    question: str,
    projections: Sequence[str],
    aggregates: Sequence[str],
    grouping: Sequence[str],
    *,
    limit: Optional[int] = None,
    set_semantics: Optional[str] = None,
) -> RowSetMode:
    """Bind explicit and existential deduplication requests to the contract."""

    if aggregates or grouping:
        return RowSetMode.PRESERVE
    if re.search(
        r"至少\s*(?:\d+(?:\.\d+)?|[一二两三四五六七八九十百千万零〇]+|N)\s*次",
        question,
        re.IGNORECASE,
    ):
        # A minimum occurrence predicate changes which objects qualify; it is
        # not a request to deduplicate the outer rows.  In particular, do not
        # let the nearby "就诊记录" existential heuristic turn 至少N次 into
        # SELECT DISTINCT.
        return RowSetMode.PRESERVE
    if re.search(
        r"多少|几(?:种|名|位|个)?|人数|数量|总数|计数",
        question,
    ):
        # DISTINCT inside COUNT is an aggregate argument, not SELECT-level row
        # deduplication.  Some public paraphrases are owned by a specialized
        # compiler even when the generic aggregate extractor stays conservative.
        return RowSetMode.PRESERVE
    if (
        re.search(r"\b(?:how much|what)\b", question, re.IGNORECASE)
        and re.search(r"\b(?:costs?|price|charge)\b", question, re.IGNORECASE)
        and not re.search(
            r"\b(?:average|mean|maximum|max|minimum|min|total|sum|count)\b",
            question,
            re.IGNORECASE,
        )
    ):
        # Public clinical questions such as "How much does X cost?" request
        # the set of observed price values, not one row per encounter. Their
        # bounded renderer uses SELECT DISTINCT so duplicate visits cannot
        # duplicate an otherwise identical answer value.
        return RowSetMode.DISTINCT
    if re.search(r"不同|去重|不重复|唯一|\bDISTINCT\b", question, re.IGNORECASE):
        return RowSetMode.DISTINCT
    if _is_cross_table_object_set_enumeration(
        question,
        projections,
        aggregates,
        grouping,
        limit=limit,
        set_semantics=set_semantics,
    ):
        return RowSetMode.DISTINCT
    patient_output = any(
        _canonical_column_value(value) in {"patient_id", "name"}
        for value in projections
    )
    visit_existence = bool(
        re.search(r"患者|病人|病患|病号", question)
        and re.search(
            r"曾经|曾有|曾到|来院就诊|就诊过|就诊记录|"
            r"去过|到过|看过病|看病",
            question,
        )
        and not re.search(
            r"从来没有|从未|未曾|没有|没在|最后|最近|最新|"
            r"最早|首次|第一次|每个患者|各个患者",
            question,
        )
    )
    return (
        RowSetMode.DISTINCT
        if patient_output and visit_existence
        else RowSetMode.PRESERVE
    )


def _relative_relation_role_output(
    question: str,
    output_text: str,
) -> Optional[tuple[str, str]]:
    """Resolve an exact graph-role name/type list after a CODE filter clause."""

    # Nested/correlated entity lookups name the graph role only as a filter;
    # their outer result is the matched kg_entities row.  Keep this path exact:
    # one uppercase relation code, one explicit head/tail role, and the closed
    # two-field suffix ``实体名称和类型``.  This must not leak relation/head/tail
    # filter nouns into the projected entity-table contract.
    entity_lookup = re.search(
        r"(?:列出|返回|查询|查找)\s*(?:在)?关系\s*"
        r"[A-Z][A-Z0-9_\-]*\s*中担任(?:头|尾)实体的\s*"
        r"实体(?:名称|名字|实体名)\s*(?:、|和|与|及|以及)\s*类型",
        question,
    )
    if entity_lookup is not None:
        return "name", "type"

    relation_code = re.search(
        r"(?:关系\s*(?:CODE|编码|代码)|relation\s*(?:code|id))\s*"
        r"(?:为|等于|是|=|:|：)?\s*[A-Z][A-Z0-9_\-]*",
        question,
        re.IGNORECASE,
    )
    if relation_code is None or not output_text:
        return None

    cleaned = re.sub(
        r"\s*(?:字段)?(?:清单|列表|明细)\s*$",
        "",
        output_text.strip(" ：:,，"),
    )
    requested = re.fullmatch(
        r"(?P<role>头|尾)实体(?:的)?(?:名称|名字|实体名)?\s*"
        r"(?:、|和|与|及|以及)\s*(?:其\s*)?"
        r"(?:(?P<type_role>头|尾)实体(?:的)?)?类型",
        cleaned,
    )
    if requested is None:
        return None
    role = requested.group("role")
    if requested.group("type_role") not in {None, role}:
        return None

    output_at = question.rfind(output_text)
    prefix = question[:output_at] if output_at >= 0 else question
    role_filter = re.search(
        r"(?:头|尾)实体(?:类型)?\s*"
        r"(?:为|等于|是|=|:|：)\s*"
        r"[A-Za-z][A-Za-z0-9_\-]*.{0,24}的\s*$",
        prefix,
        re.IGNORECASE,
    )
    if role_filter is None:
        return None
    canonical = "head" if role == "头" else "tail"
    return canonical, f"{canonical}_type"


def extract_semantic_contract(question: str) -> SemanticContract:
    """Extract a conservative semantic contract from Chinese/English question text."""

    q = _normalise_question(question)
    output_text, has_output = _extract_output_clause(q)
    grouping = _extract_grouping(q)
    aggregate_source = output_text if output_text else q
    window_analysis_request = bool(
        re.search(r"连续排名|排序编号|累计分布", q, re.IGNORECASE)
    )
    aggregate_spans = _extract_aggregates(
        aggregate_source,
        grouped=bool(
            grouping
            and not window_analysis_request
            and re.search(r"统计|计数|数量|分布|趋势|排名|多少", q)
        ),
    )
    # Grouped patient demographics operate on the one-row-per-patient table.
    # In that source, a plain 患者数 is COUNT(*), while explicit "不同/去重患者"
    # and visit-derived cohorts remain COUNT(DISTINCT patient_id).
    patient_demographic_group = bool(
        grouping
        and set(grouping).issubset({"region", "gender", "age"})
        and not re.search(
            r"就诊|就医|住院|门诊|访问|病历|记录|费用|花费|疾病|诊断|"
            r"科室|部门|药物|处方|检查",
            q,
        )
        and not re.search(r"不同|去重|不重复|唯一", aggregate_source)
    )
    scalar_patient_population = bool(
        not grouping
        and re.search(
            r"(?:患者|病人)(?:的)?(?:总数|总人数|数量|人数|有多少(?:人)?)"
            r"(?:分别)?(?:是|为)?多少?",
            q,
        )
        and not re.search(
            r"就诊|就医|住院|门诊|访问|病历|记录|费用|疾病|诊断|"
            r"科室|部门|药物|处方|检查",
            q,
        )
        and not re.search(r"不同|去重|不重复|唯一", aggregate_source)
    )
    if patient_demographic_group or scalar_patient_population:
        aggregate_spans = [
            _Span(span.start, span.end, "COUNT(*)")
            if span.value == "COUNT(DISTINCT patient_id)"
            else span
            for span in aggregate_spans
        ]
    if not aggregate_spans and output_text:
        # The compact form "统计各科室就诊次数，返回科室和次数" puts the
        # measure before the final output verb.  Carry only its semantic value
        # into the output clause and use a zero-width span for masking.
        question_aggregates = _extract_aggregates(
            q,
            grouped=bool(grouping and not window_analysis_request),
        )
        fallback_value = question_aggregates[0].value if question_aggregates else "COUNT(*)"
        aggregate_spans = []
        count_phrase = re.compile(
            r"(?:不同|去重|不重复)?\s*(?:患者|病人)?\s*"
            r"(?:总数|数量|人数|次数|条数|数|量)"
        )
        raw_output_count_matches = list(count_phrase.finditer(output_text))
        projection_aliases = _all_alias_matches(output_text, _PROJECTION_ALIASES)
        output_count_matches = [
            match
            for match in raw_output_count_matches
            if not any(
                match.start() >= alias.start
                and match.end() <= alias.end
                for alias in projection_aliases
            )
        ]
        if output_count_matches:
            aggregate_spans.extend(
                _Span(match.start(), match.end(), fallback_value)
                for match in output_count_matches
            )
    aggregates = []
    for span in aggregate_spans:
        if span.value not in aggregates:
            aggregates.append(span.value)

    projections: list[str] = []
    output_spans: list[_Span] = []
    projection_spans: list[_Span] = []
    if output_text:
        output_spans = list(aggregate_spans)
        masked = list(output_text)
        for span in aggregate_spans:
            for index in range(max(0, span.start), min(len(masked), span.end)):
                masked[index] = " "
        masked_output = "".join(masked)
        projection_spans = _all_alias_matches(masked_output, _PROJECTION_ALIASES)
        # Bare "编号" is table-relative.  In a visit-row question without an
        # explicit patient-number phrase it denotes visit_id; explicit
        # 患者编号/病人编号 continues to map to patient_id.
        if re.search(r"就诊|访问|记录", q):
            projection_spans = [
                _Span(
                    span.start,
                    span.end,
                    "visit_id"
                    if (
                        span.value == "patient_id"
                        and masked_output[span.start:span.end] == "编号"
                    )
                    else span.value,
                )
                for span in projection_spans
            ]
        last_projection_occurrence = {
            span.value: index
            for index, span in enumerate(projection_spans)
        }
        projection_spans = [
            span
            for index, span in enumerate(projection_spans)
            if last_projection_occurrence[span.value] == index
        ]
        output_spans.extend(projection_spans)
        for span in projection_spans:
            if span.value not in projections:
                projections.append(span.value)
        if (
            re.search(
                r"(?:每位|每个|各(?:位|个)?)\s*患者.{0,6}"
                r"(?:最近|最新|最后|末次|最晚|首次|第一次|最早)\s*一?次?",
                q,
            )
            and "patient_id" not in projections
        ):
            projections.insert(0, "patient_id")

    explicit_output_relations = {
        span.value
        for span in projection_spans
        if span.value in {"relation", "relation_zh"}
    }
    if (
        "relation" in grouping
        and "relation_zh" in explicit_output_relations
        and "relation" not in explicit_output_relations
    ):
        # A bare grouping mention normally means the canonical code column.
        # When the final answer explicitly names only the Chinese relation,
        # however, the local output mention disambiguates that otherwise-bare
        # dimension.  Do not use question-global relation filters for this:
        # they were already removed from projection_spans with the filter
        # prefix, preserving mixed English-filter/Chinese-output questions.
        grouping = [
            "relation_zh" if value == "relation" else value
            for value in grouping
        ]

    # A grouped result must expose its group key even when the question uses
    # the compact form "按科室统计就诊量" without an output verb.  When an
    # aggregate and explicit raw output columns coexist, those columns are the
    # natural GROUP BY keys even if the question omits the word "按".
    if (
        not grouping
        and "name" in projections
        and re.search(r"(?:累计费用|总费用|总花费|总支出|花费总额)", q)
        and re.search(r"(?:前\s*[一二两三四五六七八九十\d]+|最高|最多)", q)
        and re.search(r"(?:患者|病人)", q)
    ):
        # Ranking patients by an aggregate has patient entity grain even when
        # the answer asks only for display names and keeps SUM(cost) solely in
        # ORDER BY.
        grouping = ["patient_id", *projections]
    elif not grouping and aggregates and projections:
        grouping = list(projections)
    elif grouping and aggregates:
        # Every raw selected attribute participates in the result grain.
        # A stable entity key may remain hidden (for example GROUP BY
        # patient_id, name while only name is returned), but grouping only by
        # a non-unique display name is not an equivalent patient grain.
        grouping = [
            *grouping,
            *(value for value in projections if value not in grouping),
        ]
    # An implicit per-group key identifies every result row and therefore
    # precedes the requested row attributes.  Iterate in reverse so multiple
    # grouping keys retain their stated order when inserted at the front.
    for value in reversed(grouping):
        if value not in projections:
            if (
                value == "patient_id"
                and output_text
                and not re.search(
                    r"(?:每位|每个|各(?:位|个)?)\s*患者",
                    q,
                )
                and not any(
                    span.value == "patient_id"
                    for span in output_spans
                )
            ):
                continue
            projections.insert(0, value)

    # Preserve mixed raw/aggregate answer order only when every required output
    # can be located in the final output clause.  If implicit grouping inserted
    # an otherwise unstated key, retain the long-standing fallback order rather
    # than inventing where that key belongs in the user's answer list.
    output_slots: list[str] = []
    if output_text:
        extracted_slots: list[str] = []
        ordered_output_spans = sorted(
            output_spans,
            key=lambda item: (item.start, item.end),
        )
        # A Top-K answer can name the same aggregate twice: first as the
        # ranking key ("就诊次数最多的前2个科室"), then as the requested
        # value ("及其就诊次数").  The latter occurrence determines the
        # select-list position.  Ordinary non-repeated output clauses retain
        # their exact order.
        last_aggregate_occurrence = {
            span.value: index
            for index, span in enumerate(ordered_output_spans)
            if span.value in aggregates
        }
        for index, span in enumerate(ordered_output_spans):
            if (
                span.value in aggregates
                and last_aggregate_occurrence.get(span.value) != index
            ):
                continue
            if span.value not in extracted_slots:
                extracted_slots.append(span.value)
        required_slots = [*projections, *aggregates]
        if (
            len(extracted_slots) == len(required_slots)
            and all(
                extracted_slots.count(value) == required_slots.count(value)
                for value in required_slots
            )
        ):
            output_slots = extracted_slots

    relative_role_output = _relative_relation_role_output(q, output_text)
    if relative_role_output is not None:
        # Filter-side role/type nouns describe predicates.  The final relative
        # clause asks only for the selected graph entity and its own type.
        projections = list(relative_role_output)
        aggregates = []
        grouping = []
        output_slots = list(relative_role_output)

    # Natural legacy KG questions use clinical nouns for graph roles.  Keep
    # those roles separate from same-named visits columns, and keep a ranking
    # COUNT(*) hidden unless the question explicitly asks for the quantity.
    graph_head_top = bool(
        re.search(r"并发症最多", q)
        and re.search(r"疾病|哪个|哪种", q)
    )
    graph_tail_top = bool(
        re.search(
            r"被最多疾病.{0,8}(?:列为)?(?:临床表现|症状)|"
            r"(?:临床表现|症状).{0,8}最多疾病",
            q,
        )
    )
    graph_head_count = bool(
        re.search(r"每种疾病|各(?:种|类)?疾病", q)
        and re.search(r"(?:关联的?)?(?:症状|临床表现).{0,6}(?:数量|多少)", q)
    )
    graph_triple_head_mentions = bool(
        re.search(r"三元组", q)
        and re.search(r"头实体提及次数", q)
        and re.search(r"头实体类型|中文关系|关系中文名", q)
    )
    if graph_triple_head_mentions:
        projections = ["head", "relation_zh", "tail", "mention_count"]
        aggregates = []
        grouping = []
        output_slots = ["head", "relation_zh", "tail", "mention_count"]
    elif graph_head_top:
        projections = ["head"]
        aggregates = ["COUNT(*)"]
        grouping = ["head"]
        output_slots = ["head", "COUNT(*)"]
    elif graph_tail_top:
        projections = ["tail"]
        aggregates = ["COUNT(*)"]
        grouping = ["tail"]
        output_slots = ["tail", "COUNT(*)"]
    elif graph_head_count:
        projections = [
            "head" if value == "disease" else value
            for value in projections
        ]
        grouping = [
            "head" if value == "disease" else value
            for value in grouping
        ]
        if output_slots:
            output_slots = [
                "head" if value == "disease" else value
                for value in output_slots
            ]

    fee_rank_contract = bool(
        re.search(r"科室.+?中.{0,24}就诊按费用降序编号", q)
        and re.search(r"前\s*[一二两三四五六七八九十\d]+\s*名", q)
        and re.search(r"名次|排名", q)
    )
    patient_total_rank_contract = bool(
        re.search(r"科室.+?中按患者累计费用排名", q)
        and re.search(r"累计费用.{0,12}(?:不低于|至少|大于|高于)", q)
        and re.search(r"前\s*[一二两三四五六七八九十\d]+\s*名", q)
        and re.search(r"名次|排名", q)
    )
    observation_rank_contract = bool(
        re.search(r"三元组", q)
        and re.search(r"按头实体累计观察次数排名", q)
        and re.search(r"前\s*[一二两三四五六七八九十\d]+\s*名", q)
    )
    age_group_average_rank_contract = bool(
        re.search(r"(?:平均就诊费用|就诊平均费用|平均费用)", q)
        and re.search(r"(?:最高|最低)的年龄段", q)
    )
    prescription_top_contract = bool(
        re.search(r"处方.{0,8}(?:最多|最高)", q)
        and re.search(
            r"前\s*(?:\d+|[一二两三四五六七八九十百千万]+)\s*种药物",
            q,
        )
        and not re.search(
            r"平均|均值|费用|花费|金额|住院|患者数|人数",
            q,
        )
    )
    exam_usage_contract = bool(
        re.search(r"各(?:个)?检查项目", q)
        and re.search(r"使用次数|使用数量|使用量", q)
    )
    entity_mention_rank_contract = bool(
        re.search(r"提及次数最多", q)
        and re.search(r"疾病|症状|药物|检查", q)
        and re.search(r"哪些|有哪些|前", q)
    )
    specialized_rank_contract = (
        fee_rank_contract
        or patient_total_rank_contract
        or observation_rank_contract
    )
    if fee_rank_contract:
        projections = ["visit_id", "patient_id", "cost", "fee_rank"]
        aggregates = []
        grouping = []
        output_slots = list(projections)
    elif patient_total_rank_contract:
        projections = ["patient_id", "total_rank"]
        aggregates = ["SUM(cost)"]
        grouping = ["patient_id"]
        output_slots = ["patient_id", "SUM(cost)", "total_rank"]
    elif observation_rank_contract:
        projections = ["head", "observation_rank"]
        aggregates = ["SUM(count)"]
        grouping = ["head"]
        output_slots = ["head", "SUM(count)", "observation_rank"]
    elif age_group_average_rank_contract:
        projections = ["age"]
        aggregates = ["AVG(cost)"]
        grouping = ["age"]
        output_slots = ["age", "AVG(cost)"]
    if prescription_top_contract:
        projections = ["drug_prescribed"]
        aggregates = ["COUNT(*)"]
        grouping = ["drug_prescribed"]
        output_slots = ["drug_prescribed", "COUNT(*)"]
    elif exam_usage_contract:
        projections = ["exam"]
        aggregates = ["COUNT(*)"]
        grouping = ["exam"]
        output_slots = ["exam", "COUNT(*)"]
    elif entity_mention_rank_contract:
        projections = ["name", "mention_count"]
        aggregates = []
        grouping = []
        output_slots = ["name", "mention_count"]

    peer_group: Optional[str] = None
    peer_metric: Optional[str] = None
    peer_direction: Optional[str] = None
    if (
        re.search(r"高于|超过|大于|低于|小于", q)
        and re.search(r"全体|整体|总体|所有", q)
        and re.search(r"平均|均值", q)
    ):
        peer_group = (
            "department"
            if re.search(r"科室|部门", q)
            else "disease"
            if re.search(r"疾病|诊断", q)
            else None
        )
        peer_metric = (
            "cost"
            if re.search(r"费用|花费|开销|金额", q)
            else "los_days"
            if re.search(r"住院天数|住院时长", q)
            else None
        )
        peer_direction = (
            "DESC"
            if re.search(r"高于|超过|大于", q)
            else "ASC"
        )
        if peer_group and peer_metric:
            projections = [peer_group]
            aggregates = [f"AVG({peer_metric})"]
            grouping = [peer_group]
            output_slots = []

    count_peer_group: Optional[str] = None
    if re.search(
        r"(?:就诊次数|就诊量).{0,12}(?:超过|高于|大于).{0,12}"
        r"(?:平均每(?:种|个)?|每(?:种|个)?.{0,4}平均).{0,6}(?:疾病|科室)|"
        r"(?:超过|高于|大于).{0,6}平均每(?:种|个)?(?:疾病|科室)"
        r".{0,8}(?:就诊次数|就诊量)",
        q,
    ):
        count_peer_group = "department" if "科室" in q else "disease"
        projections = [count_peer_group]
        aggregates = ["COUNT(*)"]
        grouping = [count_peer_group]
        output_slots = []

    limit = _extract_limit(q)
    if age_group_average_rank_contract:
        limit = 1
    elif entity_mention_rank_contract:
        # The long-standing public rule exposes a bounded entity leaderboard
        # when the plural question omits an explicit K.
        limit = 10
    ordering, explicit_order = _extract_ordering(q, aggregates, grouping, limit)
    if specialized_rank_contract:
        rank_column = (
            "fee_rank"
            if fee_rank_contract
            else "total_rank"
            if patient_total_rank_contract
            else "observation_rank"
        )
        ordering = [f"{rank_column} ASC"]
        explicit_order = True
    elif age_group_average_rank_contract:
        direction = "DESC" if "最高" in q else "ASC"
        ordering = [f"AVG(cost) {direction}"]
        explicit_order = True
    elif prescription_top_contract:
        ordering = ["COUNT(*) DESC"]
        explicit_order = True
    elif entity_mention_rank_contract:
        ordering = ["mention_count DESC"]
        explicit_order = True
    elif peer_group and peer_metric and peer_direction:
        ordering = [f"AVG({peer_metric}) {peer_direction}"]
        explicit_order = True
    elif count_peer_group:
        direction = (
            "ASC"
            if re.search(r"升序|从低到高", q)
            else "DESC"
        )
        ordering = [f"COUNT(*) {direction}"]
    set_semantics = _extract_set_semantics(q)
    row_set_mode = _extract_row_set_mode(
        q,
        projections,
        aggregates,
        grouping,
        limit=limit,
        set_semantics=set_semantics,
    )

    recognized = bool(
        projections
        or aggregates
        or grouping
        or ordering
        or limit is not None
        or set_semantics is not None
    )
    incomplete_output = (
        bool(output_text)
        and not specialized_rank_contract
        and relative_role_output is None
        and _meaningful_residual(output_text, output_spans)
    )
    partial = incomplete_output
    if has_output and not (projections or aggregates):
        partial = True
    if explicit_order and not ordering:
        partial = True

    if not recognized:
        status = ContractStatus.NONE
        confidence = 0.0
    elif partial:
        status = ContractStatus.PARTIAL
        confidence = 0.55
    else:
        status = ContractStatus.COMPLETE
        confidence = 1.0

    return SemanticContract(
        projections=tuple(projections),
        aggregates=tuple(aggregates),
        output_slots=tuple(output_slots),
        grouping=tuple(grouping),
        ordering=tuple(ordering),
        limit=limit,
        confidence=confidence,
        status=status,
        set_semantics=set_semantics,
        row_set_mode=row_set_mode,
    )


def _column_text(value: str) -> str:
    """Canonicalize semantic-contract text, not a parsed SQLite identifier."""

    value = value.strip().strip('`"[]')
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.casefold()


def _sqlite_identifier_key(value: str) -> str:
    """Canonicalize parsed SQLite identifier content with ASCII-only folding.

    SQLGlot's identifier accessors have already removed a delimiter.  Preserve
    every remaining character: trimming or stripping quote-like characters
    would merge distinct quoted SQLite identifiers such as ``"patients "``
    and ``"patients"``.
    """

    return "".join(
        character.lower() if "A" <= character <= "Z" else character
        for character in value
    )


def _canonical_aggregate_text(value: str) -> Optional[str]:
    text = str(value).strip()
    match = re.fullmatch(
        r"(?i)(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*"
        r"(?:(DISTINCT)\s+)?(\*|[A-Za-z_][A-Za-z0-9_.]*)\s*\)",
        text,
    )
    if not match:
        return None
    function = match.group(1).upper()
    distinct = bool(match.group(2))
    argument = "*" if match.group(3) == "*" else _column_text(match.group(3))
    if function == "COUNT" and distinct:
        return f"COUNT(DISTINCT {argument})"
    return f"{function}({argument})"


def _canonical_column_value(value: str) -> str:
    text = str(value).strip()
    aggregate = _canonical_aggregate_text(text)
    if aggregate is not None:
        return aggregate
    text = re.sub(r"(?i)\s+AS\s+.*$", "", text).strip()
    return _column_text(text)


def _sqlglot_import():
    try:
        from sqlglot import exp, parse
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("sqlglot is required for candidate validation") from exc
    return exp, parse


# The validator is deliberately schema-bound.  A candidate may derive values
# through a CTE or subquery, but its leaves must still be one of the project's
# four read-only relations and their declared columns.
_ALLOWED_SOURCE_COLUMNS: dict[str, frozenset[str]] = {
    "patients": frozenset({"patient_id", "name", "gender", "age", "region"}),
    "visits": frozenset(
        {
            "visit_id",
            "patient_id",
            "visit_date",
            "department",
            "disease",
            "drug_prescribed",
            "exam",
            "cost",
            "los_days",
        }
    ),
    "kg_entities": frozenset({"name", "type", "mention_count"}),
    "kg_triples": frozenset(
        {
            "head",
            "head_type",
            "relation",
            "relation_zh",
            "tail",
            "tail_type",
            "confidence",
            "count",
        }
    ),
}


@dataclass(frozen=True)
class _Lineage:
    """One resolved source expression without retaining or rendering SQL text."""

    signature: Optional[str]
    aggregate_argument: Optional[str] = None
    kind: str = "opaque"


@dataclass(frozen=True)
class _Relation:
    """Named outputs, grouping semantics, and provenance of a relation.

    ``aggregate_input_safe`` records a narrow transparency proof for a derived
    relation: it has one safe source and does not group, deduplicate, limit, or
    aggregate its input rows.  Outer aggregates may otherwise silently operate
    on a changed row multiset.  ``known_single_row`` is a separate, exact proof
    used only when forwarding a completed scalar aggregate through a CROSS JOIN.
    """

    outputs: dict[str, _Lineage]
    grouping: tuple[str, ...] = ()
    valid: bool = True
    has_base_source: bool = False
    aggregate_input_safe: bool = False
    known_single_row: bool = False


@dataclass(frozen=True)
class _Source:
    relation: _Relation
    is_base: bool = False
    qualified_name_visible: bool = True


@dataclass
class _Scope:
    sources: dict[str, _Source]
    parent: Optional["_Scope"] = None
    aliases: dict[str, _Lineage] = field(default_factory=dict)
    aggregate_input_safe: bool = False
    aggregate_safe_source_aliases: frozenset[str] = frozenset()
    forwards_derived_aggregate: bool = False
    singleton_derived_aggregate_join_safe: bool = False


@dataclass
class _LineageAudit:
    reasons: list[str] = field(default_factory=list)
    relations: dict[int, _Relation] = field(default_factory=dict)
    scopes: dict[int, _Scope] = field(default_factory=dict)
    relation_scopes: dict[int, _Scope] = field(default_factory=dict)
    analysed_set_operations: set[int] = field(default_factory=set)

    def reject(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)


_BASE_RELATIONS: dict[str, _Relation] = {
    table: _Relation(
        {
            column: _Lineage(column, aggregate_argument=column, kind="base")
            for column in columns
        },
        has_base_source=True,
        aggregate_input_safe=True,
    )
    for table, columns in _ALLOWED_SOURCE_COLUMNS.items()
}


def _unwrap_expression(node: Any, exp: Any) -> Any:
    wrappers = tuple(
        kind
        for name in ("Alias", "Round", "Cast", "TryCast", "Paren")
        if (kind := getattr(exp, name, None)) is not None
    )
    while wrappers and isinstance(node, wrappers):
        node = node.this
    return node


def _is_expression_type(node: Any, exp: Any, name: str) -> bool:
    kind = getattr(exp, name, None)
    return kind is not None and isinstance(node, kind)


def _column_is_star(node: Any) -> bool:
    return bool(getattr(node, "is_star", False))


def _output_name(node: Any, exp: Any) -> Optional[str]:
    if _is_expression_type(node, exp, "Alias"):
        alias = str(getattr(node, "alias", ""))
        return _sqlite_identifier_key(alias) if alias else None
    if _is_expression_type(node, exp, "Column") and not _column_is_star(node):
        name = str(getattr(node, "name", ""))
        return _sqlite_identifier_key(name) if name else None
    return None


def _source_for_qualified_name(scope: Optional[_Scope], name: str) -> Optional[_Source]:
    current = scope
    while current is not None:
        source = current.sources.get(name)
        if source is not None and source.qualified_name_visible:
            return source
        current = current.parent
    return None


def _unqualified_source(scope: Optional[_Scope], name: str) -> Optional[_Source]:
    current = scope
    while current is not None:
        matches = [
            source
            for source in current.sources.values()
            if name in source.relation.outputs
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
        current = current.parent
    return None


def _unqualified_lineage(scope: Optional[_Scope], name: str) -> Optional[_Lineage]:
    source = _unqualified_source(scope, name)
    return source.relation.outputs.get(name) if source is not None else None


def _column_lineage(node: Any, exp: Any, scope: Optional[_Scope]) -> Optional[_Lineage]:
    if not _is_expression_type(node, exp, "Column") or _column_is_star(node):
        return None
    name = str(getattr(node, "name", ""))
    if not name:
        return None
    identifier = getattr(node, "this", None)
    if bool(getattr(identifier, "args", {}).get("quoted")) and "." in name:
        # SQLite treats "patients.age" as one literal identifier, rather than
        # a qualified patients.age reference.  Never split a quoted dotted name
        # into a real source column.
        return None
    canonical_name = _sqlite_identifier_key(name)
    if scope is None:
        return _Lineage(canonical_name, aggregate_argument=canonical_name, kind="raw")
    table = str(getattr(node, "table", ""))
    if table:
        source = _source_for_qualified_name(scope, _sqlite_identifier_key(table))
        if source is None:
            return None
        return source.relation.outputs.get(canonical_name)
    return _unqualified_lineage(scope, canonical_name)


def _column_source(node: Any, exp: Any, scope: Optional[_Scope]) -> Optional[_Source]:
    """Resolve the concrete relation supplying a direct, valid column."""

    if scope is None or _column_lineage(node, exp, scope) is None:
        return None
    name = str(getattr(node, "name", ""))
    canonical_name = _sqlite_identifier_key(name)
    table = str(getattr(node, "table", ""))
    if table:
        return _source_for_qualified_name(scope, _sqlite_identifier_key(table))
    return _unqualified_source(scope, canonical_name)


def _join_column_reference(node: Any, exp: Any) -> Optional[tuple[str, str]]:
    """Return one explicit join-column reference without resolving SQL text."""

    if not _is_expression_type(node, exp, "Column") or _column_is_star(node):
        return None
    table = _sqlite_identifier_key(str(getattr(node, "table", "")))
    column = _sqlite_identifier_key(str(getattr(node, "name", "")))
    return (table, column) if table and column else None


def _safe_visits_patient_lookup_join(
    select: Any,
    exp: Any,
) -> Optional[str]:
    """Prove the one supported direct join preserves visit-row multiplicity.

    The local schema models patients as the single patient_id lookup for visits.
    This intentionally narrow proof permits the normal visit-to-patient age
    bucket query while rejecting every other multi-source aggregate scope until
    its cardinality can be established structurally.  SQL join order does not
    change that proof: both ``visits JOIN patients`` and
    ``patients JOIN visits`` preserve the visit-row multiset.  Return the
    visits alias so callers can still forbid patient-side aggregate measures.
    """

    source_nodes = _direct_source_nodes(select)
    joins = list(select.args.get("joins") or ())
    if len(source_nodes) != 2 or len(joins) != 1:
        return None
    left, right = source_nodes
    if (
        not _is_expression_type(left, exp, "Table")
        or not _is_expression_type(right, exp, "Table")
    ):
        return None

    aliases_by_table = {
        _sqlite_identifier_key(str(getattr(node, "name", ""))): _source_alias(node)
        for node in (left, right)
    }
    if set(aliases_by_table) != {"patients", "visits"}:
        return None
    visits_alias = aliases_by_table["visits"]
    patients_alias = aliases_by_table["patients"]
    if visits_alias is None or patients_alias is None:
        return None

    join = joins[0]
    if join.args.get("side") is not None or join.args.get("kind") is not None:
        return None
    condition = join.args.get("on")
    if not _is_expression_type(condition, exp, "EQ"):
        return None
    references = {
        _join_column_reference(condition.args.get("this"), exp),
        _join_column_reference(condition.args.get("expression"), exp),
    }
    if references != {
        (visits_alias, "patient_id"),
        (patients_alias, "patient_id"),
    }:
        return None
    return visits_alias


def _aggregate_safe_source_aliases(
    select: Any,
    exp: Any,
    scope: _Scope,
) -> frozenset[str]:
    """Return direct source aliases whose measures retain aggregate semantics.

    A one-source scope is safe for that source.  The supported visits-to-patients
    lookup preserves the visit-row multiset, not the patient-row multiset, so it
    permits visit-side measures and COUNT(*) but never patient-side measures.
    """

    if not scope.sources or not all(
        source.relation.aggregate_input_safe for source in scope.sources.values()
    ):
        return frozenset()
    if len(scope.sources) == 1:
        return frozenset(scope.sources)
    visits_alias = _safe_visits_patient_lookup_join(select, exp)
    if visits_alias is None or visits_alias not in scope.sources:
        return frozenset()
    return frozenset((visits_alias,))


def _scope_has_aggregate_safe_inputs(scope: Optional[_Scope]) -> bool:
    """Whether the current scope has a completed aggregate-input safety proof."""

    return scope is None or scope.aggregate_input_safe


def _source_is_aggregate_input_safe(scope: _Scope, source: _Source) -> bool:
    """Whether this exact direct source can supply an aggregate measure."""

    return any(
        scope.sources.get(alias) is source
        for alias in scope.aggregate_safe_source_aliases
    )


def _column_signature(node: Any, exp: Any, scope: Optional[_Scope] = None) -> Optional[str]:
    lineage = _column_lineage(node, exp, scope)
    return lineage.signature if lineage is not None else None


def _aggregate_argument_signature(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> Optional[str]:
    # An aggregate measure must be a real source column.  Presentation wrappers
    # such as CAST/ROUND are allowed only *around* a completed aggregate, not
    # around its input: otherwise AVG(CAST(cost AS INTEGER)) could impersonate
    # AVG(cost).
    raw = _unwrap_alias_or_paren(node, exp)
    if scope is not None and not _scope_has_aggregate_safe_inputs(scope):
        return None
    lineage = _column_lineage(raw, exp, scope)
    if lineage is None or lineage.kind in {
        "aggregate",
        "window_aggregate",
        "age_bucket",
        "cost_bucket",
    }:
        return None
    # A derived column can retain raw lineage while its enclosing CTE has
    # changed multiplicity through GROUP BY, DISTINCT, LIMIT, or a join.  Do
    # not allow an outer aggregate to treat that transformed row set as base.
    if scope is not None:
        source = _column_source(raw, exp, scope)
        if (
            source is None
            or not source.relation.aggregate_input_safe
            or not _source_is_aggregate_input_safe(scope, source)
        ):
            return None
    return lineage.aggregate_argument or lineage.signature


def _lookup_distinct_patient_id_signature(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> Optional[str]:
    """Prove COUNT(DISTINCT patient_id) across the supported lookup join.

    The visits-to-patients lookup can repeat patient-side values, so ordinary
    patient aggregates remain unsafe.  DISTINCT over the patient key removes
    exactly that multiplicity.  Keep this exception deliberately narrower
    than general patient-side aggregation and require the already-proven
    two-base-table lookup scope.
    """

    if (
        scope is None
        or not scope.aggregate_input_safe
        or len(scope.sources) != 2
        or {
            id(source.relation)
            for source in scope.sources.values()
        }
        != {
            id(_BASE_RELATIONS["patients"]),
            id(_BASE_RELATIONS["visits"]),
        }
    ):
        return None
    raw = _unwrap_alias_or_paren(node, exp)
    lineage = _column_lineage(raw, exp, scope)
    source = _column_source(raw, exp, scope)
    if (
        lineage is None
        or lineage.signature != "patient_id"
        or source is None
        or source.relation
        not in (_BASE_RELATIONS["patients"], _BASE_RELATIONS["visits"])
    ):
        return None
    return "patient_id"


def _lookup_idempotent_patient_measure_signature(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> Optional[str]:
    """Prove MIN/MAX over patient columns across the visits lookup join.

    The lookup may repeat one patient's value for several visits, which
    invalidates AVG/SUM but cannot change MIN or MAX over the matched cohort.
    """

    if (
        scope is None
        or not scope.aggregate_input_safe
        or len(scope.sources) != 2
        or {
            id(source.relation)
            for source in scope.sources.values()
        }
        != {
            id(_BASE_RELATIONS["patients"]),
            id(_BASE_RELATIONS["visits"]),
        }
    ):
        return None
    raw = _unwrap_alias_or_paren(node, exp)
    lineage = _column_lineage(raw, exp, scope)
    source = _column_source(raw, exp, scope)
    if (
        lineage is None
        or source is None
        or source.relation is not _BASE_RELATIONS["patients"]
        or lineage.kind in {"aggregate", "window_aggregate"}
    ):
        return None
    return lineage.aggregate_argument or lineage.signature


def _aggregate_signature(
    node: Any,
    exp: Any,
    scope: Optional[_Scope] = None,
) -> Optional[str]:
    # Keep aggregate recognition exact.  The caller may explicitly unwrap
    # harmless presentation wrappers after establishing that the inner node is
    # an aggregate; doing it here would let ROUND(name) pass as raw name.
    node = _unwrap_alias_or_paren(node, exp)
    if _is_expression_type(node, exp, "Window"):
        return None
    class_name = type(node).__name__.casefold()
    function = {
        "count": "COUNT",
        "sum": "SUM",
        "avg": "AVG",
        "min": "MIN",
        "max": "MAX",
    }.get(class_name)
    if function is None:
        # Custom function nodes still expose a name in sqlglot.  Restrict this
        # fallback to the five aggregate names rather than accepting arbitrary
        # functions as measures.
        function = {
            "count": "COUNT",
            "sum": "SUM",
            "avg": "AVG",
            "min": "MIN",
            "max": "MAX",
        }.get(_sqlite_identifier_key(str(getattr(node, "name", ""))))
    if function is None:
        return None

    argument = getattr(node, "this", None)
    distinct_type = getattr(exp, "Distinct", None)
    if distinct_type is not None and isinstance(argument, distinct_type):
        expressions = list(getattr(argument, "expressions", ()) or ())
        if not expressions and getattr(argument, "this", None) is not None:
            expressions = [argument.this]
        if len(expressions) != 1:
            return None
        column = _aggregate_argument_signature(expressions[0], exp, scope)
        if column is None and function == "COUNT":
            column = _lookup_distinct_patient_id_signature(
                expressions[0],
                exp,
                scope,
            )
        if column is None:
            return None
        return f"{function}(DISTINCT {column})"

    # Do not erase transformations inside an aggregate.  CAST/ROUND are
    # presentation wrappers only when they wrap a completed aggregate, never
    # when they change its input measure.
    argument = _unwrap_alias_or_paren(argument, exp)
    star_type = getattr(exp, "Star", None)
    if star_type is not None and isinstance(argument, star_type):
        if function != "COUNT" or not _scope_has_aggregate_safe_inputs(scope):
            return None
        argument_text = "*"
    else:
        argument_text = _aggregate_argument_signature(argument, exp, scope)
        if argument_text is None and function in {"MIN", "MAX"}:
            argument_text = _lookup_idempotent_patient_measure_signature(
                argument,
                exp,
                scope,
            )
        if argument_text is None:
            return None
    return f"{function}({argument_text})"


def _sqlite_cast_is_real_affinity(node: Any, exp: Any) -> bool:
    """Whether a CAST target preserves a numeric aggregate as SQLite REAL.

    Integer and text casts are semantic transformations, not presentation
    wrappers.  SQLGlot normalizes SQLite ``REAL`` to its ``FLOAT`` data type,
    so that is the only cast target accepted around a completed aggregate.
    """

    if not _is_expression_type(node, exp, "Cast"):
        return False
    target = node.args.get("to")
    target_kind = getattr(target, "args", {}).get("this")
    target_value = getattr(target_kind, "value", target_kind)
    return str(target_value).casefold() == "float"


def _round_has_literal_precision(node: Any, exp: Any) -> bool:
    """Require ROUND precision to be a non-negative integer literal."""

    if not _is_expression_type(node, exp, "Round"):
        return False
    decimals = node.args.get("decimals")
    if decimals is None:
        return True
    literal_type = getattr(exp, "Literal", None)
    if (
        literal_type is None
        or not isinstance(decimals, literal_type)
        or bool(getattr(decimals, "is_string", False))
    ):
        return False
    return str(getattr(decimals, "this", "")).isdigit()


def _presentation_aggregate_signature(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> Optional[str]:
    """Recognize only value-preserving wrappers around a proven aggregate."""

    raw = _unwrap_alias_or_paren(node, exp)
    saw_wrapper = False
    while True:
        if _round_has_literal_precision(raw, exp):
            saw_wrapper = True
            raw = _unwrap_alias_or_paren(raw.args.get("this"), exp)
            continue
        if _sqlite_cast_is_real_affinity(raw, exp):
            saw_wrapper = True
            raw = _unwrap_alias_or_paren(raw.args.get("this"), exp)
            continue
        break
    if not saw_wrapper:
        return None
    return _aggregate_signature(raw, exp, scope)


def _unwrap_alias_or_paren(node: Any, exp: Any) -> Any:
    wrappers = tuple(
        kind
        for name in ("Alias", "Paren")
        if (kind := getattr(exp, name, None)) is not None
    )
    while wrappers and isinstance(node, wrappers):
        node = node.this
    return node


def _window_has_partition_and_order(node: Any) -> bool:
    partition = list(node.args.get("partition_by") or ())
    order = node.args.get("order")
    return bool(partition and order and getattr(order, "expressions", ()))


def _window_has_order(node: Any) -> bool:
    order = node.args.get("order")
    return bool(order and getattr(order, "expressions", ()))


def _window_partition_signatures(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> list[str]:
    signatures: list[str] = []
    for expression in node.args.get("partition_by") or ():
        signature = _column_signature(expression, exp, scope)
        if signature is None:
            return []
        signatures.append(signature)
    return signatures


def _window_order_signatures(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> list[tuple[str, str]]:
    order = node.args.get("order")
    if order is None:
        return []
    ordered_type = getattr(exp, "Ordered", None)
    signatures: list[tuple[str, str]] = []
    for expression in getattr(order, "expressions", ()):
        is_ordered = ordered_type is not None and isinstance(
            expression,
            ordered_type,
        )
        raw = expression.this if is_ordered else expression
        signature = _aggregate_signature(raw, exp, scope)
        if signature is None:
            signature = _column_signature(raw, exp, scope)
        if signature is None:
            return []
        signatures.append(
            (
                signature,
                "DESC"
                if is_ordered and bool(expression.args.get("desc"))
                else "ASC",
            )
        )
    return signatures


def _window_has_cumulative_rows_frame(node: Any) -> bool:
    spec = node.args.get("spec")
    if spec is None:
        return False
    return (
        str(spec.args.get("kind", "")).upper() == "ROWS"
        and str(spec.args.get("start", "")).upper() == "UNBOUNDED"
        and str(spec.args.get("start_side", "")).upper() == "PRECEDING"
        and str(spec.args.get("end", "")).upper() == "CURRENT ROW"
    )


def _julianday_column_lineage(node: Any, exp: Any, scope: Optional[_Scope]) -> Optional[_Lineage]:
    if type(node).__name__.casefold() != "anonymous":
        return None
    if _sqlite_identifier_key(str(getattr(node, "name", ""))) != "julianday":
        return None
    expressions = list(getattr(node, "expressions", ()) or ())
    if len(expressions) != 1:
        return None
    return _column_lineage(_unwrap_expression(expressions[0], exp), exp, scope)


def _calendar_lineage(node: Any, exp: Any, scope: Optional[_Scope]) -> Optional[_Lineage]:
    """Recognize only the calendar buckets the contract can name exactly."""

    node_kind = type(node).__name__.casefold()
    source_node: Any
    signature: Optional[str]
    if node_kind in {"strftime", "datetostr", "timetostr"}:
        format_node = node.args.get("format")
        literal_type = getattr(exp, "Literal", None)
        if (
            literal_type is None
            or not isinstance(format_node, literal_type)
            or not bool(getattr(format_node, "is_string", False))
        ):
            return None
        signature = {"%Y-%m": "month", "%Y": "year"}.get(
            str(getattr(format_node, "this", ""))
        )
        column_type = getattr(exp, "Column", None)
        if signature is None or column_type is None:
            return None
        columns = list(node.find_all(column_type))
        if len(columns) != 1:
            return None
        source_node = columns[0]
    elif node_kind == "substring":
        # SQLite SUBSTR is accepted only for the two exact, prefix-preserving
        # calendar buckets used by the deterministic compiler.  Requiring
        # literal 1-based bounds prevents an arbitrary text slice from
        # laundering itself as a month/year dimension.
        if not _literal_equals(node.args.get("start"), exp, "1"):
            return None
        signature = None
        for length, bucket in (("7", "month"), ("4", "year")):
            if _literal_equals(node.args.get("length"), exp, length):
                signature = bucket
                break
        if signature is None:
            return None
        source_node = _unwrap_expression(node.args.get("this"), exp)
    else:
        return None
    source = _column_lineage(source_node, exp, scope)
    if source is None or source.signature != "visit_date":
        return None
    return _Lineage(signature, aggregate_argument=signature, kind="calendar")


def _is_forward_gap_difference(node: Any, exp: Any, scope: Optional[_Scope]) -> bool:
    if not _is_expression_type(node, exp, "Sub"):
        return False
    current = _julianday_column_lineage(node.args.get("this"), exp, scope)
    previous = _julianday_column_lineage(node.args.get("expression"), exp, scope)
    return (
        current is not None
        and current.signature == "visit_date"
        and previous is not None
        and previous.kind == "lag_visit_date"
    )


def _is_integer_cast(node: Any, exp: Any) -> bool:
    if not _is_expression_type(node, exp, "Cast"):
        return False
    target = node.args.get("to")
    kind = getattr(getattr(target, "this", None), "value", "")
    return str(kind).upper() in {"INT", "INTEGER"}


def _literal_equals(node: Any, exp: Any, value: str) -> bool:
    if not _is_expression_type(node, exp, "Literal"):
        return False
    return not bool(getattr(node, "is_string", False)) and str(getattr(node, "this", "")) == value


def _contains_previous_null(node: Any, exp: Any, scope: Optional[_Scope]) -> bool:
    for candidate in node.walk():
        if not _is_expression_type(candidate, exp, "Is"):
            continue
        previous = _column_lineage(candidate.args.get("this"), exp, scope)
        if previous is not None and previous.kind == "lag_visit_date" and _is_expression_type(
            candidate.args.get("expression"), exp, "Null"
        ):
            return True
    return False


def _contains_forward_gap_difference(node: Any, exp: Any, scope: Optional[_Scope]) -> bool:
    return any(_is_forward_gap_difference(candidate, exp, scope) for candidate in node.walk())


def _episode_marker_lineage(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
    output_name: Optional[str],
) -> Optional[_Lineage]:
    if output_name != "new_episode" or not _is_expression_type(node, exp, "Case"):
        return None
    if node.args.get("this") is not None or not _literal_equals(node.args.get("default"), exp, "0"):
        return None
    for branch in node.args.get("ifs") or ():
        condition = branch.args.get("this")
        if not _literal_equals(branch.args.get("true"), exp, "1"):
            continue
        if (
            condition is not None
            and _contains_previous_null(condition, exp, scope)
            and _contains_forward_gap_difference(condition, exp, scope)
        ):
            return _Lineage(None, aggregate_argument="new_episode", kind="episode_marker")
    return None


def _window_lineage(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
    output_name: Optional[str],
) -> Optional[_Lineage]:
    if not _window_has_order(node):
        return None
    inner = node.args.get("this")
    class_name = type(inner).__name__.casefold()

    if (
        output_name == "episode_id"
        and class_name == "sum"
        and _window_has_partition_and_order(node)
        and _window_has_cumulative_rows_frame(node)
    ):
        marker = _column_lineage(_unwrap_expression(getattr(inner, "this", None), exp), exp, scope)
        if marker is not None and marker.kind == "episode_marker":
            return _Lineage("episode_id", aggregate_argument="episode_id", kind="episode_id")

    controlled_ranks = {
        "fee_rank": (
            "rownumber",
            ["department"],
            [("cost", "DESC"), ("visit_id", "ASC")],
        ),
        "total_rank": (
            "denserank",
            [],
            [("SUM(cost)", "DESC")],
        ),
        "observation_rank": (
            "rownumber",
            [],
            [("SUM(count)", "DESC"), ("head", "ASC")],
        ),
    }
    controlled_rank_name = (
        "fee_rank"
        if output_name in {"fee_rank", "visit_rank"}
        else output_name or ""
    )
    controlled_rank = controlled_ranks.get(controlled_rank_name)
    if controlled_rank is not None:
        expected_kind, expected_partition, expected_order = controlled_rank
        if (
            class_name == expected_kind
            and _window_partition_signatures(node, exp, scope)
            == expected_partition
            and _window_order_signatures(node, exp, scope) == expected_order
        ):
            return _Lineage(
                controlled_rank_name,
                aggregate_argument=controlled_rank_name,
                kind=controlled_rank_name,
            )
    if (
        output_name == "visit_rank"
        and class_name == "rownumber"
        and _window_has_partition_and_order(node)
    ):
        return _Lineage(
            output_name,
            aggregate_argument=output_name,
            kind=output_name,
        )

    if (
        output_name == "previous_visit_date"
        and class_name == "lag"
        and _window_has_partition_and_order(node)
    ):
        source = _column_lineage(_unwrap_expression(getattr(inner, "this", None), exp), exp, scope)
        if source is not None and source.signature == "visit_date":
            return _Lineage(
                "previous_visit_date",
                aggregate_argument="previous_visit_date",
                kind="lag_visit_date",
            )

    # A windowed COUNT/AVG/SUM answers a different question from an ordinary
    # aggregate.  Only the explicitly modeled rank/episode/lag forms above are
    # accepted; every other window expression remains unsupported.
    return None


def _gap_days_lineage(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
    output_name: Optional[str],
) -> Optional[_Lineage]:
    if output_name != "gap_days" or not _is_integer_cast(node, exp):
        return None
    if _is_forward_gap_difference(node.args.get("this"), exp, scope):
        return _Lineage("gap_days", aggregate_argument="gap_days", kind="gap_days")
    return None


def _case_direct_numeric_comparison_lineage(
    condition: Any,
    exp: Any,
    scope: Optional[_Scope],
    expected_signature: str,
    *,
    require_base: bool = False,
) -> Optional[_Lineage]:
    """Recognize one direct numeric column/literal comparison."""

    comparison_types = tuple(
        kind
        for name in ("LT", "LTE", "GT", "GTE", "EQ", "NEQ")
        if (kind := getattr(exp, name, None)) is not None
    )
    literal_type = getattr(exp, "Literal", None)
    if (
        not comparison_types
        or literal_type is None
        or not isinstance(condition, comparison_types)
    ):
        return None

    left = condition.args.get("this")
    right = condition.args.get("expression")
    for column, literal in ((left, right), (right, left)):
        if (
            not _is_expression_type(column, exp, "Column")
            or not isinstance(literal, literal_type)
            or bool(getattr(literal, "is_string", False))
        ):
            continue
        lineage = _column_lineage(column, exp, scope)
        if (
            lineage is not None
            and lineage.signature == expected_signature
            and (not require_base or lineage.kind == "base")
        ):
            return lineage
    return None


def _case_age_comparison_lineage(
    condition: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> Optional[_Lineage]:
    return _case_direct_numeric_comparison_lineage(
        condition,
        exp,
        scope,
        "age",
    )


def _case_cost_comparison_lineage(
    condition: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> Optional[_Lineage]:
    # ``cost`` exists only on visits; base provenance prevents a previously
    # derived cost_bucket from being bucketed again as raw visit cost.
    return _case_direct_numeric_comparison_lineage(
        condition,
        exp,
        scope,
        "cost",
        require_base=True,
    )


def _searched_case_bucket_lineage(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
    output_name: Optional[str],
    *,
    allowed_output_names: frozenset[str],
    expected_signature: str,
    bucket_kind: str,
    comparison_lineage: Any,
    allow_unaliased: bool,
) -> Optional[_Lineage]:
    """Validate searched CASE shape and unique string bucket labels."""

    if not _is_expression_type(node, exp, "Case") or node.args.get("this") is not None:
        return None
    if output_name is None:
        if not allow_unaliased:
            return None
    elif output_name not in allowed_output_names:
        return None

    literal_type = getattr(exp, "Literal", None)
    branches = list(node.args.get("ifs") or ())
    if literal_type is None or not branches:
        return None

    labels: set[str] = set()
    for branch in branches:
        lineage = comparison_lineage(branch.args.get("this"), exp, scope)
        if lineage is None:
            return None
        label = branch.args.get("true")
        if (
            not isinstance(label, literal_type)
            or not bool(getattr(label, "is_string", False))
        ):
            return None
        label_text = str(getattr(label, "this", ""))
        if label_text in labels:
            return None
        labels.add(label_text)

    default = node.args.get("default")
    if (
        not isinstance(default, literal_type)
        or not bool(getattr(default, "is_string", False))
    ):
        return None
    if str(getattr(default, "this", "")) in labels:
        return None
    return _Lineage(
        expected_signature,
        aggregate_argument=None,
        kind=bucket_kind,
    )


def _case_age_source_lineage(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
    output_name: Optional[str],
    *,
    allow_unaliased: bool = False,
) -> Optional[_Lineage]:
    return _searched_case_bucket_lineage(
        node,
        exp,
        scope,
        output_name,
        allowed_output_names=frozenset(("age_group", "age_band")),
        expected_signature="age",
        bucket_kind="age_bucket",
        comparison_lineage=_case_age_comparison_lineage,
        allow_unaliased=allow_unaliased,
    )


def _case_cost_source_lineage(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
    output_name: Optional[str],
    *,
    allow_unaliased: bool = False,
) -> Optional[_Lineage]:
    return _searched_case_bucket_lineage(
        node,
        exp,
        scope,
        output_name,
        allowed_output_names=frozenset(("cost_band",)),
        expected_signature="cost",
        bucket_kind="cost_bucket",
        comparison_lineage=_case_cost_comparison_lineage,
        allow_unaliased=allow_unaliased,
    )


def _expression_has_volatile_source_lineage(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> bool:
    """Whether a local input column already carries recognized volatility.

    An unsupported expression must not downgrade a volatile source output to
    ``unsupported``: an outer ORDER BY could otherwise hide RANDOM() behind
    arithmetic, casts, or another unmodeled wrapper.  Nested query columns are
    deliberately excluded because they belong to their own scope and are
    audited independently.
    """

    if scope is None:
        return False
    column_type = getattr(exp, "Column", None)
    if column_type is None:
        return False
    query_boundaries = tuple(
        kind
        for name in ("Select", "SetOperation", "Subquery", "CTE")
        if (kind := getattr(exp, name, None)) is not None
    )
    for child in node.walk():
        if not isinstance(child, column_type) or _column_is_star(child):
            continue
        parent = getattr(child, "parent", None)
        while parent is not None and parent is not node:
            if query_boundaries and isinstance(parent, query_boundaries):
                break
            parent = getattr(parent, "parent", None)
        if parent is not node:
            continue
        lineage = _column_lineage(child, exp, scope)
        if lineage is not None and lineage.kind == "volatile":
            return True
    return False


def _expression_lineage(
    node: Any,
    exp: Any,
    *,
    scope: Optional[_Scope] = None,
    aliases: Optional[dict[str, _Lineage]] = None,
    resolve_alias: bool = False,
    output_name: Optional[str] = None,
    allow_unaliased_case: bool = False,
) -> Optional[_Lineage]:
    output_name = output_name or _output_name(node, exp)
    raw = _unwrap_alias_or_paren(node, exp)
    # Preserve volatility through a derived relation so an outer ORDER BY cannot
    # hide RANDOM() behind an otherwise ordinary-looking output column.
    if _contains_nondeterministic(raw) or _expression_has_volatile_source_lineage(
        raw,
        exp,
        scope,
    ):
        return _Lineage(None, kind="volatile")
    if _is_expression_type(raw, exp, "Window"):
        return _window_lineage(raw, exp, scope, output_name)
    marker = _episode_marker_lineage(raw, exp, scope, output_name)
    if marker is not None:
        return marker
    gap_days = _gap_days_lineage(raw, exp, scope, output_name)
    if gap_days is not None:
        return gap_days
    if _is_expression_type(raw, exp, "Case"):
        # CASE lineage is intentionally terminal: only the narrowly modeled
        # age and cost buckets may satisfy raw-dimension contracts.
        age_bucket = _case_age_source_lineage(
            raw,
            exp,
            scope,
            output_name,
            allow_unaliased=allow_unaliased_case,
        )
        if age_bucket is not None:
            return age_bucket
        return _case_cost_source_lineage(
            raw,
            exp,
            scope,
            output_name,
            allow_unaliased=allow_unaliased_case,
        )

    aggregate = _aggregate_signature(raw, exp, scope)
    if aggregate is None:
        # ROUND(CAST(AVG(cost) AS REAL), 2) is a presentation-only aggregate
        # wrapper.  It is recognized only after the unwrapped inner expression
        # has independently proven to be a supported aggregate.
        aggregate = _presentation_aggregate_signature(raw, exp, scope)
    if aggregate is not None:
        return _Lineage(aggregate, aggregate_argument=output_name or aggregate, kind="aggregate")
    if _is_expression_type(raw, exp, "Column"):
        column_name = _sqlite_identifier_key(str(getattr(raw, "name", "")))
        if (
            resolve_alias
            and not str(getattr(raw, "table", ""))
            and aliases
            and column_name in aliases
        ):
            return aliases[column_name]
        return _column_lineage(raw, exp, scope)
    star_type = getattr(exp, "Star", None)
    if star_type is not None and isinstance(raw, star_type):
        return _Lineage("*", aggregate_argument="*", kind="star")
    literal_type = getattr(exp, "Literal", None)
    if literal_type is not None and isinstance(raw, literal_type):
        literal_kind = "string" if bool(getattr(raw, "is_string", False)) else "number"
        return _Lineage(
            f"__literal__:{literal_kind}:{str(getattr(raw, 'this', ''))}",
            kind="literal",
        )
    calendar = _calendar_lineage(raw, exp, scope)
    if calendar is not None:
        return calendar
    return None


def _expression_signature(
    node: Any,
    exp: Any,
    *,
    scope: Optional[_Scope] = None,
    aliases: Optional[dict[str, _Lineage]] = None,
    resolve_alias: bool = False,
    allow_unaliased_case: bool = False,
) -> Optional[str]:
    lineage = _expression_lineage(
        node,
        exp,
        scope=scope,
        aliases=aliases,
        resolve_alias=resolve_alias,
        allow_unaliased_case=allow_unaliased_case,
    )
    return lineage.signature if lineage is not None else None


def _outer_select(tree: Any, exp: Any) -> Optional[Any]:
    node = tree
    wrappers = tuple(
        kind
        for name in ("Subquery", "Paren")
        if (kind := getattr(exp, name, None)) is not None
    )
    while wrappers and isinstance(node, wrappers):
        node = node.this
    select_type = getattr(exp, "Select", None)
    return node if select_type is not None and isinstance(node, select_type) else None


def _forbidden_read_nodes(exp: Any) -> tuple[type[Any], ...]:
    names = (
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Create",
        "Alter",
        "Attach",
        "Detach",
        "Command",
        "Pragma",
        "Merge",
        "Replace",
        "TruncateTable",
        "Set",
    )
    return tuple(kind for name in names if (kind := getattr(exp, name, None)) is not None)


_ALLOWED_SQLITE_ANONYMOUS_FUNCTIONS = frozenset(
    {
        # SQLite planner hints are semantic no-ops.
        "likely",
        "unlikely",
        "likelihood",
        # Deterministic when supplied a source value; current-time spellings are
        # still detected by the existing volatility audit.
        "time",
        "datetime",
        "julianday",
        "unixepoch",
        "timediff",
    }
)


def _unapproved_sqlite_function_reason(tree: Any, exp: Any) -> Optional[str]:
    """Return a fail-closed reason for any unapproved function in the AST.

    SQLite extensions expose side-effecting functions such as ``writefile`` as
    ordinary scalar calls, and SQLGlot represents unknown calls as Anonymous.
    Inspect the entire statement tree, including predicates, CTEs, and unused
    branches, so such a call cannot hide outside the final select list.
    """

    function_type = getattr(exp, "Func", None)
    if function_type is None:
        return "SQL function safety cannot be established"
    for node in tree.walk():
        if not isinstance(node, function_type):
            continue
        key = type(node).__name__.casefold()
        if key == "anonymous":
            name = _sqlite_identifier_key(str(getattr(node, "name", "")))
            if name in _ALLOWED_SQLITE_ANONYMOUS_FUNCTIONS:
                continue
            if name == "randomblob":
                return "nondeterministic or unsafe SQLite function is not allowed: randomblob"
            return f"unsafe or unapproved SQLite function is not allowed: {name or '<unknown>'}"
        # Named SQLGlot expression classes include both recognized built-ins
        # and structural nodes such as CASE/AND/COLLATE.  They remain subject
        # to the existing lineage and volatility gates.  Extension and unknown
        # SQLite calls are Anonymous and therefore fail closed above.
    return None


def _parse_candidate(sql: str) -> tuple[Any, Any, Any]:
    exp, parse = _sqlglot_import()
    try:
        statements = [tree for tree in parse(sql, read="sqlite") if tree is not None]
    except Exception as exc:
        raise ValueError(f"parse failure: {exc}") from exc
    if len(statements) != 1:
        raise ValueError("parse failure: expected exactly one SQL statement")
    tree = statements[0]
    if any(tree.find(kind) is not None for kind in _forbidden_read_nodes(exp)):
        raise PermissionError("candidate is not read-only")
    function_reason = _unapproved_sqlite_function_reason(tree, exp)
    if function_reason is not None:
        raise PermissionError(function_reason)
    predicate_nodes: list[Any] = []
    for kind_name in ("Where", "Having"):
        kind = getattr(exp, kind_name, None)
        if kind is not None:
            predicate_nodes.extend(
                node.this
                for node in tree.find_all(kind)
                if getattr(node, "this", None) is not None
            )
    join_type = getattr(exp, "Join", None)
    if join_type is not None:
        predicate_nodes.extend(
            node.args.get("on")
            for node in tree.find_all(join_type)
            if node.args.get("on") is not None
        )
    if any(
        _contains_local_nondeterministic(node, exp)
        for node in predicate_nodes
    ):
        raise PermissionError(
            "nondeterministic SQLite function is not allowed in a filter predicate"
        )
    outer = _outer_select(tree, exp)
    set_operation_type = getattr(exp, "SetOperation", None)
    if outer is None and set_operation_type is not None and isinstance(tree, set_operation_type):
        outer = tree
    if outer is None:
        raise TypeError("candidate is not a SELECT or compound read query")
    return exp, tree, outer


def _aliases_for_select(select: Any, exp: Any, scope: Optional[_Scope]) -> dict[str, _Lineage]:
    aliases: dict[str, _Lineage] = {}
    duplicate_aliases: set[str] = set()
    alias_type = getattr(exp, "Alias", None)
    if alias_type is None:
        return aliases
    for expression in getattr(select, "expressions", ()):
        if not isinstance(expression, alias_type):
            continue
        alias = _sqlite_identifier_key(str(getattr(expression, "alias", "")))
        if not alias:
            continue
        if alias in aliases:
            duplicate_aliases.add(alias)
            continue
        lineage = _expression_lineage(
            expression.this,
            exp,
            scope=scope,
            output_name=alias,
        )
        if lineage is not None and lineage.signature is not None:
            aliases[alias] = lineage
    for alias in duplicate_aliases:
        aliases.pop(alias, None)
    return aliases


def _projection_signatures(
    select: Any,
    exp: Any,
    scope: Optional[_Scope],
    aliases: dict[str, _Lineage],
) -> tuple[list[str], list[str], list[str]]:
    """Return raw outputs, aggregate outputs, and their original select order."""

    projections: list[str] = []
    aggregates: list[str] = []
    select_list: list[str] = []
    for expression in getattr(select, "expressions", ()):
        signature = _expression_signature(
            expression,
            exp,
            scope=scope,
            aliases=aliases,
            resolve_alias=False,
        )
        if signature is None:
            signature = "<unsupported expression>"
        select_list.append(signature)
        if _canonical_aggregate_text(signature) is not None:
            aggregates.append(signature)
        else:
            projections.append(signature)
    return projections, aggregates, select_list


def _unwrap_ordinal_reference(
    node: Any,
    exp: Any,
    *,
    allow_collation: bool = False,
) -> Any:
    """Remove basic wrappers while retaining whether collation remains semantic."""

    names = ("Alias", "Paren", "Collate") if allow_collation else ("Alias", "Paren")
    wrappers = tuple(kind for name in names if (kind := getattr(exp, name, None)) is not None)
    while wrappers and isinstance(node, wrappers):
        node = node.this
    return node


_SQLITE_ORDINAL_MAX = (1 << 31) - 1


def _literal_ordinal(raw: Any, exp: Any) -> Optional[int]:
    """Return a SQLite-sized integer literal ordinal from an unwrapped node."""

    literal_type = getattr(exp, "Literal", None)
    if (
        literal_type is None
        or not isinstance(raw, literal_type)
        or bool(getattr(raw, "is_string", False))
    ):
        return None
    text = str(getattr(raw, "this", ""))
    if not text.isdigit():
        return None
    try:
        value = int(text)
    except ValueError:
        # CPython may reject exceptionally long decimal strings.  Candidate
        # validation must abstain rather than let a parser-shaped input escape
        # the read-only gate as an exception.
        return None
    # SQLite's ORDER/GROUP positional aliases are signed 32-bit integers.
    # Larger numeric literals remain ordinary expressions rather than slots.
    return value if value <= _SQLITE_ORDINAL_MAX else None


def _sqlite_likelihood_probability(node: Any, exp: Any) -> bool:
    """Whether a likelihood() hint has a plainly valid SQLite probability."""

    raw = _unwrap_ordinal_reference(node, exp)
    literal_type = getattr(exp, "Literal", None)
    if (
        literal_type is None
        or not isinstance(raw, literal_type)
        or bool(getattr(raw, "is_string", False))
    ):
        return False
    try:
        value = float(str(getattr(raw, "this", "")))
    except (TypeError, ValueError, OverflowError):
        return False
    return 0.0 <= value <= 1.0


def _sqlite_likely_argument(node: Any, exp: Any) -> Optional[Any]:
    """Return the transparent argument of a valid SQLite likely-family hint."""

    function_name = _sqlite_identifier_key(str(getattr(node, "name", "")))
    if not function_name:
        function_name = type(node).__name__.casefold()
    arguments = tuple(getattr(node, "expressions", ()) or ())
    if function_name in {"likely", "unlikely"} and len(arguments) == 1:
        return arguments[0]
    if (
        function_name == "likelihood"
        and len(arguments) == 2
        and _sqlite_likelihood_probability(arguments[1], exp)
    ):
        return arguments[0]
    return None


def _unwrap_sqlite_ordinal_reference(
    node: Any,
    exp: Any,
    *,
    allow_collation: bool,
) -> Any:
    """Strip wrappers SQLite ignores while resolving ORDER/GROUP ordinals."""

    while True:
        raw = _unwrap_ordinal_reference(node, exp, allow_collation=allow_collation)
        likely_argument = _sqlite_likely_argument(raw, exp)
        if likely_argument is None:
            return raw
        node = likely_argument


def _sqlite_integer_ordinal(
    node: Any,
    exp: Any,
    *,
    allow_collation: bool,
    allow_ambiguous_hex: bool,
) -> Optional[tuple[int, bool]]:
    """Return a SQLite integer ordinal and whether it used ambiguous hex syntax."""

    raw = _unwrap_sqlite_ordinal_reference(node, exp, allow_collation=allow_collation)
    neg_type = getattr(exp, "Neg", None)
    if neg_type is not None and isinstance(raw, neg_type):
        value = _sqlite_integer_ordinal(
            raw.this,
            exp,
            allow_collation=allow_collation,
            allow_ambiguous_hex=allow_ambiguous_hex,
        )
        return (-value[0], value[1]) if value is not None else None

    ordinal = _literal_ordinal(raw, exp)
    if ordinal is not None:
        return ordinal, False

    hex_string_type = getattr(exp, "HexString", None)
    if not allow_ambiguous_hex or hex_string_type is None or not isinstance(raw, hex_string_type):
        return None
    text = str(getattr(raw, "this", ""))
    if not re.fullmatch(r"[0-9A-Fa-f]+", text):
        return None
    try:
        value = int(text, 16)
    except ValueError:
        return None
    return (value, True) if value <= _SQLITE_ORDINAL_MAX else None


def _numeric_literal_ordinal(node: Any, exp: Any) -> Optional[int]:
    """Return an unambiguous ordinal without erasing ordering collation."""

    value = _sqlite_integer_ordinal(
        node,
        exp,
        allow_collation=False,
        allow_ambiguous_hex=False,
    )
    return value[0] if value is not None and not value[1] else None


def _paging_numeric_literal_ordinal(node: Any, exp: Any) -> Optional[int]:
    """Return an ordinal for volatility checks, where COLLATE preserves its slot."""

    value = _sqlite_integer_ordinal(
        node,
        exp,
        allow_collation=True,
        allow_ambiguous_hex=False,
    )
    return value[0] if value is not None and not value[1] else None


def _ambiguous_hex_ordinal(node: Any, exp: Any) -> Optional[int]:
    """Return a possible SQLite hex ordinal for fail-closed paging checks.

    SQLGlot's SQLite AST represents integer ``0x2`` and blob ``X'02'``
    spellings as the same HexString shape.  Callers must use this only to
    reject a volatile ordering path, never to accept grouping or ordering
    semantics that depend on the ambiguous spelling.
    """

    value = _sqlite_integer_ordinal(
        node,
        exp,
        allow_collation=True,
        allow_ambiguous_hex=True,
    )
    return value[0] if value is not None and value[1] else None


def _group_signatures(
    select: Any,
    exp: Any,
    scope: Optional[_Scope],
    aliases: dict[str, _Lineage],
    projections: Sequence[str],
    *,
    include_inherited: bool = False,
) -> list[str]:
    group = select.args.get("group") or select.args.get("group_by")
    if group is None:
        if include_inherited:
            inherited = _transparent_derived_grouping(select, exp, scope)
            if inherited is not None:
                return list(inherited)
        return []
    result: list[str] = []
    for expression in getattr(group, "expressions", ()):
        # SQLite gives source columns precedence over SELECT aliases in GROUP
        # BY.  Resolve an output alias only when no visible source can shadow
        # it; otherwise preserve real source-column resolution.  Its valid
        # likely-family hints are transparent, but COLLATE remains visible
        # because it changes grouping semantics.
        signature_node = _unwrap_sqlite_ordinal_reference(
            expression,
            exp,
            allow_collation=False,
        )
        signature = _expression_signature(
            signature_node,
            exp,
            scope=scope,
            aliases=aliases,
            resolve_alias=_is_direct_group_alias(signature_node, select, exp, scope),
            allow_unaliased_case=True,
        )
        ordinal = _numeric_literal_ordinal(expression, exp)
        if ordinal is not None and 1 <= ordinal <= len(projections):
            signature = projections[ordinal - 1]
        result.append(signature or "<unsupported expression>")
    return result


def _order_signatures(
    select: Any,
    exp: Any,
    scope: Optional[_Scope],
    aliases: dict[str, _Lineage],
    projections: Sequence[str],
) -> tuple[list[tuple[str, str, str]], list[Any]]:
    order = select.args.get("order") or select.args.get("order_by")
    if order is None:
        return [], []
    ordered_type = getattr(exp, "Ordered", None)
    values: list[tuple[str, str, str]] = []
    nodes: list[Any] = []
    for item in getattr(order, "expressions", ()):
        is_ordered = ordered_type is not None and isinstance(item, ordered_type)
        node = item.this if is_ordered else item
        # SQLite's valid likely-family hints are planner-only wrappers.  They
        # preserve an ORDER BY key's semantic identity, while COLLATE remains
        # visible so a changed collation cannot satisfy the default contract.
        signature_node = _unwrap_sqlite_ordinal_reference(
            node,
            exp,
            allow_collation=False,
        )
        signature = _expression_signature(
            signature_node,
            exp,
            scope=scope,
            aliases=aliases,
            resolve_alias=True,
        )
        ordinal = _numeric_literal_ordinal(node, exp)
        if ordinal is not None and 1 <= ordinal <= len(projections):
            signature = projections[ordinal - 1]
        if signature is None:
            signature = "<unsupported expression>"
        desc = bool(item.args.get("desc")) if is_ordered else False
        nulls_first = item.args.get("nulls_first") if is_ordered else None
        if nulls_first is True:
            null_placement = "NULLS FIRST"
        elif nulls_first is False:
            null_placement = "NULLS LAST"
        else:
            # Do not silently erase an unrecognized parser representation when
            # a contract requires this ordering slot.
            null_placement = "<unsupported null placement>"
        values.append((signature, "DESC" if desc else "ASC", null_placement))
        nodes.append(node)
    return values, nodes


def _direct_source_nodes(select: Any) -> list[Any]:
    nodes: list[Any] = []
    from_clause = select.args.get("from_") or select.args.get("from")
    if from_clause is not None:
        first = getattr(from_clause, "this", None)
        if first is not None:
            nodes.append(first)
        nodes.extend(item for item in getattr(from_clause, "expressions", ()) if item is not None)
    for join in select.args.get("joins") or ():
        source = getattr(join, "this", None)
        if source is not None:
            nodes.append(source)
    return nodes


def _source_alias(node: Any) -> Optional[str]:
    alias = str(getattr(node, "alias_or_name", ""))
    return _sqlite_identifier_key(alias) if alias else None


def _invalid_relation() -> _Relation:
    return _Relation({}, valid=False)


def _rename_relation_outputs(
    relation: _Relation,
    names: Sequence[str],
    audit: _LineageAudit,
) -> _Relation:
    canonical_names = [_sqlite_identifier_key(str(name)) for name in names if str(name)]
    if len(canonical_names) != len(names) or len(set(canonical_names)) != len(canonical_names):
        audit.reject("ambiguous CTE output names")
        return _invalid_relation()
    outputs = list(relation.outputs.values())
    if len(outputs) != len(canonical_names):
        audit.reject("CTE column list does not match its projection")
        return _invalid_relation()
    renamed = _Relation(
        outputs=dict(zip(canonical_names, outputs)),
        grouping=relation.grouping,
        valid=relation.valid,
        has_base_source=relation.has_base_source,
        aggregate_input_safe=relation.aggregate_input_safe,
        known_single_row=relation.known_single_row,
    )
    source_scope = audit.relation_scopes.get(id(relation))
    if source_scope is not None:
        audit.relation_scopes[id(renamed)] = source_scope
    return renamed


def _values_relation(node: Any, exp: Any, names: Sequence[str], audit: _LineageAudit) -> _Relation:
    if not names:
        audit.reject("VALUES CTE requires explicit output names")
        return _invalid_relation()
    literal_type = getattr(exp, "Literal", None)
    tuple_type = getattr(exp, "Tuple", None)
    rows = list(getattr(node, "expressions", ()) or ())
    for row in rows:
        values = (
            list(getattr(row, "expressions", ()) or ())
            if tuple_type is not None and isinstance(row, tuple_type)
            else [row]
        )
        if len(values) != len(names) or any(
            literal_type is None or not isinstance(value, literal_type) for value in values
        ):
            audit.reject("VALUES CTE contains unsupported expressions")
            return _invalid_relation()
    canonical_names = [_sqlite_identifier_key(str(name)) for name in names]
    if len(set(canonical_names)) != len(canonical_names):
        audit.reject("ambiguous CTE output names")
        return _invalid_relation()
    return _Relation(
        {
            name: _Lineage(None, aggregate_argument=name, kind="literal_source")
            for name in canonical_names
        },
        known_single_row=len(rows) == 1,
    )


def _resolve_local_ctes(
    query: Any,
    exp: Any,
    audit: _LineageAudit,
    *,
    visible_ctes: dict[str, _Relation],
    blocked_ctes: frozenset[str],
) -> tuple[dict[str, _Relation], frozenset[str]]:
    """Bind one query root's local CTEs without assuming it is a SELECT.

    SQLite attaches a WITH clause to a compound SetOperation root, rather than
    to either SELECT branch.  Keep the existing left-to-right CTE proof for
    every query root so those branch scopes never fall back to base relations.
    """

    with_clause = query.args.get("with_")
    local_ctes = list(getattr(with_clause, "expressions", ()) or ())
    local_names = [
        _sqlite_identifier_key(str(getattr(cte, "alias_or_name", "")))
        for cte in local_ctes
    ]
    duplicate_names = {
        name for name in local_names if name and local_names.count(name) > 1
    }
    recursive = bool(getattr(with_clause, "args", {}).get("recursive"))
    if recursive:
        audit.reject("recursive CTEs are not supported")

    ctes = dict(visible_ctes)
    pending = set(blocked_ctes) | {name for name in local_names if name}
    for cte, name in zip(local_ctes, local_names):
        if not name:
            audit.reject("unnamed CTE")
            continue
        if name in duplicate_names:
            audit.reject(f"duplicate CTE name: {name}")
            ctes[name] = _invalid_relation()
            pending.discard(name)
            continue
        if name in _BASE_RELATIONS:
            audit.reject(f"CTE shadows base relation: {name}")
            ctes[name] = _invalid_relation()
            pending.discard(name)
            continue
        if recursive:
            ctes[name] = _invalid_relation()
            pending.discard(name)
            continue
        cte_declared_names = list(getattr(cte, "alias_column_names", ()) or ())
        relation = _relation_from_query(
            cte.this,
            exp,
            audit,
            visible_ctes=ctes,
            blocked_ctes=frozenset(pending),
            declared_names=cte_declared_names,
        )
        ctes[name] = relation
        pending.discard(name)
    return ctes, frozenset(pending)


def _relation_from_query(
    node: Any,
    exp: Any,
    audit: _LineageAudit,
    *,
    visible_ctes: dict[str, _Relation],
    blocked_ctes: frozenset[str],
    declared_names: Sequence[str] = (),
) -> _Relation:
    select = _outer_select(node, exp)
    if select is not None:
        relation, _ = _analyse_select(
            select,
            exp,
            audit,
            parent=None,
            visible_ctes=visible_ctes,
            blocked_ctes=blocked_ctes,
            declared_names=declared_names,
            derived=True,
        )
        return _rename_relation_outputs(relation, declared_names, audit) if declared_names else relation
    if _is_expression_type(node, exp, "Values"):
        return _values_relation(node, exp, declared_names, audit)
    if _is_expression_type(node, exp, "SetOperation"):
        # Compound CTE/source bodies still need their local WITH environment
        # audited, even though the validator does not synthesize a relation
        # proof across branches.
        _analyse_set_operation(
            node,
            exp,
            audit,
            parent=None,
            visible_ctes=visible_ctes,
            blocked_ctes=blocked_ctes,
        )
        audit.reject("unsupported compound derived relation")
        return _invalid_relation()
    audit.reject("unsupported derived relation")
    return _invalid_relation()


def _source_from_node(
    node: Any,
    exp: Any,
    audit: _LineageAudit,
    *,
    visible_ctes: dict[str, _Relation],
    blocked_ctes: frozenset[str],
    allow_unnamed_derived: bool = False,
) -> tuple[Optional[str], _Source]:
    if _is_expression_type(node, exp, "Table"):
        raw_name = str(getattr(node, "name", ""))
        identifier = getattr(node, "this", None)
        alias = _source_alias(node)
        if bool(getattr(identifier, "args", {}).get("quoted")) and "." in raw_name:
            # A quoted dot is part of one SQLite identifier, not a database
            # qualifier.  Do not split it and accidentally authorize its suffix
            # as one of the schema-bound base relations.
            audit.reject("quoted dotted source relation is not supported")
            return alias, _Source(_invalid_relation())
        name = _sqlite_identifier_key(raw_name)
        if not name or alias is None:
            audit.reject("unnamed source relation")
            return alias, _Source(_invalid_relation())
        if node.args.get("db") is not None or node.args.get("catalog") is not None:
            audit.reject("qualified database sources are not supported")
            return alias, _Source(_invalid_relation())
        # SQLite searches every CTE name in the innermost WITH context before
        # considering outer contexts.  A local name that is still pending must
        # therefore mask an inherited CTE with the same spelling.
        if name in blocked_ctes:
            audit.reject(f"forward or cyclic CTE reference: {name}")
            return alias, _Source(_invalid_relation())
        if name in visible_ctes:
            return alias, _Source(visible_ctes[name])
        relation = _BASE_RELATIONS.get(name)
        if relation is None:
            audit.reject(f"forbidden source relation: {name or '<unnamed>'}")
            return alias, _Source(_invalid_relation())
        return alias, _Source(relation, is_base=True)

    if _is_expression_type(node, exp, "Subquery"):
        alias = _source_alias(node)
        if alias is None:
            if not allow_unnamed_derived:
                audit.reject("derived source requires an alias")
                return None, _Source(_invalid_relation())
            relation = _relation_from_query(
                node.this,
                exp,
                audit,
                visible_ctes=visible_ctes,
                blocked_ctes=blocked_ctes,
            )
            # SQLite permits a single unaliased derived source and resolves its
            # outputs only as unqualified names.  The internal key is never a
            # SQL-visible qualifier, so a fabricated table prefix cannot gain
            # provenance through this compatibility path.
            return "\x00unnamed-derived", _Source(
                relation,
                qualified_name_visible=False,
            )
        relation = _relation_from_query(
            node.this,
            exp,
            audit,
            visible_ctes=visible_ctes,
            blocked_ctes=blocked_ctes,
        )
        return alias, _Source(relation)

    audit.reject("unsupported source relation")
    return None, _Source(_invalid_relation())


def _expand_projection_star(
    expression: Any,
    exp: Any,
    scope: _Scope,
    audit: _LineageAudit,
) -> Optional[list[tuple[str, _Lineage]]]:
    if _is_expression_type(expression, exp, "Star"):
        audit.reject("wildcard projection is not supported")
        return []
    if not _is_expression_type(expression, exp, "Column") or not _column_is_star(expression):
        return None
    table = _sqlite_identifier_key(str(getattr(expression, "table", "")))
    source = _source_for_qualified_name(scope, table) if table else None
    if source is None or source.is_base or not source.relation.valid:
        audit.reject("wildcard projection is not supported")
        return []
    return list(source.relation.outputs.items())


def _relation_outputs(
    select: Any,
    exp: Any,
    scope: _Scope,
    audit: _LineageAudit,
    *,
    declared_names: Sequence[str] = (),
    derived: bool,
) -> tuple[dict[str, _Lineage], bool]:
    canonical_names = (
        [_sqlite_identifier_key(str(name)) for name in declared_names if str(name)]
        if declared_names
        else []
    )
    items: list[tuple[str, _Lineage]] = []
    for expression in getattr(select, "expressions", ()):
        expanded = _expand_projection_star(expression, exp, scope, audit)
        if expanded is not None:
            items.extend(expanded)
            continue
        # A CTE column list defines the outward name even when its SELECT item
        # is unaliased.  Feed that name into expression lineage before output
        # collection so modeled derived dimensions (for example age_group CASE)
        # are judged by their actual public name.
        declared_name = (
            canonical_names[len(items)]
            if len(items) < len(canonical_names)
            else None
        )
        name = declared_name or _output_name(expression, exp)
        if name is None:
            if derived:
                audit.reject("derived projection requires an explicit output name")
            continue
        lineage = _expression_lineage(expression, exp, scope=scope, output_name=name)
        items.append((name, lineage or _Lineage(None, kind="unsupported")))

    if declared_names:
        if len(canonical_names) != len(declared_names) or len(canonical_names) != len(items):
            audit.reject("CTE column list does not match its projection")
            return {}, False
        items = list(zip(canonical_names, (lineage for _, lineage in items)))

    outputs: dict[str, _Lineage] = {}
    valid = True
    for name, lineage in items:
        if name in outputs:
            if derived:
                audit.reject(f"ambiguous derived output: {name}")
            valid = False
            continue
        outputs[name] = lineage
    return outputs, valid


def _explicit_aliases(select: Any, exp: Any, scope: _Scope, audit: _LineageAudit) -> dict[str, _Lineage]:
    aliases: dict[str, _Lineage] = {}
    alias_type = getattr(exp, "Alias", None)
    if alias_type is None:
        return aliases
    for expression in getattr(select, "expressions", ()):
        if not isinstance(expression, alias_type):
            continue
        alias = _sqlite_identifier_key(str(getattr(expression, "alias", "")))
        if not alias:
            continue
        if alias in aliases:
            audit.reject(f"duplicate output alias: {alias}")
            aliases.pop(alias, None)
            continue
        lineage = _expression_lineage(expression.this, exp, scope=scope, output_name=alias)
        if lineage is not None and (
            lineage.signature is not None or lineage.kind == "volatile"
        ):
            aliases[alias] = lineage
    return aliases


def _compound_output_alias_index(
    name: str,
    exp: Any,
    output_queries: Sequence[Any],
) -> Optional[int]:
    """Resolve a compound output alias in SQLite's left-to-right branch order."""

    for query in output_queries:
        matches = [
            index
            for index, expression in enumerate(getattr(query, "expressions", ()) or (), start=1)
            if _output_name(expression, exp) == name
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            # SQLite's first matching SELECT owns the name.  Do not guess a
            # slot when that branch itself exposes the name more than once.
            return None
    return None


def _sqlite_transparent_reference_parent(node: Any, exp: Any) -> Any:
    """Return the first nontransparent parent of an SQLite alias reference."""

    wrappers = tuple(
        kind
        for name in ("Alias", "Paren", "Collate")
        if (kind := getattr(exp, name, None)) is not None
    )
    child = node
    parent = getattr(child, "parent", None)
    while parent is not None:
        if wrappers and isinstance(parent, wrappers):
            child = parent
            parent = getattr(parent, "parent", None)
            continue
        # SQLite treats a valid likely-family hint as transparent while it
        # resolves a GROUP BY or ORDER BY output alias.  Track the immediately enclosed
        # child so parentheses/collation inside the hint remain transparent.
        if _sqlite_likely_argument(parent, exp) is child:
            child = parent
            parent = getattr(parent, "parent", None)
            continue
        return parent
    return None


def _is_compound_order_output_alias(node: Any, exp: Any) -> bool:
    """Whether an unqualified column names a compound query's output slot."""

    if (
        not _is_expression_type(node, exp, "Column")
        or _column_is_star(node)
        or str(getattr(node, "table", ""))
    ):
        return False
    parent = _sqlite_transparent_reference_parent(node, exp)
    ordered_type = getattr(exp, "Ordered", None)
    if ordered_type is None or not isinstance(parent, ordered_type):
        return False
    order = getattr(parent, "parent", None)
    query = getattr(order, "parent", None)
    set_operation_type = getattr(exp, "SetOperation", None)
    if set_operation_type is None or not isinstance(query, set_operation_type):
        return False
    leaves, _ = _set_operation_select_leaves(query, exp)
    name = _sqlite_identifier_key(str(getattr(node, "name", "")))
    return bool(name) and _compound_output_alias_index(name, exp, leaves) is not None


def _column_belongs_to_scope(node: Any, select: Any, exp: Any) -> bool:
    parent = getattr(node, "parent", None)
    while parent is not None and parent is not select:
        if _is_expression_type(parent, exp, "Select") or _is_expression_type(parent, exp, "CTE"):
            return False
        parent = getattr(parent, "parent", None)
    return parent is select


def _is_direct_order_alias(node: Any, select: Any, exp: Any, scope: _Scope) -> bool:
    if str(getattr(node, "table", "")):
        return False
    name = _sqlite_identifier_key(str(getattr(node, "name", "")))
    if name not in scope.aliases:
        return False
    parent = _sqlite_transparent_reference_parent(node, exp)
    ordered_type = getattr(exp, "Ordered", None)
    if ordered_type is None or not isinstance(parent, ordered_type):
        return False
    order = select.args.get("order") or select.args.get("order_by")
    return getattr(parent, "parent", None) is order


def _visible_source_has_output(scope: Optional[_Scope], name: str) -> bool:
    """Whether SQLite can resolve an unqualified name as a source column."""

    current = scope
    while current is not None:
        if any(name in source.relation.outputs for source in current.sources.values()):
            return True
        current = current.parent
    return False


def _is_direct_group_alias(node: Any, select: Any, exp: Any, scope: _Scope) -> bool:
    """Allow a GROUP BY output alias only when no source column can shadow it."""

    if not _is_expression_type(node, exp, "Column") or _column_is_star(node):
        return False
    if str(getattr(node, "table", "")):
        return False
    name = _sqlite_identifier_key(str(getattr(node, "name", "")))
    if not name or name not in scope.aliases or _visible_source_has_output(scope, name):
        return False
    group = select.args.get("group") or select.args.get("group_by")
    return _sqlite_transparent_reference_parent(node, exp) is group


def _audit_scope_columns(select: Any, exp: Any, scope: _Scope, audit: _LineageAudit) -> None:
    column_type = getattr(exp, "Column", None)
    if column_type is None:
        return
    for column in select.find_all(column_type):
        if not _column_belongs_to_scope(column, select, exp) or _column_is_star(column):
            continue
        if (
            _is_direct_order_alias(column, select, exp, scope)
            or _is_direct_group_alias(column, select, exp, scope)
            or _is_compound_order_output_alias(column, exp)
        ):
            continue
        if _column_lineage(column, exp, scope) is not None:
            continue
        table = str(getattr(column, "table", ""))
        name = str(getattr(column, "name", ""))
        reference = f"{table}.{name}" if table else name
        audit.reject(f"unknown or ambiguous source column: {reference or '<unnamed>'}")


def _transparent_derived_grouping(
    select: Any,
    exp: Any,
    scope: Optional[_Scope],
) -> Optional[tuple[str, ...]]:
    if scope is None or len(scope.sources) != 1:
        return None
    source = next(iter(scope.sources.values()))
    if source.is_base or not source.relation.valid or not source.relation.grouping:
        return None
    for expression in getattr(select, "expressions", ()):
        raw = _unwrap_alias_or_paren(expression, exp)
        if not _is_expression_type(raw, exp, "Column") or _column_is_star(raw):
            return None
        if _column_lineage(raw, exp, scope) is None:
            return None
    return source.relation.grouping


def _derived_relation_is_aggregate_input_safe(
    select: Any,
    exp: Any,
    scope: _Scope,
    outputs: dict[str, _Lineage],
) -> bool:
    """Prove that a derived relation does not reshape aggregate input rows.

    Filtering remains allowed here: the validator already permits predicates on
    direct aggregate queries, while this narrow check prevents structural
    laundering that changes multiplicity (deduplication, grouping, sampling,
    aggregates, or joins) before an otherwise valid outer aggregate.
    """

    if len(scope.sources) != 1 or not _scope_has_aggregate_safe_inputs(scope):
        return False
    if any(
        select.args.get(name) is not None
        for name in ("distinct", "group", "group_by", "limit", "offset", "having")
    ):
        return False
    select_type = getattr(exp, "Select", None)
    if select_type is not None and any(
        nested is not select for nested in select.find_all(select_type)
    ):
        # A correlated/scalar/IN subquery can apply its own deduplication,
        # grouping, sampling, or membership test before the outer aggregate.
        # This narrow transparency proof intentionally accepts only direct
        # predicates over the one source relation.
        return False
    return not any(
        lineage.kind in {"aggregate", "window_aggregate"}
        for lineage in outputs.values()
    )


def _has_singleton_proof_barrier(select: Any, *, allow_where: bool) -> bool:
    """Reject clauses outside the deliberately tiny singleton proof language."""

    clauses = {
        "distinct",
        "group",
        "group_by",
        "having",
        "limit",
        "offset",
        "order",
        "order_by",
        "qualify",
    }
    if not allow_where:
        clauses.add("where")
    return any(select.args.get(name) is not None for name in clauses)


def _scope_has_allowed_base_source(scope: _Scope) -> bool:
    return any(
        source.is_base or source.relation.has_base_source
        for source in scope.sources.values()
    )


def _literal_select_is_known_single_row(
    select: Any,
    scope: _Scope,
    outputs: dict[str, _Lineage],
    *,
    outputs_valid: bool,
) -> bool:
    """Prove a source-less, literal-only SELECT returns exactly one row."""

    return (
        outputs_valid
        and bool(outputs)
        and not scope.sources
        and not _has_singleton_proof_barrier(select, allow_where=False)
        and all(lineage.kind == "literal" for lineage in outputs.values())
    )


def _scalar_aggregate_select_is_known_single_row(
    select: Any,
    scope: _Scope,
    outputs: dict[str, _Lineage],
    *,
    outputs_valid: bool,
) -> bool:
    """Prove a supported, non-grouped aggregate relation has one output row."""

    return (
        outputs_valid
        and bool(outputs)
        and bool(scope.sources)
        and _scope_has_allowed_base_source(scope)
        and _scope_has_aggregate_safe_inputs(scope)
        and not _has_singleton_proof_barrier(select, allow_where=True)
        and all(lineage.kind == "aggregate" for lineage in outputs.values())
    )


def _transparent_singleton_passthrough_is_known_single_row(
    select: Any,
    exp: Any,
    scope: _Scope,
    outputs: dict[str, _Lineage],
    *,
    outputs_valid: bool,
) -> bool:
    """Propagate an exact singleton proof through direct column pass-through."""

    if (
        not outputs_valid
        or not outputs
        or len(scope.sources) != 1
        or _has_singleton_proof_barrier(select, allow_where=False)
    ):
        return False
    source = next(iter(scope.sources.values()))
    if source.is_base or not source.relation.valid or not source.relation.known_single_row:
        return False
    for expression in getattr(select, "expressions", ()):
        raw = _unwrap_alias_or_paren(expression, exp)
        if not _is_expression_type(raw, exp, "Column") or _column_is_star(raw):
            return False
        if _column_source(raw, exp, scope) is not source:
            return False
        if _column_lineage(raw, exp, scope) is None:
            return False
    return True


def _derived_relation_is_known_single_row(
    select: Any,
    exp: Any,
    scope: _Scope,
    outputs: dict[str, _Lineage],
    *,
    outputs_valid: bool,
) -> bool:
    """Return only an exact structural proof that a derived relation is singleton."""

    return (
        _literal_select_is_known_single_row(
            select,
            scope,
            outputs,
            outputs_valid=outputs_valid,
        )
        or _scalar_aggregate_select_is_known_single_row(
            select,
            scope,
            outputs,
            outputs_valid=outputs_valid,
        )
        or _transparent_singleton_passthrough_is_known_single_row(
            select,
            exp,
            scope,
            outputs,
            outputs_valid=outputs_valid,
        )
    )


def _scope_is_exact_singleton_derived_aggregate_join(
    select: Any,
    exp: Any,
    scope: _Scope,
) -> bool:
    """Prove a direct CROSS JOIN preserves one scalar aggregate row exactly."""

    if (
        len(scope.sources) < 2
        or _has_singleton_proof_barrier(select, allow_where=False)
    ):
        return False
    source_nodes = _direct_source_nodes(select)
    joins = list(select.args.get("joins") or ())
    if len(source_nodes) != len(scope.sources) or len(joins) != len(source_nodes) - 1:
        return False
    source_aliases = [_source_alias(node) for node in source_nodes]
    if (
        any(alias is None for alias in source_aliases)
        or len(set(source_aliases)) != len(source_aliases)
        or set(source_aliases) != set(scope.sources)
    ):
        return False
    if any(
        source.is_base
        or not source.relation.valid
        or not source.relation.known_single_row
        for source in scope.sources.values()
    ):
        return False
    for join in joins:
        if _sqlite_identifier_key(str(join.args.get("kind", ""))) != "cross":
            return False
        if any(
            join.args.get(name) is not None
            for name in ("on", "using", "side", "method")
        ) or bool(join.args.get("natural")):
            return False

    projections = list(getattr(select, "expressions", ()) or ())
    if not projections:
        return False
    for expression in projections:
        raw = _unwrap_alias_or_paren(expression, exp)
        if not _is_expression_type(raw, exp, "Column") or _column_is_star(raw):
            return False
        source = _column_source(raw, exp, scope)
        lineage = _column_lineage(raw, exp, scope)
        if (
            source is None
            or source.is_base
            or not source.relation.valid
            or lineage is None
            or lineage.kind != "aggregate"
        ):
            return False
    return True


def _analyse_select(
    select: Any,
    exp: Any,
    audit: _LineageAudit,
    *,
    parent: Optional[_Scope],
    visible_ctes: dict[str, _Relation],
    blocked_ctes: frozenset[str],
    derived: bool,
    declared_names: Sequence[str] = (),
) -> tuple[_Relation, _Scope]:
    identity = id(select)
    cached = audit.relations.get(identity)
    cached_scope = audit.scopes.get(identity)
    if cached is not None and cached_scope is not None:
        return cached, cached_scope

    ctes, pending = _resolve_local_ctes(
        select,
        exp,
        audit,
        visible_ctes=visible_ctes,
        blocked_ctes=blocked_ctes,
    )

    scope = _Scope({} , parent=parent)
    source_nodes = _direct_source_nodes(select)
    for source_node in source_nodes:
        alias, source = _source_from_node(
            source_node,
            exp,
            audit,
            visible_ctes=ctes,
            blocked_ctes=frozenset(pending),
            allow_unnamed_derived=len(source_nodes) == 1,
        )
        if alias is None:
            continue
        if alias in scope.sources:
            audit.reject(f"ambiguous source alias: {alias}")
            continue
        scope.sources[alias] = source

    scope.aggregate_safe_source_aliases = _aggregate_safe_source_aliases(select, exp, scope)
    scope.aggregate_input_safe = bool(scope.aggregate_safe_source_aliases)
    scope.aliases = _explicit_aliases(select, exp, scope, audit)
    outputs, outputs_valid = _relation_outputs(
        select,
        exp,
        scope,
        audit,
        declared_names=declared_names,
        derived=derived,
    )
    scope.forwards_derived_aggregate = _scope_forwards_derived_aggregate(
        select,
        exp,
        scope,
    )
    scope.singleton_derived_aggregate_join_safe = (
        _scope_is_exact_singleton_derived_aggregate_join(select, exp, scope)
    )
    projection_signatures = [lineage.signature or "<unsupported expression>" for lineage in outputs.values()]
    grouping = tuple(
        _group_signatures(
            select,
            exp,
            scope,
            scope.aliases,
            projection_signatures,
            # Derived pass-through layers must retain grouped-input semantics so
            # a later outer query cannot launder a grouped aggregate as scalar.
            include_inherited=derived,
        )
    )
    has_base_source = _scope_has_allowed_base_source(scope)
    relation = _Relation(
        outputs=outputs,
        grouping=grouping,
        valid=outputs_valid,
        has_base_source=has_base_source,
        aggregate_input_safe=(
            _derived_relation_is_aggregate_input_safe(select, exp, scope, outputs)
            if derived
            else _scope_has_aggregate_safe_inputs(scope)
        ),
        known_single_row=(
            _derived_relation_is_known_single_row(
                select,
                exp,
                scope,
                outputs,
                outputs_valid=outputs_valid,
            )
            if derived
            else False
        ),
    )
    audit.relations[identity] = relation
    audit.scopes[identity] = scope
    audit.relation_scopes[id(relation)] = scope

    _audit_scope_columns(select, exp, scope, audit)
    _analyse_nested_query_roots(
        select,
        exp,
        audit,
        parent=scope,
        visible_ctes=ctes,
        blocked_ctes=pending,
    )
    return relation, scope


def _analyse_set_operation(
    query: Any,
    exp: Any,
    audit: _LineageAudit,
    *,
    parent: Optional[_Scope],
    visible_ctes: dict[str, _Relation],
    blocked_ctes: frozenset[str],
) -> None:
    """Analyse one compound query root under its own lexical CTE environment."""

    set_operation_type = getattr(exp, "SetOperation", None)
    select_type = getattr(exp, "Select", None)
    if (
        set_operation_type is None
        or select_type is None
        or not isinstance(query, set_operation_type)
    ):
        return
    identity = id(query)
    if identity in audit.analysed_set_operations:
        return
    audit.analysed_set_operations.add(identity)

    ctes, pending = _resolve_local_ctes(
        query,
        exp,
        audit,
        visible_ctes=visible_ctes,
        blocked_ctes=blocked_ctes,
    )
    wrappers = tuple(
        kind
        for name in ("Subquery", "Paren")
        if (kind := getattr(exp, name, None)) is not None
    )
    values_type = getattr(exp, "Values", None)

    def analyse_branch(node: Any) -> None:
        while wrappers and isinstance(node, wrappers):
            node = node.this
        if isinstance(node, set_operation_type):
            _analyse_set_operation(
                node,
                exp,
                audit,
                parent=parent,
                visible_ctes=ctes,
                blocked_ctes=pending,
            )
        elif isinstance(node, select_type):
            _analyse_select(
                node,
                exp,
                audit,
                parent=parent,
                visible_ctes=ctes,
                blocked_ctes=pending,
                derived=True,
            )
        elif values_type is not None and isinstance(node, values_type):
            # Values does not supply a source scope, but scalar subqueries in
            # its rows still need the same lexical source/CTE audit as every
            # other nested query root.
            _analyse_query_roots_in_value(
                node,
                exp,
                audit,
                parent=parent,
                visible_ctes=ctes,
                blocked_ctes=pending,
            )
        else:
            # Do not silently treat an unknown compound branch as a proven
            # relation.  Literal VALUES branches retain their existing paging
            # provenance handling and do not supply a source scope.
            audit.reject("unsupported compound query branch")

    left = query.args.get("this")
    right = query.args.get("expression")
    if left is None or right is None:
        audit.reject("unsupported compound query branch")
        return
    analyse_branch(left)
    analyse_branch(right)

    # A compound root can own ORDER BY, LIMIT, OFFSET, and other expression
    # clauses.  Those expressions are outside either branch but can contain
    # scalar Select/SetOperation roots, so audit them under this root's local
    # CTE environment as well.
    for name, value in query.args.items():
        if name not in {"with_", "this", "expression"}:
            _analyse_query_roots_in_value(
                value,
                exp,
                audit,
                parent=parent,
                visible_ctes=ctes,
                blocked_ctes=pending,
            )


def _analyse_query_roots_in_value(
    value: Any,
    exp: Any,
    audit: _LineageAudit,
    *,
    parent: Optional[_Scope],
    visible_ctes: dict[str, _Relation],
    blocked_ctes: frozenset[str],
) -> None:
    """Dispatch nearest nested query roots under one lexical environment."""

    select_type = getattr(exp, "Select", None)
    set_operation_type = getattr(exp, "SetOperation", None)
    values_type = getattr(exp, "Values", None)

    if isinstance(value, (list, tuple)):
        for item in value:
            _analyse_query_roots_in_value(
                item,
                exp,
                audit,
                parent=parent,
                visible_ctes=visible_ctes,
                blocked_ctes=blocked_ctes,
            )
        return
    if not hasattr(value, "args"):
        return
    if values_type is not None and isinstance(value, values_type):
        # SQLite permits a WITH clause on a VALUES query root.  Current
        # SQLGlot versions reject that spelling, but if parser support changes
        # it must receive the same local CTE binding and fail-closed policy as
        # SELECT and compound query roots.
        ctes, pending = _resolve_local_ctes(
            value,
            exp,
            audit,
            visible_ctes=visible_ctes,
            blocked_ctes=blocked_ctes,
        )
        for name, child in value.args.items():
            if name != "with_":
                _analyse_query_roots_in_value(
                    child,
                    exp,
                    audit,
                    parent=parent,
                    visible_ctes=ctes,
                    blocked_ctes=pending,
                )
        return
    if select_type is not None and isinstance(value, select_type):
        wrappers = tuple(
            kind
            for name in ("Subquery", "Paren")
            if (kind := getattr(exp, name, None)) is not None
        )
        context = getattr(value, "parent", None)
        while wrappers and isinstance(context, wrappers):
            context = getattr(context, "parent", None)
        exposed_contexts = tuple(
            kind
            for name in ("From", "Join", "Lateral", "CTE")
            if (kind := getattr(exp, name, None)) is not None
        )
        _analyse_select(
            value,
            exp,
            audit,
            parent=parent,
            visible_ctes=visible_ctes,
            blocked_ctes=blocked_ctes,
            # Only a relation exposed through FROM/JOIN/CTE needs stable
            # outward output names.  Scalar, EXISTS, and IN subqueries are
            # consumed by position and may legally project an unnamed
            # aggregate or literal.
            derived=bool(exposed_contexts and isinstance(context, exposed_contexts)),
        )
        return
    if set_operation_type is not None and isinstance(value, set_operation_type):
        _analyse_set_operation(
            value,
            exp,
            audit,
            parent=parent,
            visible_ctes=visible_ctes,
            blocked_ctes=blocked_ctes,
        )
        return
    for child in getattr(value, "args", {}).values():
        _analyse_query_roots_in_value(
            child,
            exp,
            audit,
            parent=parent,
            visible_ctes=visible_ctes,
            blocked_ctes=blocked_ctes,
        )


def _analyse_nested_query_roots(
    select: Any,
    exp: Any,
    audit: _LineageAudit,
    *,
    parent: _Scope,
    visible_ctes: dict[str, _Relation],
    blocked_ctes: frozenset[str],
) -> None:
    """Analyse nested roots without revisiting this Select's local CTE bodies."""

    # Local CTEs are resolved before scope construction.  Revisiting their
    # bodies here could cache a SELECT with the wrong lexical environment.
    for name, value in select.args.items():
        if name != "with_":
            _analyse_query_roots_in_value(
                value,
                exp,
                audit,
                parent=parent,
                visible_ctes=visible_ctes,
                blocked_ctes=blocked_ctes,
            )


def _expected_ordering(ordering: Sequence[str]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for item in ordering:
        text = str(item).strip()
        null_match = re.search(r"(?is)\s+NULLS\s+(FIRST|LAST)\s*$", text)
        null_placement: Optional[str] = None
        if null_match:
            null_placement = f"NULLS {null_match.group(1).upper()}"
            text = text[:null_match.start()].rstrip()
        match = re.match(r"(?is)^(.*?)(?:\s+|:)(ASC|DESC)\s*$", text)
        if match:
            key = match.group(1).strip()
            direction = match.group(2).upper()
        else:
            key = text
            direction = "ASC"
        if null_placement is None:
            # SQLite's default null placement depends on the effective sort
            # direction, so an omitted modifier is still semantic.
            null_placement = "NULLS LAST" if direction == "DESC" else "NULLS FIRST"
        aggregate = _canonical_aggregate_text(key)
        result.append(
            (aggregate or _canonical_column_value(key), direction, null_placement)
        )
    return result


def _sqlite_current_time_literal(node: Any) -> bool:
    """Whether one designated SQLite time value is a recognized current-time token."""

    return (
        type(node).__name__.casefold() == "literal"
        and bool(getattr(node, "is_string", False))
        and str(getattr(node, "this", "")).casefold()
        in {"now", "subsec", "subsecond"}
    )


def _sqlite_time_value_is_current(node: Any) -> bool:
    """Follow only SQLGlot's transparent adapters around a time-value argument."""

    if node is None:
        return False
    if _sqlite_current_time_literal(node):
        return True
    class_name = type(node).__name__.casefold()
    if class_name in {"currentdate", "currenttime", "currenttimestamp"}:
        return True
    if class_name in {"cast", "paren", "tsordstotimestamp"}:
        return _sqlite_time_value_is_current(node.args.get("this"))
    return False


def _is_sqlite_current_time_expression(node: Any) -> bool:
    """Whether a parser-recognized SQLite date/time call depends on current time.

    Inspect only a function's designated time-value argument.  In particular,
    ``strftime('now', visit_date)`` has a deterministic source time value; the
    format literal must not be mistaken for the SQLite ``'now'`` token.
    """

    class_name = type(node).__name__.casefold()
    if class_name == "date":
        time_value = node.args.get("this")
        return time_value is None or _sqlite_time_value_is_current(time_value)
    if class_name in {"strftime", "datetostr", "timetostr"}:
        time_value = node.args.get("this")
        return time_value is None or _sqlite_time_value_is_current(time_value)

    # SQLite keeps several date/time functions as Anonymous nodes.  The first
    # argument is their time value and defaults to ``'now'`` when absent.
    if class_name != "anonymous":
        return False
    function_name = _sqlite_identifier_key(str(getattr(node, "name", "")))
    arguments = tuple(getattr(node, "expressions", ()) or ())
    if function_name in {"time", "datetime", "julianday", "unixepoch"}:
        return not arguments or _sqlite_time_value_is_current(arguments[0])
    if function_name == "timediff":
        # timediff() has two time-value arguments and no implicit-now form.
        return any(_sqlite_time_value_is_current(argument) for argument in arguments[:2])
    return False


def _contains_nondeterministic(node: Any) -> bool:
    """Reject recognized volatile ordering functions, including anonymous SQL functions."""

    return any(_is_nondeterministic_expression(child) for child in node.walk())


def _is_nondeterministic_expression(node: Any) -> bool:
    volatile = {
        "rand",
        "random",
        "randomblob",
        "current_date",
        "current_time",
        "current_timestamp",
    }
    class_name = type(node).__name__.casefold()
    if any(
        token in class_name
        for token in ("rand", "random", "currentdate", "currenttime", "currenttimestamp")
    ):
        return True
    if _is_sqlite_current_time_expression(node):
        return True
    # Unknown SQLite functions parse as Anonymous.  Do not inspect a
    # Column.name here: a deterministic source column can share a volatile
    # function's spelling.
    function_name = _sqlite_identifier_key(str(getattr(node, "name", "")))
    return class_name == "anonymous" and function_name in volatile


def _contains_local_nondeterministic(node: Any, exp: Any) -> bool:
    """Ignore volatility owned by a nested query with its own semantic audit."""

    boundaries = tuple(
        kind
        for name in ("Select", "SetOperation", "Subquery", "CTE")
        if (kind := getattr(exp, name, None)) is not None
    )
    for child in node.walk():
        if not _is_nondeterministic_expression(child):
            continue
        current = getattr(child, "parent", None)
        while current is not None and current is not node:
            if boundaries and isinstance(current, boundaries):
                break
            current = getattr(current, "parent", None)
        if current is node:
            return True
    return False


def _order_node_is_nondeterministic(
    node: Any,
    exp: Any,
    scope: Optional[_Scope],
    aliases: dict[str, _Lineage],
) -> bool:
    """Detect direct volatility and volatility propagated through an ORDER BY column."""

    if _contains_nondeterministic(node):
        return True
    column_type = getattr(exp, "Column", None)
    if column_type is None:
        return False
    for child in node.walk():
        if not isinstance(child, column_type) or _column_is_star(child):
            continue
        lineage = _expression_lineage(
            child,
            exp,
            scope=scope,
            aliases=aliases,
            resolve_alias=True,
        )
        if lineage is not None and lineage.kind == "volatile":
            return True
    return False


def _active_contract(contract: SemanticContract) -> bool:
    return bool(
        contract.projections
        or contract.aggregates
        or contract.output_slots
        or contract.grouping
        or contract.ordering
        or contract.limit is not None
        or contract.set_semantics is not None
    )


def _paging_query_nodes(tree: Any, exp: Any) -> list[Any]:
    """Return query roots that can own SQLite ORDER BY, LIMIT, or OFFSET."""

    query_types = tuple(
        kind
        for name in ("Select", "SetOperation", "Values")
        if (kind := getattr(exp, name, None)) is not None
    )
    if not query_types:
        return []
    return [node for node in tree.walk() if isinstance(node, query_types)]


def _select_has_values_source(select: Any, exp: Any) -> bool:
    """Whether SQLGlot normalized a VALUES branch into a SELECT source."""

    values_type = getattr(exp, "Values", None)
    return values_type is not None and any(
        isinstance(source, values_type) for source in _direct_source_nodes(select)
    )


def _set_operation_select_leaves(query: Any, exp: Any) -> tuple[list[Any], bool]:
    """Return compound SELECT leaves and whether any output branch is unresolved."""

    set_operation_type = getattr(exp, "SetOperation", None)
    select_type = getattr(exp, "Select", None)
    if set_operation_type is None or select_type is None or not isinstance(query, set_operation_type):
        return [], True
    wrappers = tuple(
        kind
        for name in ("Subquery", "Paren")
        if (kind := getattr(exp, name, None)) is not None
    )
    leaves: list[Any] = []
    unresolved = False

    def visit(node: Any) -> None:
        nonlocal unresolved
        while wrappers and isinstance(node, wrappers):
            node = node.this
        if isinstance(node, set_operation_type):
            left = node.args.get("this")
            right = node.args.get("expression")
            if left is None or right is None:
                unresolved = True
                return
            visit(left)
            visit(right)
        elif isinstance(node, select_type):
            if _select_has_values_source(node, exp):
                unresolved = True
            else:
                leaves.append(node)
        else:
            # VALUES and other compound branches have no Select scope whose
            # output lineage can prove an ordered paging path is deterministic.
            unresolved = True

    visit(query)
    return leaves, unresolved


def _query_order_contexts(
    query: Any,
    exp: Any,
    audit: _LineageAudit,
) -> tuple[list[tuple[Any, _Scope]], bool]:
    """Return output scopes and whether any ordered output is unresolvable."""

    scope = audit.scopes.get(id(query))
    if scope is not None:
        return [(query, scope)], False
    set_operation_type = getattr(exp, "SetOperation", None)
    if set_operation_type is None or not isinstance(query, set_operation_type):
        return [], True
    leaves, unresolved = _set_operation_select_leaves(query, exp)
    contexts: list[tuple[Any, _Scope]] = []
    for leaf in leaves:
        scope = audit.scopes.get(id(leaf))
        if scope is None:
            unresolved = True
        else:
            contexts.append((leaf, scope))
    return contexts, unresolved or not contexts


def _compound_order_output_index(
    node: Any,
    exp: Any,
    contexts: Sequence[tuple[Any, _Scope]],
) -> Optional[int]:
    """Resolve a compound ORDER BY ordinal or output alias."""

    ordinal = _paging_numeric_literal_ordinal(node, exp)
    if ordinal is None:
        ordinal = _ambiguous_hex_ordinal(node, exp)
    if ordinal is not None:
        return ordinal
    raw = _unwrap_sqlite_ordinal_reference(node, exp, allow_collation=True)
    if (
        not _is_expression_type(raw, exp, "Column")
        or _column_is_star(raw)
        or str(getattr(raw, "table", ""))
        or not contexts
    ):
        return None
    name = _sqlite_identifier_key(str(getattr(raw, "name", "")))
    return _compound_output_alias_index(name, exp, [query for query, _ in contexts]) if name else None


def _query_window_order_is_nondeterministic(
    query: Any,
    exp: Any,
    audit: _LineageAudit,
) -> bool:
    """Whether this SELECT owns a window order with volatile provenance."""

    select_type = getattr(exp, "Select", None)
    window_type = getattr(exp, "Window", None)
    if (
        select_type is None
        or window_type is None
        or not isinstance(query, select_type)
    ):
        return False
    scope = audit.scopes.get(id(query))
    if scope is None:
        return False
    ordered_type = getattr(exp, "Ordered", None)
    for window in query.find_all(window_type):
        # Nested SELECTs have their own scope and are inspected on their own
        # pass through _paging_query_nodes().
        if not _column_belongs_to_scope(window, query, exp):
            continue
        order = window.args.get("order")
        if order is None:
            continue
        if _contains_nondeterministic(order):
            return True
        for item in getattr(order, "expressions", ()):
            node = item.this if ordered_type is not None and isinstance(item, ordered_type) else item
            if _order_node_is_nondeterministic(node, exp, scope, scope.aliases):
                return True
    return False


def _query_order_risk(
    query: Any,
    exp: Any,
    audit: _LineageAudit,
) -> tuple[bool, bool]:
    """Return recognized volatility and unresolved provenance for ORDER BY."""

    order = query.args.get("order") or query.args.get("order_by")
    if order is None:
        return False, False
    if _contains_nondeterministic(order):
        return True, False

    contexts, unresolved = _query_order_contexts(query, exp, audit)
    if not contexts:
        return False, True
    set_operation_type = getattr(exp, "SetOperation", None)
    compound = set_operation_type is not None and isinstance(query, set_operation_type)
    ordered_type = getattr(exp, "Ordered", None)
    for item in getattr(order, "expressions", ()):
        node = item.this if ordered_type is not None and isinstance(item, ordered_type) else item
        if compound:
            ordinal = _compound_order_output_index(node, exp, contexts)
            if ordinal is None:
                # A compound ORDER BY may only reference an output position,
                # output alias, or exact output expression.  If it does not
                # resolve as one of those, preserve raw provenance checks but
                # fail closed rather than treating a branch-local source name
                # as the compound output alias.
                for _, scope in contexts:
                    if _order_node_is_nondeterministic(node, exp, scope, scope.aliases):
                        return True, unresolved
                unresolved = True
                continue
        else:
            for _, scope in contexts:
                if _order_node_is_nondeterministic(node, exp, scope, scope.aliases):
                    return True, unresolved
            ordinal = _paging_numeric_literal_ordinal(node, exp)
            if ordinal is None:
                ordinal = _ambiguous_hex_ordinal(node, exp)
            if ordinal is None:
                continue
        for output_query, scope in contexts:
            expressions = list(getattr(output_query, "expressions", ()) or ())
            if not 1 <= ordinal <= len(expressions):
                unresolved = True
                continue
            lineage = _expression_lineage(
                expressions[ordinal - 1],
                exp,
                scope=scope,
                aliases=scope.aliases,
                resolve_alias=False,
            )
            if lineage is not None and lineage.kind == "volatile":
                return True, unresolved
            if compound and lineage is None:
                unresolved = True
    return False, unresolved


def _query_is_scalar_subquery(query: Any, exp: Any) -> bool:
    """Whether a query root is consumed as a SQLite scalar subquery.

    A SELECT nested beneath a FROM/JOIN source or an IN/EXISTS predicate is
    relation-valued.  Every other Subquery wrapper is scalar-valued in SQLite,
    where its ORDER BY selects the first returned row even without LIMIT.
    """

    subquery_type = getattr(exp, "Subquery", None)
    parent = getattr(query, "parent", None)
    if subquery_type is None or not isinstance(parent, subquery_type):
        return False
    context = getattr(parent, "parent", None)
    if context is None:
        return False
    relation_or_set_contexts = tuple(
        kind
        for name in ("From", "Join", "Lateral", "In", "Exists", "Any", "All")
        if (kind := getattr(exp, name, None)) is not None
    )
    return not (
        relation_or_set_contexts and isinstance(context, relation_or_set_contexts)
    )


def _semantic_paging_reasons(
    tree: Any,
    outer: Any,
    exp: Any,
    audit: _LineageAudit,
) -> list[str]:
    """Reject paging and volatile ordering that can alter contract membership."""

    reasons: list[str] = []

    def reject(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    outer_has_limit = outer.args.get("limit") is not None
    for query in _paging_query_nodes(tree, exp):
        if query.args.get("offset") is not None:
            reject("offset clause is not permitted for an active semantic contract")
        if _query_window_order_is_nondeterministic(query, exp, audit):
            # A ROW_NUMBER (including a named WINDOW definition) can turn its
            # order into row membership through an outer rank predicate, even
            # when the query itself has no ORDER BY clause.
            reject("nondeterministic window ordering is not permitted for an active semantic contract")

        nested = query is not outer
        has_limit = query.args.get("limit") is not None
        scalar_subquery = _query_is_scalar_subquery(query, exp)
        volatile_order, unresolved_order = _query_order_risk(query, exp, audit)
        if scalar_subquery:
            # SQLite consumes the first row of a scalar subquery.  Its ORDER BY
            # therefore controls a value or membership decision even when it
            # has no explicit LIMIT and the outer query is not Top-K.
            if volatile_order:
                reject("nondeterministic ordering controls a scalar subquery")
            if unresolved_order:
                reject("ordered scalar-subquery provenance cannot be established")
        if nested and has_limit:
            reject("nested limit clause is not permitted for an active semantic contract")
            if volatile_order:
                reject("nondeterministic ordering controls a nested limit")
            if unresolved_order:
                reject("ordered paging provenance cannot be established for a nested limit")
        if nested and outer_has_limit:
            # SQLite can retain an inner relation's ordering when the outer
            # query consumes its prefix.  An AST-only audit cannot prove that
            # a nested volatile order is semantically dead, so active Top-K
            # contracts reject it rather than risk unstable membership.
            if volatile_order:
                reject("nondeterministic ordering controls an outer limit")
            if unresolved_order:
                reject("ordered paging provenance cannot be established for an outer limit")
    return reasons


def _scope_forwards_derived_aggregate(select: Any, exp: Any, scope: _Scope) -> bool:
    """Whether a SELECT forwards a scalar aggregate from a derived source."""

    aggregate_kinds = {"aggregate", "window_aggregate"}
    for expression in getattr(select, "expressions", ()):
        raw = _unwrap_alias_or_paren(expression, exp)
        if not _is_expression_type(raw, exp, "Column"):
            continue
        if _column_is_star(raw):
            table = _sqlite_identifier_key(str(getattr(raw, "table", "")))
            source = _source_for_qualified_name(scope, table) if table else None
            if (
                source is not None
                and not source.is_base
                and source.relation.valid
                and not source.relation.grouping
                and any(
                    lineage.kind in aggregate_kinds
                    for lineage in source.relation.outputs.values()
                )
            ):
                return True
            continue
        source = _column_source(raw, exp, scope)
        lineage = _column_lineage(raw, exp, scope)
        if (
            source is not None
            and not source.is_base
            and source.relation.valid
            and not source.relation.grouping
            and lineage is not None
            and lineage.kind in aggregate_kinds
        ):
            return True
    return False


def _reachable_scope_ids(audit: _LineageAudit, root_scope: _Scope) -> frozenset[int]:
    """Follow evaluated source and nested-query edges from the outer result.

    Local CTE bodies are audited eagerly so malformed or forbidden definitions
    remain hard failures.  Their aggregate/grouping provenance only affects the
    result when an evaluated scope resolves that CTE as a source, however.
    """

    children: dict[int, list[_Scope]] = {}
    for candidate in audit.scopes.values():
        if candidate.parent is not None:
            children.setdefault(id(candidate.parent), []).append(candidate)

    reachable: set[int] = set()
    pending = [root_scope]
    while pending:
        scope = pending.pop()
        identity = id(scope)
        if identity in reachable:
            continue
        reachable.add(identity)
        pending.extend(children.get(identity, ()))
        for source in scope.sources.values():
            source_scope = audit.relation_scopes.get(id(source.relation))
            if source_scope is not None:
                pending.append(source_scope)
    return frozenset(reachable)


def _reachable_row_set_mode_reasons(
    tree: Any,
    exp: Any,
    audit: _LineageAudit,
    reachable_scope_ids: frozenset[int],
    row_set_mode: RowSetMode,
    *,
    outer_select: Any = None,
) -> list[str]:
    """Require SELECT-level deduplication to match the bound row-set mode."""

    select_type = getattr(exp, "Select", None)
    if select_type is None:
        return ["SELECT DISTINCT safety cannot be established"]
    reachable_distinct: list[Any] = []
    for node in tree.walk():
        if not isinstance(node, select_type):
            continue
        scope = audit.scopes.get(id(node))
        if (
            scope is not None
            and id(scope) in reachable_scope_ids
            and node.args.get("distinct") is not None
        ):
            reachable_distinct.append(node)
    if row_set_mode is RowSetMode.PRESERVE and reachable_distinct:
        return ["unrequested SELECT DISTINCT changes result cardinality"]
    if (
        row_set_mode is RowSetMode.DISTINCT
        and outer_select is not None
        and outer_select.args.get("distinct") is None
        and not _outer_select_is_inherently_unique_patient_id(outer_select, exp)
    ):
        return ["requested SELECT DISTINCT is missing"]
    return []


def _outer_select_is_inherently_unique_patient_id(
    outer_select: Any,
    exp: Any,
) -> bool:
    """Prove one base-patient row per projected patient identifier.

    Correlated EXISTS clauses filter an outer ``patients`` row but cannot
    duplicate it. Requiring a syntactic DISTINCT for that shape would reject a
    query whose row set is already unique by the project's declared
    ``patients.patient_id`` key. The proof deliberately excludes joins,
    derived sources, additional projections, and non-key patient attributes.
    """

    if outer_select.args.get("joins"):
        return False
    from_clause = outer_select.args.get("from_")
    source = getattr(from_clause, "this", None)
    if not _is_expression_type(source, exp, "Table"):
        return False
    if _sqlite_identifier_key(str(getattr(source, "name", ""))) != "patients":
        return False
    expressions = list(getattr(outer_select, "expressions", ()) or ())
    if len(expressions) != 1:
        return False
    projection = _unwrap_alias_or_paren(expressions[0], exp)
    if (
        not _is_expression_type(projection, exp, "Column")
        or _column_is_star(projection)
        or _sqlite_identifier_key(str(getattr(projection, "name", "")))
        != "patient_id"
    ):
        return False
    qualifier = _sqlite_identifier_key(str(getattr(projection, "table", "")))
    source_alias = _sqlite_identifier_key(
        str(getattr(source, "alias_or_name", ""))
    )
    return not qualifier or qualifier == source_alias


def _reachable_raw_cross_join_reasons(
    tree: Any,
    exp: Any,
    audit: _LineageAudit,
    reachable_scope_ids: frozenset[int],
) -> list[str]:
    """Reject result-reachable cartesian joins without a singleton proof."""

    select_type = getattr(exp, "Select", None)
    join_type = getattr(exp, "Join", None)
    if select_type is None or join_type is None:
        return ["join cardinality safety cannot be established"]
    for node in tree.walk():
        if not isinstance(node, select_type):
            continue
        scope = audit.scopes.get(id(node))
        if (
            scope is None
            or id(scope) not in reachable_scope_ids
            or scope.singleton_derived_aggregate_join_safe
        ):
            continue
        for join in node.args.get("joins") or ():
            if not isinstance(join, join_type):
                continue
            kind = _sqlite_identifier_key(str(join.args.get("kind", "")))
            has_predicate = (
                join.args.get("on") is not None
                or join.args.get("using") is not None
                or bool(join.args.get("natural"))
            )
            if kind == "cross" or not has_predicate:
                return [
                    "raw-row CROSS JOIN cardinality has no supported singleton proof"
                ]
    return []


def _bucket_projection_reasons(
    select: Any,
    exp: Any,
    scope: _Scope,
    _actual_grouping: Sequence[str],
) -> list[str]:
    """Require a CASE-derived age/cost dimension to be consumed as a bucket."""

    expressions = list(getattr(select, "expressions", ()) or ())
    group = select.args.get("group") or select.args.get("group_by")
    group_expressions = list(getattr(group, "expressions", ()) or ())
    for index, expression in enumerate(expressions, start=1):
        lineage = _expression_lineage(expression, exp, scope=scope)
        if lineage is None or lineage.kind not in {"age_bucket", "cost_bucket"}:
            continue
        dimension = "age" if lineage.kind == "age_bucket" else "cost"
        raw = _unwrap_alias_or_paren(expression, exp)
        if _is_expression_type(raw, exp, "Case"):
            matched = False
            for group_expression in group_expressions:
                group_raw = _unwrap_sqlite_ordinal_reference(
                    group_expression,
                    exp,
                    allow_collation=False,
                )
                if (
                    _is_expression_type(group_raw, exp, "Case")
                    and group_raw == raw
                ):
                    matched = True
                    break
                if _is_direct_group_alias(group_raw, select, exp, scope):
                    alias = _sqlite_identifier_key(
                        str(getattr(group_raw, "name", ""))
                    )
                    if alias == _output_name(expression, exp):
                        matched = True
                        break
                if _numeric_literal_ordinal(group_expression, exp) == index:
                    matched = True
                    break
            if not matched:
                return [
                    f"CASE-derived {dimension} bucket cannot impersonate "
                    f"an ungrouped raw {dimension}"
                ]
        else:
            matched = False
            for group_expression in group_expressions:
                if _numeric_literal_ordinal(group_expression, exp) == index:
                    matched = True
                    break
                group_raw = _unwrap_sqlite_ordinal_reference(
                    group_expression,
                    exp,
                    allow_collation=False,
                )
                group_lineage = _expression_lineage(
                    group_raw,
                    exp,
                    scope=scope,
                    aliases=scope.aliases,
                    resolve_alias=_is_direct_group_alias(
                        group_raw,
                        select,
                        exp,
                        scope,
                    ),
                    allow_unaliased_case=True,
                )
                if (
                    group_lineage is not None
                    and group_lineage.signature == lineage.signature
                    and group_lineage.kind == lineage.kind
                ):
                    matched = True
                    break
            if not matched:
                return [
                    f"derived {dimension} bucket cannot impersonate "
                    f"an ungrouped raw {dimension}"
                ]
    return []


def _derived_aggregate_source_join_reasons(
    audit: _LineageAudit,
    reachable_scope_ids: frozenset[int],
) -> list[str]:
    """Reject unsafe, result-reachable forwarding of derived aggregates."""

    if any(
        id(scope) in reachable_scope_ids
        and len(scope.sources) > 1
        and scope.forwards_derived_aggregate
        and not scope.singleton_derived_aggregate_join_safe
        for scope in audit.scopes.values()
    ):
        return [
            "aggregate provenance cannot be preserved through a joined "
            "derived aggregate source"
        ]
    return []


def _grouped_derived_source_join_reasons(
    audit: _LineageAudit,
    reachable_scope_ids: frozenset[int],
) -> list[str]:
    """Reject reachable joins that can hide grouped derived semantics."""

    for scope in audit.scopes.values():
        if id(scope) not in reachable_scope_ids or len(scope.sources) <= 1:
            continue
        if any(
            not source.is_base
            and source.relation.valid
            and source.relation.grouping
            for source in scope.sources.values()
        ):
            return [
                "grouping provenance cannot be preserved through a joined "
                "grouped derived source"
            ]
    return []


def _compound_set_semantics(tree: Any, exp: Any) -> Optional[str]:
    """Return one homogeneous compound-set meaning, or None if mixed/unsafe."""

    set_operation_type = getattr(exp, "SetOperation", None)
    if set_operation_type is None or not isinstance(tree, set_operation_type):
        return None
    mapping = {
        "union": "any_of",
        "intersect": "all_of",
        "except": "difference",
    }
    meanings: set[str] = set()
    for node in tree.walk():
        if not isinstance(node, set_operation_type):
            continue
        meaning = mapping.get(type(node).__name__.casefold())
        if meaning is None or node.args.get("distinct") is False:
            return None
        meanings.add(meaning)
    return next(iter(meanings)) if len(meanings) == 1 else None


def _compound_order_signatures(
    query: Any,
    exp: Any,
    audit: _LineageAudit,
    select_list: Sequence[str],
) -> list[tuple[str, str, str]]:
    """Resolve compound ORDER BY keys through their shared output slots."""

    order = query.args.get("order") or query.args.get("order_by")
    if order is None:
        return []
    contexts, unresolved = _query_order_contexts(query, exp, audit)
    if unresolved or not contexts:
        return [("<unsupported expression>", "ASC", "<unsupported null placement>")]
    ordered_type = getattr(exp, "Ordered", None)
    values: list[tuple[str, str, str]] = []
    for item in getattr(order, "expressions", ()):
        is_ordered = ordered_type is not None and isinstance(item, ordered_type)
        node = item.this if is_ordered else item
        ordinal = _compound_order_output_index(node, exp, contexts)
        if ordinal is None or not 1 <= ordinal <= len(select_list):
            signature = "<unsupported expression>"
        else:
            signature = select_list[ordinal - 1]
        desc = bool(item.args.get("desc")) if is_ordered else False
        nulls_first = item.args.get("nulls_first") if is_ordered else None
        if nulls_first is True:
            null_placement = "NULLS FIRST"
        elif nulls_first is False:
            null_placement = "NULLS LAST"
        else:
            null_placement = "<unsupported null placement>"
        values.append((signature, "DESC" if desc else "ASC", null_placement))
    return values


def _literal_limit(query: Any, exp: Any) -> Optional[int]:
    limit = query.args.get("limit")
    if limit is None:
        return None
    expression = limit.args.get("expression")
    literal_type = getattr(exp, "Literal", None)
    if literal_type is None or not isinstance(expression, literal_type):
        return None
    raw = str(getattr(expression, "this", ""))
    if not raw.isdigit() or bool(getattr(expression, "is_string", False)):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _validate_compound_candidate(
    contract: SemanticContract,
    tree: Any,
    exp: Any,
    *,
    strict: bool,
) -> ValidationResult:
    """Validate a homogeneous UNION/INTERSECT/EXCEPT without bypassing slots."""

    audit = _LineageAudit()
    _analyse_set_operation(
        tree,
        exp,
        audit,
        parent=None,
        visible_ctes={},
        blocked_ctes=frozenset(),
    )
    leaves, unresolved = _set_operation_select_leaves(tree, exp)
    reasons = list(audit.reasons)
    if unresolved or len(leaves) < 2:
        reasons.append("compound query branches cannot be resolved")
    reachable_scope_ids: set[int] = set()
    for leaf in leaves:
        leaf_scope = audit.scopes.get(id(leaf))
        if leaf_scope is not None:
            reachable_scope_ids.update(_reachable_scope_ids(audit, leaf_scope))
    reasons.extend(
        _reachable_row_set_mode_reasons(
            tree,
            exp,
            audit,
            frozenset(reachable_scope_ids),
            contract.row_set_mode,
        )
    )
    reasons.extend(
        _reachable_raw_cross_join_reasons(
            tree,
            exp,
            audit,
            frozenset(reachable_scope_ids),
        )
    )

    actual_set_semantics = _compound_set_semantics(tree, exp)
    if actual_set_semantics is None:
        reasons.append("compound query must use one distinct set operator")
    elif contract.set_semantics is None:
        reasons.append("unexpected compound set semantics")
    elif actual_set_semantics != contract.set_semantics:
        reasons.append(
            "compound set semantic mismatch: "
            f"expected {contract.set_semantics}, got {actual_set_semantics}"
        )

    if audit.reasons or unresolved or len(leaves) < 2:
        return ValidationResult(False, tuple(dict.fromkeys(reasons)))

    if not _active_contract(contract):
        return ValidationResult(False, ("compound query has no active semantic contract",))

    reasons.extend(_semantic_paging_reasons(tree, tree, exp, audit))
    expected_projections = [_canonical_column_value(item) for item in contract.projections]
    expected_aggregates = [
        _canonical_aggregate_text(item) or str(item).strip().upper()
        for item in contract.aggregates
    ]
    expected_select_list = [
        _canonical_aggregate_text(item) or _canonical_column_value(item)
        for item in contract.ordered_outputs
    ]
    expected_grouping = [_canonical_column_value(item) for item in contract.grouping]
    has_output_contract = bool(
        contract.projections
        or contract.aggregates
        or contract.output_slots
    )
    canonical_branch_select_list: Optional[list[str]] = None

    for index, leaf in enumerate(leaves, start=1):
        scope = audit.scopes.get(id(leaf))
        relation = audit.relations.get(id(leaf))
        if scope is None or relation is None or not relation.has_base_source:
            reasons.append(f"compound branch {index} has no allowed base source")
            continue
        actual_projections, actual_aggregates, actual_select_list = _projection_signatures(
            leaf,
            exp,
            scope,
            scope.aliases,
        )
        actual_projections = [
            _canonical_column_value(item) for item in actual_projections
        ]
        actual_aggregates = [
            _canonical_aggregate_text(item) or str(item).strip().upper()
            for item in actual_aggregates
        ]
        actual_select_list = [
            _canonical_aggregate_text(item) or _canonical_column_value(item)
            for item in actual_select_list
        ]
        if canonical_branch_select_list is None:
            canonical_branch_select_list = actual_select_list
        elif actual_select_list != canonical_branch_select_list:
            reasons.append("compound branches expose different output semantics")
        if has_output_contract and actual_projections != expected_projections:
            reasons.append(
                f"compound branch {index} projection semantic mismatch: "
                f"expected {expected_projections}, got {actual_projections}"
            )
        if has_output_contract and actual_aggregates != expected_aggregates:
            reasons.append(
                f"compound branch {index} aggregate semantic mismatch: "
                f"expected {expected_aggregates}, got {actual_aggregates}"
            )
        if has_output_contract and actual_select_list != expected_select_list:
            reasons.append(
                f"compound branch {index} select-list order mismatch: "
                f"expected {expected_select_list}, got {actual_select_list}"
            )
        actual_grouping = _group_signatures(
            leaf,
            exp,
            scope,
            scope.aliases,
            actual_select_list,
            include_inherited=True,
        )
        if expected_grouping:
            if not actual_grouping and (strict or contract.status is ContractStatus.COMPLETE):
                reasons.append(f"compound branch {index} grouping clause missing")
            elif actual_grouping != expected_grouping:
                reasons.append(
                    f"compound branch {index} grouping semantic mismatch: "
                    f"expected {expected_grouping}, got {actual_grouping}"
                )
        elif actual_grouping and (strict or contract.status is ContractStatus.COMPLETE):
            reasons.append(
                f"compound branch {index} has unexpected grouping: {actual_grouping}"
            )
        reasons.extend(
            _bucket_projection_reasons(
                leaf,
                exp,
                scope,
                actual_grouping,
            )
        )

    expected_ordering = _expected_ordering(contract.ordering)
    if expected_ordering:
        actual_ordering = _compound_order_signatures(
            tree,
            exp,
            audit,
            canonical_branch_select_list or (),
        )
        if len(actual_ordering) < len(expected_ordering):
            reasons.append(
                "compound ordering clause missing or too short: "
                f"expected prefix {expected_ordering}, got {actual_ordering}"
            )
        else:
            for index, expected in enumerate(expected_ordering):
                if actual_ordering[index] != expected:
                    reasons.append(
                        "compound ordering semantic mismatch: "
                        f"expected {expected}, got {actual_ordering[index]}"
                    )

    actual_limit = _literal_limit(tree, exp)
    if contract.limit is None:
        if tree.args.get("limit") is not None:
            reasons.append("unexpected limit clause")
    elif actual_limit is None:
        if strict or contract.status is ContractStatus.COMPLETE:
            reasons.append("limit clause missing")
    elif actual_limit != contract.limit:
        reasons.append(f"limit mismatch: expected {contract.limit}, got {actual_limit}")

    return ValidationResult(not reasons, tuple(dict.fromkeys(reasons)))


def _simple_set_shape_matches(
    tree: Any,
    exp: Any,
    semantics: str,
) -> bool:
    """Require a structural lowering before qualifier-level proof runs."""

    if semantics == "any_of":
        return tree.find(exp.In) is not None or tree.find(exp.Or) is not None
    if semantics == "difference":
        return any(
            isinstance(node.this, (exp.Exists, exp.In))
            for node in tree.find_all(exp.Not)
        )
    if semantics == "all_of":
        conjunctive_exists = 0
        for exists in tree.find_all(exp.Exists):
            current = getattr(exists, "parent", None)
            safe = True
            while current is not None and not isinstance(current, exp.Where):
                if isinstance(current, (exp.Or, exp.Not)):
                    safe = False
                    break
                current = getattr(current, "parent", None)
            if safe:
                conjunctive_exists += 1
        return conjunctive_exists >= 2
    return False


def validate_candidate(
    contract: SemanticContract,
    sql: str,
    *,
    strict: bool = True,
) -> ValidationResult:
    """Validate one candidate's outer SELECT semantics without changing it."""

    try:
        exp, tree, select = _parse_candidate(sql)
    except RuntimeError as exc:
        return ValidationResult(False, (str(exc),))
    except PermissionError as exc:
        return ValidationResult(False, (str(exc),))
    except TypeError as exc:
        return ValidationResult(False, (str(exc),))
    except ValueError as exc:
        return ValidationResult(False, (str(exc),))

    set_operation_type = getattr(exp, "SetOperation", None)
    if set_operation_type is not None and isinstance(select, set_operation_type):
        return _validate_compound_candidate(
            contract,
            tree,
            exp,
            strict=strict,
        )

    audit = _LineageAudit()
    relation, scope = _analyse_select(
        select,
        exp,
        audit,
        parent=None,
        visible_ctes={},
        blocked_ctes=frozenset(),
        derived=False,
    )
    if audit.reasons:
        reasons = list(audit.reasons)
        if _active_contract(contract):
            for reason in _semantic_paging_reasons(tree, select, exp, audit):
                if reason not in reasons:
                    reasons.append(reason)
        return ValidationResult(False, tuple(reasons))
    if not relation.has_base_source:
        return ValidationResult(False, ("candidate has no allowed base source",))

    reachable_scope_ids = _reachable_scope_ids(audit, scope)
    structural_shape_reasons = [
        *_reachable_row_set_mode_reasons(
            tree,
            exp,
            audit,
            reachable_scope_ids,
            contract.row_set_mode,
            outer_select=select,
        ),
        *_reachable_raw_cross_join_reasons(
            tree,
            exp,
            audit,
            reachable_scope_ids,
        ),
    ]

    # NONE means that extraction found no reliable semantic slots.  It still
    # receives the structural read-only and provenance gates above, but no
    # guessed output semantics.
    if not _active_contract(contract):
        return ValidationResult(
            not structural_shape_reasons,
            tuple(structural_shape_reasons),
        )

    paging_reasons = _semantic_paging_reasons(tree, select, exp, audit)
    grouped_join_reasons = _grouped_derived_source_join_reasons(
        audit,
        reachable_scope_ids,
    )
    derived_aggregate_join_reasons = _derived_aggregate_source_join_reasons(
        audit,
        reachable_scope_ids,
    )
    aliases = scope.aliases
    actual_projections, actual_aggregates, actual_select_list = _projection_signatures(
        select,
        exp,
        scope,
        aliases,
    )
    expected_projections = [_canonical_column_value(item) for item in contract.projections]
    expected_aggregates = [
        _canonical_aggregate_text(item) or str(item).strip().upper()
        for item in contract.aggregates
    ]
    has_output_contract = bool(
        contract.projections
        or contract.aggregates
        or contract.output_slots
    )
    reasons = list(paging_reasons)
    reasons.extend(structural_shape_reasons)
    reasons.extend(grouped_join_reasons)
    reasons.extend(derived_aggregate_join_reasons)
    if (
        contract.set_semantics is not None
        and not _simple_set_shape_matches(tree, exp, contract.set_semantics)
    ):
        reasons.append(
            f"missing structural lowering for {contract.set_semantics} set semantics"
        )

    if has_output_contract and actual_projections != expected_projections:
        if len(actual_projections) != len(expected_projections):
            reasons.append(
                "projection count mismatch: "
                f"expected {len(expected_projections)}, got {len(actual_projections)}"
            )
        reasons.append(
            "projection semantic mismatch: "
            f"expected {expected_projections}, got {actual_projections}"
        )

    if has_output_contract and actual_aggregates != expected_aggregates:
        reasons.append(
            "aggregate function/measure/distinct mismatch: "
            f"expected {expected_aggregates}, got {actual_aggregates}"
        )

    expected_select_list = [
        _canonical_aggregate_text(item) or _canonical_column_value(item)
        for item in contract.ordered_outputs
    ]
    if has_output_contract and actual_select_list != expected_select_list:
        reasons.append(
            "select-list order mismatch: "
            f"expected {expected_select_list}, got {actual_select_list}"
        )

    actual_grouping = _group_signatures(
        select,
        exp,
        scope,
        aliases,
        actual_select_list,
        # A transparent pass-through of a grouped CTE still has grouping
        # semantics, including when the contract expects a scalar aggregate.
        include_inherited=True,
    )
    expected_grouping = [_canonical_column_value(item) for item in contract.grouping]
    if expected_grouping:
        if not actual_grouping:
            if strict or contract.status is ContractStatus.COMPLETE:
                reasons.append("grouping clause missing")
        elif actual_grouping != expected_grouping:
            reasons.append(
                "grouping semantic mismatch: "
                f"expected {expected_grouping}, got {actual_grouping}"
            )
    elif actual_grouping and (strict or contract.status is ContractStatus.COMPLETE):
        reasons.append(f"unexpected grouping clause: got {actual_grouping}")
    reasons.extend(
        _bucket_projection_reasons(
            select,
            exp,
            scope,
            actual_grouping,
        )
    )

    actual_ordering, order_nodes = _order_signatures(
        select,
        exp,
        scope,
        aliases,
        actual_select_list,
    )
    expected_ordering = _expected_ordering(contract.ordering)
    if expected_ordering:
        if len(actual_ordering) < len(expected_ordering):
            if strict or contract.status is ContractStatus.COMPLETE:
                reasons.append(
                    "ordering clause missing or too short: "
                    f"expected prefix {expected_ordering}, got {actual_ordering}"
                )
        else:
            for index, expected in enumerate(expected_ordering):
                if actual_ordering[index] != expected:
                    reasons.append(
                        "ordering key/direction/null-placement mismatch: "
                        f"expected {expected}, got {actual_ordering[index]}"
                    )

    has_volatile_outer_order = any(
        _order_node_is_nondeterministic(node, exp, scope, aliases)
        for node in order_nodes
    )
    outer_has_limit = select.args.get("limit") is not None
    if outer_has_limit:
        # Any outer limit, including one the contract did not request, makes
        # every ordering key a membership rule.  Do not let an ungrounded cap
        # hide a volatile sample behind a limit=None contract.
        if has_volatile_outer_order:
            reasons.append("nondeterministic ordering controls a top-k limit")
    elif expected_ordering and any(
        _order_node_is_nondeterministic(node, exp, scope, aliases)
        for node in order_nodes[len(expected_ordering):]
    ):
        reasons.append("nondeterministic ordering tie key")
    # Without an order requirement or a limit, a deterministic candidate order
    # is harmless and is intentionally not rejected.

    limit = select.args.get("limit")
    if contract.limit is None:
        if limit is not None:
            reasons.append("unexpected limit clause")
    else:
        actual_limit: Optional[int] = None
        if limit is not None:
            expression = limit.args.get("expression")
            literal_type = getattr(exp, "Literal", None)
            if literal_type is not None and isinstance(expression, literal_type):
                raw = str(getattr(expression, "this", ""))
                if raw.isdigit() and not bool(getattr(expression, "is_string", False)):
                    try:
                        actual_limit = int(raw)
                    except ValueError:
                        actual_limit = None
        if actual_limit is None:
            if strict or contract.status is ContractStatus.COMPLETE:
                reasons.append("limit clause missing")
        elif actual_limit != contract.limit:
            reasons.append(f"limit mismatch: expected {contract.limit}, got {actual_limit}")

    return ValidationResult(not reasons, tuple(reasons))


__all__ = [
    "ContractStatus",
    "RowSetMode",
    "SemanticContract",
    "ValidationResult",
    "extract_semantic_contract",
    "validate_candidate",
]
