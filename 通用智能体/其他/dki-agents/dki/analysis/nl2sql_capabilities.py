"""Deterministic semantic capability tags for NL2SQL questions.

The classifier is intentionally observational: it never executes SQL and it does
not depend on database contents.  Question text is always inspected; callers may
optionally provide SQL text or an already-parsed ``sqlglot`` expression to add
structural evidence.  The returned labels have a stable order so reports and
fixtures remain reproducible.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

CAPABILITY_ORDER: tuple[str, ...] = (
    "schema_linking",
    "grain",
    "joins_cardinality",
    "count_distinct",
    "aggregation_stage",
    "preservation",
    "correlated",
    "cte",
    "set_operation",
    "window",
    "ties",
    "date",
    "null",
    "output_contract",
)

_CAPABILITY_SET = frozenset(CAPABILITY_ORDER)
_CAPABILITY_ALIASES = {
    "schema linking": "schema_linking",
    "schema-linking": "schema_linking",
    "joins/cardinality": "joins_cardinality",
    "join/cardinality": "joins_cardinality",
    "joins cardinality": "joins_cardinality",
    "count distinct": "count_distinct",
    "count-distinct": "count_distinct",
    "aggregation stage": "aggregation_stage",
    "aggregation-stage": "aggregation_stage",
    "set": "set_operation",
    "set operation": "set_operation",
    "set-operation": "set_operation",
    "output contract": "output_contract",
    "output-contract": "output_contract",
    "nulls": "null",
    "dates": "date",
}

_SCHEMA_LINKING_RE = re.compile(
    r"(?:患者|病人|就诊|病历|疾病|诊断|科室|药物|处方|检查|费用|住院|"
    r"地区|城市|姓名|性别|年龄|图谱|实体|关系|三元组|置信度|观察次数|"
    r"提及次数|记录|数据|patient|visit|disease|diagnos|department|drug|"
    r"exam|cost|region|gender|age|entity|relation|triple|table|column|row)",
    re.IGNORECASE,
)
_GRAIN_RE = re.compile(
    r"(?:每\s*(?:位|个|种|类|月|年|日)|各\s*(?:位|个|种|类|月|年|日)?|"
    r"按.{0,18}?(?:统计|分组|汇总)|分别|粒度|维度|一行|每组|去重|不同|唯一|"
    r"per\s+\w+|for\s+each|group(?:ed)?\s+by|distinct)",
    re.IGNORECASE,
)
_JOIN_RE = re.compile(
    r"(?:连接|关联|匹配|对应|跨表|患者.{0,16}(?:就诊|病历|科室|疾病)|"
    r"(?:就诊|病历|科室|疾病).{0,16}(?:患者|姓名|性别|年龄|地区)|"
    r"头实体.{0,16}尾实体|join|cardinality|one[- ]to[- ]many|many[- ]to[- ]many)",
    re.IGNORECASE,
)
_COUNT_DISTINCT_RE = re.compile(
    r"(?:COUNT\s*\(\s*DISTINCT|去重(?:后)?(?:计数|统计|人数|数量)|"
    r"不同(?:的)?(?:患者|实体|疾病|科室|记录).{0,8}(?:数|数量|多少)|"
    r"唯一(?:的)?(?:患者|实体|记录).{0,8}(?:数|数量|多少)|"
    r"distinct\s+(?:count|patient|entity))",
    re.IGNORECASE,
)
_PATIENT_CARDINALITY_RE = re.compile(
    r"(?:患者数|患者人数|病人数|病人人数|多少位患者|多少名患者|unique patients?)",
    re.IGNORECASE,
)
_VISIT_CONTEXT_RE = re.compile(
    r"(?:就诊|病历|科室|疾病|visit|encounter|admission)", re.IGNORECASE
)
_AGGREGATION_RE = re.compile(
    r"(?:统计|汇总|总数|数量|次数|人数|条数|总和|合计|平均|均值|最高|最低|"
    r"最大|最小|占比|比例|百分比|累计|分组|HAVING|高于.{0,8}平均|"
    r"低于.{0,8}平均|count|sum|avg|average|min|max|aggregate|group\s+by|having)",
    re.IGNORECASE,
)
_PRESERVATION_RE = re.compile(
    r"(?:包括|包含)(?:没有|无).{0,16}(?:记录|就诊|匹配|关系|边)|"
    r"(?:即使|即便).{0,12}(?:没有|无|为空).{0,20}(?:保留|返回|列出)|"
    r"(?:没有|无)(?:就诊|匹配|关系|记录|边).{0,16}(?:也|仍).{0,12}(?:保留|返回|列出)|"
    r"(?:零|0)\s*(?:次|条|个)|补零|不丢失|完整保留|左(?:外)?连接|全外连接|"
    r"left\s+(?:outer\s+)?join|full\s+(?:outer\s+)?join|preserv(?:e|ing)",
    re.IGNORECASE,
)
_CORRELATED_RE = re.compile(
    r"(?:相关子查询|correlated\s+subquery|各自的|该(?:科室|疾病|患者|地区|"
    r"关系|实体|分组)|每(?:个|位|种).{0,20}(?:高于|低于|等于|超过|不少于|"
    r"不高于).{0,16}(?:平均|最高|最低|最大|最小))",
    re.IGNORECASE,
)
_CTE_RE = re.compile(
    r"(?:\bCTE\b|公用表表达式|公共表表达式|WITH\s+[A-Za-z_]|"
    r"先在.{0,20}(?:汇总|筛选|计算).{0,20}再)",
    re.IGNORECASE,
)
_SET_RE = re.compile(
    r"(?:\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|并集|交集|差集|"
    r"既.{0,24}又|都(?:出现|有|就诊|包含)|在.{0,20}(?:但|却).{0,20}(?:没|未|不)|"
    r"同时满足.{0,16}(?:两|多)|set\s+operation)",
    re.IGNORECASE,
)
_WINDOW_RE = re.compile(
    r"(?:窗口函数|\bOVER\s*\(|\bPARTITION\s+BY\b|ROW_NUMBER|DENSE_RANK|"
    r"(?<![A-Za-z_])RANK\s*\(|\bLAG\s*\(|\bLEAD\s*\(|累计|滚动|"
    r"移动平均|环比|上一(?:次|条|月)|下一(?:次|条|月)|排名|名次)",
    re.IGNORECASE,
)
_TIES_RE = re.compile(
    r"(?:并列|同分|同名次|并列时|并列保留|全部并列|with\s+ties|"
    r"dense_rank|(?<!row_)(?<!dense_)rank\s*\()",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?:\d{4}[-年/]\d{1,2}(?:[-月/]\d{1,2}日?)?|\d{4}\s*年|"
    r"日期|就诊时间|时间范围|截至|之前|之后|以后|以来|期间|最近|最早|最新|"
    r"按(?:日|周|月|季|季度|年)|每(?:日|周|月|季度|年)|逐(?:日|周|月|年)|"
    r"同比|环比|date|datetime|timestamp|year|month|quarter|week)",
    re.IGNORECASE,
)
_NULL_RE = re.compile(
    r"(?:NULL|空值|为空|非空|不为空|缺失值?|未填写|未知值|无值|"
    r"没有(?:匹配|对应|记录|数据)|is\s+(?:not\s+)?null|coalesce|nullif)",
    re.IGNORECASE,
)
_OUTPUT_CONTRACT_RE = re.compile(
    r"(?:返回|列出|给出|显示|输出|展示|只要|仅返回|分别是多少|"
    r"按.{0,24}(?:升序|降序|排序|排列)|前\s*\d+|后\s*\d+|"
    r"保留\s*\d+\s*位小数|四舍五入|字段顺序|列顺序|"
    r"return|list|show|display|output|top\s*\d+|bottom\s*\d+|"
    r"order(?:ed)?\s+by|round(?:ed)?)",
    re.IGNORECASE,
)


def _normalise_question(question: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(question or "")).strip(),
    )


def canonical_capability_name(value: Any) -> str:
    """Return a stable snake-case label for one capability name."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    if not text:
        return ""
    if text in _CAPABILITY_ALIASES:
        return _CAPABILITY_ALIASES[text]
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return _CAPABILITY_ALIASES.get(text, text)


def normalise_capabilities(value: Any) -> tuple[str, ...]:
    """Normalise explicit capability metadata into a deterministic tuple.

    Mappings are treated as ``label -> enabled`` flags.  Unknown non-empty labels
    are retained after snake-case normalisation so external development sets can
    add dimensions without losing their metadata.
    """

    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw_values = [key for key, enabled in value.items() if bool(enabled)]
    elif isinstance(value, str):
        raw_values = [
            token
            for token in re.split(r"[,;|]+", value)
            if token.strip()
        ]
    elif isinstance(value, Iterable):
        raw_values = list(value)
    else:
        raw_values = [value]

    labels = {canonical_capability_name(item) for item in raw_values}
    labels.discard("")
    known = [name for name in CAPABILITY_ORDER if name in labels]
    unknown = sorted(labels - _CAPABILITY_SET)
    return tuple((*known, *unknown))


def _parse_sql_tree(sql_or_ast: Any) -> tuple[Any | None, Any | None]:
    if sql_or_ast is None:
        return None, None
    try:
        from sqlglot import exp, parse_one
    except ImportError:
        return None, None
    if isinstance(sql_or_ast, str):
        if not sql_or_ast.strip():
            return None, exp
        try:
            return parse_one(sql_or_ast, read="sqlite"), exp
        except Exception:
            # Classification is advisory.  Invalid candidate SQL must not make a
            # question-level report non-deterministically fail.
            return None, exp
    return sql_or_ast, exp


def _nodes(tree: Any, kind: Any) -> list[Any]:
    try:
        return list(tree.find_all(kind))
    except Exception:
        return []


def _nearest_select(node: Any, exp: Any) -> Any | None:
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, exp.Select):
            return parent
        parent = getattr(parent, "parent", None)
    return None


def _scope_aliases(select: Any, exp: Any) -> set[str]:
    aliases: set[str] = set()
    for table in _nodes(select, exp.Table):
        if _nearest_select(table, exp) is not select:
            continue
        alias = str(getattr(table, "alias_or_name", "") or "").strip().casefold()
        name = str(getattr(table, "name", "") or "").strip().casefold()
        if alias:
            aliases.add(alias)
        if name:
            aliases.add(name)
    return aliases


def _has_correlated_reference(tree: Any, exp: Any) -> bool:
    selects = _nodes(tree, exp.Select)
    for select in selects:
        ancestor = getattr(select, "parent", None)
        outer_aliases: set[str] = set()
        while ancestor is not None:
            if isinstance(ancestor, exp.Select):
                outer_aliases.update(_scope_aliases(ancestor, exp))
            ancestor = getattr(ancestor, "parent", None)
        if not outer_aliases:
            continue
        local_aliases = _scope_aliases(select, exp)
        for column in _nodes(select, exp.Column):
            if _nearest_select(column, exp) is not select:
                continue
            qualifier = str(getattr(column, "table", "") or "").strip().casefold()
            if qualifier and qualifier in outer_aliases and qualifier not in local_aliases:
                return True
    return False


def _count_distinct_in_tree(tree: Any, exp: Any) -> bool:
    for count in _nodes(tree, exp.Count):
        argument = getattr(count, "this", None)
        if isinstance(argument, exp.Distinct):
            return True
        try:
            if count.find(exp.Distinct) is not None:
                return True
        except Exception:
            pass
    return False


def _tree_flags(tree: Any, exp: Any) -> dict[str, bool]:
    flags = {name: False for name in CAPABILITY_ORDER}
    if tree is None or exp is None:
        return flags

    tables = _nodes(tree, exp.Table)
    columns = _nodes(tree, exp.Column)
    selects = _nodes(tree, exp.Select)
    joins = _nodes(tree, exp.Join)
    groups = _nodes(tree, exp.Group)
    distincts = _nodes(tree, exp.Distinct)
    windows = _nodes(tree, exp.Window)
    aggregates = _nodes(tree, exp.AggFunc)
    ctes = _nodes(tree, exp.CTE)
    set_nodes = []
    for name in ("Union", "Intersect", "Except"):
        kind = getattr(exp, name, None)
        if kind is not None:
            set_nodes.extend(_nodes(tree, kind))

    flags["schema_linking"] = bool(tables or columns)
    flags["grain"] = bool(groups or distincts or aggregates or windows)
    flags["joins_cardinality"] = bool(joins or len({table.name for table in tables}) > 1)
    flags["count_distinct"] = _count_distinct_in_tree(tree, exp)
    flags["aggregation_stage"] = bool(aggregates)
    flags["correlated"] = _has_correlated_reference(tree, exp)
    flags["cte"] = bool(ctes)
    flags["set_operation"] = bool(set_nodes)
    flags["window"] = bool(windows)

    for join in joins:
        side = str(join.args.get("side") or "").strip().casefold()
        kind = str(join.args.get("kind") or "").strip().casefold()
        if side in {"left", "right", "full"} or kind in {"left", "right", "full"}:
            flags["preservation"] = True
            break

    rank_types = tuple(
        kind
        for name in ("Rank", "DenseRank")
        if (kind := getattr(exp, name, None)) is not None
    )
    if rank_types:
        flags["ties"] = any(isinstance(node, rank_types) for node in tree.walk())

    date_function_names = {
        "date",
        "datetime",
        "strftime",
        "julianday",
        "unixepoch",
        "timediff",
        "date_diff",
        "datediff",
    }
    date_column_names = {
        "date",
        "datetime",
        "timestamp",
        "visit_date",
        "created_at",
        "updated_at",
    }
    if any(
        str(getattr(column, "name", "") or "").casefold() in date_column_names
        or str(getattr(column, "name", "") or "").casefold().endswith("_date")
        or str(getattr(column, "name", "") or "").casefold().endswith("_time")
        for column in columns
    ):
        flags["date"] = True
    for node in tree.walk():
        key = str(getattr(node, "key", "") or "").casefold()
        name = str(getattr(node, "name", "") or "").casefold()
        if key in date_function_names or name in date_function_names:
            flags["date"] = True
        if isinstance(node, exp.Literal) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}(?:[ T].*)?", str(node.this)
        ):
            flags["date"] = True

    null_types = tuple(
        kind
        for name in ("Null", "Coalesce", "Nullif")
        if (kind := getattr(exp, name, None)) is not None
    )
    if null_types:
        flags["null"] = any(isinstance(node, null_types) for node in tree.walk())

    # Every parsed SELECT has a concrete answer shape.  Richer clauses make that
    # contract visible even when the question text is unavailable.
    flags["output_contract"] = bool(selects)
    return flags


def classify_nl2sql_capability_flags(
    question: str,
    sql: Any = None,
    *,
    sql_ast: Any = None,
) -> dict[str, bool]:
    """Return all capability dimensions as deterministic boolean flags.

    ``sql`` may be SQLite text or a parsed ``sqlglot`` expression.  ``sql_ast``
    is an explicit synonym useful to callers that already parsed the statement;
    when supplied it takes precedence over ``sql``.
    """

    text = _normalise_question(question)
    flags = {
        "schema_linking": bool(_SCHEMA_LINKING_RE.search(text)),
        "grain": bool(_GRAIN_RE.search(text)),
        "joins_cardinality": bool(_JOIN_RE.search(text)),
        "count_distinct": bool(
            _COUNT_DISTINCT_RE.search(text)
            or (
                _PATIENT_CARDINALITY_RE.search(text)
                and _VISIT_CONTEXT_RE.search(text)
            )
        ),
        "aggregation_stage": bool(_AGGREGATION_RE.search(text)),
        "preservation": bool(_PRESERVATION_RE.search(text)),
        "correlated": bool(_CORRELATED_RE.search(text)),
        "cte": bool(_CTE_RE.search(text)),
        "set_operation": bool(_SET_RE.search(text)),
        "window": bool(_WINDOW_RE.search(text)),
        "ties": bool(_TIES_RE.search(text)),
        "date": bool(_DATE_RE.search(text)),
        "null": bool(_NULL_RE.search(text)),
        "output_contract": bool(_OUTPUT_CONTRACT_RE.search(text)),
    }

    tree, exp = _parse_sql_tree(sql_ast if sql_ast is not None else sql)
    ast_flags = _tree_flags(tree, exp)
    for name in CAPABILITY_ORDER:
        flags[name] = bool(flags[name] or ast_flags[name])
    return flags


def classify_nl2sql_capabilities(
    question: str,
    sql: Any = None,
    *,
    sql_ast: Any = None,
) -> tuple[str, ...]:
    """Return enabled semantic capability labels in ``CAPABILITY_ORDER``."""

    flags = classify_nl2sql_capability_flags(question, sql, sql_ast=sql_ast)
    return tuple(name for name in CAPABILITY_ORDER if flags[name])


# Short, discoverable compatibility aliases for evaluation/development callers.
classify_capabilities = classify_nl2sql_capabilities
capability_flags = classify_nl2sql_capability_flags

__all__ = [
    "CAPABILITY_ORDER",
    "canonical_capability_name",
    "normalise_capabilities",
    "classify_nl2sql_capability_flags",
    "classify_nl2sql_capabilities",
    "classify_capabilities",
    "capability_flags",
]
