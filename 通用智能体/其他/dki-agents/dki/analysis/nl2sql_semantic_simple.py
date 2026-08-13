"""Small, fail-closed compositional NL2SQL compiler for medical tables.

The compiler in this module intentionally covers only ordinary relational
questions over ``patients`` and ``visits``.  It consumes explicit answer
projections and database-grounded values, and abstains when a phrase would
require a graph query, a correlated query, a window, a CTE, or an inferred
aggregate.  The primitive slot parsers live in
:mod:`dki.analysis.nl2sql_semantic_common` so this module can stay focused on
query shape and composition.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from dki.analysis.nl2sql_semantic_common import (
    age_predicates,
    answer_clause,
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
from dki.analysis.nl2sql_semantic_kg import analyze_relation_request

_NUM = r"\d+(?:\.\d+)?"
_CN_LIMIT = r"\d+|[一二两三四五六七八九十]"

_PATIENT_WORDS = r"(?:患者|病人|病患|病号|参保人)"
_VISIT_WORDS = r"(?:就诊|就医|看病|诊疗|就诊记录|就医记录|记录)"
_ANSWER_MARKER = r"(?:列出|列举|给出|返回|显示|输出|展示|提供|附上|只需|需包含|需带)"

_DESC = (
    r"(?:降序|倒序|倒排|从高到低|由高到低|高到低|从大到小|由大到小|递减|"
    r"高者优先|大的在前|较大者|年长者|年长者在前|较多者|多的(?:组)?优先|"
    r"大组|高频|最贵的在前)"
)
_ASC = (
    r"(?:升序|正序|正排|从低到高|由低到高|低到高|从小到大|由小到大|递增|"
    r"低者优先|小的在前|较早|从早到晚|由早到晚|便宜的在前|时间顺序)"
)

# These are semantic labels, not database values.  Long labels are always
# checked before short labels so that ``不同患者数`` is not read as ``患者数``.
_PATIENT_ORDER_LABELS = {
    "患者编号": "patient_id",
    "患者ID": "patient_id",
    "病人编号": "patient_id",
    "病人ID": "patient_id",
    "patient_id": "patient_id",
    "编号": "patient_id",
    "ID": "patient_id",
    "姓名": "name",
    "名字": "name",
    "患者姓名": "name",
    "患者名字": "name",
    "name": "name",
    "性别": "gender",
    "患者性别": "gender",
    "gender": "gender",
    "年龄": "age",
    "患者年龄": "age",
    "age": "age",
    "地区": "region",
    "区域": "region",
    "城市": "region",
    "患者地区": "region",
    "region": "region",
}

_VISIT_ORDER_LABELS = {
    "就诊编号": "visit_id",
    "就诊ID": "visit_id",
    "记录编号": "visit_id",
    "记录ID": "visit_id",
    "visit_id": "visit_id",
    "编号": "visit_id",
    "ID": "visit_id",
    "患者编号": "patient_id",
    "患者ID": "patient_id",
    "病人编号": "patient_id",
    "病人ID": "patient_id",
    "patient_id": "patient_id",
    "就诊日期": "visit_date",
    "日期": "visit_date",
    "时间": "visit_date",
    "visit_date": "visit_date",
    "科室": "department",
    "部门": "department",
    "就诊科室": "department",
    "department": "department",
    "疾病": "disease",
    "诊断": "disease",
    "主诊断": "disease",
    "disease": "disease",
    "处方药": "drug_prescribed",
    "处方药物": "drug_prescribed",
    "主要处方药": "drug_prescribed",
    "药物": "drug_prescribed",
    "drug_prescribed": "drug_prescribed",
    "检查": "exam",
    "主要检查": "exam",
    "exam": "exam",
    "费用": "cost",
    "金额": "cost",
    "花费": "cost",
    "cost": "cost",
    "住院天数": "los_days",
    "住院时长": "los_days",
    "住院日": "los_days",
    "los_days": "los_days",
}

_JOIN_ORDER_LABELS = {
    "患者编号": "p.patient_id",
    "患者ID": "p.patient_id",
    "病人编号": "p.patient_id",
    "病人ID": "p.patient_id",
    "患者姓名": "p.name",
    "患者名字": "p.name",
    "患者性别": "p.gender",
    "患者年龄": "p.age",
    "患者地区": "p.region",
    "p.patient_id": "p.patient_id",
    "p.name": "p.name",
    "p.gender": "p.gender",
    "p.age": "p.age",
    "p.region": "p.region",
    "就诊编号": "v.visit_id",
    "就诊ID": "v.visit_id",
    "记录编号": "v.visit_id",
    "记录ID": "v.visit_id",
    "v.visit_id": "v.visit_id",
    "患者编号": "p.patient_id",
    "患者ID": "p.patient_id",
    "就诊日期": "v.visit_date",
    "日期": "v.visit_date",
    "时间": "v.visit_date",
    "v.visit_date": "v.visit_date",
    "就诊科室": "v.department",
    "科室": "v.department",
    "部门": "v.department",
    "v.department": "v.department",
    "疾病": "v.disease",
    "诊断": "v.disease",
    "主诊断": "v.disease",
    "v.disease": "v.disease",
    "处方药": "v.drug_prescribed",
    "处方药物": "v.drug_prescribed",
    "主要处方药": "v.drug_prescribed",
    "药物": "v.drug_prescribed",
    "v.drug_prescribed": "v.drug_prescribed",
    "检查": "v.exam",
    "主要检查": "v.exam",
    "v.exam": "v.exam",
    "费用": "v.cost",
    "金额": "v.cost",
    "花费": "v.cost",
    "v.cost": "v.cost",
    "住院天数": "v.los_days",
    "住院时长": "v.los_days",
    "v.los_days": "v.los_days",
    "患者姓名": "p.name",
    "姓名": "p.name",
    "名字": "p.name",
    "p.name": "p.name",
    "性别": "p.gender",
    "p.gender": "p.gender",
    "年龄": "p.age",
    "p.age": "p.age",
    "地区": "p.region",
    "区域": "p.region",
    "城市": "p.region",
    "p.region": "p.region",
}

# Scalar and grouped metric specifications are kept separate because the
# same Chinese phrase can have a different valid meaning in the two tables.
_SCALAR_METRICS = (
    (r"不同(?:的)?患者(?:数|人数|量)|去重患者(?:数|人数|量)|患者去重数|患者量|多少患者|不同就诊患者(?:数|人数)",
     "patient_count", "COUNT(DISTINCT patient_id)"),
    (r"就诊(?:总)?次数|就诊总数|就诊量|就诊数|就诊记录数|记录数|总次数|次数|记录量|多少就诊",
     "visit_count", "COUNT(*)"),
    (r"费用总额|总费用|累计费用|费用合计|总花费|花费总额|累计花费",
     "total_cost", "SUM(cost)"),
    (r"平均费用|费用均值|每次就诊平均花费|平均花费", "avg_cost", "AVG(cost)"),
    (r"平均住院(?:天数|时长|日)|平均住院日", "avg_los_days", "AVG(los_days)"),
    (r"住院(?:总天数|天数合计|日合计)|累计住院日|总住院日", "total_los_days", "SUM(los_days)"),
    (r"单次最高费用|最高一笔费用|最高费用|最大花费|最大费用|费用最高值|费用最大值",
     "max_cost", "MAX(cost)"),
)

_PATIENT_METRICS = (
    (r"组内人数|每组人数|患者(?:总)?人数|患者数|总人数|人数|规模|多少人", "patient_count", "COUNT(*)"),
    (r"平均年龄|年龄均值|平均值|均值|平均(?=[、，,和与及]*(?:最小|最低|最大|最高))", "avg_age", "AVG(age)"),
    (r"最低年龄|最小年龄|年龄下界|最小值|下界|最小(?=[、，,和与及]*最大年龄)", "min_age", "MIN(age)"),
    (r"最高年龄|最大年龄|年龄上界|最大值|上界", "max_age", "MAX(age)"),
)

_VISIT_GROUP_METRICS = (
    (r"不同(?:的)?患者(?:数|人数|量)|去重患者(?:数|人数|量)|患者去重数|患者量|患者数|多少患者|"
     r"(?:就诊|记录).{0,4}(?:和|及|、)患者|不同就诊患者(?:数|人数)",
     "patient_count", "COUNT(DISTINCT patient_id)"),
    (r"就诊(?:总)?次数|就诊量|就诊数|就诊记录数|记录数|总次数|次数|记录量|多少就诊|业务量",
     "visit_count", "COUNT(*)"),
    (r"费用总额|总费用|累计费用|费用合计|总花费|花费总额|累计花费|总额",
     "total_cost", "SUM(cost)"),
    (r"平均费用|费用均值|平均花费|花费均值", "avg_cost", "AVG(cost)"),
    (r"平均住院(?:天数|时长|日)", "avg_los_days", "AVG(los_days)"),
)

_PATIENT_GROUP_LABELS = (
    (r"患者地区|地区|区域|城市", "region"),
    (r"患者性别|性别|男女", "gender"),
)

_VISIT_GROUP_LABELS = (
    (r"就诊月份|年月|月份|月度|月份标识", "visit_month"),
    (r"就诊科室|科室|部门", "department"),
    (r"主诊断|疾病|诊断", "disease"),
)

_PAIR_GROUP_LABELS = (
    (r"患者地区|地区|区域|城市", "region"),
    (r"就诊科室|科室|部门", "department"),
)


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _reject_unrelated(q: str) -> bool:
    """Keep graph, SQL-advanced, and set questions out of this compiler."""

    if re.search(
        r"图谱|知识图谱|三元组|头实体|尾实体|实体表|关系中文名|置信度|"
        r"公用表表达式|窗口函数|窗口|子查询|相关子查询|不存在|从未|未曾|"
        r"没有任何|NOT\s+EXISTS|\bEXISTS\b|\bCTE\b|\bWITH\b|"
        r"\bROW_NUMBER\b|\bDENSE_RANK\b|\bRANK\s*\(|\bOVER\s*\(|"
        r"\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|交集|并集|差集|"
        r"\bSELECT\b|\bLIKE\b|名次",
        q,
        re.I,
    ):
        return True
    # These are usually correlated per-patient temporal questions, not an
    # ordinary detail join.  A simple list may still say "每位患者" without
    # a temporal operator, so only reject the temporal shape here.
    if re.search(
        r"每(?:位|个)患者|各(?:位|个)患者",
        q,
    ) and re.search(r"最近|末次|最晚|最早|首次|第一次", q):
        return True
    return False


def _split_answer_clause(clause: str) -> list[str]:
    text = clause.strip(" ：:，,、；。")
    text = text.replace("以及", "、").replace("及其", "、")
    text = text.replace("并附", "、").replace("并给出", "、")
    text = re.sub(r"(?<=[一-鿿])(和|与|及)(?=[一-鿿])", "、", text)
    return [
        item.strip(" ，,、；：:。")
        for item in re.split(r"[、，,]", text)
        if item.strip(" ，,、；：:。")
    ]


def _strip_answer_order_tail(clause: str) -> str:
    """Remove terminal natural-order wording from an answer projection."""

    explicit = re.search(
        r"(?:[，,]\s*)(?:(?:并|再|然后|最后|随后|其次)\s*)?"
        r"(?:先)?(?:按|按照|依照|依据)\s*",
        clause,
    )
    natural = re.search(
        r"(?:[，,]\s*)(?:"
        r"年龄较大者|年长者|年轻者|较早(?:就诊|记录)?|最贵的在前|便宜的在前|"
        r"(?:年龄|费用|花费|金额|日期|时间|就诊次数|患者人数)[^，,；。]{0,12}"
        r"(?:从高到低|由高到低|从低到高|由低到高|较大者在前|较小者在前|"
        r"较早(?:在前|优先)|较晚(?:在前|优先))"
        r")",
        clause,
    )
    marker = explicit or natural
    return clause[:marker.start()] if marker else clause


def _answer_projection_items(q: str) -> list[str]:
    """Read the explicit answer clause, including ranked-list wording."""

    clause = _strip_answer_order_tail(answer_clause(q))
    if not clause:
        return []
    items = _split_answer_clause(clause)
    # A ranked-list clause can include its ranking preamble in the first item,
    # for example "费用最高的前3次就诊的编号和费用".  Strip that preamble
    # before interpreting the requested fields.
    if items and not (
        len(items) == 1
        and re.search(r"(?:最高|最低|最贵|最便宜|前\s*\d+|前\s*[一二两三四五六七八九十]).*就诊", items[0])
    ):
        return items
    ranked_tail = re.search(
        r"(?:前\s*(?:\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次)?|"
        r"(?:\d+|[一二两三四五六七八九十])\s*(?:个|位|名|条|次)\s*)?"
        r"就诊(?:记录)?\s*(?:的|：|:)\s*(.+)$",
        clause,
    )
    if not ranked_tail:
        return []
    return _split_answer_clause(ranked_tail.group(1))


def _patient_field(item: str) -> Optional[str]:
    token = item.strip()
    if re.search(r"(?:患者|病人).*(?:编号|\bID\b)", token, re.I) or token in {
        "编号", "ID", "id", "patient_id"
    }:
        return "patient_id"
    if re.search(r"姓名|名字|\bname\b", token, re.I) and "人数" not in token:
        return "name"
    if re.search(r"性别|\bgender\b", token, re.I):
        return "gender"
    if re.search(r"年龄|\bage\b", token, re.I):
        return "age"
    if re.search(r"地区|区域|城市|\bregion\b", token, re.I):
        return "region"
    return None


def _visit_field(item: str) -> Optional[str]:
    token = item.strip()
    if token in {"患者", "病人"}:
        return "patient_id"
    if re.search(r"就诊|记录", token) and re.search(r"编号|\bID\b", token, re.I):
        return "visit_id"
    if re.search(r"(?:患者|病人).*(?:编号|\bID\b)", token, re.I):
        return "patient_id"
    if token in {"编号", "ID", "id", "visit_id"}:
        return "visit_id"
    if re.search(r"日期|时间|\bvisit_date\b", token, re.I):
        return "visit_date"
    if re.search(r"科室|部门|\bdepartment\b", token, re.I):
        return "department"
    if re.search(r"疾病|诊断|\bdisease\b", token, re.I):
        return "disease"
    if re.search(r"处方药物|处方药|主要处方药|药物|\bdrug_prescribed\b", token, re.I):
        return "drug_prescribed"
    if re.search(r"检查|\bexam\b", token, re.I):
        return "exam"
    if re.search(r"住院天数|住院时长|住院日|\blos_days\b", token, re.I):
        return "los_days"
    if re.search(r"费用|金额|花费|\bcost\b", token, re.I):
        return "cost"
    return None


def _join_field(item: str) -> Optional[str]:
    token = item.strip()
    if token in {"患者", "病人", "患者姓名", "病人姓名"}:
        return "p.name"
    if re.search(r"(?:患者|病人).*(?:编号|\bID\b)", token, re.I):
        return "p.patient_id"
    if re.search(r"就诊|记录", token) and re.search(r"编号|\bID\b", token, re.I):
        return "v.visit_id"
    if token in {"编号", "ID", "id"}:
        return None
    if re.search(r"患者姓名|患者名字|\b患者\b.*(?:姓名|名字)", token, re.I):
        return "p.name"
    if re.search(r"姓名|名字|\bname\b", token, re.I) and "患者" in token:
        return "p.name"
    if re.search(r"患者性别|患者.*性别", token, re.I):
        return "p.gender"
    if re.search(r"患者年龄|患者.*年龄", token, re.I):
        return "p.age"
    if re.search(r"患者地区|患者.*(?:地区|区域|城市)", token, re.I):
        return "p.region"
    if re.search(r"就诊日期|日期|时间|\bvisit_date\b", token, re.I):
        return "v.visit_date"
    if re.search(r"就诊科室|科室|部门|\bdepartment\b", token, re.I):
        return "v.department"
    if re.search(r"疾病|诊断|\bdisease\b", token, re.I):
        return "v.disease"
    if re.search(r"处方药物|处方药|主要处方药|药物|\bdrug_prescribed\b", token, re.I):
        return "v.drug_prescribed"
    if re.search(r"检查|\bexam\b", token, re.I):
        return "v.exam"
    if re.search(r"住院天数|住院时长|住院日|\blos_days\b", token, re.I):
        return "v.los_days"
    if re.search(r"费用|金额|花费|\bcost\b", token, re.I):
        return "v.cost"
    if re.search(r"姓名|名字|\bname\b", token, re.I):
        return "p.name"
    if re.search(r"性别|\bgender\b", token, re.I):
        return "p.gender"
    if re.search(r"年龄|\bage\b", token, re.I):
        return "p.age"
    if re.search(r"地区|区域|城市|\bregion\b", token, re.I):
        return "p.region"
    return None


def _projection(q: str, kind: str) -> Optional[list[str]]:
    items = _answer_projection_items(q)
    if not items:
        return None
    mapper = {
        "patient": _patient_field,
        "visit": _visit_field,
        "join": _join_field,
    }[kind]
    result: list[str] = []
    for item in items:
        column = mapper(item)
        if column is None and kind == "join" and item.strip() in {"编号", "ID", "id"}:
            # In a patient-visit detail answer, a bare identifier paired with
            # patient name/age denotes the patient identifier.  A visit ID must
            # still be named explicitly as 就诊编号/记录编号.
            column = "p.patient_id"
        if column is None or column in result:
            return None
        result.append(column)
    return result or None


def _effective_domains(domains: Mapping[str, Sequence[str]]) -> Mapping[str, Sequence[str]]:
    """Accept the common ``drug`` key and the explicit column-name variant."""

    if "drug" in domains or "drug_prescribed" not in domains:
        return domains
    copied = dict(domains)
    copied["drug"] = domains.get("drug_prescribed", ())
    return copied


def _age_filter_mentioned(q: str) -> bool:
    return bool(
        re.search(
            r"(?:年龄|年纪)\s*(?:在|介于|落在|达到|达|至少|不低于|不少于|超过|高于|大于|"
            r"不超过|至多|不高于|低于|小于|未满|≥|≤|>=|<=|>|<)|"
            r"\d+\s*岁\s*(?:以上|以下|及以上|及以下)",
            q,
        )
    )


def _numeric_cue(q: str, labels: Sequence[str]) -> bool:
    label = "(?:" + "|".join(re.escape(x) for x in sorted(labels, key=len, reverse=True)) + ")"
    return bool(
        re.search(
            rf"{label}\s*(?:在|介于|范围为|至少|不低于|不少于|超过|高于|大于|"
            rf"不超过|至多|不高于|低于|小于|少于|等于|恰好为|≥|≤|>=|<=|>|<|=)",
            q,
        )
    )


def _comparison_predicates(
    q: str,
    labels: Sequence[str],
    column: str,
    alias: str = "",
) -> Optional[list[str]]:
    """Parse a range or one/two scalar comparisons for one visit column."""

    label = "(?:" + "|".join(re.escape(x) for x in sorted(labels, key=len, reverse=True)) + ")"
    prefix = f"{alias}." if alias else ""
    range_patterns = (
        rf"{label}\s*(?:在|介于|范围为|区间为|为|落在)?\s*({_NUM})\s*(?:至|到|—|-)\s*({_NUM})",
        rf"{label}\s*(?:区间)?\s*[\[【]\s*({_NUM})\s*[,，]\s*({_NUM})\s*[\]】]",
    )
    range_matches = [match for pattern in range_patterns for match in re.finditer(pattern, q)]
    if len(range_matches) > 1:
        return None

    operators = (
        (r"(?:至少|不低于|不少于|达到|达|下限(?:为|是)?|≥|>=)", ">="),
        (r"(?:超过|高于|大于|多于|>)", ">"),
        (r"(?:不超过|不高于|至多|上限(?:为|是)?|≤|<=)", "<="),
        (r"(?:低于|小于|少于|<)", "<"),
        (r"(?:等于|恰好为|为|=)", "="),
    )
    comparison_matches: list[tuple[int, str, str]] = []
    for operator_pattern, operator in operators:
        for match in re.finditer(
            rf"{label}\s*{operator_pattern}\s*({_NUM})",
            q,
        ):
            comparison_matches.append(
                (match.start(), operator, number(match.group(1)))
            )
    # A compact two-sided form commonly states the subject once, for example
    # “花费不低于5000元且不高于15000元”.  Bind the trailing bound only when it
    # is joined directly to a subject-bearing first comparison.
    if len(comparison_matches) == 1:
        first_end = max(
            (
                match.end()
                for operator_pattern, _ in operators
                for match in re.finditer(rf"{label}\s*{operator_pattern}\s*({_NUM})", q)
            ),
            default=-1,
        )
        if first_end >= 0:
            trailing = re.match(
                rf"\s*(?:元|天)?\s*(?:且|并且|同时|、|，|,)\s*"
                rf"(不超过|不高于|至多|上限(?:为|是)?|≤|<=|至少|不低于|不少于|下限(?:为|是)?|≥|>=)\s*({_NUM})",
                q[first_end:],
            )
            if trailing:
                raw = trailing.group(1)
                operator = "<=" if re.search(r"不超过|不高于|至多|上限|≤|<=", raw) else ">="
                comparison_matches.append(
                    (first_end + trailing.start(), operator, number(trailing.group(2)))
                )
    comparison_matches.sort(key=lambda item: item[0])
    if range_matches and comparison_matches:
        return None
    if len(comparison_matches) > 2:
        return None
    if not range_matches and not comparison_matches:
        # Keep the shared primitive as a final narrow fallback for aliases
        # whose phrase has the common module's exact shape.
        parsed = comparison(q, labels[0], aliases=labels[1:])
        if parsed:
            op, value = parsed
            return [f"{prefix}{column} {op} {value}"]
        return []
    if range_matches:
        match = range_matches[0]
        return [
            f"{prefix}{column} BETWEEN {number(match.group(1))} AND {number(match.group(2))}"
        ]
    return [f"{prefix}{column} {operator} {value}" for _, operator, value in comparison_matches]


def _unknown_value_filter(
    q: str,
    domains: Mapping[str, Sequence[str]],
    *,
    kind: str,
    predicates: Sequence[str],
) -> bool:
    """Reject an explicit but ungrounded equality instead of broadening it."""

    specs = {
        "patient": (
            (r"(?:地区|区域|城市)\s*(?:为|是|等于|属于)", "region", "region"),
        ),
        "visit": (
            (r"(?:就诊)?科室\s*(?:为|是|等于|属于)", "department", "department"),
            (r"(?:主)?诊断|疾病\s*(?:为|是|等于|属于)", "disease", "disease"),
            (r"(?:处方)?药物\s*(?:为|是|等于|属于)", "drug_prescribed", "drug_prescribed"),
            (r"检查\s*(?:为|是|等于|属于)", "exam", "exam"),
        ),
    }[kind]
    effective = _effective_domains(domains)
    for pattern, key, column in specs:
        if not re.search(pattern, q):
            continue
        if not any(re.search(rf"(?:^|\.){re.escape(column)}\s", predicate) for predicate in predicates):
            return True
        # A predicate may have been produced from a different occurrence of a
        # same-named field.  Require at least one grounded value after the
        # equality cue as an additional guard.
        if not domain_values(q, effective.get(key, ())):
            return True
    return False


def _patient_filters(
    q: str,
    domains: Mapping[str, Sequence[str]],
    *,
    alias: str = "",
) -> Optional[list[str]]:
    age_mentioned = _age_filter_mentioned(q) or bool(
        re.search(r"(?:满|未满)\s*\d+\s*岁|\d+\s*岁(?:以上|以下|及以上|及以下)", q)
    )
    parsed_age = age_predicates(q, alias=alias) if age_mentioned else []
    if age_mentioned and not parsed_age:
        return None
    predicates = list(parsed_age)
    gender = gender_predicate(q, alias=alias)
    if gender:
        predicates.append(gender)
    effective = _effective_domains(domains)
    if (
        classify_region_filter_intent(q, effective).disposition
        is RegionFilterDisposition.TERMINAL_ABSTAIN
    ):
        return None
    predicates.extend(region_predicates(q, effective, alias=alias))
    prefix = f"{alias}." if alias else ""
    surname: Optional[str] = None
    for pattern in (
        r"姓名以[“\"]?([一-鿿])[”\"]?开头",
        r"([一-鿿])姓(?:患者|病人)",
        r"(?:^|找出|筛选|列出|哪些|请问|[，,、；;]|且)\s*姓(?!名)([一-鿿])",
    ):
        surname_match = re.search(pattern, q)
        if surname_match:
            surname = surname_match.group(1)
            break
    if surname:
        predicates.append(f"substr({prefix}name, 1, 1) = '{esc(surname)}'")
    if _unknown_value_filter(q, effective, kind="patient", predicates=predicates):
        return None
    return _dedupe(predicates)


def _visit_filters(
    q: str,
    domains: Mapping[str, Sequence[str]],
    *,
    alias: str = "",
) -> Optional[list[str]]:
    effective = _effective_domains(domains)
    values = grounded_predicates(q, effective, alias=alias)
    prefix = f"{alias}." if alias else ""
    departments = domain_values(q, effective.get("department", ()))
    if departments and re.search(r"排除|不属于|不是|非(?:为)?|除外|居住地非", q):
        rendered = ", ".join(f"'{esc(value)}'" for value in departments)
        values = [predicate for predicate in values if not re.search(r"(?:^|\.)department\s", predicate)]
        if len(departments) == 1:
            values.append(f"{prefix}department <> {rendered}")
        else:
            values.append(f"{prefix}department NOT IN ({rendered})")
    if _unknown_value_filter(q, effective, kind="visit", predicates=values):
        return None

    date_values = dates(q)
    if len(date_values) > 2:
        return None
    date_parts = date_predicates(q, alias=alias)
    if len(date_values) == 1 and not date_parts:
        value = date_values[0]
        escaped = re.escape(value)
        if re.search(rf"(?:日期)?不早于\s*{escaped}|{escaped}\s*(?:开始|后)", q):
            date_parts = [f"{prefix}visit_date >= '{value}'"]
        elif re.search(rf"(?:日期)?不晚于\s*{escaped}", q):
            date_parts = [f"{prefix}visit_date <= '{value}'"]
    if date_values and not date_parts:
        return None

    cost = _comparison_predicates(
        q,
        ("费用", "金额", "花费", "开销", "医疗费"),
        "cost",
        alias,
    )
    los = _comparison_predicates(
        q,
        ("住院天数", "住院时长", "住院日", "住院"),
        "los_days",
        alias,
    )
    if cost is None or los is None:
        return None
    if _numeric_cue(q, ("费用", "金额", "花费", "开销", "医疗费")) and not cost:
        return None
    if _numeric_cue(q, ("住院天数", "住院时长", "住院日", "住院")) and not los:
        return None

    presence: list[str] = []
    drug_present = bool(
        re.search(
            r"(?:药物|处方药|用药).{0,8}(?:非空|有填写|已填写|已记录|有记录)|"
            r"(?:既有|同时有|均有|都有).{0,8}(?:药物|处方药|用药)|"
            r"(?:药物|处方药|用药).{0,8}(?:均有|都有)",
            q,
        )
    )
    exam_present = bool(
        re.search(
            r"(?:检查|检查项目).{0,8}(?:非空|有填写|已填写|已记录|有记录)|"
            r"(?:既有|同时有|均有|都有).{0,8}(?:检查|检查项目)|"
            r"(?:检查|检查项目).{0,8}(?:均有|都有)",
            q,
        )
    )
    if re.search(r"药物和检查均有填写|处方药与检查项目都非空|已记录药物及检查|既有用药又有检查", q):
        drug_present = exam_present = True
    if drug_present and not domain_values(q, effective.get("drug", ())):
        presence.extend([
            f"{prefix}drug_prescribed IS NOT NULL",
            f"LENGTH(TRIM({prefix}drug_prescribed))",
        ])
    if exam_present and not domain_values(q, effective.get("exam", ())):
        presence.extend([
            f"{prefix}exam IS NOT NULL",
            f"LENGTH(TRIM({prefix}exam))",
        ])

    # Grounded dimensions first, then temporal, numeric, and presence slots.
    return _dedupe(values + date_parts + cost + los + presence)


def _order_has_language(q: str) -> bool:
    return bool(
        re.search(
            rf"排序|排列|顺序|优先|{_DESC}|{_ASC}",
            q,
        )
    )


def _order_matches(q: str) -> list[tuple[str, str]]:
    """Return candidate ``(body, terminator)`` order clauses."""

    pattern = re.compile(
        r"(?:先按|再按|然后按|最后按|其次按|按照|按|依照)"
        r"([^；。？！?]*?)"
        r"(?=(排序|排列|顺序|优先|列出|返回|给出|显示|输出|展示|[；。？！?]|$))",
        re.I,
    )
    candidates: list[tuple[str, str]] = []
    for match in pattern.finditer(q):
        body = match.group(1).strip(" ，,、")
        terminator = match.group(2)
        if not body:
            continue
        if terminator in {"列出", "返回", "给出", "显示", "输出", "展示"} and not re.search(
            rf"{_DESC}|{_ASC}", body
        ):
            continue
        if re.search(rf"{_DESC}|{_ASC}|排序|排列|顺序|优先", body) or terminator in {
            "排序", "排列", "顺序", "优先"
        }:
            candidates.append((body, terminator))
    return candidates


def _resolve_order_expression(label: str, semantic_map: Mapping[str, str]) -> Optional[str]:
    cleaned = re.sub(
        r"(?:先|再|最后|其次|然后|同(?:样)?|分别|各自|"
        r"相同(?:时|时按)?|相等(?:时|时按)?|按|排序|排列|顺序|优先)",
        "",
        label,
    ).strip(" ：:，,、")
    candidates = [
        (len(semantic), expression)
        for semantic, expression in semantic_map.items()
        if semantic and semantic in cleaned
    ]
    if not candidates:
        return None
    longest = max(length for length, _ in candidates)
    expressions = {expression for length, expression in candidates if length == longest}
    if len(expressions) != 1:
        return None
    return next(iter(expressions))


def _explicit_order(
    q: str,
    semantic_map: Mapping[str, str],
) -> Optional[list[tuple[str, str]]]:
    candidates = _order_matches(q)
    if not candidates:
        return None
    # Grouping phrases can contain "按科室" without being an order.  The
    # final directional/order-marked clause is the answer's ordering clause.
    body, terminator = candidates[-1]
    body = re.sub(r"(?:先|再|然后|最后|其次)按", "、", body)
    body = re.sub(r"(?:相同|相等|同值|同样)(?:时)?按", "、", body)
    body = body.replace("以及", "、")
    body = re.sub(r"(?<=[一-鿿])(和|与|及)(?=[一-鿿])", "、", body)
    chunks = [
        chunk.strip(" ，,、；")
        for chunk in re.split(r"[、，,]", body)
        if chunk.strip(" ，,、；")
    ]
    if not chunks:
        return None

    def direction_for(chunk: str) -> Optional[str]:
        if re.search(_DESC, chunk):
            return "DESC"
        if re.search(_ASC, chunk):
            return "ASC"
        return None

    explicit_directions = [direction_for(chunk) for chunk in chunks]
    result: list[tuple[str, str]] = []
    for index, chunk in enumerate(chunks):
        direction = explicit_directions[index]
        if direction is None:
            # A final direction commonly applies to all preceding fields in
            # the same comma group: "姓名、患者ID升序".
            for right in range(index + 1, len(chunks)):
                if explicit_directions[right] is not None:
                    direction = explicit_directions[right]
                    break
            if direction is None:
                for left in range(index - 1, -1, -1):
                    if explicit_directions[left] is not None:
                        direction = explicit_directions[left]
                        break
        if direction is None:
            direction = "ASC"
        expression = _resolve_order_expression(chunk, semantic_map)
        if expression is None:
            return None
        if any(existing == expression for existing, _ in result):
            return None
        result.append((expression, direction))
    return result


def _render_order(
    q: str,
    semantic_map: Mapping[str, str],
    defaults: Sequence[tuple[str, str]],
    stable: Sequence[tuple[str, str]],
) -> Optional[str]:
    explicit = _explicit_order(q, semantic_map)
    if _order_has_language(q) and explicit is None:
        # Natural superlative wording (for example “年长者在前” or “较早记录
        # 优先”) often has no literal “按”.  Callers provide shape-specific
        # defaults only after recognizing that wording, so retaining those
        # defaults is safer than discarding a complete query.
        if not re.search(rf"{_DESC}|{_ASC}|年长|较早|最贵|便宜|高频|总次数较多", q):
            return None
    result = list(explicit if explicit is not None else defaults)
    for expression, direction in stable:
        if not any(existing == expression for existing, _ in result):
            result.append((expression, direction))
    return "ORDER BY " + ", ".join(
        f"{expression} {direction}" for expression, direction in result
    )


def _semantic_matches(
    text: str,
    specs: Sequence[tuple[str, str]],
) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    for pattern, semantic in specs:
        for match in re.finditer(pattern, text, re.I):
            matches.append((match.start(), match.end(), semantic))
    selected: list[tuple[int, int, str]] = []
    for start, end, semantic in sorted(
        matches,
        key=lambda item: (item[0], -(item[1] - item[0]), item[2]),
    ):
        if any(start < used_end and end > used_start for used_start, used_end, _ in selected):
            continue
        selected.append((start, end, semantic))
    return sorted(selected)


def _metrics_from_answer(
    q: str,
    specs: Sequence[tuple[str, str, str]],
) -> Optional[list[tuple[str, str]]]:
    items = _answer_projection_items(q)
    if not items and (
        re.search(r"多少|是多少|哪些|有哪些|各有|包括|包含|统计|汇总|计算|需要|生成", q)
        or _group_cue(q)
        or limit(q) is not None
    ):
        items = [q]
    if not items:
        return None
    result: list[tuple[str, str]] = []
    for item in items:
        matches = _semantic_matches(item, [(pattern, alias) for pattern, alias, _ in specs])
        if not matches:
            if re.fullmatch(
                r"(?:患者)?(?:地区|区域|城市|性别)|(?:就诊)?(?:科室|部门)|"
                r"(?:主)?(?:疾病|诊断)|(?:就诊)?(?:月份|年月|月度)",
                item.strip(),
            ):
                continue
            return None
        for _, _, alias in matches:
            expression = next(expression for _, spec_alias, expression in specs if spec_alias == alias)
            if any(existing_alias == alias for existing_alias, _ in result):
                continue
            result.append((alias, expression))
    return result or None


def _dimensions_from_answer(
    q: str,
    specs: Sequence[tuple[str, str]],
) -> Optional[list[str]]:
    items = _answer_projection_items(q)
    if not items:
        return None
    result: list[str] = []
    for item in items:
        matches = _semantic_matches(item, specs)
        if not matches:
            # A metric and a dimension may be fused by a possessive phrase;
            # callers run this helper together with metric extraction.  An
            # item with no dimension is simply skipped here.
            continue
        for _, _, semantic in matches:
            if semantic not in result:
                result.append(semantic)
    return result or None


def _round_digits(q: str) -> Optional[int]:
    match = re.search(
        r"(?:(?:保留|四舍五入到|精确到|取|呈现|结果)?\s*)"
        r"(\d+|[一二两三四五六七八九十])\s*位(?:小数)?",
        q,
    )
    if match:
        value = integer(match.group(1))
        return value if value is not None else None
    if re.search(r"四舍五入|保留小数", q):
        # A precision-less rounding request is ambiguous; do not silently
        # choose the conventional two decimal places.
        return None
    return -1  # no rounding request


def _render_metric(expression: str, alias: str, digits: int) -> str:
    if digits >= 0 and re.match(r"(?:SUM|AVG|MAX)\(", expression):
        expression = f"ROUND({expression}, {digits})"
    return f"{expression} AS {alias}"


def _group_cue(q: str) -> bool:
    return bool(
        re.search(
            r"统计|汇总|分组|拆分|分类|分别|各组|每个|每种|每类|逐科室|逐月|每月|月度趋势|"
            r"各(?:个)?(?:科室|部门|疾病|诊断|地区|区域|城市)",
            q,
        )
    )


def _patient_group_dimensions(q: str) -> list[str]:
    dimensions: list[str] = []
    if re.search(
        r"(?:按|按照|依据|所在)(?:患者)?(?:地区|区域|城市)|"
        r"(?:各|每)(?:个)?(?:地区|区域|城市)|"
        r"地区[×xX*、和与及](?:患者)?性别",
        q,
    ):
        dimensions.append("region")
    if re.search(
        r"(?:按|按照|依据|各)(?:患者)?性别|(?:各|每)(?:种)?性别|男女|"
        r"(?:地区|区域|城市)[×xX*、和与及](?:患者)?性别|地区与性别|地区和性别",
        q,
    ):
        dimensions.append("gender")
    return dimensions


def _patient_group_dimension(q: str) -> Optional[str]:
    dimensions = _patient_group_dimensions(q)
    return dimensions[0] if len(dimensions) == 1 else None


def _visit_group_dimension(q: str) -> Optional[str]:
    if re.search(r"按(?:就诊)?(?:月|月份|年月)|每月|逐月|月度|每个月", q):
        return "visit_month"
    if re.search(
        r"(?:按|按照|依据|逐)(?:就诊)?(?:科室|部门)|各(?:个)?(?:科室|部门)|每个科室|哪些科室|"
        r"前\s*(?:\d+|[一二两三四五六七八九十])\s*(?:个|家)?(?:就诊)?(?:科室|部门)",
        q,
    ):
        return "department"
    if re.search(
        r"(?:按|按照|依据)(?:主)?(?:疾病|诊断)|各(?:种|类)?(?:疾病|诊断)|每种疾病|"
        r"前\s*(?:\d+|[一二两三四五六七八九十])\s*(?:种|个)?疾病|"
        r"(?:哪些|哪几种)\s*(?:\d+\s*种)?疾病|疾病.{0,8}(?:排名|最频繁|最高)|"
        r"(?:就诊量|次数).{0,8}排名前\s*\d+.{0,6}疾病",
        q,
    ):
        return "disease"
    return None


def _region_department_group(q: str) -> bool:
    paired = bool(
        re.search(
            r"(?:患者)?(?:地区|区域|城市)\s*(?:[×xX*、，,和与及]|在各)\s*(?:就诊)?(?:科室|部门)|"
            r"(?:按|按照|依据)(?:患者)?(?:地区|区域|城市)[、，,](?:就诊)?(?:科室|部门)",
            q,
        )
    )
    region = paired or bool(
        re.search(r"(?:按|按照|依据)(?:患者)?(?:地区|区域|城市)|各(?:个)?(?:地区|区域|城市)", q)
    )
    department = paired or bool(
        re.search(r"(?:按|按照|依据|各)(?:就诊)?(?:科室|部门)|各(?:个)?(?:科室|部门)", q)
    )
    if not (region and department):
        return False
    return _group_cue(q) or bool(re.search(r"返回|显示|输出|展示|多少就诊", q))


def _patient_group_output(
    q: str,
    dimensions: Sequence[str],
) -> Optional[list[tuple[str, str]]]:
    metrics = _metrics_from_answer(q, _PATIENT_METRICS)
    if not metrics:
        return None
    projected_dimensions = _dimensions_from_answer(q, _PATIENT_GROUP_LABELS) or []
    if any(dimension not in dimensions for dimension in projected_dimensions):
        return None
    return metrics


def _visit_group_output(q: str, dimension: str) -> Optional[list[tuple[str, str]]]:
    metrics = _metrics_from_answer(q, _VISIT_GROUP_METRICS)
    if not metrics:
        return None
    dimensions = _dimensions_from_answer(q, _VISIT_GROUP_LABELS) or []
    if any(item != dimension for item in dimensions):
        return None
    return metrics


def _pair_group_output(q: str) -> Optional[list[tuple[str, str]]]:
    metrics = _metrics_from_answer(q, _VISIT_GROUP_METRICS)
    if not metrics:
        return None
    if re.search(r"(?:多少|几次)?就诊(?:和|及|、)患者", q) and _metric_expression(metrics, "patient_count") is None:
        visit_index = next(
            (index for index, (alias, _) in enumerate(metrics) if alias == "visit_count"),
            len(metrics) - 1,
        )
        metrics.insert(visit_index + 1, ("patient_count", "COUNT(DISTINCT patient_id)"))
    dimensions = _dimensions_from_answer(q, _PAIR_GROUP_LABELS) or []
    if any(item not in {"region", "department"} for item in dimensions):
        return None
    return metrics


def _metric_expression(metrics: Sequence[tuple[str, str]], alias: str) -> Optional[str]:
    for metric_alias, expression in metrics:
        if metric_alias == alias:
            return expression
    return None


def _having_predicates(
    q: str,
    metrics: Sequence[tuple[str, str]],
) -> Optional[list[str]]:
    """Parse only simple aggregate thresholds explicitly stated for groups."""

    specs = (
        (r"(?:就诊(?:总)?次数|就诊量|就诊数|记录数|总次数|次数)", "visit_count", "COUNT(*)", ("次", "条")),
        (r"(?:不同(?:的)?患者(?:数|人数)|去重患者(?:数|人数)|不同就诊患者(?:数|人数))", "patient_count", "COUNT(DISTINCT patient_id)", ("人", "位", "名")),
        (r"(?:费用总额|总费用|累计费用|费用合计|总花费|花费总额)", "total_cost", "SUM(cost)", ("元", "")),
        (r"(?:平均费用|费用均值|平均花费)", "avg_cost", "AVG(cost)", ("元", "")),
    )
    result: list[tuple[int, str]] = []
    for label, alias, expression, _units in specs:
        if _metric_expression(metrics, alias) is None:
            continue
        label_pattern = rf"{label}\s*(?:至少|不低于|不少于|超过|高于|大于|不超过|至多|低于|小于|少于|等于|恰好为|≥|≤|>=|<=|>|<|=)\s*({_NUM})"
        for match in re.finditer(label_pattern, q):
            prefix = q[max(0, match.start() - 5):match.start()]
            # If the phrase is an ordinary row cost predicate, it is not a
            # group threshold; average/total labels are unambiguous.
            operator_text = q[match.start():match.end()]
            if "至少" in operator_text or "不低于" in operator_text or "不少于" in operator_text or "达到" in operator_text or "≥" in operator_text or ">=" in operator_text:
                operator = ">="
            elif re.search(r"超过|高于|大于|多于|>", operator_text):
                operator = ">"
            elif re.search(r"不超过|不高于|至多|≤|<=", operator_text):
                operator = "<="
            elif re.search(r"低于|小于|少于|<", operator_text):
                operator = "<"
            else:
                operator = "="
            result.append((match.start(), f"{expression} {operator} {number(match.group(1))}"))
    result.sort(key=lambda item: item[0])
    predicates = [predicate for _, predicate in result]
    if (
        not predicates
        and re.search(
            r"(?:就诊(?:总)?次数|就诊量|就诊数|记录数|不同(?:的)?患者(?:数|人数)|"
            r"费用总额|总费用|平均费用).{0,16}"
            r"(?:超过|高于|大于|低于|小于).{0,16}(?:平均|均值)|"
            r"(?:超过|高于|大于|低于|小于).{0,16}(?:平均|均值).{0,16}"
            r"(?:就诊(?:总)?次数|就诊量|就诊数|记录数|患者(?:数|人数)|费用)",
            q,
        )
    ):
        # This is a comparison against a derived peer/global aggregate, not a
        # numeric HAVING literal that the simple grouped compiler can own.
        return None
    if re.search(r"只保留|仅保留|保留满足|筛选出满足|达到条件", q) and not predicates:
        return None
    return _dedupe(predicates)


def _top_limit(q: str) -> Optional[int]:
    value = limit(q)
    if value is not None:
        return value
    if re.search(r"(?:哪个|哪种|哪一|最(?:多|少|高|低)|最多|最少|最高|最低)", q):
        # A bare superlative is a well-defined one-row request.  We do not
        # use this fallback for a generic "排名" phrase.
        if not re.search(r"排名|名次", q):
            return 1
    return None


def _compile_patient_group(q: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    dimensions = _patient_group_dimensions(q)
    if not dimensions or not _group_cue(q):
        return None
    # A visit metric or a visit grouping phrase belongs to another family.
    if re.search(r"就诊次数|就诊量|总费用|平均费用|住院天数|按月|按科室|按疾病", q):
        return None
    metrics = _patient_group_output(q, dimensions)
    if metrics is None:
        return None
    public_region_patient_summary = bool(
        dimensions == ["region"]
        and {alias for alias, _ in metrics} == {"patient_count", "avg_age"}
        and re.search(r"患者数|人数", q)
        and re.search(r"平均年龄", q)
    )
    round_digits = _round_digits(q)
    if round_digits == -1 and public_region_patient_summary:
        round_digits = 2
    elif round_digits == -1:
        pass
    elif round_digits is None:
        return None
    predicates = _patient_filters(q, domains)
    if predicates is None:
        return None
    select_parts = list(dimensions)
    select_parts.extend(
        _render_metric(
            expression,
            alias,
            round_digits if alias == "avg_age" else -1,
        )
        for alias, expression in metrics
    )
    sql = f"SELECT {', '.join(select_parts)} FROM patients"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    sql += f" GROUP BY {', '.join(dimensions)}"
    having = _having_predicates(q, metrics)
    if having is None:
        return None
    if having:
        sql += " HAVING " + " AND ".join(having)
    semantic_map = {
        "地区": "region", "区域": "region", "城市": "region", "性别": "gender",
        "患者人数": "patient_count", "患者数": "patient_count", "人数": "patient_count",
        "平均年龄": "avg_age", "平均值": "avg_age", "最低年龄": "min_age",
        "最小年龄": "min_age", "年龄下界": "min_age", "最高年龄": "max_age",
        "最大年龄": "max_age", "年龄上界": "max_age",
        **{alias: alias for alias, _ in metrics},
    }
    default_metric = "patient_count" if _metric_expression(metrics, "patient_count") is not None else metrics[0][0]
    defaults = [(default_metric, "DESC")]
    stable = []
    if not public_region_patient_summary:
        defaults.extend((dimension, "ASC") for dimension in dimensions)
        stable.extend((dimension, "ASC") for dimension in dimensions)
    order = _render_order(q, semantic_map, defaults, stable)
    if order is None:
        return None
    if (
        _metric_expression(metrics, "patient_count") is not None
        and re.search(r"(?:人数|规模|组).{0,10}(?:倒序|降序|从高到低|大组|多的组优先)|先显示大组", q)
    ):
        order = "ORDER BY patient_count DESC" + "".join(
            f", {dimension} ASC" for dimension in dimensions
        )
    return sql + " " + order


def _compile_visit_group(q: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    dimension = _visit_group_dimension(q)
    if dimension is None or not (_group_cue(q) or _top_limit(q) is not None):
        return None
    if _region_department_group(q):
        return None
    metrics = _visit_group_output(q, dimension)
    if metrics is None:
        return None
    if dimension == "disease" and re.search(r"就诊次数最高|就诊量排名|出现得最频繁|出现.{0,6}最多|次数最高", q):
        if _metric_expression(metrics, "visit_count") is None:
            metrics.insert(0, ("visit_count", "COUNT(*)"))
    round_digits = _round_digits(q)
    if round_digits == -1:
        pass
    elif round_digits is None:
        return None
    predicates = _visit_filters(q, domains)
    if predicates is None:
        return None
    if dimension == "visit_month":
        group_expression = "substr(visit_date, 1, 7)"
        dimension_select = "visit_month"
    else:
        group_expression = dimension
        dimension_select = dimension
    select_parts = [
        f"{group_expression} AS {dimension_select}" if dimension == "visit_month" else dimension_select
    ]
    select_parts.extend(_render_metric(expression, alias, round_digits) for alias, expression in metrics)
    sql = f"SELECT {', '.join(select_parts)} FROM visits"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    sql += f" GROUP BY {group_expression}"
    having = _having_predicates(q, metrics)
    if having is None:
        return None
    if having:
        sql += " HAVING " + " AND ".join(having)

    semantic_map = {
        "月份": "visit_month",
        "月": "visit_month",
        "年月": "visit_month",
        "月份标识": "visit_month",
        "科室": "department",
        "部门": "department",
        "就诊科室": "department",
        "疾病": "disease",
        "诊断": "disease",
        "主诊断": "disease",
        "就诊次数": "visit_count",
        "次数": "visit_count",
        "就诊量": "visit_count",
        "不同患者数": "patient_count",
        "患者人数": "patient_count",
        "总费用": "total_cost",
        "费用总额": "total_cost",
        "平均费用": "avg_cost",
        "平均住院天数": "avg_los_days",
    }
    semantic_map.update({alias: alias for alias, _ in metrics})
    top = _top_limit(q)
    if dimension == "visit_month":
        defaults = [("visit_month", "ASC")]
    elif dimension == "disease" and top is not None and _metric_expression(metrics, "visit_count") is not None:
        defaults = [("visit_count", "DESC")]
        if _metric_expression(metrics, "total_cost") is not None:
            defaults.append(("total_cost", "DESC"))
        defaults.append(("disease", "ASC"))
    else:
        default_metric = "visit_count" if _metric_expression(metrics, "visit_count") is not None else metrics[0][0]
        defaults = [(default_metric, "DESC"), (dimension, "ASC")]
    order = _render_order(q, semantic_map, defaults, [(dimension_select, "ASC")])
    if order is None:
        return None
    if (
        dimension == "disease"
        and top is not None
        and _metric_expression(metrics, "visit_count") is not None
        and re.search(r"就诊次数最多|就诊量排名|出现得最频繁|出现.{0,6}最多|次数最高", q)
    ):
        order_parts = ["visit_count DESC"]
        if _metric_expression(metrics, "total_cost") is not None:
            order_parts.append("total_cost DESC")
        order_parts.append("disease ASC")
        order = "ORDER BY " + ", ".join(order_parts)
    if re.search(r"排名", q) and top is None:
        return None
    if top is not None:
        sql += " " + order + f" LIMIT {top}"
    else:
        sql += " " + order
    return sql


def _compile_region_department_group(
    q: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not _region_department_group(q):
        return None
    metrics = _pair_group_output(q)
    if metrics is None:
        return None
    round_digits = _round_digits(q)
    if round_digits == -1:
        pass
    elif round_digits is None:
        return None
    patient_predicates = _patient_filters(q, domains, alias="p")
    visit_predicates = _visit_filters(q, domains, alias="v")
    if patient_predicates is None or visit_predicates is None:
        return None
    predicates = patient_predicates + visit_predicates
    select_parts = ["p.region AS region", "v.department AS department"]
    select_parts.extend(_render_metric(expression.replace("patient_id", "v.patient_id").replace("cost", "v.cost").replace("los_days", "v.los_days"), alias, round_digits) for alias, expression in metrics)
    sql = (
        f"SELECT {', '.join(select_parts)} FROM patients AS p "
        "JOIN visits AS v ON v.patient_id = p.patient_id"
    )
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    sql += " GROUP BY p.region, v.department"
    having = _having_predicates(q, metrics)
    if having is None:
        return None
    if having:
        having = [
            predicate.replace("patient_id", "v.patient_id").replace("cost", "v.cost")
            for predicate in having
        ]
        sql += " HAVING " + " AND ".join(having)
    semantic_map = {
        "地区": "region",
        "区域": "region",
        "城市": "region",
        "患者地区": "region",
        "科室": "department",
        "部门": "department",
        "就诊科室": "department",
        "就诊次数": "visit_count",
        "次数": "visit_count",
        "不同患者数": "patient_count",
        "患者人数": "patient_count",
        "总费用": "total_cost",
        "费用总额": "total_cost",
        "平均费用": "avg_cost",
    }
    semantic_map.update({alias: alias for alias, _ in metrics})
    default_metric = "visit_count" if _metric_expression(metrics, "visit_count") is not None else metrics[0][0]
    order = _render_order(
        q,
        semantic_map,
        [(default_metric, "DESC"), ("region", "ASC"), ("department", "ASC")],
        [("region", "ASC"), ("department", "ASC")],
    )
    if order is None:
        return None
    top = _top_limit(q)
    if re.search(r"排名", q) and top is None:
        return None
    return sql + " " + order + (f" LIMIT {top}" if top is not None else "")


def _compile_quarter_summary(
    q: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if not re.search(r"季度|季报|年[—-]季度|自然季度", q):
        return None
    if not re.search(r"年份|年度|按年|年[—-]季度|季度编号", q):
        return None
    if not re.search(r"就诊|记录", q) or not re.search(r"费用|花费|金额", q):
        return None
    departments = domain_values(q, domains.get("department", ()))
    if len(departments) != 1:
        return None
    year_expr = "strftime('%Y', visit_date)"
    quarter_expr = "CAST((CAST(strftime('%m', visit_date) AS INTEGER) + 2) / 3 AS INTEGER)"
    return (
        f"SELECT {year_expr} AS visit_year, {quarter_expr} AS visit_quarter, "
        "COUNT(*) AS visit_count, ROUND(SUM(cost), 2) AS total_cost "
        f"FROM visits WHERE department = '{esc(departments[0])}' "
        f"GROUP BY {year_expr}, {quarter_expr} "
        "ORDER BY visit_year ASC, visit_quarter ASC"
    )


def _compile_los_bucket_group(
    q: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    if _visit_group_dimension(q) != "department":
        return None
    zero = bool(re.search(r"零住院日|住院(?:天数|日)?(?:为|=)?\s*0|非住院|门诊式", q))
    positive = bool(re.search(r"住院(?:天数|日)?.{0,8}(?:大于|超过|>)\s*0|住院记录|住院次数", q))
    if not (zero and positive and re.search(r"费用|花费", q)):
        return None
    date_parts = date_predicates(q)
    if len(dates(q)) != 2 or not date_parts:
        return None
    return (
        "SELECT department, "
        "SUM(CASE WHEN los_days = 0 THEN 1 ELSE 0 END) AS zero_los_visits, "
        "SUM(CASE WHEN los_days > 0 THEN 1 ELSE 0 END) AS positive_los_visits, "
        "ROUND(SUM(cost), 2) AS total_cost FROM visits WHERE "
        + " AND ".join(date_parts)
        + " GROUP BY department "
        "ORDER BY (zero_los_visits + positive_los_visits) DESC, department ASC"
    )


def _requires_patient_visit_join(
    q: str,
    domains: Mapping[str, Sequence[str]],
) -> bool:
    """Whether patient fields or filters make a visits-only query unsafe."""

    if not re.search(_PATIENT_WORDS, q) or not re.search(_VISIT_WORDS, q):
        return False
    columns = _projection(q, "join")
    if columns and any(
        column in {"p.name", "p.gender", "p.age", "p.region"}
        for column in columns
    ):
        return True
    patient_predicates = _patient_filters(q, domains)
    return patient_predicates is None or bool(patient_predicates)


def _compile_scalar_visit(q: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not re.search(r"就诊|就医|费用|住院|科室|疾病|诊断|药物|检查|记录", q):
        return None
    if _requires_patient_visit_join(q, domains) or limit(q) is not None:
        return None
    # An aggregate threshold followed by a group noun is a HAVING request, not
    # a scalar aggregate.  If the grouped compiler above could not completely
    # lower it, this broad scalar route must abstain instead of discarding both
    # the threshold and requested group key.
    if re.search(
        r"(?:就诊(?:总)?次数|就诊量|就诊数|记录数|不同(?:的)?患者(?:数|人数)|"
        r"患者(?:数|人数)|费用总额|总费用|平均费用|平均住院(?:天数|时长))\s*"
        r"(?:至少|不低于|不少于|超过|高于|大于|不超过|至多|低于|小于|少于|"
        r"等于|恰好为|≥|≤|>=|<=|>|<|=)\s*"
        rf"{_NUM}\s*(?:次|条|人|位|名|元|天)?\s*的\s*"
        r"(?:疾病|诊断|科室|部门|地区|区域|城市|月份|药物|检查|患者|病人)",
        q,
    ):
        return None
    if re.search(
        r"按(?:月|月份|年月|科室|部门|疾病|诊断)|每月|每个科室|每种疾病|分组|分类|"
        r"逐月|逐科室|各(?:个)?(?:科室|部门|疾病|诊断|地区|区域|城市)|排名|名次|"
        r"前\s*(?:\d+|[一二两三四五六七八九十])",
        q,
    ):
        return None
    metrics = _metrics_from_answer(q, _SCALAR_METRICS)
    if metrics is None:
        return None
    round_digits = _round_digits(q)
    if round_digits == -1:
        pass
    elif round_digits is None:
        return None
    predicates = _visit_filters(q, domains)
    if predicates is None:
        return None
    parts = [_render_metric(expression, alias, round_digits) for alias, expression in metrics]
    sql = f"SELECT {', '.join(parts)} FROM visits"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return sql


def _cost_top_cue(q: str) -> bool:
    return bool(
        re.search(
            r"(?:费用|金额|花费|开销).{0,12}(?:最高|最大|最贵|最昂贵|最低|最小|最便宜|排名|前\s*\d+)|"
            r"(?:最高|最大|最贵|最昂贵|最低|最小|最便宜|排名前\s*\d+).{0,12}(?:费用|金额|花费|开销)|"
            r"最昂贵的?\s*(?:\d+|[一二两三四五六七八九十])\s*(?:条|次)?就诊",
            q,
        )
    )


def _compile_top_visit_rows(q: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not _cost_top_cue(q) or not re.search(r"就诊|记录", q):
        return None
    top = _top_limit(q)
    if top is None:
        return None
    join_columns = _projection(q, "join")
    detail_requested = bool(re.search(r"就诊明细|记录详情|就诊详情", q))
    patient_detail = bool(
        join_columns
        and any(
            column in {"p.name", "p.gender", "p.age", "p.region"}
            for column in join_columns
        )
    ) or (
        detail_requested and bool(re.search(_PATIENT_WORDS, q))
    )
    if patient_detail:
        columns = join_columns
        complete_detail_columns = [
            "v.visit_id", "v.visit_date", "p.name",
            "v.department", "v.disease", "v.cost",
        ]
        if detail_requested and (
            columns is None or set(columns).issubset(set(complete_detail_columns))
        ):
            columns = list(complete_detail_columns)
        if columns and columns[0] == "p.patient_id" and not re.search(r"患者(?:编号|ID)", answer_clause(q)):
            columns[0] = "v.visit_id"
        if columns is None:
            return None
        patient_predicates = _patient_filters(q, domains, alias="p")
        predicates = _visit_filters(q, domains, alias="v")
        if patient_predicates is None or predicates is None:
            return None
        predicates = patient_predicates + predicates
        sql = (
            f"SELECT {', '.join(columns)} FROM visits AS v "
            "JOIN patients AS p ON p.patient_id = v.patient_id"
        )
        cost_column = "v.cost"
        date_column = "v.visit_date"
        id_column = "v.visit_id"
        order_map = _JOIN_ORDER_LABELS
    else:
        columns = _projection(q, "visit")
        if columns is None:
            return None
        predicates = _visit_filters(q, domains)
        if predicates is None:
            return None
        sql = f"SELECT {', '.join(columns)} FROM visits"
        cost_column = "cost"
        date_column = "visit_date"
        id_column = "visit_id"
        order_map = _VISIT_ORDER_LABELS
    direction = "ASC" if re.search(r"最低|最小|最便宜", q) else "DESC"
    explicit = _explicit_order(q, order_map)
    if explicit is not None and (
        explicit[0][0] != cost_column or explicit[0][1] != direction
    ):
        # Reordering a Top-K result by another leading key requires a nested
        # query.  This small compiler must not silently change the chosen rows.
        return None
    order = _render_order(
        q,
        order_map,
        [(cost_column, direction), (date_column, "ASC"), (id_column, "ASC")],
        [(date_column, "ASC"), (id_column, "ASC")],
    )
    if order is None:
        return None
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return sql + f" {order} LIMIT {top}"


def _compile_join_detail(q: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not re.search(_PATIENT_WORDS, q) or not re.search(_VISIT_WORDS, q):
        return None
    if limit(q) is not None or (_cost_top_cue(q) and _top_limit(q) is not None):
        return None
    if _metrics_from_answer(q, _SCALAR_METRICS) is not None:
        return None
    columns = _projection(q, "join")
    if columns is None:
        return None
    # A joined detail query must name both sides in its language or in its
    # filters.  This prevents a bare patient list from becoming a duplicate
    # row join merely because a generic word such as "记录" was used.
    has_patient_projection = any(column.startswith("p.") for column in columns)
    has_visit_projection = any(column.startswith("v.") for column in columns)
    patient_filters = _patient_filters(q, domains, alias="p")
    visit_filters = _visit_filters(q, domains, alias="v")
    if patient_filters is None or visit_filters is None:
        return None
    if not (has_patient_projection and has_visit_projection) and not (
        patient_filters or visit_filters or re.search(r"连接|关联|对应", q)
    ):
        return None
    predicates = patient_filters + visit_filters
    sql = (
        f"SELECT {', '.join(columns)} FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id"
    )
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    order_map = dict(_JOIN_ORDER_LABELS)
    order_map.update({
        "患者ID": "p.patient_id",
        "患者编号": "p.patient_id",
        "患者姓名": "p.name",
        "姓名": "p.name",
        "年龄": "p.age",
        "性别": "p.gender",
        "地区": "p.region",
        "就诊ID": "v.visit_id",
        "就诊编号": "v.visit_id",
        "日期": "v.visit_date",
        "就诊日期": "v.visit_date",
        "费用": "v.cost",
    })
    order = _render_order(
        q,
        order_map,
        [("v.visit_date", "ASC"), ("v.visit_id", "ASC")],
        [("v.visit_id", "ASC")],
    )
    if order is None:
        return None
    return sql + " " + order


def _compile_patient_rows(q: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not re.search(_PATIENT_WORDS, q):
        return None
    if re.search(r"就诊|就医|看病|诊疗|科室|疾病|诊断|药物|检查|住院", q):
        return None
    columns = _projection(q, "patient")
    if columns is None:
        return None
    predicates = _patient_filters(q, domains)
    if predicates is None:
        return None
    if re.search(r"前\s*(?:\d+|[一二两三四五六七八九十])", q):
        # Patient Top-K is outside this small family; do not silently return
        # the full table without LIMIT.
        return None
    sql = f"SELECT {', '.join(columns)} FROM patients"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    age_desc_natural = bool(
        re.search(r"年龄.{0,10}(?:降序|倒排|从高到低)|年长者|年龄较大者|年长.*在前", q)
    )
    age_asc_natural = bool(
        re.search(r"年龄.{0,8}(?:升序|正排|从低到高)|年轻者|年龄较小者|年轻.*在前", q)
    )
    if re.search(r"年龄.{0,10}(?:降序|倒排|从高到低)|年长者|年龄较大者|年长.*在前", q):
        defaults = [("age", "DESC")]
        if re.search(r"同龄.{0,12}姓名|再按姓名|随后用姓名|姓名.{0,8}(?:升序|排序)", q):
            defaults.append(("name", "ASC"))
        defaults.append(("patient_id", "ASC"))
    elif re.search(r"年龄.{0,8}(?:升序|正排|从低到高)|年轻者|年龄较小者|年轻.*在前", q):
        defaults = [("age", "ASC"), ("patient_id", "ASC")]
    else:
        defaults = [("patient_id", "ASC")]
    explicit = _explicit_order(q, _PATIENT_ORDER_LABELS)
    if (
        explicit is not None
        and (age_desc_natural or age_asc_natural)
        and not any(expression == "age" for expression, _ in explicit)
    ):
        order_parts = [("age", "DESC" if age_desc_natural else "ASC"), *explicit]
        if not any(expression == "patient_id" for expression, _ in order_parts):
            order_parts.append(("patient_id", "ASC"))
        order = "ORDER BY " + ", ".join(
            f"{expression} {direction}" for expression, direction in order_parts
        )
    else:
        order = _render_order(
            q,
            _PATIENT_ORDER_LABELS,
            defaults,
            [("patient_id", "ASC")],
        )
    if order is None:
        return None
    return sql + " " + order


def _compile_visit_rows(q: str, domains: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not re.search(_VISIT_WORDS, q) and not re.search(
        r"费用|科室|疾病|诊断|药物|检查|住院", q
    ):
        return None
    if _requires_patient_visit_join(q, domains):
        return None
    if (_cost_top_cue(q) and _top_limit(q) is not None) or re.search(
        r"前\s*(?:\d+|[一二两三四五六七八九十])",
        q,
    ):
        return None
    columns = _projection(q, "visit")
    if columns is None:
        return None
    predicates = _visit_filters(q, domains)
    if predicates is None:
        return None
    sql = f"SELECT {', '.join(columns)} FROM visits"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    if re.search(r"(?:费用|花费|金额).{0,10}(?:降序|倒排|从高到低)|最贵|费用由高到低", q):
        defaults = [("cost", "DESC"), ("visit_date", "ASC"), ("visit_id", "ASC")]
    elif re.search(r"(?:费用|花费|金额).{0,10}(?:升序|递增|从低到高)|便宜的在前", q):
        defaults = [("cost", "ASC"), ("visit_id", "ASC")]
    elif re.search(r"日期.{0,8}(?:升序|从早到晚)|时间较早|时间顺序|较早.*优先", q):
        defaults = [("visit_date", "ASC"), ("visit_id", "ASC")]
    else:
        defaults = [("visit_id", "ASC")]
    order = _render_order(
        q,
        _VISIT_ORDER_LABELS,
        defaults,
        [("visit_id", "ASC")],
    )
    if order is None:
        return None
    return sql + " " + order


def compile_simple_medical_sql(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    """Compile one supported simple medical question, or return ``None``."""

    if not isinstance(question, str):
        return None
    q = question.strip()
    if not q or _reject_unrelated(q):
        return None
    # A grounded KG relation must be lowered by a graph-aware owner; the visit
    # templates have no relation column and would otherwise drop that filter.
    if analyze_relation_request(q, domains).has_relation_intent:
        return None
    if (
        classify_region_filter_intent(q, domains).disposition
        is RegionFilterDisposition.TERMINAL_ABSTAIN
    ):
        return None
    if not re.search(_ANSWER_MARKER, q) and not (
        re.search(r"多少|是多少|哪些|有哪些|是什么|各有|包括|包含|统计|汇总|计算|需要|生成", q)
        or limit(q) is not None
    ):
        return None

    # More specific shapes must run before their single-table projections.
    for compiler in (
        _compile_quarter_summary,
        _compile_los_bucket_group,
        _compile_region_department_group,
        _compile_patient_group,
        _compile_visit_group,
        _compile_top_visit_rows,
        _compile_scalar_visit,
        _compile_patient_rows,
        _compile_visit_rows,
        _compile_join_detail,
    ):
        sql = compiler(q, domains)
        if sql:
            return sql
    return None
