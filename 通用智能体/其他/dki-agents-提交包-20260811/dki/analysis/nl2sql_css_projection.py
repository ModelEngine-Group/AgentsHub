"""Deterministic projection of public CSS development rows onto the sealed schema.

The projector is deliberately independent from :mod:`dki.analysis.nl2sql`.
It consumes the public CSS Spider-style SQL annotation, maps only medically
compatible columns, grounds literals in the sealed four-table database, and
rejects every row whose semantics cannot be represented without guessing.

The official CSS test split is not needed by this module.  A caller can freeze
this source and later apply the same deterministic transformation once inside
the private blind-consumption protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.analysis.nl2sql_sqlite import connect_readonly


PROJECTION_SEED = "20260725-css-four-table-v1"
PROJECTION_SEED_V2 = "20260726-public-official-test-v2"
AGGREGATIONS = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")
CONDITION_OPERATORS = (
    "NOT",
    "BETWEEN",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "IN",
    "LIKE",
    "IS",
    "EXISTS",
)

# Only names whose medical meaning is stable across CSS development schemas are
# admitted.  Identifier/code companions and institution-specific fields are
# intentionally absent: mapping them would silently change the question.
COLUMN_CONCEPTS: dict[str, str] = {
    # Patient identity and demographics.
    "PERSON_ID": "patient_id",
    "RYBH": "patient_id",
    "OVERALL_CD_PERSON": "patient_id",
    "PERSON_NM": "name",
    "PERSON_INFO_XM": "name",
    "XM": "name",
    "HZXM": "name",
    "PERSON_SEX": "gender",
    "PERSON_INFO_XBMC": "gender",
    "XBMC": "gender",
    "PERSON_AGE": "age",
    "NLY": "age",
    "NLS": "age",
    "PERSON_INFO_CSD": "region",
    "PERSON_INFO_JGMC": "region",
    "CSD": "region",
    "JGMC": "region",
    # Encounter identity and dates.
    "MED_CLINIC_ID": "visit_id",
    "JZLSH": "visit_id",
    "MZJZLSH": "visit_id",
    "CLINIC_ID": "visit_id",
    "IN_HOSP_DATE": "visit_date",
    "CLINIC_SLT_DATE": "visit_date",
    "JZKSRQ": "visit_date",
    "RYDJSJ": "visit_date",
    "RYSJ": "visit_date",
    "EXP_OCC_DATE": "visit_date",
    "STA_DATE": "visit_date",
    "JYRQ": "visit_date",
    "BGRQ": "visit_date",
    "SQRQ": "visit_date",
    "CJRQ": "visit_date",
    "BGSJ": "visit_date",
    "SHSJ": "visit_date",
    "CYSJ": "visit_date",
    "OUT_HOSP_DATE": "visit_date",
    # Encounter attributes.
    "MED_ORG_DEPT_NM": "department",
    "JZKSMC": "department",
    "KSMC": "department",
    "SQKSMC": "department",
    "JYKSMC": "department",
    "CYKSMC": "department",
    "IN_DIAG_DIS_NM": "disease",
    "OUT_DIAG_DIS_NM": "disease",
    "JZZDSM": "disease",
    "MZZDMC": "disease",
    "ZYZDMC": "disease",
    "MZZYZDZZMC": "disease",
    "ZYZYZDZZMC": "disease",
    "SOC_SRT_DIRE_NM": "drug_prescribed",
    "MED_DIRE_NM": "drug_prescribed",
    "JCZBMC": "exam",
    "JCXMMC": "exam",
    "JYXMMC": "exam",
    "MED_AMOUT": "cost",
    "AMOUNT": "cost",
    "IN_HOSP_DAYS": "los_days",
}

TARGET_COLUMNS = {
    "patient_id": "p.patient_id",
    "name": "p.name",
    "gender": "p.gender",
    "age": "p.age",
    "region": "p.region",
    "visit_id": "v.visit_id",
    "visit_date": "v.visit_date",
    "department": "v.department",
    "disease": "v.disease",
    "drug_prescribed": "v.drug_prescribed",
    "exam": "v.exam",
    "cost": "v.cost",
    "los_days": "v.los_days",
}
PATIENT_CONCEPTS = {"patient_id", "name", "gender", "age", "region"}
VISIT_CONCEPTS = {
    "visit_id",
    "visit_date",
    "department",
    "disease",
    "drug_prescribed",
    "exam",
    "cost",
    "los_days",
}
NUMERIC_CONCEPTS = {"age", "cost", "los_days"}
DATE_CONCEPTS = {"visit_date"}


class ProjectionRejected(ValueError):
    """A fail-closed row rejection with a stable aggregate reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CSSSchema:
    db_id: str
    columns: tuple[tuple[int, str], ...]
    table_names: tuple[str, ...]

    def concept(self, column_index: int) -> str:
        if column_index == 0:
            return "star"
        if column_index < 0 or column_index >= len(self.columns):
            raise ProjectionRejected("column_index_out_of_range")
        _, original_name = self.columns[column_index]
        concept = COLUMN_CONCEPTS.get(original_name.upper())
        if not concept:
            raise ProjectionRejected("unmapped_source_column")
        return concept


@dataclass(frozen=True)
class TargetDomains:
    joined_rows: tuple[Mapping[str, Any], ...]
    patient_rows: tuple[Mapping[str, Any], ...]
    dates: tuple[str, ...]


@dataclass(frozen=True)
class RenderedExpression:
    sql: str
    concept: str
    concepts: frozenset[str]
    aggregation: str = ""


@dataclass(frozen=True)
class LiteralBinding:
    original: Any
    replacement: Any


@dataclass(frozen=True)
class ProjectedItem:
    item: dict[str, Any]
    skeleton_role: str
    skeleton_sha256: str


def _stable_int(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big")


def _source_row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_css_schemas(
    tables_path: Path,
    *,
    allowed_db_ids: Iterable[str] | None = None,
) -> dict[str, CSSSchema]:
    rows = json.loads(Path(tables_path).read_text(encoding="utf-8"))
    allowed = None if allowed_db_ids is None else {str(value) for value in allowed_db_ids}
    result: dict[str, CSSSchema] = {}
    for row in rows:
        db_id = str(row["db_id"])
        if allowed is not None and db_id not in allowed:
            continue
        if db_id in result:
            raise ValueError(f"duplicate CSS schema id: {db_id}")
        columns = tuple(
            (int(table_index), str(name))
            for table_index, name in row["column_names_original"]
        )
        result[db_id] = CSSSchema(
            db_id=db_id,
            columns=columns,
            table_names=tuple(str(name) for name in row["table_names_original"]),
        )
    if allowed is not None and set(result) != allowed:
        missing = sorted(allowed - set(result))
        raise ValueError(f"CSS metadata is missing development schemas: {missing}")
    return result


def load_target_domains(db_path: Path) -> TargetDomains:
    connection = connect_readonly(db_path)
    connection.row_factory = sqlite3.Row
    try:
        joined = tuple(
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    p.patient_id AS patient_id,
                    p.name AS name,
                    p.gender AS gender,
                    p.age AS age,
                    p.region AS region,
                    v.visit_id AS visit_id,
                    v.visit_date AS visit_date,
                    v.department AS department,
                    v.disease AS disease,
                    v.drug_prescribed AS drug_prescribed,
                    v.exam AS exam,
                    v.cost AS cost,
                    v.los_days AS los_days
                FROM visits v
                JOIN patients p ON p.patient_id = v.patient_id
                ORDER BY v.visit_id
                """
            )
        )
        patients = tuple(
            dict(row)
            for row in connection.execute(
                """
                SELECT patient_id, name, gender, age, region
                FROM patients
                ORDER BY patient_id
                """
            )
        )
        dates = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT visit_date FROM visits ORDER BY visit_date"
            )
            if row[0]
        )
    finally:
        connection.close()
    if not joined or not patients or not dates:
        raise ValueError("sealed four-table database has empty projection domains")
    return TargetDomains(joined_rows=joined, patient_rows=patients, dates=dates)


def skeleton_role(skeleton: str) -> str:
    """Assign whole AST families before either development or holdout selection."""
    digest = hashlib.sha256(
        f"{PROJECTION_SEED}\0{skeleton}".encode("utf-8")
    ).hexdigest()
    # Thirty percent of structural families are reserved without looking at
    # any official test question.  The larger reserve protects the >=120 item
    # blind floor after strict semantic rejection.
    return "holdout" if int(digest[:8], 16) % 10 < 3 else "development"


def skeleton_role_v2(skeleton: str) -> str:
    """Reserve 65% of target AST families for the larger official-test pool.

    V2 combines CSS and EHRSQL.  The shared target-AST partition is computed
    before either official test split is projected, so a structure can never
    occur in both development and holdout even when it appears in both source
    benchmarks.
    """
    digest = hashlib.sha256(
        f"{PROJECTION_SEED_V2}\0{skeleton}".encode("utf-8")
    ).hexdigest()
    return "holdout" if int(digest[:8], 16) % 20 < 13 else "development"


def _normalise_select(select_payload: Any) -> tuple[bool, Sequence[Any]]:
    if (
        isinstance(select_payload, list)
        and len(select_payload) == 2
        and isinstance(select_payload[0], bool)
        and isinstance(select_payload[1], list)
    ):
        return bool(select_payload[0]), select_payload[1]
    if isinstance(select_payload, list):
        return False, select_payload
    raise ProjectionRejected("invalid_select_annotation")


def _render_col_unit(
    unit: Any,
    schema: CSSSchema,
    *,
    allow_bare_star: bool = False,
) -> RenderedExpression:
    if not isinstance(unit, list) or len(unit) < 2:
        raise ProjectionRejected("invalid_column_unit")
    aggregation_id = int(unit[0])
    if aggregation_id < 0 or aggregation_id >= len(AGGREGATIONS):
        raise ProjectionRejected("unsupported_aggregation")
    concept = schema.concept(int(unit[1]))
    distinct = bool(unit[2]) if len(unit) > 2 else False
    if concept == "star":
        column_sql = "*"
        concepts: frozenset[str] = frozenset()
    else:
        column_sql = TARGET_COLUMNS[concept]
        concepts = frozenset({concept})
    aggregation = AGGREGATIONS[aggregation_id]
    if not aggregation and concept == "star" and not allow_bare_star:
        raise ProjectionRejected("unbounded_star_projection")
    if aggregation in {"SUM", "AVG"} and concept not in NUMERIC_CONCEPTS:
        raise ProjectionRejected("non_numeric_aggregation")
    if aggregation in {"MAX", "MIN"} and concept == "star":
        raise ProjectionRejected("aggregate_star_not_supported")
    if distinct and concept == "star":
        raise ProjectionRejected("distinct_star")
    inner = f"DISTINCT {column_sql}" if distinct else column_sql
    sql = f"{aggregation}({inner})" if aggregation else inner
    return RenderedExpression(
        sql=sql,
        concept=concept,
        concepts=concepts,
        aggregation=aggregation,
    )


def _render_val_unit(
    unit: Any,
    schema: CSSSchema,
    *,
    allow_bare_star: bool = False,
) -> RenderedExpression:
    if not isinstance(unit, list) or len(unit) < 2:
        raise ProjectionRejected("invalid_value_unit")
    operation_id = int(unit[0])
    if operation_id != 0:
        raise ProjectionRejected("arithmetic_value_unit")
    if len(unit) > 2 and unit[2] is not None:
        raise ProjectionRejected("binary_value_unit")
    return _render_col_unit(
        unit[1],
        schema,
        allow_bare_star=allow_bare_star,
    )


def _with_outer_aggregation(
    expression: RenderedExpression,
    aggregation_id: int,
) -> RenderedExpression:
    if aggregation_id < 0 or aggregation_id >= len(AGGREGATIONS):
        raise ProjectionRejected("unsupported_outer_aggregation")
    aggregation = AGGREGATIONS[aggregation_id]
    if not aggregation:
        return expression
    if expression.aggregation:
        raise ProjectionRejected("nested_aggregation")
    if aggregation in {"SUM", "AVG"} and expression.concept not in NUMERIC_CONCEPTS:
        raise ProjectionRejected("non_numeric_aggregation")
    if aggregation in {"MAX", "MIN"} and expression.concept == "star":
        raise ProjectionRejected("aggregate_star_not_supported")
    return RenderedExpression(
        sql=f"{aggregation}({expression.sql})",
        concept=expression.concept,
        concepts=expression.concepts,
        aggregation=aggregation,
    )


def _anchor_for(
    domains: TargetDomains,
    row_sha256: str,
    attempt: int,
) -> Mapping[str, Any]:
    index = _stable_int(PROJECTION_SEED, row_sha256, attempt) % len(domains.joined_rows)
    return domains.joined_rows[index]


def _date_replacement(
    domains: TargetDomains,
    anchor: Mapping[str, Any],
    operator: str,
    position: int,
) -> str:
    anchor_date = str(anchor["visit_date"])
    anchor_index = min(
        range(len(domains.dates)),
        key=lambda index: abs(
            int(domains.dates[index].replace("-", ""))
            - int(anchor_date.replace("-", ""))
        ),
    )
    if operator == "BETWEEN":
        offset = -5 if position == 0 else 5
        return domains.dates[
            min(max(anchor_index + offset, 0), len(domains.dates) - 1)
        ]
    if operator in (">", ">="):
        offset = -1 if operator == ">" else 0
    elif operator in ("<", "<="):
        offset = 1 if operator == "<" else 0
    else:
        offset = 0
    return domains.dates[
        min(max(anchor_index + offset, 0), len(domains.dates) - 1)
    ]


def _numeric_replacement(
    concept: str,
    anchor: Mapping[str, Any],
    operator: str,
    position: int,
    *,
    aggregation: str,
) -> int | float:
    if aggregation == "COUNT" or concept == "star":
        if operator == ">":
            return 0
        if operator == "<":
            return 999
        if operator == "BETWEEN":
            return 0 if position == 0 else 999
        return 1
    raw = anchor.get(concept)
    if raw is None:
        raise ProjectionRejected("empty_target_literal")
    value = float(raw)
    step = 1.0 if concept in {"age", "los_days"} else max(1.0, round(value * 0.05, 2))
    if operator == "BETWEEN":
        value = value - step if position == 0 else value + step
    elif operator == ">":
        value -= step
    elif operator == "<":
        value += step
    if concept in {"age", "los_days"}:
        return max(0, int(round(value)))
    return round(max(0.0, value), 2)


def _replacement_for(
    *,
    expression: RenderedExpression,
    original: Any,
    operator: str,
    position: int,
    anchor: Mapping[str, Any],
    domains: TargetDomains,
) -> Any:
    concept = expression.concept
    if concept in DATE_CONCEPTS:
        if not re.fullmatch(
            r"\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?",
            str(original).strip(),
        ):
            raise ProjectionRejected("source_literal_type_mismatch")
        return _date_replacement(domains, anchor, operator, position)
    if concept in NUMERIC_CONCEPTS or expression.aggregation == "COUNT" or concept == "star":
        if concept in NUMERIC_CONCEPTS:
            try:
                float(original)
            except (TypeError, ValueError) as exc:
                raise ProjectionRejected("source_literal_type_mismatch") from exc
        return _numeric_replacement(
            concept,
            anchor,
            operator,
            position,
            aggregation=expression.aggregation,
        )
    if concept not in anchor:
        raise ProjectionRejected("literal_without_target_domain")
    value = anchor[concept]
    if value is None or str(value) == "":
        raise ProjectionRejected("empty_target_literal")
    return value


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _render_conditions(
    payload: Any,
    schema: CSSSchema,
    *,
    anchor: Mapping[str, Any],
    domains: TargetDomains,
) -> tuple[str, frozenset[str], list[LiteralBinding]]:
    if not payload:
        return "", frozenset(), []
    if not isinstance(payload, list):
        raise ProjectionRejected("invalid_condition_list")
    fragments: list[str] = []
    concepts: set[str] = set()
    bindings: list[LiteralBinding] = []
    expect_condition = True
    for entry in payload:
        if expect_condition:
            if not isinstance(entry, list) or len(entry) < 4:
                raise ProjectionRejected("invalid_condition")
            not_flag = bool(entry[0])
            operator_id = int(entry[1])
            if operator_id <= 0 or operator_id >= len(CONDITION_OPERATORS):
                raise ProjectionRejected("unsupported_condition_operator")
            operator = CONDITION_OPERATORS[operator_id]
            expression = _render_val_unit(entry[2], schema)
            concepts.update(expression.concepts)
            if operator in {"IN", "EXISTS"} or isinstance(entry[3], dict):
                raise ProjectionRejected("nested_condition_not_supported")
            if operator == "BETWEEN":
                if len(entry) < 5 or isinstance(entry[4], (dict, list)):
                    raise ProjectionRejected("invalid_between_condition")
                first = _replacement_for(
                    expression=expression,
                    original=entry[3],
                    operator=operator,
                    position=0,
                    anchor=anchor,
                    domains=domains,
                )
                second = _replacement_for(
                    expression=expression,
                    original=entry[4],
                    operator=operator,
                    position=1,
                    anchor=anchor,
                    domains=domains,
                )
                bindings.extend(
                    [
                        LiteralBinding(entry[3], first),
                        LiteralBinding(entry[4], second),
                    ]
                )
                condition_sql = (
                    f"{expression.sql} BETWEEN {_sql_literal(first)} "
                    f"AND {_sql_literal(second)}"
                )
            else:
                if isinstance(entry[3], list):
                    raise ProjectionRejected("literal_list_not_supported")
                replacement = _replacement_for(
                    expression=expression,
                    original=entry[3],
                    operator=operator,
                    position=0,
                    anchor=anchor,
                    domains=domains,
                )
                bindings.append(LiteralBinding(entry[3], replacement))
                condition_sql = (
                    f"{expression.sql} {operator} {_sql_literal(replacement)}"
                )
            if not_flag:
                condition_sql = f"NOT ({condition_sql})"
            fragments.append(condition_sql)
            expect_condition = False
        else:
            connector = str(entry).strip().upper()
            if connector not in {"AND", "OR"}:
                raise ProjectionRejected("unsupported_condition_connector")
            fragments.append(connector)
            expect_condition = True
    if expect_condition:
        raise ProjectionRejected("trailing_condition_connector")
    return " ".join(fragments), frozenset(concepts), bindings


def _source_literal_variants(value: Any) -> list[str]:
    raw = str(value)
    variants = {raw}
    date_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(?:[ T].*)?", raw)
    if date_match:
        year, month, day = (int(part) for part in date_match.groups())
        chinese_digits = "零一二三四五六七八九"

        def chinese_integer(number: int) -> str:
            if number < 10:
                return chinese_digits[number]
            if number == 10:
                return "十"
            tens, ones = divmod(number, 10)
            prefix = "" if tens == 1 else chinese_digits[tens]
            return f"{prefix}十{chinese_digits[ones] if ones else ''}"

        chinese_year = "".join(chinese_digits[int(digit)] for digit in f"{year:04d}")
        short_year_digits = f"{year % 100:02d}"
        chinese_short_year = "".join(
            chinese_digits[int(digit)] for digit in short_year_digits
        )
        mixed_short_year = "".join(
            digit if digit == "0" else chinese_digits[int(digit)]
            for digit in short_year_digits
        )
        mixed_full_years = {
            "".join(
                digit if mask & (1 << index) else chinese_digits[int(digit)]
                for index, digit in enumerate(f"{year:04d}")
            )
            for mask in range(16)
        }
        chinese_month = chinese_integer(month)
        chinese_day = chinese_integer(day)
        variants.update(
            {
                f"{year}年{month}月{day}日",
                f"{year:04d}年{month:02d}月{day:02d}日",
                f"{year % 100:02d}年{month}月{day}日",
                f"{year % 100:02d}年{month:02d}月{day:02d}日",
                f"{year:04d}-{month}-{day}",
                f"{year % 100:02d}-{month:02d}-{day:02d}",
                f"{year % 100:02d}-{month}-{day}",
                f"{year:04d}.{month}.{day}",
                f"{year % 100:02d}.{month}.{day}",
                f"{year:04d}/{month}/{day}",
                f"{year % 100:02d}/{month}/{day}",
                f"{chinese_year}年{chinese_month}月{chinese_day}日",
                f"{chinese_short_year}年{chinese_month}月{chinese_day}日",
                f"{mixed_short_year}年{chinese_month}月{chinese_day}日",
                *{
                    f"{mixed_year}年{chinese_month}月{chinese_day}日"
                    for mixed_year in mixed_full_years
                },
            }
        )
    if isinstance(value, float) and value.is_integer():
        variants.add(str(int(value)))
    return sorted(variants, key=len, reverse=True)


def _replace_question_literals(
    question: str,
    bindings: Sequence[LiteralBinding],
) -> str:
    result = unicodedata.normalize("NFKC", str(question))
    for binding in bindings:
        replacement = str(binding.replacement)
        replaced = False
        for variant in _source_literal_variants(binding.original):
            if variant and variant in result:
                result = result.replace(variant, replacement)
                replaced = True
        if not replaced:
            raise ProjectionRejected("source_literal_not_found_in_question")
    return re.sub(r"\s+", " ", result).strip()


def _from_sql(concepts: set[str]) -> str:
    uses_patient = bool(concepts & PATIENT_CONCEPTS)
    uses_visit = bool(concepts & VISIT_CONCEPTS)
    if uses_patient and uses_visit:
        return "patients p JOIN visits v ON p.patient_id = v.patient_id"
    if uses_patient:
        return "patients p"
    if uses_visit:
        return "visits v"
    raise ProjectionRejected("query_has_no_mapped_table")


def _category_and_difficulty(
    *,
    concepts: set[str],
    select_expressions: Sequence[RenderedExpression],
    where_sql: str,
    group_sql: str,
    having_sql: str,
    order_sql: str,
    limit: int | None,
) -> tuple[str, str]:
    uses_join = bool(concepts & PATIENT_CONCEPTS) and bool(concepts & VISIT_CONCEPTS)
    has_aggregate = any(expression.aggregation for expression in select_expressions)
    if having_sql:
        category = "having"
    elif order_sql and limit is not None:
        category = "top_k"
    elif "visit_date" in concepts and where_sql:
        category = "temporal"
    elif uses_join:
        category = "multi_table_join"
    elif group_sql or has_aggregate:
        category = "aggregation_grouping"
    else:
        category = "filtering"
    score = (
        (1 if uses_join else 0)
        + (1 if group_sql else 0)
        + (1 if having_sql else 0)
        + (1 if order_sql else 0)
        + (1 if limit is not None else 0)
        + (1 if has_aggregate else 0)
    )
    if score >= 5:
        difficulty = "extra-hard"
    elif score >= 3:
        difficulty = "extra"
    elif score >= 1:
        difficulty = "hard"
    else:
        difficulty = "medium"
    return category, difficulty


def _validate_question_semantics(
    question: str,
    select_expressions: Sequence[RenderedExpression],
) -> None:
    normalized = unicodedata.normalize("NFKC", question)
    # These nouns denote entities/attributes absent from the four-table
    # destination.  Keeping such a question while dropping the corresponding
    # source column would create a fluent but false benchmark item.
    unsupported = re.compile(
        r"报告单|报告号|报告编号|仪器(?:编号|名称)?|医生|医师|床位|"
        r"医疗机构|医院(?:编号|编码)|票据|处方(?:编号|编码)"
    )
    if unsupported.search(normalized):
        raise ProjectionRejected("unsupported_question_semantics")
    for expression in select_expressions:
        concept = expression.concept
        if concept == "cost" and not re.search(
            r"费用|医疗费|金额|花费|结算|支付|多少钱|成本",
            normalized,
        ):
            raise ProjectionRejected("selected_concept_not_grounded")
        if concept == "los_days" and not re.search(
            r"住院|天数|时长|多少天",
            normalized,
        ):
            raise ProjectionRejected("selected_concept_not_grounded")
        if concept == "visit_date" and not re.search(
            r"日期|时间|哪天|几号|何时|什么时候|来就诊",
            normalized,
        ):
            raise ProjectionRejected("selected_concept_not_grounded")


def _normalise_select_semantics(
    question: str,
    expression: RenderedExpression,
) -> RenderedExpression:
    if (
        expression.aggregation == "COUNT"
        and expression.concept == "patient_id"
        and "DISTINCT" not in expression.sql.upper()
        and re.search(
            r"人数|多少人|多少.{0,12}人|人.{0,12}多少|"
            r"多少(?:名|位)?患者|患者(?:一共|总共|共有|有)?多少",
            question,
        )
    ):
        return RenderedExpression(
            sql=f"COUNT(DISTINCT {TARGET_COLUMNS['patient_id']})",
            concept=expression.concept,
            concepts=expression.concepts,
            aggregation=expression.aggregation,
        )
    return expression


def _render_flat_query(
    sql: Mapping[str, Any],
    schema: CSSSchema,
    *,
    question: str,
    anchor: Mapping[str, Any],
    domains: TargetDomains,
) -> tuple[str, list[LiteralBinding], str, str]:
    if any(sql.get(name) is not None for name in ("intersect", "union", "except")):
        raise ProjectionRejected("set_operation_not_supported")
    from_payload = sql.get("from")
    if not isinstance(from_payload, Mapping):
        raise ProjectionRejected("missing_from_annotation")
    table_units = from_payload.get("table_units", [])
    if any(unit and str(unit[0]).lower() != "table_unit" for unit in table_units):
        raise ProjectionRejected("nested_from_not_supported")

    distinct, select_payload = _normalise_select(sql.get("select"))
    select_expressions: list[RenderedExpression] = []
    concepts: set[str] = set()
    for item in select_payload:
        if not isinstance(item, list) or len(item) != 2:
            raise ProjectionRejected("invalid_select_item")
        expression = _with_outer_aggregation(
            _render_val_unit(
                item[1],
                schema,
                allow_bare_star=bool(int(item[0])),
            ),
            int(item[0]),
        )
        expression = _normalise_select_semantics(question, expression)
        select_expressions.append(expression)
        concepts.update(expression.concepts)
    if not select_expressions:
        raise ProjectionRejected("empty_select")
    _validate_question_semantics(question, select_expressions)

    where_sql, where_concepts, where_bindings = _render_conditions(
        sql.get("where"),
        schema,
        anchor=anchor,
        domains=domains,
    )
    concepts.update(where_concepts)

    group_expressions = [
        _render_col_unit(unit, schema) for unit in (sql.get("groupBy") or [])
    ]
    if any(expression.aggregation for expression in group_expressions):
        raise ProjectionRejected("aggregate_group_key")
    for expression in group_expressions:
        concepts.update(expression.concepts)
    group_sql = ", ".join(expression.sql for expression in group_expressions)

    having_sql, having_concepts, having_bindings = _render_conditions(
        sql.get("having"),
        schema,
        anchor=anchor,
        domains=domains,
    )
    concepts.update(having_concepts)

    order_payload = sql.get("orderBy") or []
    order_sql = ""
    if order_payload:
        if (
            not isinstance(order_payload, list)
            or len(order_payload) != 2
            or str(order_payload[0]).lower() not in {"asc", "desc"}
        ):
            raise ProjectionRejected("invalid_order_by")
        order_expressions = [
            _render_val_unit(unit, schema) for unit in order_payload[1]
        ]
        for expression in order_expressions:
            concepts.update(expression.concepts)
        order_sql = ", ".join(
            f"{expression.sql} {str(order_payload[0]).upper()}"
            for expression in order_expressions
        )

    limit_value = sql.get("limit")
    limit: int | None = None
    limit_bindings: list[LiteralBinding] = []
    if limit_value is not None:
        limit = max(1, min(int(limit_value), 10))
        limit_bindings.append(LiteralBinding(limit_value, limit))

    from_sql = _from_sql(concepts)
    select_sql = ", ".join(expression.sql for expression in select_expressions)
    query = f"SELECT {'DISTINCT ' if distinct else ''}{select_sql} FROM {from_sql}"
    if where_sql:
        query += f" WHERE {where_sql}"
    if group_sql:
        query += f" GROUP BY {group_sql}"
    if having_sql:
        query += f" HAVING {having_sql}"
    if order_sql:
        query += f" ORDER BY {order_sql}"
    if limit is not None:
        query += f" LIMIT {limit}"
    category, difficulty = _category_and_difficulty(
        concepts=concepts,
        select_expressions=select_expressions,
        where_sql=where_sql,
        group_sql=group_sql,
        having_sql=having_sql,
        order_sql=order_sql,
        limit=limit,
    )
    return (
        query,
        [*where_bindings, *having_bindings, *limit_bindings],
        category,
        difficulty,
    )


def project_css_row(
    row: Mapping[str, Any],
    schema: CSSSchema,
    domains: TargetDomains,
    db_path: Path,
    *,
    source_split: str,
    required_role: str | None = None,
    max_attempts: int = 48,
    partition_version: str = "v1",
) -> ProjectedItem:
    """Project one public CSS row or raise :class:`ProjectionRejected`."""
    if source_split not in {"train", "dev", "test"}:
        raise ValueError("source_split must be train, dev, or test")
    if required_role not in {None, "development", "holdout"}:
        raise ValueError("required_role must be development, holdout, or None")
    if partition_version not in {"v1", "v2"}:
        raise ValueError("partition_version must be v1 or v2")
    if str(row.get("db_id")) != schema.db_id:
        raise ProjectionRejected("schema_id_mismatch")
    question = str(row.get("question", "")).strip()
    if not question:
        raise ProjectionRejected("empty_question")
    sql_annotation = row.get("sql")
    if not isinstance(sql_annotation, Mapping):
        raise ProjectionRejected("missing_structured_sql")
    source_sha = _source_row_sha256(row)
    last_reason = "no_nonempty_projection"
    for attempt in range(max_attempts):
        try:
            anchor = _anchor_for(domains, source_sha, attempt)
            gold_sql, bindings, category, difficulty = _render_flat_query(
                sql_annotation,
                schema,
                question=question,
                anchor=anchor,
                domains=domains,
            )
            projected_question = _replace_question_literals(question, bindings)
            skeleton = validate_gold_sql(gold_sql)
            role = (
                skeleton_role_v2(skeleton)
                if partition_version == "v2"
                else skeleton_role(skeleton)
            )
            if required_role is not None and role != required_role:
                raise ProjectionRejected("skeleton_reserved_for_other_role")
            validate_question_sql_grounding(projected_question, gold_sql)
            _, rows = execute_complete_readonly(gold_sql, db_path)
            if not rows:
                raise ProjectionRejected("empty_result")
            normalized = normalise_question(projected_question)
            projection_seed = (
                PROJECTION_SEED_V2
                if partition_version == "v2"
                else PROJECTION_SEED
            )
            identity = hashlib.sha256(
                (
                    f"{projection_seed}\0{source_split}\0{source_sha}\0"
                    f"{normalized}\0{gold_sql}"
                ).encode("utf-8")
            ).hexdigest()
            item = {
                "id": f"css-{source_split}-{identity[:20]}",
                "question": projected_question,
                "gold_sql": gold_sql,
                "difficulty": difficulty,
                "category": category,
                "provenance": {
                    "benchmark": "CSS",
                    "source_split": source_split,
                    "source_db_id_sha256": hashlib.sha256(
                        schema.db_id.encode("utf-8")
                    ).hexdigest(),
                    "source_question_id_sha256": hashlib.sha256(
                        str(row.get("question_id", "")).encode("utf-8")
                    ).hexdigest(),
                    "source_row_sha256": source_sha,
                    "projection_seed": projection_seed,
                    "partition_version": partition_version,
                    "literal_attempt": attempt,
                    "skeleton_sha256": hashlib.sha256(
                        skeleton.encode("utf-8")
                    ).hexdigest(),
                    "skeleton_role": role,
                },
            }
            return ProjectedItem(
                item=item,
                skeleton_role=role,
                skeleton_sha256=item["provenance"]["skeleton_sha256"],
            )
        except ProjectionRejected as exc:
            last_reason = exc.reason
            if exc.reason == "skeleton_reserved_for_other_role":
                raise
            if exc.reason not in {
                "empty_result",
                "empty_target_literal",
                "source_literal_not_found_in_question",
            }:
                raise
        except (RuntimeError, ValueError, sqlite3.Error) as exc:
            last_reason = f"validation_or_execution:{type(exc).__name__}"
    raise ProjectionRejected(last_reason)


def deduplicate_projected_items(
    items: Iterable[ProjectedItem],
) -> list[ProjectedItem]:
    by_question: dict[str, ProjectedItem] = {}
    by_id: dict[str, ProjectedItem] = {}
    for projected in items:
        item = projected.item
        normalized = normalise_question(str(item["question"]))
        if normalized in by_question:
            continue
        item_id = str(item["id"])
        if item_id in by_id:
            raise ValueError(f"duplicate projected item id: {item_id}")
        by_question[normalized] = projected
        by_id[item_id] = projected
    return sorted(
        by_id.values(),
        key=lambda projected: (
            projected.item["difficulty"],
            projected.item["category"],
            projected.item["id"],
        ),
    )
