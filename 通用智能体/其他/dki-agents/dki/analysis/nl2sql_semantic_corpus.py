"""Deterministic semantic-family corpus construction for the four-table NL2SQL schema.

The builder is deliberately separate from the runtime, evaluator, and sealing
code.  It reads domains from one live SQLite database, renders only queries over
``patients``, ``visits``, ``kg_entities``, and ``kg_triples``, validates every
query as read-only SQL, consumes every result completely, and partitions whole
families into train and OOD development splits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.analysis.nl2sql_sqlite import bounded_execution, connect_readonly


SCHEMA_VERSION = 1
DEFAULT_PARTITION_SEED = "track-c-semantic-families-v1"
DEFAULT_DEV_FRACTION = 0.25

REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "patients": ("patient_id", "name", "gender", "age", "region"),
    "visits": (
        "visit_id",
        "patient_id",
        "visit_date",
        "department",
        "disease",
        "drug_prescribed",
        "exam",
        "cost",
        "los_days",
    ),
    "kg_entities": ("name", "type", "mention_count"),
    "kg_triples": (
        "head",
        "head_type",
        "relation",
        "relation_zh",
        "tail",
        "tail_type",
        "confidence",
        "count",
    ),
}

REQUIRED_COVERAGE = frozenset(
    {
        "count_vs_distinct",
        "join_amplification",
        "zero_match_preservation",
        "where_having",
        "cte_propagation",
        "set_duplicate_semantics",
        "windows_ties",
        "dates",
        "null_anti_join",
        "output_order_precision",
    }
)


@dataclass(frozen=True)
class _Draft:
    template_id: str
    variant_id: str
    question: str
    gold_sql: str
    semantic_family: str
    join_path: tuple[str, ...]
    near_duplicate_family: str
    coverage: tuple[str, ...]
    domain_bindings: Mapping[str, Any]
    mutation_profiles: tuple[str, ...]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite SQL literals are forbidden")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _usable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 96:
        return None
    if any(character in text for character in ("\n", "\r", "\x00", "「", "」")):
        return None
    return text


def _query_all(connection, sql: str, parameters: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
    with bounded_execution(connection):
        return [tuple(row) for row in connection.execute(sql, tuple(parameters)).fetchall()]


def inspect_live_schema(db_path: Path) -> dict[str, tuple[str, ...]]:
    """Validate and return the required live four-table schema."""

    connection = connect_readonly(Path(db_path))
    try:
        observed: dict[str, tuple[str, ...]] = {}
        for table, required_columns in REQUIRED_SCHEMA.items():
            rows = _query_all(connection, f'PRAGMA table_info("{table}")')
            columns = tuple(str(row[1]) for row in rows)
            if not columns:
                raise ValueError(f"required NL2SQL table is missing: {table}")
            missing = sorted(set(required_columns) - set(columns))
            if missing:
                raise ValueError(
                    f"required columns are missing from {table}: {', '.join(missing)}"
                )
            observed[table] = columns
        return observed
    finally:
        connection.close()


def _text_domain(connection, table: str, column: str) -> tuple[str, ...]:
    if table not in REQUIRED_SCHEMA or column not in REQUIRED_SCHEMA[table]:
        raise ValueError(f"domain identifier is outside the four-table schema: {table}.{column}")
    rows = _query_all(
        connection,
        f'SELECT DISTINCT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL '
        f'ORDER BY CAST("{column}" AS TEXT) COLLATE BINARY',
    )
    values: list[str] = []
    for row in rows:
        value = _usable_text(row[0])
        if value is not None and value not in values:
            values.append(value)
    return tuple(values)


def _joined_domain(connection, column: str) -> tuple[str, ...]:
    if column not in {"region", "gender"}:
        raise ValueError(f"unsupported joined patient domain: {column}")
    rows = _query_all(
        connection,
        f'SELECT DISTINCT p."{column}" FROM patients AS p '
        "JOIN visits AS v ON v.patient_id = p.patient_id "
        f'WHERE p."{column}" IS NOT NULL '
        f'ORDER BY CAST(p."{column}" AS TEXT) COLLATE BINARY',
    )
    values: list[str] = []
    for row in rows:
        value = _usable_text(row[0])
        if value is not None and value not in values:
            values.append(value)
    return tuple(values)


def _first_values(values: Sequence[str], limit: int = 2) -> tuple[str, ...]:
    return tuple(values[: max(1, int(limit))])


def _require_values(label: str, values: Sequence[str], minimum: int = 1) -> tuple[str, ...]:
    if len(values) < minimum:
        raise ValueError(
            f"live database needs at least {minimum} usable {label} value(s); found {len(values)}"
        )
    return tuple(values)


def _having_threshold(connection, department: str) -> int:
    rows = _query_all(
        connection,
        "SELECT COUNT(*) AS n FROM visits "
        "WHERE department = ? AND disease IS NOT NULL "
        "GROUP BY disease ORDER BY n ASC, CAST(disease AS TEXT) COLLATE BINARY",
        (department,),
    )
    if not rows:
        raise ValueError(f"department has no non-NULL disease groups: {department!r}")
    threshold = int(rows[0][0])
    if threshold <= 0:
        raise ValueError("derived HAVING threshold must be positive")
    return threshold


def _date_pairs(dates: Sequence[str]) -> tuple[tuple[str, str], ...]:
    if not dates:
        return ()
    if len(dates) == 1:
        return ((dates[0], dates[0]),)
    candidates = [
        (dates[0], dates[len(dates) // 2]),
        (dates[max(0, len(dates) // 2 - 1)], dates[-1]),
    ]
    result: list[tuple[str, str]] = []
    for pair in candidates:
        if pair not in result:
            result.append(pair)
    return tuple(result)


def _normalise_ast_argument(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    return re.sub(r"\s+", "_", text)


def build_ast_metadata(sql: str) -> dict[str, Any]:
    """Return a literal-free skeleton plus stable structural family features."""

    skeleton = validate_gold_sql(sql)
    try:
        from sqlglot import exp, parse_one
    except ImportError as exc:
        raise RuntimeError("sqlglot is required for Track C AST metadata") from exc

    tree = parse_one(sql, read="sqlite")
    aggregate_types: list[str] = []
    aggregate_classes = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
    for node in tree.walk():
        if not isinstance(node, aggregate_classes):
            continue
        label = node.key.upper()
        if isinstance(node, exp.Count) and node.find(exp.Distinct) is not None:
            label = "COUNT_DISTINCT"
        if label not in aggregate_types:
            aggregate_types.append(label)

    join_types: list[str] = []
    for join in tree.find_all(exp.Join):
        side = _normalise_ast_argument(join.args.get("side"))
        kind = _normalise_ast_argument(join.args.get("kind"))
        label = "_".join(part for part in (side, kind, "join") if part) or "join"
        join_types.append(label)

    window_functions: list[str] = []
    for window in tree.find_all(exp.Window):
        function = getattr(window.this, "key", type(window.this).__name__)
        label = str(function).upper()
        if label not in window_functions:
            window_functions.append(label)

    if isinstance(tree, exp.Union):
        set_root = "union" if tree.args.get("distinct") is not False else "union_all"
    elif isinstance(tree, exp.Intersect):
        set_root = "intersect"
    elif isinstance(tree, exp.Except):
        set_root = "except"
    else:
        set_root = "none"

    select_distinct = any(
        select.args.get("distinct") is not None for select in tree.find_all(exp.Select)
    )
    features = {
        "root": tree.key.upper(),
        "set_root": set_root,
        "cte_count": sum(1 for _ in tree.find_all(exp.CTE)),
        "join_types": join_types,
        "aggregate_types": aggregate_types,
        "window_functions": window_functions,
        "has_group_by": tree.find(exp.Group) is not None,
        "has_having": tree.find(exp.Having) is not None,
        "has_order_by": tree.find(exp.Order) is not None,
        "has_limit": tree.find(exp.Limit) is not None,
        "has_correlated_subquery": "NOT EXISTS" in skeleton or "EXISTS" in skeleton,
        "select_distinct": select_distinct,
    }
    family = ";".join(
        (
            f"root={features['root']}",
            f"set={features['set_root']}",
            f"cte={features['cte_count']}",
            "joins=" + (",".join(join_types) or "none"),
            "aggregates=" + (",".join(aggregate_types) or "none"),
            "windows=" + (",".join(window_functions) or "none"),
            f"group={int(features['has_group_by'])}",
            f"having={int(features['has_having'])}",
            f"order={int(features['has_order_by'])}",
            f"distinct={int(select_distinct)}",
        )
    )
    return {
        "skeleton": skeleton,
        "skeleton_sha256": _sha256_text(skeleton),
        "family": family,
        "family_sha256": _sha256_text(family),
        "features": features,
    }


def _typed_result_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "inf" if value > 0 else "-inf"
        else:
            encoded = value.hex()
        return {"type": "float", "value": encoded}
    return {"type": "text", "value": str(value)}


def result_fingerprint(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Hash a complete SQLite result, preserving column and row order and types."""

    payload = {
        "columns": [str(column) for column in columns],
        "rows": [
            [_typed_result_value(value) for value in row]
            for row in rows
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def execute_with_fingerprint(sql: str, db_path: Path) -> dict[str, Any]:
    """Completely execute one read-only query and return deterministic evidence."""

    columns, rows = execute_complete_readonly(sql, Path(db_path))
    return {
        "complete": True,
        "readonly": True,
        "columns": [str(column) for column in columns],
        "row_count": len(rows),
        "result_sha256": result_fingerprint(columns, rows),
    }


def _build_drafts(db_path: Path) -> list[_Draft]:
    inspect_live_schema(db_path)
    connection = connect_readonly(Path(db_path))
    try:
        departments = _require_values(
            "department", _text_domain(connection, "visits", "department"), 2
        )
        regions = _require_values("region", _text_domain(connection, "patients", "region"))
        joined_regions = _require_values("joined region", _joined_domain(connection, "region"))
        joined_genders = _require_values("joined gender", _joined_domain(connection, "gender"))
        entity_types = _require_values(
            "knowledge-graph entity type", _text_domain(connection, "kg_entities", "type")
        )
        relation_labels = _require_values(
            "knowledge-graph relation label",
            _text_domain(connection, "kg_triples", "relation_zh"),
        )
        raw_dates = _text_domain(connection, "visits", "visit_date")
        dates = _require_values(
            "ISO visit date",
            tuple(value for value in raw_dates if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)),
        )

        drafts: list[_Draft] = []

        def add(
            *,
            template_id: str,
            variant_id: str,
            question: str,
            gold_sql: str,
            semantic_family: str,
            join_path: Iterable[str] = (),
            near_duplicate_family: str,
            coverage: Iterable[str],
            domain_bindings: Mapping[str, Any],
            mutation_profiles: Iterable[str],
        ) -> None:
            drafts.append(
                _Draft(
                    template_id=template_id,
                    variant_id=variant_id,
                    question=question,
                    gold_sql=gold_sql,
                    semantic_family=semantic_family,
                    join_path=tuple(join_path),
                    near_duplicate_family=near_duplicate_family,
                    coverage=tuple(sorted(set(coverage))),
                    domain_bindings=dict(domain_bindings),
                    mutation_profiles=tuple(mutation_profiles),
                )
            )

        for index, department in enumerate(_first_values(departments), start=1):
            literal = _sql_literal(department)
            add(
                template_id="visit_count_cardinality",
                variant_id=f"department-{index}",
                question=(
                    f"在科室「{department}」的就诊中，严格按顺序返回就诊记录数和去重患者数。"
                ),
                gold_sql=(
                    "SELECT COUNT(*) AS visit_rows, "
                    "COUNT(DISTINCT patient_id) AS distinct_patients "
                    f"FROM visits WHERE department = {literal}"
                ),
                semantic_family="count_rows_vs_distinct_patients",
                near_duplicate_family="visit_count_cardinality",
                coverage=("count_vs_distinct", "output_order_precision"),
                domain_bindings={"department": department},
                mutation_profiles=("remove_distinct",),
            )

        for index, region in enumerate(_first_values(joined_regions), start=1):
            literal = _sql_literal(region)
            add(
                template_id="join_amplification",
                variant_id=f"region-{index}",
                question=(
                    f"对地区「{region}」的患者做患者—就诊内连接，严格按顺序返回"
                    "去重患者数和连接后的就诊行数。"
                ),
                gold_sql=(
                    "SELECT COUNT(DISTINCT p.patient_id) AS distinct_patients, "
                    "COUNT(v.visit_id) AS joined_visit_rows "
                    "FROM patients AS p JOIN visits AS v "
                    "ON v.patient_id = p.patient_id "
                    f"WHERE p.region = {literal}"
                ),
                semantic_family="join_amplification_cardinality",
                join_path=("patients.patient_id--inner-->visits.patient_id",),
                near_duplicate_family="join_count_cardinality",
                coverage=("count_vs_distinct", "join_amplification"),
                domain_bindings={"region": region},
                mutation_profiles=("remove_distinct",),
            )

        for index, region in enumerate(_first_values(regions), start=1):
            literal = _sql_literal(region)
            add(
                template_id="patient_zero_match",
                variant_id=f"region-{index}",
                question=(
                    f"列出地区「{region}」的每位患者编号、姓名和就诊次数；没有就诊的患者"
                    "也必须保留为0次，并按患者编号升序。"
                ),
                gold_sql=(
                    "SELECT p.patient_id, p.name, COUNT(v.visit_id) AS visit_count "
                    "FROM patients AS p LEFT JOIN visits AS v "
                    "ON v.patient_id = p.patient_id "
                    f"WHERE p.region = {literal} "
                    "GROUP BY p.patient_id, p.name "
                    "ORDER BY p.patient_id ASC"
                ),
                semantic_family="patient_zero_match_preservation",
                join_path=("patients.patient_id--left-->visits.patient_id",),
                near_duplicate_family="patient_zero_match_by_region",
                coverage=("zero_match_preservation",),
                domain_bindings={"region": region},
                mutation_profiles=("left_join_to_inner_patient",),
            )

        for index, department in enumerate(_first_values(departments), start=1):
            threshold = _having_threshold(connection, department)
            literal = _sql_literal(department)
            add(
                template_id="where_having",
                variant_id=f"department-{index}",
                question=(
                    f"先在 WHERE 中筛选科室「{department}」，再按主诊断分组，并在 HAVING "
                    f"中仅保留就诊次数至少{threshold}次的主诊断；返回主诊断和就诊次数，"
                    "次数降序、主诊断升序。"
                ),
                gold_sql=(
                    "SELECT disease, COUNT(*) AS visit_count FROM visits "
                    f"WHERE department = {literal} GROUP BY disease "
                    f"HAVING COUNT(*) >= {threshold} "
                    "ORDER BY visit_count DESC, disease ASC"
                ),
                semantic_family="where_before_group_having_after_group",
                near_duplicate_family="where_having_department_counts",
                coverage=("where_having",),
                domain_bindings={"department": department, "threshold": threshold},
                mutation_profiles=("having_boundary_strictened",),
            )

        for index, department in enumerate(_first_values(departments), start=1):
            literal = _sql_literal(department)
            add(
                template_id="cte_filter_propagation",
                variant_id=f"department-{index}",
                question=(
                    f"用 CTE 先筛选科室「{department}」的就诊并传递患者编号和费用，外层按患者"
                    "汇总，严格依次返回患者编号和四舍五入到2位的小计费用，并按小计费用降序、"
                    "患者编号升序。"
                ),
                gold_sql=(
                    "WITH scoped_visits AS ("
                    "SELECT patient_id, cost FROM visits "
                    f"WHERE department = {literal}) "
                    "SELECT patient_id, ROUND(SUM(cost), 2) AS total_cost "
                    "FROM scoped_visits GROUP BY patient_id "
                    "ORDER BY total_cost DESC, patient_id ASC"
                ),
                semantic_family="cte_filter_propagation",
                near_duplicate_family="cte_department_costs",
                coverage=("cte_propagation", "output_order_precision"),
                domain_bindings={"department": department},
                mutation_profiles=("drop_cte_filter",),
            )

        left_department, right_department = departments[0], departments[1]
        left_literal = _sql_literal(left_department)
        right_literal = _sql_literal(right_department)
        set_bindings = {
            "filter_column": "department",
            "left_value": left_department,
            "right_value": right_department,
        }
        add(
            template_id="set_union_all",
            variant_id="preserve-duplicates",
            question=(
                f"将科室「{left_department}」与「{right_department}」的患者编号用 UNION ALL "
                "合并，保留所有重复行，返回一列患者编号并按患者编号升序。"
            ),
            gold_sql=(
                "SELECT patient_id FROM visits "
                f"WHERE department = {left_literal} UNION ALL "
                "SELECT patient_id FROM visits "
                f"WHERE department = {right_literal} ORDER BY patient_id ASC"
            ),
            semantic_family="set_bag_union_preserves_duplicates",
            near_duplicate_family="set_union_duplicate_pair",
            coverage=("set_duplicate_semantics",),
            domain_bindings=set_bindings,
            mutation_profiles=("union_all_to_union",),
        )
        add(
            template_id="set_union_distinct",
            variant_id="remove-duplicates",
            question=(
                f"将科室「{left_department}」与「{right_department}」的患者编号用 UNION 合并，"
                "删除重复行，返回一列患者编号并按患者编号升序。"
            ),
            gold_sql=(
                "SELECT patient_id FROM visits "
                f"WHERE department = {left_literal} UNION "
                "SELECT patient_id FROM visits "
                f"WHERE department = {right_literal} ORDER BY patient_id ASC"
            ),
            semantic_family="set_union_removes_duplicates",
            near_duplicate_family="set_union_duplicate_pair",
            coverage=("set_duplicate_semantics",),
            domain_bindings=set_bindings,
            mutation_profiles=("union_to_union_all",),
        )

        for index, department in enumerate(_first_values(departments), start=1):
            literal = _sql_literal(department)
            add(
                template_id="window_rank_ties",
                variant_id=f"department-{index}",
                question=(
                    f"在科室「{department}」内按费用从高到低排名；费用并列必须使用 RANK 共享"
                    "名次。严格依次返回就诊编号、科室、费用和名次，并按名次、就诊编号升序。"
                ),
                gold_sql=(
                    "SELECT visit_id, department, cost, "
                    "RANK() OVER (PARTITION BY department ORDER BY cost DESC) AS cost_rank "
                    f"FROM visits WHERE department = {literal} "
                    "ORDER BY cost_rank ASC, visit_id ASC"
                ),
                semantic_family="window_rank_preserves_ties",
                near_duplicate_family="window_rank_by_department",
                coverage=("windows_ties", "output_order_precision"),
                domain_bindings={"department": department},
                mutation_profiles=("rank_to_row_number",),
            )

        for index, (start_date, end_date) in enumerate(_date_pairs(dates), start=1):
            add(
                template_id="inclusive_date_range",
                variant_id=f"range-{index}",
                question=(
                    f"统计从{start_date}到{end_date}（含首尾）的每日就诊次数；严格依次返回"
                    "就诊日期和次数，并按日期升序。"
                ),
                gold_sql=(
                    "SELECT visit_date, COUNT(*) AS visit_count FROM visits "
                    f"WHERE visit_date >= {_sql_literal(start_date)} "
                    f"AND visit_date <= {_sql_literal(end_date)} "
                    "GROUP BY visit_date ORDER BY visit_date ASC"
                ),
                semantic_family="inclusive_date_boundaries",
                near_duplicate_family="inclusive_date_ranges",
                coverage=("dates",),
                domain_bindings={"start_date": start_date, "end_date": end_date},
                mutation_profiles=("exclude_date_upper_bound",),
            )

        add(
            template_id="null_safe_anti_join",
            variant_id="patients-without-visits",
            question=(
                "列出没有任何就诊记录的患者编号；即使就诊表的患者编号存在 NULL，也必须正确"
                "保留未就诊患者，并按患者编号升序。"
            ),
            gold_sql=(
                "SELECT p.patient_id FROM patients AS p "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id) "
                "ORDER BY p.patient_id ASC"
            ),
            semantic_family="null_safe_anti_join",
            join_path=("patients.patient_id--anti_exists-->visits.patient_id",),
            near_duplicate_family="patient_anti_join_null_semantics",
            coverage=("null_anti_join", "zero_match_preservation"),
            domain_bindings={},
            mutation_profiles=("not_exists_to_not_in",),
        )

        for index, gender in enumerate(_first_values(joined_genders), start=1):
            literal = _sql_literal(gender)
            add(
                template_id="regional_output_contract",
                variant_id=f"gender-{index}",
                question=(
                    f"对于性别「{gender}」且有就诊的患者，按地区汇总；严格依次返回地区、"
                    "四舍五入到2位的平均费用、去重患者数，并按平均费用降序、地区升序。"
                ),
                gold_sql=(
                    "SELECT p.region AS region, "
                    "ROUND(AVG(v.cost), 2) AS average_cost, "
                    "COUNT(DISTINCT p.patient_id) AS distinct_patients "
                    "FROM patients AS p JOIN visits AS v "
                    "ON v.patient_id = p.patient_id "
                    f"WHERE p.gender = {literal} GROUP BY p.region "
                    "ORDER BY average_cost DESC, p.region ASC"
                ),
                semantic_family="ordered_precise_regional_summary",
                join_path=("patients.patient_id--inner-->visits.patient_id",),
                near_duplicate_family="regional_output_contract_by_gender",
                coverage=(
                    "count_vs_distinct",
                    "join_amplification",
                    "output_order_precision",
                ),
                domain_bindings={"gender": gender, "precision": 2},
                mutation_profiles=(
                    "reduce_round_precision",
                    "swap_projection_order",
                    "reverse_primary_sort",
                ),
            )

        for index, relation_label in enumerate(_first_values(relation_labels), start=1):
            literal = _sql_literal(relation_label)
            add(
                template_id="kg_row_observation_counts",
                variant_id=f"relation-{index}",
                question=(
                    f"对中文关系「{relation_label}」，严格依次返回三元组行数和观察次数总和。"
                ),
                gold_sql=(
                    "SELECT COUNT(*) AS triple_rows, "
                    "SUM(count) AS observation_total FROM kg_triples "
                    f"WHERE relation_zh = {literal}"
                ),
                semantic_family="kg_rows_vs_observation_sum",
                near_duplicate_family="kg_relation_count_measures",
                coverage=("output_order_precision",),
                domain_bindings={"relation_zh": relation_label},
                mutation_profiles=(),
            )

        for index, entity_type in enumerate(_first_values(entity_types), start=1):
            literal = _sql_literal(entity_type)
            add(
                template_id="kg_tail_zero_match",
                variant_id=f"type-{index}",
                question=(
                    f"对类型「{entity_type}」的知识图谱实体，列出实体名称和作为尾实体出现的次数；"
                    "从未作为尾实体出现的实体也保留为0次，并按次数降序、名称升序。"
                ),
                gold_sql=(
                    "SELECT e.name, COUNT(k.tail) AS tail_occurrences "
                    "FROM kg_entities AS e LEFT JOIN kg_triples AS k "
                    "ON k.tail = e.name "
                    f"WHERE e.type = {literal} GROUP BY e.name "
                    "ORDER BY tail_occurrences DESC, e.name ASC"
                ),
                semantic_family="kg_entity_zero_match_preservation",
                join_path=("kg_entities.name--left-->kg_triples.tail",),
                near_duplicate_family="kg_tail_coverage_by_type",
                coverage=("zero_match_preservation",),
                domain_bindings={"entity_type": entity_type},
                mutation_profiles=("left_join_to_inner_kg",),
            )
        return drafts
    finally:
        connection.close()


def _materialise_drafts(db_path: Path, drafts: Sequence[_Draft]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ordinal, draft in enumerate(drafts, start=1):
        validate_question_sql_grounding(draft.question, draft.gold_sql)
        ast = build_ast_metadata(draft.gold_sql)
        execution = execute_with_fingerprint(draft.gold_sql, db_path)
        records.append(
            {
                "id": f"track-c-{ordinal:04d}-{draft.template_id}-{draft.variant_id}",
                "question": draft.question,
                "gold_sql": draft.gold_sql,
                "semantic_family": draft.semantic_family,
                "join_path": list(draft.join_path),
                "ast": ast,
                "family_metadata": {
                    "template_id": draft.template_id,
                    "variant_id": draft.variant_id,
                    "near_duplicate_family": draft.near_duplicate_family,
                    "coverage": list(draft.coverage),
                    "domain_bindings": dict(draft.domain_bindings),
                    "mutation_profiles": list(draft.mutation_profiles),
                },
                "execution": execution,
            }
        )
    return records


def _family_values(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    metadata = record["family_metadata"]
    values: list[tuple[str, str]] = [
        ("semantic", str(record["semantic_family"])),
        ("skeleton", str(record["ast"]["skeleton_sha256"])),
        ("near_duplicate", str(metadata["near_duplicate_family"])),
    ]
    join_path = tuple(str(item) for item in record.get("join_path", ()))
    if join_path:
        values.append(
            (
                "join_path",
                json.dumps(join_path, ensure_ascii=False, separators=(",", ":")),
            )
        )
    return tuple(values)


def _partition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dev_fraction: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < float(dev_fraction) < 1:
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    if len(records) < 2:
        raise ValueError("family partitioning requires at least two corpus rows")

    disjoint = _DisjointSet(len(records))
    owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for family in _family_values(record):
            previous = owner.setdefault(family, index)
            disjoint.union(index, previous)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[disjoint.find(index)].append(index)
    if len(components) < 2:
        raise ValueError(
            "all semantic/join/skeleton/near-duplicate families collapsed into one component"
        )

    component_rows: list[tuple[str, list[int]]] = []
    for indexes in components.values():
        member_ids = sorted(str(records[index]["id"]) for index in indexes)
        component_id = _sha256_text("\n".join(member_ids))
        component_rows.append((component_id, sorted(indexes)))
    component_rows.sort(
        key=lambda item: (_sha256_text(f"{seed}\0{item[0]}"), item[0])
    )

    target_dev_rows = max(1, int(round(len(records) * float(dev_fraction))))
    selected_dev_components: set[str] = set()
    selected_rows = 0
    for position, (component_id, indexes) in enumerate(component_rows):
        components_remaining = len(component_rows) - position - 1
        if selected_rows >= target_dev_rows or components_remaining < 1:
            break
        selected_dev_components.add(component_id)
        selected_rows += len(indexes)
    if not selected_dev_components:
        selected_dev_components.add(component_rows[0][0])
    if len(selected_dev_components) == len(component_rows):
        selected_dev_components.remove(component_rows[-1][0])

    component_for_index = {
        index: component_id
        for component_id, indexes in component_rows
        for index in indexes
    }
    train: list[dict[str, Any]] = []
    ood_dev: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        row = dict(source)
        metadata = dict(row["family_metadata"])
        component_id = component_for_index[index]
        metadata["partition_component"] = component_id
        row["family_metadata"] = metadata
        if component_id in selected_dev_components:
            row["split"] = "ood_dev"
            ood_dev.append(row)
        else:
            row["split"] = "train"
            train.append(row)

    train.sort(key=lambda row: str(row["id"]))
    ood_dev.sort(key=lambda row: str(row["id"]))
    if not train or not ood_dev:
        raise ValueError("deterministic family partition produced an empty split")

    separation = validate_family_separation(train, ood_dev)
    return train, ood_dev, {
        "strategy": (
            "connected components over semantic_family, non-empty join_path, "
            "AST skeleton, and near_duplicate_family"
        ),
        "seed": str(seed),
        "requested_dev_fraction": float(dev_fraction),
        "component_count": len(component_rows),
        "train_components": len(component_rows) - len(selected_dev_components),
        "ood_dev_components": len(selected_dev_components),
        "train_rows": len(train),
        "ood_dev_rows": len(ood_dev),
        "family_separation": separation,
    }


def validate_family_separation(
    train: Sequence[Mapping[str, Any]],
    ood_dev: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Raise on leakage and return explicit per-family separation evidence."""

    def collect(rows: Sequence[Mapping[str, Any]], kind: str) -> set[str]:
        values: set[str] = set()
        for row in rows:
            metadata = row["family_metadata"]
            if kind == "semantic_family":
                values.add(str(row["semantic_family"]))
            elif kind == "ast_skeleton":
                values.add(str(row["ast"]["skeleton_sha256"]))
            elif kind == "near_duplicate_family":
                values.add(str(metadata["near_duplicate_family"]))
            elif kind == "join_path":
                path = tuple(str(item) for item in row.get("join_path", ()))
                if path:
                    values.add(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            else:
                raise ValueError(f"unknown family kind: {kind}")
        return values

    result: dict[str, bool] = {}
    for kind in (
        "semantic_family",
        "join_path",
        "ast_skeleton",
        "near_duplicate_family",
    ):
        overlap = collect(train, kind) & collect(ood_dev, kind)
        if overlap:
            raise ValueError(f"{kind} leaked across train/OOD splits: {sorted(overlap)}")
        result[kind] = True
    return result


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def build_semantic_corpus(
    db_path: Path,
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: str = DEFAULT_PARTITION_SEED,
) -> dict[str, Any]:
    """Build, execute, and family-partition the deterministic Track C corpus."""

    database = Path(db_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {database}")
    schema = inspect_live_schema(database)
    drafts = _build_drafts(database)
    records = _materialise_drafts(database, drafts)
    observed_coverage = sorted(
        {
            str(tag)
            for record in records
            for tag in record["family_metadata"]["coverage"]
        }
    )
    missing_coverage = sorted(REQUIRED_COVERAGE - set(observed_coverage))
    if missing_coverage:
        raise ValueError(
            "semantic corpus is missing required coverage: " + ", ".join(missing_coverage)
        )

    train, ood_dev, partition = _partition_records(
        records,
        dev_fraction=dev_fraction,
        seed=seed,
    )
    train_bytes = _jsonl_bytes(train)
    dev_bytes = _jsonl_bytes(ood_dev)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nl2sql_semantic_family_corpus",
        "database_sha256": _sha256_file(database),
        "live_schema": {table: list(columns) for table, columns in sorted(schema.items())},
        "required_coverage": sorted(REQUIRED_COVERAGE),
        "observed_coverage": observed_coverage,
        "row_count": len(records),
        "semantic_family_count": len({row["semantic_family"] for row in records}),
        "ast_skeleton_count": len({row["ast"]["skeleton_sha256"] for row in records}),
        "near_duplicate_family_count": len(
            {row["family_metadata"]["near_duplicate_family"] for row in records}
        ),
        "partition": partition,
        "files": {
            "train.jsonl": {
                "rows": len(train),
                "sha256": _sha256_bytes(train_bytes),
            },
            "ood_dev.jsonl": {
                "rows": len(ood_dev),
                "sha256": _sha256_bytes(dev_bytes),
            },
        },
    }
    return {
        "train": train,
        "ood_dev": ood_dev,
        "manifest": manifest,
        "train_bytes": train_bytes,
        "ood_dev_bytes": dev_bytes,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_semantic_corpus(
    db_path: Path,
    output_dir: Path,
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: str = DEFAULT_PARTITION_SEED,
) -> dict[str, Any]:
    """Build the corpus and atomically write train, OOD-dev, and manifest files."""

    result = build_semantic_corpus(
        db_path,
        dev_fraction=dev_fraction,
        seed=seed,
    )
    output = Path(output_dir)
    train_path = output / "train.jsonl"
    dev_path = output / "ood_dev.jsonl"
    manifest_path = output / "manifest.json"
    manifest_bytes = (
        json.dumps(
            result["manifest"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(train_path, result["train_bytes"])
    _atomic_write(dev_path, result["ood_dev_bytes"])
    _atomic_write(manifest_path, manifest_bytes)
    return {
        "train_path": train_path,
        "ood_dev_path": dev_path,
        "manifest_path": manifest_path,
        "train_rows": len(result["train"]),
        "ood_dev_rows": len(result["ood_dev"]),
        "manifest": result["manifest"],
    }


def load_corpus_inputs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load generated JSONL files or corpus directories in deterministic order."""

    files: list[Path] = []
    for supplied in paths:
        path = Path(supplied)
        if path.is_dir():
            for name in ("train.jsonl", "ood_dev.jsonl"):
                candidate = path / name
                if not candidate.is_file():
                    raise FileNotFoundError(f"corpus split is missing: {candidate}")
                files.append(candidate)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"corpus input does not exist: {path}")

    records: dict[str, dict[str, Any]] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    raise ValueError(f"corpus row lacks a string id at {path}:{line_number}")
                if row["id"] in records:
                    raise ValueError(f"duplicate corpus id: {row['id']}")
                records[row["id"]] = row
    if not records:
        raise ValueError("corpus inputs contain no rows")
    return [records[key] for key in sorted(records)]
