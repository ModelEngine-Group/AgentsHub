from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from analysis.open_sql.schema_catalog import allowed_fields, allowed_tables, get_schema_catalog

ALLOWED_FUNCTIONS = {
    "ABS",
    "AVG",
    "CAST",
    "COALESCE",
    "COUNT",
    "DATE",
    "DATETIME",
    "IFNULL",
    "JULIANDAY",
    "LIKE",
    "LENGTH",
    "LOWER",
    "MAX",
    "MIN",
    "NULLIF",
    "ROUND",
    "ROW_NUMBER",
    "STRFTIME",
    "SUBSTR",
    "SUBSTRING",
    "SUM",
    "TIME_TO_STR",
    "TOTAL",
    "TS_OR_DS_TO_TIMESTAMP",
    "TRIM",
    "UPPER",
}
STRUCTURAL_FUNCTIONS = {"AND", "CASE", "IF", "OR"}
SYSTEM_TABLES = {"sqlite_master", "sqlite_schema", "sqlite_sequence", "pragma_table_info"}
MAX_CTES = 8
MAX_SUBQUERY_DEPTH = 6
DEFAULT_RESULT_LIMIT = 500


@dataclass(frozen=True)
class GuardError:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _error(code: str, message: str) -> GuardError:
    return GuardError(code=code, message=message)


def _cte_sources(expression: exp.Expression, cte_names: set[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for cte in expression.find_all(exp.CTE):
        name = cte.alias_or_name
        sources = {table.name for table in cte.this.find_all(exp.Table) if table.name not in cte_names}
        result[name] = sources
    return result


def _cte_outputs(expression: exp.Expression) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for cte in expression.find_all(exp.CTE):
        select = cte.this if isinstance(cte.this, exp.Select) else cte.this.find(exp.Select)
        result[cte.alias_or_name] = {item.alias_or_name for item in (select.expressions if select is not None else [])}
    return result


def _table_aliases(expression: exp.Expression, cte_names: set[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in expression.find_all(exp.Table):
        name = table.name
        aliases[name] = name
        aliases[table.alias_or_name] = name
    for name in cte_names:
        aliases[name] = name
    return aliases


def _projection_aliases(expression: exp.Expression) -> set[str]:
    aliases: set[str] = set()
    for select in expression.find_all(exp.Select):
        for projection in select.expressions:
            if projection.alias:
                aliases.add(projection.alias)
    return aliases


def _function_name(function: exp.Func) -> str:
    name = function.sql_name().upper()
    if name == "ANONYMOUS":
        name = str(getattr(function, "name", "") or "").upper()
    return name


def _flatten_and(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    return [node]


def _source_candidates(name: str, cte_sources: dict[str, set[str]]) -> set[str]:
    return cte_sources.get(name, {name})


def _allowed_join_pairs(catalog: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    pairs: set[tuple[str, str, str, str]] = set()
    for item in catalog.get("joins") or []:
        left = (item["left_table"], item["left_field"], item["right_table"], item["right_field"])
        right = (item["right_table"], item["right_field"], item["left_table"], item["left_field"])
        pairs.update((left, right))
    return pairs


def _validate_join(
    join: exp.Join,
    aliases: dict[str, str],
    cte_sources: dict[str, set[str]],
    allowed_pairs: set[tuple[str, str, str, str]],
) -> list[GuardError]:
    errors: list[GuardError] = []
    on = join.args.get("on")
    using = join.args.get("using")
    if using:
        errors.append(_error("join_using_not_allowed", "JOIN USING is not allowed; use an explicit registered equality join."))
        return errors
    if on is None:
        errors.append(_error("join_condition_missing", "Every JOIN must have an explicit ON condition."))
        return errors
    terms = _flatten_and(on)
    for term in terms:
        if not isinstance(term, exp.EQ) or not isinstance(term.left, exp.Column) or not isinstance(term.right, exp.Column):
            errors.append(_error("join_condition_not_allowed", "JOIN conditions must be conjunctions of registered column equalities."))
            continue
        left, right = term.left, term.right
        if not left.table or not right.table:
            errors.append(_error("join_column_unqualified", "JOIN columns must be table-qualified."))
            continue
        left_name = aliases.get(left.table, left.table)
        right_name = aliases.get(right.table, right.table)
        valid = any(
            (left_source, left.name, right_source, right.name) in allowed_pairs
            for left_source in _source_candidates(left_name, cte_sources)
            for right_source in _source_candidates(right_name, cte_sources)
        )
        if not valid:
            errors.append(_error("join_pair_not_allowed", f"Join not registered: {left_name}.{left.name} = {right_name}.{right.name}."))
    return errors


def _subquery_depth(node: exp.Expression) -> int:
    depth = 0
    current = node.parent
    while current is not None:
        if isinstance(current, (exp.Subquery, exp.CTE)):
            depth += 1
        current = current.parent
    return depth


def _validate_columns(
    expression: exp.Expression,
    aliases: dict[str, str],
    cte_names: set[str],
    cte_sources: dict[str, set[str]],
    cte_outputs: dict[str, set[str]],
    field_map: dict[str, set[str]],
) -> list[GuardError]:
    errors: list[GuardError] = []
    for column in expression.find_all(exp.Column):
        name = column.name
        if name == "*":
            continue
        select = column.find_ancestor(exp.Select)
        scope_tables = []
        if select is not None:
            scope_tables = [
                table
                for table in select.find_all(exp.Table)
                if table.find_ancestor(exp.Select) is select
            ]
        scope_projection_aliases = {
            projection.alias
            for projection in (select.expressions if select is not None else [])
            if projection.alias
        }
        subquery_outputs: set[str] = set()
        if select is not None:
            for subquery in select.find_all(exp.Subquery):
                if subquery.find_ancestor(exp.Select) is not select:
                    continue
                inner = subquery.this if isinstance(subquery.this, exp.Select) else subquery.this.find(exp.Select)
                if inner is not None:
                    subquery_outputs.update(item.alias_or_name for item in inner.expressions)
        if column.table:
            table_name = aliases.get(column.table)
            if table_name is None:
                errors.append(_error("unknown_table_alias", f"Unknown table alias: {column.table}."))
                continue
            if table_name in cte_names:
                sources = cte_sources.get(table_name, set())
                if not any(name in field_map.get(source, set()) for source in sources) and name not in cte_outputs.get(table_name, set()) and name not in scope_projection_aliases:
                    errors.append(_error("cte_field_not_allowed", f"Field not exposed by allowed CTE sources: {column.table}.{name}."))
            elif name not in field_map.get(table_name, set()):
                errors.append(_error("field_not_allowed", f"Field not allowed: {column.table}.{name}."))
            continue
        if name in scope_projection_aliases or name in subquery_outputs:
            continue
        candidates: set[str] = set()
        for table in scope_tables:
            table_name = aliases.get(table.alias_or_name, table.name)
            for source in _source_candidates(table_name, cte_sources):
                if name in field_map.get(source, set()):
                    candidates.add(source)
        if not candidates:
            errors.append(_error("field_not_allowed", f"Unqualified field is not allowed: {name}."))
        elif len(candidates) > 1:
            errors.append(_error("ambiguous_field", f"Unqualified field is ambiguous: {name}."))
    return errors


def _validate_wildcards(expression: exp.Expression) -> list[GuardError]:
    errors: list[GuardError] = []
    for star in expression.find_all(exp.Star):
        if star.find_ancestor(exp.Count):
            continue
        column = star.parent if isinstance(star.parent, exp.Column) else None
        select = star.find_ancestor(exp.Select)
        tables = list(select.find_all(exp.Table)) if select else []
        joins = list(select.find_all(exp.Join)) if select else []
        if column is not None and column.table and not joins:
            continue
        if len(tables) != 1 or joins:
            errors.append(_error("wildcard_not_allowed", "Bare * is only allowed for a single whitelisted table."))
    return errors


def _deduplicate(errors: Iterable[GuardError]) -> list[GuardError]:
    seen: set[tuple[str, str]] = set()
    result: list[GuardError] = []
    for item in errors:
        key = (item.code, item.message)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def validate_sql(
    sql: str | None,
    schema_catalog: dict[str, Any] | None = None,
    *,
    max_result_rows: int = DEFAULT_RESULT_LIMIT,
) -> dict[str, Any]:
    if not sql or not str(sql).strip():
        error = _error("empty_sql", "SQL is empty.")
        return {"safe": False, "reason": error.message, "errors": [error.to_dict()], "normalized_sql": None, "warnings": []}
    raw = str(sql).strip()
    if "--" in raw or "/*" in raw or "*/" in raw:
        error = _error("comment_not_allowed", "SQL comments are not allowed.")
        return {"safe": False, "reason": error.message, "errors": [error.to_dict()], "normalized_sql": None, "warnings": []}
    try:
        statements = parse(raw, read="sqlite")
    except ParseError:
        error = _error("parse_error", "SQL could not be parsed as a supported SQLite query.")
        return {"safe": False, "reason": error.message, "errors": [error.to_dict()], "normalized_sql": None, "warnings": []}
    if len(statements) != 1:
        error = _error("multiple_statements", "Exactly one SQL statement is allowed.")
        return {"safe": False, "reason": error.message, "errors": [error.to_dict()], "normalized_sql": None, "warnings": []}
    expression = statements[0]
    if not isinstance(expression, exp.Select):
        error = _error("statement_not_select", "Only a single SELECT or WITH ... SELECT statement is allowed.")
        return {"safe": False, "reason": error.message, "errors": [error.to_dict()], "normalized_sql": None, "warnings": []}

    catalog = schema_catalog or get_schema_catalog()
    table_set = allowed_tables(catalog)
    field_map = allowed_fields(catalog)
    cte_names = {cte.alias_or_name for cte in expression.find_all(exp.CTE)}
    cte_sources = _cte_sources(expression, cte_names)
    cte_outputs = _cte_outputs(expression)
    aliases = _table_aliases(expression, cte_names)
    errors: list[GuardError] = []
    if len(cte_names) > MAX_CTES:
        errors.append(_error("too_many_ctes", f"At most {MAX_CTES} CTEs are allowed."))
    for table in expression.find_all(exp.Table):
        name = table.name
        if name in SYSTEM_TABLES:
            errors.append(_error("system_table_not_allowed", f"System table is not allowed: {name}."))
        elif name not in table_set and name not in cte_names:
            errors.append(_error("table_not_allowed", f"Table not allowed: {name}."))
    for function in expression.find_all(exp.Func):
        name = _function_name(function)
        if name and name not in ALLOWED_FUNCTIONS and name not in STRUCTURAL_FUNCTIONS:
            errors.append(_error("function_not_allowed", f"Function not allowed: {name}."))
    errors.extend(_validate_columns(expression, aliases, cte_names, cte_sources, cte_outputs, field_map))
    errors.extend(_validate_wildcards(expression))
    allowed_pairs = _allowed_join_pairs(catalog)
    for join in expression.find_all(exp.Join):
        errors.extend(_validate_join(join, aliases, cte_sources, allowed_pairs))
    for subquery in expression.find_all(exp.Subquery):
        if _subquery_depth(subquery) > MAX_SUBQUERY_DEPTH:
            errors.append(_error("subquery_depth_exceeded", f"Subquery depth exceeds {MAX_SUBQUERY_DEPTH}."))
    errors = _deduplicate(errors)
    if errors:
        return {
            "safe": False,
            "reason": "; ".join(item.message for item in errors),
            "errors": [item.to_dict() for item in errors],
            "normalized_sql": None,
            "warnings": [],
            "ast_dialect": "sqlite",
        }
    warnings: list[str] = []
    if expression.args.get("limit") is None:
        expression = expression.limit(max(1, min(int(max_result_rows), DEFAULT_RESULT_LIMIT)))
        warnings.append("LIMIT was automatically added to keep the query bounded.")
    return {
        "safe": True,
        "reason": "",
        "errors": [],
        "normalized_sql": expression.sql(dialect="sqlite"),
        "warnings": warnings,
        "ast_dialect": "sqlite",
    }
