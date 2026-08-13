"""Shared validation for sealed NL2SQL development and blind manifests."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from dki.analysis.nl2sql_sqlite import DEFAULT_PROGRESS_OPS
from dki.analysis.nl2sql_sqlite import DEFAULT_QUERY_TIMEOUT_SECONDS
from dki.analysis.nl2sql_sqlite import DEFAULT_VM_STEP_BUDGET
from dki.analysis.nl2sql_sqlite import bounded_execution
from dki.analysis.nl2sql_sqlite import connect_readonly

RESULT_FINGERPRINT_VERSION = "nl2sql-complete-result-v1"
DEFAULT_COMPLETE_RESULT_MAX_ROWS = 100_000
DEFAULT_COMPLETE_RESULT_MAX_CANONICAL_BYTES = 16 * 1024 * 1024

RowSemantics = Literal["ordered", "unordered"]


class CompleteResultLimitError(RuntimeError):
    """Raised when complete-result execution or serialization exceeds a bound."""


class CompleteResultSerializationError(TypeError):
    """Raised when a result contains a scalar that cannot be encoded safely."""


@dataclass(frozen=True, slots=True)
class CompleteResultFingerprint:
    """One fully materialized read result and its deterministic typed identity."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_semantics: RowSemantics
    result_sha256: str
    canonical_size_bytes: int

    @property
    def column_arity(self) -> int:
        return len(self.columns)

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _require_positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _validate_complete_result_options(
    row_semantics: RowSemantics,
    max_rows: int,
    max_canonical_bytes: int,
) -> None:
    if row_semantics not in {"ordered", "unordered"}:
        raise ValueError("row_semantics must be 'ordered' or 'unordered'")
    _require_positive_integer(max_rows, "max_rows")
    _require_positive_integer(max_canonical_bytes, "max_canonical_bytes")


def _typed_result_scalar(value: Any) -> list[str]:
    """Encode one SQLite scalar without conflating Python scalar types."""
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["boolean", "true" if value else "false"]
    if type(value) is int:
        return ["integer", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise CompleteResultSerializationError(
                "non-finite floating-point result values are not supported"
            )
        return ["real", value.hex()]
    if type(value) is str:
        return ["text", value]
    if type(value) is bytes:
        return ["blob", base64.b64encode(value).decode("ascii")]
    raise CompleteResultSerializationError(
        f"unsupported complete-result scalar type: {type(value).__name__}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CompleteResultSerializationError(
            "complete result cannot be serialized canonically"
        ) from exc


def _canonicalize_complete_result(
    columns: Iterable[str],
    rows: Iterable[Iterable[Any]],
    *,
    row_semantics: RowSemantics,
    max_rows: int,
    max_canonical_bytes: int,
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...], bytes]:
    _validate_complete_result_options(
        row_semantics,
        max_rows,
        max_canonical_bytes,
    )
    try:
        column_tuple = tuple(columns)
    except TypeError as exc:
        raise CompleteResultSerializationError("columns must be iterable") from exc
    if any(type(column) is not str for column in column_tuple):
        raise CompleteResultSerializationError("column names must be strings")

    materialized_rows: list[tuple[Any, ...]] = []
    canonical_rows: list[tuple[bytes, list[list[str]]]] = []
    encoded_row_bytes = 0
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise CompleteResultSerializationError("rows must be iterable") from exc

    for row_index, row in enumerate(iterator):
        if row_index >= max_rows:
            raise CompleteResultLimitError(
                f"complete result exceeds max_rows={max_rows}"
            )
        if isinstance(row, (str, bytes, bytearray, memoryview)):
            raise CompleteResultSerializationError(
                f"result row {row_index} must be an iterable of scalar values"
            )
        try:
            row_tuple = tuple(row)
        except TypeError as exc:
            raise CompleteResultSerializationError(
                f"result row {row_index} must be iterable"
            ) from exc
        if len(row_tuple) != len(column_tuple):
            raise CompleteResultSerializationError(
                f"result row {row_index} has arity {len(row_tuple)}; "
                f"expected {len(column_tuple)}"
            )
        typed_row = [_typed_result_scalar(value) for value in row_tuple]
        row_bytes = _canonical_json_bytes(typed_row)
        encoded_row_bytes += len(row_bytes)
        if encoded_row_bytes > max_canonical_bytes:
            raise CompleteResultLimitError(
                "complete-result row serialization exceeds "
                f"max_canonical_bytes={max_canonical_bytes}"
            )
        materialized_rows.append(row_tuple)
        canonical_rows.append((row_bytes, typed_row))

    if row_semantics == "unordered":
        canonical_rows.sort(key=lambda item: item[0])

    payload = {
        "column_arity": len(column_tuple),
        "columns": list(column_tuple),
        "format": RESULT_FINGERPRINT_VERSION,
        "row_count": len(materialized_rows),
        "row_semantics": row_semantics,
        "rows": [typed_row for _, typed_row in canonical_rows],
    }
    encoded_payload = _canonical_json_bytes(payload)
    if len(encoded_payload) > max_canonical_bytes:
        raise CompleteResultLimitError(
            "complete-result canonical payload exceeds "
            f"max_canonical_bytes={max_canonical_bytes}"
        )
    return column_tuple, tuple(materialized_rows), encoded_payload


def canonical_complete_result_payload(
    columns: Iterable[str],
    rows: Iterable[Iterable[Any]],
    *,
    row_semantics: RowSemantics = "ordered",
    max_rows: int = DEFAULT_COMPLETE_RESULT_MAX_ROWS,
    max_canonical_bytes: int = DEFAULT_COMPLETE_RESULT_MAX_CANONICAL_BYTES,
) -> bytes:
    """Return the bounded canonical typed payload for one complete result."""
    _, _, payload = _canonicalize_complete_result(
        columns,
        rows,
        row_semantics=row_semantics,
        max_rows=max_rows,
        max_canonical_bytes=max_canonical_bytes,
    )
    return payload


def fingerprint_complete_result(
    columns: Iterable[str],
    rows: Iterable[Iterable[Any]],
    *,
    row_semantics: RowSemantics = "ordered",
    max_rows: int = DEFAULT_COMPLETE_RESULT_MAX_ROWS,
    max_canonical_bytes: int = DEFAULT_COMPLETE_RESULT_MAX_CANONICAL_BYTES,
) -> str:
    """Return a SHA-256 identity over every typed scalar in a complete result."""
    payload = canonical_complete_result_payload(
        columns,
        rows,
        row_semantics=row_semantics,
        max_rows=max_rows,
        max_canonical_bytes=max_canonical_bytes,
    )
    return hashlib.sha256(payload).hexdigest()


ALLOWED_TABLES = {"patients", "visits", "kg_entities", "kg_triples"}
ALLOWED_DIFFICULTIES = {"medium", "hard", "extra", "extra-hard"}
ALLOWED_CATEGORIES = {
    "filtering",
    "aggregation_grouping",
    "top_k",
    "multi_table_join",
    "temporal",
    "knowledge_graph",
    "having",
    "nested_correlated",
    "cte",
    "set_operation",
    "window",
}


def _normalise_development_label(
    value: Any,
    *,
    label: str,
    allowed: set[str],
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"development {label} must be a string")
    normalised = unicodedata.normalize("NFKC", value).strip().casefold()
    if normalised not in allowed:
        raise ValueError(f"development {label} is not allowlisted: {value!r}")
    return normalised


def normalise_development_difficulty(value: Any) -> str:
    """Return one canonical, allowlisted development difficulty."""
    return _normalise_development_label(
        value,
        label="difficulty",
        allowed=ALLOWED_DIFFICULTIES,
    )


def normalise_development_category(value: Any) -> str:
    """Return one canonical, allowlisted development category."""
    return _normalise_development_label(
        value,
        label="category",
        allowed=ALLOWED_CATEGORIES,
    )


def normalise_development_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a development row with strict canonical difficulty/category labels."""
    if not isinstance(row, Mapping):
        raise ValueError("development item must be a JSON object")
    normalised = dict(row)
    normalised["difficulty"] = normalise_development_difficulty(
        normalised.get("difficulty")
    )
    normalised["category"] = normalise_development_category(
        normalised.get("category")
    )
    return normalised


def normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question).strip()).casefold()


def validate_gold_sql(sql: str) -> str:
    """Validate one SQLite read query and return its AST skeleton."""
    try:
        from sqlglot import exp, parse
    except ImportError as exc:  # never fall back to regex-only validation
        raise RuntimeError("sqlglot is required for the NL2SQL blind protocol") from exc
    statements = parse(sql, read="sqlite")
    if len(statements) != 1:
        raise ValueError("gold_sql must contain exactly one statement")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Subquery, exp.Union, exp.Intersect, exp.Except)) and not tree.find(exp.Select):
        raise ValueError("gold_sql must be a SQLite SELECT/WITH query")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Attach, exp.Command)
    if any(tree.find(kind) for kind in forbidden):
        raise ValueError("gold_sql must be read-only")
    # sqlglot represents a CTE reference as ``exp.Table`` as well.  Treat only
    # base relations as access-control subjects; otherwise every legitimate
    # WITH query is falsely rejected because its local alias is not one of the
    # four sealed tables.
    cte_names = {
        str(cte.alias_or_name).casefold()
        for cte in tree.find_all(exp.CTE)
        if str(cte.alias_or_name).strip()
    }
    tables = {table.name.casefold() for table in tree.find_all(exp.Table)}
    unknown = tables - ALLOWED_TABLES - cte_names
    if unknown:
        raise ValueError(f"gold_sql references forbidden tables: {sorted(unknown)}")
    def anonymise(node):
        if isinstance(node, exp.Literal):
            return exp.Literal.string("__LITERAL__")
        return node
    return re.sub(r"\s+", " ", tree.transform(anonymise).sql(dialect="sqlite")).strip().upper()


def validate_question_sql_grounding(question: str, sql: str) -> None:
    """Require predicate and LIMIT literals to be explicitly grounded in the question.

    Execution alone cannot prove that a curated question states every condition
    used by its gold SQL.  This fail-closed check deliberately covers only
    literals that directly constrain rows or result cardinality; formatting
    literals inside functions (for example ``strftime('%Y', ...)``) are not
    treated as user predicates.
    """
    validate_gold_sql(sql)
    try:
        from sqlglot import exp, parse_one
    except ImportError as exc:
        raise RuntimeError("sqlglot is required for NL2SQL semantic grounding") from exc

    tree = parse_one(sql, read="sqlite")
    literals = []
    binary_predicates = tuple(
        kind
        for name in ("EQ", "NEQ", "GT", "GTE", "LT", "LTE", "Like", "ILike")
        if (kind := getattr(exp, name, None)) is not None
    )
    for predicate in tree.walk():
        if isinstance(predicate, binary_predicates):
            for key in ("this", "expression"):
                value = predicate.args.get(key)
                if isinstance(value, exp.Literal):
                    literals.append(value)
        elif isinstance(predicate, exp.Between):
            for key in ("low", "high"):
                value = predicate.args.get(key)
                if isinstance(value, exp.Literal):
                    literals.append(value)
        elif isinstance(predicate, exp.In):
            literals.extend(
                value for value in predicate.expressions
                if isinstance(value, exp.Literal)
            )
        elif isinstance(predicate, (exp.Limit, exp.Offset)):
            value = predicate.args.get("expression")
            if isinstance(value, exp.Literal):
                literals.append(value)

    normalised_question = unicodedata.normalize("NFKC", question).casefold()
    compact_question = re.sub(r"\s+", "", normalised_question)
    # Keep word boundaries for numeric grounding.  The compact form is useful
    # for matching CJK/string literals, but it turns an English phrase such as
    # ``top 2 diagnoses`` into ``top2diagnoses`` and falsely hides the number.
    numeric_tokens = re.findall(
        r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.])",
        normalised_question,
    )

    def grounded(literal) -> bool:
        raw = str(literal.this)
        if literal.is_string:
            if raw == "":
                raise ValueError("empty predicate literal cannot be grounded in a question")
            candidate = unicodedata.normalize("NFKC", raw).casefold()
            if candidate in compact_question:
                return True
            if "%" in candidate or "_" in candidate:
                unescaped = candidate.replace("%", "").replace("_", "")
                return bool(unescaped) and unescaped in compact_question
            return False
        try:
            expected = Decimal(raw)
        except InvalidOperation:
            return raw.casefold() in compact_question
        for token in numeric_tokens:
            try:
                if Decimal(token) == expected:
                    return True
            except InvalidOperation:
                continue
        return False

    missing = []
    seen = set()
    for literal in literals:
        identity = (literal.is_string, str(literal.this))
        if identity in seen:
            continue
        seen.add(identity)
        if not grounded(literal):
            missing.append(str(literal.this))
    if missing:
        raise ValueError(
            "gold predicate literal(s) not grounded in question: "
            + ", ".join(repr(value) for value in missing)
        )


def execute_complete_readonly(sql: str, db_path: Path):
    """Execute validated SQL without the UI evaluator's 200-row truncation."""
    validate_gold_sql(sql)
    conn = connect_readonly(Path(db_path))
    try:
        with bounded_execution(conn):
            cursor = conn.execute(sql.strip().rstrip(";"))
            columns = [column[0] for column in cursor.description]
            rows = [tuple(row) for row in cursor.fetchall()]
        return columns, rows
    finally:
        conn.close()


def execute_complete_readonly_fingerprint(
    sql: str,
    db_path: Path,
    *,
    row_semantics: RowSemantics = "ordered",
    max_rows: int = DEFAULT_COMPLETE_RESULT_MAX_ROWS,
    max_canonical_bytes: int = DEFAULT_COMPLETE_RESULT_MAX_CANONICAL_BYTES,
    max_vm_steps: int = DEFAULT_VM_STEP_BUDGET,
    timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    progress_ops: int = DEFAULT_PROGRESS_OPS,
) -> CompleteResultFingerprint:
    """Execute and fingerprint every row of one bounded read-only query.

    No fingerprint is returned unless execution, full cursor consumption, row
    validation, typed serialization, and all configured bounds succeed.
    """
    _validate_complete_result_options(
        row_semantics,
        max_rows,
        max_canonical_bytes,
    )
    validate_gold_sql(sql)
    conn = connect_readonly(Path(db_path))
    try:
        with bounded_execution(
            conn,
            max_vm_steps=max_vm_steps,
            timeout_seconds=timeout_seconds,
            progress_ops=progress_ops,
        ):
            cursor = conn.execute(sql.strip().rstrip(";"))
            if cursor.description is None:
                raise CompleteResultSerializationError(
                    "complete-result query did not return columns"
                )
            columns, rows, payload = _canonicalize_complete_result(
                (column[0] for column in cursor.description),
                cursor,
                row_semantics=row_semantics,
                max_rows=max_rows,
                max_canonical_bytes=max_canonical_bytes,
            )
        return CompleteResultFingerprint(
            columns=columns,
            rows=rows,
            row_semantics=row_semantics,
            result_sha256=hashlib.sha256(payload).hexdigest(),
            canonical_size_bytes=len(payload),
        )
    finally:
        conn.close()


execute_complete_readonly_with_fingerprint = execute_complete_readonly_fingerprint
