"""Engine-blind semantic-grounding preflight for NL2SQL curators.

This file is safe to copy into an isolated curator pack: it contains no
translation rules, prompts, development examples, or evaluator predictions.
It only checks that direct SQL predicate and cardinality literals are stated
explicitly by the paired natural-language question.  The freezer enforces the
same literal policy again before it creates a development/holdout split.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def missing_grounded_literals(question: str, sql: str) -> tuple[str, ...]:
    """Return ungrounded direct literals, rejecting empty string predicates."""

    try:
        from sqlglot import exp, parse_one
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError("sqlglot is required for curator grounding preflight") from exc

    statements = [item for item in sql.split(";") if item.strip()]
    if len(statements) != 1:
        raise ValueError("gold_sql must contain exactly one statement")
    tree = parse_one(sql, read="sqlite")
    literals: list[Any] = []
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
                value for value in predicate.expressions if isinstance(value, exp.Literal)
            )
        elif isinstance(predicate, (exp.Limit, exp.Offset)):
            value = predicate.args.get("expression")
            if isinstance(value, exp.Literal):
                literals.append(value)

    normalized_question = unicodedata.normalize("NFKC", str(question)).casefold()
    compact_question = re.sub(r"\s+", "", normalized_question)
    numeric_tokens = re.findall(
        r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.])",
        normalized_question,
    )

    def grounded(literal: Any) -> bool:
        raw = str(literal.this)
        if literal.is_string:
            if raw == "":
                raise ValueError("empty predicate literal is forbidden")
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

    missing: list[str] = []
    seen: set[tuple[bool, str]] = set()
    for literal in literals:
        identity = (bool(literal.is_string), str(literal.this))
        if identity in seen:
            continue
        seen.add(identity)
        if not grounded(literal):
            missing.append(str(literal.this))
    return tuple(missing)


def validate_corpus(path: Path) -> dict[str, int | bool]:
    """Validate every JSONL row and return aggregate-only facts."""

    total = passed = failed = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                missing = missing_grounded_literals(
                    str(row.get("question", "")), str(row.get("gold_sql", ""))
                )
                if missing:
                    raise ValueError("one or more SQL literals are not grounded")
            except (KeyError, TypeError, ValueError):
                failed += 1
            else:
                passed += 1
    return {
        "n_rows": total,
        "n_passed": passed,
        "n_failed": failed,
        "gate_passed": total > 0 and failed == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    result = validate_corpus(args.corpus)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
