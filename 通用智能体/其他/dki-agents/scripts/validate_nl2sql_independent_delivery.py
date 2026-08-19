"""Validate an independent NL2SQL curator handoff without exposing its rows.

This pre-freeze validator never exposes corpus rows to its caller.  It verifies
the five-file hash/attestation contract and then parses, grounds, and executes
every gold query under the exact bounded read-only evaluator contract.  Both
success and failure receipts remain aggregate-only: no question, gold SQL,
prediction, row index, or row-level audit value is copied into them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_nl2sql_blind import _validate_independent_v11_attestation
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    normalise_development_metadata,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)


PRIVATE_COMPONENT = "outputs/private/"
PUBLIC_COMPONENT = "outputs/remote_runs/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    """Count JSONL records without decoding or displaying a record."""

    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def _validate_opaque_corpus(
    *, corpus: Path, database: Path, expected_count: int
) -> dict[str, int]:
    """Validate every private row while returning aggregate counts only."""

    ids: set[str] = set()
    questions: set[str] = set()
    gold_sql: set[str] = set()
    validated = 0
    executed = 0
    try:
        stream = corpus.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("opaque corpus record contract failed") from exc
    with stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                row = normalise_development_metadata(raw)
                item_id = str(row["id"])
                question = str(row["question"])
                question_key = normalise_question(question)
                sql = str(row["gold_sql"]).strip()
                validate_gold_sql(sql)
                validate_question_sql_grounding(question, sql)
            except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("opaque corpus semantic validation failed") from exc
            if item_id in ids or question_key in questions or sql in gold_sql:
                raise ValueError("opaque corpus uniqueness contract failed")
            ids.add(item_id)
            questions.add(question_key)
            gold_sql.add(sql)
            validated += 1
            try:
                execute_complete_readonly(sql, database)
            except sqlite3.OperationalError as exc:
                if str(exc).strip().casefold() == "interrupted":
                    raise ValueError("opaque corpus bounded execution failed") from None
                raise ValueError("opaque corpus execution failed") from None
            except (OSError, RuntimeError, sqlite3.DatabaseError, ValueError):
                raise ValueError("opaque corpus execution failed") from None
            executed += 1
    if validated != expected_count:
        raise ValueError("opaque corpus record count failed")
    return {
        "semantic_validation_count": validated,
        "bounded_execution_success_count": executed,
    }


def validate_delivery(
    *,
    corpus: Path,
    attestation: Path,
    aggregate_audit: Path,
    execution_audit: Path,
    sha256s: Path,
    database: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Return aggregate validation facts or raise before any blind freeze."""

    paths = {
        "corpus": Path(corpus).resolve(),
        "attestation": Path(attestation).resolve(),
        "aggregate_audit": Path(aggregate_audit).resolve(),
        "execution_audit": Path(execution_audit).resolve(),
        "sha256s": Path(sha256s).resolve(),
    }
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if any(PRIVATE_COMPONENT not in path.as_posix() for path in paths.values()):
        raise ValueError("independent curator handoff must remain below outputs/private")
    if not Path(database).resolve().is_file():
        raise FileNotFoundError(f"sealed database does not exist: {database}")
    if any(not path.is_file() for path in paths.values()):
        missing = sorted(name for name, path in paths.items() if not path.is_file())
        raise FileNotFoundError("missing independent handoff member: " + ", ".join(missing))

    record_count = _count_nonempty_lines(paths["corpus"])
    # The shared validator checks all five handoff files, including opaque
    # corpus/audit hashes, aggregate declarations, and the database binding.
    _pack, curator = _validate_independent_v11_attestation(
        corpus_path=paths["corpus"],
        database_path=Path(database).resolve(),
        attestation_path=paths["attestation"],
        aggregate_audit_path=paths["aggregate_audit"],
        execution_audit_path=paths["execution_audit"],
        sha256s_path=paths["sha256s"],
        n_rows=record_count,
    )
    if record_count != expected_count:
        raise ValueError(
            "independent curation count does not meet the exact contract: "
            f"expected {expected_count}, received {record_count}"
        )
    semantic = _validate_opaque_corpus(
        corpus=paths["corpus"],
        database=Path(database).resolve(),
        expected_count=expected_count,
    )
    return {
        "source_version": curator["source_version"],
        "record_count": record_count,
        "corpus_sha256": _sha256(paths["corpus"]),
        "database_sha256": _sha256(Path(database).resolve()),
        **semantic,
        "handoff_sha256": {
            name: _sha256(path)
            for name, path in sorted(paths.items())
        },
    }


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"independent delivery receipt exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_file_hashes(paths: dict[str, Path]) -> dict[str, str | None]:
    return {
        name: _sha256(path) if path.is_file() else None
        for name, path in sorted(paths.items())
    }


def _failure_code(exc: BaseException) -> str:
    """Return a public, aggregate-only failure category.

    The validator intentionally does not emit exception text because a future
    validation helper could include a row identifier in an exception.  These
    stable categories are sufficient to repair pre-freeze protocol defects
    without exposing questions, SQL, predictions, or failed examples.
    """

    message = str(exc)
    categories = (
        ("opaque corpus bounded execution failed", "bounded_execution_failed"),
        ("opaque corpus semantic validation failed", "semantic_validation_failed"),
        ("opaque corpus uniqueness contract failed", "uniqueness_contract_failed"),
        ("opaque corpus execution failed", "gold_execution_failed"),
        ("opaque corpus record contract failed", "record_contract_failed"),
        ("opaque corpus record count failed", "record_count_failed"),
        ("unsupported independent curator source version", "unsupported_source_version"),
        ("corpus binding failed", "corpus_binding_failed"),
        ("database binding failed", "database_binding_failed"),
        ("isolation declarations are incomplete", "isolation_declaration_failed"),
        ("execution declarations are incomplete", "execution_declaration_failed"),
        ("aggregate audit failed", "aggregate_audit_failed"),
        ("SHA256SUMS file set differs", "sha256s_file_set_failed"),
        ("SHA256SUMS mismatch", "sha256s_hash_failed"),
        ("execution audit row count differs", "execution_audit_count_failed"),
        ("expected", "record_count_failed"),
        ("missing independent handoff member", "handoff_member_missing"),
        ("sealed database does not exist", "database_missing"),
    )
    for needle, code in categories:
        if needle in message:
            return code
    return "independent_delivery_contract_failed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--aggregate-audit", type=Path, required=True)
    parser.add_argument("--execution-audit", type=Path, required=True)
    parser.add_argument("--sha256s", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_tree_sha256):
        raise SystemExit("source-tree-sha256 must be a lowercase SHA-256")
    if PUBLIC_COMPONENT not in output.as_posix():
        raise SystemExit("public validation receipt must remain below outputs/remote_runs")
    if output.exists():
        raise SystemExit(f"independent delivery receipt exists: {output}")
    handoff_paths = {
        "aggregate_audit": args.aggregate_audit.resolve(),
        "attestation": args.attestation.resolve(),
        "corpus": args.corpus.resolve(),
        "execution_audit": args.execution_audit.resolve(),
        "sha256s": args.sha256s.resolve(),
    }
    observed_count = (
        _count_nonempty_lines(handoff_paths["corpus"])
        if handoff_paths["corpus"].is_file()
        else None
    )
    try:
        result = validate_delivery(
            corpus=args.corpus,
            attestation=args.attestation,
            aggregate_audit=args.aggregate_audit,
            execution_audit=args.execution_audit,
            sha256s=args.sha256s,
            database=args.database,
            expected_count=args.expected_count,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        payload = {
            "schema_version": 1,
            "kind": "nl2sql_independent_delivery_validation",
            "status": "REJECTED_PRE_FREEZE",
            "gate_passed": False,
            "reason": "independent_delivery_validation_failed",
            "error_class": type(exc).__name__,
            "failure_code": _failure_code(exc),
            "contains_questions": False,
            "contains_gold_sql": False,
            "contains_predictions": False,
            "contains_error_samples": False,
            "expected_count": args.expected_count,
            "observed_record_count": observed_count,
            "source_tree_sha256": args.source_tree_sha256,
            "validator_sha256": _sha256(Path(__file__)),
            "handoff_sha256": _safe_file_hashes(handoff_paths),
        }
        _write_once(output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 2

    payload = {
        "schema_version": 1,
        "kind": "nl2sql_independent_delivery_validation",
        "status": "VALIDATED_UNFROZEN",
        "gate_passed": True,
        "contains_questions": False,
        "contains_gold_sql": False,
        "contains_predictions": False,
        "contains_error_samples": False,
        "expected_count": args.expected_count,
        "source_tree_sha256": args.source_tree_sha256,
        "validator_sha256": _sha256(Path(__file__)),
        "result": result,
    }
    _write_once(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
