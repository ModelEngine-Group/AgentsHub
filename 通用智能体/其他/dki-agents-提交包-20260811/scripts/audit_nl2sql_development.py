"""Audit one verified NL2SQL development snapshot without weakening blind gates.

Invalid semantic-grounding rows are reported only in the private diagnostic
artifact.  Public output is a fixed allowlisted aggregate and is permanently
ineligible for a development quality gate whenever an exclusion occurred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_development_evidence import (
    private_development_output_path,
    public_development_output_path,
    publish_development_json_pair_once,
    recover_development_json_pair,
    require_fresh_outputs,
    sha256_file,
    strict_development_jsonl_rows,
    verified_evaluation_context,
)
from dki.analysis.nl2sql_eval import evaluate_set
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    normalise_development_metadata,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)


_PUBLIC_OVERALL_FIELDS = (
    "n",
    "n_correct",
    "accuracy",
    "n_answered",
    "n_abstained",
    "coverage",
    "execution_accuracy_answered",
    "strict_result_contract",
)


def sha256(path: Path) -> str:
    return sha256_file(path)


def _load_development_rows(input_bytes: bytes) -> list[dict[str, Any]]:
    raw_rows = strict_development_jsonl_rows(input_bytes)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        rows.append(normalise_development_metadata(raw))
    return rows


def _fixed_public_audit_payload(
    *,
    common: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build the public body from an explicit, non-extensible allowlist."""
    try:
        overall = {field: result[field] for field in _PUBLIC_OVERALL_FIELDS}
    except KeyError as exc:
        raise ValueError(f"audit result lacks required public field: {exc.args[0]}") from exc
    return {
        "schema_version": 1,
        "rid": common["rid"],
        "kind": "public_development_diagnostic_aggregate",
        "dataset_role": "development_diagnostic_invalid_excluded",
        "selection_used": True,
        "created_once": True,
        "gate_eligible": False,
        "engine": "rule",
        "input_sha256": common["input_sha256"],
        "database_sha256": common["database_sha256"],
        "development_evidence_sha256": common["development_evidence_sha256"],
        "development_evidence_kind": "official_public_v2_four_table_development_receipt",
        "source_tree_sha256": common["source_tree_sha256"],
        "engine_sha256": common["engine_sha256"],
        "evaluator_sha256": common["evaluator_sha256"],
        "protocol_sha256": common["protocol_sha256"],
        "audit_script_sha256": common["audit_script_sha256"],
        "n_total": common["n_total"],
        "n_valid": common["n_valid"],
        "n_invalid": common["n_invalid"],
        "invalid_exclusion_reason": "semantic_grounding_validation_failed",
        "overall": overall,
        "decision": "diagnostic_only_not_a_development_gate",
        "contains_questions_or_sql": False,
    }


def _context_identity(context) -> dict[str, Any]:
    return {
        "rid": context.rid,
        "input_sha256": context.input_sha256,
        "database_sha256": context.database_sha256,
        "development_evidence_sha256": context.evidence_sha256,
        "source_tree_sha256": context.source_tree_sha256,
        "trusted_evidence": context.trusted_evidence_binding,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--development-evidence",
        "--development-receipt",
        dest="development_evidence",
        type=Path,
        required=True,
        help="public official-v2 receipt binding this private development input and DB",
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    args = parser.parse_args()

    try:
        with verified_evaluation_context(
            input_path=args.input,
            database_path=args.db,
            evidence_path=args.development_evidence,
            source_tree_sha256=args.source_tree_sha256,
            root=ROOT,
        ) as context:
            private_output = private_development_output_path(
                args.private_output,
                root=ROOT,
                rid=context.rid,
            )
            public_output = public_development_output_path(
                args.public_output,
                root=ROOT,
                rid=context.rid,
            )
            if recover_development_json_pair(
                private_path=private_output,
                public_path=public_output,
                root=ROOT,
                rid=context.rid,
                expected_identity=_context_identity(context),
            ):
                print(
                    json.dumps(
                        {
                            "recovered": True,
                            "gate_eligible": False,
                            "public_output": str(public_output),
                        },
                        ensure_ascii=False,
                    )
                )
                return 0
            require_fresh_outputs(private_output, public_output)

            rows = _load_development_rows(context.input_bytes)
            ids: set[str] = set()
            questions: set[str] = set()
            valid_rows: list[dict[str, Any]] = []
            invalid: list[dict[str, Any]] = []
            for index, row in enumerate(rows, start=1):
                item_id = row["id"]
                question_key = normalise_question(row["question"])
                if item_id in ids or question_key in questions:
                    raise ValueError("development ids and normalised questions must be unique")
                ids.add(item_id)
                questions.add(question_key)
                validate_gold_sql(row["gold_sql"])
                execute_complete_readonly(
                    row["gold_sql"],
                    context.database_snapshot_path,
                )
                try:
                    validate_question_sql_grounding(row["question"], row["gold_sql"])
                except Exception as exc:
                    invalid.append(
                        {
                            "index": index,
                            "id_sha256": hashlib.sha256(
                                item_id.encode("utf-8")
                            ).hexdigest(),
                            "failure_class": type(exc).__name__,
                            "failure": str(exc),
                        }
                    )
                else:
                    valid_rows.append(row)

            result = evaluate_set(
                valid_rows,
                "rule",
                context.database_snapshot_path,
                strict=True,
            )
            common = {
                "schema_version": 1,
                "rid": context.rid,
                "dataset_role": "development_diagnostic_invalid_excluded",
                "selection_used": True,
                "created_once": True,
                "gate_eligible": False,
                "engine": "rule",
                "input_sha256": context.input_sha256,
                "database_sha256": context.database_sha256,
                "development_evidence_sha256": context.evidence_sha256,
                "development_evidence_kind": "official_public_v2_four_table_development_receipt",
                "trusted_evidence": context.trusted_evidence_binding,
                "source_tree_sha256": context.source_tree_sha256,
                "engine_sha256": sha256(ROOT / "dki" / "analysis" / "nl2sql.py"),
                "evaluator_sha256": sha256(ROOT / "dki" / "analysis" / "nl2sql_eval.py"),
                "protocol_sha256": sha256(ROOT / "dki" / "analysis" / "nl2sql_protocol.py"),
                "audit_script_sha256": sha256(Path(__file__)),
                "n_total": len(rows),
                "n_valid": len(valid_rows),
                "n_invalid": len(invalid),
                "invalid_exclusion_reason": "semantic_grounding_validation_failed",
            }
            private_payload = {
                **common,
                "kind": "private_development_diagnostic_details",
                "invalid": invalid,
                "details": result["details"],
            }
            public_payload = _fixed_public_audit_payload(
                common=common,
                result=result,
            )
            publish_development_json_pair_once(
                private_path=private_output,
                private_payload=private_payload,
                public_path=public_output,
                public_payload=public_payload,
                root=ROOT,
                rid=context.rid,
            )
            print(
                json.dumps(
                    {
                        "n_total": len(rows),
                        "n_valid": len(valid_rows),
                        "n_invalid": len(invalid),
                        "accuracy": result["accuracy"],
                        "coverage": result["coverage"],
                        "gate_eligible": False,
                        "public_output": str(public_output),
                    },
                    ensure_ascii=False,
                )
            )
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
