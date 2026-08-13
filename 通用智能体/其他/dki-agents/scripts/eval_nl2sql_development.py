"""Evaluate one verified NL2SQL development snapshot for selection only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_development_evidence import (
    private_development_output_path,
    publish_private_development_json_once,
    require_fresh_outputs,
    sha256_file,
    strict_development_jsonl_rows,
    verified_evaluation_context,
)
from dki.analysis.nl2sql_eval import evaluate_set
from dki.analysis.nl2sql_protocol import (
    normalise_development_metadata,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)


def sha256(path: Path) -> str:
    return sha256_file(path)


def _load_development_rows(input_bytes: bytes) -> list[dict[str, Any]]:
    raw_rows = strict_development_jsonl_rows(input_bytes)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = normalise_development_metadata(raw)
        validate_gold_sql(row["gold_sql"])
        validate_question_sql_grounding(row["question"], row["gold_sql"])
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows) or len(
        {normalise_question(row["question"]) for row in rows}
    ) != len(rows):
        raise ValueError("development ids and normalised questions must be unique")
    return rows


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-tree-sha256",
        required=True,
        help="immutable included-tree identity bound by the development evidence",
    )
    args = parser.parse_args()

    try:
        with verified_evaluation_context(
            input_path=args.input,
            database_path=args.db,
            evidence_path=args.development_evidence,
            source_tree_sha256=args.source_tree_sha256,
            root=ROOT,
        ) as context:
            output = private_development_output_path(
                args.output,
                root=ROOT,
                rid=context.rid,
            )
            require_fresh_outputs(output)
            rows = _load_development_rows(context.input_bytes)
            result = evaluate_set(
                rows,
                "rule",
                context.database_snapshot_path,
                strict=True,
            )
            payload = {
                "schema_version": 1,
                "rid": context.rid,
                "dataset_role": "development_selection_used",
                "selection_used": True,
                "created_once": True,
                "engine": "rule",
                "input_sha256": context.input_sha256,
                "database_sha256": context.database_sha256,
                "development_evidence_sha256": context.evidence_sha256,
                "development_evidence_kind": "official_public_v2_four_table_development_receipt",
                "trusted_evidence": context.trusted_evidence_binding,
                "engine_source_sha256": sha256(ROOT / "dki" / "analysis" / "nl2sql.py"),
                "source_tree_sha256": context.source_tree_sha256,
                "evaluator_sha256": sha256(Path(__file__)),
                "overall": {
                    key: value for key, value in result.items() if key != "details"
                },
                "details": result["details"],
            }
            publish_private_development_json_once(
                path=output,
                payload=payload,
                root=ROOT,
                rid=context.rid,
            )
            print(
                json.dumps(
                    {
                        "n": result["n"],
                        "accuracy": result["accuracy"],
                        "coverage": result["coverage"],
                        "output": str(output),
                    },
                    ensure_ascii=False,
                )
            )
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
