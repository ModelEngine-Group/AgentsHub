"""Run a small, deterministic, development-only NL2SQL Agent smoke.

The selector takes the first ``N`` rows per public category after sorting by
SHA-256 of the normalized question.  It never selects by id, SQL, correctness,
or a prior prediction.  The frozen holdout is neither loaded nor evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (str(ROOT), str(SCRIPT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from eval_nl2sql_frozen_development import (  # noqa: E402
    _load_development,
    _nl2sql_for_evaluation,
    _publish_json_once,
    _rid_below,
    _sha256,
    _validate_engine_strategy,
)
from dki.analysis.database import database_matches_manifest  # noqa: E402
from dki.analysis.nl2sql_eval import evaluate_set  # noqa: E402
from dki.analysis.nl2sql_protocol import normalise_question  # noqa: E402
from dki.analysis.nl2sql_seal import (  # noqa: E402
    _agent_runtime_profile,
    _assert_freeze_unconsumed,
    _freeze_contract,
    _safe_frozen_member,
    _verify_source_tree_manifest,
)
from dki.pipeline.artifacts import validate_task2_artifact  # noqa: E402


def _question_digest(row: dict[str, Any]) -> str:
    question = normalise_question(str(row["question"]))
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _select_development_rows(
    rows: list[dict[str, Any]],
    *,
    per_category: int,
) -> list[dict[str, Any]]:
    """Select a deterministic category-stratified smoke without answer use."""

    if type(per_category) is not int or not 1 <= per_category <= 10:
        raise ValueError("per_category must be an integer between 1 and 10")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = row.get("category")
        if type(category) is not str or not category.strip():
            raise ValueError("development row lacks a non-empty category")
        grouped[category].append(row)
    if not grouped:
        raise ValueError("development is empty")

    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        candidates = sorted(grouped[category], key=_question_digest)
        if len(candidates) < per_category:
            raise ValueError(
                f"category {category!r} has fewer than {per_category} rows"
            )
        selected.extend(candidates[:per_category])
    return selected


def _selector_sha256(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "category": str(row["category"]),
            "question_sha256": _question_digest(row),
        }
        for row in rows
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _aggregate_result(result: dict[str, Any]) -> dict[str, Any]:
    details = result.get("details")
    if not isinstance(details, list):
        raise ValueError("smoke evaluator lacks details")
    by_category: dict[str, dict[str, Any]] = {}
    categories = sorted({str(row.get("category")) for row in details})
    for category in categories:
        members = [row for row in details if str(row.get("category")) == category]
        correct = sum(bool(row.get("correct")) for row in members)
        answered = sum(row.get("engine") not in (None, "none") for row in members)
        by_category[category] = {
            "n": len(members),
            "correct": correct,
            "accuracy": round(correct / max(1, len(members)), 4),
            "answered": answered,
            "coverage": round(answered / max(1, len(members)), 4),
        }
    return {
        "n": int(result["n"]),
        "n_correct": int(result["n_correct"]),
        "accuracy": float(result["accuracy"]),
        "n_answered": int(result["n_answered"]),
        "n_abstained": int(result["n_abstained"]),
        "coverage": float(result["coverage"]),
        "execution_accuracy_answered": float(
            result["execution_accuracy_answered"]
        ),
        "strict_result_contract": bool(result["strict_result_contract"]),
        "by_category": by_category,
        "by_difficulty": result["by_difficulty"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--engine", choices=["agent"], default="agent")
    parser.add_argument(
        "--llm-strategy",
        choices=["legacy_sql", "query_plan", "query_plan_v2"],
        default="legacy_sql",
    )
    parser.add_argument(
        "--agent-rule-policy",
        choices=["rule_first", "model_first", "plan_first"],
        default="rule_first",
    )
    parser.add_argument(
        "--llm-reasoning-mode",
        choices=["disabled", "adaptive"],
        default="disabled",
    )
    parser.add_argument("--llm-completion-tokens", type=int, default=1024)
    parser.add_argument("--per-category", type=int, default=1)
    args = parser.parse_args()

    try:
        _validate_engine_strategy(
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )
        if not 1 <= args.per_category <= 10:
            raise ValueError("per_category must be between 1 and 10")

        manifest = args.manifest.resolve()
        private_output = args.private_output.resolve()
        public_output = args.public_output.resolve()
        freeze_rid = _rid_below(manifest, "private")
        source_tree = _verify_source_tree_manifest(args.source_tree_manifest)
        candidate_rid = source_tree.get("rid")
        if (
            type(candidate_rid) is not str
            or not candidate_rid
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in candidate_rid
            )
        ):
            raise ValueError("included-tree manifest lacks a candidate RID")
        if _rid_below(private_output, "private") != candidate_rid:
            raise ValueError("private smoke output must use the candidate RID")
        if _rid_below(public_output, "remote_runs") != candidate_rid:
            raise ValueError("public smoke output must use the candidate RID")

        expected_private = (
            ROOT
            / "outputs"
            / "private"
            / candidate_rid
            / "nl2sql_development_smoke"
            / freeze_rid
            / "reasoning_smoke.json"
        ).resolve()
        expected_public = (
            ROOT
            / "outputs"
            / "remote_runs"
            / candidate_rid
            / "nl2sql"
            / "reasoning_smoke_attestation.json"
        ).resolve()
        if private_output != expected_private or public_output != expected_public:
            raise ValueError("smoke outputs do not match the fixed RID-scoped paths")
        if private_output.exists() or public_output.exists():
            raise FileExistsError("reasoning smoke output already exists")

        _assert_freeze_unconsumed(manifest)
        validate_task2_artifact(args.kg, task1_path=args.task1_manifest)
        if not database_matches_manifest(args.db, args.task1_manifest, args.kg):
            raise ValueError("database does not match Task 1/KG lineage")
        freeze = _freeze_contract(
            freeze_manifest=manifest,
            task1_manifest=args.task1_manifest,
            kg_path=args.kg,
            db_path=args.db,
        )
        if (
            freeze.get("kind")
            != "independent_curated_v3_four_table_blind_freeze"
        ):
            raise ValueError("reasoning smoke requires an independent-v3 freeze")
        freeze_source_tree_sha256 = str(freeze.get("source_version", ""))
        if len(freeze_source_tree_sha256) != 64:
            raise ValueError("freeze lacks a valid source-tree identity")
        development = freeze.get("development")
        if not isinstance(development, dict):
            raise ValueError("freeze lacks development binding")
        development_path = _safe_frozen_member(
            manifest.parent,
            development.get("path"),
            "independent-v3 development",
        )
        if (
            not development_path.is_file()
            or development_path.stat().st_size != development.get("size")
            or _sha256(development_path) != development.get("sha256")
        ):
            raise ValueError("frozen development binding failed")

        rows = _load_development(development_path)
        selected = _select_development_rows(
            rows,
            per_category=args.per_category,
        )
        nl = _nl2sql_for_evaluation(
            args.db,
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )
        result = evaluate_set(selected, "agent", args.db, nl=nl, strict=True)
        aggregate = _aggregate_result(result)
        profile = _agent_runtime_profile(
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )
        common = {
            "schema_version": 1,
            "rid": candidate_rid,
            "freeze_rid": freeze_rid,
            "dataset_role": "development_selection_used_smoke",
            "selection_used": True,
            "created_once": True,
            "engine": "agent",
            "per_category": args.per_category,
            "selector": "sha256(normalise_question), first N per category",
            "selector_sha256": _selector_sha256(selected),
            "freeze_manifest_sha256": _sha256(manifest),
            "development_input_sha256": _sha256(development_path),
            "database_sha256": _sha256(args.db),
            "source_tree_sha256": source_tree["included_tree_sha256"],
            "evaluator_sha256": _sha256(Path(__file__)),
            "agent_profile": profile,
            "overall": aggregate,
        }
        private_payload = {
            **common,
            "kind": "private_development_reasoning_smoke",
            "details": result["details"],
        }
        public_payload = {
            **common,
            "kind": "development_reasoning_smoke_public_attestation",
            "contains_questions": False,
            "contains_ids": False,
            "contains_gold_sql": False,
            "contains_predictions": False,
            "contains_error_samples": False,
        }
        _publish_json_once(private_output, private_payload)
        public_payload["private_results_sha256"] = _sha256(private_output)
        _publish_json_once(public_output, public_payload)
        print(
            json.dumps(
                {
                    "rid": candidate_rid,
                    "n": aggregate["n"],
                    "n_correct": aggregate["n_correct"],
                    "accuracy": aggregate["accuracy"],
                    "coverage": aggregate["coverage"],
                    "public_output": str(public_output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
