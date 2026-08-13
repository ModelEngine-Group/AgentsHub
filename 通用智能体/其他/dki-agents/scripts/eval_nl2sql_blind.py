"""Evaluate a sealed NL2SQL blind set under a create-once protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.database import database_matches_manifest
from dki.analysis.nl2sql_eval import evaluate_set
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_protocol import execute_complete_readonly, validate_gold_sql
from dki.analysis.nl2sql_seal import verify_candidate_seal
from dki.pipeline.artifacts import manifest_path, validate_task2_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _rid_below(path: Path, scope: str) -> str:
    parts = Path(path).resolve().parts
    for index in range(len(parts) - 2):
        if (
            parts[index].casefold() == "outputs"
            and parts[index + 1].casefold() == scope.casefold()
        ):
            return parts[index + 2]
    raise ValueError(f"artifact is not below outputs/{scope}/<RID>: {path}")


def _write_json_create_once(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise SystemExit(f"artifact already exists and must not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_once_outputs(
    manifest: Path,
    private_output: Path,
    public_output: Path,
) -> tuple[Path, Path, Path, str, str]:
    # POSIX marker string: comparing against ``as_posix()`` output must not
    # depend on the host separator (``Path("outputs/private/")`` stringifies
    # with backslashes on Windows and would never match).
    private_marker = "outputs/private/"
    manifest_resolved = manifest.resolve()
    private_output = private_output.resolve()
    public_output = public_output.resolve()
    try:
        manifest_rid = _rid_below(manifest_resolved, "private")
        private_rid = _rid_below(private_output, "private")
        public_rid = _rid_below(public_output, "remote_runs")
    except ValueError:
        manifest_rid = None
        private_rid = None
        public_rid = None
    if (
        manifest_rid is None
        or private_rid is None
        or public_rid is None
        or private_marker not in manifest_resolved.as_posix()
        or private_marker not in private_output.as_posix()
    ):
        raise SystemExit(
            "blind manifest and private/evaluation outputs must be under "
            "outputs/private/<RID>"
        )
    if manifest_rid != private_rid:
        raise SystemExit(
            "manifest and private detailed output must share the freeze RID"
        )
    manifest_dir = manifest_resolved.parent
    consumption_receipt = manifest_dir / "evaluation_receipt.json"
    started_receipt = manifest_dir / "consumed-started.json"
    invalid_receipt = manifest_dir / "consumed-invalid.json"
    occupied = [
        str(path)
        for path in (
            private_output,
            public_output,
            consumption_receipt,
            started_receipt,
            invalid_receipt,
        )
        if path.exists()
    ]
    if occupied:
        raise SystemExit(
            "create-once NL2SQL blind-eval output collision:\n  "
            + "\n  ".join(occupied)
        )
    return (
        consumption_receipt,
        started_receipt,
        invalid_receipt,
        manifest_rid,
        public_rid,
    )


def _evaluate_gates(result: dict, hard_details: list[dict]) -> dict:
    n_total = int(result["n"])
    n_hard = int(len(hard_details))
    n_hard_correct = int(sum(bool(detail.get("correct")) for detail in hard_details))
    n_hard_answered = int(
        sum(detail.get("engine") not in (None, "none") for detail in hard_details)
    )
    # The prospective overall gate is intentionally strict.  At 120 items,
    # 102 correct is exactly 85% and must fail; 103 is the first passing
    # numerator.  Keep this in integer arithmetic so presentation rounding
    # cannot change a release decision.
    overall_passed = n_total > 0 and (
        int(result["n_correct"]) * 100 > 85 * n_total
    )
    hard_passed = n_hard > 0 and (
        n_hard_correct * 100 >= 70 * n_hard
    )
    hard_subset_present = n_hard >= 40
    failures = []
    if not overall_passed:
        failures.append("overall_failed")
    if not hard_subset_present:
        failures.append("hard_subset_insufficient")
    if not hard_passed:
        failures.append("hard_failed")
    return {
        "overall_threshold": 0.85,
        "hard_threshold": 0.70,
        "overall_comparison": "100 * correct > 85 * n",
        "hard_comparison": "100 * hard_correct >= 70 * hard_n",
        "overall_passed": overall_passed,
        "hard_passed": hard_passed,
        "hard_subset_present": hard_subset_present,
        "hard_n": n_hard,
        "passed": bool(overall_passed and hard_passed and hard_subset_present),
        "overall_numerator": int(result["n_correct"]),
        "overall_denominator": n_total,
        "hard_numerator": n_hard_correct,
        "hard_denominator": n_hard,
        "hard_answered": n_hard_answered,
        "hard_coverage_numerator": n_hard_answered,
        "hard_coverage_denominator": n_hard,
        "gate_reason": "passed" if not failures else "+".join(failures),
    }


def _build_hard_result(
    result: dict,
    hard_details: list[dict],
) -> dict:
    n_hard = len(hard_details)
    n_hard_correct = int(sum(bool(detail.get("correct")) for detail in hard_details))
    n_hard_answered = int(
        sum(detail.get("engine") not in (None, "none") for detail in hard_details)
    )
    if n_hard == 0:
        accuracy = 0.0
        coverage = 0.0
    else:
        accuracy = n_hard_correct / n_hard
        coverage = n_hard_answered / n_hard
    return {
        "n": n_hard,
        "n_correct": n_hard_correct,
        "n_answered": n_hard_answered,
        "n_abstained": n_hard - n_hard_answered,
        "accuracy": accuracy,
        "coverage": coverage,
        "accuracy_num": n_hard_correct,
        "accuracy_den": n_hard,
        "coverage_num": n_hard_answered,
        "coverage_den": n_hard,
        "derived_from_overall_single_inference": True,
    }


def _nl2sql_for_evaluation(
    database: Path,
    *,
    engine: str,
    llm_strategy: str,
    agent_rule_policy: str,
    llm_reasoning_mode: str = "disabled",
    llm_completion_tokens: int = 1024,
) -> NL2SQL:
    if engine != "agent" and (
        llm_strategy != "legacy_sql"
        or agent_rule_policy != "rule_first"
        or llm_reasoning_mode != "disabled"
        or llm_completion_tokens != 1024
    ):
        raise ValueError("non-default Agent LLM options require engine=agent")
    if engine == "agent" and (llm_strategy, agent_rule_policy) not in {
        ("legacy_sql", "rule_first"),
        ("legacy_sql", "model_first"),
        ("query_plan", "rule_first"),
        ("query_plan", "plan_first"),
        ("query_plan_v2", "rule_first"),
        ("query_plan_v2", "plan_first"),
        ("query_plan_v3", "rule_first"),
        ("query_plan_v3", "plan_first"),
    }:
        raise ValueError("unsupported Agent LLM strategy/rule-policy combination")
    return NL2SQL(
        db_path=database,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
        llm_fail_closed=(engine == "agent"),
    )


def _preflight_agent_runtime(
    selected_nl2sql: NL2SQL,
    *,
    engine: str,
    database: Path,
    llm_reasoning_mode: str,
    llm_completion_tokens: int,
) -> dict:
    """Probe the sealed provider before creating the consumption marker.

    The constant probe contains no development or holdout text.  It verifies
    the exact adapter, credentials and model route with a read-only query, so
    provider/import/auth/timeout failures remain safe pre-marker failures.
    """

    if engine != "agent":
        return {"status": "not_applicable", "engine": engine}
    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("Agent runtime dependency openai is not importable")

    from dki.llm import chat, llm_available

    if not llm_available():
        raise RuntimeError("Agent runtime credentials/model/endpoint are not configured")
    started = time.monotonic()
    response = chat(
        "Return only this read-only SQLite query with no explanation: SELECT 1;",
        temperature=0.0,
        max_tokens=min(512, llm_completion_tokens),
        timeout=45,
        reasoning_mode=llm_reasoning_mode,
    )
    sql = selected_nl2sql._clean_sql(response)
    if sql is None:
        raise RuntimeError("Agent runtime preflight did not return valid read-only SQL")
    validate_gold_sql(sql)
    columns, rows = execute_complete_readonly(sql, database)
    if len(columns) != 1 or rows != [(1,)]:
        raise RuntimeError("Agent runtime preflight returned the wrong SELECT 1 result")
    return {
        "status": "passed",
        "engine": engine,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "question_or_holdout_content_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-seal", type=Path, required=True)
    parser.add_argument("--engine", choices=["rule", "agent"], default="rule")
    parser.add_argument(
        "--llm-strategy",
        choices=["legacy_sql", "query_plan", "query_plan_v2", "query_plan_v3"],
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
    parser.add_argument("--db", type=Path, default=ROOT / "outputs" / "insight.db")
    parser.add_argument("--kg", type=Path, default=ROOT / "outputs" / "kg" / "medical_kg.json")
    parser.add_argument("--task1-manifest", type=Path, default=manifest_path())
    parser.add_argument("--private-output", type=Path, required=True, help="private per-item detailed result")
    parser.add_argument("--public-output", type=Path, required=True, help="public aggregate attestation")
    args = parser.parse_args()

    (
        consumption_receipt,
        started_receipt,
        invalid_receipt,
        freeze_rid,
        public_rid,
    ) = _require_once_outputs(
        args.manifest.resolve(),
        args.private_output.resolve(),
        args.public_output.resolve(),
    )

    try:
        selected_nl2sql = _nl2sql_for_evaluation(
            args.db,
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )
        freeze = _load_json(args.manifest)
        if freeze.get("schema_version") not in {2, 3, 4}:
            raise ValueError("blind freeze schema version is unsupported")
        if not freeze.get("created_once"):
            raise ValueError("blind freeze is not create-once")
        if freeze.get("selection_used") is not False:
            raise ValueError("blind freeze selection_used must be false")
        if not isinstance(freeze.get("holdout"), dict):
            raise ValueError("blind freeze holdout declaration is missing")

        try:
            candidate_seal = verify_candidate_seal(
                args.candidate_seal,
                freeze_manifest=args.manifest,
                task1_manifest=args.task1_manifest,
                kg_path=args.kg,
                db_path=args.db,
                engine=args.engine,
                llm_strategy=args.llm_strategy,
                agent_rule_policy=args.agent_rule_policy,
                llm_reasoning_mode=args.llm_reasoning_mode,
                llm_completion_tokens=args.llm_completion_tokens,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"pre-consumption candidate-seal verification failed: {exc}") from exc

        candidate_rid = candidate_seal.get("candidate_rid")
        if candidate_rid is not None and candidate_rid != public_rid:
            raise RuntimeError(
                "public blind attestation must use the sealed candidate RID"
            )
        sealed_freeze_rid = candidate_seal.get("freeze_rid")
        if sealed_freeze_rid is not None and sealed_freeze_rid != freeze_rid:
            raise RuntimeError("candidate seal freeze RID differs from manifest")

        agent_runtime_preflight = _preflight_agent_runtime(
            selected_nl2sql,
            engine=args.engine,
            database=args.db,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )

        # create marker before touching holdout content so partial/aborted runs are
        # treated as consumed-invalid after this point
        _write_json_create_once(
            started_receipt,
            {
                "schema_version": 1,
                "status": "consumed-started",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "freeze_manifest_sha256": _sha256(args.manifest),
                "candidate_seal_sha256": _sha256(args.candidate_seal),
                "source_tree_sha256": candidate_seal["source_tree_sha256"],
                "candidate_rid": candidate_rid or public_rid,
                "freeze_rid": freeze_rid,
                "agent_profile": candidate_seal.get("agent_profile"),
                "agent_runtime_preflight": agent_runtime_preflight,
            },
        )

        holdout_path = args.manifest.parent / freeze["holdout"]["path"]
        holdout_declared_sha256 = str(freeze["holdout"].get("sha256", ""))
        if not holdout_path.is_file():
            raise FileNotFoundError(f"holdout not found: {holdout_path}")
        if _sha256(holdout_path) != holdout_declared_sha256:
            raise RuntimeError("holdout hash differs from freeze manifest")
        items = _load_jsonl(holdout_path)
        if len(items) != freeze.get("n_holdout"):
            raise RuntimeError("holdout count differs from freeze manifest")
        if len(items) < 120:
            raise RuntimeError("holdout has fewer than 120 items")

        validate_task2_artifact(args.kg, task1_path=args.task1_manifest)
        if not database_matches_manifest(args.db, args.task1_manifest, args.kg):
            raise RuntimeError("database does not match sealed Task 1 / KG provenance")

        result = evaluate_set(
            items,
            args.engine,
            args.db,
            nl=selected_nl2sql,
            strict=True,
            # Parallelize only LLM prediction. evaluate_set replays predictions
            # through serial SQL execution/scoring, preserving query budgets and
            # result order while avoiding a mutable NL2SQL instance across threads.
            nl_factory=lambda: _nl2sql_for_evaluation(
                args.db,
                engine=args.engine,
                llm_strategy=args.llm_strategy,
                agent_rule_policy=args.agent_rule_policy,
                llm_reasoning_mode=args.llm_reasoning_mode,
                llm_completion_tokens=args.llm_completion_tokens,
            ),
        )
        hard_details = [
            detail
            for detail in result["details"]
            if str(detail.get("difficulty", "")).lower()
            in {"hard", "extra", "extra-hard"}
        ]
        gates = _evaluate_gates(result, hard_details)
        hard_result = _build_hard_result(result, hard_details)
        if hard_result["n"] < 40:
            gates["hard_subset_present"] = False
            gates["passed"] = False

        private_payload = {
            "schema_version": 1,
            "kind": "private_detailed_results",
            "freeze_manifest_sha256": _sha256(args.manifest),
            "candidate_seal_sha256": _sha256(args.candidate_seal),
            "task1_manifest_sha256": _sha256(args.task1_manifest),
            "kg_sha256": _sha256(args.kg),
            "database_sha256": _sha256(args.db),
            "evaluator_sha256": _sha256(Path(__file__)),
            "agent_profile": candidate_seal.get("agent_profile"),
            "details": result["details"],
            "overall": {
                "n": result["n"],
                "n_correct": result["n_correct"],
                "n_answered": result["n_answered"],
                "coverage_num": result["n_answered"],
                "coverage_den": result["n"],
            },
            "hard": hard_result,
        }

        public_payload = {
            "schema_version": 1,
            "kind": "blind_evaluation_public_attestation",
            "status": "consumed",
            "gate_passed": gates["passed"],
            "freeze_manifest": str(args.manifest),
            "freeze_manifest_sha256": _sha256(args.manifest),
            "candidate_seal_sha256": _sha256(args.candidate_seal),
            "source_tree_sha256": candidate_seal["source_tree_sha256"],
            "candidate_rid": candidate_rid or public_rid,
            "freeze_rid": freeze_rid,
            "holdout_sha256": _sha256(holdout_path),
            "engine": args.engine,
            "agent_profile": candidate_seal.get("agent_profile"),
            "task1_manifest_sha256": _sha256(args.task1_manifest),
            "kg_sha256": _sha256(args.kg),
            "database_sha256": _sha256(args.db),
            "evaluator_sha256": _sha256(Path(__file__)),
            "overall": {
                "n": result["n"],
                "n_correct": result["n_correct"],
                "n_answered": result["n_answered"],
                "n_abstained": result["n_abstained"],
                "accuracy_num": result["n_correct"],
                "accuracy_den": result["n"],
                "coverage_num": result["n_answered"],
                "coverage_den": result["n"],
                "execution_accuracy_answered_num": result["n_correct"],
                "execution_accuracy_answered_den": result["n_answered"],
                "coverage": result["coverage"],
                "strict_result_contract": result["strict_result_contract"],
                "by_difficulty": result["by_difficulty"],
            },
            "hard": hard_result,
            "hard_subset_present": gates["hard_subset_present"],
            "hard_n": gates["hard_n"],
            "hard_accuracy_num": gates["hard_numerator"],
            "hard_accuracy_den": gates["hard_denominator"],
            "hard_coverage_num": gates["hard_coverage_numerator"],
            "hard_coverage_den": gates["hard_coverage_denominator"],
            "gate": gates,
            "gate_reason": gates["gate_reason"],
            "public_attestation_rules": {
                "strict_scoring": True,
                "no_question_or_gold_sql_leakage": True,
                "single_inference_pass": True,
                "selection_used": False,
            },
        }

        _write_json_create_once(args.private_output, private_payload)
        _write_json_create_once(args.public_output, public_payload)
        _write_json_create_once(
            consumption_receipt,
            {
                "schema_version": 1,
                "status": "consumed",
                "consumed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "freeze_manifest_sha256": _sha256(args.manifest),
                "candidate_seal_sha256": _sha256(args.candidate_seal),
                "holdout_sha256": _sha256(holdout_path),
                "private_detailed_path": str(args.private_output.resolve()),
                "private_detailed_sha256": _sha256(args.private_output),
                "public_attestation_path": str(args.public_output.resolve()),
                "public_attestation_sha256": _sha256(args.public_output),
                "gate_passed": gates["passed"],
                "overall_accuracy_num": result["n_correct"],
                "overall_accuracy_den": result["n"],
                "hard_accuracy_num": hard_result["n_correct"],
                "hard_accuracy_den": hard_result["n"],
                "agent_profile": candidate_seal.get("agent_profile"),
            },
        )
    except Exception as exc:
        # If execution reaches here after started marker, this is a consumed-invalid
        # scenario and should never be reused.
        if started_receipt.exists() and not invalid_receipt.exists():
            _write_json_create_once(
                invalid_receipt,
                {
                    "schema_version": 1,
                    "status": "consumed-invalid",
                    "invalidated_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),
                    "freeze_manifest_sha256": _sha256(args.manifest)
                    if args.manifest.is_file()
                    else None,
                    "candidate_seal_sha256": _sha256(args.candidate_seal)
                    if args.candidate_seal.is_file()
                    else None,
                    "failure_class": type(exc).__name__,
                    "failure_message": str(exc),
                    "instructions": "retry requires new sealed holdout",
                },
            )
        raise

    print(
        json.dumps(
            {
                "accuracy": result["accuracy"],
                "coverage": result["coverage"],
                "abstention_rate": 1 - result["coverage"],
                "hard_accuracy": hard_result["accuracy"],
                "gate_passed": gates["passed"],
                "output": str(args.public_output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if gates["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
