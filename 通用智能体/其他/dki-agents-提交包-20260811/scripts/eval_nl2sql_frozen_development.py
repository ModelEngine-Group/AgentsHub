"""Evaluate only the development member of an unconsumed independent freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.database import database_matches_manifest
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_development_evidence import (
    strict_development_jsonl_rows,
)
from dki.analysis.nl2sql_eval import evaluate_set
from dki.analysis.nl2sql_protocol import (
    normalise_development_metadata,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.analysis.nl2sql_seal import (
    _agent_runtime_profile,
    _assert_freeze_unconsumed,
    _freeze_contract,
    _safe_frozen_member,
    _verify_source_tree_manifest,
)
from dki.pipeline.artifacts import validate_task2_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rid_below(path: Path, scope: str) -> str:
    parts = Path(path).resolve().parts
    for index in range(len(parts) - 2):
        if (
            parts[index].casefold() == "outputs"
            and parts[index + 1].casefold() == scope.casefold()
        ):
            return parts[index + 2]
    raise ValueError(f"artifact is not below outputs/{scope}/<RID>: {path}")


def _publish_json_once(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"create-once artifact exists: {path}")
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


def _load_development(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    questions: set[str] = set()
    for raw in strict_development_jsonl_rows(
        path.read_bytes(),
        label="independent frozen development",
    ):
        row = normalise_development_metadata(raw)
        validate_gold_sql(row["gold_sql"])
        validate_question_sql_grounding(row["question"], row["gold_sql"])
        item_id = str(row["id"])
        question_key = normalise_question(str(row["question"]))
        if item_id in ids or question_key in questions:
            raise ValueError("development ids and normalised questions must be unique")
        ids.add(item_id)
        questions.add(question_key)
        rows.append(row)
    return rows


def _validate_engine_strategy(
    *,
    engine: str,
    llm_strategy: str,
    agent_rule_policy: str,
    llm_reasoning_mode: str = "disabled",
    llm_completion_tokens: int = 1024,
    llm_rounds: int = 2,
) -> None:
    if llm_reasoning_mode not in {"disabled", "adaptive"}:
        raise ValueError("unsupported Agent LLM reasoning mode")
    if (
        type(llm_completion_tokens) is not int
        or not 1 <= llm_completion_tokens <= 131072
    ):
        raise ValueError(
            "Agent LLM completion tokens must be an integer between 1 and 131072"
        )
    if type(llm_rounds) is not int or not 1 <= llm_rounds <= 6:
        raise ValueError("Agent LLM rounds must be an integer between 1 and 6")
    if engine != "agent" and (
        llm_strategy != "legacy_sql"
        or agent_rule_policy != "rule_first"
        or llm_reasoning_mode != "disabled"
        or llm_completion_tokens != 1024
    ):
        raise ValueError(
            "non-default Agent LLM options require engine=agent"
        )
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


def _nl2sql_for_evaluation(
    db_path: Path,
    *,
    engine: str,
    llm_strategy: str,
    agent_rule_policy: str,
    llm_reasoning_mode: str = "disabled",
    llm_completion_tokens: int = 1024,
    llm_rounds: int = 2,
    ablation_config=None,
) -> NL2SQL:
    """Build the exact product engine whose strategy is recorded in evidence."""

    _validate_engine_strategy(
        engine=engine,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
        llm_rounds=llm_rounds,
    )
    constructor_kwargs = dict(
        db_path=db_path,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
        llm_rounds=llm_rounds,
        llm_fail_closed=(engine == "agent"),
    )
    # Only ablation arms carry the extra option; the default evaluation
    # engine keeps the exact historical constructor binding.
    if ablation_config is not None:
        constructor_kwargs["ablation_config"] = ablation_config
    return NL2SQL(**constructor_kwargs)


def _load_ablation_config(path: Path):
    """Load a fail-closed rule-ablation manifest for one evaluation arm."""

    from dki.analysis.nl2sql_ablation import NL2SQLAblationConfig

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NL2SQLAblationConfig.from_manifest_payload(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
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
    parser.add_argument(
        "--llm-completion-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--llm-rounds",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ablation-manifest",
        type=Path,
        default=None,
        help=(
            "Optional rule-ablation arm manifest (profile_name + cluster/rule "
            "switches); its policy and profile hashes are bound into evidence."
        ),
    )
    args = parser.parse_args()

    try:
        _validate_engine_strategy(
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
            llm_rounds=args.llm_rounds,
        )
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
            raise ValueError("included-tree manifest lacks a safe candidate RID")
        if (
            _rid_below(private_output, "private") != candidate_rid
            or _rid_below(public_output, "remote_runs") != candidate_rid
        ):
            raise ValueError(
                "private/public development artifacts must use the candidate RID"
            )
        expected_private = (
            ROOT
            / "outputs"
            / "private"
            / candidate_rid
            / "nl2sql_development"
            / freeze_rid
            / "development_selection_used.json"
        )
        expected_public = (
            ROOT
            / "outputs"
            / "remote_runs"
            / candidate_rid
            / "nl2sql"
            / "independent_v3_development_attestation.json"
        )
        if private_output != expected_private:
            raise ValueError(
                "private output must be candidate-RID scoped and freeze-RID bound"
            )
        if public_output != expected_public.resolve():
            raise ValueError(
                "public output must be the fixed independent-v3 aggregate path"
            )
        if private_output.exists() or public_output.exists():
            raise FileExistsError("development evaluation output already exists")

        _assert_freeze_unconsumed(manifest)
        validate_task2_artifact(args.kg, task1_path=args.task1_manifest)
        if not database_matches_manifest(
            args.db,
            args.task1_manifest,
            args.kg,
        ):
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
            raise ValueError("development evaluator requires an independent-v3 freeze")
        source_tree_sha256 = source_tree["included_tree_sha256"]
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
        ablation_config = (
            _load_ablation_config(args.ablation_manifest)
            if args.ablation_manifest is not None
            else None
        )
        nl = _nl2sql_for_evaluation(
            args.db,
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
            llm_rounds=args.llm_rounds,
            ablation_config=ablation_config,
        )
        agent_profile = (
            _agent_runtime_profile(
                llm_strategy=args.llm_strategy,
                agent_rule_policy=args.agent_rule_policy,
                llm_reasoning_mode=args.llm_reasoning_mode,
                llm_completion_tokens=args.llm_completion_tokens,
            )
            if args.engine == "agent"
            else None
        )
        result = evaluate_set(
            rows,
            args.engine,
            args.db,
            nl=nl,
            strict=True,
            # Each worker needs its own engine because NL2SQL keeps per-query
            # state; the factory reproduces the exact evaluated configuration.
            nl_factory=lambda: _nl2sql_for_evaluation(
                args.db,
                engine=args.engine,
                llm_strategy=args.llm_strategy,
                agent_rule_policy=args.agent_rule_policy,
                llm_reasoning_mode=args.llm_reasoning_mode,
                llm_completion_tokens=args.llm_completion_tokens,
                llm_rounds=args.llm_rounds,
                ablation_config=ablation_config,
            ),
        )
        overall = {key: value for key, value in result.items() if key != "details"}
        common = {
            "schema_version": 2,
            "rid": candidate_rid,
            "candidate_rid": candidate_rid,
            "freeze_rid": freeze_rid,
            "dataset_role": "development_selection_used",
            "selection_used": True,
            "created_once": True,
            "engine": args.engine,
            "development_evidence_kind": (
                "independent_curated_v3_four_table_development_receipt"
            ),
            "freeze_manifest_sha256": _sha256(manifest),
            "input_sha256": _sha256(development_path),
            "database_sha256": _sha256(args.db),
            "engine_source_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql.py"
            ),
            "source_tree_sha256": source_tree_sha256,
            "freeze_source_tree_sha256": freeze_source_tree_sha256,
            "candidate_source_differs_from_freeze": (
                source_tree_sha256 != freeze_source_tree_sha256
            ),
            "evaluator_sha256": _sha256(Path(__file__)),
            "protocol_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
            ),
            "overall": overall,
        }
        if agent_profile is not None:
            common["agent_profile"] = agent_profile
        if ablation_config is not None:
            common["ablation"] = {
                "manifest_sha256": _sha256(args.ablation_manifest),
                "profile_name": ablation_config.profile_name,
                "policy_hash": ablation_config.policy_hash,
                "profile_hash": ablation_config.profile_hash,
                "registry_hash": ablation_config.registry.registry_hash,
            }
        private_payload = {
            **common,
            "kind": "private_independent_development_results",
            "details": result["details"],
        }
        public_payload = {
            **common,
            "kind": "independent_development_public_attestation",
            "contains_questions": False,
            "contains_gold_sql": False,
            "contains_predictions": False,
            "contains_error_samples": False,
        }
        _publish_json_once(private_output, private_payload)
        public_payload["private_results_path"] = str(private_output)
        public_payload["private_results_sha256"] = _sha256(private_output)
        _publish_json_once(public_output, public_payload)
        print(
            json.dumps(
                {
                    "rid": candidate_rid,
                    "freeze_rid": freeze_rid,
                    "n": result["n"],
                    "n_correct": result["n_correct"],
                    "accuracy": result["accuracy"],
                    "coverage": result["coverage"],
                    "private_output": str(private_output),
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
