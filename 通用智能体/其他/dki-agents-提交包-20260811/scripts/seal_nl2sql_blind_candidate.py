"""Create the immutable pre-consumption seal for NL2SQL blind evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_seal import seal_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument(
        "--execution-bundle",
        type=Path,
        required=True,
        help="create-once static import-closure bundle for the sealed runtime",
    )
    parser.add_argument(
        "--dependency-spec",
        type=Path,
        default=ROOT / "pyproject.toml",
    )
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--engine", choices=["rule", "agent"], default="rule")
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
    args = parser.parse_args()
    if "outputs/private/" in args.output.resolve().as_posix():
        raise SystemExit("candidate seal is an aggregate public evidence artifact")
    try:
        payload = seal_candidate(
            output=args.output,
            freeze_manifest=args.manifest,
            development_receipt=args.development_receipt,
            regression_receipt=args.regression_receipt,
            task1_manifest=args.task1_manifest,
            kg_path=args.kg,
            db_path=args.db,
            source_tree_manifest=args.source_tree_manifest,
            execution_bundle=args.execution_bundle,
            dependency_spec=args.dependency_spec,
            dependency_lock=args.dependency_lock,
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "kind": payload["kind"],
                "status": payload["status"],
                "engine": payload["engine"],
                "source_tree_sha256": payload["source_tree_sha256"],
                "holdout_opened_by_sealer": False,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
