"""CLI demo entrypoint for task 2."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.llm_config import load_llm_config
from src.pipelines.task2_kg_pipeline import run_task2_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run task 2 medical KG agent.")
    parser.add_argument("--input", default=None, help="Medical text input path.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated graph JSON.")
    parser.add_argument("--question", default="高血压有哪些症状和用药？", help="Question to answer.")
    parser.add_argument("--task-request", default=None, help="Natural language task description.")
    parser.add_argument("--llm-config", default=None, help="LLM config path (.env or .json).")
    parser.add_argument("--local-model", default=None, help="Local model path for planning.")
    parser.add_argument("--serve", action="store_true", help="Start REST API server.")
    parser.add_argument("--host", default="127.0.0.1", help="API host.")
    parser.add_argument("--port", type=int, default=8002, help="API port.")
    parser.add_argument("--neo4j-uri", default=None, help="Neo4j bolt URI (e.g. bolt://localhost:7687).")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username.")
    password_group = parser.add_mutually_exclusive_group()
    password_group.add_argument("--neo4j-password", default=None, help="Neo4j password.")
    password_group.add_argument(
        "--neo4j-password-file",
        default=None,
        help="Read the Neo4j password from an ignored local file.",
    )
    parser.add_argument(
        "--relation-backend",
        default="rule",
        choices=["rule", "cpu", "npu"],
        help="Relation scoring backend: rule (default), cpu (tensorized), npu (Ascend tensorized).",
    )
    parser.add_argument(
        "--graph-file",
        default=None,
        help="Reuse an existing graph JSON for query-only plans (skips rebuild).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.serve:
        from src.pipelines.task2_api_server import serve
        serve(host=args.host, port=args.port)
        return 0

    llm_config = load_llm_config(args.llm_config) if args.llm_config else None
    if args.llm_config and not llm_config:
        print("LLM config is missing or incomplete.")
        print(f"  config: {args.llm_config}")
        print("  Required: OPENAI_API_KEY + OPENAI_BASE_URL (.env) or api_key + base_url (.json)")
        return 2

    neo4j_config = None
    if args.neo4j_uri:
        try:
            neo4j_password = _resolve_neo4j_password(args)
        except ValueError as exc:
            print(str(exc))
            return 2
        neo4j_config = {
            "uri": args.neo4j_uri,
            "user": args.neo4j_user,
            "password": neo4j_password,
        }

    result = run_task2_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        question=args.question,
        task_request=args.task_request,
        llm_config=llm_config,
        local_model_path=args.local_model,
        neo4j_config=neo4j_config,
        relation_backend=args.relation_backend,
        graph_file=args.graph_file,
    )
    artifacts = result.artifacts
    print(f"\n{'='*60}")
    print("  Task 2 Knowledge Graph Agent")
    print(f"{'='*60}")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    if result.status != "completed":
        print(f"  Error: {artifacts.get('error', {}).get('message', 'unknown error')}")
        return 1

    graph = artifacts["graph"]
    qa = artifacts["qa"]
    quality = artifacts["quality_report"]
    metrics = quality["metrics"]
    plan = artifacts.get("plan", {})
    extraction = artifacts.get("extraction", {})

    # Mode summary -- shows what was actually used
    planner_mode = plan.get("planner_mode", "rule")
    llm_chunks = extraction.get("llm_chunks_processed", None)
    has_llm = llm_chunks is not None and llm_chunks > 0
    neo4j_artifact = artifacts.get("neo4j")
    neo4j_connected = (
        neo4j_artifact
        and neo4j_artifact.get("status") == "completed"
    )
    local_model_used = args.local_model and Path(args.local_model).is_dir()

    mode_parts = [f"planner={planner_mode}"]
    mode_parts.append("LLM=active" if has_llm else "LLM=off")
    mode_parts.append("Neo4j=connected" if neo4j_connected else "Neo4j=off")
    mode_parts.append("local_model=active" if local_model_used else "local_model=off")
    relation_scoring = artifacts.get("relation_scoring", {})
    mode_parts.append(f"relation={relation_scoring.get('backend', 'rule')}")
    print(f"\n  [Mode] {' | '.join(mode_parts)}")

    if relation_scoring.get("mode") == "tensorized":
        print(
            f"  [Relation] tensorized backend={relation_scoring.get('scoring_backend')} "
            f"device={relation_scoring.get('scoring_device')} "
            f"candidates={relation_scoring.get('candidate_count')} "
            f"status={relation_scoring.get('status')}"
        )

    # LLM evidence
    if llm_chunks is not None:
        if llm_chunks > 0:
            print(f"  [LLM]  {llm_chunks} chunk(s) processed by LLM")
        else:
            print("  [LLM]  LLM was configured but produced 0 chunks (fell back to rules)")
    elif args.llm_config:
        print("  [LLM]  Config provided but could not be loaded")

    # Graph summary
    print(
        f"\n  [Graph] "
        f"{graph['node_count']} nodes / {graph['edge_count']} edges / "
        f"{graph['triple_count']} triples"
    )
    print(f"  [Output] {graph['output_path']}")

    # Quality report
    print(
        f"  [Quality] "
        f"entities={metrics['entity_total']} / "
        f"relations={metrics.get('relation_type_count', 0)} / "
        f"coverage={metrics.get('relation_coverage', 0)} / "
        f"evidence_edges={metrics['evidence_edge_count']}"
    )

    # QA
    print(f"\n  [QA] {qa['status']}")
    print(f"  [Answer] {qa['answer']}")

    # Plan
    operators = plan.get("operators", [])
    print(f"\n  [Plan] [{planner_mode}] {' -> '.join(operators)}")

    # Neo4j evidence
    if neo4j_artifact:
        if neo4j_connected:
            print(
                f"\n  [Neo4j] Connected to {neo4j_artifact.get('uri', '?')} "
                f"(database: {neo4j_artifact.get('database', 'neo4j')})"
            )
            print(
                f"  [Neo4j] Persisted {neo4j_artifact.get('node_count', 0)} nodes, "
                f"{neo4j_artifact.get('edge_count', 0)} edges"
            )
        else:
            reason = neo4j_artifact.get("message", neo4j_artifact.get("status", "unknown"))
            print(f"\n  [Neo4j] Not connected: {reason}")

    # Run state
    run_state = artifacts.get("run_state", {})
    step_summary = ", ".join(
        step["name"] + ":" + step["status"]
        for step in run_state.get("steps", [])
    )
    print(f"\n  [Steps] {run_state.get('status', '?')} [{step_summary}]")
    print(f"{'='*60}")
    return 0


def _resolve_neo4j_password(args: argparse.Namespace) -> str:
    if args.neo4j_password_file:
        password = Path(args.neo4j_password_file).read_text(
            encoding="utf-8"
        ).strip()
        if not password:
            raise ValueError("Neo4j password file must not be empty.")
        return password
    if args.neo4j_password:
        return args.neo4j_password
    raise ValueError(
        "Neo4j password is required when --neo4j-uri is set; use "
        "--neo4j-password-file or --neo4j-password."
    )


if __name__ == "__main__":
    raise SystemExit(main())
