"""CLI demo entrypoint for task 3."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.llm_config import load_llm_config
from src.pipelines.task3_insight_pipeline import run_task3_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run task 3 graph-driven analysis agent.")
    parser.add_argument("--graph-file", default=None, help="Task-2 graph JSON path.")
    parser.add_argument("--output-dir", default=None, help="Directory for task-3 analysis output.")
    parser.add_argument("--question", default="哪些疾病关联最多症状？")
    parser.add_argument(
        "--task-request",
        default="分析图谱核心枢纽、社区结构与关键路径并生成可视化",
        help="Analysis request that drives which analytics operators run.",
    )
    parser.add_argument("--llm-config", default=None, help="LLM config path (.env or .json).")
    parser.add_argument("--local-model", default=None, help="Path to a locally fine-tuned model for planning and NL2SQL.")
    parser.add_argument("--serve", action="store_true", help="Start REST API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.serve:
        from src.pipelines.task3_api_server import serve, set_llm_config

        llm_config = load_llm_config(args.llm_config) if args.llm_config else None
        set_llm_config(llm_config)
        serve(host=args.host, port=args.port)
        return 0

    llm_config = load_llm_config(args.llm_config) if args.llm_config else None
    if args.llm_config and not llm_config:
        print("LLM config is missing or incomplete.")
        print(f"  config: {args.llm_config}")
        print("  Required: OPENAI_API_KEY + OPENAI_BASE_URL")
        return 2

    result = run_task3_pipeline(
        graph_file=args.graph_file,
        output_dir=args.output_dir,
        question=args.question,
        task_request=args.task_request,
        llm_config=llm_config,
        local_model_path=args.local_model,
    )

    print(f"\n{'='*60}")
    print("  Task 3 Graph Analysis Agent")
    print(f"{'='*60}")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")

    if result.status != "completed":
        print(f"  Error: {result.artifacts.get('error', {}).get('message', 'unknown error')}")
        return 1

    artifacts = result.artifacts
    graph = artifacts["graph"]
    quality = artifacts["quality_report"]
    nl2sql = artifacts["nl2sql"]
    plan = artifacts.get("plan", {})
    centrality = artifacts.get("centrality", {})

    # Mode summary
    planner_mode = plan.get("planner_mode", "rule")
    translator = nl2sql.get("translator", "template")
    mode_parts = [f"planner={planner_mode}", f"NL2SQL={translator}"]
    if args.local_model:
        mode_parts.append("local_model=active")
    if llm_config:
        mode_parts.append("LLM=active")
    print(f"\n  [Mode] {' | '.join(mode_parts)}")

    # Graph summary
    print(f"\n  [Graph] {graph.get('node_count', 0)} nodes / {graph.get('edge_count', 0)} edges")

    # Centrality
    top_hubs = centrality.get("top_hubs", [])[:3]
    if top_hubs:
        hub_str = ", ".join(f"{h['name']}({h['degree']})" for h in top_hubs)
        print(f"  [Hubs] {hub_str}")
    print(f"  [Hubs Backend] {centrality.get('top_hubs_backend', 'unknown')}")

    # NL2SQL
    print(f"\n  [NL2SQL] {nl2sql['intent']} (via {translator})")
    if nl2sql.get("rows"):
        print(f"  [SQL Result] {len(nl2sql['rows'])} rows -- top: {nl2sql['rows'][0]}")

    # Visualizations
    charts = artifacts["visualizations"]["charts"]
    print(f"\n  [Charts] {', '.join(charts.keys())}")

    # Quality
    print(f"\n  [Quality] {quality['status']}")

    # Outputs
    print(f"\n  [Output] {artifacts['export']['output_path']}")
    print(f"  [Report] {artifacts['insight_report']['html_path']}")
    print(f"  [Dashboard] {artifacts['insight_report']['dashboard_path']}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
