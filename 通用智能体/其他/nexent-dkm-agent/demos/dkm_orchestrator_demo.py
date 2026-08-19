"""CLI demo for the cross-task DKM orchestrator."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.dkm_orchestrator import DKMOrchestrator, plan_dkm_workflow
from src.agents.planner_comparison import compare_dkm_orchestrator_planners
from src.common.llm_config import load_llm_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DKM cross-task orchestrator.")
    parser.add_argument(
        "--request",
        default="请清洗医疗文本，构建知识图谱并生成分析洞察",
        help="Natural-language DKM workflow request.",
    )
    parser.add_argument("--question", default=None, help="Optional QA / NL2SQL question.")
    parser.add_argument("--input", default=None, help="Optional task-1 input path.")
    parser.add_argument("--graph-file", default=None, help="Optional task-3 graph JSON path.")
    parser.add_argument("--output-dir", default=None, help="Output root for orchestrated stages.")
    parser.add_argument("--datamate-url", default="http://localhost:18000")
    parser.add_argument("--datamate-mode", choices=["dry_run", "submit"], default="dry_run")
    parser.add_argument("--plan-only", action="store_true", help="Only print the workflow plan.")
    parser.add_argument(
        "--compare-planners",
        action="store_true",
        help="Print rule vs enhanced DKM orchestrator plans side by side.",
    )
    parser.add_argument("--llm", action="store_true", help="Use LLM-assisted DKM planning.")
    parser.add_argument("--llm-config", default=None)
    parser.add_argument("--local-model", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm_config = None
    if args.llm:
        config_path = args.llm_config or str(ROOT / ".local" / "llm_config.env")
        llm_config = load_llm_config(config_path)
        if not llm_config:
            print("LLM mode requested but config is missing or invalid.")
            return 2

    if args.plan_only:
        plan = plan_dkm_workflow(args.request, question=args.question, llm_config=llm_config)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.compare_planners:
        comparison = compare_dkm_orchestrator_planners(
            args.request,
            question=args.question,
            llm_config=llm_config,
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0

    datamate_url = None if args.datamate_url.lower() == "none" else args.datamate_url
    orchestrator = DKMOrchestrator(
        llm_config=llm_config,
        local_model_path=args.local_model,
        datamate_base_url=datamate_url,
        datamate_mode=args.datamate_mode,
    )
    result = orchestrator.run(
        request=args.request,
        output_root=args.output_dir,
        question=args.question,
        text_input=args.input,
        graph_file=args.graph_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
