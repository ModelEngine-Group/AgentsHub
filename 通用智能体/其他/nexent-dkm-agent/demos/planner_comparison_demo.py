"""CLI demo comparing rule vs enhanced planner outputs across DKM tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.planner_comparison import compare_all_planners
from src.common.llm_config import load_llm_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rule-based and enhanced planner outputs for tasks 1-3."
    )
    parser.add_argument("--llm", action="store_true", help="Include LLM-enhanced plans.")
    parser.add_argument("--llm-config", default=None)
    parser.add_argument("--local-model", default=None)
    parser.add_argument(
        "--graph-file",
        default=None,
        help="Optional task-2 graph JSON for task-3 planning context.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report output path.",
    )
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

    report = compare_all_planners(
        llm_config=llm_config,
        local_model_path=args.local_model,
    )
    if args.graph_file:
        from src.agents.planner_comparison import compare_task3_planners

        report["task3"] = compare_task3_planners(
            "分析图谱统计、关联与可视化洞察",
            question="哪些疾病关联最多症状？",
            graph_file=args.graph_file,
            llm_config=llm_config,
            local_model_path=args.local_model,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
