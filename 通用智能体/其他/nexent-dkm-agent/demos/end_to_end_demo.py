"""Demo: run the full data -> knowledge -> insight closed loop.

Usage:
    python demos/end_to_end_demo.py
    python demos/end_to_end_demo.py --text data/samples/task1_medical_notes.txt
    python demos/end_to_end_demo.py --llm-config .local/llm_config.env --datamate-url http://localhost:18000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.llm_config import load_llm_config
from src.pipelines.end_to_end_pipeline import run_end_to_end_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Task1->Task2->Task3 closed loop.")
    parser.add_argument("--text", default=None, help="Raw medical text input path (defaults to sample).")
    parser.add_argument("--output-root", default=None, help="Output root directory.")
    parser.add_argument("--question", default="高血压有哪些症状和用药？")
    parser.add_argument(
        "--analysis-request",
        default="分析图谱核心枢纽、社区结构与关键路径并生成可视化",
        help="Task-3 analysis request (drives which analytics operators run).",
    )
    parser.add_argument("--llm-config", default=None, help="LLM config path (.env or .json).")
    parser.add_argument(
        "--datamate-url",
        default=None,
        help="DataMate Python backend base URL. Omit or use 'none' to disable.",
    )
    parser.add_argument(
        "--datamate-mode",
        choices=["dry_run", "submit"],
        default="dry_run",
        help="dry_run only prepares payloads; submit posts to DataMate.",
    )
    parser.add_argument("--datamate-timeout", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm_config = load_llm_config(args.llm_config) if args.llm_config else None
    if args.llm_config and not llm_config:
        print("LLM config is missing or incomplete.")
        print(f"  config: {args.llm_config}")
        print("  Required: OPENAI_API_KEY + OPENAI_BASE_URL (.env) or api_key + base_url (.json)")
        return 2

    datamate_url = None
    if args.datamate_url and args.datamate_url.lower() != "none":
        datamate_url = args.datamate_url

    report = run_end_to_end_pipeline(
        text_input=args.text,
        output_root=args.output_root,
        question=args.question,
        analysis_request=args.analysis_request,
        llm_config=llm_config,
        datamate_base_url=datamate_url,
        datamate_timeout=args.datamate_timeout,
        datamate_mode=args.datamate_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
