"""Collect rule vs LLM planner evidence for competition defense.

Tries live LLM planning when `--llm-config` is reachable; otherwise merges the
recorded snapshot in `benchmarks/data/planner_llm_snapshot.json`.

Usage:
    python demos/planner_llm_evidence_demo.py
    python demos/planner_llm_evidence_demo.py --llm --llm-config .local/llm_config.env
    python demos/planner_llm_evidence_demo.py --output benchmarks/reports/planner_llm_evidence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.planner_comparison import collect_planner_llm_evidence
from src.common.llm_config import load_llm_config

DEFAULT_SNAPSHOT = ROOT / "benchmarks" / "data" / "planner_llm_snapshot.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "reports" / "planner_llm_evidence.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rule vs LLM planner evidence.")
    parser.add_argument("--llm", action="store_true", help="Attempt live LLM-enhanced planning.")
    parser.add_argument("--llm-config", default=None, help="LLM config (.env or .json).")
    parser.add_argument("--local-model", default=None, help="Optional local model adapter path.")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT), help="Recorded LLM fallback JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Evidence JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm_config = None
    if args.llm:
        config_path = args.llm_config or str(ROOT / ".local" / "llm_config.env")
        llm_config = load_llm_config(config_path)
        if not llm_config:
            print(f"LLM config missing or invalid: {config_path}")
            print("Falling back to recorded snapshot if available.")

    report = collect_planner_llm_evidence(
        llm_config=llm_config,
        snapshot_path=args.snapshot,
        local_model_path=args.local_model,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
