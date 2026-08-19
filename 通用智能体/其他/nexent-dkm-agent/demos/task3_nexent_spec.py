"""Print the task-3 Nexent-compatible agent spec."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.analysis_agent.nexent_adapter import build_nexent_agent_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print task-3 Nexent agent spec.")
    parser.add_argument("--model-name", default="main_model")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            build_nexent_agent_spec(model_name=args.model_name, output_dir=args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
