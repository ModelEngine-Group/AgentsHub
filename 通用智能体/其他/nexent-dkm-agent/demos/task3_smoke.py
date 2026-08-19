"""Run a compact reviewer smoke check for task 3."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.task3_smoke import run_task3_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check task-3 reviewer path.")
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_task3_smoke(
        graph_file=args.graph_file,
        output_dir=args.output_dir,
        question=args.question,
        iterations=args.iterations,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
