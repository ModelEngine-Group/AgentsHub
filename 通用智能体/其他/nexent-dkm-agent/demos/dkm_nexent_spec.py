"""Print the unified DKM Nexent agent spec with all three task tools."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.dkm_nexent_suite import build_dkm_nexent_suite_spec
from src.common.integration import build_integration_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print unified DKM Nexent agent spec.")
    parser.add_argument("--model-name", default="main_model")
    parser.add_argument("--datamate-url", default="http://localhost:18000")
    parser.add_argument("--nexent-url", default="http://localhost:3000")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Also probe Nexent/DataMate availability and print integration report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = build_dkm_nexent_suite_spec(
        model_name=args.model_name,
        datamate_base_url=args.datamate_url,
        output_root=args.output_root,
    )
    payload: dict = {"agent_spec": spec}
    if args.probe:
        payload["integration_report"] = build_integration_report(
            datamate_url=args.datamate_url,
            nexent_url=args.nexent_url,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
