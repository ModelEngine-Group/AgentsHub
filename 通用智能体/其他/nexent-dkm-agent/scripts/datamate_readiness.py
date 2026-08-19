"""Return a compact DataMate readiness result for deployment scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.integration import probe_datamate  # noqa: E402


def evaluate_readiness(report: dict[str, Any]) -> dict[str, Any]:
    """Require every core business probe before declaring DataMate ready."""

    successful = int(report.get("successful_core_probes") or 0)
    total = int(report.get("core_probe_count") or 0)
    ready = (
        report.get("status") == "available"
        and total > 0
        and successful == total
    )
    health = report.get("health")
    return {
        "ready": ready,
        "status": report.get("status", "unavailable"),
        "readiness_basis": report.get("readiness_basis", "unknown"),
        "successful_core_probes": successful,
        "core_probe_count": total,
        "health_status": (
            health.get("status", "unknown")
            if isinstance(health, dict)
            else "unknown"
        ),
        "base_url": report.get("base_url"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check DataMate health and database-backed core APIs."
    )
    parser.add_argument("--url", default="http://localhost:18000")
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness = evaluate_readiness(
        probe_datamate(args.url, timeout=args.timeout)
    )
    print(json.dumps(readiness, ensure_ascii=False))
    return 0 if readiness["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
