"""Fail-closed CLI for the NL2SQL contamination audit.

Loads candidate and historical JSONL corpora, runs the disjointness audit and
the runtime import-closure scan, writes the private report and the text-free
public attestation, and exits nonzero unless both are clean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_contamination import (  # noqa: E402
    audit_contamination,
    scan_runtime_closure,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            for key in ("id", "question", "gold_sql"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number}: missing field {key!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: contains no records (fail-closed)")
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument(
        "--historical-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Repeatable: every historical/consumed corpus must be supplied.",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--private-report-out", type=Path, required=True)
    parser.add_argument("--public-attestation-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        candidates = _load_jsonl(args.candidate_jsonl)
        historical: list[dict] = []
        for path in args.historical_jsonl:
            historical.extend(_load_jsonl(path))
        report = audit_contamination(candidates, historical)
        closure = scan_runtime_closure(args.repo_root)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[contamination-audit] FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2

    private_payload = {
        "audit": report.private_report(),
        "runtime_closure": {
            "clean": closure.clean,
            "n_modules": closure.n_modules,
            "findings": [
                {
                    "module_path": finding.module_path,
                    "line_number": finding.line_number,
                    "fragment": finding.fragment,
                }
                for finding in closure.findings
            ],
        },
    }
    public_payload = {
        "audit": report.public_attestation(),
        "runtime_closure": closure.public_summary(),
    }
    _write_json(args.private_report_out, private_payload)
    _write_json(args.public_attestation_out, public_payload)

    clean = report.clean and closure.clean
    status = "CLEAN" if clean else "CONTAMINATED"
    print(
        f"[contamination-audit] {status}: "
        f"audit_clean={report.clean} closure_clean={closure.clean} "
        f"hits={dict(sorted(report.hit_counts.items()))}"
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
