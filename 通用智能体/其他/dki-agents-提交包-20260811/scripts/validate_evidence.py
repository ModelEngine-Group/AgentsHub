"""Validate one or more typed evidence claims and referenced artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence import (
    EvidenceValidationError,
    validate_claim_files,
    verify_protected_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("claims", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    failures = []
    for path in args.claims:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("protected"), list):
                verify_protected_manifest(payload, args.root)
                print(f"[OK] {path}: protected manifest ({len(payload['protected'])} artifact(s))")
                continue
            claims = payload if isinstance(payload, list) else [payload]
            for claim in claims:
                validate_claim_files(claim, args.root)
            print(f"[OK] {path}: {len(claims)} claim(s)")
        except (OSError, json.JSONDecodeError, EvidenceValidationError) as exc:
            failures.append((path, exc))
            print(f"[FAIL] {path}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
