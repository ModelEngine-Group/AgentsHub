"""Retired unsafe remote launcher.

The historical launcher uploaded source/data, killed matching processes, and
started an obsolete training recipe. Current CMeEE diagnostics now intentionally
reject that recipe, and the kill-first behavior can disrupt unrelated work.

This module is deliberately fail-closed. Build and inspect the local bundle with
``prepare_remote_npu_bundle.py`` first; after a controlled host is available,
execute its verified runbook step-by-step through the pinned SSH helper.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "Refusing obsolete automatic remote launch. Build the local credential-free "
        "bundle with `python scripts/prepare_remote_npu_bundle.py --include-cblue`, "
        "inspect dist/remote_npu/REMOTE_NPU_RUNBOOK.json, then run the target-host "
        "preflight before any training or OM load.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
