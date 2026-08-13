"""Fail if README / CMeEE metrics disagree."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from normalize_cmeee_metrics import main_check  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_check())
