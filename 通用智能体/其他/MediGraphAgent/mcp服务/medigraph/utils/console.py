"""Console helpers. Force UTF-8 stdout/stderr so Chinese output does not crash on
the default Windows GBK code page."""
from __future__ import annotations

import sys


def enable_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
