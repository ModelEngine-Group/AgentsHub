"""Create the static source-closure artifact required by a blind candidate seal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_source_closure import (
    DEFAULT_ENTRY_MODULES,
    build_source_closure_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--entry-module", action="append")
    args = parser.parse_args()
    try:
        payload = build_source_closure_manifest(
            args.output,
            source_root=args.source_root,
            entry_modules=args.entry_module or DEFAULT_ENTRY_MODULES,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "closure_sha256": payload["closure_sha256"],
                "n_files": len(payload["files"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
