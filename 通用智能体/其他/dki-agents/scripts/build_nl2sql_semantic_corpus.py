"""Build the deterministic four-table semantic-family NL2SQL corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_semantic_corpus import (  # noqa: E402
    DEFAULT_DEV_FRACTION,
    DEFAULT_PARTITION_SEED,
    write_semantic_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate execution-verified train/OOD-dev NL2SQL data while keeping "
            "semantic, join, skeleton, and near-duplicate families intact."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="live four-table SQLite database")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--seed", default=DEFAULT_PARTITION_SEED)
    args = parser.parse_args()
    try:
        result = write_semantic_corpus(
            args.db,
            args.output_dir,
            dev_fraction=args.dev_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "train": str(result["train_path"].resolve()),
                "ood_dev": str(result["ood_dev_path"].resolve()),
                "manifest": str(result["manifest_path"].resolve()),
                "train_rows": result["train_rows"],
                "ood_dev_rows": result["ood_dev_rows"],
                "coverage": result["manifest"]["observed_coverage"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
