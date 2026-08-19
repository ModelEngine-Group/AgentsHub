"""Build executable, execution-distinguishable NL2SQL hard negatives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_hard_negatives import write_hard_negatives  # noqa: E402
from dki.analysis.nl2sql_semantic_corpus import load_corpus_inputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one labelled semantic mutation per candidate and retain it only "
            "after a base-database or controlled-temporary-SQLite witness."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="live four-table SQLite database")
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        required=True,
        help="semantic-corpus directory or JSONL split; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        rows = load_corpus_inputs(args.corpus)
        result = write_hard_negatives(
            args.db,
            rows,
            args.output,
            manifest_path=args.manifest,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(result["output_path"].resolve()),
                "manifest": str(result["manifest_path"].resolve()),
                "rows": result["row_count"],
                "mutation_labels": result["manifest"]["mutation_labels"],
                "base_database_witnesses": result["manifest"][
                    "base_database_witnesses"
                ],
                "controlled_temporary_sqlite_witnesses": result["manifest"][
                    "controlled_temporary_sqlite_witnesses"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
