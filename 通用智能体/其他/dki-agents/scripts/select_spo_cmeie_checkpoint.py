"""Publish the one canonical calibration checkpoint selection for a RID."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.nlp.cmeie_run_protocol import (  # noqa: E402
    canonical_run_paths,
    load_run_binding,
)
from dki.nlp.cmeie_selection_protocol import (  # noqa: E402
    canonical_selection_path,
    compute_canonical_selection,
    load_and_validate_selection_receipt,
    publish_canonical_selection_once,
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        run_binding = load_run_binding(args.run_root)
        paths = canonical_run_paths(run_binding)
        candidate = _absolute(args.candidate)
        matching_seeds = [
            seed
            for seed in (42, 43, 44)
            if candidate == paths.seed_model(seed)
        ]
        if len(matching_seeds) != 1:
            raise ValueError(
                "candidate must equal one canonical seed candidate path"
            )
        seed = matching_seeds[0]
        output = canonical_selection_path(run_binding, seed)
        if _absolute(args.output) != output:
            raise ValueError(
                "selection output must equal the canonical seed selection path"
            )
        if os.path.lexists(output):
            raise FileExistsError(
                f"checkpoint-selection output exists: {output}"
            )
        payload = compute_canonical_selection(
            candidate,
            run_binding=run_binding,
        )
        if payload["seed"] != seed:
            raise ValueError(
                "canonical selection seed changed after attestation validation"
            )
        publish_canonical_selection_once(
            output,
            payload,
            run_binding=run_binding,
        )
        validated = load_and_validate_selection_receipt(
            output,
            run_binding=run_binding,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            validated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
