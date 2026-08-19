"""分层 NL2SQL 执行准确率评测（rule + auto/LLM SOTA 管线），落盘证据。

- rule：确定性主路径（离线、零 LLM），报基础集与合并集分层 EX。
- auto：规则命中即用规则；硬结构让位给 LLM（schema-linking + self-consistency +
  执行反馈自纠），报合并集分层 EX 与 LLM 兜底命中率。
- holdout：从 hard 集按稳定哈希冻结子集，禁止再用于调参；单独落盘。

用法：
  python scripts/eval_nl2sql_layered.py --engine rule
  python scripts/eval_nl2sql_layered.py --engine rule --holdout
  python scripts/eval_nl2sql_layered.py --engine auto --samples 4 --rounds 2 --limit-hard 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql import NL2SQL  # noqa: E402
from dki.analysis.nl2sql_eval import (  # noqa: E402
    EVAL_SET, HARD_EVAL_SET, _load_items, _score_item,
)
from dki.paths import OUTPUT_DIR  # noqa: E402

HOLDOUT_SEED = "dki-nl2sql-holdout-v1"
HOLDOUT_FRACTION = 0.25  # 25% of hard set frozen


def _publish_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"evaluation output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _item_key(item: dict) -> str:
    return str(item.get("id") or item.get("question") or json.dumps(item, ensure_ascii=False, sort_keys=True))


def _split_holdout(hard_items: list) -> tuple[list, list]:
    """Deterministic freeze: hash(id+seed) rank; top fraction = holdout."""
    ranked = sorted(
        hard_items,
        key=lambda it: hashlib.sha256(f"{HOLDOUT_SEED}\0{_item_key(it)}".encode("utf-8")).hexdigest(),
    )
    n_hold = max(1, int(round(len(ranked) * HOLDOUT_FRACTION))) if ranked else 0
    holdout = ranked[:n_hold]
    develop = ranked[n_hold:]
    return develop, holdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["rule", "auto"], default="rule")
    ap.add_argument("--db", type=Path, default=OUTPUT_DIR / "insight.db")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit-hard", type=int, default=0, help="仅评前 N 道难题（0=全部）")
    ap.add_argument(
        "--holdout", action="store_true",
        help="evaluate frozen hard holdout only (do not use for tuning)",
    )
    ap.add_argument(
        "--dev-only", action="store_true",
        help="evaluate development hard split only (excludes holdout)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="use the non-truncating evaluator and require exact result columns",
    )
    ap.add_argument(
        "--source-tree-sha256",
        default="",
        help="optional immutable included-tree identity for a formal gate receipt",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.source_tree_sha256 and not re.fullmatch(
        r"[0-9a-f]{64}", args.source_tree_sha256
    ):
        raise SystemExit("source-tree-sha256 must be a lowercase SHA-256")
    if args.output is None:
        if args.holdout:
            args.output = OUTPUT_DIR / "eval" / f"nl2sql_layered_holdout_{args.engine}_eval.json"
        else:
            args.output = OUTPUT_DIR / "eval" / "nl2sql_layered_eval.json"

    base = _load_items(EVAL_SET)
    for it in base:
        it.setdefault("difficulty", "easy")
        it.setdefault("category", "basic")
    hard = _load_items(HARD_EVAL_SET)
    if args.limit_hard > 0:
        hard = hard[: args.limit_hard]
    hard_dev, hard_hold = _split_holdout(hard)

    if args.holdout:
        items = hard_hold
        dataset_role = "holdout_locked"
        selection_used = False
        holdout_status = (
            f"frozen {len(hard_hold)}/{len(hard)} hard items via sha256({HOLDOUT_SEED}); "
            "do not retune against this split"
        )
    elif args.dev_only:
        items = base + hard_dev
        dataset_role = "development_selection_used"
        selection_used = True
        holdout_status = (
            f"development excludes frozen holdout ({len(hard_hold)} hard items); "
            "safe for implementation regression"
        )
    else:
        items = base + hard
        dataset_role = "development_selection_used"
        selection_used = True
        holdout_status = (
            "legacy combined set (includes future holdout items); "
            "prefer --dev-only / --holdout for new claims"
        )

    nl = NL2SQL(args.db, llm_samples=args.samples, llm_rounds=args.rounds)
    details = [
        _score_item(
            nl,
            it,
            args.engine,
            args.db,
            strict=args.strict,
        )
        for it in items
    ]
    n_correct = sum(1 for d in details if d.get("correct"))
    n_answered = sum(
        1 for d in details if d.get("engine") not in (None, "none")
    )

    layers = {}
    for d in details:
        lay = d.get("difficulty") or "unspecified"
        layers.setdefault(lay, {"n": 0, "correct": 0})
        layers[lay]["n"] += 1
        layers[lay]["correct"] += 1 if d.get("correct") else 0
    for v in layers.values():
        v["accuracy"] = round(v["correct"] / max(1, v["n"]), 4)

    llm_used = sum(1 for d in details if d.get("engine") == "llm")
    llm_correct = sum(1 for d in details if d.get("engine") == "llm" and d.get("correct"))
    payload = {
        "schema_version": 3,
        "dataset_role": dataset_role,
        "selection_used": selection_used,
        "holdout_status": holdout_status,
        "holdout_seed": HOLDOUT_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "n_hard_total": len(hard),
        "n_hard_holdout": len(hard_hold),
        "n_hard_dev": len(hard_dev),
        "engine": args.engine,
        "llm_samples": args.samples,
        "llm_rounds": args.rounds,
        "strict_result_contract": args.strict,
        "source_tree_sha256": args.source_tree_sha256 or None,
        "database_sha256": hashlib.sha256(args.db.read_bytes()).hexdigest(),
        "engine_source_sha256": hashlib.sha256(
            (ROOT / "dki" / "analysis" / "nl2sql.py").read_bytes()
        ).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_files": {
            "base": {
                "path": str(EVAL_SET.resolve()),
                "sha256": hashlib.sha256(EVAL_SET.read_bytes()).hexdigest(),
            },
            "hard": {
                "path": str(HARD_EVAL_SET.resolve()),
                "sha256": hashlib.sha256(HARD_EVAL_SET.read_bytes()).hexdigest(),
            },
        },
        "n": len(items),
        "n_correct": n_correct,
        "execution_accuracy": round(n_correct / max(1, len(items)), 4),
        "n_answered": n_answered,
        "n_abstained": len(items) - n_answered,
        "coverage": round(n_answered / max(1, len(items)), 4),
        "execution_accuracy_answered": round(
            n_correct / max(1, n_answered), 4
        ),
        "by_difficulty": layers,
        "llm_fallback": {"used": llm_used, "correct": llm_correct,
                         "accuracy": round(llm_correct / max(1, llm_used), 4) if llm_used else None},
        "details": details,
    }
    _publish_once(args.output, payload)
    print(json.dumps({k: payload[k] for k in
                      ("engine", "dataset_role", "n", "n_correct", "execution_accuracy",
                       "n_answered", "n_abstained", "coverage",
                       "execution_accuracy_answered", "by_difficulty",
                       "llm_fallback", "holdout_status")},
                     ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
