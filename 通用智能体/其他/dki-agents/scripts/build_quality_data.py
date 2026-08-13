"""构建严格隔离的质量增强训练/验证集。

输入 data/ood_train/quality_raw.jsonl 由 gen_ood_quality.py 新生成，和
 data/ood/ood_eval.jsonl（最终 OOD 测试集）严格独立：
- quality_raw → seed 固定切 70% train / 15% dev / 15% internal-holdout；
- 最终 OOD test 的标签和正文不进入训练/切分；仅比较 SHA-256 文本哈希以拒绝泄漏；
- 原始 data/ner/train.jsonl (160 docs) 始终保留，用于防止质量微调遗忘正式病历能力。

产物：data/ood_train/{train,dev,holdout}.jsonl + manifest.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent
ENTITY_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}
FINAL_OOD_TEST = ROOT / "data" / "ood" / "ood_eval.jsonl"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _load_test_hashes(path: Path) -> Set[str]:
    """只读取最终测试集的文本哈希做防泄漏审计，不使用标签或内容训练。"""
    if not path.exists():
        return set()
    hashes = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                hashes.add(_text_hash(json.loads(line)["text"]))
    return hashes


def _validate_rows(rows: List[dict], forbidden_hashes: Set[str]) -> dict:
    seen_ids, seen_hashes = set(), set()
    entity_count = 0
    for i, row in enumerate(rows, 1):
        doc_id = str(row.get("doc_id", "")).strip()
        text = str(row.get("text", "")).strip()
        entities = row.get("entities")
        if not doc_id or not text or not isinstance(entities, list) or not entities:
            raise ValueError(f"第 {i} 条缺少 doc_id/text/entities")
        text_hash = _text_hash(text)
        if doc_id in seen_ids or text_hash in seen_hashes:
            raise ValueError(f"第 {i} 条存在重复 doc_id 或 text: {doc_id}")
        if text_hash in forbidden_hashes:
            raise ValueError(f"第 {i} 条与最终 OOD 测试文本重合: {doc_id}")
        seen_ids.add(doc_id)
        seen_hashes.add(text_hash)
        for ent in entities:
            mention = str(ent.get("mention", "")).strip()
            if not mention or mention not in text or ent.get("type") not in ENTITY_TYPES:
                raise ValueError(f"第 {i} 条存在非法实体标注: {ent!r}")
            entity_count += 1
    return {"documents": len(rows), "entities": entity_count, "test_hash_overlap": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description="切分独立 OOD 质量训练数据")
    ap.add_argument("--input", default=str(ROOT / "data" / "ood_train" / "quality_raw.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "ood_train"))
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--dev-ratio", type=float, default=0.15)
    ap.add_argument("--test-file", default=str(FINAL_OOD_TEST),
                    help="仅用于文本哈希防泄漏审计；标签不进入训练或切分")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    if len(rows) < 20:
        raise RuntimeError(f"质量样本仅 {len(rows)} 条，至少需要 20 条才能切分")
    audit = _validate_rows(rows, _load_test_hashes(Path(args.test_file)))
    random.Random(args.seed).shuffle(rows)
    n_train = int(len(rows) * args.train_ratio)
    n_dev = int(len(rows) * args.dev_ratio)
    splits = {
        "train": rows[:n_train],
        "dev": rows[n_train:n_train + n_dev],
        "holdout": rows[n_train + n_dev:],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "seed": args.seed,
        "source": str(Path(args.input)),
        "counts": {name: len(data) for name, data in splits.items()},
        "audit": audit,
        "test_isolation": (
            "data/ood/ood_eval.jsonl labels never enter training or model selection; "
            "only SHA-256 text hashes are compared to reject exact leakage"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
