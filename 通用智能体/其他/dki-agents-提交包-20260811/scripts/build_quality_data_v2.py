"""审计并分层切分广域质量增强训练数据。

按临床主题的确定性哈希分层，防止某一专科只出现在 dev/holdout；对所有历史 OOD 和
盲测文本仅比较 SHA-256，拒绝精确泄漏。标注冲突、单字检查/治疗、类型缺失和类别覆盖
不足会在训练前失败，而不是静默进入模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
ENTITY_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}

TOPIC_RULES = [
    ("respiratory_ent", r"肺|咳|痰|哮喘|鼻|咽|喉|耳|呼吸"),
    ("digestive", r"胃|肠|腹|肝|胆|胰|便|消化|反酸"),
    ("neuro_sleep", r"头痛|眩晕|神经|癫痫|失眠|睡眠|焦虑|抑郁"),
    ("orthopedic_rheum", r"骨|关节|踝|膝|腰|颈|痛风|扭伤|骨折"),
    ("skin_allergy", r"皮肤|红疹|丘疹|瘙痒|过敏|湿疹|痤疮"),
    ("urinary_renal", r"尿|肾|前列腺|膀胱|睾丸"),
    ("gyne_obstetric", r"月经|子宫|卵巢|孕|胎|乳腺|阴道"),
    ("pediatric", r"患儿|儿童|婴儿|新生儿|家长代诉"),
    ("eye_oral", r"眼|视力|牙|口腔|牙龈|舌"),
    ("cardio_heme_endo", r"心|血压|血糖|甲状腺|贫血|血液|肿瘤"),
]


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _iter_forbidden(paths: Iterable[Path]):
    for path in paths:
        for row in _read_jsonl(path):
            if row.get("text"):
                yield _hash(row["text"])


def _forbidden_hashes(out_dir: Path, input_path: Path) -> Set[str]:
    paths = [ROOT / "data" / "ner" / "train.jsonl", ROOT / "data" / "ner" / "eval.jsonl"]
    paths.extend((ROOT / "data" / "ood").glob("*.jsonl"))
    paths.extend((ROOT / "data" / "ood_blind").glob("*.jsonl"))
    paths.extend(
        path for path in (ROOT / "data" / "ood_train").glob("*.jsonl")
        if path.name not in {"train.jsonl", "dev.jsonl", "holdout.jsonl"}
    )
    paths.extend(
        path for path in out_dir.glob("*.jsonl")
        if path != input_path and path.name not in {"train.jsonl", "dev.jsonl", "holdout.jsonl"}
    )
    return set(_iter_forbidden(paths))


def _topic(text: str) -> str:
    for name, pattern in TOPIC_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return "mixed_other"


def _validate_rows(rows: List[dict], forbidden: Set[str]) -> dict:
    ids, hashes = set(), set()
    counts = Counter()
    topic_counts = Counter()
    documents_with_treatment = 0
    for index, row in enumerate(rows, 1):
        doc_id = str(row.get("doc_id", "")).strip()
        text = str(row.get("text", "")).strip()
        entities = row.get("entities")
        if not doc_id or not text or not isinstance(entities, list) or not entities:
            raise ValueError(f"第 {index} 条缺少 doc_id/text/entities")
        digest = _hash(text)
        if doc_id in ids or digest in hashes:
            raise ValueError(f"第 {index} 条重复: {doc_id}")
        if digest in forbidden:
            raise ValueError(f"第 {index} 条与历史 OOD/盲测文本重合: {doc_id}")
        ids.add(doc_id)
        hashes.add(digest)
        mention_types: Dict[str, Set[str]] = defaultdict(set)
        row_types = set()
        for entity in entities:
            mention = str(entity.get("mention", "")).strip()
            entity_type = entity.get("type")
            if not mention or mention not in text or entity_type not in ENTITY_TYPES:
                raise ValueError(f"第 {index} 条非法实体: {entity!r}")
            if len(mention) == 1 and entity_type in {"Exam", "Treatment"}:
                raise ValueError(f"第 {index} 条含单字碎片: {entity!r}")
            mention_types[mention].add(entity_type)
            row_types.add(entity_type)
            counts[entity_type] += 1
        conflicts = {mention: types for mention, types in mention_types.items() if len(types) > 1}
        if conflicts:
            raise ValueError(f"第 {index} 条同 mention 类型冲突: {conflicts}")
        if len(row_types) < 3:
            raise ValueError(f"第 {index} 条实体类型少于 3 类: {doc_id}")
        documents_with_treatment += "Treatment" in row_types
        topic_counts[_topic(text)] += 1
    for required in ("Disease", "Symptom", "Drug", "Exam", "Treatment"):
        if counts[required] == 0:
            raise ValueError(f"整个数据集缺少 {required} 标注")
    return {
        "documents": len(rows),
        "entities": sum(counts.values()),
        "entity_types": dict(sorted(counts.items())),
        "topics": dict(sorted(topic_counts.items())),
        "documents_with_treatment": documents_with_treatment,
        "forbidden_hash_overlap": 0,
    }


def _stratified_split(rows: List[dict], seed: int,
                      train_ratio: float, dev_ratio: float) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[_topic(row["text"])].append(row)
    splits = {"train": [], "dev": [], "holdout": []}
    for topic, group in sorted(groups.items()):
        group.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{topic}:{row['doc_id']}".encode("utf-8")
        ).hexdigest())
        n = len(group)
        n_dev = max(1, round(n * dev_ratio)) if n >= 5 else 0
        n_holdout = max(1, n - int(n * train_ratio) - n_dev) if n >= 5 else 0
        if n_dev + n_holdout >= n:
            n_dev, n_holdout = 1, 0
        n_train = n - n_dev - n_holdout
        splits["train"].extend(group[:n_train])
        splits["dev"].extend(group[n_train:n_train + n_dev])
        splits["holdout"].extend(group[n_train + n_dev:])
    for name in splits:
        splits[name].sort(key=lambda row: row["doc_id"])
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="审计并切分广域 OOD 质量数据")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "ood_train_v2" / "quality_raw.jsonl"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "ood_train_v2"))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--allow-development-source", action="store_true",
                        help="允许明确重分类的旧开发/误差分析集进入训练；仍禁止 final2 fresh blind")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    rows = _read_jsonl(input_path)
    if len(rows) < 30:
        raise RuntimeError(f"广域质量样本仅 {len(rows)} 条，至少需要 30 条")
    forbidden = _forbidden_hashes(out_dir, input_path)
    if args.allow_development_source:
        for development_path in (
            ROOT / "data" / "ood_blind" / "blind_eval.jsonl",
            ROOT / "data" / "ood_blind" / "final_blind_eval.jsonl",
        ):
            forbidden.difference_update(_iter_forbidden([development_path]))
    audit = _validate_rows(rows, forbidden)
    splits = _stratified_split(rows, args.seed, args.train_ratio, args.dev_ratio)
    if not splits["dev"] or not splits["holdout"]:
        raise RuntimeError("dev/holdout 为空，无法进行隔离选择与评测")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "seed": args.seed,
        "source": str(input_path),
        "counts": {name: len(data) for name, data in splits.items()},
        "split_topics": {
            name: dict(sorted(Counter(_topic(row["text"]) for row in data).items()))
            for name, data in splits.items()
        },
        "audit": audit,
        "blind_isolation": (
            "fresh final2 and future blind labels are excluded; only SHA-256 text hashes are read "
            "to reject exact overlap. With --allow-development-source, explicitly reclassified "
            "old development/error-analysis sets may enter training."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
