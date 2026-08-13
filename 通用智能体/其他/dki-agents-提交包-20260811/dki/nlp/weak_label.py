"""远程监督弱标注器：用 dict 后端（F1≈99.97%）自动生成 NER 训练标签。

核心思路（见 docs/ner_finetuning.md）：
- gold entities 只有 (name, type)、span=null，无法直接做 span 级训练标签；
- 但 dict 后端的 `_resolve_spans` 已能在文本中定位干净的字符级 span，
  且对本合成语料 F1≈99.97%——用它对（清洗后）文本做远程监督弱标注，
  自动得到字符级实体 span，无需人工标注；
- 可选用 gold 的 (name, type) 集合过滤 dict 的越界/歧义匹配（降噪），
  使弱标签质量≈金标。

产物（data/ner/）：
- train.jsonl / eval.jsonl：每行 {"doc_id","text","spans":[[start,end,type],...]}
- label_list.json：{"labels":[...13...], "id2label":{...}}

训练脚本（scripts/train_ner.py）读这些 char span，用 fast tokenizer 的
offset_mapping 对齐到 token 级 BIO 标签（对中文 BERT 稳健，无需假设 char==token）。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from dki.nlp.labels import KNOWN_TYPES, LABELS, OUTSIDE, id2label
from dki.paths import GOLD_DATA_DIR, OUTPUT_DIR, PROJECT_ROOT, RAW_DATA_DIR

# 复用算子的词典加载与 span 定位（单一事实来源，避免逻辑漂移）
from dki.operators.medical_entity_extractor.process import _resolve_spans, load_vocab

Span = Tuple[int, int, str]  # (start, end, type)


def spans_to_bio(text: str, spans: Sequence[Span]) -> List[str]:
    """字符级 BIO 序列（长度 = len(text)）。重叠时后写入者覆盖，调用方应先消歧。

    仅用于测试/校验 BIO 合法性；训练走 offset_mapping 的 token 级对齐。
    """
    tags = [OUTSIDE] * len(text)
    for start, end, etype in spans:
        if etype not in KNOWN_TYPES:
            raise ValueError(f"未知实体类型 {etype!r}")
        if not (0 <= start < end <= len(text)):
            continue
        tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{etype}"
    return tags


def _load_gold_pairs(doc_id: str, gold_dir: Path) -> Optional[Set[Tuple[str, str]]]:
    """读取某文档金标实体的 (name, type) 集合；无金标返回 None（表示不过滤）。"""
    f = gold_dir / f"{doc_id}.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text(encoding="utf-8"))
    return {(str(r["name"]).strip(), str(r["type"]).strip()) for r in rows}


def label_one(
    text: str,
    vocab: Dict[str, Tuple[str, str]],
    gold_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, object]:
    """对单条文本做远程监督弱标注。

    gold_pairs 非 None 时，只保留 (canonical_name, type) 命中金标的 span（降噪）。
    返回 {"text", "spans":[[start,end,type],...]}（spans 按 start 升序）。
    """
    hits = [w for w in vocab if w in text]
    resolved = _resolve_spans(text, hits, vocab)  # 已做最长匹配消歧、无重叠
    spans: List[List[object]] = []
    for e in resolved:
        etype = e["type"]
        if etype not in KNOWN_TYPES:
            continue
        if gold_pairs is not None and (e["name"], etype) not in gold_pairs:
            continue
        s, en = e["span"]
        spans.append([int(s), int(en), etype])
    spans.sort(key=lambda x: x[0])
    return {"text": text, "spans": spans}


def _iter_source_texts(clean_dir: Path) -> List[Tuple[str, str]]:
    """优先用清洗后文本（outputs/clean），缺失则回退原始病历（data/raw）。

    返回 [(doc_id, text)]，doc_id 与 gold 文件名 stem 对齐（如 emr_0001）。
    """
    src = clean_dir if clean_dir.exists() and any(clean_dir.glob("*.txt")) else RAW_DATA_DIR
    out = []
    for f in sorted(src.glob("*.txt")):
        out.append((f.stem, f.read_text(encoding="utf-8")))
    return out


def build_ner_dataset(
    clean_dir: Optional[Path] = None,
    gold_dir: Optional[Path] = None,
    dict_path: Optional[str] = None,
    gold_filter: bool = True,
    split: float = 0.8,
    seed: int = 42,
    out_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """构建 NER 弱标注数据集（train/eval jsonl + label_list.json）。

    - gold_filter=True：用金标 (name,type) 过滤 dict 越界匹配（弱标≈金标质量）；
    - split=0.8, seed=42：确定性切分，eval 集训练不可见，用于真实泛化评估。
    """
    clean_dir = Path(clean_dir) if clean_dir else (OUTPUT_DIR / "clean")
    gold_dir = Path(gold_dir) if gold_dir else (GOLD_DATA_DIR / "entities")
    out_dir = Path(out_dir) if out_dir else (PROJECT_ROOT / "data" / "ner")
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(dict_path)
    docs = _iter_source_texts(clean_dir)
    if not docs:
        raise FileNotFoundError(
            f"没有可用文本：既无 {clean_dir}/*.txt 也无 {RAW_DATA_DIR}/*.txt；"
            f"可先运行 python -m dki gen-data / task1。"
        )

    records = []
    total_spans = 0
    for doc_id, text in docs:
        gold_pairs = _load_gold_pairs(doc_id, gold_dir) if gold_filter else None
        rec = label_one(text, vocab, gold_pairs)
        rec["doc_id"] = doc_id
        total_spans += len(rec["spans"])
        records.append(rec)

    rng = random.Random(seed)
    rng.shuffle(records)
    n_train = int(len(records) * split)
    train, eval_ = records[:n_train], records[n_train:]

    def _dump(path: Path, rows):
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    _dump(out_dir / "train.jsonl", train)
    _dump(out_dir / "eval.jsonl", eval_)
    (out_dir / "label_list.json").write_text(
        json.dumps({"labels": LABELS, "id2label": {str(k): v for k, v in id2label.items()}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "n_docs": len(records),
        "n_train": len(train),
        "n_eval": len(eval_),
        "total_spans": total_spans,
        "avg_spans_per_doc": round(total_spans / max(len(records), 1), 2),
        "gold_filter": gold_filter,
        "seed": seed,
        "source": "clean" if (clean_dir.exists() and any(clean_dir.glob("*.txt"))) else "raw",
        "out_dir": str(out_dir),
        "num_labels": len(LABELS),
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats
