"""OM 精度验证：静态 OM 解码带类型实体 → 实体级 F1 vs 金标。

补齐"NPU 成绩只验吞吐、不验精度"的软肋：对同一 OM（FP16 或 INT8），跑真实
病历，用 VectorizedBioDecoder.decode_typed 还原 (类型, 字符跨度) → mention →
经词典归一到规范名 → 实体级 P/R/F1 vs data/gold/entities。

FP16 与 INT8 各跑一轮，即可给出"量化后 F1 掉点"的真实数字。

用法（服务器）：
    python3 scripts/eval_om_ner.py \
        --om outputs/bench/npu_int8/ner_ft_bs64_seq256_int8.om \
        --model-name models/medical_ner_bert \
        --output outputs/bench/npu_int8/eval_om_int8.json \
        --batch 64 --seq-len 256 --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="OM NER 精度验证（实体级 F1 vs 金标）")
    ap.add_argument("--om", type=Path, required=True)
    ap.add_argument("--model-name", default="bert-base-chinese")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=200, help="评测文档数上限")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--label", default=None, help="结果标签，如 fp16 / int8")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from acl_om_pipeline_bench import StaticOmRunner, VectorizedBioDecoder, _tokenize
    from dki.nlp.eval_ner import evaluate_entities
    from dki.operators.medical_entity_extractor.process import load_vocab
    from dki.paths import GOLD_DATA_DIR, OUTPUT_DIR, RAW_DATA_DIR

    # 用清洗文本（与训练/OM 口径一致），缺失回退原始
    clean_dir = OUTPUT_DIR / "clean"
    src = clean_dir if clean_dir.exists() and any(clean_dir.glob("*.txt")) else RAW_DATA_DIR
    files = sorted(src.glob("*.txt"))[: args.limit]
    doc_ids = [f.stem for f in files]
    texts = [f.read_text(encoding="utf-8") for f in files]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    decoder = VectorizedBioDecoder(AutoConfig.from_pretrained(args.model_name).id2label)
    vocab = load_vocab()
    runner = StaticOmRunner(args.om, args.device)

    pred_by_doc: Dict[str, List[dict]] = {}
    try:
        shape = runner.output_shape
        n_elems = int(np.prod(shape))
        for i in range(0, len(texts), args.batch):
            chunk_ids = doc_ids[i:i + args.batch]
            chunk_txt = texts[i:i + args.batch]
            real = len(chunk_txt)
            if real < args.batch:  # 尾批补齐到定长
                chunk_txt = chunk_txt + [chunk_txt[-1]] * (args.batch - real)
            arrays, offsets = _tokenize(tokenizer, chunk_txt, args.seq_len)
            logits = runner.execute(arrays)[0]
            pred = logits[:n_elems].reshape(shape).argmax(axis=-1)
            typed = decoder.decode_typed(pred, offsets)
            for j, did in enumerate(chunk_ids):  # 只回填真实样本
                ents = []
                for e in typed[j]:
                    surface = chunk_txt[j][e["char_start"]:e["char_end"]]
                    canonical = vocab.get(surface, (surface, e["type"]))[0]
                    ents.append({"name": canonical, "type": e["type"], "mention": surface})
                pred_by_doc[did] = ents
    finally:
        runner.close()

    res = evaluate_entities(pred_by_doc, gold_dir=GOLD_DATA_DIR / "entities", doc_ids=doc_ids)
    out = {
        "om": str(args.om),
        "label": args.label or args.om.stem,
        "n_docs": len(doc_ids),
        "batch_size": args.batch,
        "seq_len": args.seq_len,
        "entity_f1": res,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"label": out["label"], "precision": res["precision"],
                      "recall": res["recall"], "f1": res["f1"],
                      "tp": res["tp"], "fp": res["fp"], "fn": res["fn"]},
                     ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
