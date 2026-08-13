"""可选 quality-max：用独立 OpenAI 兼容 LLM 做高质量实体抽取。

该模式不接入默认 dict/BERT/RBT6/OM 路径，只用于低吞吐、高质量场景。输入按批发送，
输出 mention 必须逐字存在于原文；所有结果再做同类型跨度去重。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_ood import _gold_spans, _pred_spans, _prf_span

ENTITY_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}


def _prompt(rows: List[dict]) -> str:
    payload = [{"doc_id": row["doc_id"], "text": row["text"]} for row in rows]
    return f"""你是中文医学 NER 专家。完整抽取下列病历的所有医疗实体。
类型只允许 Disease/Symptom/Drug/Exam/Treatment/Department。
mention 必须是原文逐字连续子串；name 给医学规范名。检查结果/数值属于 Exam，治疗动作属于
Treatment，不要漏英文缩写、口语、体征；不要把普通修饰词或剂量频次单独当实体。
只输出 JSON 数组：
[{{"doc_id":"...","entities":[{{"mention":"原文","name":"规范名","type":"Symptom"}}]}}]

病历：
{json.dumps(payload, ensure_ascii=False)}
"""


def _extract_json(raw: str) -> list:
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _locate_entities(text: str, entities: list) -> list:
    output, used = [], []
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("type") not in ENTITY_TYPES:
            continue
        mention = str(entity.get("mention", "")).strip()
        name = str(entity.get("name", mention)).strip()
        if not mention or not name:
            continue
        start = 0
        while True:
            index = text.find(mention, start)
            if index < 0:
                break
            span = (index, index + len(mention))
            if not any(entity["type"] == old_type and not (span[1] <= s or span[0] >= e)
                       for s, e, old_type in used):
                output.append({"mention": mention, "name": name, "type": entity["type"],
                               "span": [span[0], span[1]], "source": "quality-llm"})
                used.append((span[0], span[1], entity["type"]))
                break
            start = index + 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="评测可选 LLM quality-max NER")
    parser.add_argument("--ood", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    rows = [json.loads(line) for line in Path(args.ood).open(encoding="utf-8") if line.strip()]
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_API_URL"],
        timeout=180,
    )
    model = os.environ["LLM_MODEL_NAME"]
    predictions = {}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _prompt(batch)}],
            temperature=0.0,
            max_tokens=5000,
        )
        raw = response.choices[0].message.content or ""
        by_id = {item.get("doc_id"): item.get("entities", [])
                 for item in _extract_json(raw) if isinstance(item, dict)}
        for row in batch:
            predictions[row["doc_id"]] = _locate_entities(
                row["text"], by_id.get(row["doc_id"], []),
            )
        print(f"processed={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    gold = {row["doc_id"]: _gold_spans(row["text"], row["entities"]) for row in rows}
    pred = {row["doc_id"]: _pred_spans(row["text"], predictions[row["doc_id"]]) for row in rows}
    result = {
        "n_docs": len(rows),
        "n_gold_entities": sum(len(value) for value in gold.values()),
        "model": model,
        "mode": "quality-max-independent-llm",
        "quality_max": _prf_span(pred, gold),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
