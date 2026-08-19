"""批量生成严格隔离的口语 OOD 训练数据（LLM 生成 + 逐字 mention 标注）。

与 data/ood/ood_eval.jsonl（最终测试集）严格隔离：本脚本生成的新样本只能进入
质量增强模型训练/验证，绝不能覆盖或混入现有 OOD 测试集。

相对逐条生成的 gen_ood_eval.py，本脚本每次请求 5 个病例，减少 MiMo 推理模型的
网络/思考开销；特意要求高频 OOD 漏项：口语症状、简称、同义词、非规范检查表达。

使用：
  export OOD_LLM_KEY=... OOD_LLM_URL=https://.../v1 OOD_LLM_MODEL=mimo-v2.5
  python scripts/gen_ood_quality.py --n 80 --output data/ood_train/quality_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}

_PROMPT = """你是医学 NER 鲁棒性数据构造专家。生成 {n} 条彼此不同的虚构中文门诊病历，纯用于模型训练。

每条必须是 JSON 对象，格式：
{{"text":"病历原文", "entities":[{{"mention":"原文逐字连续片段","name":"医学规范名","type":"Disease"}}]}}
整体只输出一个 JSON 数组，不要任何说明。

规则：
- 每条 35-100 字，包含至少 4 个实体，类型仅 Disease/Symptom/Drug/Exam/Treatment/Department；
- mention 必须逐字出现在 text 中，name 是该 mention 的医学规范名；
- 特别覆盖口语化、同义词、非规范表达和真实门诊书写，如：没睡好/睡不踏实、没劲儿/乏力、
  脑袋晕乎乎/头晕、高压/血压升高、看东西发花/视物模糊、嘴发干/口干、手抖/震颤、
  胸口发闷/胸闷、查血糖/空腹血糖检测、量血压/血压检测；
- 疾病、症状、药物、检查组合要多样，不能反复只写高血压和糖尿病；
- 不要免责声明，不要治疗建议，只给虚构病历和标注。"""


def _chat(url: str, key: str, model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 6500,
        "temperature": 0.95,
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if text:
                return text
        except Exception as exc:
            if attempt == 2:
                print(f"  [LLM error] {str(exc)[:120]}")
        time.sleep(3 * (attempt + 1))
    return ""


def _parse(raw: str) -> List[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        rows = json.loads(raw[start:end + 1])
    except Exception:
        return []
    valid = []
    for row in rows if isinstance(rows, list) else []:
        text = str(row.get("text", "")).replace("\n", " ").strip() if isinstance(row, dict) else ""
        if not (35 <= len(text) <= 160):
            continue
        ents = []
        for ent in row.get("entities", []):
            if not isinstance(ent, dict) or ent.get("type") not in _TYPES:
                continue
            mention = str(ent.get("mention", "")).strip()
            name = str(ent.get("name", mention)).strip()
            if mention and mention in text and name:
                ents.append({"mention": mention, "name": name, "type": ent["type"]})
        if len(ents) >= 3:
            valid.append({"text": text, "entities": ents})
    return valid


def main() -> None:
    ap = argparse.ArgumentParser(description="批量生成 OOD 质量训练数据")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--batch-cases", type=int, default=5)
    ap.add_argument("--output", default=str(ROOT / "data" / "ood_train" / "quality_raw.jsonl"))
    args = ap.parse_args()

    key = os.environ["OOD_LLM_KEY"]
    url = os.environ.get("OOD_LLM_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    model = os.environ.get("OOD_LLM_MODEL", "mimo-v2.5")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    seen = set()
    calls = 0
    with output.open("w", encoding="utf-8") as fh:
        while len(records) < args.n and calls < (args.n * 2):
            calls += 1
            raw = _chat(url, key, model, _PROMPT.format(n=args.batch_cases))
            rows = _parse(raw)
            for row in rows:
                # 防止模型重复模板；同 text 不重复
                if row["text"] in seen:
                    continue
                seen.add(row["text"])
                row["doc_id"] = f"quality_{len(records):04d}"
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(row)
                if len(records) >= args.n:
                    break
            print(f"  call={calls}, accepted={len(records)}/{args.n}")
    print(f"[OK] {len(records)} 条质量训练 OOD 样本 → {output}")


if __name__ == "__main__":
    main()
