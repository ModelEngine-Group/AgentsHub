"""OOD 鲁棒性评测集生成：用外部 LLM 生成多样化病历 + LLM 独立标注实体。

动机：现有 train/eval 都源自同一合成生成器 + 同一词典，弱标签也来自该词典——
模型本质在蒸馏词典。held-out F1 只测"同分布泛化"，测不出真正的鲁棒性/过拟合。

本脚本用外部 LLM（小米 MiMo）：
1. 生成与合成语料**不同表述风格**的门诊病历（口语化、不同句式、真实噪声）；
2. 让 LLM **独立标注**其中的实体（疾病/症状/药物/检查/治疗/科室）——
   标注不依赖我们的词典，故可作为 OOD 金标。

产物：data/ood/ood_eval.jsonl，每行 {"text","entities":[{"name","type"}]}。
用 dki.nlp.eval_ner.evaluate_entities 即可对任意后端算 OOD F1。

用法（服务器，有网）：
    export OOD_LLM_KEY=... OOD_LLM_URL=https://.../v1 OOD_LLM_MODEL=mimo-v2.5
    python3 scripts/gen_ood_eval.py --n 60 --output data/ood/ood_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TYPES = ["Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"]

_GEN_PROMPT = """你是医学 NLP 数据构造专家。请虚构 1 段门诊病历片段（纯虚构，仅用于命名实体识别模型测试）。
要求：
- 必须包含至少 2 种疾病、2 个症状、1 种药物、1 项检查；
- 用与教科书不同的、口语化或不规范的表述（如患者自述口吻、简写、错别字风格）；
- 30-80 字，单段，不要分点，不要任何解释或免责声明；
- 直接输出病历文本本身。"""

_LABEL_PROMPT = """你是医学实体标注专家。从下面的病历文本中抽取所有医疗实体。
实体类型限定：Disease(疾病)、Symptom(症状)、Drug(药物)、Exam(检查)、Treatment(治疗方式)、Department(科室)。
只输出 JSON 数组，每个元素形如 {{"mention":"原文中的确切片段","name":"医学规范名","type":"Disease"}}。
- mention 必须是病历原文里**逐字出现的连续子串**（口语化也照抄，如"没劲儿"、"脑袋晕乎乎的"）；
- name 是该 mention 对应的医学规范名（如"高压"→"高血压"、"没睡好"→"失眠"）。
不要输出任何解释。

病历文本：
{text}"""


def _chat(url: str, key: str, model: str, prompt: str, temperature: float, max_tokens: int = 2000) -> Optional[str]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                d = json.load(resp)
            msg = d["choices"][0]["message"]
            # MiMo 等推理模型把思考放 reasoning_content，答案在 content；content 空时兜底
            content = (msg.get("content") or "").strip()
            return content or None
        except Exception as e:
            if attempt == 2:
                print(f"  [chat err] {str(e)[:120]}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _parse_entities(raw: str) -> List[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    # 截取第一个 [ 到最后一个 ]
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        items = json.loads(raw[i:j + 1])
    except Exception:
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("type") in _TYPES and it.get("name"):
            out.append({
                "mention": str(it.get("mention", it.get("name"))).strip(),
                "name": str(it["name"]).strip(),
                "type": it["type"],
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 OOD NER 评测集（LLM 生成 + LLM 独立标注）")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--output", default=str(ROOT / "data" / "ood" / "ood_eval.jsonl"))
    ap.add_argument("--gen-temp", type=float, default=0.95)
    args = ap.parse_args()

    key = os.environ["OOD_LLM_KEY"]
    url = os.environ.get("OOD_LLM_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    model = os.environ.get("OOD_LLM_MODEL", "mimo-v2.5")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    fh = out_path.open("w", encoding="utf-8")  # 增量写：中途中断也保留已生成的
    try:
        for i in range(args.n):
            text = _chat(url, key, model, _GEN_PROMPT, args.gen_temp)
            if not text or len(text) < 15:
                print(f"  [{i}] 生成失败/过短，跳过")
                continue
            text = text.replace("\n", " ").strip()
            ents_raw = _chat(url, key, model, _LABEL_PROMPT.format(text=text), 0.1)
            if not ents_raw:
                print(f"  [{i}] 标注失败，跳过")
                continue
            ents = _parse_entities(ents_raw)
            # 只保留 mention 逐字出现在 text 中的实体（去 LLM 幻觉/规范化偏差）
            ents = [e for e in ents if e["mention"] and e["mention"] in text]
            if not ents:
                print(f"  [{i}] 无有效实体，跳过")
                continue
            rec = {"doc_id": f"ood_{i:04d}", "text": text, "entities": ents}
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}] OK {len(ents)} 实体 (累计 {len(records)})")
    finally:
        fh.close()
    total_ents = sum(len(r["entities"]) for r in records)
    print(f"[OK] {len(records)} 条 OOD 病历 / {total_ents} 实体 → {out_path}")


if __name__ == "__main__":
    main()
