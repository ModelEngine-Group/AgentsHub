"""生成冻结后的全新 OOD 盲测集（先生成文本，再独立标注）。

与质量训练/校准数据完全隔离：
- 生成提示覆盖十个临床系统，明确限制高血压/糖尿病占比，避免旧 OOD 的主题塌缩；
- 第一次 LLM 调用只生成文本，第二次调用独立标注，不把生成时的实体答案直接当金标；
- 与 data/ner、data/ood_train、data/ood 的现有文本做 SHA-256 精确去重；
- 默认拒绝覆盖已有输出。模型和规则冻结后才运行本脚本，生成后只评一次。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
ENTITY_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}
TOPIC_GROUPS = [
    "呼吸系统与耳鼻喉",
    "消化、肝胆与营养",
    "神经系统与睡眠",
    "骨科、风湿与疼痛",
    "皮肤、过敏与感染",
    "泌尿、肾脏与男性健康",
    "妇科、产科与乳腺",
    "儿科常见门诊",
    "眼科、口腔与五官",
    "心血管、血液与内分泌（本组不得超过全部病例的五分之一）",
]


def _chat(url: str, key: str, model: str, prompt: str, temperature: float,
          timeout: int = 180) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3500,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                message = json.load(response)["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if content:
                return content
        except Exception as exc:
            if attempt == 2:
                print(f"  [LLM error] {str(exc)[:160]}")
        time.sleep(3 * (attempt + 1))
    return ""


def _json_array(raw: str) -> List:
    """提取响应中最后一个完整 JSON 数组，兼容模型在前面输出含方括号的思考文本。"""
    candidates = []
    cursor = 0
    while True:
        start = raw.find("[", cursor)
        if start < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        parsed_end = None
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(raw[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, list):
                        candidates.append(value)
                    parsed_end = index + 1
                    break
        cursor = parsed_end if parsed_end is not None else start + 1
    return candidates[-1] if candidates else []


def _generate_prompt(topics: List[str], count: int) -> str:
    return f"""你是中文临床 NLP 盲测语料构造员。只生成病历文本，不做实体标注。
请输出 JSON 字符串数组，共 {count} 条，每条 45-130 字，主题依次覆盖：{'、'.join(topics)}。

硬性要求：
- 病历之间疾病、症状、药物和检查尽量不重复，覆盖不同年龄、性别和书写风格；
- 混合患者口述、医生简写、错别字、英文缩写、数值、同义表达，但必须保持医学可理解；
- 每条自然包含至少 4 个可抽取医疗实体，并覆盖至少 3 种类型；
- 不要集中写高血压、糖尿病、头晕、二甲双胍、血糖/血压检查；这些旧主题在本批最多出现 1 条；
- 不要免责声明、解释、编号或 markdown，只输出 JSON 字符串数组。
"""


def _label_prompt(texts: List[str]) -> str:
    numbered = [{"index": i, "text": text} for i, text in enumerate(texts)]
    return f"""你是独立的中文医学 NER 标注员。下面文本由另一轮调用生成，你没有它的实体答案。
逐条完整标注所有医疗实体，类型仅限 Disease/Symptom/Drug/Exam/Treatment/Department。
mention 必须是原文逐字连续子串，name 是规范名；不要从一个 mention 臆造两个互相冲突的类型。
只输出 JSON 数组，格式：
[{{"index":0,"entities":[{{"mention":"原文片段","name":"规范名","type":"Disease"}}]}}]

文本：
{json.dumps(numbered, ensure_ascii=False)}
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _existing_hashes(paths: List[Path]) -> Set[str]:
    hashes: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("text"):
                    hashes.add(_hash(row["text"]))
    return hashes


def _validated_records(texts: List[str], labels: List, min_entities: int = 4,
                       min_types: int = 3) -> List[dict]:
    by_index: Dict[int, list] = {}
    for item in labels:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            by_index[item["index"]] = item.get("entities", [])
    records = []
    for index, text in enumerate(texts):
        if not (35 <= len(text) <= 180):
            continue
        entities, seen = [], set()
        for ent in by_index.get(index, []):
            if not isinstance(ent, dict) or ent.get("type") not in ENTITY_TYPES:
                continue
            mention = str(ent.get("mention", "")).strip()
            name = str(ent.get("name", mention)).strip()
            key = (mention, ent.get("type"))
            if not mention or mention not in text or not name or key in seen:
                continue
            seen.add(key)
            entities.append({"mention": mention, "name": name, "type": ent["type"]})
        if len(entities) >= min_entities and len({e["type"] for e in entities}) >= min_types:
            records.append({"text": text, "entities": entities})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="生成冻结质量模式后的全新 OOD 盲测集")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--min-entities", type=int, default=4)
    parser.add_argument("--min-types", type=int, default=3)
    parser.add_argument("--batch-cases", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--label-url", default=None,
                        help="可选独立标注端点；默认沿用 OOD_LLM_URL")
    parser.add_argument("--label-key-env", default="OOD_LLM_KEY",
                        help="独立标注 API key 的环境变量名")
    parser.add_argument("--label-model", default=None,
                        help="可选独立标注模型；默认沿用 OOD_LLM_MODEL")
    parser.add_argument("--output", default=str(ROOT / "data" / "ood_blind" / "blind_eval.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="从已增量写入的记录继续")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"盲测集已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    forbidden_paths = [
        ROOT / "data" / "ner" / "train.jsonl",
        ROOT / "data" / "ner" / "eval.jsonl",
        ROOT / "data" / "ood_train" / "quality_raw.jsonl",
        ROOT / "data" / "ood" / "ood_eval.jsonl",
    ]
    blind_dir = ROOT / "data" / "ood_blind"
    if blind_dir.exists():
        forbidden_paths.extend(path for path in blind_dir.glob("*.jsonl") if path != output)
    forbidden = _existing_hashes(forbidden_paths)
    key = os.environ["OOD_LLM_KEY"]
    url = os.environ.get("OOD_LLM_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    model = os.environ.get("OOD_LLM_MODEL", "mimo-v2.5")
    label_url = args.label_url or url
    label_key = os.environ[args.label_key_env]
    label_model = args.label_model or model
    accepted, calls, seen = [], 0, set(forbidden)
    if output.exists() and args.resume:
        with output.open(encoding="utf-8") as existing:
            accepted = [json.loads(line) for line in existing if line.strip()]
        seen.update(_hash(record["text"]) for record in accepted)
        print(f"[resume] existing={len(accepted)}/{args.n}", flush=True)
    mode = "a" if accepted else "w"
    with output.open(mode, encoding="utf-8") as fh:
        while len(accepted) < args.n and calls < args.n * 3:
            calls += 1
            topics = [TOPIC_GROUPS[(calls * args.batch_cases + i) % len(TOPIC_GROUPS)]
                      for i in range(args.batch_cases)]
            raw_texts = _json_array(_chat(
                url, key, model, _generate_prompt(topics, args.batch_cases), 1.0,
                args.request_timeout,
            ))
            texts = [str(text).replace("\n", " ").strip() for text in raw_texts
                     if isinstance(text, str)]
            labels = _json_array(_chat(
                label_url, label_key, label_model, _label_prompt(texts), 0.1,
                args.request_timeout,
            )) if texts else []
            for record in _validated_records(texts, labels, args.min_entities, args.min_types):
                digest = _hash(record["text"])
                if digest in seen:
                    continue
                seen.add(digest)
                record["doc_id"] = f"blind_{len(accepted):04d}"
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                accepted.append(record)
                if len(accepted) >= args.n:
                    break
            print(f"  calls={calls}, accepted={len(accepted)}/{args.n}", flush=True)
    manifest = {
        "documents": len(accepted),
        "entities": sum(len(row["entities"]) for row in accepted),
        "generation_calls": calls,
        "forbidden_text_hashes": len(forbidden),
        "exact_overlap": 0,
        "protocol": "model/rules frozen before generation; evaluate once; do not tune on this set",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
