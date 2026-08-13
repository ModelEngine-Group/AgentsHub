"""生成广域、类别均衡的医学 NER 困难训练语料。

与盲测协议隔离：本脚本只写 data/ood_train_v2，且会对 ID、既有质量数据、
所有 OOD/盲测文本做 SHA-256 精确去重。病例文本生成、实体初标和标签质控使用
三次独立 LLM 调用；自动审计会拒绝缺失必备锚点、同 mention 类型冲突及类型过少的样本。

用法：
  python scripts/gen_ood_quality_v2.py --n 240
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
ENTITY_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}

TOPIC_GROUPS = [
    "呼吸系统与耳鼻喉",
    "消化、肝胆、肛肠与营养",
    "神经系统、精神心理与睡眠",
    "骨科、运动损伤、风湿与疼痛",
    "皮肤、过敏、感染与免疫",
    "泌尿、肾脏与男性健康",
    "妇科、产科与乳腺",
    "儿科与新生儿",
    "眼科、口腔与五官",
    "心血管、血液、内分泌与肿瘤",
]

STYLE_LENSES = [
    "患者口述和错别字，包含至少一个口语症状",
    "医生简写，包含英文检查缩写和带单位数值",
    "体格检查描述，包含体征以及检查结果",
    "用药记录，包含商品剂型、剂量或给药缩写",
    "处置记录，包含非药物治疗或生活方式干预",
    "多疾病鉴别，包含否定病史，但不要标注被否定实体",
]

ANCHOR_PATTERNS = {
    "Disease": (r"诊断[:：]", r"诊断为", r"考虑为", r"既往.*病史", r"感染", r"炎", r"症", r"病", r"瘤", r"骨折"),
    "Symptom": (r"主诉", r"自诉", r"查体", r"疼", r"痛", r"肿", r"热", r"痒", r"乏", r"困难"),
    "Drug": (r"给予", r"口服", r"处方", r"用药", r"片", r"胶囊", r"注射", r"滴眼液", r"软膏"),
    "Exam": (r"检查", r"化验", r"超声", r"B超", r"CT", r"MRI", r"X线", r"评分", r"\d+(?:\.\d+)?(?:mg|mmHg|mmol|μmol|U/L|×10)"),
    "Treatment": (r"治疗", r"处理", r"固定", r"训练", r"教育", r"饮食", r"手术", r"理疗", r"随访"),
    "Department": (r"科", r"门诊"),
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _forbidden_hashes(output: Path) -> Set[str]:
    paths = [
        ROOT / "data" / "ner" / "train.jsonl",
        ROOT / "data" / "ner" / "eval.jsonl",
        ROOT / "data" / "ood_train" / "quality_raw.jsonl",
    ]
    paths.extend((ROOT / "data" / "ood").glob("*.jsonl"))
    paths.extend((ROOT / "data" / "ood_blind").glob("*.jsonl"))
    paths.extend(
        path for path in (ROOT / "data" / "ood_train_v2").glob("*.jsonl")
        if path != output
    )
    return {_hash(row["text"]) for row in _iter_jsonl(paths) if row.get("text")}


def _chat(url: str, key: str, model: str, prompt: str, temperature: float,
          timeout: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 7500,
        "temperature": temperature,
    }).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                message = json.load(response)["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if content:
                return content
        except Exception as exc:
            if attempt == 2:
                print(f"  [LLM error] {str(exc)[:160]}", flush=True)
        time.sleep(3 * (attempt + 1))
    return ""


def _json_array(raw: str) -> List:
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


def _generation_prompt(topics: List[str], styles: List[str]) -> str:
    assignments = [
        f"{i + 1}. {topic}；{style}"
        for i, (topic, style) in enumerate(zip(topics, styles))
    ]
    return f"""你是中文临床 NLP 训练语料构造员。只生成病历文本，不做实体标注。
输出恰好 {len(assignments)} 条 70-180 字的 JSON 字符串数组，对应：
{chr(10).join(assignments)}

硬性要求：
- 每条自然包含 Disease、Symptom、Drug、Exam 中至少三类；全批至少两条明确含 Treatment；
- 每条至少 7 个可抽取实体，兼顾疾病、症状/体征、药物、检查、处置；
- 刻意覆盖罕见检查、英文缩写、带单位数值、药物剂型、长症状边界与非药物治疗；
- 同一批病例的疾病、药物、检查尽量不重复；不得集中使用高血压、糖尿病、头晕、二甲双胍；
- 否定病史必须写清楚，避免把不存在的病误作阳性事实；
- 不要说明、编号、markdown 或实体答案，只输出 JSON 字符串数组。
"""


def _label_prompt(texts: List[str]) -> str:
    payload = [{"index": i, "text": text} for i, text in enumerate(texts)]
    return f"""你是独立的中文医学 NER 标注员，未参与文本生成。完整标注下列病历。
类型仅限 Disease/Symptom/Drug/Exam/Treatment/Department。mention 必须是原文逐字连续子串，
name 是规范名。必须检查并标注：口语症状、所有阳性体征、完整疾病名、完整药名、英文缩写、
带单位检验项、量表评分、影像/化验，以及明确实施或建议的非药物治疗。

边界规则：药物 mention 可包含剂型但不含剂量/频次；Exam mention 包含检查名和紧邻结果时可整体标注；
不要把否认/无/未见后的疾病标为阳性实体；同一个 mention 不得给冲突类型；不要只标单字动作。
逐条复核原文，宁可补全高置信实体，不得臆造原文不存在的 mention。
只输出 JSON 数组：
[{{"index":0,"entities":[{{"mention":"原文","name":"规范名","type":"Exam"}}]}}]

病历：
{json.dumps(payload, ensure_ascii=False)}
"""


def _review_prompt(texts: List[str], labels: List) -> str:
    payload = [{"index": i, "text": text} for i, text in enumerate(texts)]
    return f"""你是资深中文医学 NER 标注质控员。逐条复核初标，输出修正后的完整实体列表。
重点补齐阳性体征、长口语症状、完整疾病名、完整药名/剂型、英文缩写、带单位检验、量表、
影像检查和非药物治疗；删除否定事实、普通词、剂量频次碎片、单字动作及重复/冲突类型。
mention 必须是原文逐字连续子串，类型仅限 Disease/Symptom/Drug/Exam/Treatment/Department。
只输出与输入 index 对齐的 JSON 数组，不要解释。

文本：
{json.dumps(payload, ensure_ascii=False)}

初标：
{json.dumps(labels, ensure_ascii=False)}
"""


def _anchor_types(text: str) -> Set[str]:
    return {
        entity_type
        for entity_type, patterns in ANCHOR_PATTERNS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    }


def _clean_entities(text: str, raw_entities: list) -> List[dict]:
    candidates: Dict[tuple, dict] = {}
    mention_types: Dict[str, Set[str]] = {}
    for entity in raw_entities if isinstance(raw_entities, list) else []:
        if not isinstance(entity, dict) or entity.get("type") not in ENTITY_TYPES:
            continue
        mention = str(entity.get("mention", "")).strip()
        name = str(entity.get("name", mention)).strip()
        if not mention or mention not in text or not name:
            continue
        if len(mention) == 1 and entity["type"] in {"Exam", "Treatment"}:
            continue
        mention_types.setdefault(mention, set()).add(entity["type"])
        candidates[(mention, entity["type"])] = {
            "mention": mention,
            "name": name,
            "type": entity["type"],
        }
    conflicts = {mention for mention, types in mention_types.items() if len(types) > 1}
    entities = [
        entity for (mention, _), entity in candidates.items()
        if mention not in conflicts
    ]
    entities.sort(key=lambda entity: (text.find(entity["mention"]), -len(entity["mention"])))
    return entities


def _validated_records(texts: List[str], labels: List, min_entities: int = 6) -> List[dict]:
    by_index = {
        item["index"]: item.get("entities", [])
        for item in labels
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    records = []
    for index, text in enumerate(texts):
        if not (50 <= len(text) <= 220):
            continue
        entities = _clean_entities(text, by_index.get(index, []))
        types = {entity["type"] for entity in entities}
        required = _anchor_types(text) & {"Disease", "Symptom", "Drug", "Exam"}
        if len(entities) < min_entities or len(types) < 3 or len(required - types) > 1:
            continue
        records.append({"text": text, "entities": entities})
    return records


def _stats(rows: List[dict]) -> dict:
    type_counts = Counter(
        entity["type"] for row in rows for entity in row["entities"]
    )
    return {
        "documents": len(rows),
        "entities": sum(type_counts.values()),
        "entity_types": dict(sorted(type_counts.items())),
        "documents_with_treatment": sum(
            any(entity["type"] == "Treatment" for entity in row["entities"])
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成广域、类别均衡 OOD 困难训练数据")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--batch-cases", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--label-url", default=None,
                        help="可选独立标注端点；默认沿用 OOD_LLM_URL")
    parser.add_argument("--label-key-env", default="OOD_LLM_KEY",
                        help="独立标注 API key 的环境变量名")
    parser.add_argument("--label-model", default=None,
                        help="可选独立标注模型；默认沿用 OOD_LLM_MODEL")
    parser.add_argument("--skip-review", action="store_true",
                        help="跳过第三次质控调用；仍保留独立生成/标注与确定性审计")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "ood_train_v2" / "quality_raw.jsonl"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"训练集已存在，使用 --resume 或 --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    forbidden = _forbidden_hashes(output)
    key = os.environ["OOD_LLM_KEY"]
    url = os.environ.get("OOD_LLM_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    model = os.environ.get("OOD_LLM_MODEL", "mimo-v2.5")
    label_url = args.label_url or url
    label_key = os.environ[args.label_key_env]
    label_model = args.label_model or model

    accepted: List[dict] = []
    if output.exists() and args.resume:
        accepted = list(_iter_jsonl([output]))
        print(f"[resume] existing={len(accepted)}/{args.n}", flush=True)
    seen = set(forbidden) | {_hash(row["text"]) for row in accepted}
    mode = "a" if accepted else "w"
    max_calls = max(args.n, 40)
    calls = 0
    with output.open(mode, encoding="utf-8") as fh:
        while len(accepted) < args.n and calls < max_calls:
            calls += 1
            offset = len(accepted) + calls
            topics = [
                TOPIC_GROUPS[(offset + i) % len(TOPIC_GROUPS)]
                for i in range(args.batch_cases)
            ]
            styles = [
                STYLE_LENSES[(offset + i * 2) % len(STYLE_LENSES)]
                for i in range(args.batch_cases)
            ]
            raw_texts = _json_array(_chat(
                url, key, model, _generation_prompt(topics, styles), 1.0,
                args.request_timeout,
            ))
            texts = [
                str(text).replace("\n", " ").strip()
                for text in raw_texts if isinstance(text, str)
            ]
            labels = _json_array(_chat(
                label_url, label_key, label_model, _label_prompt(texts), 0.1,
                args.request_timeout,
            )) if texts else []
            reviewed = _json_array(_chat(
                label_url, label_key, label_model, _review_prompt(texts, labels), 0.0,
                args.request_timeout,
            )) if labels and not args.skip_review else []
            final_labels = reviewed if reviewed else labels
            for record in _validated_records(texts, final_labels):
                digest = _hash(record["text"])
                if digest in seen:
                    continue
                seen.add(digest)
                record["doc_id"] = f"quality_v2_{len(accepted):04d}"
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                accepted.append(record)
                if len(accepted) >= args.n:
                    break
            print(f"calls={calls}, accepted={len(accepted)}/{args.n}", flush=True)

    manifest = {
        **_stats(accepted),
        "generation_calls": calls,
        "forbidden_text_hashes": len(forbidden),
        "exact_overlap": 0,
        "protocol": (
            "independent generation/label/review calls; training only; "
            "all existing OOD/blind hashes forbidden"
        ),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
