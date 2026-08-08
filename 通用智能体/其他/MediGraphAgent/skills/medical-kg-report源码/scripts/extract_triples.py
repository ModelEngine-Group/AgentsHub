#!/usr/bin/env python3
"""Extract ontology-constrained medical relation triples from text via an
OpenAI-compatible API. Uses only `openai` (available in the Nexent runtime).
Prints the result as JSON to stdout.
"""
import argparse
import json
import os
import re

RELATIONS = ("has_symptom", "recommend_drug", "need_examination", "complication",
             "contraindication", "treated_in_department", "positive_marker",
             "negative_marker", "associated_gene", "has_morphology", "located_in", "subtype_of")

_SYSTEM = "你是严谨的医学关系抽取引擎，只输出 JSON，不要任何解释。"
_PROMPT = """从医学文本中抽取实体间关系三元组。relation 只能取：{rels}。
输出 JSON 对象：{{"triples":[{{"head":"","head_type":"","relation":"","tail":"","tail_type":"","confidence":0.0}}]}}

文本：
\"\"\"
{text}
\"\"\"
"""


def _extract_json(text: str):
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--api-key", default=os.getenv("SILICONFLOW_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    ap.add_argument("--api-base", default=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen/Qwen3.5-35B-A3B"))
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key or "EMPTY", base_url=args.api_base, timeout=120)
    resp = client.chat.completions.create(
        model=args.model, temperature=0.2,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": _PROMPT.format(rels="/".join(RELATIONS), text=args.text[:4000])}],
    )
    data = _extract_json(resp.choices[0].message.content or "")
    triples = [t for t in (data.get("triples", []) if isinstance(data, dict) else [])
               if isinstance(t, dict) and t.get("relation") in RELATIONS and t.get("head") and t.get("tail")]
    print(json.dumps({"status": "success", "count": len(triples), "triples": triples}, ensure_ascii=False))


if __name__ == "__main__":
    main()
