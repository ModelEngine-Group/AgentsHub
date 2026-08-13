"""图谱问答：无需 LLM 的模板问答引擎（LLM 可用时可对答案润色/兜底自由问法）。

支持问法（正则模板 + 实体链接）：
- X 有哪些症状 / X 的临床表现          -> HAS_SYMPTOM
- X 用什么药 / 吃什么药 / 治疗药物      -> TREATED_BY_DRUG
- X 怎么治疗 / 治疗方式                -> TREATED_BY (+ 药物合并展示)
- X 做什么检查 / 如何确诊              -> DIAGNOSED_BY
- X 有什么并发症                       -> HAS_COMPLICATION
- X 挂什么科 / 属于哪个科室            -> BELONGS_TO_DEPT
- 什么药对 X 禁忌 / X 不能用什么药     -> CONTRAINDICATED_FOR（反向）
- 出现 Y 可能是什么病                  -> HAS_SYMPTOM（反向）
- X 和 Y 有什么关系                    -> find_path
"""

import re
from typing import Any, Dict, List, Optional

from dki.kg.schema import relation_zh
from dki.kg.store import KnowledgeGraph

_PATTERNS = [
    # (正则, 关系, 方向 out/in, 答案模板)
    (r"(?P<e>[^，。？?的]+?)(?:有哪些|有什么|的)(?:症状|临床表现|表现)", "HAS_SYMPTOM", "out",
     "{e} 的常见临床表现：{items}"),
    (r"(?P<e>[^，。？?的]+?)(?:用什么药|吃什么药|的治疗药物|用药|首选什么药)", "TREATED_BY_DRUG", "out",
     "{e} 的常用治疗药物：{items}"),
    (r"(?P<e>[^，。？?的]+?)(?:怎么治疗|如何治疗|的治疗方式|怎么治)", "TREATED_BY", "out",
     "{e} 的治疗方式：{items}"),
    (r"(?P<e>[^，。？?的]+?)(?:做什么检查|需要哪些检查|如何确诊|怎么确诊|的诊断检查)", "DIAGNOSED_BY", "out",
     "{e} 的诊断/评估检查：{items}"),
    (r"(?P<e>[^，。？?的]+?)(?:有什么并发症|有哪些并发症|的并发症|会引起什么|可能并发)", "HAS_COMPLICATION", "out",
     "{e} 可能的并发症：{items}"),
    (r"(?P<e>[^，。？?的]+?)(?:挂什么科|挂哪个科|属于哪个科室|看什么科|的所属科室)", "BELONGS_TO_DEPT", "out",
     "{e} 建议就诊科室：{items}"),
    (r"(?:什么药对)?(?P<e>[^，。？?的]+?)(?:患者)?(?:不能用什么药|禁用什么药|用药禁忌|禁忌)", "CONTRAINDICATED_FOR", "in",
     "对 {e} 患者需禁用/慎用的药物：{items}"),
    (r"(?:出现|患有|表现为)?(?P<e>[^，。？?的]+?)(?:可能是什么病|可能得了什么病|常见于什么疾病|是什么病的症状)", "HAS_SYMPTOM", "in",
     "症状「{e}」可能相关的疾病：{items}"),
]

_PATH_PATTERN = re.compile(r"(?P<a>[^，。？?和与]+?)[和与](?P<b>[^，。？?有]+?)(?:之间)?有(?:什么|何)关系")


def answer_question(kg: KnowledgeGraph, question: str) -> Dict[str, Any]:
    """图谱模板问答。返回 {answered, answer, evidence, structured}。"""
    q = question.strip()

    m = _PATH_PATTERN.search(q)
    if m:
        a, b = m.group("a").strip(), m.group("b").strip()
        path = kg.find_path(a, b, max_hops=3)
        if path:
            steps = " → ".join(
                f"{t['head']} -[{relation_zh(t['relation'])}]-> {t['tail']}" for t in path
            )
            return {
                "answered": True,
                "answer": f"{a} 与 {b} 的图谱关联路径：{steps}",
                "structured": path,
                "evidence": [e for t in path for e in t.get("evidence", [])][:5],
            }
        return {"answered": False, "answer": f"图谱中未找到 {a} 与 {b} 的关联路径。", "structured": [], "evidence": []}

    for pattern, relation, direction, tpl in _PATTERNS:
        m = re.search(pattern, q)
        if not m:
            continue
        raw = m.group("e").strip()
        canonical = kg.resolve_name(raw)
        if canonical is None:
            continue  # 实体不在图谱，试下一个模板
        neighbors = kg.neighbors(canonical, relation=relation, direction=direction)
        if not neighbors:
            return {
                "answered": False,
                "answer": f"图谱中暂无「{canonical}」的{relation_zh(relation)}信息。",
                "structured": [],
                "evidence": [],
            }
        items = "、".join(
            f"{n['entity']}（置信度{n['confidence']:.2f}）" for n in neighbors[:8]
        )
        # 取证据句
        evidence: List[Dict[str, str]] = []
        for n in neighbors[:3]:
            key_head, key_tail = (canonical, n["entity"]) if direction == "out" else (n["entity"], canonical)
            t = kg._triples.get((key_head, relation, key_tail))
            if t:
                evidence.extend(t.get("evidence", [])[:1])
        return {
            "answered": True,
            "answer": tpl.format(e=canonical, items=items),
            "structured": neighbors,
            "evidence": evidence,
        }

    return {
        "answered": False,
        "answer": "未能匹配到图谱问答模板。可尝试：『X有哪些症状』『X用什么药』『X有什么并发症』『X和Y有什么关系』等问法。",
        "structured": [],
        "evidence": [],
    }
