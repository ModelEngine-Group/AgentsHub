"""GraphRAG QA Agent (Task 2): answer questions over the knowledge graph.

Pipeline:
  1. Run medical_ner on the question to locate anchor entities.
  2. Resolve anchors to graph nodes; traverse a 1~2 hop subgraph for evidence.
  3. Feed the subgraph triples to the LLM to compose a grounded answer.
  4. Return the answer plus provenance (the triples / source documents hit).

If no graph anchors are found, it degrades gracefully and tells the user.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from medigraph.graph.base import GraphStore, get_graph_store
from medigraph.graph.vector_store import LocalVectorStore
from medigraph.operators.base import get_operator, load_default_operators
from medigraph.schema.cmeie_schema import predicate_zh
from medigraph.schema.ontology import RELATION_TYPES

_SYSTEM = "你是医疗知识问答助手。优先依据提供的知识图谱三元组证据作答，可参考文本片段补充；不要编造；证据不足时如实说明。"

_ANSWER_PROMPT = """用户问题：{question}

【知识图谱三元组证据】（结构化、可溯源，每条含来源文档）：
{evidence}

【相关文本片段】（向量检索补充，仅供参考）：
{passages}

【多跳推理路径】（按路径置信度排序）：
{paths}

请基于上述证据用中文回答问题，要求：
1. 优先使用三元组证据；文本片段仅作补充。不要臆造。
2. 在答案末尾用「依据：」列出你引用的关键三元组。
3. 若证据不足以回答，请明确说明。
"""

# Question intent -> ontology relations.  This keeps the evidence budget focused
# on what the user actually asked instead of returning arbitrary high-degree
# neighbours first (important on the 26k-edge CM3KG graph).
#: Each rule maps question keywords to relation KEYS, not to one graph's naming
#: scheme -- deliberately pairing the CM3KG-import compact key (`has_symptom`,
#: `recommend_drug`, ...) with its CMeIE-V2 counterpart (`cmeie:clinical_
#: manifestation`, `cmeie:drug_treatment`, ...) wherever both graphs express the
#: same relationship. graph.json (CM3KG import) and graph_scaled.json (self-
#: produced, neural-GPLinker-built over the full 53-row CMeIE schema) use
#: different relation vocabularies for overlapping concepts; a rule that only
#: recognised the CM3KG key returned an *empty* intent list for the same
#: question asked against the self-produced graph. With no matched intent,
#: `select_evidence` falls back to "top-24 by confidence, no relation-type
#: balancing" -- and since `cmeie:clinical_manifestation` is by far the most
#: numerous relation in the self-produced graph (~28% of all edges), it can
#: bury the specific relation type the question actually asked about (e.g. a
#: "相关（导致）" or "病因" question ends up starved of related_cause/etiology
#: evidence and answered mostly from clinical_manifestation edges instead).
#: Found via benchmarks/build_kg_qa_human.py's graph_scaled.json-grounded
#: question set, which exercises the real agent end to end rather than the
#: graph-traversal self-consistency check in eval_kg_qa.py.
_INTENT_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("症状", "表现", "体征", "symptom"), ("has_symptom", "cmeie:clinical_manifestation")),
    (("药物", "用药", "治疗药", "吃什么药", "drug"), ("recommend_drug", "cmeie:drug_treatment")),
    (("检查", "检验", "筛查", "诊断方法", "examination"),
     ("need_examination", "cmeie:laboratory_examination", "cmeie:imaging_examination",
      "cmeie:auxiliary_examination", "cmeie:histological_examination", "cmeie:endoscopic_examination")),
    (("并发", "并发症", "风险", "complication"), ("complication", "cmeie:complication")),
    (("科室", "挂号", "就诊", "department"), ("treated_in_department", "cmeie:department")),
    (("手术", "操作", "procedure"), ("treated_by_procedure", "cmeie:surgical_treatment")),
    (("阳性", "标志物", "marker"), ("positive_marker",)),
    (("阴性",), ("negative_marker",)),
    (("基因", "突变", "gene"), ("associated_gene",)),
    (("形态", "镜下", "病理表现", "morphology"), ("has_morphology",)),
    (("部位", "位于", "发生于", "location"), ("located_in", "cmeie:onset_site")),
    (("亚型", "分型", "subtype"), ("subtype_of",)),
    (("禁忌", "contraindication"), ("contraindication",)),
    (("不良反应", "副作用", "adverse"), ("adverse_reaction",)),
    # CMeIE-only categories with no CM3KG counterpart in the compact ontology.
    (("相关", "关联", "有关", "related"), ("cmeie:related_cause", "cmeie:related_symptom", "cmeie:related_transformation")),
    (("病因", "原因", "为什么会得", "etiology"), ("cmeie:etiology",)),
    (("鉴别诊断", "如何区分", "differential"), ("cmeie:differential_diagnosis",)),
    (("病理分型", "分类", "classification"), ("cmeie:pathological_classification",)),
    (("高危因素", "危险因素", "risk factor"), ("cmeie:high_risk_factor", "cmeie:risk_assessment_factor"),),
    (("辅助治疗", "非药物治疗", "adjuvant"), ("cmeie:adjuvant_therapy",)),
    (("化疗", "chemotherapy"), ("cmeie:chemotherapy",)),
    (("放疗", "放射治疗", "radiotherapy"), ("cmeie:radiotherapy",)),
    (("预防", "prevention"), ("cmeie:prevention",)),
    (("预后", "存活率", "生存率", "prognosis"), ("cmeie:prognosis_status", "cmeie:prognosis_survival_rate")),
    (("多发群体", "易感人群", "susceptible"), ("cmeie:susceptible_population",)),
    (("发病率", "incidence"), ("cmeie:incidence",)),
    (("发病年龄", "onset age"), ("cmeie:onset_age",)),
    (("遗传", "genetic"), ("cmeie:genetic_factor",)),
    (("发病机制", "pathogenesis"), ("cmeie:pathogenesis",)),
]


def _relation_label(triple: dict) -> str:
    """Chinese label for one evidence edge's relation, across both graph flavours.

    `graph.json` (CM3KG import) stores an explicit `relation_zh` per edge.
    `graph_scaled.json` (self-produced, neural-GPLinker-built) does not -- its
    edges carry the raw `cmeie:xxx` schema key -- so without a fallback the QA
    prompt (and thus the LLM's composed answer) would show literal strings like
    "cmeie:pathological_classification" instead of "病理分型". Falls through
    explicit relation_zh -> CM3KG ontology label -> CMeIE predicate label -> the
    raw key, so it degrades gracefully rather than raising on an unknown key.
    """
    relation = triple.get("relation", "")
    return (
        triple.get("relation_zh")
        or RELATION_TYPES.get(relation)
        or predicate_zh(relation)
    )


def infer_intent_relations(question: str) -> list[str]:
    """Infer requested relation types deterministically from the question."""
    q = question.lower()
    out: list[str] = []
    for keywords, relations in _INTENT_RULES:
        if any(keyword.lower() in q for keyword in keywords):
            for relation in relations:
                if relation not in out:
                    out.append(relation)
    return out


def rank_evidence(
    evidence: list[dict],
    intent_relations: list[str],
    anchors: list[str] | None = None,
) -> list[dict]:
    """Rank relevant relation types first, then confidence and provenance."""
    priority = {relation: index for index, relation in enumerate(intent_relations)}
    anchor_set = set(anchors or [])

    def key(triple: dict) -> tuple:
        relation = str(triple.get("relation", ""))
        head = str(triple.get("head", ""))
        tail = str(triple.get("tail", ""))
        return (
            0 if relation in priority else 1,
            priority.get(relation, len(priority)),
            0 if head in anchor_set else 1 if tail in anchor_set else 2,
            -float(triple.get("confidence", 0.0) or 0.0),
            0 if triple.get("source") else 1,
            head,
            tail,
        )

    return sorted(evidence, key=key)


def select_evidence(
    evidence: list[dict],
    intent_relations: list[str],
    limit: int = 24,
    anchors: list[str] | None = None,
) -> list[dict]:
    """Balance evidence across requested relations and keep prompts concise."""
    if not intent_relations:
        return evidence[:limit]
    focused = [triple for triple in evidence if triple.get("relation") in intent_relations]
    if not focused:
        return evidence[:limit]
    anchor_set = set(anchors or [])
    direct_outgoing = [
        triple for triple in focused if str(triple.get("head", "")) in anchor_set
    ]
    if direct_outgoing:
        return direct_outgoing[:limit]
    per_relation = 12 if len(intent_relations) == 1 else 8
    counts: dict[str, int] = {}
    selected = []
    for triple in focused:
        relation = str(triple.get("relation", ""))
        if counts.get(relation, 0) >= per_relation:
            continue
        selected.append(triple)
        counts[relation] = counts.get(relation, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def score_answer_confidence(evidence: list[dict], passages: list[dict] | None = None) -> dict:
    """Compute an explainable answer-level confidence and safety grade."""
    passages = passages or []
    values = [float(item.get("confidence", 0.0) or 0.0) for item in evidence]
    if values:
        evidence_score = sum(values) / len(values)
        source_ratio = sum(bool(item.get("source") or item.get("sources")) for item in evidence) / len(evidence)
        score = 0.9 * evidence_score + 0.1 * source_ratio
    elif passages:
        score = 0.5
        source_ratio = sum(bool(item.get("source")) for item in passages) / len(passages)
    else:
        score = source_ratio = 0.0
    grade = "high" if score >= 0.85 else "medium" if score >= 0.65 else "low"
    return {
        "score": round(score, 4),
        "grade": grade,
        "evidence_mean": round(sum(values) / len(values), 4) if values else 0.0,
        "source_coverage": round(source_ratio, 4),
    }


class QAAgent:
    def __init__(self, llm: Any | None = None, store: GraphStore | None = None, hops: int = 2,
                 vector_store: LocalVectorStore | None = None, top_k: int = 3,
                 min_answer_confidence: float = 0.55):
        if llm is None:
            from medigraph.llm.client import LLMClient
            llm = LLMClient()
        self.llm = llm
        load_default_operators(llm=llm)
        self.ner = get_operator("medical_ner")
        self.store = store or get_graph_store()
        self.hops = hops
        self.vector_store = vector_store  # optional: enables hybrid retrieval
        self.top_k = top_k
        self.min_answer_confidence = min_answer_confidence

    def answer(
        self,
        question: str,
        verbose: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        """Grounded answer over the graph (+ optional vector passages).

        When `on_token` is supplied the composition step streams and each delta is
        handed to the callback as it arrives; the return value is unchanged, so
        callers that want the whole payload (tests, CLI, MCP) are unaffected. This
        is what lets the HTTP layer emit SSE without duplicating retrieval.
        """
        intent_relations = infer_intent_relations(question)
        # 1) anchors via NER on the question
        ents = self.ner.run({"text": question}).get("entities", [])
        anchor_names = [e["name"] for e in ents]
        if verbose:
            print(f"[QA] question anchors: {anchor_names}")

        # 2) resolve to graph nodes + traverse
        resolved = self.store.find_entities(anchor_names) if anchor_names else []
        # Deterministic fallback: graph node names occurring literally in the
        # question can still anchor retrieval when the LLM NER misses.
        if not resolved:
            resolved = self.store.find_entities([question])
        evidence: list[dict] = []
        reasoning_paths: list[dict] = []
        seen = set()
        for node in resolved:
            for t in self.store.neighbors(node, hops=self.hops):
                key = (t["head"], t["relation"], t["tail"])
                if key not in seen:
                    seen.add(key)
                    evidence.append(t)
            if hasattr(self.store, "traverse_paths"):
                reasoning_paths.extend(self.store.traverse_paths(node, hops=min(3, max(1, self.hops))))

        evidence_total = len(evidence)
        evidence = rank_evidence(evidence, intent_relations, resolved)
        evidence = select_evidence(evidence, intent_relations, anchors=resolved)
        reasoning_paths.sort(
            key=lambda path: (
                -sum(relation in intent_relations for relation in path.get("relations", [])),
                -float(path.get("confidence", 0.0)),
                path.get("hops", 0),
            )
        )
        reasoning_paths = reasoning_paths[:12]

        # 2b) vector retrieval (hybrid GraphRAG): top-k passages for the question
        passages: list[dict] = []
        if self.vector_store is not None and self.vector_store.size:
            qvec = self.llm.embed([question])
            if qvec and qvec[0]:
                passages = self.vector_store.search(qvec[0], k=self.top_k)
        if verbose:
            print(f"[QA] graph evidence: {len(evidence)} triples; vector passages: {len(passages)}")

        if not evidence and not passages:
            return {
                "question": question,
                "answer": "知识图谱与文本库中均未找到与该问题相关的证据，无法作答。",
                "anchors": anchor_names,
                "resolved_entities": resolved,
                "intent_relations": intent_relations,
                "retrieval_mode": "none",
                "evidence_total": 0,
                "evidence_used": 0,
                "evidence": [],
                "passages": [],
                "citations": [],
                "reasoning_paths": [],
                "answer_confidence": score_answer_confidence([], []),
                "refused": True,
            }

        answer_confidence = score_answer_confidence(evidence, passages)
        if answer_confidence["score"] < self.min_answer_confidence:
            return {
                "question": question,
                "answer": (
                    "已检索到相关线索，但证据置信度不足，基于医疗安全策略暂不生成确定性答案。"
                    "建议补充高质量来源或由专业人员复核。"
                ),
                "anchors": anchor_names,
                "resolved_entities": resolved,
                "intent_relations": intent_relations,
                "retrieval_mode": "hybrid" if passages else "graph",
                "evidence_total": evidence_total,
                "evidence_used": len(evidence),
                "evidence": evidence,
                "passages": passages,
                "citations": [],
                "reasoning_paths": reasoning_paths,
                "answer_confidence": answer_confidence,
                "refused": True,
            }

        # 3) compose grounded answer from both evidence sources
        evidence_str = "\n".join(
            f"  - {t['head']} --[{_relation_label(t)}]--> {t['tail']} "
            f"(conf={t.get('confidence')}, 来源={t.get('source','')})"
            for t in evidence[:40]
        ) or "  (无)"
        passages_str = "\n".join(
            f"  - [{p.get('source','')}] {p['text'][:200].strip()}" for p in passages
        ) or "  (无)"
        paths_str = "\n".join(
            "  - "
            + " -> ".join(
                f"{step['head']} -[{_relation_label(step)}]-> {step['tail']}"
                for step in path["steps"]
            )
            + f" (path_conf={path['confidence']})"
            for path in reasoning_paths[:8]
        ) or "  (无)"
        prompt = _ANSWER_PROMPT.format(
            question=question,
            evidence=evidence_str,
            passages=passages_str,
            paths=paths_str,
        )
        if on_token is not None:
            pieces: list[str] = []
            for piece in self.llm.chat_stream(prompt, system=_SYSTEM, temperature=0.2):
                pieces.append(piece)
                on_token(piece)
            answer = "".join(pieces)
        else:
            answer = self.llm.chat(prompt, system=_SYSTEM, temperature=0.2)
        citations = [
            {
                "id": index,
                "triple": f"{t['head']} --[{_relation_label(t)}]--> {t['tail']}",
                "relation": t["relation"],
                "confidence": t.get("confidence"),
                "source": t.get("source", ""),
            }
            for index, t in enumerate(evidence[:12], start=1)
        ]

        return {
            "question": question,
            "answer": answer,
            "anchors": anchor_names,
            "resolved_entities": resolved,
            "intent_relations": intent_relations,
            "retrieval_mode": "hybrid" if passages else "graph",
            "evidence_total": evidence_total,
            "evidence_used": len(evidence),
            "evidence": evidence[:40],
            "passages": passages,
            "citations": citations,
            "reasoning_paths": reasoning_paths,
            "answer_confidence": answer_confidence,
            "refused": False,
        }
