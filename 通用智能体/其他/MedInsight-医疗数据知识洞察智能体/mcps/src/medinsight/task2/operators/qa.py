from __future__ import annotations

from typing import Any

from medinsight.common.operators.base import OperatorSpec


SYMPTOM_LIKE_TYPES = {"症状", "体征", "临床表现", "医学事件"}
CLINICAL_CONTEXT_PREDICATES = {"临床表现", "因果相关", "相关（导致）"}
CLINICAL_REVERSE_KEYWORDS = ("哪些病", "什么病", "哪种病", "哪些疾病", "什么疾病", "见于", "会出现")

QUESTION_ENTITY_ALIASES = {
    "感冒": ["普通感冒", "流行性感冒", "风寒感冒"],
    "流感": ["甲型流感", "流行性感冒"],
    "发烧": ["发热"],
}

QUESTION_RELATION_HINTS = {
    "症状": ["临床表现"],
    "表现": ["临床表现"],
    "哪些病": ["临床表现", "因果相关", "相关（导致）"],
    "什么病": ["临床表现", "因果相关", "相关（导致）"],
    "哪些疾病": ["临床表现", "因果相关", "相关（导致）"],
    "什么疾病": ["临床表现", "因果相关", "相关（导致）"],
    "见于": ["临床表现", "因果相关", "相关（导致）"],
    "治疗": ["药物治疗", "辅助治疗", "手术治疗", "化疗", "放射治疗"],
    "医治": ["药物治疗", "辅助治疗", "手术治疗", "化疗", "放射治疗"],
    "诊治": ["药物治疗", "辅助治疗", "手术治疗", "化疗", "放射治疗"],
    "怎么治": ["药物治疗", "辅助治疗", "手术治疗", "化疗", "放射治疗"],
    "如何治": ["药物治疗", "辅助治疗", "手术治疗", "化疗", "放射治疗"],
    "治法": ["药物治疗", "辅助治疗", "手术治疗", "化疗", "放射治疗"],
    "疗法": ["药物治疗", "辅助治疗", "手术治疗", "化疗", "放射治疗"],
    "药": ["药物治疗"],
    "用药": ["药物治疗"],
    "管理": ["辅助治疗", "药物治疗"],
    "生活方式": ["辅助治疗"],
    "饮食": ["辅助治疗"],
    "检查": ["实验室检查", "影像学检查", "辅助检查", "内窥镜检查", "组织学检查", "筛查"],
    "并发": ["并发症"],
    "病因": ["病因", "因果相关", "相关（导致）"],
    "原因": ["病因", "因果相关", "相关（导致）"],
    "导致": ["因果相关", "相关（导致）"],
    "引起": ["因果相关", "相关（导致）"],
    "造成": ["因果相关", "相关（导致）"],
    "编码": ["ICD编码", "手术编码"],
    "代码": ["ICD编码", "手术编码"],
    "ICD": ["ICD编码"],
    "icd": ["ICD编码"],
    "手术编码": ["手术编码"],
    "标准": ["标准化为", "同义词"],
    "归一": ["标准化为"],
    "别名": ["同义词", "标准化为"],
    "分型": ["病理分型"],
    "类型": ["病理分型"],
    "科室": ["就诊科室"],
    "预防": ["预防", "辅助治疗"],
    "部位": ["发病部位"],
}


class KnowledgeGraphQAOperator:
    spec = OperatorSpec(
        name="kg_qa_agent",
        description="Answer medical questions by querying the generated knowledge graph.",
        input_schema={"type": "object", "properties": {"graph": {"type": "object"}, "question": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"answer": {"type": "object"}}},
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        graph = payload.get("graph") or {}
        question = str(payload.get("question") or "").strip()
        answer = answer_question(graph, question)
        return {"answer": answer}


def answer_question(graph: dict[str, Any], question: str, max_facts: int = 10) -> dict[str, Any]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_by_id = {node["node_id"]: node for node in nodes}
    predicates = infer_predicates(question)
    matched_nodes = find_question_entities(nodes, question, edges, predicates)
    if asks_multi_entity_question(question) and len(matched_nodes) > 1:
        multi_answer = build_multi_entity_answer(
            matched_nodes=matched_nodes,
            question=question,
            predicates=predicates,
            edges=edges,
            node_by_id=node_by_id,
            max_facts=max_facts,
        )
        if multi_answer:
            return multi_answer
    matched_node = matched_nodes[0] if matched_nodes else None
    if not matched_node:
        return {
            "question": question,
            "status": "not_found",
            "answer": "未在当前知识图谱中找到问题对应的医学实体。",
            "facts": [],
        }
    return answer_for_matched_node(
        matched_node=matched_node,
        question=question,
        predicates=predicates,
        edges=edges,
        node_by_id=node_by_id,
        max_facts=max_facts,
    )


def answer_for_matched_node(
    *,
    matched_node: dict[str, Any],
    question: str,
    predicates: list[str],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    max_facts: int,
) -> dict[str, Any]:
    if asks_reverse_clinical_question(question):
        context_answer = build_symptom_context_answer(
            matched_node=matched_node,
            question=question,
            predicates=predicates,
            edges=edges,
            node_by_id=node_by_id,
            max_facts=max_facts,
        )
        if context_answer:
            return context_answer
    facts = collect_outgoing_facts(matched_node, predicates, edges, node_by_id, max_facts)
    if not facts:
        facts = collect_related_disease_facts(matched_node, predicates, edges, node_by_id, max_facts)
    if not facts:
        context_answer = build_symptom_context_answer(
            matched_node=matched_node,
            question=question,
            predicates=predicates,
            edges=edges,
            node_by_id=node_by_id,
            max_facts=max_facts,
        )
        if context_answer:
            return context_answer
        return {
            "question": question,
            "status": "no_relation",
            "answer": f"当前图谱中找到了 {matched_node['name']}，但没有匹配到该问题需要的关系。",
            "entity": matched_node,
            "facts": [],
        }
    grouped = group_fact_objects(facts)
    parts = [f"{predicate}：{'、'.join(objects)}" for predicate, objects in grouped.items()]
    related_subjects = related_fact_subjects(matched_node, facts)
    answer_name = str(matched_node["name"])
    if related_subjects:
        answer_name = f"{answer_name}（参考相关分型：{'、'.join(related_subjects[:4])}）"
    evidence_paths = build_evidence_paths(facts)
    return {
        "question": question,
        "status": "answered",
        "entity": matched_node,
        "predicates": predicates,
        "answer": build_evidence_grounded_answer(answer_name, parts, evidence_paths),
        "facts": facts,
        "evidence_paths": evidence_paths,
    }


def collect_outgoing_facts(
    matched_node: dict[str, Any],
    predicates: list[str],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    max_facts: int,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for edge in edges:
        if edge.get("source") != matched_node["node_id"]:
            continue
        if predicates and edge.get("predicate") not in predicates:
            continue
        target = node_by_id.get(edge.get("target"))
        if not target:
            continue
        facts.append(
            {
                "subject": matched_node["name"],
                "predicate": edge["predicate"],
                "object": target["name"],
                "object_type": target["type"],
                "evidence": (edge.get("evidence") or [])[:1],
            }
        )
        if len(facts) >= max_facts:
            break
    return facts


def collect_related_disease_facts(
    matched_node: dict[str, Any],
    predicates: list[str],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    max_facts: int,
) -> list[dict[str, Any]]:
    related_ids = related_disease_node_ids(matched_node, edges, node_by_id)
    if not related_ids:
        return []
    facts: list[dict[str, Any]] = []
    for edge in edges:
        if edge.get("source") not in related_ids:
            continue
        if predicates and edge.get("predicate") not in predicates:
            continue
        target = node_by_id.get(edge.get("target"))
        source = node_by_id.get(edge.get("source"))
        if not target or not source:
            continue
        facts.append(
            {
                "subject": source["name"],
                "predicate": edge["predicate"],
                "object": target["name"],
                "object_type": target["type"],
                "evidence": (edge.get("evidence") or [])[:1],
                "via_anchor": matched_node["name"],
            }
        )
        if len(facts) >= max_facts:
            break
    return facts


def related_disease_node_ids(
    matched_node: dict[str, Any],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    if matched_node.get("type") != "疾病":
        return []
    node_id = matched_node["node_id"]
    related: list[str] = []
    for edge in edges:
        if edge.get("predicate") != "病理分型":
            continue
        other_id = ""
        if edge.get("source") == node_id:
            other_id = str(edge.get("target") or "")
        elif edge.get("target") == node_id:
            other_id = str(edge.get("source") or "")
        if not other_id or other_id == node_id:
            continue
        other = node_by_id.get(other_id)
        if other and other.get("type") == "疾病" and other_id not in related:
            related.append(other_id)
    return related


def group_fact_objects(facts: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for fact in facts:
        values = grouped.setdefault(str(fact["predicate"]), [])
        value = str(fact["object"])
        if value not in values:
            values.append(value)
    return grouped


def related_fact_subjects(anchor_node: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    anchor_name = str(anchor_node.get("name") or "")
    for fact in facts:
        if fact.get("via_anchor") != anchor_name:
            continue
        subject = str(fact.get("subject") or "")
        if subject and subject != anchor_name and subject not in names:
            names.append(subject)
    return names


def asks_multi_entity_question(question: str) -> bool:
    return any(keyword in question for keyword in ("分别", "各自", "和", "与", "及"))


def build_multi_entity_answer(
    *,
    matched_nodes: list[dict[str, Any]],
    question: str,
    predicates: list[str],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    max_facts: int,
) -> dict[str, Any] | None:
    sections: list[str] = []
    all_facts: list[dict[str, Any]] = []
    answered_entities: list[dict[str, Any]] = []
    for node in matched_nodes[:5]:
        result = answer_for_matched_node(
            matched_node=node,
            question=question,
            predicates=predicates,
            edges=edges,
            node_by_id=node_by_id,
            max_facts=max(3, max_facts // 2),
        )
        if result.get("status") != "answered":
            continue
        facts = result.get("facts") or []
        grouped = group_fact_objects(facts)
        parts = [f"{predicate}：{'、'.join(objects)}" for predicate, objects in grouped.items()]
        related_subjects = related_fact_subjects(node, facts)
        name = str(node.get("name") or "")
        if related_subjects:
            name = f"{name}（参考相关分型：{'、'.join(related_subjects[:3])}）"
        sections.append(f"{name}：" + "；".join(parts) + "。")
        all_facts.extend(facts)
        answered_entities.append(node)
    if not sections:
        return None
    evidence_paths = build_evidence_paths(all_facts[:max_facts])
    answer_lines = [f"{index}. {section}" for index, section in enumerate(sections, start=1)]
    if evidence_paths:
        answer_lines.extend(["", "证据路径："])
        for index, item in enumerate(select_diverse_evidence_paths(evidence_paths), start=1):
            answer_lines.append(f"{index}. {item['path']}")
            if item.get("evidence"):
                answer_lines.append(f"   来源文本：{item['evidence']}")
    return {
        "question": question,
        "status": "answered",
        "entity": answered_entities[0],
        "entities": answered_entities,
        "predicates": predicates,
        "answer": "\n".join(answer_lines),
        "facts": all_facts[:max_facts],
        "evidence_paths": evidence_paths,
        "multi_entity": True,
    }


def build_symptom_context_answer(
    *,
    matched_node: dict[str, Any],
    question: str,
    predicates: list[str],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    max_facts: int,
) -> dict[str, Any] | None:
    node_id = matched_node["node_id"]
    requested_clinical_context = not predicates or bool(set(predicates) & CLINICAL_CONTEXT_PREDICATES)
    if not requested_clinical_context:
        return None

    incoming_edges = [
        edge
        for edge in edges
        if edge.get("target") == node_id and edge.get("predicate") in CLINICAL_CONTEXT_PREDICATES
    ]
    if not incoming_edges:
        return None

    related_sources: list[str] = []
    co_occurring_symptoms: list[str] = []
    facts: list[dict[str, Any]] = []
    source_ids: list[str] = []

    for edge in incoming_edges:
        source = node_by_id.get(edge.get("source"))
        if not source:
            continue
        append_unique(source_ids, str(edge.get("source")))
        append_unique(related_sources, str(source.get("name") or ""))
        facts.append(
            {
                "subject": source.get("name"),
                "predicate": edge.get("predicate"),
                "object": matched_node.get("name"),
                "object_type": matched_node.get("type"),
                "evidence": (edge.get("evidence") or [])[:1],
                "direction": "incoming",
            }
        )
        if len(facts) >= max_facts:
            break

    source_id_set = set(source_ids)
    for edge in edges:
        if edge.get("source") not in source_id_set:
            continue
        if edge.get("target") == node_id or edge.get("predicate") != "临床表现":
            continue
        target = node_by_id.get(edge.get("target"))
        if not target or target.get("name") == matched_node.get("name"):
            continue
        target_name = str(target.get("name") or "")
        matched_name = str(matched_node.get("name") or "")
        if equivalent_symptom_name(matched_name, target_name):
            continue
        if is_symptom_like(target) and evidence_mentions_pair(edge, matched_name, target_name):
            append_unique(co_occurring_symptoms, str(target.get("name") or ""))

    facts = facts[:max_facts]
    source_preview = "、".join(related_sources[:6])
    symptom_preview = "、".join(co_occurring_symptoms[:8])
    matched_name = str(matched_node.get("name") or "")
    answer_parts = [f"{matched_name}本身是一个症状/临床表现，当前图谱中它更多作为其他疾病或医学事件的表现/结果出现。"]

    if co_occurring_symptoms and asks_for_symptom_context(question, matched_node):
        answer_parts.append(f"同源证据中与它共同出现的表现包括：{symptom_preview}。")
    if related_sources:
        answer_parts.append(f"图谱中与它相关的疾病或场景包括：{source_preview}。")
    answer_parts.append("以上结论来自当前已构建的医疗知识图谱证据。")

    return {
        "question": question,
        "status": "answered",
        "entity": matched_node,
        "predicates": predicates,
        "answer": "".join(answer_parts),
        "facts": facts,
        "evidence_paths": build_evidence_paths(facts),
        "reverse_lookup": True,
        "co_occurring_symptoms": co_occurring_symptoms[:8],
        "related_sources": related_sources[:6],
    }


def build_evidence_grounded_answer(entity_name: str, parts: list[str], evidence_paths: list[dict[str, Any]]) -> str:
    answer = f"{entity_name}的相关知识包括：" + "；".join(parts) + "。"
    if not evidence_paths:
        return answer
    lines = [answer, "", "证据路径："]
    for index, item in enumerate(select_diverse_evidence_paths(evidence_paths), start=1):
        lines.append(f"{index}. {item['path']}")
        if item.get("evidence"):
            lines.append(f"   来源文本：{item['evidence']}")
    return "\n".join(lines)


def select_diverse_evidence_paths(evidence_paths: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_predicates: set[str] = set()
    for item in evidence_paths:
        predicate = str(item.get("predicate") or "")
        if predicate in seen_predicates:
            continue
        selected.append(item)
        seen_predicates.add(predicate)
        if len(selected) >= limit:
            return selected
    for item in evidence_paths:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def build_evidence_paths(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for fact in facts:
        evidence_items = fact.get("evidence") or []
        first_evidence = evidence_items[0] if evidence_items else {}
        if isinstance(first_evidence, dict):
            evidence_text = str(first_evidence.get("text") or "")
            source_doc = str(first_evidence.get("doc_id") or "")
            source_refs = first_evidence.get("source_refs") or []
        else:
            evidence_text = str(first_evidence or "")
            source_doc = ""
            source_refs = []
        paths.append(
            {
                "path": f"{fact.get('subject')} --{fact.get('predicate')}--> {fact.get('object')}",
                "subject": fact.get("subject"),
                "predicate": fact.get("predicate"),
                "object": fact.get("object"),
                "evidence": evidence_text,
                "source_doc": source_doc,
                "source_refs": source_refs,
            }
        )
    return paths


def asks_for_symptom_context(question: str, matched_node: dict[str, Any]) -> bool:
    if is_symptom_like(matched_node) and any(keyword in question for keyword in ("症状", "表现", "伴随", "同时")):
        return True
    return not any(keyword in question for keyword in CLINICAL_REVERSE_KEYWORDS)


def asks_reverse_clinical_question(question: str) -> bool:
    return any(keyword in question for keyword in CLINICAL_REVERSE_KEYWORDS)


def is_symptom_like(node: dict[str, Any]) -> bool:
    return str(node.get("type") or "") in SYMPTOM_LIKE_TYPES


def equivalent_symptom_name(anchor: str, target: str) -> bool:
    if anchor == target:
        return True
    return {anchor, target} == {"发烧", "发热"}


def evidence_mentions_pair(edge: dict[str, Any], anchor: str, target: str) -> bool:
    texts = [str(item.get("text") or item) for item in edge.get("evidence") or []]
    anchors = {anchor}
    if anchor == "发烧":
        anchors.add("发热")
    if anchor == "发热":
        anchors.add("发烧")
    return any(any(item and item in text for item in anchors) and target in text for text in texts)


def append_unique(items: list[str], value: str) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in items:
        items.append(cleaned)


def find_question_entity(
    nodes: list[dict[str, Any]],
    question: str,
    edges: list[dict[str, Any]] | None = None,
    predicates: list[str] | None = None,
) -> dict[str, Any] | None:
    matches = find_question_entities(nodes, question, edges, predicates)
    return matches[0] if matches else None


def find_question_entities(
    nodes: list[dict[str, Any]],
    question: str,
    edges: list[dict[str, Any]] | None = None,
    predicates: list[str] | None = None,
) -> list[dict[str, Any]]:
    candidates = [node for node in nodes if node.get("name") and node["name"] in question]
    if not candidates:
        alias_names: list[str] = []
        for mention, aliases in QUESTION_ENTITY_ALIASES.items():
            if mention in question:
                alias_names.extend(aliases)
        candidates = [node for node in nodes if node.get("name") in alias_names]
    if not candidates:
        return []
    candidates = dedupe_nodes_by_id(candidates)
    if asks_reverse_clinical_question(question):
        symptom_candidates = [
            node
            for node in candidates
            if str(node.get("type") or "") in {"症状", "体征", "临床表现"}
        ]
        if symptom_candidates:
            candidates = symptom_candidates
        incoming_score = build_incoming_score(edges or [], predicates or [])
        candidates.sort(
            key=lambda item: (
                -incoming_score.get(item["node_id"], 0),
                item.get("type") != "疾病",
                -len(item.get("name", "")),
            )
        )
        return candidates
    outgoing_score = build_outgoing_score(edges or [], predicates or [])
    candidates.sort(
        key=lambda item: (
            -outgoing_score.get(item["node_id"], 0),
            item.get("type") != "疾病",
            question.find(str(item.get("name") or "")) if str(item.get("name") or "") in question else 9999,
            -len(item.get("name", "")),
        )
    )
    return candidates


def dedupe_nodes_by_id(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        result.append(node)
    return result


def build_incoming_score(edges: list[dict[str, Any]], predicates: list[str]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for edge in edges:
        if predicates and edge.get("predicate") not in predicates:
            continue
        target = str(edge.get("target") or "")
        if not target:
            continue
        scores[target] = scores.get(target, 0) + 1
    return scores


def build_outgoing_score(edges: list[dict[str, Any]], predicates: list[str]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for edge in edges:
        if predicates and edge.get("predicate") not in predicates:
            continue
        source = str(edge.get("source") or "")
        if not source:
            continue
        scores[source] = scores.get(source, 0) + 1
    return scores


def infer_predicates(question: str) -> list[str]:
    predicates: list[str] = []
    for keyword, relation_names in QUESTION_RELATION_HINTS.items():
        if keyword in question:
            for relation in relation_names:
                if relation not in predicates:
                    predicates.append(relation)
    return predicates


def build_qa_examples(graph: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    disease_nodes = [node for node in graph.get("nodes", []) if node.get("type") == "疾病"]
    questions: list[str] = []
    for node in disease_nodes[:limit]:
        questions.append(f"{node['name']}有什么症状？")
        questions.append(f"{node['name']}怎么治疗？")
    return [answer_question(graph, question) for question in questions[:limit]]
