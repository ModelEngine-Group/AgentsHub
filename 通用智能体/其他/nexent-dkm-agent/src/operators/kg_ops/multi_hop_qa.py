"""Multi-hop reasoning and evidence-chain QA for task 2.

Extends the single-hop graph QA with:
- Multi-hop path traversal
- Evidence chain construction
- LLM-enhanced natural language answer generation
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

from src.common.llm_config import openai_extra_kwargs

logger = logging.getLogger(__name__)

_MAX_HOPS = 4
_MAX_PATHS = 10


def multi_hop_query(
    graph: dict[str, Any],
    start_entity: str,
    target_entity: str | None = None,
    max_hops: int = 3,
    max_paths: int = 5,
) -> dict[str, Any]:
    """Find multi-hop paths between entities in the KG.

    Parameters
    ----------
    graph:
        The medical knowledge graph dict.
    start_entity:
        Name of the starting entity.
    target_entity:
        Optional target entity.  If *None*, returns all entities
        reachable within *max_hops*.
    max_hops:
        Maximum number of hops to traverse.
    max_paths:
        Maximum number of paths to return.
    """
    from src.operators.kg_ops.query import find_graph_entities

    start_match = find_graph_entities(start_entity, graph, limit=1)
    if not start_match["matches"]:
        return {
            "status": "unmatched",
            "start_entity": start_entity,
            "target_entity": target_entity,
            "paths": [],
        }

    start_node = start_match["matches"][0]
    node_lookup = {node["id"]: node for node in graph.get("nodes", [])}
    edge_list = graph.get("edges", [])

    target_id = None
    if target_entity:
        target_match = find_graph_entities(target_entity, graph, limit=1)
        if target_match["matches"]:
            target_id = target_match["matches"][0]["id"]

    adjacency = _build_adjacency(edge_list)
    paths = _find_paths(
        adjacency, start_node["id"], target_id, max_hops, max_paths
    )

    result_paths = []
    for path in paths:
        result_paths.append(_format_path(path, node_lookup, edge_list))

    return {
        "status": "matched" if result_paths else "unmatched",
        "start_entity": start_entity,
        "target_entity": target_entity,
        "start_node": start_node,
        "path_count": len(result_paths),
        "paths": result_paths,
    }


def build_evidence_chain(
    graph: dict[str, Any],
    question: str,
    max_hops: int = 2,
) -> dict[str, Any]:
    """Build an evidence chain from a question by traversing the graph.

    Detects entities in the question, finds their neighborhoods,
    and constructs reasoning chains.
    """
    from src.operators.kg_ops.query import find_graph_entities

    # Find all entities mentioned in the question
    mentioned_entities = []
    for node in graph.get("nodes", []):
        if node["name"] in question:
            match = find_graph_entities(node["name"], graph, limit=1)
            if match["matches"]:
                mentioned_entities.append(match["matches"][0])

    if not mentioned_entities:
        return {
            "status": "no_entities",
            "question": question,
            "chains": [],
        }

    node_lookup = {node["id"]: node for node in graph.get("nodes", [])}
    edge_list = graph.get("edges", [])
    adjacency = _build_adjacency(edge_list)

    chains = []
    for entity in mentioned_entities[:3]:  # Limit to top 3 entities
        paths = _find_paths(
            adjacency, entity["id"], None, max_hops, _MAX_PATHS
        )
        for path in paths[:3]:  # Top 3 paths per entity
            chain = _format_evidence_chain(path, node_lookup, edge_list, question)
            if chain:
                chains.append(chain)

    return {
        "status": "chains_found" if chains else "no_chains",
        "question": question,
        "entity_count": len(mentioned_entities),
        "chain_count": len(chains),
        "chains": chains,
    }


def answer_with_evidence_chain(
    question: str,
    graph: dict[str, Any],
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a question using evidence-chain reasoning with optional LLM.

    Falls back to template-based answers if LLM is unavailable.
    """
    chain_result = build_evidence_chain(graph, question, max_hops=3)
    from src.operators.kg_ops.qa import answer_graph_question

    base_answer = answer_graph_question(question, graph)

    if chain_result["status"] != "chains_found":
        return {
            **base_answer,
            "evidence_chain": chain_result,
            "reasoning_depth": "single_hop",
        }

    # Build enriched answer from chains
    chain_facts = _extract_facts_from_chains(chain_result["chains"])
    if base_answer.get("status") == "answered" and chain_facts:
        enriched = f"{base_answer['answer']} 推理链路：{'; '.join(chain_facts[:5])}。"
    elif chain_facts:
        enriched = f"根据图谱推理：{'; '.join(chain_facts[:5])}。"
    else:
        enriched = base_answer.get("answer", "无法回答。")

    result = {
        **base_answer,
        "answer": enriched,
        "evidence_chain": chain_result,
        "reasoning_depth": "multi_hop",
        "chain_facts": chain_facts,
    }

    # Optionally enhance with LLM
    if llm_config:
        try:
            result["answer"] = _llm_enhance_answer(question, enriched, chain_facts, llm_config)
            result["reasoning_depth"] = "multi_hop_llm"
        except Exception:
            logger.warning("LLM answer enhancement failed; using template.", exc_info=True)

    return result


def _build_adjacency(edge_list: list[dict[str, Any]]) -> dict[str, list[tuple[str, dict]]]:
    adj: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for edge in edge_list:
        source = edge.get("source", "")
        target = edge.get("target", "")
        adj[source].append((target, edge))
        adj[target].append((source, edge))
    return dict(adj)


def _find_paths(
    adjacency: dict[str, list[tuple[str, dict]]],
    start_id: str,
    target_id: str | None,
    max_hops: int,
    max_paths: int,
) -> list[list[tuple[str, str, dict]]]:
    """BFS path finding."""
    paths = []
    queue: deque[tuple[str, list[tuple[str, str, dict]], set[str]]] = deque()
    queue.append((start_id, [], {start_id}))

    while queue and len(paths) < max_paths:
        current, path, visited = queue.popleft()
        if len(path) > max_hops:
            continue
        if target_id and current == target_id and path:
            paths.append(path)
            continue
        if not target_id and path:
            paths.append(path)

        for neighbor, edge in adjacency.get(current, []):
            if neighbor not in visited and len(path) < max_hops:
                queue.append((
                    neighbor,
                    path + [(current, neighbor, edge)],
                    visited | {neighbor},
                ))

    return paths


def _format_path(
    path: list[tuple[str, str, dict]],
    node_lookup: dict[str, dict],
    edge_list: list[dict[str, Any]],
) -> dict[str, Any]:
    steps = []
    for source_id, target_id, edge in path:
        source_name = node_lookup.get(source_id, {}).get("name", source_id)
        target_name = node_lookup.get(target_id, {}).get("name", target_id)
        predicate = edge.get("predicate", "unknown")
        steps.append({
            "source": source_name,
            "predicate": predicate,
            "target": target_name,
            "confidence": edge.get("confidence", 0.0),
        })
    return {
        "hop_count": len(steps),
        "steps": steps,
        "entities": [s["source"] for s in steps] + ([steps[-1]["target"]] if steps else []),
    }


def _format_evidence_chain(
    path: list[tuple[str, str, dict]],
    node_lookup: dict[str, dict],
    edge_list: list[dict[str, Any]],
    question: str,
) -> dict[str, Any] | None:
    if not path:
        return None
    formatted = _format_path(path, node_lookup, edge_list)
    formatted["question_relevance"] = _assess_relevance(formatted, question)
    return formatted


def _assess_relevance(path_data: dict[str, Any], question: str) -> str:
    entities = path_data.get("entities", [])
    relevant = sum(1 for e in entities if e and e in question)
    if relevant >= 2:
        return "high"
    elif relevant >= 1:
        return "medium"
    return "low"


def _extract_facts_from_chains(chains: list[dict[str, Any]]) -> list[str]:
    facts = []
    seen = set()
    for chain in chains:
        for step in chain.get("steps", []):
            fact = f"{step['source']} {step['predicate']} {step['target']}"
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
    return facts


# Reasoning models need headroom beyond the visible answer length, otherwise
# the response truncates to empty and silently falls back to the template.
_DEFAULT_LLM_MAX_TOKENS = 4096


def _llm_enhance_answer(
    question: str,
    template_answer: str,
    chain_facts: list[str],
    llm_config: dict[str, Any],
) -> str:
    import openai

    facts_text = "\n".join(f"- {f}" for f in chain_facts[:8])
    prompt = (
        f"问题：{question}\n\n"
        f"已知知识图谱事实：\n{facts_text}\n\n"
        f"模板回答：{template_answer}\n\n"
        f"请基于以上事实，用更自然、准确的语言重新组织回答，保留关键医学信息。"
    )
    client = openai.OpenAI(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
    )
    response = client.chat.completions.create(
        model=llm_config.get("model_name", "glm-5.1"),
        messages=[
            {"role": "system", "content": "你是医疗知识问答助手，基于知识图谱事实回答问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=llm_config.get("max_tokens", _DEFAULT_LLM_MAX_TOKENS),
        timeout=llm_config.get("timeout", 180.0),
        **openai_extra_kwargs(llm_config),
    )
    return response.choices[0].message.content or template_answer
