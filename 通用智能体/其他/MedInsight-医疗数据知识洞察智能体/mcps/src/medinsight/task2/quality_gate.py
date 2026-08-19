from __future__ import annotations

from collections import Counter
from typing import Any


CORE_DISEASE_PREDICATES = {
    "临床表现",
    "病因",
    "病理分型",
    "辅助检查",
    "药物治疗",
    "辅助治疗",
    "并发症",
    "就诊科室",
}


def evaluate_kg_quality_gate(
    graph: dict[str, Any],
    valid_triples: list[dict[str, Any]],
    invalid_triples: list[dict[str, Any]],
    qa_examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    qa_examples = qa_examples or []
    edge_count = len(edges)
    node_count = len(nodes)
    total_triples = len(valid_triples) + len(invalid_triples)

    valid_rate = safe_ratio(len(valid_triples), total_triples)
    evidence_coverage = safe_ratio(sum(1 for edge in edges if edge.get("evidence")), edge_count)
    duplicate_edge_ratio = duplicate_ratio(edges)
    isolated_node_ratio = isolated_ratio(nodes, edges)
    predicate_coverage = safe_ratio(len({edge.get("predicate") for edge in edges}), graph.get("schema", {}).get("predicate_count") or 0)
    source_diversity = len(source_dataset_counts(edges))
    coding_edges = sum(1 for edge in edges if edge.get("predicate") in {"ICD编码", "手术编码"})
    qa_answered_ratio = safe_ratio(sum(1 for item in qa_examples if item.get("status") == "answered"), len(qa_examples))
    graph_source = str((graph.get("metadata") or {}).get("source") or "")
    is_generic_text_graph = graph_source == "generic_medical_text"
    disease_completeness = disease_relation_completeness(graph)
    knowledge_gaps = detect_knowledge_gaps(graph)

    rules = [
        numeric_rule("triple_valid_rate", "三元组有效率", valid_rate, pass_min=0.98, warn_min=0.9, weight=22, critical=True),
        numeric_rule("evidence_coverage", "证据覆盖率", evidence_coverage, pass_min=0.95, warn_min=0.8, weight=18, critical=True),
        inverse_rule("duplicate_edge_ratio", "重复边比例", duplicate_edge_ratio, pass_max=0.0, warn_max=0.02, weight=12),
        inverse_rule("isolated_node_ratio", "孤立节点比例", isolated_node_ratio, pass_max=0.15, warn_max=0.35, weight=10),
        numeric_rule("schema_predicate_coverage", "Schema 谓词覆盖率", predicate_coverage, pass_min=0.1, warn_min=0.03, weight=10),
        numeric_rule("qa_answered_ratio", "问答样例命中率", qa_answered_ratio, pass_min=0.4, warn_min=0.1, weight=10),
    ]
    if is_generic_text_graph:
        rules.append(
            numeric_rule(
                "disease_relation_completeness",
                "疾病关系完整度",
                disease_completeness["average_score"],
                pass_min=0.45,
                warn_min=0.25,
                weight=18,
            )
        )
    else:
        rules.extend(
            [
                count_rule("source_diversity", "来源数据多样性", source_diversity, pass_min=3, warn_min=2, weight=10),
                count_rule("coding_edge_count", "医学编码增强边数量", coding_edges, pass_min=1, warn_min=0, weight=8),
            ]
        )
    score = weighted_score(rules)
    failed = [rule for rule in rules if rule["status"] == "failed"]
    warned = [rule for rule in rules if rule["status"] == "warning"]
    critical_failed = [rule for rule in failed if rule.get("critical")]
    status = "failed" if critical_failed or score < 60 else "warning" if failed or warned or score < 85 else "passed"
    return {
        "status": status,
        "passed": status != "failed",
        "score": score,
        "summary": build_summary(status, score, failed, warned),
        "metrics": {
            "node_count": node_count,
            "edge_count": edge_count,
            "valid_triples": len(valid_triples),
            "invalid_triples": len(invalid_triples),
            "triple_valid_rate": valid_rate,
            "evidence_coverage": evidence_coverage,
            "duplicate_edge_ratio": duplicate_edge_ratio,
            "isolated_node_ratio": isolated_node_ratio,
            "schema_predicate_coverage": predicate_coverage,
            "source_dataset_counts": dict(source_dataset_counts(edges)),
            "coding_edge_count": coding_edges,
            "qa_answered_ratio": qa_answered_ratio,
            "disease_relation_completeness": disease_completeness,
            "knowledge_gap_count": len(knowledge_gaps),
        },
        "knowledge_gaps": knowledge_gaps,
        "rules": rules,
        "recommendations": build_recommendations(failed + warned),
    }


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def duplicate_ratio(edges: list[dict[str, Any]]) -> float:
    if not edges:
        return 0.0
    keys = [(edge.get("source"), edge.get("predicate"), edge.get("target")) for edge in edges]
    return round((len(keys) - len(set(keys))) / len(keys), 4)


def isolated_ratio(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> float:
    if not nodes:
        return 0.0
    connected: set[str] = set()
    for edge in edges:
        if edge.get("source"):
            connected.add(str(edge["source"]))
        if edge.get("target"):
            connected.add(str(edge["target"]))
    isolated = sum(1 for node in nodes if str(node.get("node_id")) not in connected)
    return round(isolated / len(nodes), 4)


def source_dataset_counts(edges: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for edge in edges:
        source = (edge.get("properties") or {}).get("source_dataset") or "unknown"
        counts[str(source)] += 1
    return counts


def disease_relation_completeness(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    predicate_by_disease: dict[str, set[str]] = {}
    edge_count_by_disease: Counter = Counter()
    for edge in edges:
        source = node_by_id.get(str(edge.get("source")))
        if not source or source.get("type") != "疾病":
            continue
        disease = str(source.get("name") or "")
        predicate = str(edge.get("predicate") or "")
        if predicate in CORE_DISEASE_PREDICATES:
            predicate_by_disease.setdefault(disease, set()).add(predicate)
        edge_count_by_disease[disease] += 1

    if not predicate_by_disease:
        return {
            "average_score": 0.0,
            "disease_count": 0,
            "complete_disease_count": 0,
            "low_coverage_diseases": [],
        }

    scores: dict[str, float] = {}
    for disease, predicates in predicate_by_disease.items():
        scores[disease] = min(len(predicates) / 3, 1.0)
    average_score = round(sum(scores.values()) / len(scores), 4)
    complete_diseases = [disease for disease, score in scores.items() if score >= 1.0]
    low_coverage = [
        {
            "disease": disease,
            "predicate_count": len(predicate_by_disease[disease]),
            "edge_count": edge_count_by_disease[disease],
            "predicates": sorted(predicate_by_disease[disease]),
        }
        for disease, score in sorted(scores.items(), key=lambda item: (item[1], item[0]))
        if score < 0.67
    ][:10]
    return {
        "average_score": average_score,
        "disease_count": len(predicate_by_disease),
        "complete_disease_count": len(complete_diseases),
        "low_coverage_diseases": low_coverage,
    }


def detect_knowledge_gaps(graph: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    predicate_by_disease: dict[str, set[str]] = {}
    edge_count_by_disease: Counter = Counter()
    for edge in edges:
        source = node_by_id.get(str(edge.get("source")))
        if not source or source.get("type") != "疾病":
            continue
        disease = str(source.get("name") or "")
        predicate = str(edge.get("predicate") or "")
        if predicate in CORE_DISEASE_PREDICATES:
            predicate_by_disease.setdefault(disease, set()).add(predicate)
            edge_count_by_disease[disease] += 1

    gaps: list[dict[str, Any]] = []
    required_groups = {
        "clinical": {"临床表现"},
        "diagnosis": {"辅助检查"},
        "treatment": {"药物治疗", "辅助治疗"},
        "cause_or_type": {"病因", "病理分型"},
    }
    for disease, predicates in predicate_by_disease.items():
        if edge_count_by_disease[disease] < 2:
            continue
        missing_groups = [
            group_name
            for group_name, group_predicates in required_groups.items()
            if not predicates & group_predicates
        ]
        if not missing_groups:
            continue
        gaps.append(
            {
                "disease": disease,
                "existing_predicates": sorted(predicates),
                "missing_groups": missing_groups,
                "suggestion": build_gap_suggestion(disease, missing_groups),
            }
        )
    gaps.sort(key=lambda item: (-len(item["missing_groups"]), item["disease"]))
    return gaps[:limit]


def build_gap_suggestion(disease: str, missing_groups: list[str]) -> str:
    group_text = {
        "clinical": "临床表现",
        "diagnosis": "检查/诊断",
        "treatment": "治疗/管理",
        "cause_or_type": "病因/分型",
    }
    missing = "、".join(group_text.get(item, item) for item in missing_groups)
    return f"{disease} 当前缺少 {missing} 关系，建议补充包含这些信息的医疗文本后重新生成图谱。"


def numeric_rule(
    rule_id: str,
    name: str,
    value: float,
    pass_min: float,
    warn_min: float,
    weight: int,
    critical: bool = False,
) -> dict[str, Any]:
    status = "passed" if value >= pass_min else "warning" if value >= warn_min else "failed"
    return make_rule(rule_id, name, value, f">={pass_min}", status, weight, critical)


def inverse_rule(
    rule_id: str,
    name: str,
    value: float,
    pass_max: float,
    warn_max: float,
    weight: int,
    critical: bool = False,
) -> dict[str, Any]:
    status = "passed" if value <= pass_max else "warning" if value <= warn_max else "failed"
    return make_rule(rule_id, name, value, f"<={pass_max}", status, weight, critical)


def count_rule(
    rule_id: str,
    name: str,
    value: int,
    pass_min: int,
    warn_min: int,
    weight: int,
    critical: bool = False,
) -> dict[str, Any]:
    status = "passed" if value >= pass_min else "warning" if value >= warn_min else "failed"
    return make_rule(rule_id, name, value, f">={pass_min}", status, weight, critical)


def make_rule(
    rule_id: str,
    name: str,
    value: int | float,
    threshold: str,
    status: str,
    weight: int,
    critical: bool,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "name": name,
        "value": value,
        "threshold": threshold,
        "status": status,
        "weight": weight,
        "critical": critical,
    }


def weighted_score(rules: list[dict[str, Any]]) -> float:
    total_weight = sum(int(rule["weight"]) for rule in rules)
    if not total_weight:
        return 0.0
    status_score = {"passed": 1.0, "warning": 0.6, "failed": 0.0}
    score = sum(status_score[rule["status"]] * int(rule["weight"]) for rule in rules)
    return round(score * 100 / total_weight, 2)


def build_summary(status: str, score: float, failed: list[dict[str, Any]], warned: list[dict[str, Any]]) -> str:
    if status == "passed":
        return f"KG quality gate passed with score {score}."
    if status == "warning":
        return f"KG quality gate produced warnings with score {score}; review {len(warned) + len(failed)} rule(s)."
    return f"KG quality gate failed with score {score}; critical graph quality rules need attention."


def build_recommendations(rules: list[dict[str, Any]]) -> list[str]:
    mapping = {
        "triple_valid_rate": "检查关系 schema 映射和三元组生成逻辑，降低无效三元组比例。",
        "evidence_coverage": "保留每条边的原始证据文本和 source doc，避免无证据事实入图。",
        "duplicate_edge_ratio": "在图谱构建阶段继续按 source-predicate-target 去重。",
        "isolated_node_ratio": "减少未连接实体入图，或增加实体归一化与关系补全。",
        "schema_predicate_coverage": "扩大 CMeIE split 或启用更多增强词典，提高关系类型覆盖。",
        "source_diversity": "启用 CMeEE、CMeIE、CHIP-CDN、CMedCausal 等多源数据增强。",
        "coding_edge_count": "启用 ICD-10 和手术词表增强，补充医学编码关系。",
        "qa_answered_ratio": "扩充图谱关系或优化问题意图到谓词的映射规则。",
        "disease_relation_completeness": "优先补充每个核心疾病的症状、检查、治疗、病因/分型等多关系证据，减少只含单一关系的疾病节点。",
    }
    recommendations: list[str] = []
    for rule in rules:
        item = mapping.get(str(rule.get("id")))
        if item and item not in recommendations:
            recommendations.append(item)
    return recommendations
