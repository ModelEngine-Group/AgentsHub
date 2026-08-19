from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from medinsight.common.utils import ensure_dir, write_json, write_jsonl
from medinsight.task2.agent import MedicalKnowledgeGraphAgent
from medinsight.task2.datasets import (
    TASK2_RAW_DIR,
    dataset_manifest,
    load_chip_cdn_split,
    load_cmeee_split,
    load_cmeie_schema,
    load_cmeie_split,
    load_cmedcausal_split,
    load_icd10_terms,
    load_surgery_lexicon,
)
from medinsight.task2.operators.graph_builder import MedicalKnowledgeGraphBuilder
from medinsight.task2.operators.qa import answer_question, build_qa_examples
from medinsight.task2.operators import MedicalConflictDetector, MedicalEntityExtractor, MedicalRelationExtractor, MedicalTripleRepairer
from medinsight.task2.operators.text_kg_extraction import TEXT_KG_RELATION_SCHEMAS
from medinsight.task2.operators.triple_builder import MedicalTripleBuilder, MedicalTripleValidator
from medinsight.task2.graph_store import generate_graph_html
from medinsight.task2.quality_gate import evaluate_kg_quality_gate


DEFAULT_TASK2_INPUT = TASK2_RAW_DIR
DEFAULT_TASK2_OUTPUT = Path("artifacts/task2/kg_demo")
DEFAULT_TASK2_TEXT_OUTPUT = Path("artifacts/task2/text_kg_demo")
DEFAULT_SHARED_KG_DIR = Path("data/shared/knowledge_graph")
DEFAULT_TASK2_TASK = "基于公开中文医疗实体和关系抽取数据，构建医疗知识图谱并支持图谱问答。"
DEFAULT_TASK2_TEXT_TASK = "从用户输入的医疗文本自动抽取实体和关系，生成可问答的医疗知识图谱。"


def run_task2_kg_build(
    input_dir: Path = DEFAULT_TASK2_INPUT,
    output_dir: Path = DEFAULT_TASK2_OUTPUT,
    shared_output_dir: Path = DEFAULT_SHARED_KG_DIR,
    task: str = DEFAULT_TASK2_TASK,
    cmeee_split: str = "train",
    cmeie_split: str = "train",
    cmeee_limit: int | None = 2000,
    cmeie_limit: int | None = 2000,
    chip_cdn_split: str = "train",
    cmedcausal_split: str = "train",
    chip_cdn_limit: int | None = 0,
    cmedcausal_limit: int | None = 0,
    icd10_limit: int | None = 0,
    surgery_lexicon_limit: int | None = 0,
    enrichment_enabled: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(shared_output_dir)
    cmeee_records = load_cmeee_split(input_dir, cmeee_split, cmeee_limit)
    cmeie_records = load_cmeie_split(input_dir, cmeie_split, cmeie_limit)
    chip_cdn_records = load_chip_cdn_split(input_dir, chip_cdn_split, chip_cdn_limit) if enrichment_enabled else []
    cmedcausal_records = load_cmedcausal_split(input_dir, cmedcausal_split, cmedcausal_limit) if enrichment_enabled else []
    icd10_terms = load_icd10_terms(input_dir, icd10_limit) if enrichment_enabled else []
    surgery_terms = load_surgery_lexicon(input_dir, surgery_lexicon_limit) if enrichment_enabled else []
    cmeie_schema = load_cmeie_schema(input_dir)
    result = MedicalKnowledgeGraphAgent().run(
        cmeee_records=cmeee_records,
        cmeie_records=cmeie_records,
        cmeie_schema=cmeie_schema,
        chip_cdn_records=chip_cdn_records,
        cmedcausal_records=cmedcausal_records,
        icd10_terms=icd10_terms,
        surgery_terms=surgery_terms,
        task=task,
        cmeee_split=cmeee_split,
        cmeie_split=cmeie_split,
        chip_cdn_split=chip_cdn_split,
        cmedcausal_split=cmedcausal_split,
        enrichment_enabled=enrichment_enabled,
    )
    write_json(output_dir / "entities.json", result["entities"])
    write_json(output_dir / "relations.json", result["relations"])
    write_json(output_dir / "enhanced_relations.json", result["enhanced_relations"])
    write_json(output_dir / "triples.json", result["valid_triples"])
    write_json(output_dir / "invalid_triples.json", result["invalid_triples"])
    write_json(output_dir / "graph.json", result["graph"])
    write_json(output_dir / "kg_build_report.json", result["report"])
    write_json(output_dir / "qa_examples.json", result["qa_examples"])
    write_json(output_dir / "events.json", {"knowledge_graph_build": result["events"]})
    write_json(output_dir / "dataset_manifest.json", dataset_manifest(input_dir))

    shared_graph_path = shared_output_dir / "task2_medical_kg.json"
    shared_summary_path = shared_output_dir / "task2_medical_kg_summary.json"
    shared_triples_path = shared_output_dir / "task2_medical_triples.jsonl"
    write_json(shared_graph_path, result["graph"])
    write_json(
        shared_summary_path,
        {
            "task": "task2",
            "graph_path": str(shared_graph_path),
            "triples_path": str(shared_triples_path),
            "source_report": str(output_dir / "kg_build_report.json"),
            "node_count": result["graph"]["metadata"]["node_count"],
            "edge_count": result["graph"]["metadata"]["edge_count"],
            "entity_types": result["graph"]["schema"]["entity_types"],
            "relation_count": result["graph"]["schema"]["predicate_count"],
            "kg_quality_gate": result["report"]["kg_quality_gate"],
            "task3_reuse": {
                "recommended_input": str(shared_graph_path),
                "description": "Task3 can load this graph for graph-aware analytics, NL2SQL grounding, and BI visualization.",
            },
        },
    )
    write_jsonl(shared_triples_path, result["valid_triples"])
    return {
        "status": "ok",
        "report_path": str(output_dir / "kg_build_report.json"),
        "graph_path": str(output_dir / "graph.json"),
        "shared_graph_path": str(shared_graph_path),
        "shared_summary_path": str(shared_summary_path),
        "shared_triples_path": str(shared_triples_path),
        "qa_examples_path": str(output_dir / "qa_examples.json"),
        "quality": result["report"]["quality"],
        "kg_quality_gate": result["report"]["kg_quality_gate"],
        "performance": result["report"]["performance"],
        "plan": result["report"]["plan"],
    }


def run_task2_text_kg_build(
    text: str,
    output_dir: Path = DEFAULT_TASK2_TEXT_OUTPUT,
    shared_output_dir: Path = DEFAULT_SHARED_KG_DIR,
    task: str = DEFAULT_TASK2_TEXT_TASK,
    doc_id: str = "custom-medical-text-000001",
    question: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    ensure_dir(output_dir)
    ensure_dir(shared_output_dir)
    medical_text_records = parse_medical_text_records(text, doc_id)
    normalized_text = "\n".join(str(record.get("text") or "") for record in medical_text_records if record.get("text"))

    entity_extractor = MedicalEntityExtractor()
    relation_extractor = MedicalRelationExtractor()
    triple_builder = MedicalTripleBuilder()
    triple_validator = MedicalTripleValidator()
    triple_repairer = MedicalTripleRepairer()
    conflict_detector = MedicalConflictDetector()
    graph_builder = MedicalKnowledgeGraphBuilder()
    traces: list[dict[str, Any]] = []

    plan = {
        "task_understanding": {
            "canonical_task": "用户医疗文本知识图谱生成",
            "input_mode": "processed_medical_text",
            "goal": task,
        },
        "operator_chain": [
            {"step": 1, "tool": entity_extractor.spec.name, "reason": "从任务一处理后的医疗文本抽取疾病、症状、药物、检查等实体。"},
            {"step": 2, "tool": relation_extractor.spec.name, "reason": "从任务一处理后的医疗文本抽取临床表现、治疗、检查、并发等关系。"},
            {"step": 3, "tool": triple_builder.spec.name, "reason": "将候选关系转为标准 SPO 三元组并去重。"},
            {"step": 4, "tool": triple_validator.spec.name, "reason": "按通用医疗文本图谱 schema 校验三元组。"},
            {"step": 5, "tool": triple_repairer.spec.name, "reason": "对可安全修复的 schema 类型错误进行自动纠正。"},
            {"step": 6, "tool": conflict_detector.spec.name, "reason": "检测重复三元组和实体类型冲突，避免低质量事实进入图谱。"},
            {"step": 7, "tool": graph_builder.spec.name, "reason": "构建节点、边和图谱 schema，供问答和任务三复用。"},
        ],
    }

    entity_result = timed_operator(
        entity_extractor.spec.name,
        lambda: entity_extractor.run({"medical_text_records": medical_text_records, "doc_id": doc_id}),
        traces,
    )
    relation_result = timed_operator(
        relation_extractor.spec.name,
        lambda: relation_extractor.run({"medical_text_records": medical_text_records, "doc_id": doc_id}),
        traces,
    )
    triple_result = timed_operator(
        triple_builder.spec.name,
        lambda: triple_builder.run({"relations": relation_result["relations"]}),
        traces,
    )
    initial_validation_result = timed_operator(
        triple_validator.spec.name,
        lambda: triple_validator.run({"triples": triple_result["triples"], "schema": TEXT_KG_RELATION_SCHEMAS}),
        traces,
    )
    repair_result = timed_operator(
        triple_repairer.spec.name,
        lambda: triple_repairer.run(
            {
                "invalid_triples": initial_validation_result["invalid_triples"],
                "schema": TEXT_KG_RELATION_SCHEMAS,
            }
        ),
        traces,
    )
    repaired_validation_result = timed_operator(
        f"{triple_validator.spec.name}_after_repair",
        lambda: triple_validator.run(
            {
                "triples": repair_result["repaired_triples"],
                "schema": TEXT_KG_RELATION_SCHEMAS,
            }
        ),
        traces,
    )
    valid_triples = initial_validation_result["valid_triples"] + repaired_validation_result["valid_triples"]
    invalid_triples = repair_result["unrepaired_triples"] + repaired_validation_result["invalid_triples"]
    validation_summary = merge_validation_summaries(
        len(triple_result["triples"]),
        valid_triples,
        invalid_triples,
        initial_validation_result["summary"],
        repaired_validation_result["summary"],
    )
    conflict_result = timed_operator(
        conflict_detector.spec.name,
        lambda: conflict_detector.run({"triples": valid_triples}),
        traces,
    )
    graph_result = timed_operator(
        graph_builder.spec.name,
        lambda: graph_builder.run(
            {
                "triples": valid_triples,
                "schema": TEXT_KG_RELATION_SCHEMAS,
            }
        ),
        traces,
    )
    custom_import_id = f"custom:{doc_id}"
    graph_result["graph"].setdefault("metadata", {})
    graph_result["graph"]["metadata"].update(
        {
            "name": "task2_pending_medical_knowledge_graph",
            "source": "task1_processed_medical_text",
            "custom_import_id": custom_import_id,
            "doc_id": doc_id,
            "input_mode": "task1_processed_medical_text",
            "description": "Generated by Task2 generic medical text operators from Task1 processed medical text.",
        }
    )

    qa_examples = build_qa_examples(graph_result["graph"], limit=4)
    quality_gate = evaluate_kg_quality_gate(
        graph_result["graph"],
        valid_triples,
        invalid_triples,
        qa_examples,
    )
    qa_answer = answer_question(graph_result["graph"], question) if question else None
    total_duration_ms = round((time.perf_counter() - started) * 1000, 3)

    write_json(output_dir / "entities.json", entity_result["entities"])
    write_json(output_dir / "relations.json", relation_result["relations"])
    write_json(output_dir / "triples.json", valid_triples)
    write_json(output_dir / "invalid_triples.json", invalid_triples)
    write_json(output_dir / "repaired_triples.json", repair_result["repaired_triples"])
    write_json(output_dir / "conflicts.json", conflict_result["conflicts"])
    write_json(output_dir / "graph.json", graph_result["graph"])
    write_json(output_dir / "qa_examples.json", qa_examples)
    write_json(output_dir / "qa_answer.json", qa_answer or {})
    (output_dir / "input_text.txt").write_text(normalized_text, encoding="utf-8")

    report = {
        "task": task,
        "runtime": "nexent_text_kg_agent_tool_contract",
        "plan": plan,
        "input": {
            "doc_id": doc_id,
            "text_length": len(normalized_text),
            "record_count": len(medical_text_records),
            "official_source_record_count": sum(1 for record in medical_text_records if record.get("source_refs")),
            "source_dataset": "task1_processed_medical_text",
            "upstream_task": "task1_data_processing_agent",
            "note": "The graph is generated from Task1-processed medical text records by generic Task2 operators, not imported from an existing KG.",
        },
        "quality": {
            "entities": len(entity_result["entities"]),
            "relations": len(relation_result["relations"]),
            "unique_triples": len(triple_result["triples"]),
            "valid_triples": len(valid_triples),
            "invalid_triples": len(invalid_triples),
            "repaired_triples": len(repair_result["repaired_triples"]),
            "unrepaired_triples": len(repair_result["unrepaired_triples"]),
            "conflict_count": conflict_result["summary"]["conflict_count"],
            "graph_nodes": graph_result["summary"]["node_count"],
            "graph_edges": graph_result["summary"]["edge_count"],
            "triple_valid_rate": validation_summary["valid_rate"],
        },
        "schema": {
            "entity_types": graph_result["graph"]["schema"]["entity_types"],
            "relation_schemas": TEXT_KG_RELATION_SCHEMAS,
            "relation_count": len(TEXT_KG_RELATION_SCHEMAS),
        },
        "kg_quality_gate": quality_gate,
        "self_verification": {
            "schema_validation": {
                "valid_triples": len(valid_triples),
                "invalid_triples": len(invalid_triples),
                "valid_rate": validation_summary["valid_rate"],
                "invalid_reasons": validation_summary.get("invalid_reasons", {}),
            },
            "repair": {
                "repaired_triples": len(repair_result["repaired_triples"]),
                "unrepaired_triples": len(repair_result["unrepaired_triples"]),
                "repair_rate": repair_result["repair_report"]["repair_rate"],
                "repair_reasons": repair_result["repair_report"].get("repair_reasons", {}),
            },
            "conflicts": {
                "conflict_count": conflict_result["summary"]["conflict_count"],
                "conflict_types": conflict_result["summary"].get("conflict_types", {}),
                "passed": conflict_result["summary"]["passed"],
            },
            "evidence_check": {
                "evidence_coverage": quality_gate["metrics"]["evidence_coverage"],
                "note": "Every accepted graph edge must keep at least one source evidence sentence.",
            },
            "graph_quality": {
                "duplicate_edge_ratio": quality_gate["metrics"]["duplicate_edge_ratio"],
                "isolated_node_ratio": quality_gate["metrics"]["isolated_node_ratio"],
                "disease_relation_completeness": quality_gate["metrics"].get("disease_relation_completeness"),
            },
            "knowledge_gaps": quality_gate.get("knowledge_gaps", []),
        },
        "performance": {
            "total_duration_ms": total_duration_ms,
            "records_per_second": round(len(medical_text_records) / (total_duration_ms / 1000), 3) if total_duration_ms else 0.0,
            "operator_latency_ms": {
                trace["operator"]: trace["duration_ms"]
                for trace in traces
                if trace.get("status") == "done"
            },
            "operator_panel": build_operator_performance_panel(traces, len(medical_text_records)),
        },
        "operator_traces": traces,
    }
    write_json(output_dir / "kg_build_report.json", report)

    shared_graph_path = shared_output_dir / "task2_pending_medical_kg.json"
    shared_summary_path = shared_output_dir / "task2_pending_medical_kg_summary.json"
    shared_triples_path = shared_output_dir / "task2_pending_medical_triples.jsonl"
    visualization_path = output_dir / "graph_visualization.html"
    write_json(shared_graph_path, graph_result["graph"])
    write_jsonl(shared_triples_path, valid_triples)
    visualization = generate_graph_html(
        shared_graph_path,
        visualization_path,
        "本次输入医疗数据生成的知识图谱",
    )
    write_json(
        shared_summary_path,
        {
            "task": "task2_pending_text",
            "graph_path": str(shared_graph_path),
            "triples_path": str(shared_triples_path),
            "visualization_path": str(visualization_path),
            "source_report": str(output_dir / "kg_build_report.json"),
            "node_count": graph_result["graph"]["metadata"]["node_count"],
            "edge_count": graph_result["graph"]["metadata"]["edge_count"],
            "entity_types": graph_result["graph"]["schema"]["entity_types"],
            "relation_count": graph_result["graph"]["schema"]["predicate_count"],
            "kg_quality_gate": quality_gate,
            "task3_reuse": {
                "recommended_input": "data/shared/knowledge_graph/task2_user_medical_kg_library.json",
                "description": "Only user-confirmed graphs are stored in the Task2 library for downstream Task3 reuse.",
            },
        },
    )
    return {
        "status": "ok",
        "report_path": str(output_dir / "kg_build_report.json"),
        "graph_path": str(output_dir / "graph.json"),
        "shared_graph_path": str(shared_graph_path),
        "shared_summary_path": str(shared_summary_path),
        "shared_triples_path": str(shared_triples_path),
        "visualization_path": str(visualization_path),
        "visualization": visualization,
        "qa_answer": qa_answer,
        "quality": report["quality"],
        "kg_quality_gate": quality_gate,
        "performance": report["performance"],
        "plan": plan,
    }


def merge_validation_summaries(
    input_triples: int,
    valid_triples: list[dict[str, Any]],
    invalid_triples: list[dict[str, Any]],
    *summaries: dict[str, Any],
) -> dict[str, Any]:
    invalid_reasons: Counter[str] = Counter()
    for summary in summaries:
        invalid_reasons.update(summary.get("invalid_reasons") or {})
    return {
        "input_triples": input_triples,
        "valid_triples": len(valid_triples),
        "invalid_triples": len(invalid_triples),
        "valid_rate": round(len(valid_triples) / input_triples, 4) if input_triples else 0.0,
        "invalid_reasons": dict(invalid_reasons),
    }


def build_operator_performance_panel(traces: list[dict[str, Any]], record_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        duration_ms = float(trace.get("duration_ms") or 0.0)
        duration_seconds = duration_ms / 1000
        rows.append(
            {
                "operator": trace.get("operator"),
                "status": trace.get("status"),
                "device": "CPU",
                "latency_ms": duration_ms,
                "throughput_records_per_second": round(record_count / duration_seconds, 3) if duration_seconds else 0.0,
                "error": trace.get("error"),
            }
        )
    return rows


def timed_operator(operator: str, func: Any, traces: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = func()
        traces.append(
            {
                "operator": operator,
                "status": "done",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        return result
    except Exception as exc:
        traces.append(
            {
                "operator": operator,
                "status": "failed",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": str(exc),
            }
        )
        raise


def parse_medical_text_records(text: str, fallback_doc_id: str) -> list[dict[str, Any]]:
    value = str(text or "").strip()
    if not value:
        return [{"doc_id": fallback_doc_id, "text": ""}]
    if value[0] not in "[{":
        return [{"doc_id": fallback_doc_id, "text": value}]
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return [{"doc_id": fallback_doc_id, "text": value}]
    records = extract_records_from_payload(payload, fallback_doc_id)
    return records or [{"doc_id": fallback_doc_id, "text": value}]


def extract_records_from_payload(payload: Any, fallback_doc_id: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("records", "documents", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return extract_records_from_payload(value, fallback_doc_id)
        for key in ("text", "content", "medical_text", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return [build_text_record(payload, fallback_doc_id, value)]
        return []
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                item_doc_id = str(item.get("doc_id") or item.get("id") or f"{fallback_doc_id}-{index:04d}")
                item_records = extract_records_from_payload(item, item_doc_id)
                records.extend(item_records)
            elif isinstance(item, str) and item.strip():
                records.append({"doc_id": f"{fallback_doc_id}-{index:04d}", "text": item.strip()})
        return records
    return []


def build_text_record(item: dict[str, Any], fallback_doc_id: str, text: str) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    record = {
        "doc_id": item.get("doc_id") or item.get("id") or fallback_doc_id,
        "text": text.strip(),
    }
    for meta_key in (
        "source_refs",
        "dataset",
        "split",
        "source_file",
        "source_index",
        "disease",
        "disease_topic",
        "title",
    ):
        if meta_key in item:
            record[meta_key] = item[meta_key]
        elif meta_key in metadata:
            record[meta_key] = metadata[meta_key]
    if metadata:
        record["metadata"] = metadata
    return record
