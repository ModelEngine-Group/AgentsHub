from __future__ import annotations

from collections import Counter
import time
from typing import Any

from medinsight.common.agent import AgentRun
from medinsight.task2.operators import (
    KnowledgeGraphQAOperator,
    MedicalConflictDetector,
    MedicalEntityExtractor,
    MedicalGraphEnricher,
    MedicalKnowledgeGraphBuilder,
    MedicalRelationExtractor,
    MedicalTripleBuilder,
    MedicalTripleRepairer,
    MedicalTripleValidator,
)
from medinsight.task2.operators.qa import build_qa_examples
from medinsight.task2.quality_gate import evaluate_kg_quality_gate


class MedicalKnowledgeGraphAgent:
    def __init__(self) -> None:
        self.entity_extractor = MedicalEntityExtractor()
        self.relation_extractor = MedicalRelationExtractor()
        self.graph_enricher = MedicalGraphEnricher()
        self.triple_builder = MedicalTripleBuilder()
        self.triple_validator = MedicalTripleValidator()
        self.triple_repairer = MedicalTripleRepairer()
        self.conflict_detector = MedicalConflictDetector()
        self.graph_builder = MedicalKnowledgeGraphBuilder()
        self.qa_agent = KnowledgeGraphQAOperator()

    def run(
        self,
        cmeee_records: list[dict[str, Any]],
        cmeie_records: list[dict[str, Any]],
        cmeie_schema: list[dict[str, Any]],
        task: str,
        chip_cdn_records: list[dict[str, Any]] | None = None,
        cmedcausal_records: list[dict[str, Any]] | None = None,
        icd10_terms: list[dict[str, Any]] | None = None,
        surgery_terms: list[dict[str, Any]] | None = None,
        cmeee_split: str = "train",
        cmeie_split: str = "train",
        chip_cdn_split: str = "train",
        cmedcausal_split: str = "train",
        enrichment_enabled: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        run = AgentRun()
        traces: list[dict[str, Any]] = []
        chip_cdn_records = chip_cdn_records or []
        cmedcausal_records = cmedcausal_records or []
        icd10_terms = icd10_terms or []
        surgery_terms = surgery_terms or []
        plan = self._build_plan(
            task,
            cmeee_records,
            cmeie_records,
            cmeie_schema,
            chip_cdn_records,
            cmedcausal_records,
            icd10_terms,
            surgery_terms,
            enrichment_enabled,
        )
        run.emit("planning", "done", "Generated task2 knowledge graph build plan.", plan=plan)

        entity_result = self._timed(
            self.entity_extractor.spec.name,
            lambda: self.entity_extractor.run(
                {
                    "cmeee_records": cmeee_records,
                    "cmeie_records": cmeie_records,
                    "cmeee_split": cmeee_split,
                    "cmeie_split": cmeie_split,
                }
            ),
            traces,
        )
        run.emit("entity_extraction", "done", "Extracted medical entities.", **entity_result["summary"])

        relation_result = self._timed(
            self.relation_extractor.spec.name,
            lambda: self.relation_extractor.run({"cmeie_records": cmeie_records, "cmeie_split": cmeie_split}),
            traces,
        )
        run.emit("relation_extraction", "done", "Extracted medical relations.", **relation_result["summary"])

        enrichment_result = {"relations": [], "schemas": [], "summary": {"total_enhanced_relations": 0}}
        if enrichment_enabled:
            enrichment_result = self._timed(
                self.graph_enricher.spec.name,
                lambda: self.graph_enricher.run(
                    {
                        "chip_cdn_records": chip_cdn_records,
                        "cmedcausal_records": cmedcausal_records,
                        "icd10_terms": icd10_terms,
                        "surgery_terms": surgery_terms,
                        "chip_cdn_split": chip_cdn_split,
                        "cmedcausal_split": cmedcausal_split,
                    }
                ),
                traces,
            )
            run.emit("graph_enrichment", "done", "Enhanced KG relations with normalization and causal data.", **enrichment_result["summary"])

        all_relations = relation_result["relations"] + enrichment_result["relations"]
        combined_schema = cmeie_schema + enrichment_result.get("schemas", [])
        triple_result = self._timed(
            self.triple_builder.spec.name,
            lambda: self.triple_builder.run({"relations": all_relations}),
            traces,
        )
        run.emit("triple_building", "done", "Built canonical triples.", **triple_result["summary"])

        validation_result = self._timed(
            self.triple_validator.spec.name,
            lambda: self.triple_validator.run(
                {
                    "triples": triple_result["triples"],
                    "schema": cmeie_schema,
                    "extra_schema": enrichment_result.get("schemas", []),
                }
            ),
            traces,
        )
        run.emit("triple_validation", "done", "Validated triples against CMeIE schema.", **validation_result["summary"])

        repair_result = self._timed(
            self.triple_repairer.spec.name,
            lambda: self.triple_repairer.run(
                {
                    "invalid_triples": validation_result["invalid_triples"],
                    "schema": cmeie_schema,
                    "extra_schema": enrichment_result.get("schemas", []),
                }
            ),
            traces,
        )
        run.emit("triple_repair", "done", "Repaired safe schema type mismatches.", **repair_result["repair_report"])

        repaired_validation_result = self._timed(
            f"{self.triple_validator.spec.name}_after_repair",
            lambda: self.triple_validator.run(
                {
                    "triples": repair_result["repaired_triples"],
                    "schema": cmeie_schema,
                    "extra_schema": enrichment_result.get("schemas", []),
                }
            ),
            traces,
        )
        valid_triples = validation_result["valid_triples"] + repaired_validation_result["valid_triples"]
        invalid_triples = repair_result["unrepaired_triples"] + repaired_validation_result["invalid_triples"]
        validation_summary = self._merge_validation_summaries(
            len(triple_result["triples"]),
            valid_triples,
            invalid_triples,
            validation_result["summary"],
            repaired_validation_result["summary"],
        )

        conflict_result = self._timed(
            self.conflict_detector.spec.name,
            lambda: self.conflict_detector.run({"triples": valid_triples}),
            traces,
        )
        run.emit("conflict_detection", "done", "Detected duplicate and schema-type conflicts.", **conflict_result["summary"])

        graph_result = self._timed(
            self.graph_builder.spec.name,
            lambda: self.graph_builder.run({"triples": valid_triples, "schema": combined_schema}),
            traces,
        )
        run.emit("graph_building", "done", "Built medical knowledge graph.", **graph_result["summary"])

        qa_examples = build_qa_examples(graph_result["graph"])
        run.emit("kg_qa", "done", "Generated graph QA examples.", question_count=len(qa_examples))
        quality_gate = evaluate_kg_quality_gate(
            graph_result["graph"],
            valid_triples,
            invalid_triples,
            qa_examples,
        )
        run.emit(
            "kg_quality_gate",
            "done" if quality_gate["passed"] else "failed",
            "Evaluated medical knowledge graph quality gate.",
            gate_status=quality_gate["status"],
            score=quality_gate["score"],
            summary=quality_gate["summary"],
            recommendations=quality_gate["recommendations"],
        )

        total_duration_ms = round((time.perf_counter() - started) * 1000, 3)
        report = {
            "task": task,
            "runtime": "nexent_kg_agent_tool_contract",
            "plan": plan,
            "input": {
                "cmeee_records": len(cmeee_records),
                "cmeie_records": len(cmeie_records),
                "chip_cdn_records": len(chip_cdn_records),
                "cmedcausal_records": len(cmedcausal_records),
                "icd10_terms": len(icd10_terms),
                "surgery_terms": len(surgery_terms),
                "cmeie_schema_relations": len(cmeie_schema),
                "cmeee_split": cmeee_split,
                "cmeie_split": cmeie_split,
                "chip_cdn_split": chip_cdn_split,
                "cmedcausal_split": cmedcausal_split,
                "enrichment_enabled": enrichment_enabled,
            },
            "quality": {
                "entities": len(entity_result["entities"]),
                "relations": len(relation_result["relations"]),
                "enhanced_relations": len(enrichment_result["relations"]),
                "enhanced_from_chip_cdn": enrichment_result["summary"].get("from_chip_cdn", 0),
                "enhanced_from_icd10": enrichment_result["summary"].get("from_icd10", 0),
                "enhanced_from_surgery_lexicon": enrichment_result["summary"].get("from_surgery_lexicon", 0),
                "enhanced_from_cmedcausal": enrichment_result["summary"].get("from_cmedcausal", 0),
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
                "relation_count": len(combined_schema),
                "relation_schemas": combined_schema,
                "cmeie_relation_count": len(cmeie_schema),
                "enhancement_relation_schemas": enrichment_result.get("schemas", []),
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
                    "note": "Accepted edges keep source evidence text for evidence-grounded KGQA.",
                },
                "knowledge_gaps": quality_gate.get("knowledge_gaps", []),
            },
            "performance": {
                "total_duration_ms": total_duration_ms,
                "records_per_second": round(
                    (
                        len(cmeee_records)
                        + len(cmeie_records)
                        + len(chip_cdn_records)
                        + len(cmedcausal_records)
                        + len(icd10_terms)
                        + len(surgery_terms)
                    )
                    / (total_duration_ms / 1000),
                    3,
                )
                if total_duration_ms
                else 0.0,
                "operator_latency_ms": {
                    trace["operator"]: trace["duration_ms"]
                    for trace in traces
                    if trace.get("status") == "done"
                },
                "operator_panel": self._operator_performance_panel(
                    traces,
                    len(cmeee_records)
                    + len(cmeie_records)
                    + len(chip_cdn_records)
                    + len(cmedcausal_records)
                    + len(icd10_terms)
                    + len(surgery_terms),
                ),
            },
            "operator_traces": traces,
        }
        return {
            "entities": entity_result["entities"],
            "relations": relation_result["relations"],
            "enhanced_relations": enrichment_result["relations"],
            "triples": triple_result["triples"],
            "valid_triples": valid_triples,
            "invalid_triples": invalid_triples,
            "graph": graph_result["graph"],
            "qa_examples": qa_examples,
            "events": run.event_dicts(),
            "report": report,
        }

    def _build_plan(
        self,
        task: str,
        cmeee_records: list[dict[str, Any]],
        cmeie_records: list[dict[str, Any]],
        cmeie_schema: list[dict[str, Any]],
        chip_cdn_records: list[dict[str, Any]],
        cmedcausal_records: list[dict[str, Any]],
        icd10_terms: list[dict[str, Any]],
        surgery_terms: list[dict[str, Any]],
        enrichment_enabled: bool,
    ) -> dict[str, Any]:
        operator_chain = [
            {
                "step": 1,
                "tool": self.entity_extractor.spec.name,
                "reason": "从 CMeEE 标注和 CMeIE SPO 锚点抽取疾病、症状、药物、检查等实体。",
            },
            {
                "step": 2,
                "tool": self.relation_extractor.spec.name,
                "reason": "从 CMeIE SPO 标注抽取医学关系。",
            },
        ]
        next_step = 3
        if enrichment_enabled:
            operator_chain.append(
                {
                    "step": next_step,
                    "tool": self.graph_enricher.spec.name,
                    "reason": "融合 CHIP-CDN 术语标准化、ICD-10 编码、手术词表和 CMedCausal 因果关系，提升图谱问答覆盖面。",
                }
            )
            next_step += 1
        operator_chain.extend(
            [
                {
                    "step": next_step,
                    "tool": self.triple_builder.spec.name,
                    "reason": "将关系转成标准 subject-predicate-object 三元组并去重。",
                },
                {
                    "step": next_step + 1,
                    "tool": self.triple_validator.spec.name,
                    "reason": "使用 CMeIE 53 类 schema 和增强 schema 校验实体类型、关系类型和三元组一致性。",
                },
                {
                    "step": next_step + 2,
                    "tool": self.triple_repairer.spec.name,
                    "reason": "对可安全修复的 schema 类型错误进行自动纠正，并保留修复原因。",
                },
                {
                    "step": next_step + 3,
                    "tool": self.conflict_detector.spec.name,
                    "reason": "检测重复三元组和实体类型冲突，过滤低可信事实。",
                },
                {
                    "step": next_step + 4,
                    "tool": self.graph_builder.spec.name,
                    "reason": "生成任务三可复用的节点、边和图谱 schema。",
                },
                {
                    "step": next_step + 5,
                    "tool": self.qa_agent.spec.name,
                    "reason": "基于图谱执行可解释问答。",
                },
            ]
        )
        return {
            "task_understanding": {
                "canonical_task": "医疗知识图谱生成与问答",
                "input_mode": "public_medical_text_with_entity_relation_annotations",
                "goal": task,
            },
            "operator_chain": operator_chain,
            "input_profile": {
                "cmeee_records": len(cmeee_records),
                "cmeie_records": len(cmeie_records),
                "chip_cdn_records": len(chip_cdn_records),
                "cmedcausal_records": len(cmedcausal_records),
                "icd10_terms": len(icd10_terms),
                "surgery_terms": len(surgery_terms),
                "schema_relations": len(cmeie_schema),
                "enrichment_enabled": enrichment_enabled,
            },
            "nexent_role": "The same chain is exposed as task2 tools for Nexent Outer API registration.",
        }

    def _merge_validation_summaries(
        self,
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

    def _operator_performance_panel(self, traces: list[dict[str, Any]], record_count: int) -> list[dict[str, Any]]:
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

    def _timed(self, operator: str, func: Any, traces: list[dict[str, Any]]) -> dict[str, Any]:
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
