from __future__ import annotations

import time
from typing import Any, Dict

from medinsight.common.agent import AgentRun
from medinsight.common.nexent_adapter import NexentAdapter
from medinsight.common.operators.datamate_adapter import DataMateAdapter
from medinsight.task1.models.flow_planner import DEFAULT_MODEL_PATH, FlowPlannerModel
from medinsight.task1.operators.text_processing import (
    MedicalRecordNormalizer,
    MedicalTextCleaner,
    MedicalTextLoader,
    MedicalTextSegmenter,
    RawRecordProfiler,
)
from medinsight.task1.operators.table_processing import (
    MedicalTableDocumentBuilder,
    MedicalTableNormalizer,
    MedicalTableProfiler,
)
from medinsight.task1.quality_gate import evaluate_quality_gate
from medinsight.task1.scheduler import build_batch_scheduler


class DataProcessingAgent:
    def __init__(self) -> None:
        self.planner = NexentAdapter()
        self.datamate = DataMateAdapter()
        self.profiler = RawRecordProfiler()
        self.normalizer = MedicalRecordNormalizer()
        self.loader = MedicalTextLoader()
        self.cleaner = MedicalTextCleaner()
        self.segmenter = MedicalTextSegmenter()
        self.table_profiler = MedicalTableProfiler()
        self.table_normalizer = MedicalTableNormalizer()
        self.table_document_builder = MedicalTableDocumentBuilder()
        self.flow_model = FlowPlannerModel.load(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None

    def run(self, records: list[dict[str, Any]], task: str, scheduler_enabled: bool = True) -> Dict[str, Any]:
        started = time.perf_counter()
        run = AgentRun()
        tools = [
            self.profiler.spec.name,
            self.normalizer.spec.name,
            self.loader.spec.name,
            self.cleaner.spec.name,
            self.segmenter.spec.name,
            self.table_profiler.spec.name,
            self.table_normalizer.spec.name,
            self.table_document_builder.spec.name,
        ]
        traces: list[dict[str, Any]] = []

        try:
            datamate_context = self.datamate.discover_capabilities()
            run.emit(
                "datamate_discovery",
                "done" if datamate_context["connected"] else "warning",
                "Discovered DataMate runtime capabilities.",
                connected=datamate_context["connected"],
                base_url=datamate_context["base_url"],
                cleaning_template_count=len(datamate_context["cleaning_templates"]),
                operator_total=datamate_context["operator_catalog"].get("total"),
                fallback_used=datamate_context.get("fallback", {}).get("used"),
                fallback_reason=datamate_context.get("fallback", {}).get("reason"),
                missing_services=datamate_context.get("fallback", {}).get("missing_services", []),
                error=datamate_context.get("error"),
            )

            profiled = self.datamate.execute(self.profiler, {"records": records})
            traces.append(profiled["_trace"])
            run.emit("profiling", "done", "Profiled raw medical records.", **profiled["profile"])

            scheduler = build_batch_scheduler(task, profiled["profile"], scheduler_enabled)
            run.emit(
                "batch_scheduling",
                "done" if scheduler["enabled"] else "warning",
                "Selected data processing execution policy.",
                **scheduler,
            )

            executable_tools = self._select_tools(tools, profiled["profile"])
            plan = self._build_plan(task, executable_tools, profiled["profile"])
            plan["batch_scheduler"] = scheduler
            plan["explainability"] = self._explain_plan(task, plan, profiled["profile"], scheduler, executable_tools)
            run.emit("planning", "done", "Generated data processing plan.", plan=plan)

            if profiled["profile"].get("detected_mode") == "structured":
                table_profiled = self.datamate.execute(self.table_profiler, {"rows": records})
                traces.append(table_profiled["_trace"])
                run.emit("table_profiling", "done", "Profiled structured medical table.", **table_profiled["profile"])

                normalized = self.datamate.execute(self.table_normalizer, {"rows": records})
                traces.append(normalized["_trace"])
                run.emit(
                    "table_normalizing",
                    "done",
                    "Normalized structured table rows.",
                    count=len(normalized["rows"]),
                    anomalies=normalized["anomalies"],
                )

                loaded = self.datamate.execute(self.table_document_builder, {"rows": normalized["rows"]})
                traces.append(loaded["_trace"])
                run.emit("table_document_building", "done", "Converted table rows into Document objects.", count=len(loaded["documents"]))
                normalized_records = normalized["rows"]
                table_profile = table_profiled["profile"]
            else:
                normalized = self.datamate.execute(
                    self.normalizer,
                    {
                        "records": records,
                        "use_optimized_text_ops": scheduler["use_optimized_text_ops"],
                    },
                )
                traces.append(normalized["_trace"])
                run.emit(
                    "normalizing",
                    "done",
                    "Normalized raw records for downstream operators.",
                    count=len(normalized["records"]),
                    anomalies=normalized["anomalies"],
                )

                loaded = self.datamate.execute(self.loader, {"records": normalized["records"]})
                traces.append(loaded["_trace"])
                run.emit("loading", "done", "Loaded raw medical records.", count=len(loaded["documents"]))
                normalized_records = normalized["records"]
                table_profile = None

            optimized_payload = {
                **loaded,
                "parallel_workers": scheduler["parallel_workers"],
                "enable_content_cache": scheduler["enable_content_cache"],
                "use_optimized_text_ops": scheduler["use_optimized_text_ops"],
            }
            cleaned = self.datamate.execute(self.cleaner, optimized_payload)
            traces.append(cleaned["_trace"])
            run.emit(
                "cleaning",
                "done",
                "Cleaned medical text.",
                count=len(cleaned["documents"]),
                optimization=cleaned.get("optimization", {}),
            )

            segmented = self.datamate.execute(
                self.segmenter,
                {
                    **cleaned,
                    "parallel_workers": scheduler["parallel_workers"],
                    "enable_content_cache": scheduler["enable_content_cache"],
                    "use_optimized_text_ops": scheduler["use_optimized_text_ops"],
                },
            )
            traces.append(segmented["_trace"])
            run.emit(
                "segmenting",
                "done",
                "Segmented documents.",
                count=len(segmented["segments"]),
                optimization=segmented.get("optimization", {}),
            )
            total_duration_ms = round((time.perf_counter() - started) * 1000, 3)
            records_per_second = round(len(records) / (total_duration_ms / 1000), 3) if total_duration_ms else 0.0
            quality = {
                "input_records": len(records),
                "normalized_records": len(normalized_records),
                "documents": len(cleaned["documents"]),
                "segments": len(segmented["segments"]),
                "anomaly_count": len(normalized["anomalies"]),
                "warning_count": sum(1 for item in normalized["anomalies"] if item.get("level") == "warning"),
                "error_count": sum(1 for item in normalized["anomalies"] if item.get("level") == "error"),
                "recovered_records": len(
                    {
                        item.get("doc_id")
                        for item in normalized["anomalies"]
                        if item.get("level") == "warning"
                    }
                ),
                "dropped_records": sum(
                    1 for item in normalized["anomalies"] if item.get("action") == "dropped_record"
                ),
                "success_rate": round(len(normalized_records) / len(records), 4) if records else 0.0,
            }
            quality_gate = evaluate_quality_gate(profiled["profile"], quality, normalized["anomalies"])
            run.emit(
                "quality_gate",
                "done" if quality_gate["passed"] else "failed",
                "Evaluated data quality gate.",
                gate_status=quality_gate["status"],
                summary=quality_gate["summary"],
                recommendations=quality_gate["recommendations"],
            )
            report = {
                "task": task,
                "runtime": (
                    "nexent_datamate_connected"
                    if datamate_context.get("connected")
                    else "nexent_datamate_unavailable_with_explicit_local_fallback"
                ),
                "plan": plan,
                "datamate_integration": datamate_context,
                "fallback_policy": {
                    "fallback_used": bool(datamate_context.get("fallback", {}).get("used")),
                    "fallback_reason": datamate_context.get("fallback", {}).get("reason"),
                    "missing_services": datamate_context.get("fallback", {}).get("missing_services", []),
                    "service_checks": datamate_context.get("service_checks", []),
                },
                "input_profile": profiled["profile"],
                "quality": quality,
                "quality_gate": quality_gate,
                "performance": {
                    "total_duration_ms": total_duration_ms,
                    "records_per_second": records_per_second,
                    "batch_scheduler": scheduler,
                    "operator_latency_ms": {
                        trace["operator"]: trace["duration_ms"]
                        for trace in traces
                        if trace.get("status") == "done"
                    },
                    "optimized_operators": [
                        {
                            "operator": "medical_text_cleaner",
                            "optimized_text_ops": scheduler["use_optimized_text_ops"],
                            **cleaned.get("optimization", {}),
                        },
                        {
                            "operator": "medical_text_segmenter",
                            "optimized_text_ops": scheduler["use_optimized_text_ops"],
                            **segmented.get("optimization", {}),
                        },
                    ],
                },
                "table_profile": table_profile,
                "operator_traces": traces,
                "anomalies": normalized["anomalies"],
            }
            return {
                "documents": cleaned["documents"],
                "segments": segmented["segments"],
                "events": run.event_dicts(),
                "plan": plan,
                "report": report,
            }
        except Exception as exc:
            run.emit("data_processing", "failed", "Data processing failed.", error=str(exc))
            raise

    def _build_plan(self, task: str, tools: list[str], profile: dict[str, Any]) -> dict[str, Any]:
        deterministic_plan = self.planner.plan(task, tools)
        if not self.flow_model:
            return deterministic_plan

        sample = {
            "task": self._canonical_flow_task(task, profile),
            "data_mode": profile.get("detected_mode", "hybrid"),
            "optimization_goal": "quality_first",
            "profile": {
                "record_bucket": "small" if profile.get("record_count", 0) < 100 else "large",
                "text_coverage_bucket": self._bucket(profile.get("text_coverage", 0)),
                "structured_coverage_bucket": self._bucket(profile.get("structured_field_coverage", 0)),
                "noise_types": [],
                "missing_fields": [],
            },
        }
        predicted_tools = [tool for tool in self.flow_model.predict(sample) if tool in tools]
        if profile.get("detected_mode") == "structured" and not any(tool.startswith("medical_table_") for tool in predicted_tools):
            predicted_tools = tools
        if not predicted_tools:
            predicted_tools = tools
        scores = self.flow_model.predict_scores(sample)
        plan = self.planner.plan(task, tools)
        plan["runtime"] = "flow_planner_model+deterministic_rules"
        plan["flow_planner_model"] = {
            "model_path": str(DEFAULT_MODEL_PATH),
            "parameter_count_estimate": self.flow_model.parameter_count_estimate(),
            "canonical_task": sample["task"],
            "recommended_tools": predicted_tools,
            "scores": {tool: scores.get(tool, 0.0) for tool in tools},
        }
        return plan

    def _explain_plan(
        self,
        task: str,
        plan: dict[str, Any],
        profile: dict[str, Any],
        scheduler: dict[str, Any],
        tools: list[str],
    ) -> dict[str, Any]:
        intent = plan.get("intent", {})
        scores = (plan.get("flow_planner_model") or {}).get("scores", {})
        canonical_task = (plan.get("flow_planner_model") or {}).get("canonical_task") or self._canonical_flow_task(task, profile)
        evidence = {
            "detected_mode": profile.get("detected_mode"),
            "record_count": profile.get("record_count"),
            "text_coverage": profile.get("text_coverage"),
            "structured_field_coverage": profile.get("structured_field_coverage"),
            "entity_terms_coverage": profile.get("entity_terms_coverage"),
            "avg_text_length": profile.get("avg_text_length"),
        }
        operator_explanations = []
        for step in plan.get("steps", []):
            tool = step.get("tool")
            operator_explanations.append(
                {
                    "tool": tool,
                    "reason": step.get("reason"),
                    "model_score": scores.get(tool),
                    "evidence": self._tool_evidence(tool, profile),
                }
            )
        return {
            "name": "explainable_planner",
            "task_understanding": {
                "original_task": task,
                "canonical_task": canonical_task,
                "active_intents": [key for key, value in intent.items() if value],
            },
            "data_evidence": evidence,
            "operator_selection": operator_explanations,
            "optimization_policy": {
                "strategy": scheduler.get("strategy"),
                "batch_size": scheduler.get("batch_size"),
                "use_optimized_text_ops": scheduler.get("use_optimized_text_ops"),
                "reason": scheduler.get("reason"),
            },
            "explanation_summary": [
                f"识别为 {profile.get('detected_mode')} 数据，记录数 {profile.get('record_count')}。",
                f"任务归一为：{canonical_task}。",
                f"选择 {len(tools)} 个可执行算子，并使用 {scheduler.get('strategy')} 调度策略。",
            ],
        }

    def _tool_evidence(self, tool: str, profile: dict[str, Any]) -> str:
        if tool == "raw_record_profiler":
            return "所有输入先画像，作为后续规划、门控和调度依据。"
        if tool in {"medical_record_normalizer", "medical_table_normalizer"}:
            return f"字段数 {profile.get('field_count', profile.get('column_count', 'unknown'))}，需要统一输入契约。"
        if tool in {"medical_text_loader", "medical_table_document_builder"}:
            return "下游清洗和切分统一以 Document 对象执行。"
        if tool == "medical_text_cleaner":
            return f"文本覆盖率 {profile.get('text_coverage')}，需要归一化空白和标点。"
        if tool == "medical_text_segmenter":
            return "任务要求生成可复用句子片段。"
        if tool == "medical_table_profiler":
            return "检测到结构化表格，需要先分析字段覆盖和列模式。"
        return "该算子由任务意图和数据画像共同触发。"

    def _select_tools(self, tools: list[str], profile: dict[str, Any]) -> list[str]:
        mode = profile.get("detected_mode", "hybrid")
        if mode == "structured":
            selected = [
                "raw_record_profiler",
                "medical_table_profiler",
                "medical_table_normalizer",
                "medical_table_document_builder",
                "medical_text_cleaner",
                "medical_text_segmenter",
            ]
        else:
            selected = [
                "raw_record_profiler",
                "medical_record_normalizer",
                "medical_text_loader",
                "medical_text_cleaner",
                "medical_text_segmenter",
            ]
        available = set(tools)
        return [tool for tool in selected if tool in available]

    def _canonical_flow_task(self, task: str, profile: dict[str, Any]) -> str:
        mode = profile.get("detected_mode", "hybrid")
        if "去重" in task:
            return "去重后生成标准化医学片段"
        if mode == "structured":
            return "处理结构化表格并映射字段"
        if "异常" in task or profile.get("text_coverage", 1) < 0.8:
            return "缺字段数据修复和质量检查"
        if mode == "text":
            return "清洗医疗文本并切分句子"
        return "公开医疗数据 ETL"

    def _bucket(self, value: Any) -> str:
        number = float(value or 0)
        if number >= 0.8:
            return "high"
        if number >= 0.4:
            return "medium"
        return "low"
