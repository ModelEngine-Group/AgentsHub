from __future__ import annotations

from medgraph_agent.core.models import WorkflowPlan, WorkflowStep, stable_id, utc_now


class Planner:
    """Deterministic offline planner for data-to-knowledge-to-insight tasks."""

    def plan(self, task: str) -> WorkflowPlan:
        task_text = task.strip() or "构建医疗数据处理、知识图谱和分析闭环"
        steps = [
            WorkflowStep(
                name="ingest",
                operator="data_ingestion",
                intent="接入 JSONL/CSV/TXT 医疗数据，保留来源与元数据。",
            ),
            WorkflowStep(
                name="clean",
                operator="text_cleaning",
                intent="清洗空白、标点和重复格式，产出可复用的结构化记录。",
                depends_on=["ingest"],
            ),
            WorkflowStep(
                name="extract_entities",
                operator="entity_recognition",
                intent="识别疾病、症状、药物、检查、治疗、科室、风险因素。",
                depends_on=["clean"],
            ),
            WorkflowStep(
                name="extract_relations",
                operator="relation_extraction",
                intent="抽取诊断、治疗、症状、并发、禁忌和科室归属关系。",
                depends_on=["extract_entities"],
            ),
            WorkflowStep(
                name="validate_triples",
                operator="triple_validation",
                intent="按图谱 schema 校验三元组类型和置信度。",
                depends_on=["extract_relations"],
            ),
        ]
        return WorkflowPlan(
            id=stable_id("plan", task_text, utc_now()),
            task=task_text,
            created_at=utc_now(),
            steps=steps,
        )
