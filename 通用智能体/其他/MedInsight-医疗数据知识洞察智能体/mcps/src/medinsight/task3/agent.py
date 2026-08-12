from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from medinsight.common.agent import AgentRun
from medinsight.task3.service import DEFAULT_OUTPUT_DIR, DEFAULT_TASK, run_analysis, run_query


class GraphInsightAgent:
    """Task3 graph-aware analysis agent.

    The agent reads Task1 processing artifacts and Task2 knowledge graph output,
    then builds a relational analysis layer for graph-aware NL2SQL, evidence
    paths, insight cards, and BI visualization. It does not clean raw data or
    rebuild the knowledge graph.
    """

    def plan(self, task: str = DEFAULT_TASK) -> dict[str, Any]:
        return {
            "task": task,
            "runtime": "graph_schema_rules+nl2sql_templates",
            "steps": [
                {"order": 1, "tool": "task1_artifact_reader", "reason": "只读读取任务一报告、processed documents 和 segments。"},
                {"order": 2, "tool": "medical_graph_loader", "reason": "读取任务二生成并保存的知识图谱。"},
                {"order": 3, "tool": "graph_schema_profiler", "reason": "识别实体类型、关系类型、属性结构和可分析任务。"},
                {"order": 4, "tool": "graph_to_sqlite_builder", "reason": "映射为 task1_documents、task1_segments、medical_triples、evidence_paths 等关系表。"},
                {"order": 5, "tool": "graph_metric_calculator", "reason": "计算分布、中心性、药物关联、知识缺口和数据到知识转化指标。"},
                {"order": 6, "tool": "nl2sql_query_agent", "reason": "把自然语言分析问题映射到可审计 SQL。"},
                {"order": 7, "tool": "sql_verifier", "reason": "检查 SQL 只读、可执行和字段一致性。"},
                {"order": 8, "tool": "evidence_path_analyzer", "reason": "绑定统计结果和图谱证据路径。"},
                {"order": 9, "tool": "bi_visualization_builder", "reason": "生成 BI 图表、图谱子图和洞察报告。"},
            ],
        }

    def run_analysis(
        self,
        graph_path: str | Path | None = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        task: str = DEFAULT_TASK,
        task1_report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        run = AgentRun()
        plan = self.plan(task)
        run.emit("planning", "done", "Generated graph-aware analysis plan.", plan=plan)
        result = run_analysis(
            graph_path=graph_path,
            output_dir=output_dir,
            task=task,
            task1_report_path=task1_report_path,
        )
        run.emit(
            "analysis",
            "done",
            "Generated graph metrics, relational analysis store and visualization.",
            report_path=result["report_path"],
            visualization_path=result["visualization_path"],
            nl2sql_accuracy=result["nl2sql_evaluation"]["accuracy"],
        )
        result["plan"] = plan
        result["events"] = run.event_dicts()
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result

    def ask(
        self,
        question: str,
        graph_path: str | Path | None = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        task1_report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        run = AgentRun()
        plan = self.plan(f"执行图谱分析需求：{question}")
        run.emit("planning", "done", "Generated NL2SQL analysis plan.", plan=plan)
        result = run_query(
            question=question,
            graph_path=graph_path,
            output_dir=output_dir,
            task1_report_path=task1_report_path,
        )
        run.emit(
            "nl2sql_query",
            "done",
            "Mapped user analysis request to SQL and executed it on the graph analysis store.",
            intent=result["intent"],
            sql=result["sql"],
            row_count=len(result["rows"]),
        )
        result["plan"] = plan
        result["events"] = run.event_dicts()
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result
