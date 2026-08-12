"""Data analysis agent for task 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from operators import (
    BuildMedicalSqliteOperator,
    CentralityAnalysisOperator,
    DiseaseSymptomCountOperator,
    DrugDiseaseCountOperator,
    EvaluateNL2SQLOperator,
    GraphStatisticsOperator,
    NL2AnalysisOperator,
    SaveJsonOperator,
)


class DataAnalysisAgent:
    """Analyze task 1 and task 2 outputs, then generate charts and a report."""

    def __init__(self, outputs_dir: Path | str) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.graph_path = self.outputs_dir / "medical_kg.json"
        self.records_path = self.outputs_dir / "cleaned_records.csv"
        self.sqlite_path = self.outputs_dir / "medical_records.db"
        self.charts_dir = self.outputs_dir / "charts"
        self.report_path = self.outputs_dir / "analysis_report.md"
        self.summary_path = self.outputs_dir / "analysis_summary.json"

    def plan(self, task_description: str) -> list[dict[str, str]]:
        if not task_description.strip():
            raise ValueError("任务描述不能为空")
        return [
            {"step": "load_graph", "operator": "json.load", "goal": "读取医疗知识图谱"},
            {"step": "graph_statistics", "operator": "GraphStatisticsOperator", "goal": "统计节点、边、实体类型和关系类型"},
            {"step": "disease_symptom_count", "operator": "DiseaseSymptomCountOperator", "goal": "统计疾病关联症状数量"},
            {"step": "drug_disease_count", "operator": "DrugDiseaseCountOperator", "goal": "统计药物关联疾病数量"},
            {"step": "centrality_analysis", "operator": "CentralityAnalysisOperator", "goal": "分析图谱中心节点"},
            {"step": "visualize", "operator": "matplotlib", "goal": "生成柱状图"},
            {"step": "nl2sql", "operator": "BuildMedicalSqliteOperator/EvaluateNL2SQLOperator", "goal": "构建 SQLite 并评估 NL2SQL"},
            {"step": "report", "operator": "markdown", "goal": "生成分析报告"},
        ]

    def run(self, task_description: str) -> dict[str, Any]:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        plan = self.plan(task_description)
        graph = self._load_graph()

        selected_task = NL2AnalysisOperator().execute(task_description)
        statistics = GraphStatisticsOperator().execute(graph)
        disease_symptom_counts = DiseaseSymptomCountOperator().execute(graph)
        drug_disease_counts = DrugDiseaseCountOperator().execute(graph)
        centrality = CentralityAnalysisOperator().execute(graph, top_k=10)
        sqlite_result = BuildMedicalSqliteOperator().execute(str(self.records_path), str(self.sqlite_path))
        nl2sql_metrics = EvaluateNL2SQLOperator().execute()

        disease_chart = self._save_bar_chart(
            disease_symptom_counts[:10],
            label_key="disease",
            value_key="symptom_count",
            title="Disease Symptom Count Top 10",
            ylabel="Symptom Count",
            file_name="disease_symptom_count.png",
        )
        drug_chart = self._save_bar_chart(
            drug_disease_counts[:10],
            label_key="drug",
            value_key="disease_count",
            title="Drug Disease Count Top 10",
            ylabel="Disease Count",
            file_name="drug_disease_count.png",
        )

        report = self._build_report(
            statistics=statistics,
            disease_symptom_counts=disease_symptom_counts,
            drug_disease_counts=drug_disease_counts,
            centrality=centrality,
            nl2sql_metrics=nl2sql_metrics,
        )
        self.report_path.write_text(report, encoding="utf-8")

        summary = {
            "task": "task3_data_analysis_agent",
            "task_description": task_description,
            "selected_task": selected_task,
            "status": "success",
            "plan": plan,
            "statistics": statistics,
            "disease_symptom_counts": disease_symptom_counts,
            "drug_disease_counts": drug_disease_counts,
            "centrality": centrality,
            "sqlite_result": sqlite_result,
            "nl2sql_metrics": nl2sql_metrics,
            "outputs": {
                "disease_symptom_chart": disease_chart,
                "drug_disease_chart": drug_chart,
                "sqlite": str(self.sqlite_path),
                "report": str(self.report_path),
                "summary": str(self.summary_path),
            },
        }
        SaveJsonOperator().execute(summary, str(self.summary_path))
        return summary

    def _load_graph(self) -> dict[str, list[dict[str, str]]]:
        if not self.graph_path.exists():
            raise FileNotFoundError(f"医疗知识图谱不存在: {self.graph_path}")
        with open(self.graph_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_bar_chart(
        self,
        rows: list[dict[str, Any]],
        label_key: str,
        value_key: str,
        title: str,
        ylabel: str,
        file_name: str,
    ) -> str:
        path = self.charts_dir / file_name
        # 使用实际的疾病/药物名称作为横坐标标签
        labels = [str(row[label_key]) for row in rows]
        values = [int(row[value_key]) for row in rows]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color="#4C78A8")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)

    def _build_report(
        self,
        statistics: dict[str, Any],
        disease_symptom_counts: list[dict[str, Any]],
        drug_disease_counts: list[dict[str, Any]],
        centrality: list[dict[str, Any]],
        nl2sql_metrics: dict[str, Any],
    ) -> str:
        top_disease = disease_symptom_counts[0] if disease_symptom_counts else {"disease": "无", "symptom_count": 0}
        top_drug = drug_disease_counts[0] if drug_disease_counts else {"drug": "无", "disease_count": 0}
        top_node = centrality[0] if centrality else {"node": "无", "degree_centrality": 0}

        return f"""# 医疗知识图谱分析报告

## 总体统计

- 节点数量：{statistics['node_count']}
- 边数量：{statistics['edge_count']}
- 实体类型分布：{statistics['entity_type_count']}
- 关系类型分布：{statistics['relation_type_count']}

## 关键发现

- 关联症状最多的疾病是：{top_disease['disease']}，症状数量为 {top_disease['symptom_count']}。
- 关联疾病最多的药物是：{top_drug['drug']}，疾病数量为 {top_drug['disease_count']}。
- 中心性最高的节点是：{top_node['node']}，中心性分数为 {top_node['degree_centrality']}。
- NL2SQL 模板评估准确率：{nl2sql_metrics['accuracy']:.0%}（{nl2sql_metrics['correct']}/{nl2sql_metrics['total']}）。

## 图表输出

- `outputs/charts/disease_symptom_count.png`
- `outputs/charts/drug_disease_count.png`

## NL2SQL MVP

- SQLite 数据库：`outputs/medical_records.db`
- 评估样本数：{nl2sql_metrics['total']}
- 正确映射数：{nl2sql_metrics['correct']}
- 准确率：{nl2sql_metrics['accuracy']:.0%}
"""
