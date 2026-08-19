"""FastAPI service for the data-knowledge-insight demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agents import DataAnalysisAgent, DataProcessingAgent, KGQAAgent, KnowledgeGraphAgent


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

app = FastAPI(
    title="Data Knowledge Insight Demo API",
    description="基于数据处理、知识图谱和数据分析智能体的 Demo API。",
    version="0.1.0",
)


class TaskRequest(BaseModel):
    """Generic task request with an optional natural-language description."""

    task_description: str = Field(default="", description="自然语言任务描述")


class QuestionRequest(BaseModel):
    """Knowledge graph question request."""

    question: str = Field(..., description="图谱问答问题，例如：肺炎有哪些症状？")


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok", "service": "data-knowledge-insight-demo"}


@app.post("/task1/run")
def run_task1(request: TaskRequest) -> dict[str, Any]:
    """Run task 1 data processing agent."""
    description = request.task_description or "请清洗医疗CSV数据，抽取核心字段，并处理医疗文本"
    return DataProcessingAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR).run(description)


@app.post("/task2/build-kg")
def build_knowledge_graph(request: TaskRequest) -> dict[str, Any]:
    """Run task 2 knowledge graph construction agent."""
    description = request.task_description or "请从医疗文本中抽取实体关系，并生成医疗知识图谱"
    return KnowledgeGraphAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR).run(description)


@app.post("/task2/ask")
def ask_knowledge_graph(request: QuestionRequest) -> dict[str, Any]:
    """Answer a question from the generated medical knowledge graph."""
    return KGQAAgent(outputs_dir=OUTPUTS_DIR).run(request.question)


@app.post("/task3/analyze")
def analyze_data(request: TaskRequest) -> dict[str, Any]:
    """Run task 3 data analysis agent."""
    description = request.task_description or "请分析医疗知识图谱，生成统计结果、图表和报告"
    return DataAnalysisAgent(outputs_dir=OUTPUTS_DIR).run(description)
