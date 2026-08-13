"""三个智能体的构建工厂。"""

from dki.agents.runtime import build_agent
from dki.agents.tools import (
    BuildKgTool,
    ChartTool,
    KgQueryTool,
    ListOperatorsTool,
    Nl2SqlTool,
    PublishTask1ArtifactsTool,
    PreviewDataTool,
    RunPipelineTool,
    SaveReportTool,
    TaskStatusTool,
)


def build_data_processing_agent(model=None, observer=None, max_steps: int = 8):
    """任务一：数据处理智能体。"""
    return build_agent(
        name="data_processing_agent",
        description="理解数据处理需求，自主编排 DataMate 算子流水线并跟踪执行。",
        tools=[ListOperatorsTool(), PreviewDataTool(), RunPipelineTool(), TaskStatusTool(), PublishTask1ArtifactsTool()],
        prompt_name="data_agent",
        model=model,
        observer=observer,
        max_steps=max_steps,
    )


def build_kg_agent(model=None, observer=None, max_steps: int = 8):
    """任务二：知识图谱生成与问答智能体。"""
    return build_agent(
        name="knowledge_graph_agent",
        description="自动编排知识抽取算子构建医疗知识图谱，并基于图谱问答。",
        tools=[
            ListOperatorsTool(), PreviewDataTool(), RunPipelineTool(), TaskStatusTool(),
            BuildKgTool(), KgQueryTool(),
        ],
        prompt_name="kg_agent",
        model=model,
        observer=observer,
        max_steps=max_steps,
    )


def build_insight_agent(model=None, observer=None, max_steps: int = 10):
    """任务三：数据分析与 BI 可视化智能体。"""
    return build_agent(
        name="insight_agent",
        description="图谱感知 + NL2SQL 数据分析 + BI 可视化报告生成。",
        tools=[KgQueryTool(), Nl2SqlTool(), ChartTool(), SaveReportTool()],
        prompt_name="insight_agent",
        model=model,
        observer=observer,
        max_steps=max_steps,
    )
