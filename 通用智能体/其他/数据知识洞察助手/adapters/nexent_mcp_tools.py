"""Nexent-compatible MCP tool wrappers for the demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import DataAnalysisAgent, DataProcessingAgent, KGQAAgent, KnowledgeGraphAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

try:
    from fastmcp import FastMCP  # pyright: ignore[reportMissingImports]
except ImportError:
    FastMCP = None


def process_medical_data_tool(
    task_description: str,
    data_dir: str | None = None,
    outputs_dir: str | None = None,
) -> dict[str, Any]:
    """Run task 1 medical data processing and return a JSON-serializable result."""
    # Always use local paths, ignore any paths from Nexent (which may be S3 paths)
    agent = DataProcessingAgent(
        data_dir=DEFAULT_DATA_DIR,
        outputs_dir=DEFAULT_OUTPUTS_DIR,
    )
    return agent.run(task_description)


def build_medical_knowledge_graph_tool(
    task_description: str,
    data_dir: str | None = None,
    outputs_dir: str | None = None,
) -> dict[str, Any]:
    """Run task 2 medical knowledge graph construction."""
    # Always use local paths, ignore any paths from Nexent (which may be S3 paths)
    agent = KnowledgeGraphAgent(
        data_dir=DEFAULT_DATA_DIR,
        outputs_dir=DEFAULT_OUTPUTS_DIR,
    )
    return agent.run(task_description)


def ask_medical_kg_tool(
    question: str,
    outputs_dir: str | None = None,
) -> dict[str, Any]:
    """Answer a medical question from the generated knowledge graph."""
    # Always use local paths, ignore any paths from Nexent (which may be S3 paths)
    graph_path = DEFAULT_OUTPUTS_DIR / "medical_kg.json"
    if not graph_path.exists():
        return _missing_prerequisite(
            tool="ask_medical_kg",
            required=[graph_path],
            message="请先运行 build_medical_knowledge_graph 生成医疗知识图谱。",
        )

    agent = KGQAAgent(outputs_dir=DEFAULT_OUTPUTS_DIR)
    return agent.run(question)


def analyze_medical_data_tool(
    task_description: str,
    outputs_dir: str | None = None,
) -> dict[str, Any]:
    """Analyze processed medical data and the generated knowledge graph."""
    # Always use local paths, ignore any paths from Nexent (which may be S3 paths)
    required_paths = [
        DEFAULT_OUTPUTS_DIR / "medical_kg.json",
        DEFAULT_OUTPUTS_DIR / "cleaned_records.csv",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        return _missing_prerequisite(
            tool="analyze_medical_data",
            required=missing_paths,
            message="请先运行 process_medical_data 和 build_medical_knowledge_graph，再执行医疗数据分析。",
        )

    agent = DataAnalysisAgent(outputs_dir=DEFAULT_OUTPUTS_DIR)
    return agent.run(task_description)


def _missing_prerequisite(tool: str, required: list[Path], message: str) -> dict[str, Any]:
    """Return a JSON-serializable prerequisite error for Nexent tool calls."""
    return {
        "status": "missing_prerequisite",
        "tool": tool,
        "required": [str(path) for path in required],
        "message": message,
    }


mcp = FastMCP("data-knowledge-insight") if FastMCP is not None else None
if mcp is not None:
    mcp.tool(
        name="process_medical_data",
        description="清洗医疗 CSV 数据并处理医疗文本，返回任务一执行摘要。",
    )(process_medical_data_tool)
    mcp.tool(
        name="build_medical_knowledge_graph",
        description="从医疗文本中抽取实体关系并构建医疗知识图谱。",
    )(build_medical_knowledge_graph_tool)
    mcp.tool(
        name="ask_medical_kg",
        description="基于已生成的医疗知识图谱回答疾病症状、用药等问题。",
    )(ask_medical_kg_tool)
    mcp.tool(
        name="analyze_medical_data",
        description="分析清洗后的医疗数据和知识图谱，生成统计结果、图表和报告。",
    )(analyze_medical_data_tool)


if __name__ == "__main__":
    if mcp is None:
        raise SystemExit("FastMCP is not installed. Install it before starting the MCP server.")
    mcp.run(transport="sse", port=8001)
