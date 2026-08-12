"""
任务三 NL2SQL MCP 工具。
"""
from mcp_server.tools import mcp
from mcp_server.tools.task3_runtime import get_task3_analysis_service

@mcp.tool
def execute_nl2sql(question: str) -> dict:
    """兼容旧工具名，走统一任务三分析链并返回同源证据。"""
    return get_task3_analysis_service().analyze(question)
