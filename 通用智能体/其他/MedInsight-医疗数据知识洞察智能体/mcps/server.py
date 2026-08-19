from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from medinsight_tools import (
    analyze_medical_graph,
    ask_medical_knowledge_graph,
    build_medical_knowledge_graph,
    process_medical_data,
    query_medical_graph_analysis,
    run_data_knowledge_insight_pipeline,
)


mcp = FastMCP(
    "MedInsight Agent MCP",
    host=os.getenv("MEDINSIGHT_MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MEDINSIGHT_MCP_PORT", "18100")),
)

mcp.tool()(process_medical_data)
mcp.tool()(build_medical_knowledge_graph)
mcp.tool()(ask_medical_knowledge_graph)
mcp.tool()(analyze_medical_graph)
mcp.tool()(query_medical_graph_analysis)
mcp.tool()(run_data_knowledge_insight_pipeline)


@mcp.custom_route("/", methods=["GET"])
async def service_status(_: Request) -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MedInsight MCP</title>
  <style>
    body { margin: 0; font: 16px/1.6 system-ui, sans-serif; color: #202124; background: #f7f8fa; }
    main { width: min(720px, calc(100% - 40px)); margin: 72px auto; }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    p { margin: 8px 0; color: #5f6368; }
    code { padding: 2px 6px; border: 1px solid #dadce0; border-radius: 4px; background: #fff; }
    .status { color: #137333; font-weight: 700; }
  </style>
</head>
<body>
  <main>
    <h1>MedInsight Agent MCP</h1>
    <p class="status">服务运行正常</p>
    <p>已注册 6 个“数据-知识-洞察”工具。</p>
    <p>健康检查：<code>/health</code></p>
    <p>Nexent SSE 地址：<code>/sse</code>。该地址是持续连接，不是普通网页。</p>
  </main>
</body>
</html>"""
    )


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "MedInsight Agent MCP",
            "transport": "sse",
            "sse_endpoint": "/sse",
            "tool_count": 6,
        }
    )


if __name__ == "__main__":
    transport = os.getenv("MEDINSIGHT_MCP_TRANSPORT", "sse")
    mcp.run(transport=transport)
