"""
医学数据智能体 MCP 服务入口。

该模块注册任务一、任务二、任务三工具并启动 FastMCP 服务。
"""

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastmcp import FastMCP

mcp = FastMCP('medical-ai', version='3.4.0')

# 让子模块能 import 到同一个 mcp 实例
import mcp_server.tools
mcp_server.tools.mcp = mcp
mcp_server.mcp = mcp

# ---- 导入工具模块（内部使用 @mcp.tool 注册）----
import mcp_server.tools.task2_extract
import mcp_server.tools.task3_query
import mcp_server.tools.task3_nl2sql
import mcp_server.tools.task2_pipeline
import mcp_server.tools.task1_data

# ---- FastMCP 服务入口 ----
def main():
    host = os.environ.get('MCP_HOST', '0.0.0.0')
    port = int(os.environ.get('MCP_PORT', '8900'))
    mcp.run(transport='streamable-http', host=host, port=port)

if __name__ == '__main__':
    main()
