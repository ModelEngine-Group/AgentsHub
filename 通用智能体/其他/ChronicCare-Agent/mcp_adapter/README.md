# ChronicCare MCP Adapter

`mcp_adapter` 是一个面向 Nexent 的轻量适配层：

- 接收 MCP 风格工具调用
- 转发到已部署的 ChronicCare Tool Server
- 返回适合 Nexent 展示的中文摘要和结构化 JSON

默认环境变量：

```bash
export CHRONICCARE_TOOL_SERVER_URL=http://127.0.0.1:18088
export CHRONICCARE_MCP_HOST=0.0.0.0
export CHRONICCARE_MCP_PORT=18188
export CHRONICCARE_MCP_TRANSPORT=streamable-http
```

直接启动：

```bash
python scripts/run_mcp_adapter.py
```

或者：

```bash
python -m mcp_adapter.server --host 0.0.0.0 --port 18188
```
