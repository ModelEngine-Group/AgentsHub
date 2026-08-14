# 在Nexent中注册ChronicCare MCP

建议在 Nexent 的 “MCP 工具” 页面添加远程服务：

- Tool Server：`http://<server-host>:18088`
- MCP Adapter：`http://<server-host>:18188`
- 首选 MCP endpoint：`http://<server-host>:18188/mcp`
- 可选 SSE endpoint：`http://<server-host>:18188/sse`

步骤：

1. 打开 Nexent 的 MCP 工具页面。
2. 新增远程 MCP 服务。
3. 将 `<server-host>` 替换成部署机器的域名或 IP，优先填写 `http://<server-host>:18188/mcp`。
4. 如果页面要求 SSE 地址，尝试填写 `http://<server-host>:18188/sse`。
5. 刷新工具列表，确认工具范围与`integrations/nexent/chroniccare_tool_manifest.json`一致。
6. 创建或编辑智能体时勾选 ChronicCare 工具。

当前接入方式为远程Streamable HTTP MCP；Nexent需使用支持远程HTTP MCP的配置入口。
