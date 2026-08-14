# Nexent接入步骤

1. 启动`chroniccare-runtime`，确认Tool Server和MCP Adapter健康检查通过。
2. 依据`chroniccare_mcp_config.example.json`注册Streamable HTTP MCP Endpoint。
3. 加载`chroniccare_tool_manifest.json`中的33个默认绑定工具，并配置`chroniccare_agent_prompt.md`。
4. 在Nexent中验证健康检查、数据规模、图谱摘要、分析查询和调用追踪。
