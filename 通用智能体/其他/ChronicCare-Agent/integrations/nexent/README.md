# Nexent集成材料

本目录保存ChronicCare-Agent接入Nexent所需的正式材料。

## 文件说明

- `chroniccare_agent_prompt.md`：单智能体系统提示词与工具路由约束。
- `chroniccare_mcp_config.example.json`：Streamable HTTP MCP连接示例。
- `chroniccare_tool_manifest.json`：当前Nexent Agent默认绑定工具清单。
- `demo_questions.md`：前端功能验证问题集。
- `setup_guide.md`：Nexent接入步骤。

## 工具范围

后端MCP服务当前公开38个工具；默认清单选择其中33个绑定给Nexent Agent，覆盖健康检查、数据规模、DataMate执行与状态、NPU增强、知识图谱、疾病与风险分析、未来随访、Open SQL、动态DAG、报告和调用追踪。5个历史兼容入口保留后端实现但不默认绑定，2个调试工具继续保留绑定。实际默认范围以`chroniccare_tool_manifest.json`为准。
