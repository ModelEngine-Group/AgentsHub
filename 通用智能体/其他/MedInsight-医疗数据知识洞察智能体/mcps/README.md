# MedInsight MCP 服务

该服务将 MedInsight 的三条能力链封装为 Nexent 可调用的 MCP 工具：

| 工具 | 作用 |
|---|---|
| `process_medical_data` | 医疗 JSON 数据画像、规范化、清洗、切分和质量门控 |
| `build_medical_knowledge_graph` | 实体/关系抽取、三元组校验、修复、冲突检测和图谱构建 |
| `ask_medical_knowledge_graph` | 基于已生成图谱和证据路径的问答 |
| `analyze_medical_graph` | 图谱 Schema 感知、指标计算、NL2SQL 评测和 BI 看板 |
| `query_medical_graph_analysis` | 自然语言分析请求转换为可审计的只读 SQL |
| `run_data_knowledge_insight_pipeline` | 端到端执行“数据—知识—洞察”闭环 |

启动后默认 SSE 地址为 `http://127.0.0.1:18100/sse`。
