# Nexent 智能体配置

三个子目录分别对应一个可导入 Nexent 的智能体配置：

| 目录 | 智能体 | 主要能力 |
| --- | --- | --- |
| `数据处理智能体/` | 数据处理智能体 | 医疗资料注册、格式整理和质量检查 |
| `知识图谱智能体/` | 知识图谱智能体 | 医学实体关系抽取和三元组入库 |
| `数据分析智能体/` | 数据分析智能体 | 疾病查询、统计分析和图表结果交付 |

每个目录中的 `agent.json` 均按 Nexent 导出格式整理，文件结构包含 Nexent 导入所需的 `agent_id`、`agent_info` 和 `mcp_info`。导入其他环境时，按以下顺序操作：

1. 展开 `knowledge_base/` 中的运行数据库；
2. 启动 `mcps/mediflow-mcp/`，在 Nexent 注册并扫描 `medical-ai`；
3. 导入对应的 `agent.json`；
4. 为智能体选择目标环境的模型，确认 MCP 工具绑定；
5. 如使用 Nexent 原生知识库，重新导入源文档并绑定目标知识库；
6. 保存并发布。

三个子目录已经放入对应的 `agent.json`。配置文件只描述模型、提示词、工具关系和知识库绑定，不会携带知识库中的全部原始文档。因此知识库迁移方式见 [`../knowledge_base/README.md`](../knowledge_base/README.md)。导入其他环境前，应将 `mcp_info` 中的示例地址改为目标 MCP 服务地址，并确认目标环境已注册 `medical-ai`；同时按目标环境生成的索引标识更新 `knowledge_base_search` 的 `index_names`。公开发布前请确认配置不包含密码、API Key、Cookie 或个人目录。
