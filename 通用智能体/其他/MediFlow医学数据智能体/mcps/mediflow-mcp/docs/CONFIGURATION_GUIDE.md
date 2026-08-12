# 配置说明

服务运行时读取 `mcps/mediflow-mcp/.env.runtime`。可以从 `.env.example` 复制一份，再按目标环境填写；根目录 `config/` 还提供一份便于阅读的 YAML 参考。

## 常用配置

| 配置项 | 作用 |
| --- | --- |
| `CCF_LLM_API_KEY` | 模型服务密钥 |
| `CCF_LLM_BASE_URL` | OpenAI 兼容的模型接口地址 |
| `CCF_LLM_MODEL` | 使用的模型名称 |
| `CCF_NEXENT_CONFIG_BASE` | Nexent 配置服务地址 |
| `CCF_NEXENT_RUNTIME_BASE` | Nexent 运行服务地址 |
| `CCF_MCP_URL` | Nexent 访问 MCP 的地址 |
| `CCF_MCP_SERVICE_NAME` | MCP 在 Nexent 中的注册名，默认 `medical-ai` |
| `MCP_HOST` / `MCP_PORT` | MCP 监听地址和端口 |
| `CCF_DATAMATE_BASE` | DataMate API 地址 |
| `CCF_DATAMATE_GATEWAY` | DataMate 网关地址 |
| `CCF_DATASET_VOLUME` | DataMate 数据集文件目录 |
| `CCF_DATAMATE_OPERATOR_VOLUME` | DataMate 自定义算子目录 |
| `CCF_DATA_ROOT` | 本地待处理资料目录 |
| `CCF_TASK2_KG_DB` | 知识图谱数据库路径 |
| `CCF_TASK3_ANALYTICS_DB` | 分析数据库路径 |
| `CCF_MINERU_API` | PDF 解析服务地址 |

`CCF_TASK2_KG_DB` 和 `CCF_TASK3_ANALYTICS_DB` 是代码中保留的兼容变量名，分别指向知识图谱数据库和分析数据库，不代表额外的平台服务。

## 抽取参数

以下变量用于调节医学关系抽取的增强路径：

- `CCF_TASK2_BACKEND`：`offline` 使用本地抽取，`hybrid` 在本地结果基础上调用模型补充和复核；
- `CCF_TASK2_CASCADE_MAX_GAP_SEGMENTS`：单条记录允许送入模型补充的缺口句子数；
- `CCF_TASK2_CASCADE_MAX_GAP_SEGMENTS_TOTAL`：一次数据集处理的缺口上限；
- `CCF_TASK2_CASCADE_GAP_WORKERS`：并发补充请求数；
- `CCF_TASK2_CASCADE_GAP_BATCH_SIZE`：每次补充请求合并的记录数；
- `CCF_TASK2_CASCADE_MAX_REVIEW_CANDIDATES`：模型复核候选上限；
- `CCF_TASK2_LLM_GAP_AUTO_ACCEPT_CONFIDENCE`：通过原文证据检查后进入快速通道的最低置信度。

## 外部入口

| 配置项 | 作用 |
| --- | --- |
| `CCF_NEXENT_FRONTEND_URL` | Nexent 页面地址 |
| `CCF_DATAMATE_FRONTEND_URL` | DataMate 页面地址 |
| `CCF_ANALYTICS_FRONTEND_URL` | 可选分析前端地址 |
| `CCF_PUBLIC_DOMAIN` | 未单独配置入口时用于生成公开地址 |

分析前端是可选外部服务，本 MCP 包不负责启动它。若不使用，可留空。

## 安全与迁移

- 真实密码、Token、Cookie 和模型密钥只保存于 `.env.runtime`，不要提交；
- 新环境先展开 `knowledge_base/` 数据库，再检查数据库路径；
- 在 Nexent 中注册 MCP 并扫描工具后，再导入智能体配置；
- Nexent 原生知识库的源文档、嵌入模型和绑定关系在 Nexent 中单独配置，说明见 [`../../../knowledge_base/README.md`](../../../knowledge_base/README.md)。
