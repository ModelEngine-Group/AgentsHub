# 数据库与知识库说明

本目录包含 MCP 服务直接读取的运行数据资产。为避免单个数据库文件过大，资产以 ZIP 保存；在归档根目录执行 `python prepare_runtime_assets.py` 后，脚本会校验 SHA-256 并展开到运行代码需要的位置。

## 随包提供的资产

| 压缩文件 | 展开文件 | 用途 |
| --- | --- | --- |
| `task2_medical_kg.db.zip` | `data/task2_medical_kg.db` | 医学实体、关系、三元组、来源和质量记录 |
| `task3_analytics.db.zip` | `data/task3_analytics.db` | 疾病事实、统计视图和自然语言分析所需的数据表 |
| `noise_kb.db.zip` | `operators/llm_noise_filter/noise_kb.db` | 语义噪声识别和质量审计规则 |
| `term_kb.db.zip` | `operators/medical_term_normalizer/term_kb.db` | 医学术语和缩写映射 |
| `task2_assets.zip` | `data/task2/*.json` | 实体词典、关系词表和可靠性配置 |

文件大小、目标路径和校验值记录在 [`MANIFEST.json`](MANIFEST.json)。这些数据库是运行产物，已经包含在归档包中，不需要再从服务器下载。

## 可重新导入 Nexent 的文本源文件

`nexent/reconstructed_source_documents/` 中有 122 个文本源文件（121 个医学 JSON 文件和 1 个文本文件）。它们依据当前 Nexent 服务端的只读文档内容重建，用于在新的 Nexent 环境中重新建立向量索引；它们不是 Nexent 数据库文件，也不替代原始数据提供方的授权文件。使用前请确认目标环境允许导入和再分发这些内容。

建议使用与运行时一致的 1024 维嵌入模型建立新知识库，然后批量上传上述文件。导入完成后，在 Nexent 中复制新索引标识，更新两个需要文本检索的 Agent 配置中 `knowledge_base_search` 的 `index_names`，并重新扫描 MCP 工具后发布。

## 与 Nexent 知识库的区别

随包数据库是 MCP 代码直接打开的 SQLite 和 JSON 文件；Nexent 知识库则由 Nexent 服务负责文档上传、分片、向量化和检索，两者不是同一种资产。

当前运行环境中的旧 `ccf_medical_kb` 属于 Nexent 服务端知识库。旧索引的文档向量为 768 维，而运行时查询向量为 1024 维，因此不应继续把旧索引作为混合检索的默认绑定。归档同时提供了按只读文档内容重建的源文件和一个使用 1024 维嵌入模型的并行索引配置；目标环境仍需重新上传、等待索引完成并按环境生成的索引标识绑定。

如果目标环境需要使用文本知识库：

1. 使用归档中的重建源文件，或准备拥有再分发权限的原始医学文档；
2. 在 Nexent 中新建知识库，并使用 1024 维嵌入模型；
3. 上传文档并等待索引完成；
4. 将知识库绑定到需要文本依据的智能体；
5. 用“疾病定义、病因、症状、治疗依据”等问题验证检索结果。

如果没有可授权的文本源文件，可以继续使用随包的知识图谱数据库完成结构化查询，也可以由使用者提供新的、可再分发文档集。
