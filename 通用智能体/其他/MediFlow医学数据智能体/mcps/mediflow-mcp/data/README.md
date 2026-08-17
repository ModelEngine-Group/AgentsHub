# 运行数据目录

`data/` 是 MCP 服务展开数据库和接收本地输入资料的位置。归档包把数据库压缩保存在上级 `knowledge_base/`；在归档根目录执行 `python prepare_runtime_assets.py` 后，资产会展开到这里及相关算子目录。

| 路径 | 作用 |
| --- | --- |
| `task2_medical_kg.db` | 医学知识图谱数据库 |
| `task3_analytics.db` | 疾病事实和统计分析数据库 |
| `input/` | 使用者提供的 TXT、CSV、JSON、JSONL 或 PDF 输入目录 |

输入资料由使用者提供，或直接使用已经注册到 DataMate 的数据集。数据库版本、目标路径和校验值见 [`../../../knowledge_base/MANIFEST.json`](../../../knowledge_base/MANIFEST.json)。
