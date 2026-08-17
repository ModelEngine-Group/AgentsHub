# 医学核心能力

`core/` 保存可复用的医学语义处理能力，供 MCP 工具、DataMate 算子和离线数据库构建代码调用。

## 能力范围

| 能力 | 主要模块 |
| --- | --- |
| 文本预处理和质量判断 | `text_preprocessor.py`、`text_quality.py` |
| 医学实体与关系抽取 | `medical_extraction_service.py`、`medical_offline_extraction.py` |
| 术语和缩写规范化 | `medical_normalize.py`、`medical_lexicon.py` |
| 抽取结果校验 | `medical_extraction_validation.py`、`medical_reliability.py` |
| 疾病事实查询 | `medical_query_engine.py` |
| 自然语言统计查询 | `nl2sql.py` |
| 数据结构定义 | `schemas.py` |
| 模型调用 | `llm_client.py` |

## 调用关系

```text
Nexent 智能体
  → MCP 工具
  → core 医学语义处理
  → 知识图谱库或分析库
  → 文字、表格和图表结果
```

DataMate 的文件清洗由 `operators/` 中的算子完成；`core/` 负责共享的医学语义能力，不直接管理 Nexent 页面，也不负责外部平台账号。
