---
name: 数据知识洞察分析
description: 按照数据处理、知识图谱、图谱问答和数据分析的依赖顺序编排本地工具，并规范化结果输出。
tags:
  - 数据处理
  - 知识图谱
  - 数据分析
  - 医疗示例
---

# 数据知识洞察分析

## 适用场景

用于处理仓库 `data/` 目录中的医疗示例 CSV 和文本数据，完成“数据 → 知识 → 洞察”闭环。

## 工具编排

1. 数据清洗或文本处理：调用 `process_medical_data`。
2. 需要知识图谱：调用 `build_medical_knowledge_graph`。
3. 需要基于图谱问答：确认 `medical_kg.json` 存在后调用 `ask_medical_kg`。
4. 需要统计、图表或报告：确认 `medical_kg.json` 和 `cleaned_records.csv` 存在后调用 `analyze_medical_data`。

## 输出要求

- 先报告每个工具的 `status`，再列出关键结果和生成文件。
- 遇到 `missing_prerequisite` 时，指出缺少的文件和应先调用的工具。
- 遇到 `failed` 或错误时，保留错误信息，不将任务描述为成功。
- 图谱问答返回答案时同时保留可追溯的证据三元组（如果工具提供）。
- 医疗示例数据仅用于演示数据处理与分析流程，不构成诊断或治疗建议。

## 本地运行

在 Agent 根目录启动 MCP 服务：

```bash
uv sync
uv run python -m adapters.nexent_mcp_tools
```

服务默认地址：`http://127.0.0.1:8001/sse`。
