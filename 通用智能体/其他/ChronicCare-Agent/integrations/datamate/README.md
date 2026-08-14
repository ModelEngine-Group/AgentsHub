# ChronicCare DataMate算子包

`integrations/datamate/`保存DataMate算子目录、便携式参考实现、算子元数据、编排清单和NPU增强源码。正式全流程由外部`datamate-runtime`中的同名算子执行，本地源码用于交付审查、独立算子复现和NPU增强集成。

目标：

- 记录 DataMate 容器内 11 个慢病算子的正式名称与阶段映射
- 保留与 `tool_server/`、`mcp_adapter/`、Nexent 的衔接证据
- 说明外部`datamate-runtime`正式执行入口与本地交付源码的对应关系

## 算子数量

当前提供 11 个算子：

- `chronic_file_ingest`
- `chronic_table_clean`
- `chronic_field_normalize`
- `chronic_text_split`
- `chronic_entity_extract`
- `chronic_relation_extract`
- `chronic_triple_validate`
- `chronic_kg_build`
- `chronic_sqlite_loader`
- `chronic_nl2sql_analyze`
- `chronic_report_pack`

## 运行方式

当前主流程：

```bash
python3 scripts/run_datamate_full_pipeline.py
python3 scripts/sync_datamate_outputs_to_mainline.py
python3 scripts/check_datamate_full_pipeline.py
```

## 设计说明

- 11个正式算子在`datamate-runtime`容器中通过`datamate.ops.mapper.*`路径执行。
- 本目录保留对应算子的便携式参考源码和元数据，NPU增强算子直接从本目录加载。
- DataMate负责数据集与算子链路；Tool Server负责API化；MCP Adapter负责工具包装；Nexent负责智能体规划与调度。

## 增强数据同步

- 新增 5 张慢病管理表会被 DataMate 风格链路识别。
- SQLite loader 会装载新增表。
- KG 构建会补充增强版风险与随访管理实体。
- NL2SQL 分析问题扩展到 35 个。
- 报告打包会输出增强版评测报告与当前指标文件。

医疗安全说明：本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。
