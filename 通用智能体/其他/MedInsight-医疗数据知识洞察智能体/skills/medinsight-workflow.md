# MedInsight 数据—知识—洞察工作流

## 使用时机

当用户需要处理医疗文本或表格、生成医疗知识图谱、基于图谱问答，或对图谱执行统计分析和可视化时使用本技能。

## 规划规则

1. 原始数据必须先调用 `process_medical_data`。
2. 仅将任务一的处理文档交给 `build_medical_knowledge_graph`。
3. 问答只能调用 `ask_medical_knowledge_graph`，不得脱离图谱补写医学事实。
4. 统计、关联、完整度和趋势分析调用 `query_medical_graph_analysis`。
5. 需要完整报告、BI 看板和 NL2SQL 评测时调用 `analyze_medical_graph`。
6. 用户明确要求端到端执行时，优先调用 `run_data_knowledge_insight_pipeline`。

## 输出约束

- 明确区分工具返回的事实、质量指标和系统建议。
- 保留产物路径、执行状态、异常数量、质量门控结果和证据路径。
- DataMate 不可用时必须展示 `fallback_used`、`fallback_reason` 和 `missing_services`。
- 医疗问答仅供技术演示和知识检索，不作为诊断或用药依据。
