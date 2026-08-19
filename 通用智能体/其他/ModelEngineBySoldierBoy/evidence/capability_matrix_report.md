# 智能体能力评估矩阵

- 生成时间：2026-07-06T17:13:25+08:00
- 检查项：17
- 通过：17
- 部分：0
- 失败：0

## 明细

| 任务 | 能力 | 状态 | 关键指标 | 证据 |
| --- | --- | --- | --- | --- |
| task1 | 任务理解与动态计划 | `pass` | trace_steps=14 | `data/processed/data_processing_trace.jsonl`<br>`demo/task1_processing_report.md` |
| task1 | 结构化与非结构化输入处理 | `pass` | csv={'documents': 2, 'structured': True, 'status': 'completed'}, txt={'documents': 2, 'unstructured': True, 'status': 'completed'}, invalid={'documents': 3, 'warning_count': 4, 'recovery_action_count': 3, 'status': 'completed_with_warnings'} | `data/raw/medical_demo_structured.csv`<br>`data/raw/medical_demo_notes.txt` |
| task1 | 状态跟踪与异常恢复 | `pass` | documents=3, warning_count=4, recovery_action_count=3, status=completed_with_warnings | `data/raw/medical_demo_invalid.jsonl`<br>`demo/task1_invalid_processing_report.md` |
| task2 | 医疗 KG schema 与关系覆盖 | `pass` | relation_type_count=15, entity_pair_type_count=16 | `graph/kg_schema.json`<br>`graph/kg_validation_report.json` |
| task2 | 核心算子编排与后端可插拔 | `pass` | backend_trace_events=41 | `graph/kg_build_trace.jsonl`<br>`operators/medical_pipeline.py` |
| task2 | 三元组质量与可解释问答 | `pass` | accepted_triples=3403, rejected_triples=0, qa_count=2, gold_eval_enabled=True, gold_eval_f1=0.9724, gold_eval_recall=1.0, gold_eval_matched=3220, gold_eval_total=3220, backend_probe_enabled=True, backend_probe_mode=lexicon_assisted_backend, backend_probe_f1=0.7722, backend_probe_recall=0.7531, backend_probe_documents=40 | `graph/triples.jsonl`<br>`graph/kg_validation_report.json`<br>`demo/qa_trace.jsonl` |
| task3 | 图谱复用与分析查询 | `pass` | query_count=15, analysis_query_rows=15 | `demo/analysis_summary.json`<br>`demo/analysis_queries.jsonl` |
| task3 | 任务一/二产物复用闭环 | `pass` | lineage_ready=True, closed_loop_ready=True, task1_documents=320, task2_triples=3403 | `demo/task3_lineage.json`<br>`demo/task3_closed_loop_status.json` |
| task3 | 图谱结构理解与查询能力识别 | `pass` | node_count=4468, edge_count=3403, supported_analysis_modes=['统计分析', '关联分析', '趋势/结构代理分析', '风险与因果路径', 'BI图表', '图结构可视化'] | `demo/graph_understanding.json`<br>`demo/analysis_planning.json` |
| task3 | NL2SQL 准确率门槛 | `pass` | accuracy=1.0, passed=70, total=70 | `demo/nl2sql_eval_report.json`<br>`demo/nl2sql_eval_report.md` |
| task3 | BI 图表与图谱子图展示 | `pass` | subgraph_count=3, bi_chart_count=25, graph_visual_count=4 | `demo/dashboard.html`<br>`demo/charts/*.json`<br>`demo/subgraphs/*.jsonl` |
| chain | 三智能体端到端链路 | `pass` | document_count=320, entity_count=4468, triple_count=3403, qa_count=3, analysis_query_count=15, analysis_subgraph_count=3, nl2sql_total=70, nl2sql_passed=70, nl2sql_accuracy=1.0, kg_candidate_triples=3418, kg_accepted_triples=3403, kg_rejected_triples=0, data_quality_warning_count=0, external_document_count=300, external_triple_count=3233, external_entity_count=793 | `demo/orchestration_summary.json`<br>`demo/orchestration_trace.jsonl`<br>`demo/orchestration_report.md` |
| platform | DataMate/Nexent 真实联通与检索 | `pass` | datamate_reachable=True, nexent_reachable=True, datamate_search_success=True | `demo/integration/platform_evidence.json`<br>`demo/integration/datamate_search_trace.jsonl` |
| task1 | 基于 Nexent OpenAPI/MCP 构建数据处理智能体 | `pass` | openapi_registered=True, tool_visible=True, runtime_ready=True, run_attempted=True, run_ok=True | `demo/integration/task1_nexent_openapi.json`<br>`demo/integration/nexent_task1_tool_trace.jsonl`<br>`demo/integration/nexent_task1_runtime_evidence.json` |
| task2 | 基于 Nexent OpenAPI/MCP 构建医疗知识图谱智能体 | `pass` | openapi_registered=True, tool_visible=True, runtime_ready=True, implemented=True, evidence_collected=True, run_attempted=True, run_ok=True, latest_kg_ready=True | `demo/integration/task2_nexent_openapi.json`<br>`demo/integration/nexent_task2_tool_trace.jsonl`<br>`demo/integration/nexent_task2_runtime_evidence.json` |
| task3 | 基于 Nexent OpenAPI/MCP 构建图谱驱动数据分析智能体 | `pass` | openapi_registered=True, tool_visible=True, runtime_ready=True, implemented=True, evidence_collected=True, run_attempted=True, run_ok=True, latest_analysis_ready=True, latest_interactive_ready=True | `demo/integration/task3_nexent_openapi.json`<br>`demo/integration/nexent_task3_tool_trace.jsonl`<br>`demo/integration/nexent_task3_runtime_evidence.json` |
| external | 官方 CBLUE 数据边界 | `pass` | cblue_mode=official_ready, cblue_official_ready=True | `docs/cblue_evidence_report.md` |

## 边界说明

- `pass` 表示当前仓库产物和本机运行结果能直接证明该能力存在。
- `partial` 表示能力有入口或部分证据，但不足以作为完整完成声明。
- `fail` 表示当前证据不足或与要求冲突。
- 官方 CBLUE 项只记录真实数据口径；没有官方数据时不宣称官方数据验收通过。
