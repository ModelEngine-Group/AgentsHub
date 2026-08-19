# 文档模块索引

本目录存放长期维护文档。临时运行产物与截图放在 `outputs/competition_evidence/`；可提交的基准报告放在 `benchmarks/reports/`。

## 快速阅读路径

| 场景 | 推荐阅读顺序 |
| --- | --- |
| 快速了解项目 | [技术答辩材料](competition_defense_document.md) → [答辩材料说明](competition_defense_outline.md) |
| 核对技术实现 | [架构说明](architecture.md) → 对应任务文档 → [依赖与环境](dependencies.md) |
| 复现演示与测试 | [依赖与环境](dependencies.md) → [部署记录](preparation.md) → [任务一](task1_data_agent.md) / [任务二](task2_kg_agent.md) / [任务三](task3_analysis_agent.md) |
| 在线集成 | [部署记录](preparation.md) → [在线集成](online_integration.md) |
| 核对 NPU 结论 | [NPU 优化说明](npu_optimization.md) → [服务器环境](server_environment.md) |

## 文档目录

| 文档 | 内容 |
| --- | --- |
| [架构说明](architecture.md) | Agent / Operator / Pipeline 分层、三任务闭环、外部集成边界 |
| [依赖与环境](dependencies.md) | Python 依赖、LLM 配置、路径安全 |
| [部署记录](preparation.md) | Windows/WSL 环境、Nexent/DataMate 部署与复现边界 |
| [在线集成](online_integration.md) | Nexent OpenAPI 导入、DataMate 提交、安全写入开关 |
| [任务一](task1_data_agent.md) | 数据处理智能体：规划、清洗、DataMate、API |
| [任务二](task2_kg_agent.md) | 知识图谱：抽取、问答、Neo4j、NPU |
| [任务三](task3_analysis_agent.md) | 图谱分析：NL2SQL、可视化、闭环 |
| [技术答辩材料](competition_defense_document.md) | 答辩正文源稿与证据索引 |
| [答辩材料说明](competition_defense_outline.md) | HTML 打包、证据采集步骤 |
| [本地小模型微调](local_model_finetune.md) | 三任务 QLoRA 训练与验证 |
| [NPU 优化说明](npu_optimization.md) | Ascend CPU/NPU 基线对照与复现命令 |
| [服务器环境](server_environment.md) | 910B3 硬件栈与验证快照 |
| [基准测试](../benchmarks/README.md) | CPU/NPU 基准脚本与报告索引 |
| [答辩提交包](../competition_submission/README.md) | 可提交的 HTML 证据包入口 |

## 维护约定

- 正文使用中文；命令、文件名、API 字段保留原文。
- 粗体仅用于小节标题（如 **1.1 建设目标**）与表格「结果」列中的 pytest/benchmark 计数（如 **437/437**、**76/76**）；说明性文字、日期与文件名不用粗体（文件名用反引号）。
- 任务文档写实现与复现，答辩文档写交付证据，环境文档写部署，避免同一内容多处复制。
- 涉及准确率、加速比、pytest 数量等结论，须指向可运行命令或 `benchmarks/reports/` 中的报告。
- 新增或重命名文档时，同步更新本文件与根目录 [README.md](../README.md) 的文档模块导航。
