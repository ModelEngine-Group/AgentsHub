# 赛题评分项 ↔ 实现证据映射

> 便于评审快速定位：每个评分项对应的代码位置、运行命令与量化证据。

## 任务一：数据处理智能体（30 分）

| 评分项 | 分值 | 实现证据 |
|---|---:|---|
| 架构设计合理性（Agent 与数据处理模块解耦） | 5 | 四层单向依赖：智能体（`dki/agents/`）→ 工具（`dki/agents/tools/`）→ 执行器（`dki/pipeline/runner.py`）→ 算子（`dki/operators/`）。智能体不触碰算子实现；算子可独立上传 DataMate 平台运行（`dist/operators/*.zip`）。详见 `docs/architecture.md` §1 |
| 智能体能力（任务规划与自动执行） | 10 | ReAct 闭环：侦察（list_operators/preview_data）→ 规划 → 执行（run_pipeline）→ 跟踪（get_task_status）→ 异常恢复（工具结构化错误 + constraint 提示词的诊断-调整-重试流程）。在线模式 `.\run.ps1 task1 --agent`；离线确定性验证 `tests/test_agents.py::test_scripted_agent_full_react_loop` |
| 功能完整性（流程可跑通、结果正确） | 10 | `.\run.ps1 task1`：200 份病历 ETL，输出 200 份零残留（HTML/全角/不可见字符/重复句/PII 全清，测试断言 `tests/test_operators.py::test_cleaner_removes_noise`）。任务报告 `outputs/tasks/*.json` |
| 工程质量 | 5 | 历史验收基线为 302 passed / 4 skipped。当前候选的两次最终全仓测试、JUnit、node IDs、skip reasons 和 clean-room 复验必须由随包 `FINAL_STATUS.json` 绑定；开发阶段核心测试不能冒充最终全测。 |
| **加分：构造数据集 + 微调 <1B 小模型 + 编排增强** | +? | dict 后端（F1≈99.97%）远程监督弱标注 → 微调 `bert-base-chinese`（101.7M «1B，13 BIO 标签，`scripts/train_ner.py`）。**40 份训练不可见验证集实体 F1：微调 BERT 0.9936 vs dict 教师 0.9974 vs 未微调基线 0.0**（`outputs/ner_finetune/metrics.json`）。微调模型自动接回 bert 后端（真实医疗类型 + 规范名归一，向后兼容占位）；新增 `configs/pipeline_kg_build_bert.yaml` 编排增强变体。详见 `docs/ner_finetuning.md` |

**与 DataMate/Nexent 集成深度**：MedicalTextCleaner 内部真实调用 DataMate 内置算子
（HtmlTagCleaner/FullWidthCharacterCleaner/ExtraSpaceCleaner/AnonymizedPhoneNumber/
AnonymizedIdNumber 等 8 个）；执行器与 `process.yaml`/RayDataset 语义同构；任务状态机
对齐 CleaningTaskStatus——逐条对照表见 `docs/architecture.md` §2/§3。

## 任务二：知识图谱问答智能体（40 分 + NPU 10 分）

| 评分项 | 分值 | 实现证据 |
|---|---:|---|
| 知识图谱设计（实体/关系定义清晰） | 10 | 6 实体类型 × 7 关系类型 + 定义域/值域约束（`dki/kg/schema.py`）；覆盖赛题要求的诊断/治疗/并发关系 |
| 算子设计与实现（可复用、接口清晰、低时延） | 10 | 3 个核心算子均为 DataMate 规范包（实体/关系/三元组校验），CPU 实测 p50 0.13~0.23ms、吞吐 4k-6.4k docs/s（`outputs/bench/operator_bench.md`）；可打包上市场（`python scripts/package_operators.py`） |
| 智能体编排能力 | 10 | KG 智能体一句话触发全链构建（build_knowledge_graph 工具封装 5 算子流水线 + 跨文档聚合）；`.\run.ps1 task2 --agent` |
| 结果质量（合理、可解释） | 10 | **三栏**：① 合成 F1≈99.97%（只证明受控流水线正确性）；② 同分布微调/OOD；③ CMeEE 冻结均衡基线 0.6031 与 OM 官方 dev 0.613020。CMeIE 完整 text-to-SPO 只有在 calibration≥0.45、三 seed outer median≥0.50/worst≥0.48 且官方 dev 开发确认≥0.50 后才记 PASS；当前候选值由 `FINAL_STATUS.json` 的哈希绑定摘要填写，任何 waiver、gold-subject 或开发分解不得替代。 |
| NPU 算子优化（加分） | 10 | CMeEE schema-v3 FP16 OM 真实 pyACL 验收：326.098 docs/s，CPU ONNX 18.393 docs/s，17.7295x；官方 dev F1 差 0.000194；0.6641 J/doc、扣 idle 0.1571。证据 `docs/remote_npu_acceptance_20260725.md`。 |

图谱问答：模板引擎 9 类问法（无 LLM，`.\run.ps1 task2-qa --question "高血压有哪些症状？"`，
带证据句）+ 智能体自由问答（LLM 模式）。

## 任务三：数据分析智能体与可视化（30 分）

| 评分项 | 分值 | 实现证据 |
|---|---:|---|
| 有效复用任务一、二产出 | 5 | SQLite 同时装载任务一结构化表（patients/visits）与任务二图谱（kg_entities/kg_triples）（`dki/analysis/database.py`）；BI 报告内嵌图谱力导向关系图 |
| 智能体能力（NL2SQL 准确率越高越好） | 10 | `QueryPlan v2` 与确定性 rule 共用 SQLGlot 安全门。development、known regression 与 prospective holdout 严格分栏；v28 已合法一次性消费但 overall 未过门，永久禁用。新集合必须在最终源码密封后只消费一次，以整数门 `100×correct > 85×n`、`100×hard_correct ≥ 70×hard_n` 判定；最终只引用 aggregate-only attestation 与 `FINAL_STATUS.json`。 |
| 可视化与图谱结合 | 10 | ECharts 单文件报告：**关键洞察 Top-N**（数据驱动自动萃取）+ 概览卡片 + 柱/折/饼 + **年龄段×疾病热力图** + **疾病→科室桑基图** + 图谱力导向图（分类图例/邻接高亮/可拖拽）+ **图谱驱动关联洞察**（疾病-症状-药物跨表 JOIN）+ 每图数据驱动中文解读（`.\run.ps1 task3` → `outputs/reports/insight_report.html`） |
| 工程实现 | 5 | SQL 只读白名单防注入；报告模板与数据分离；智能体图表工具会话式累积 |

## 鼓励方向对照

| 方向 | 本项目 |
|---|---|
| 创新性设计 | manifest 驱动的数据→图谱→分析闭环；真实 CoreAgent 异步状态/恢复轨迹；Graph+SQL 工具链；关系逐谓词校准；CMeEE checkpoint→ONNX→ATC→schema-v3 manifest→pyACL→官方质量→同步能耗的 fail-closed 发布链。DataMate Mapper 仍为 fill=1，不夸称平台原生微批。 |
| 工程可复现性 | 一键环境脚本 + 确定性种子 + 远端分环境测试 + 完整 stdout/JUnit/node IDs/skip reasons + create-once 演示 receipt + 中文 Web 驾驶舱与 Notebook 辅助复现。当前候选的精确测试数量由 `FINAL_STATUS.json` 绑定。 |
| 性能量化对比 | 9 算子吞吐/延迟基准；NL2SQL 覆盖率/拒答率/执行准确率；CMeEE/CMeIE P/R/F1；当前 CMeEE NPU 三次中位数、p50/p95、CPU 对照、官方质量差与同步能效。 |
| 深度集成 | Nexent：CoreAgent/OpenAIModel/MessageObserver/Tool 范式/三段式提示词；DataMate：base_op 继承/OPERATORS 注册/process.yaml 同构/内置算子复用/AC 自动机复用/算子市场包/NPU 钩子 |
| 端到端闭环 | `.\run.ps1 demo-ui` 启动中文证据驾驶舱；每次点击使用独立 create-once 输出运行 `.\run.ps1 demo-all --agent --offline` 的数据 → 清洗 → 图谱 → 问答 → NL2SQL → BI 闭环，并保留哈希绑定 receipt。 |
