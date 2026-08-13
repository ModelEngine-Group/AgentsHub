# 评分表 ↔ 证据 ↔ 一键命令（评委速览一页纸）

> 30 秒定位每一分：左列是赛题评分项，中列是可复核证据文件，右列是一键复现命令。
> 数据为确定性种子合成（`--seed 42`）；真实基准（CBLUE）另见 `docs/cblue_datacard.md`。

## 任务一：数据处理智能体（30）
- 架构设计合理性(5)：四层单向依赖 → `docs/architecture.md` §1。
- 智能体能力(10)：ReAct 侦察→规划→执行→跟踪→恢复 → `tests/test_agents.py::test_scripted_agent_full_react_loop`；在线 `.\run.ps1 task1 --agent`。
- 功能完整性(10)：200 病历 ETL 零残留 → `.\run.ps1 task1`；`outputs/tasks/*.json`。
- 工程质量(5)：pytest 全绿 + 一键脚本 + 文档 → `.\run.ps1`（见 §测试）。
- 加分：<1B 微调 → `docs/ner_finetuning.md`；`models/medical_ner_bert`（bert F1 0.9936）。

## 任务二：知识图谱问答智能体（40 + NPU 10）
- 图谱设计(10)：6×7 本体 + 域/值域约束 → `dki/kg/schema.py`。
- 算子设计低时延(10)：5 规范算子，CPU 实测吞吐/延迟 → `outputs/bench/operator_bench.md`（`.\run.ps1 bench-ops`）：
  实体抽取 4.7k docs/s、关系 4.1k、三元组校验 5.4k、清洗组内置算子 6k–16k docs/s。
- 编排能力(10)：一句话触发全链 → `.\run.ps1 task2 --agent`。
- 结果质量(10)：**三栏**——① 合成 **F1 99.97%**（受控流水线）；② 同分布微调/OOD；③ CMeEE 均衡基线 0.6031 + OM 官方 dev 0.613020。CMeIE 完整 text-to-SPO 必须通过 calibration、三 seed family-disjoint outer 和官方 dev 开发确认；当前候选只从 `FINAL_STATUS.json` 的哈希绑定摘要读取，gold-subject、selection-used 或 waiver 均不能替代 ≥0.50 门。
- NPU 优化(10)：CMeEE FP16 OM 326.098 docs/s、CPU 18.393、17.7295x、官方 F1 差 0.000194、0.6641 J/doc；见 `docs/remote_npu_acceptance_20260725.md`。

## 任务三：数据分析智能体与可视化（30）
- 有效复用(5)：SQLite 同时装载任务一结构化表 + 任务二图谱 → `dki/analysis/database.py`。
- 智能体能力/NL2SQL(10)：`QueryPlan v2` 与确定性 rule 共用有界计划和 SQLGlot 安全门。所有 development 与已消费集合只作 selection-used 历史；新 prospective 集合在最终源码密封后只消费一次，overall 严格 `>0.85`、hard+ `≥0.70`。页面与报告只读取 aggregate-only attestation/`FINAL_STATUS.json`，不公开问题或 SQL。
- 可视化+图谱(10)：ECharts 单文件报告——概览卡片 + 柱/折/饼 + **年龄×疾病热力图** + **疾病→科室桑基图** + 图谱力导向图 + **关键洞察 Top-N** + **图谱驱动关联洞察** → `.\run.ps1 task3`，`outputs/reports/insight_report.html`。
- 工程实现(5)：SQL 只读白名单防注入；模板与数据分离。

## 鼓励方向（决定名次）
- 创新性：契约兼容层 / ScriptedModel 离线 ReAct / GlobalPointer 嵌套 NER / INT8 带精度验证 / 进程独占 ACL + manifest 门 / typed-claim 证据体系 / final3 内容哈希门 / NL2SQL 多候选自洽。
- 工程可复现：确定性种子 + pytest 全绿 + clean-room 提交包（hash manifest + 密钥扫描）→ `scripts/build_submission_bundle.py` + `scripts/verify_submission_bundle.py`。
- 性能量化：CPU/NPU/FP16 吞吐、p50/p95、同步能效与官方质量差；NL2SQL 分层 EX/覆盖/拒答；CBLUE 真实基准。
- 深度集成：真实 DataMate `RayDataset.process` 实测（`outputs/remote_acceptance_20260720_datamate_ray.json`，200 病历 1569 三元组）+ 算子市场 ZIP + Nexent CoreAgent/Tool 范式。

## 测试与交付
- 全量测试：历史目标机基线 302 passed / 4 skipped；当前候选只在源码冻结和 clean-room RC 两个节点运行全量测试，精确 passed/skipped/node IDs/JUnit/skip reasons 由 `FINAL_STATUS.json` 与 verification receipt 绑定。
- typed claims：`python scripts/normalize_cmeee_metrics.py` + `python scripts/check_claims_metrics_consistency.py` + `python scripts/build_evidence_claims.py` + `python scripts/validate_evidence.py outputs/claims/headline_claims.json`。
- 提交包复验：`python scripts/build_submission_bundle.py` + `python scripts/verify_submission_bundle.py --bundle <zip> --python .venv\Scripts\python.exe`。
