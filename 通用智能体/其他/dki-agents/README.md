# dki-agents：基于 Nexent 的"数据—知识—洞察"智能体与算子实现

面向医疗领域，基于 **Nexent 智能体框架** 与 **DataMate 数据处理算子体系**，实现
"数据 → 知识 → 洞察" 完整闭环的三个智能体与五个可上传 DataMate 算子市场的医疗算子。

**核心量化结果（本仓库可一键复现）**

> 当前候选是否达到比赛质量门，只以随候选包生成的 `FINAL_STATUS.json` 及其
> SHA-256 绑定证据为准。README 只保存稳定的历史基线与评测口径，不把开发集、
> 已消费旧盲测或正在运行的实验写成“最新通过结果”。

| 指标 | 结果 |
|---|---|
| 知识抽取质量 · 合成受控语料（vs 1580 条金标准三元组） | 微平均 P=100% / R=99.94% / **F1=99.97%**（验证**流水线正确性**：抽取器与语料同源于一部词典） |
| 知识抽取质量 · **真实权威基准 CBLUE**（<1B） | **三栏分报**：① 合成流水线正确性 F1≈99.97%（非开放域）；② 同分布微调见 `docs/ner_finetuning.md`；③ CMeEE-V2 冻结均衡基线 F1 0.6031，已验收 OM 官方 dev **F1 0.613020**（P=0.583679/R=0.645468）。CMeIE 完整 text-to-SPO 必须依次通过 calibration、family-disjoint outer 与官方 dev 开发确认；正在运行或未过门的候选不写入 README 主指标，最终值从 `FINAL_STATUS.json` 读取。默认规则路径不变。 |
| 微调 NER（<1B 小模型，40 份训练不可见验证集实体 F1） | **微调 BERT 0.9936** vs dict 教师 0.9974 vs 未微调基线 0.0（dict 远程监督弱标注微调 `bert-base-chinese`，`docs/ner_finetuning.md`） |
| NL2SQL 执行准确率 | development、known regression 与 prospective independent holdout 严格分栏。所有已消费集合永久禁用；新候选必须在源码、依赖、数据库和 evaluator 密封后只消费一次，并满足 overall 严格 `>0.85`、hard+ `≥0.70`。当前候选的权威聚合结论只从 `FINAL_STATUS.json` 与 aggregate-only attestation 读取。 |
| 核心算子时延（CPU 实测 p50） | 实体抽取 0.232ms / 关系抽取 0.176ms / 三元组校验 0.129ms（`outputs/bench/operator_bench.md`） |
| 核心算子吞吐（100 份病历，repeat=3 中位） | 4k ~ 19k docs/s |
| CMeEE 推理 NPU 加速 | 绑定当前 CMeEE checkpoint 的 FP16 OM：三次中位数 **326.098 docs/s**，CPU ONNX 18.393 docs/s，**17.7295×**；NPU 串行/微批 200/200 一致；官方 dev F1 差 0.000194；功耗覆盖 93.6%，0.6641 J/doc（扣 idle 0.1571）。见 `docs/remote_npu_acceptance_20260725.md`。 |
| 分布外鲁棒性（span-overlap） | **质量 v2 在盲测-2** 达 0.702/0.880/0.781，较旧 quality 三指标同升（P +0.80、R +6.02、F1 +2.93 点）；在模型/规则预先冻结后生成的**全新 final3（15篇/173可定位实体）**上，单 v2=**0.821/0.740/0.778**、可选 recall mode=**0.769/0.809/0.789**，均显著优于 dict F1=0.508，且 F1 优于原 BERT 0.733。ID F1 仍为 0.9974，默认 BERT/RBT6/OM 不变；未知主题召回未稳定达到 0.9，不夸大结论（`docs/ner_finetuning.md` §9） |
| 测试 | 历史验收基线为 **302 passed / 4 skipped**。当前候选只在源码冻结与 clean-room RC 两个节点运行必要全量测试；精确 passed/skipped、node IDs、JUnit 和 skip reasons 由随包 `FINAL_STATUS.json`/verification receipt 绑定。 |
| 创新发布验收 | 真实 Nexent CoreAgent + DataMate runtime 的 `demo-all --agent --offline` 已在远端完成；CMeEE schema-v3 OM、pyACL、官方质量、三次性能与同步能耗已验收。DataMate Mapper 仍是逐样本派发，不能把独立 `execute_batch` 成绩称为平台原生 map_batches。 |

| 赛题任务 | 对应实现 | 一键体验 |
|---|---|---|
| 任务一：数据处理智能体 | `DataProcessingAgent` + 医疗文本 ETL 流水线（复用 DataMate 内置清洗/脱敏/过滤算子） | `.\run.ps1 task1` |
| 任务二：知识图谱生成与问答智能体 | `KnowledgeGraphAgent` + 实体/关系/三元组三算子 + 图谱问答 | `.\run.ps1 task2 --eval` / `.\run.ps1 task2-qa --question "..."` |
| 任务三：数据分析智能体与 BI 可视化 | `InsightAgent` + NL2SQL + ECharts 图文报告 | `.\run.ps1 task3` |
| 加分：微调 <1B 小模型（任务一） | dict 远程监督弱标注微调；**对外主数字用 OOD/final3**，同分布 0.99 仅证明蒸馏有效 | `python scripts/train_ner.py` / `docs/ner_finetuning.md` |
| 加分：NPU 算子优化（任务二） | 当前 CMeEE FP16 schema-v3 OM 已完成质量/性能/能效验收；历史 INT8 结果继续分栏 | `docs/remote_npu_acceptance_20260725.md` |
| CMeEE 质量演示链路 | `cmeee_om + hybrid RE` 配置变体 | `configs/pipeline_kg_build_cmeee_om.yaml` |
| 端到端演示 | 中文 Web 证据驾驶舱串起数据→图谱→NL2SQL→BI；Notebook 保留为辅助复现 | `.\run.ps1 demo-ui` / `notebooks/dki_end_to_end_demo.ipynb` |

## 1. 快速开始

### 1.1 环境（Windows PowerShell；Linux/macOS 用 `scripts/setup_env.sh` + `./run.sh`）

前置：安装 [uv](https://docs.astral.sh/uv/)（会自动下载 Python 3.11）；本项目与
`nexent/`、`datamate/` 两个开源仓库位于同一父目录（比赛目录布局）。

```powershell
cd dki-agents
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

脚本做三件事：① 创建 Python 3.11 venv；② 完整安装 Nexent SDK（`../nexent/sdk`，官方
支持路径，PyPI `nexent==0.1.2` 兜底）；③ 安装 DataMate 执行器包（`--no-deps` + 精选依赖，
不拉起 Ray/PostgreSQL——本项目用同构语义的本地执行器，详见架构文档）。

> 没有 DataMate 仓库也能跑：`dki/compat/` 提供与 `datamate.core.base_op` 逐条对齐的
> 契约兼容层，自动切换。

### 1.2 一键端到端（无需任何 LLM 配置）

```powershell
.\run.ps1 demo-all
```

依次执行：生成合成病历数据集（200 份，含噪声与金标准）→ 任务一 ETL 清洗 → 任务二
图谱构建 + 抽取质量评测（vs 金标准 P/R/F1）→ 图谱问答演示 → 任务三建库 + BI 报告
生成 → NL2SQL 准确率评测。产物在 `outputs/`：

```
outputs/
├── processed/manifest.json   # 任务一：hash 绑定的下游输入契约
├── processed/runs/<task>/    # 任务一：清洗文本与结构化表快照
├── kg/medical_kg.json        # 任务二：知识图谱
├── kg/medical_kg.provenance.json  # 任务二：绑定任务一 manifest 与图谱 hash
├── insight.db                # 任务三：分析数据库（SQLite）
├── reports/insight_report.html  # 任务三：BI 图文报告（浏览器打开）
├── tasks/task_id.json        # 每次流水线的任务报告（逐算子指标）
└── eval/                     # 抽取质量 / NL2SQL 评测明细
```

### 1.2.1 浏览器演示驾驶舱

安装可选演示依赖后，可在浏览器内按“数据 → 知识 → 洞察”逐页展示三个任务：

```powershell
.\run.ps1 demo-ui --host 0.0.0.0 --port 7860
```

Linux/macOS 使用 `./run.sh demo-ui --host 0.0.0.0 --port 7860`。每次点击“运行一次完整闭环”
都会创建新的 `outputs/demo_runs/<server-rid>/runs/<run-id>/`，写入 `started.json`、事件流和
完成/失败回执；运行目录不覆盖历史记录，也不会进入提交包。页面中的 CMeEE、CMeIE、NL2SQL
和 NPU 数字均来自公开摘要或哈希绑定验收文档，并标明数据角色。

### 1.3 智能体模式（需要任意 OpenAI 兼容 LLM）

```powershell
Copy-Item .env.example .env   # 填入 LLM_API_KEY / LLM_MODEL_NAME / LLM_API_URL
.\run.ps1 task1 --agent       # 数据处理智能体：自主侦察数据→选算子→编排→执行→检查→汇报
.\run.ps1 agent --which kg --task "基于任务一 manifest 构建医疗知识图谱并总结规模"
.\run.ps1 task3 --agent       # 分析智能体：图谱感知→NL2SQL→指标→图文报告
.\run.ps1 demo-all --agent --offline  # 无 LLM 凭据的确定性工具调用演示
```

不配 LLM 时，上述任务均有确定性直跑模式（去掉 `--agent`）；智能体 ReAct 循环本身可用
`ScriptedModel`（预置响应）离线验证：`python -m pytest tests/test_agents.py`。
`demo-all --agent --offline` 会由真实 CoreAgent 提交异步任务、轮询状态，并通过一次可恢复的
瞬时故障展示 `recovery_events`，不是预先拼接的状态文本。

## 2. 三个任务的实现要点

### 任务一：数据处理智能体（30 分）
- **算子编排**：`MedicalTextCleaner` 组合复用 DataMate 内置算子（HTML 清理/全角转换/
  不可见字符/多余空格/繁转简/手机号+身份证脱敏），加 DataMate 内置长度过滤算子构成 ETL；
- **任务理解与规划**：智能体先 `list_operators`+`preview_data` 侦察，再产出流水线 JSON
  提交 `run_pipeline`（与 DataMate `process.yaml` 同构）；
- **状态跟踪**：`run_pipeline` 异步返回 `task_id`，`get_task_status(wait_seconds=...)` 轮询 PENDING/RUNNING/终态与逐算子指标；
- **异常恢复**：typed failure 已接入 Runner，记录 `failure_category/recovery_action`；失败重提可携带 `parent_task_id` 形成可审计链路。

### 任务二：知识图谱生成与问答（40 分 + NPU 10 分）
- **本体**：6 实体类型 × 7 关系类型（含定义域/值域约束），见 `dki/kg/schema.py`；
- **三个核心算子**（DataMate 规范包，可打包上传算子市场 `scripts/package_operators.py`）：
  `MedicalEntityExtractor`（复用 DataMate AC 自动机 + 别名归一，LLM/BERT 后端可插拔）、
  `MedicalRelationExtractor`（触发词模式 + 章节先验共现，双级置信度）、
  `TripleBuilderValidator`（Schema 校验 + 去重合并 + 置信度过滤，输出校验统计）；
- **关系模型后端**：`MedicalRelationExtractor` 支持 `rule | cmeie | hybrid`，句内真实 mention marker、方向/类型约束及 hash 绑定阈值；联合抽取候选只有在 calibration、outer、官方 dev 三层门全部通过后才可由显式配置启用，checkpoint 不进入提交包；未通过时不替换规则默认路径；
- **图谱聚合**：跨文档 noisy-or 置信度合并 + 证据句留存（200 份病历 → 295 实体 / 479 三元组 / 7 类关系全覆盖）；
- **质量评测**：对 1580 条金标准三元组微平均 **F1=99.97%**（P=100%，R=99.94%）——
  `.\run.ps1 task2 --eval`，按关系类型细分见 `outputs/eval/kg_extraction_eval.json`
  （注：抽取器与语料同源于一部医疗词典，该指标验证的是**流水线在受控语料上的正确性**，
  开放语料泛化依赖 LLM/BERT 后端）；
- **图谱问答**：模板问答引擎（9 类问法，无 LLM 可用，回答带证据句与置信度）+ 智能体自由问答（LLM 模式）。

### 任务三：数据分析与 BI（30 分）
- **复用前序产出**：SQLite 只装载任务一 `outputs/processed/manifest.json` 指向的结构化快照与任务二图谱（kg_entities/kg_triples）；
- **NL2SQL**：确定性规则引擎与有界 `QueryPlan v2` 共享 SQLGlot 安全门，执行单语句、
  只读表白名单和严格输出列契约；LLM 仅在显式 Agent 配置下参与，不作为正式 rule 盲测回退。
  development、known regression 与一次性 prospective holdout 严格分栏；任何已消费集合永久禁用，
  README 不缓存“最新盲测数字”，最终聚合状态由 `FINAL_STATUS.json` 和公开 attestation 提供；
- **BI 报告**：ECharts 单文件 HTML——概览卡片、柱状/折线/饼图、图谱力导向关系图、
  每图附来源 SQL、样本量、数据来源和“相关不等于因果”提示
  （`outputs/reports/insight_report.html`）。

## 3. 更多文档

- [docs/architecture.md](docs/architecture.md)：架构设计、与 Nexent/DataMate 的逐条集成点
- [docs/npu_optimization.md](docs/npu_optimization.md)：NPU 优化思路、CPU/NPU 对比与复跑手册
- [docs/remote_npu_acceptance_20260725.md](docs/remote_npu_acceptance_20260725.md)：当前 CMeEE OM 的真实目标机质量、性能与能效证据
- [docs/scoring_map.md](docs/scoring_map.md)：赛题评分项 ↔ 实现证据映射
- [docs/model_card.md](docs/model_card.md)：三条模型链的指标口径、用途、风险与 NPU 绑定状态
- [docs/data_card.md](docs/data_card.md)：合成、CBLUE、NL2SQL 与盲测数据的血缘和分发边界
- [CONTRACTS.md](CONTRACTS.md)：模块接口契约（工程协作规范）

## 4. 测试与复现

```powershell
$env:PYTHONUTF8=1
& .venv\Scripts\python.exe -m pytest -q          # 全量测试（以收集结果为准）
& .venv\Scripts\python.exe scripts\verify_innovation_release.py `
    --output outputs\verification\innovation_release_cpu.json
.\run.ps1 bench-ops                              # 算子吞吐/延迟基准（CPU）

# 新 NL2SQL 独立集使用不可复用的 <RID>。先把四表只读库、schema、
# 权威公开来源的聚合审计和策展契约交给隔离策展者；pack 不含本项目引擎。
& .venv\Scripts\python.exe scripts\build_nl2sql_curator_pack.py `
    --output-dir outputs\private\<RID>\nl2sql_blind_v4\curator_pack `
    --db outputs\remote_runs\<LINEAGE_RID>\lineage\insight.db `
    --task1-manifest outputs\remote_runs\<LINEAGE_RID>\lineage\processed\manifest.json `
    --kg outputs\remote_runs\<LINEAGE_RID>\lineage\kg\medical_kg.json `
    --source-audit <EHRSQL_AGGREGATE_AUDIT.json> `
    --source-audit <CSS_AGGREGATE_AUDIT.json> `
    --source-tree-sha256 <INCLUDED_TREE_SHA256>

# 隔离策展者返回严格 300 题 corpus.jsonl 和独立复核 attestation 后，
# 才按匿名 SQL AST 整组冻结；holdout.jsonl 始终留在 outputs/private。
& .venv\Scripts\python.exe scripts\freeze_nl2sql_blind.py `
    --input outputs\private\<RID>\nl2sql_blind_v4\corpus.jsonl `
    --output outputs\private\<RID>\nl2sql_blind_v4\frozen\manifest.json `
    --db outputs\remote_runs\<LINEAGE_RID>\lineage\insight.db `
    --task1-manifest outputs\remote_runs\<LINEAGE_RID>\lineage\processed\manifest.json `
    --kg outputs\remote_runs\<LINEAGE_RID>\lineage\kg\medical_kg.json `
    --curator-pack-manifest outputs\private\<RID>\nl2sql_blind_v4\curator_pack\pack_manifest.json `
    --curator-attestation outputs\private\<RID>\nl2sql_blind_v4\curator_attestation.json `
    --source-version <INCLUDED_TREE_SHA256>

# development 达标且源码冻结后，锁定完整 Python 环境并生成候选 seal。
& .venv\Scripts\python.exe scripts\freeze_python_environment.py `
    --output outputs\remote_runs\<RID>\nl2sql\python_environment_lock.json `
    --source-tree-sha256 <INCLUDED_TREE_SHA256>
& .venv\Scripts\python.exe scripts\seal_nl2sql_blind_candidate.py `
    --output outputs\remote_runs\<RID>\nl2sql\candidate_seal.json `
    --manifest outputs\private\<RID>\nl2sql_blind_v4\frozen\manifest.json `
    --task1-manifest outputs\remote_runs\<LINEAGE_RID>\lineage\processed\manifest.json `
    --kg outputs\remote_runs\<LINEAGE_RID>\lineage\kg\medical_kg.json `
    --db outputs\remote_runs\<LINEAGE_RID>\lineage\insight.db `
    --source-tree-manifest outputs\remote_runs\<RID>\source\included_tree.json `
    --dependency-lock outputs\remote_runs\<RID>\nl2sql\python_environment_lock.json

# 下面命令会在打开 holdout 前原子创建 consumed-started.json；无论成功、
# 崩溃或指标失败都不得重跑同一集合。逐题详情仅写 private，公开文件仅含聚合。
& .venv\Scripts\python.exe scripts\eval_nl2sql_blind.py `
    --manifest outputs\private\<RID>\nl2sql_blind_v4\frozen\manifest.json `
    --candidate-seal outputs\remote_runs\<RID>\nl2sql\candidate_seal.json `
    --db outputs\remote_runs\<LINEAGE_RID>\lineage\insight.db `
    --task1-manifest outputs\remote_runs\<LINEAGE_RID>\lineage\processed\manifest.json `
    --kg outputs\remote_runs\<LINEAGE_RID>\lineage\kg\medical_kg.json `
    --private-output outputs\private\<RID>\nl2sql_blind_v4\evaluation_details.json `
    --public-output outputs\remote_runs\<RID>\nl2sql\nl2sql_blind_attestation.json
```

数据为确定性种子合成（`--seed 42`），全流程不依赖网络（智能体在线模式除外）。

## 5. 证据与交付（typed claims + clean-room 提交包）

```powershell
# 生成 validator-passing typed claims 并校验（headline 指标带工件 SHA-256）
& .venv\Scripts\python.exe scripts\build_evidence_claims.py
& .venv\Scripts\python.exe scripts\validate_evidence.py outputs\claims\headline_claims.json
# 证据托管清单（内容指纹 + provenance + bootstrap CI + HMAC 签名/时间戳）
$env:DKI_EVIDENCE_HMAC_KEY="<你的密钥>"; & .venv\Scripts\python.exe scripts\build_evidence_custody.py
# 打包 6 个算子市场 ZIP（含 OM shadow 候选）
& .venv\Scripts\python.exe scripts\package_operators.py
# 构建 clean-room 提交包（allowlist + 密钥扫描 + hash manifest）并干净目录复验
& .venv\Scripts\python.exe scripts\build_submission_bundle.py
& .venv\Scripts\python.exe scripts\verify_submission_bundle.py `
    --bundle <新生成的 clean-room ZIP> --python .venv\Scripts\python.exe `
    --agent-deps .deps
```

- 真实 DataMate 平台 smoke：`python scripts\datamate_smoke.py`（upload/import/instantiate/invoke/故障隔离）。
- NPU/Release A/Release B 的目标机验收口径与诚实边界见
  [docs/npu_optimization.md](docs/npu_optimization.md) §5.9–§5.12 与
  `outputs/verification/phase2_release_a_status_20260720.md`、`phase4_release_b_status_20260720.md`。
- 开源许可证见 [LICENSE](LICENSE)；直接依赖、模型、数据与 NPU 工具链许可清单见
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。ECharts 已随报告内联并保留本地许可说明。
