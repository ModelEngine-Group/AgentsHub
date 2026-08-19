# 数据卡

## 1. 数据资产总览

| 数据资产 | 用途 | 是否进入提交包 | 关键约束 |
|---|---|---|---|
| `data/raw`、`data/gold` | 确定性合成病历、结构化表和受控金标准 | 是 | seed 42，可用于流水线正确性与演示，不代表真实开放域质量 |
| `data/nl2sql_eval*.jsonl` | NL2SQL 开发与回归 | 是 | 已被开发过程使用，不得称为新盲测 |
| `data/cblue` | CMeEE/CMeIE 训练与评估 | 否 | 第三方研究数据，仅在授权环境本地使用 |
| `data/ood` | 历史 OOD 开发诊断 | 部分 | 非一次性盲测，可用于回归但不能替代独立验收 |
| `data/ood_blind`、`data/ood_train*` | 一次性盲测或训练隔离数据 | 否 | 物理隔离，禁止随包分发或用于后验调参 |
| `outputs/processed/runs/<task_id>` | 任务一发布快照 | 运行时生成 | 只能由 `outputs/processed/manifest.json` 引用 |

## 2. 合成医疗数据

- 生成方式：项目内确定性词典和模板，默认 seed 42。
- 内容：患者、就诊、诊断、检查、用药、治疗及相应文本表达。
- 用途：ETL、算子契约、图谱闭环、问答和 BI 的可复现演示。
- 已知偏差：实体表面词、句式、关系和分布与规则/词典高度同源，因此高 F1 只证明受控流水线正确性。
- 隐私：数据为合成数据，不应混入真实姓名、证件号、手机号或真实病历片段。

## 3. CBLUE

- CMeEE-V2：train 15,000，dev 5,000，九类实体。
- CMeIE-V2：train 14,339，dev 3,585，医疗关系抽取。
- 用途：研究和竞赛质量评估。
- 处理：`scripts/download_cblue.py` 下载后规范化为 JSONL。
- 分发：原始或规范化数据均不进入 Git、远端源码包或提交 ZIP。
- 选择纪律：官方 train 可用于训练和内部切分；官方 dev 只作锁定报告，不用于阈值、checkpoint 或规则选择。

详细来源与协议说明见 [cblue_datacard.md](cblue_datacard.md)。

## 4. NL2SQL 数据纪律

现有精选集、分层集和 hard 集均已参与开发或回归，只能报告为开发结果。新的独立评测必须：

1. 在任何新一轮 NL2SQL 修改前准备不少于 120 题；
2. 使用 `sqlglot` 规范化 SQL AST 并按 AST 组划分，防止模板同构泄漏；
3. 由 `scripts/freeze_nl2sql_blind.py` create-once 冻结 holdout、development 和哈希清单；
4. 不把 holdout 的 SQL、执行结果或错误样本用于开发；
5. 用 `scripts/eval_nl2sql_blind.py` 一次性报告执行准确率、hard 准确率、覆盖率和拒答率；
6. 只有整数判定 `100 * correct > 85 * n` 且 `100 * hard_correct >= 70 * hard_n` 才通过门禁。

冻结脚本本身不会凭空创造新问题。输入语料必须来自尚未用于本项目开发、选择或回归的独立来源。
首次评估后会在冻结目录写入 create-once `evaluation_receipt.json`；存在该回执或既有输出时，
脚本拒绝重跑，避免用同一 holdout 反复调参后仍标为盲测。

已消费或拒绝的历史集合均为不可变审计记录，永久禁止重跑：v21 已合法消费但失败；v22 在 marker 前因 bounded-execution 预检失败而拒绝；后续 v23–v27 的选择/消费状态由各自 create-once receipt 记录，不能再宣称新的 prospective 盲测。

v26 已一次性消费，overall 为 63/120、hard+ 为 46/102，两个质量门均失败。v28 也已一次性消费，overall 为 89/120（74.17%，失败），hard+ 为 67/88（76.14%，通过）。v28 的 holdout 后来被其他开发会话查看过，因此整套集合已经是 selection-used 历史，除合法的 development 回归外不得再用于候选选择、重跑或盲测声明。

下一轮必须使用全新隔离策展的 300 题交付，冻结为 development 180 / holdout 120；normalized exact、字符近重复与 AST skeleton family 均不得跨 split。实现线程只能读取 development，holdout 只能在源码、依赖、数据库、evaluator 和 candidate seal 全部固定后消费一次。公开文件只含计数、聚合指标、覆盖率、哈希与通过状态；问题、gold SQL、预测 SQL 和行级错误永远保留在 private 范围。最终生效版本由随 RC 的 `FINAL_STATUS.json` 指向，不在本数据卡中硬编码为“最新”。

## 5. 产物血缘

任务一发布 `outputs/processed/manifest.json`，记录：

- 清洗文本和结构化表的相对路径、字节数和 SHA-256；
- 原始来源、schema、任务报告及 `task_id`；
- create-once 的运行快照目录。

任务二图谱旁必须存在 `medical_kg.provenance.json`，将图谱 SHA-256 绑定到任务一 manifest。任务三数据库同时记录任务一 manifest 和任务二图谱哈希。任一上游工件变化后，下游读取会失败关闭并要求重建。

## 6. 隐私、安全与人工复核

- 清洗器提供手机号、身份证号等模式脱敏，但不能视为完整匿名化方案。
- 禁止将真实个人健康信息提交到公共 LLM、Git 仓库、提交 ZIP 或公开演示环境。
- 对真实数据应另行执行授权、最小化、访问控制、审计、保留期限和删除流程。
- 低置信实体/关系只进入候选或人工复核队列，不应自动形成临床结论。

## 7. 局限

- 合成数据覆盖有限，不能估计真实世界偏差、噪声和罕见表达。
- CBLUE 的领域、年代和标注规则不覆盖所有医院与临床场景。
- AST 分组只能降低模板泄漏，不能保证问题语义完全独立。
- 数据质量结论均为相关性和工程诊断，不构成因果或临床有效性证明。
