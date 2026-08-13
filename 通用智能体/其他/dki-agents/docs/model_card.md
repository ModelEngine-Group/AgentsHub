# 模型卡

## 1. 范围与状态

本项目包含三条相互独立的医疗信息抽取模型链。指标和 NPU 工件不得跨链合并：

| 模型链 | 用途 | 当前可引用结果 | 发布状态 |
|---|---|---|---|
| `medical_ner_bert` | 合成语料上的 13 标签 token classification | 同分布实体 F1 0.9936；历史 FP16 OM 372.799 docs/s、200/200 文档一致 | 历史验收基线，不代表 CMeEE 泛化质量 |
| `medical_ner_cmeee` | CMeEE-V2 九类嵌套实体，RoBERTa/BERT + GlobalPointer | 冻结均衡基线 F1 0.6031；OM 绑定候选官方 dev F1 0.613020，P=0.583679，R=0.645468 | schema-v3 FP16 OM 已验收；弱类型回退风险使其不自动替换均衡基线 |
| `medical_re_cmeie` | CMeIE-V2 实体对谓词分类与联合 SPO 实验 | gold-pair F1 0.6500；现有产品链官方 dev 完整 text-to-SPO F1 0.1911；CasRel selection-used calibration F1 0.322919、PRGC seed 42/43 为 0.218947/0.223683，均为已拒绝历史路线 | 当前候选只有通过 calibration≥0.45、三 seed outer 与官方 dev≥0.50 才能发布；历史 waiver 不再满足本轮比赛闭环门，默认规则路径保持不变 |

本轮在真实 Ascend 目标机完成了 CMeIE 训练/端到端评测及 CMeEE ONNX→ATC→OM→pyACL 质量、性能、能效闭环。所有失败尝试保留独立 create-once 证据，不覆盖旧基线。

## 2. 预期用途

- 从中文医疗文本中抽取疾病、症状、药物、检查、治疗、科室等实体。
- 在句内、类型兼容的实体 mention 之间抽取受 schema 约束的关系。
- 为知识图谱构建、带证据问答和聚合分析提供候选事实。
- 用于竞赛演示、研究和人工复核辅助。

不适用于临床诊断、治疗建议、自动处方、急救分诊、个体风险预测或无人工复核的医疗决策。

## 3. 输入与输出

- 输入：UTF-8 中文文本。训练和部署默认最大长度由 checkpoint 的 `globalpointer_config.json` 或 `relation_config.json` 决定。
- NER 输出：实体类型、文本、字符级半开区间 `[start, end)`、置信度和来源。
- RE 输出：主语、谓词、宾语、mention 位置、置信度和来源。
- 图谱晋级：仍需 `TripleBuilderValidator` 做类型、关系、置信度、去重和证据校验。

## 4. 训练与选择纪律

### CMeEE NER

- 数据：CBLUE CMeEE-V2 train 15,000 / dev 5,000。
- 模型与阈值只在官方 train 的 text-disjoint 内部分割上选择。
- 官方 dev 为锁定报告集，不参与 checkpoint、阈值或融合策略选择。
- 九类标签：`dis/sym/dru/equ/pro/bod/ite/mic/dep`。

### CMeIE RE

- 数据：CBLUE CMeIE-V2 train 14,339 / dev 3,585。
- 训练和正式算子共享 `dki/nlp/cmeie_pair.py` 的 mention marker 与类型签名。
- 候选仅在同一句内生成，并受方向、实体类型和 fit-only 关系 schema 约束。
- gold-pair 分类结果不能表述为端到端 SPO 质量。
- **端到端 text-to-SPO 分解（2026-08-01 官方 dev 全量复现）**：gold-NER 条件下实体检测 100% 正确（mention F1=1.0），但端到端 SPO 仅 F1 0.3683（P 0.4929 / R 0.2940，tp=4791 fp=4929 fn=11506）；**最大损失层为谓词分类/解码**——旧阈值映射下 44 个谓词仅 6 个被激活（临床表现/药物治疗/相关(导致)/同义词/辅助检查/多发群体），38 个长尾谓词零激活（模型对 gold 实体对的最大 sigmoid logit 也普遍 <0.3）。
- **逐谓词阈值校准修复（最终版）**：在 train-internal 上用坐标上升（`coordinate_micro_f1_per_predicate`）重校准 44 个谓词阈值，0.05 步长网格（0.2~0.95），官方 dev 端到端 SPO **F1 0.3683 → 0.4319**（P 0.4268 / R 0.4371，tp=7123），**激活谓词 6 → 19**（新增发病部位/影像学检查/手术治疗/放射治疗/病因/辅助治疗/阶段/预后状况/预后生存率/预防/高危因素等）。校准文件：`outputs/remote_runs/plan_v1/cmeie_predicate_threshold_calibration_h05_s43.json`（`official_dev_used_for_selection: false`，官方 dev 仅用于报告）。多 seed 集成（s43+bal_s42）与温度缩放均未超过该值，判定为 <1B 模型下的实测上限；仍未过 0.50 晋级线。
- **CasRel overfit 门（8 行，seed 42，hnr=8 + 表面文本自指过滤）实测 F1 0.9362（TP=44/FP=6/FN=0）**，为 head-only 架构在受控数据上的上限；joint 阶段（含 GPLinker 式一体化循环）在 8 行数据上所有配置梯度爆炸（encoder 梯度 16 万~40 万，joint F1 归零），已如实保留全部 create-once 证据。
- **逐谓词阈值校准结论（2026-08-01 远端实测）**：剩余 6 个 FP 全部集中在 `病理分型` 谓词（相似 subject 互相激活）。该谓词 gold object 分数跨 -0.602~+8.171，wrong object 最高 +6.095，gold 中 5 个 TP（+0.465/+0.466/+0.780/+0.797/+0.807 等）低于 wrong 峰值，gold/wrong 分数重叠（gap=-6.697），**单谓词抬高阈值必然丢失真阳性，解码层阈值/排序无法分离该类错误**——该结论同时覆盖全局阈值（0.5/±网格）与 per-predicate 阈值两种路线，均不可行。
- **显式 subject-object 交互 + 零初始化 residual context adapter（2026-08-02，真实 Ascend，seed 42）**：10 项远端单测全通过；8 行 / 44 SPO 的 1200 步 overfit 达到 P=R=F1=1.0（TP=44、FP=0、FN=0），说明头部表示与梯度链正确。正式 fit→calibration、encoder 全程冻结的 4 epoch 最佳为第 3 轮 **F1 0.299228**（P=0.275763、R=0.327057、TP=2015、FP=5292、FN=4146），较无 adapter 的 0.263861 提升 0.035367，但低于 0.45 晋级线；按协议未读取 outer 或官方 dev。证据：`outputs/remote_runs/20260802T061227Z-g30fce24-tca6d0807c94b/evidence/cmeie/adapter_frozen_formal_seed42_e4/selection.json`。
- **顶层 2 层保守 Adam 域适配（2026-08-02，真实 Ascend，seed 42）**：默认 `eps=1e-8` 时联合 overfit 从 F1 1.0 崩到 0.03019、encoder 预裁剪梯度 207,058.5。梯度拆解发现 object residual / cosine interaction 范数分别为 7.6556 / 5.1804，但相加仅 0.01470；改为 encoder `eps=1e-4`、无 weight decay 后，联合 overfit 保持 F1 1.0、encoder norm 降至 9.044。正式 4 head-only + 2 top-2 epoch 最佳 **calibration F1 0.322919**（P=0.303689、R=0.344749、TP=2124、FP=4870、FN=4037），0 non-finite、0 skipped step；仍低于 0.45，未读取 outer/官方 dev。gold-subject 分解 F1 0.406325，subject span F1 0.698527，说明 predicate-object 仍是主瓶颈。证据：`outputs/remote_runs/20260802T065042Z-g30fce24-tb3c21e75e9b0/evidence/cmeie/`。
- **PRGC fallback 最终门控（2026-08-02/03，真实 Ascend）**：query-v2 + nearest decode 在固定 8 行 / 44 SPO 上达到 44/44，证明标签、梯度和解码实现可工作；正式 calibration 的 seed 42 为 P/R/F1=0.198056/0.244765/0.218947，seed 43 为 0.199821/0.254017/0.223683，均无 non-finite 或 skipped step，但都低于 0.45。依计划未训练 seed 44、未读取 outer/官方 dev，并写入 `BLOCKED_MODEL_QUALITY` 回执。该结果说明当前失败点是跨 family 泛化而非实现链路或数值稳定性。
- **历史提交豁免（2026-08-03，已被本轮严格门取代）**：旧阶段曾记录 `WAIVED_FOR_SUBMISSION`，但它从未代表 F1≥0.50。本轮 `READY_FOR_SUBMISSION` 不接受 waiver；只有新的 calibration、三 seed outer 和官方 dev 门全部通过才可放行。旧回执继续作为不可变历史保留。

### 合成 BERT NER

- 标签来自项目词典的远程监督，适合验证受控流水线与产品 schema。
- 高同分布分数不等价于真实开放域医疗文本泛化能力。

## 5. NPU 工件契约

历史 372.799 docs/s 的 OM 属于 `medical_ner_bert`，不能绑定或归因给 CMeEE。当前 CMeEE schema-v3 FP16 OM 的正式三次中位数为 326.098 docs/s，CPU ONNX 为 18.393 docs/s，加速 17.7295 倍；官方 dev OM/ONNX F1 差 0.000194。

新的 CMeEE 发布链必须依次生成：

1. checkpoint 目录确定性树哈希；
2. ONNX 与 `cmeee_globalpointer_onnx_export` 回执；
3. ATC OM 与完整日志；
4. 真实 ACL IO 检查生成的 schema-v1 清单；
5. `bind_om_checkpoint.py` 生成的 schema-v3 清单。

schema-v3 清单会同时校验 checkpoint、ONNX 回执、OM、tokenizer、labels、ATC 日志、模型族和 IO 签名。缺少任一工件时不得加载或晋级。

当前证据见 `docs/remote_npu_acceptance_20260725.md`。NPU 串行与微批 200/200 文档完全一致；FP16 OM 与 FP32 ONNX 在 200 EMR 上实体 F1 0.997604，官方 CMeEE dev 的主指标差异用于最终质量门。

## 6. 局限与风险

- CMeEE 当前 F1 仍低于稳健生产使用要求，`ite`、`sym`、长实体和嵌套边界是主要误差源。
- CMeIE 历史端到端路线未达到内部质量目标。当前新候选必须先在 calibration 达到 0.45，才允许按协议继续 seed/outer/官方 dev；未过门时必须停止并保留失败证据。gold-subject、候选 oracle、受控 overfit 或历史 waiver 都不能替代完整 text-to-SPO 指标。
- 训练数据主要是中文医疗文本，不能假设对其他语言、医院模板或人群具有同等表现。
- 置信度不是临床概率，阈值变化会显著改变召回和误报。
- 输入可能含个人健康信息；模型目录和提交包均不应包含真实病历或可识别信息。

## 7. 复现入口

- NER 训练：`scripts/train_ner_cmeee.py`
- NER 冻结复评：`scripts/eval_ner_cmeee_checkpoint.py`
- RE 训练：`scripts/train_re_cmeie.py`
- RE 端到端评估：`scripts/eval_re_cmeie_e2e.py`
- CMeEE ONNX/OM 证据链：`scripts/export_cmeee_globalpointer_onnx.py`、`scripts/compile_cmeee_om.py`、`scripts/create_om_base_manifest.py`、`scripts/bind_om_checkpoint.py`
- 同 checkpoint CPU/NPU/能效验收：`scripts/run_npu_energy_acceptance.py`

数据来源与许可见 [data_card.md](data_card.md) 和 [cblue_datacard.md](cblue_datacard.md)。
