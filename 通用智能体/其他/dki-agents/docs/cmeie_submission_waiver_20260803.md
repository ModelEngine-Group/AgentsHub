# CMeIE 提交放行决定（2026-08-03）

## 1. 决定

项目所有者于 2026-08-03 明确决定：CMeIE 完整 text-to-SPO 的内部质量门不再阻塞本轮比赛提交。

该决定的发布状态为 `WAIVED_FOR_SUBMISSION`，只表示允许继续执行 NL2SQL 独立盲测、clean-room 验证和提交准备；它不表示 CMeIE 指标达到原内部门槛，也不改写任何历史训练、选择或阻塞回执。

## 2. 放行时的真实指标

| 口径 | 结果 | 数据角色 |
|---|---:|---|
| 当前产品链完整 text-to-SPO | F1 0.1911，P/R=0.2497/0.1548 | 历史官方 dev 开发结果 |
| 当前最强完整 text-to-SPO | F1 0.322919，P/R=0.303689/0.344749 | selection-used calibration |
| gold-subject SPO | F1 0.406325 | 条件化诊断，不是完整链 |
| gold-pair 谓词分类 | F1 0.6500 | 条件化诊断，不是完整链 |
| PRGC seed 42 / 43 | F1 0.218947 / 0.223683 | selection-used calibration |

原内部晋级线为 calibration F1≥0.45，最终质量目标为完整 text-to-SPO F1≥0.50。上述结果没有达到该内部质量目标；outer 和官方 dev 开发确认均未因本次放行而补跑或读取。

## 3. 技术路线完成情况

已按预声明范围完成 predicted-NER pair classifier、gold-pair/gold-NER 分解、阈值与集成、CMeEE NER 复用、gazetteer、11 类型 argument NER、GPLinker、predicate-span、CasRel、稳定 top-2 域适配和 PRGC fallback。当前阻塞点已经定位为 predicate-object 表示及跨 article-family 泛化，不是标签、offset、梯度链或单纯阈值故障。

不再为本轮提交继续进行 seed 扫描、逐谓词阈值、全 encoder 解冻、OOF、扩大 gazetteer 或读取 outer/官方 dev。默认离线 rule 产品路径保持不变，未达标联合模型不自动进入默认 KG 写入链。

## 4. 提交与报告口径

- 对外必须写“CMeIE 内部质量门未通过，经项目所有者明确豁免后允许提交”。
- 不得写“CMeIE 已达标”“CMeIE F1≥0.50”或把 gold-subject/gold-pair 结果冒充完整 text-to-SPO。
- 最终状态若其余门全部通过，应写 `READY_FOR_SUBMISSION_WITH_CMEIE_WAIVER`，不得写成“双严格门槛全部通过”。
- NL2SQL prospective independent 门、远端全仓测试、private-data/secret 检查和 clean-room 验证仍是不可豁免的提交条件。
- 本决定不修改 `docs/remote_npu_acceptance_20260725.md`、`outputs/remote_runs/plan_v1/submission/verification.json` 或任何历史模型与评估产物。
