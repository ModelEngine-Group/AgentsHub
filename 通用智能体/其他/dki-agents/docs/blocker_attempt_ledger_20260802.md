# Codex / Claude Code / AtomCode 双卡点联合审计与尝试台账（2026-08-02）

## 1. 审计范围与口径

本台账以当前工作树、Claude Code 的计划/对话/项目记忆、AtomCode 的最新 session snapshot、Codex/DeepSeek 的 RID 证据为输入。旧会话不整体恢复；同一指标只在数据角色一致时比较。

- `official dev`：历史已多次查看，只能称“开发确认”，不能称盲测。
- `development-selection-used` / `calibration`：允许用于选模和修规则，不能证明独立泛化。
- `overfit`：只证明实现和梯度链能工作，不能证明泛化。
- `prospective independent holdout`：在源码/依赖/数据库/evaluator 密封后一次性消费，才可用于 NL2SQL 独立门槛。
- 所有本轮 NPU 训练均固定远端 Python `/root/dki-agents-resume-20260723/venv/bin/python`、逻辑设备 0，并写入唯一 RID。

## 2. 当前结论

| 卡点 | 当前最强合规结果 | 门槛 | 判定 |
|---|---:|---:|---|
| CMeIE 完整 text-to-SPO | 现有产品链官方 dev F1 0.1911；当前最强完整链为 selection-used CasRel top-2 calibration F1 0.322919；PRGC seed 42/43 为 0.218947/0.223683 | 原内部门：calibration≥0.45 后才可进入 outer，最终≥0.50 | 技术判定仍为 **`BLOCKED_MODEL_QUALITY`**；2026-08-03 项目所有者决定 **`WAIVED_FOR_SUBMISSION`**，不补跑 outer/dev、不声称指标通过 |
| NL2SQL 引擎开发门 | v23 只读 development 180/180=100%；hard/extra/extra-hard 124/124=100%；coverage=100%；known-120 为 113/120 | development overall≥0.90、hard≥0.78、coverage≥0.95 | **当前候选开发门已解决** |
| NL2SQL 独立盲测门 | v23 已冻结：holdout 120、hard+ 86，exact/near/AST 跨 split 交集均为 0；状态 `frozen-unconsumed`。历史 v21 仅 6/120 且永久禁用 | overall>0.85、hard≥0.70，一次性消费 | **待最终源码/依赖/DB/evaluator 密封后一次性消费 v23** |

因此当前不再以 CMeIE 作为提交阻塞项，但也不能称“双严格门槛全部通过”。CMeIE 已按预声明 fallback 门停在模型质量阻塞，并由项目所有者明确豁免；提交准备状态仍由全新 NL2SQL prospective independent 一次性评测、全仓远端测试和 clean-room 验证决定。CMeEE 的历史 OM/NPU 验收和历史 clean-room ZIP 继续有效，但不能冒充当前工作树的新提交包。

## 3. 三个工具的最新工作对齐

| 工具 | 最新有效贡献 | 需要纠正的口径 |
|---|---|---|
| Claude Code | 补齐 NL2SQL terminal/fail-closed 传播；形成 CMeIE gold-pair、gold-NER、GPLinker、CasRel 等诊断链；记录 full-unfreeze 16 万～40 万级 encoder 梯度 | CasRel 的 0.89～0.93 多为受控 overfit，不是完整 text-to-SPO 泛化成绩 |
| AtomCode | HNR 2→8、表面自指过滤把 8 行 CasRel overfit 提到 F1 0.9362；定位剩余 6 个 `病理分型` 交叉激活 FP；验证 mean pool、HNR16、seed43、joint unfreeze 均无益 | 旧 v4b 180 题已经参与开发，不能标成 independent |
| Codex / DeepSeek 执行链 | 建立不可变 RID、隔离进程矩阵、候选路由诊断；修复 NL2SQL validator 误拒；实现 CMeIE 显式 subject-object 交互、adapter、顶层解冻稳定性门和真实 NPU 回执 | DeepSeek 的同进程 N1 矩阵存在模块缓存污染，最终结论以 fresh-interpreter matrix 为准 |

## 4. CMeIE：全部主要尝试与结论

| # | 方法 | 数据角色 / 结果 | 结论 |
|---:|---|---|---|
| 1 | 原始 predicted-NER → pair classifier | 完整链 F1 0.1523；校准后官方 dev 0.1911，P/R=0.2497/0.1548 | 当前产品模型基线，远低于 0.50 |
| 2 | gold argument pair 分类 | gold-pair F1 约 0.6500～0.6559 | 证明 RE 在理想候选上可用，但不能代表 text-to-SPO |
| 3 | gold NER 条件化 + 阈值校准 | 0.3683 → 0.4319，P/R=0.4268/0.4371；激活谓词 6→19 | NER 不是唯一瓶颈；关系/解码仍不足 |
| 4 | temperature scaling | 未超过 0.4319 | 拒绝；不能恢复缺失谓词 |
| 5 | 复用 CMeEE NER 后端 | 完整链约 0.1299 | 类型体系和任务不匹配 |
| 6 | CMeEE span + gazetteer 类型增强 | 完整链约 0.1162 | 候选噪声抵消类型收益 |
| 7 | train-fit gazetteer 独立候选器 | 100 条 train-internal F1 0.0669；typed mention precision 约 0.3044，候选爆炸 | rejected ablation |
| 8 | pair-gap 扩至 120 | 完整链约 0.1294 | 增加噪声，未改善召回质量 |
| 9 | pair classifier `pos_weight` cap=50 | 条件化/内部 F1 约 0.3800 | 长尾召回改善不足 |
| 10 | all-negative 训练修订 | gold/internal 可到约 0.583；官方端到端约 0.3448，仅 3 个谓词活跃 | 严重谓词塌缩，拒绝 |
| 11 | 多 seed 集成 | full 约 0.3915；300 条切片约 0.3565 | 未超过单模型 gold-NER 校准上限 |
| 12 | GPLinker seed 42/43 | 完整 SPO F1 均为 0 | 按门控停止继续试参 |
| 13 | GPLinker offset/dropped-span/指针修复 | dropped span 3201→0；32 行 overfit F1=1，但正式仍差 | 实现正确，泛化/四头联合匹配失败 |
| 14 | GPLinker 四头诊断 | entity F1 0.5477、rel-head 0.1017、rel-tail 0.1883；四条件匹配 339/7483=4.53% | 关系头/尾稀疏匹配是直接失败点 |
| 15 | predicate-span sparse / negative threshold | F1 0.0273 / 0.0422 / 0.0804 | 稀疏二维指针目标不可用 |
| 16 | predicate-span BCE frozen4 | F1 0.215009 | 优于 sparse，但仍明显不足 |
| 17 | 11 类型 argument GlobalPointer | 最佳 typed mention F1 约 0.6696；另有 seed/配方归零 | 未达到恢复 OOF 所需约 0.70～0.75 与高召回组合 |
| 18 | SoftLexicon / gazetteer 类型特征 | 多个 head-only ablation 无稳定晋级收益 | 不进入默认链 |
| 19 | CasRel 基础训练延长 | 400/1200/3000 步 overfit F1 0.8916/0.9247/0.9032 | 1200 步最佳；再延长出现过拟合 |
| 20 | CasRel HNR=8 + 自指过滤 | 8 行 overfit P/R/F1=0.88/1/0.9362，TP44/FP6/FN0 | 修复 self-loop，但 6 个 `病理分型` 交叉激活仍不可由阈值分离 |
| 21 | mean pool、HNR16、seed43 | 均低于 HNR8 seed42 | rejected ablations |
| 22 | CasRel full encoder unfreeze | overfit F1 归零；encoder gradient 16 万～40 万；正式训练停滞 | 直接解冻不可用 |
| 23 | DeepSeek 临时分类器 C2 | smoke F1 0.1066，实验不完整 | 不可晋级 |
| 24 | 显式 predicate-specific subject-object cosine 交互 | 8 行 / 44 SPO / 1200 步 overfit F1=1.0、FP=0 | 解决局部 pairing 交叉激活，但尚未证明泛化 |
| 25 | 交互层 + 零初始化 residual context adapter | calibration 4 epoch 最佳 F1 0.299228，P/R=0.275763/0.327057；较无 adapter 0.263861 提升 0.035367 | 有效但不足，未过 0.45，outer/official dev 未读取 |
| 26 | 只解冻最高 2 层，AdamW 默认 eps | overfit 联合 F1 0.03019；encoder norm 207,058.5 | 参数范围和低 LR 仍不能阻止崩溃 |
| 27 | loss-path 梯度拆解 | subject norm 0.00034；object residual 7.6556；interaction 5.1804；combined 0.01470 | 两条 object 路径依赖精细梯度抵消；默认 Adam 归一化破坏抵消 |
| 28 | top-2 + encoder Adam `eps=1e-4`、无 weight decay | overfit 联合 F1=1.0、encoder norm 9.044；正式 calibration 最佳 F1 0.322919，P/R=0.303689/0.344749；0 non-finite、0 skipped step | 稳定性根因已修复并带来 +0.023691 泛化增益，但仍未过 0.45 |
| 29 | 选中 checkpoint 的 calibration 分解 | subject span F1 0.698527，P/R=0.655607/0.747461；强制 gold subject 后 SPO F1 0.406325，P/R=0.416980/0.396202 | 剩余主瓶颈仍是 predicate-object 预测；subject recall 也未到恢复 OOF 所需约 0.85 |
| 30 | PRGC query-v2 + nearest 小样本门 | 固定 8 行 / 44 SPO 达到 44/44，dropped=0、无 non-finite/skipped step | PRGC 的标签、梯度与解码实现正确，可进入正式 seed 门 |
| 31 | PRGC 正式 seed 42/43 | seed42 P/R/F1=0.198056/0.244765/0.218947；seed43=0.199821/0.254017/0.223683 | 两 seed 均低于 0.45；按门控停止 seed44，未读取 outer/官方 dev，标记 `BLOCKED_MODEL_QUALITY` |

### CMeIE 当前根因

1. **旧产品链是两级误差传播**：NER 漏检/错边界/错类型先损失召回，pair classifier 再产生谓词混淆；gold-NER 上限约 0.4319 已证明仅换 NER 不足。
2. **GPLinker 是联合匹配过稀**：实体、关系头、关系尾四条件同时命中的比例只有 4.53%，不是阈值微调问题。
3. **CasRel pairing 已能局部解决，但域泛化不足**：显式交互可把 overfit FP 清零；冻结通用 encoder 的 calibration 仅 0.2992。
4. **直接 encoder 解冻的失败来自优化器动力学**：object residual 与 cosine interaction 的独立梯度较大但相加几乎抵消；默认 AdamW 将微小合成梯度按坐标归一化，破坏抵消并引发灾难性梯度。增大 encoder `eps` 后已在 overfit 上把 207k 降到 9.0，并把正式 calibration 从 0.2992 推到 0.3229。
5. **稳定性修复后，质量瓶颈转为目标/表示与跨 family 泛化能力**：gold subject 条件下仍只有 F1 0.4063；PRGC 虽能 44/44 过拟合，两个正式 seed 也仅约 0.22。继续调 NER、阈值、学习率、训练轮数或 seed 没有充分依据。

## 5. NL2SQL：全部主要尝试与结论

| # | 方法 | 数据角色 / 结果 | 结论 |
|---:|---|---|---|
| 1 | 原 44 题回归 | 44/44=100% | 基础规则稳定，但规模太小且已参与开发 |
| 2 | 已知 120 题 | 112/120=93.33% | development-selection-used，不能称盲测 |
| 3 | 旧 19 题难切片 | 11/19=57.89%；hard 7/15=46.67% | 已参与诊断，只能用于回归 |
| 4 | 早期 one-shot 集 | 历史 v3 约 9/120、v9 约 6/120 | 暴露规则覆盖不足；均已失去盲测资格 |
| 5 | v4b 早期编译器 | 初始 1/180=0.56%；简单语义扩展 21/180=11.67% | 证明需按查询结构而非关键词堆模板 |
| 6 | v7～v11 编译器/contract/obligation | 覆盖 join、CTE、KG、grouping、nested、window、temporal、set 等 | 候选 SQL 能力逐步形成 |
| 7 | 旧 v4b 当前回归 | 172/180=95.56%，coverage 96.11% | 成绩高但已 selection-used，不是 independent |
| 8 | 当前严格 180 题开发集 | 历史引擎 56→65→85→102/180 | 102 个已回答样本均严格执行等价，瓶颈表现为 78 次 abstain |
| 9 | semantic tri-state / AST safety / terminal propagation | 安全拒答和终态传播测试补齐 | 安全性改善，但未解释 78 个拒绝 |
| 10 | DeepSeek 同进程 N1 matrix | 输出受 `sys.modules`/模块缓存影响 | 协议无效，不用于结论 |
| 11 | fresh-interpreter lineage matrix | 当前/历史 DB 对旧引擎均为 102/180；数据库无影响 | 排除 DB 血缘为主因 |
| 12 | route diagnostic | 180/180 全部选择 v9；直接 v9 SQL 180/180 严格执行等价；最终仅 102 | 根因锁定为 validator false rejection，不是编译器缺规则 |
| 13 | generic contract / obligation 拆解 | generic 接受 106、拒 74；obligation 接受 172、拒 8 | 通用 validator 与受界编译器语义不一致 |
| 14 | v9 provenance validator + 自验证语义 admission | strict current development 180/180，四档难度全 100%，coverage 100%；107 tests passed | **开发引擎卡点已修复**；显式 output override 仍 fail-closed |
| 15 | prospective blind protocol | freeze、seal、consume-once、private/public receipt 代码已具备 | 协议门已实现，仍需真实独立数据验证泛化 |
| 16 | v21 隔离策展与机器冻结 | 300 题；development 180、holdout 120、hard+ 90；exact/near-duplicate/AST family 交集均为 0 | 数据角色、密封和一次性消费前置条件合规 |
| 17 | v21 development + 已知回归 | development 180/180；known-120 113/120，coverage 94.17%，easy 44/44 | 进入 blind 前开发门与历史回归门通过 |
| 18 | v21 一次性 blind | overall 6/120=5.00%，coverage 36/120；hard+ 6/90=6.67%；`overall_failed+hard_failed` | 合法消费但质量失败；暴露严重独立分布泛化断层，v21 不得重跑或用于调参 |
| 19 | v22 全新隔离策展交付 | 300 题；配额完整；旧 opaque validator 只验声明/哈希，曾返回 `VALIDATED_UNFROZEN`。正式 freezer 在划分前发现至少 1 条 gold SQL 触发 `bounded_execution_interrupted`；未生成 frozen 目录或消费 marker | `REJECTED_PRE_FREEZE`，v22 不修改、不重用；验证器已补成逐条语义 + bounded execution 聚合门，启用全新 create-once v23 交付 |
| 20 | v23 隔离策展、冻结与 development 调优 | 300 题冻结为 development 180 / holdout 120；holdout hard+ 86；三类泄漏交集均为 0。只读 development 从 10/180 改进到 180/180，hard+ 124/124、coverage 100%；known-120 113/120 | development 门通过；holdout 未读取，进入最终源码闭包、依赖锁和 candidate seal 阶段 |

### NL2SQL 已实施的根因修复

- 普通 contract 仍先执行；只有当它拒绝时，才允许真实 v9 编译器重新计算同一候选并校验 provenance。
- 只有候选 SQL 与受界编译器重算完全一致，且执行语义自验证通过时才能放行。
- 显式额外输出列、用户 override、非法表/写 SQL/多语句继续 BLOCKED，避免为了 coverage 放松安全门。
- 隔离矩阵每个 cell 使用一个全新 Python 解释器，排除模块缓存伪差异。

## 6. 远端设备和测试条件

- `npu-smi 25.5.1`；运行时设备名 `Ascend910_9382`，`npu-smi` 产品栏显示 `Ascend910`。
- 每个物理 chip 显示 65,536 MB HBM；本项目固定逻辑设备 0。
- 同机存在他人的 Qwen3-32B 进程 PID 31106，两个 chip 分别占约 30,744 / 32,618 MB。该进程从未被终止。
- 因共址负载，功能/质量结果有效，但本轮训练耗时和即时利用率不能作为无污染性能基准。
- 已验收且本轮未改写的 CMeEE 历史指标：官方 dev FP16 OM F1 0.613020；326.098 docs/s；NPU/CPU 17.7295×。

## 7. 关键不可变证据

- NL2SQL 最终开发矩阵：`outputs/remote_runs/20260802T060311Z-g30fce24-t0cde4e6aaee9/evidence/nl2sql/matrix/lineage_matrix_after_fix.json`
- NL2SQL 聚焦测试 JUnit：`outputs/remote_runs/20260802T060311Z-g30fce24-t0cde4e6aaee9/evidence/nl2sql/tests/pytest.junit.xml`
- CMeIE adapter selection：`outputs/remote_runs/20260802T061227Z-g30fce24-tca6d0807c94b/evidence/cmeie/adapter_frozen_formal_seed42_e4/selection.json`
- CMeIE 默认-eps top-2 拒绝回执：`outputs/remote_runs/20260802T063925Z-g30fce24-t9dfdc091f648/evidence/cmeie/top2_overfit_seed42_s1200_j100/overfit_receipt.json`
- CMeIE 分路径梯度诊断：`outputs/remote_runs/20260802T063925Z-g30fce24-t9dfdc091f648/evidence/cmeie/encoder_gradient_paths/diagnostic.json`
- CMeIE conservative-eps 正式 RID：`outputs/remote_runs/20260802T065042Z-g30fce24-tb3c21e75e9b0/`
- CMeIE conservative-eps selection：`outputs/remote_runs/20260802T065042Z-g30fce24-tb3c21e75e9b0/evidence/cmeie/top2_eps1e4_formal_seed42_e6/selection.json`
- CMeIE calibration 分解：`outputs/remote_runs/20260802T065042Z-g30fce24-tb3c21e75e9b0/evidence/cmeie/calibration_decomposition/diagnostic.json`

## 8. 发布门

只有以下条件同时成立才生成新的 final clean-room RC：

1. CMeIE 技术结论、实际指标和项目所有者 2026-08-03 的 `WAIVED_FOR_SUBMISSION` 决定完整披露；不得声称 F1≥0.50 或改写历史阻塞回执。
2. NL2SQL 新独立 holdout `n≥120`、hard/extra/extra-hard `n≥40`，一次性 overall>0.85、hard≥0.70，hash/consumption receipt 全部有效。
3. CMeEE 历史 NPU 证据哈希不变，全仓远端测试零失败，无新 skip/xpass。
4. clean-room manifest、secret scan、private-data 排除和解压后完整 pytest 全通过。

满足上述条件时，最终状态只能写 `READY_FOR_SUBMISSION_WITH_CMEIE_WAIVER`；若 NL2SQL 或工程门失败，则写 `NOT_READY_FOR_SUBMISSION`。
