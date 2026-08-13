# CMeEE 冲分研究结论与 TL-v4 方案（2026-07-24）

## 1. 现状诊断

| 口径 | F1 | 说明 |
|---|---|---|
| 本仓库已晋升 | **0.5940**（exact seed44） | 冻编码器 + 原版 GlobalPointer |
| TL-v2/v3 扫参 | 0.53–0.59 | seed/inner_dim/macbert/rlarge/解冻均未突破 |
| 公开文献强基线 | **约 0.64–0.68**（RoBERTa±GP/EGP，端到端微调） | 与我们同任务量级 |
| 社区 EGP 纯模型参考 | ~0.679（RoBERTa-large + Efficient GP，另可再融） | 证明架构天花板远高于 0.59 |

**结论：继续盲扫 seed/学习率收益 ≤0.1 点；主瓶颈是方法，不是运气。**

## 2. 根因（按期望收益排序）

1. **解码器仍是原版 GlobalPointer**  
   苏剑林 Efficient GlobalPointer：类型间共享抽取、分离分类，在 CMeEE（9 类嵌套）上原文明示优于原版，且更不易过拟合。

2. **长期冻死编码器**  
   冻编码器是我们唯一稳定不崩的配方，但把上限锁在 ~0.59。文献强结果几乎都是 **端到端微调**（或 RoBERTa-large + BiLSTM + GP）。此前 last-2 崩盘是 **学习率/调度过猛**，不是“不能解冻”。

3. **弱类型拖后腿**  
   验收证据：`ite`≈0.33、`sym`≈0.45，而 `dru/mic/dis` 明显更高。原版 GP 每类独立 Q/K，稀有类更容易欠拟合；EGP + 类型加权更对症。

4. **缺对抗/稳健训练**  
   社区冲榜常用 FGM/PGD；我们尚未接入。

5. **超参扫参已穷尽边际**  
   max_len=256、freeze、8ep、head_lr=5e-4 附近已饱和。

## 3. TL-v4「优秀方案」（不破评分纪律）

约束不变：`<1B`、checkpoint/阈值只在 train-derived internal split 上选、官方 dev 只报一次。

| 阶段 | 内容 | 目标 |
|---|---|---|
| A | 实现 **Efficient GlobalPointer**，先 **freeze 对照** | 验证架构本身能否抬过 0.594 |
| B | EGP + **稳端到端**：head warmup → 全层微调，`lr_enc≈2e-5`，`joint_encoder_eval=false`，可选 **FGM** | 冲 **0.62+** |
| C | EGP + **RoBERTa-wwm-ext-large**（仍 <1B）同协议 | 冲 **0.64–0.67** 区间 |
| D | 类型逆频加权损失（抬 `ite/sym`） | 补 macro/弱类 |
| 晋级 | **仅 official micro_f1 > 0.5940 才写 claims** | 防回退 |

## 4. 明确不做

- 不再大规模扫 last-2 高 `head_lr`（已证崩盘）
- 不把官方 dev 用于选模/选阈值
- 不宣称端到端 SPO（CMeIE 口径不变）

## 6. TL-v5 追加（2026-07-24）

在 EGP 端到端/FGM/large 全线未超 0.594 之后，改走：

1. **多 checkpoint 融合**（union / vote-k / intersect）+ **fit-only Self SoftLexicon 召回增强**；融合策略只在 internal split 上选。
2. **对齐社区 EGP RoPE**（sin/cos interleave）后重训 freeze。
3. 融合纳入 aligned EGP 后再评估一次。

晋级门槛仍为 official micro_f1 **> 0.5940**。
