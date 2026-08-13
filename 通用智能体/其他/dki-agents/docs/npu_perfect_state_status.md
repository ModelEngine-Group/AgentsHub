# Phase 4 NPU 完美态状态（fail-closed）

更新时间：2026-07-25

| 晋级门 | 结果 | 对外口径 |
|--------|------|----------|
| pyACL lifecycle / soak | 历史 2026-07-20 通过 | 可引用历史验收 JSON |
| OM exact parity | FP16b=1.0；INT8≈0.986 | 可引用；非 CMeEE GP |
| DataMate formal_microbatch | **false**（Mapper fill=1） | 仅称 probe / 本地微批 |
| hot-shape lattice 实机 | 未测 | CPU 仿真 only |
| OM shadow 默认路径 | **关闭** | 候选算子 |
| CMeEE ckpt ↔ OM | **未绑定** | 不以 CBLUE F1 冒充 OM 加速 |

**结论**：Release B **not promoted**。详见 `docs/npu_optimization.md` §5.11–§5.13。
