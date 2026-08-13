# 评分证据口径一致性（完美态 Phase 0）

> 评审速览：三个数字栏**不可混报**。本页与 `outputs/cblue/cmeee_metrics.json`、
> `README.md`、`scripts/check_claims_metrics_consistency.py` 对齐。

## 三栏叙事（强制）

| 栏 | 含义 | 主证据 | 当前数字（以磁盘为准） |
|---|---|---|---|
| A. 合成流水线正确性 | 抽取器与金标同源于词典，证明链路可跑通 | `outputs/eval/kg_extraction_eval.json` | F1≈**99.97%** |
| B. 同分布微调有效性 | dict 远程监督蒸馏 `<1B` BERT；**非开放域 SOTA** | `outputs/ner_finetune/metrics.json` + OOD/final3 | 同分布≈0.99；OOD 召回未稳达 0.9 |
| C. **真实权威基准 CBLUE** | 官方 CMeEE-V2 Micro-F1（唯一对外主报） | `outputs/cblue/cmeee_metrics.json` | **0.6031**（soft 融合地板） |

另有与 schema-v3 OM 完整绑定的部署候选：官方 dev F1 0.613020
（FP32 ONNX 0.613214，差 0.000194）。该候选必须单列为“部署候选”，不能静默
覆盖 0.6031 均衡基线，因为 `dep/ite/sym` 仍是弱类型。

## 禁止宣传语

- SoftLex-add 失败配方（~0.51）/ large freeze-only（~0.59）不得称 SOTA
- CMeIE **gold-pair 谓词 F1** 不得写成端到端 SPO（见 `cmeie_dual_eval.json` 双栏）
- 未验收的 OM shadow / DataMate Mapper `formal_microbatch=false` 不得写成平台原生微批；已验收的独立 CMeEE `execute_batch` 成绩必须明确运行口径
- 历史验收 **0.5935** 只能标「历史基线」，不得再当 headline

## 复现 / 校验

```powershell
python scripts/normalize_cmeee_metrics.py
python scripts/check_claims_metrics_consistency.py
python scripts/build_evidence_claims.py --output outputs/claims/headline_claims.json
python scripts/validate_evidence.py outputs/claims/headline_claims.json
```

## 晋级闸门（CMeEE）

仅当官方 Micro-F1 **严格大于**当前地板时更新 `cmeee_metrics.json` 与 claims：

- ≥0.624 → 对齐 CBLUE 官方 PLM 高点
- ≥0.670 → 对齐人类参考
- ≥0.700 → P0 完美态硬门槛
