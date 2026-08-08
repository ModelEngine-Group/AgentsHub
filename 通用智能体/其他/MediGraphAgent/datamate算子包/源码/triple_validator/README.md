# 三元组校验算子 (TripleValidator)

纯规则的三元组质量治理 DataMate 算子（无需 API，低时延）。

## 输入 / 输出
- 输入：`{"triples":[...]}` JSON 文本（如 MedicalRE 的输出）。
- 输出：`{"valid":[...], "rejected":[{"triple","reason"}]}`。

## 校验规则
1. Schema 合法性：关系两端实体类型必须匹配本体约束。
2. 去重归一：大小写/空格归一后去重，保留高置信。
3. 置信度过滤：低于 `minConfidence` 丢弃。
4. 冲突检测：互斥关系（如 阳性标志物 vs 阴性标志物）报警拒绝。

## 参数
| 参数 | 说明 |
| --- | --- |
| minConfidence | 最低置信度阈值（默认 0.5） |

## 打包上传
压缩包名必须为 `triple_validator.zip`。
