# 医学实体识别算子 (MedicalNER)

基于 LLM 的医疗实体抽取 DataMate 算子。读取文件文本，调用 OpenAI 兼容 API（硅基流动 / 百炼 / 本地 vLLM）抽取医学实体，把结果以 JSON 写回。

## 输入 / 输出
- 输入：文本文件（`sample['text']`）。
- 输出：`{"entities": [{"name", "type", "confidence"}]}` 的 JSON 文本。

## 实体类型
Disease(疾病)/Symptom(症状)/Drug(药物)/Examination(检查)/Procedure(手术)/Body(身体部位)/Department(科室)/Tumor(肿瘤)/Biomarker(标志物)/Gene(基因)/Morphology(形态学特征)。

## 参数（在 DataMate 界面配置）
| 参数 | 说明 |
| --- | --- |
| apiBase | OpenAI 兼容 API 地址，如 `https://api.siliconflow.cn/v1` |
| apiKey | 访问密钥（必填） |
| model | 模型名，如 `Qwen/Qwen3.6-35B-A3B` |
| temperature | 抽取建议 0.0-0.3 |
| maxChars | 单次最大字符数 |
| maxChunks | 上游存在切块时最多处理的块数，0 表示不限制 |

## 打包上传
压缩包名必须为 `medical_ner.zip`（与注册路径包名一致）。有上游 `chunker` 时会逐块抽取并合并去重。可用 `datamate_ops/build_zip.py` 一键打包。
