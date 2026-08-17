# DataMate 自定义算子

`operators/` 保存数据整理和医学抽取所需的 DataMate 自定义算子。每个算子目录通常包含 `process.py` 和 `metadata.yml`，可在 DataMate 的算子管理界面中上传和登记。

## 数据整理算子

| 算子 | 作用 |
| --- | --- |
| `emoji_cleaner/` | 移除 Emoji 和表情符号 |
| `url_remover/` | 移除 URL、邮箱等外部噪声 |
| `whitespace_normalizer/` | 归一化空白、换行和不可见字符 |
| `medical_term_normalizer/` | 规范化医学缩写和常见术语 |
| `table_column_cleaner/` | 清洗 CSV 字段并保留表格结构 |
| `json_field_cleaner/` | 清洗 JSON、JSONL 字段并保留结构 |
| `llm_noise_filter/` | 识别语义噪声并输出质量证据 |
| `medical_text_quality_filter/` | 评估文本长度、特殊字符比例和重复内容 |

## 医学抽取算子

| 算子 | 作用 |
| --- | --- |
| `medical_record_splitter/` | 将多段医学文本拆分为可抽取记录 |
| `medical_entity_extractor/` | 抽取疾病、症状、药物、检查等实体 |
| `medical_relation_extractor/` | 抽取实体之间的医学关系 |
| `medical_triple_generator/` | 生成知识图谱三元组 |
| `unified_jsonl_exporter/` | 导出统一 JSONL 结果 |

噪声规则库和术语映射库由归档根目录的 `prepare_runtime_assets.py` 展开。算子只负责单步处理，数据集编排由 MCP 服务完成，结果应保留来源或处理证据。
