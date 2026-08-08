# 医疗文本切块算子 (MedicalChunker)

按标题、段落和字符上限切分医疗文本。采用可串联 Mapper 设计：chunks 会作为流水线内部字段传给 NER/RE；若本算子是最后一步，则导出 `{"chunks":[...]}` JSON。

- `maxChars`：单块字符上限，默认 1200
- `overlap`：硬切分重叠字符，默认 80
- `maxChunks`：传给下游的最大块数，默认 8；0 表示不限制

推荐流水线：

`text_clean → chunker → medical_ner → medical_re → triple_validator`
