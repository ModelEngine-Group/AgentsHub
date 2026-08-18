# 医疗文本清洗算子 (TextClean)

纯规则 DataMate 算子，对医疗/病理文本执行网页噪声、页眉页脚、链接图片、LaTeX 和短碎片清洗。

- 输入：文本文件或上游 `text`
- 输出：清洗后的 `text`
- 参数：`minLineLength`，默认 8
- 特点：无外部 API、低延时、可作为五算子流水线第一步

推荐流水线：

`text_clean → chunker → medical_ner → medical_re → triple_validator`
