# 心血管代谢演示数据集

本目录包含 3 份合成医疗文本，用于 DataMate 五算子流水线、MediGraph 神经抽取、
知识图谱构建和 Nexent 对话演示。数据不对应真实患者，不包含个人标识信息。

建议流水线：

`text_clean -> chunker -> medical_ner -> medical_re -> triple_validator`

建议问答：

1. 心力衰竭有哪些症状和检查？
2. 高血压可能并发哪些疾病？
3. 2 型糖尿病常用哪些药物和检查？
