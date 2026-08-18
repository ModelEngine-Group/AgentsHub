# 本地抽取模型产物

## 神经模型（主力，GPU 训练，权重随仓）

由 `scripts/train_neural_extractor.py` 训练的 GPLinker（GlobalPointer 联合实体-关系抽取）：

- `neural_extractor/`：主力 macbert-large 模型（CMeIE-V2 训练；实体 F1 0.766 / 端到端三元组 0.534）；
- `neural_extractor_base/`：roberta-base 模型（用于双编码器集成对照）；
- `neural_extractor_v1/`：CMeIE-V1 训练的 macbert-large（实体 0.779 / strict SPO-F1 0.6146，超公开 GPLinker/CasRel）；
- `neural_extractor_v1_base/`：CMeIE-V1 roberta-base（集成对照）。

每个目录含 `pytorch_model.bin`、`extractor_config.json`、tokenizer 与 `train_log.json`（训练曲线 + best epoch）。
编码器权重在 `data/models/encoder/`（由 `scripts/download_encoder.py` 获取，已 gitignore，可重下载）。
无 GPU/torch 时系统自动回退下面的词典基线。

## 词典基线产物（回退/对照，CPU 确定性生成）

由 `python data/prep/build_fast_extractor.py` 确定性生成：

- `fast_extractor.json`：从 CMeIE-V2 train、DIAKG gold 与 CM3KG 结构化图谱学习的
  CPU Trie Span-NER / 关系事实快速路径（无 GPU/API 时的可复现下界）；
- `entity_linker.json`：医疗术语索引 **39,372 条**（CM3KG 7,467 + CMeIE/DIAKG 31,905）——in-KB 链接准确率 0.973 / NIL 拒绝 1.0（`outputs/eval_entity_linking.json`），真实 mention 可链接率 0.733（`outputs/eval_entity_linking_real.json`）；
- `temperature_calibration.json`：开发集拟合的温度标定参数（预测级 ECE 0.026）。

词典基线是可复现的**非神经**下界，不冒充神经模型；神经模型与其共用相同输入输出契约。
