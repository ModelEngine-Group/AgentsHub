"""dki.nlp —— 医疗 NER 微调相关的纯 Python 工具子包。

本子包刻意与 torch/transformers 解耦：
- labels.py / weak_label.py / eval_ner.py / quality_rules.py 只依赖标准库 + dki 内已有模块，
  因此 pytest 可在无 torch 的本机直接导入与测试；
- 真正需要 torch 的训练循环在 scripts/train_ner.py（本子包之外）。

设计动机见 docs/ner_finetuning.md：用 dict 后端（F1≈99.97%）对清洗文本做
远程监督弱标注，自动微调 bert-base-chinese 成真实医疗 NER。
"""
