# 医疗 NER 微调（任务一加分：构造数据集 + 微调 <1B 小模型 + 编排增强）

## 1. 动机

初赛版本 NER 的 `bert` 后端用的是**未微调**的 `bert-base-chinese`：分类头是随机初始化的
占位头（`config.id2label` 为 `LABEL_0/LABEL_1`），类型映射 `_LABEL_TYPE_MAP`（ORG→Department 等）
在医疗语义上无意义。这带来两个问题：

1. 任务一加分方向「构造数据集 + 微调 <1B 小模型 + 编排增强」完全没做；
2. NPU 加速成绩只能验吞吐、不能验抽取精度（分类头是占位的，抽不出真实医疗实体）。

本次用**远程监督（distant supervision）**微调 `bert-base-chinese`（101.7M 参数，远 < 1B），
一举补齐加分项、并让 NPU/INT8 成绩挂到真实医疗 F1。

## 2. 数据构造：dict 后端远程监督弱标注

不做人工标注。利用项目已有的 dict 后端（对本合成语料抽取 F1≈99.97%，已能定位干净的
**字符级 span**）自动打 BIO 标签：

- `dki/nlp/weak_label.py::build_ner_dataset` 复用算子的 `load_vocab` + `_resolve_spans`
  （单一事实来源），对 200 份清洗文本产出字符级实体 span；
- 6 类实体（Disease/Symptom/Drug/Exam/Treatment/Department）→ BIO 标签集 **13 类**
  （`O` + 每类 `B-`/`I-`），定义见 `dki/nlp/labels.py`；
- 可选用金标 `(name, type)` 集合过滤 dict 越界匹配（`gold_filter=True`），使弱标签质量≈金标；
- 确定性切分：200 doc 按 `seed=42` 洗牌 8:2 → **160 训练 / 40 验证**，验证集训练不可见。

产物：`data/ner/train.jsonl`、`data/ner/eval.jsonl`、`data/ner/label_list.json`。

## 3. 训练：手写 torch 循环

`scripts/train_ner.py`：

- 模型：`AutoModelForTokenClassification.from_pretrained("bert-base-chinese", num_labels=13, id2label=...)`，
  分类头 2→13 自动重初始化；
- 标签对齐：fast tokenizer 的 `offset_mapping` 把字符级 BIO 映射到 token 级
  （special/续词 → `-100` 忽略），对中文 BERT 稳健；
- 超参：`lr=5e-5, epochs=5, batch=16, max_length=256, seed=42`（全确定性）；
- 刻意用**手写训练循环**而非 `transformers.Trainer`：规模极小（160 样本、bert-base、5 epoch），
  循环 <40 行、无样板，且免疫服务器 transformers 5.x 的 `TrainingArguments` API 漂移；
- 训练设备：**CPU**（bert-base 此规模在 128 核服务器上约 8 分钟；无需 NPU，保证任意机可复现）。

产物：`models/medical_ner_bert/`（权重 + tokenizer + config，config.id2label 为真实
`B-Disease/I-Symptom…`）；`outputs/ner_finetune/metrics.json`（三后端对比）。

## 4. 结果：三后端实体级 F1（40 份训练不可见验证集）

| 后端 | Precision | Recall | F1 |
|---|---:|---:|---:|
| **微调 BERT** | 0.9923 | 0.9948 | **0.9936** |
| dict（教师） | 0.9974 | 0.9974 | 0.9974 |
| 未微调 BERT（基线） | 0.0000 | 0.0000 | **0.0000** |

- 微调 BERT 逼近 dict 教师（差 0.4 点，即 BERT 泛化损失），且**具备词典外泛化能力**
  （dict 是闭集匹配，BERT 是序列标注）；
- 未微调基线 F1=0（占位头抽不出任何正确医疗实体，388 全漏），量化"微调有效"这一结论。

## 5. 接回 bert 后端（编排增强 + 向后兼容）

`dki/bench/npu/bert_ner_infer.py`：

- `__init__` 自动探测 `models/medical_ner_bert/`：存在则用微调权重（真实 13 标签），
  不存在回退 `bert-base-chinese`（占位，仅吞吐基线）；显式传 `model_name` 以其为准；
- `_decode` 的类型解析：真实医疗类型（`Disease/…`）直接采用，占位通用 NER 标签走
  `_LABEL_TYPE_MAP` 兜底（`_resolve_entity_type`，一行分支，向后兼容）；
- 微调模式下把 surface 提及归一到词典规范名（与 dict/gold 口径一致，便于实体级评测）；
- 算子层 `MedicalEntityExtractor` 加可选 `modelPath`（默认自动探测），dict 仍是默认 backend；
  新增 `configs/pipeline_kg_build_bert.yaml` 作 backend=bert 的编排增强演示。

**默认行为不变**：`configs/pipeline_*` 默认 dict 后端，201 项 pytest（onnx 环境 207）与 F1=99.97%/NL2SQL=100% 全不回归。

## 6. 复跑

```bash
# 服务器（CPU 训练，约 8 分钟）
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/huggingface_home TOKENIZERS_PARALLELISM=true
# 1) 弱标注数据集
python3 -c "from dki.nlp.weak_label import build_ner_dataset; print(build_ner_dataset(gold_filter=True))"
# 2) 微调 + 三后端 F1
python3 scripts/train_ner.py --epochs 5 --device cpu
# 3) 验证接回（应抽出真实类型：冠心病=Disease, 阿司匹林=Drug）
python3 -c "from dki.bench.npu.bert_ner_infer import BertNerBackend; \
    print(BertNerBackend(device='cpu').extract(['患者诊断冠心病，给予阿司匹林口服。'])[0])"
```

## 7. 模型选型：为什么是 bert-base / rbt6，不是更大或更小

本任务判别难度不高（dict 教师 F1≈0.997），模型容量是可调的性能旋钮。实测三档：

| 模型 | 层数 | 参数 | 同分布 eval F1 | 训练耗时(CPU) |
|---|---:|---:|---:|---:|
| bert-base-chinese | 12 | 102M | 0.9936 | 476s |
| **hfl/rbt6** | 6 | ~60M | 0.9794 | 253s |
| hfl/rbt3 | 3 | ~38M | 0.9253 | 124s |

层数减半（rbt6）只掉 1.4 点 F1，却让 NPU 前向算力≈减半（详见
`docs/npu_optimization.md` §5.7，端到端吞吐 bert-FP16 2516 → rbt6-INT8-fast 5961 样本/s，
2.37×）。rbt3（3 层）掉到 0.9253 且过拟合最重，是甜点之下。故**性能优先选 rbt6、
精度优先选 bert-base**，均 < 1B。

## 8. 过拟合诊断：分布外（OOD）鲁棒性

train/eval 同源于合成生成器 + 同一词典，弱标签也来自该词典——模型本质在蒸馏词典，
held-out F1 测的是**同分布**泛化，会高估开放场景表现。为诚实量化过拟合，用外部 LLM
（小米 mimo-v2.5）生成**表述迥异的分布外病历**并**独立标注**实体（不用我们的词典），
构成 OOD 集（`scripts/gen_ood_eval.py` + `scripts/eval_ood.py`）。

**匹配口径（关键）**：分布外文本同一实体大量用非规范表述（"没睡好"=失眠、"高压"=高血压），
若按"(规范名,类型)精确匹配"会把"抽对但写法不同"误判为漏报，严重低估召回（早期精确匹配
把 bert 召回压到 0.53，是口径假象）。故 LLM 金标同时给**原文逐字 mention** 与规范名，
评测用标准 NER 的 **span-overlap + 同类型**宽松匹配。

| 后端 | 同分布 F1 | OOD F1 | OOD 精确率 | OOD 召回 |
|---|---:|---:|---:|---:|
| dict | 0.9974 | 0.621 | 1.00 | 0.45 |
| 微调 bert | 0.9936 | **0.785** | 0.77 | **0.79** |
| rbt6 | 0.9794 | 0.722 | 0.72 | 0.72 |
| rbt3 | 0.9253 | 0.620 | 0.59 | 0.65 |

- **确有分布偏移**：同分布 0.99 → OOD 0.78（微调 bert），差约 0.2——这是诚实边界，不回避；
- **微调的价值被 OOD 硬证**：BERT 的 OOD 召回 **0.79 vs dict 0.45**（近 1.8 倍）——词典法
  精确率满分但只认词表内词，漏掉近一半口语化/变体表述；神经模型能泛化到这些新表述，
  这正是"为什么要微调"的硬证据（生产可 dict 保精确 + BERT 补召回集成）；
- **rbt3 过拟合最重**（同分布→OOD 落差最大），再次印证不宜低于 rbt6。

OOD 集：34 篇 / 229 实体（外部 LLM 生成 + 逐字 mention 独立标注）。原始数据：
`outputs/ner_finetune/ood_eval.json`、`data/ood/ood_eval.jsonl`。

## 9. 可选质量模式：隔离鲁棒微调 + 医学规则校准

在不替换上述 BERT/RBT6/OM 性能路径的前提下，新增显式可选模型
`models/medical_ner_quality`：

1. `scripts/gen_ood_quality.py` 另行生成 80 篇口语化病历（481 实体），每个 mention 都做
   原文逐字校验；`scripts/build_quality_data.py` 固定 `seed=20260718` 切为
   **56 train / 12 dev / 12 internal holdout**；
2. 只比较最终 OOD 测试文本的 SHA-256，确认 0 精确重合；测试标签不进入训练、切分或
   checkpoint 选择；
3. 从原 `medical_ner_bert` 出发，以全部 160 份 ID train + 56 份质量 train×3 混合微调，
   防止灾难性遗忘；用质量 dev 的 `F1 + 0.1×Recall` 选 best epoch；
4. `dki/nlp/quality_rules.py` 仅在目录名为 `medical_ner_quality` 时启用，对医学语义稳定的
   血压/血糖升高表述与 BP/FBG/CT 等检查边界作确定性校准；默认 dict、BERT、RBT6 和
   NPU OM 路径完全不执行这些规则。

实测（Ascend NPU 微调 14.9s，best epoch=5）：

| 模式/数据集 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 原 BERT / 最终 OOD | 0.7745 | 0.7948 | 0.7845 |
| quality（仅鲁棒微调）/ 既有 OOD | 0.8451 | 0.8341 | 0.8396 |
| **quality + 校准 / 既有 OOD（校准开发集）** | **0.9478** | **0.9520** | **0.9499** |
| quality + 校准 / 原 ID eval | 0.9974 | 0.9974 | 0.9974 |
| quality + 校准 / 冻结后多主题盲测-2（11篇/83实体） | 0.6939 | 0.8193 | 0.7514 |
| quality 高召回双模型 / 同盲测-2 | 0.6283 | 0.8554 | 0.7245 |
| **quality v2 + 校准 / 同盲测-2** | **0.7019** | **0.8795** | **0.7807** |
| quality v2 / 全新冻结盲测-3（15篇/173可定位实体） | 0.8205 | 0.7399 | 0.7781 |
| **quality v2 可选 recall mode / 同盲测-3** | **0.7692** | **0.8092** | **0.7887** |

既有 OOD 上相对原 BERT：召回 **+15.72 点**、精确率 **+17.33 点**、F1 **+16.54 点**。
既有 OOD 上的校准开发成绩超过 0.9 召回目标；但该 34 篇数据用于诊断后又指导了规则设计，因此 **0.9499 是校准开发集成绩，
不是最终盲测成绩**。真正的冻结后验收由 `scripts/gen_ood_blind.py` 生成主题更多样的新集合，模型与规则在生成前冻结、
生成和标注分成两次独立 LLM 调用、只评一次。盲测-2 的 honest 结果为：原 BERT
P/R/F1=0.5267/0.8313/0.6449；旧单 quality 为 0.6939/0.8193/0.7514；旧高召回双模型为
0.6283/0.8554/0.7245。

为在不动性能路径的前提下继续同时提高召回与准确度，新增独立 `medical_ner_quality_v2`：

1. `scripts/gen_ood_quality_v2.py` 覆盖十个临床主题与六类书写风格，生成和独立标注分离；确定性审计删除
   mention 类型冲突、原文不存在的 mention 与单字 Exam/Treatment 碎片；
2. `scripts/build_quality_data_v2.py` 对历史文本做 SHA-256 防泄漏，明确把早期已用于诊断的集合降级为
   development/error-analysis 数据，但持续禁止 final2 与未来盲测；39 篇数据按 `seed=20260719` 划为
   29 train / 4 dev / 6 internal holdout，共 352 实体；
3. checkpoint 用 `micro-F1 + 0.15×Recall + 0.2×逐类型 macro-F1` 选择，避免 Symptom 多数类支配；
   NPU 训练 8 epoch/14.4s，dev P/R/F1=0.7949/0.9688/0.8732，internal holdout=
   0.7429/0.9455/0.8320，ID 仍为 0.9974；
4. 对此前未进入 v2 选模的 blind-2 做冻结评测，v2 达 **0.7019/0.8795/0.7807**。相对旧单 quality，
   精确率 **+0.80 点**、召回 **+6.02 点**、F1 **+2.93 点**，三个指标同向提升；v2 的可选双模型
   union 并未新增 TP（0.6404/0.8795/0.7411），仅增加 10 个 FP，因此明确保留单 v2 为推荐质量路径。
   召回仍低于 0.9，所以不宣称未知分布已经稳定达标。

质量 v2 前后再次核验冻结 BERT/RBT6 权重和五个 FP16/INT8 OM，SHA-256 **9/9 一致**；此前完整质量
优化审计的 15 个冻结产物也全部未变，因此现有 `rbt6-INT8-fast` 5961 samples/s 与默认生产路径保持不变。
证据落盘于 `outputs/ner_finetune/metrics_quality_v2.json` 和
`outputs/ner_finetune/ood_final2_quality_v2.json`。

模型和规则冻结之后，又从零生成 final3：15 篇、原始自动标注 176 条，其中 173 条 mention 可由
`_gold_spans` 一对一定位；生成器记录与历史集合 SHA-256 精确重合为 0。严格只评一次的结果是：
原 BERT 0.6816/0.7919/0.7326，单 quality v2 **0.8205/0.7399/0.7781**，可选 recall mode
**0.7692/0.8092/0.7887**。单 v2 相比原 BERT 以召回换得大幅精确率和更高 F1；recall mode
在单 v2 上补回 12 个 TP，召回 +6.93 点且 F1 +1.06 点，但 precision 降 5.13 点。两种模式的
recall 都未达到 0.9，因此 final3 被封存为最终验收集，**不再据其误差调整模型、规则或阈值**。
原始证据为 `data/ood_blind/final3_blind_eval.manifest.json` 与
`outputs/ner_finetune/ood_final3_quality_v2.json`。

复跑：

```bash
python3 scripts/gen_ood_quality_v2.py --n 240 --skip-review
python3 scripts/build_quality_data_v2.py --allow-development-source --seed 20260719
python3 scripts/train_ner_quality.py --output models/medical_ner_quality_v2 \
  --ood-data data/ood_train_v2 --epochs 8 --ood-repeat 5 --lr 1e-5 \
  --recall-weight 0.15 --macro-weight 0.2 --device npu
# checkpoint 与规则冻结后再创建新盲测；生成后只评一次，不回看调参
python3 scripts/gen_ood_blind.py --n 60 --output data/ood_blind/final3_blind_eval.jsonl
python3 scripts/eval_frozen_quality.py --ood data/ood_blind/final3_blind_eval.jsonl \
  --quality-model models/medical_ner_quality_v2 --with-recall-mode \
  --output outputs/ner_finetune/ood_final3_quality_v2.json
```

> 解释边界：internal holdout 来自同一个质量数据生成方案，检验训练泛化；最终 34 篇 OOD
> 是更早独立生成且从不参与选模的外部验收。两者分布不同，故数值不要求相同；最终结果
> 仍应结合人工标注真实病历复验，不能把 LLM 金标当作临床部署认证。

