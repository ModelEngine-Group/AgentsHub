# NPU 算子优化实现与性能验证方案

> 加分项对应：任务二「NPU 算子优化实现（10 分）：适合 NPU 场景，有对比分析，优化思路清晰」

## 1. 哪个算子适合 NPU（场景分析）

| 算子 | 计算特征 | NPU 适配结论 |
|---|---|---|
| MedicalTextCleaner / SectionSplitter | 正则/字符串处理，访存密集、分支多 | **不适合**——无矩阵运算，NPU 无收益 |
| MedicalEntityExtractor（dict 后端） | AC 自动机多模式匹配，CPU 实测 **7746 docs/s、p50=0.113ms** | **不适合**——已是微秒级，且指针跳转型负载与 NPU 架构不匹配 |
| MedicalEntityExtractor（**bert 后端**） | Transformer 前向推理（GEMM 密集） | **适合**——批量矩阵乘正是昇腾 Cube 单元的目标负载 |
| MedicalRelationExtractor（useLlm=true） | LLM 推理 | 适合（走推理服务，不在本算子进程内优化） |

结论：NPU 优化落点为 `MedicalEntityExtractor(backend="bert")`——词典后端负责高吞吐低成本
的确定性抽取，BERT 后端面向开放文本泛化（词典外实体），二者互补。**词典后端的
CPU 实测数据同时说明了"算子处理时延低"这一考核点**（见 `outputs/bench/operator_bench.md`）。

## 2. 昇腾 NPU 优化点（代码：`dki/bench/npu/bert_ner_infer.py`）

| # | 优化 | 实现 | 原理 |
|---|---|---|---|
| 1 | **静态 shape 分桶** | 变长文本 pad 到 64/128/256 三档桶（`SEQ_BUCKETS` + `padding="max_length"`） | 昇腾图编译按 shape 缓存；动态 shape 每个新长度都触发重新编译（秒级开销）。分桶后仅 3×batch 档组合，编译缓存全命中 |
| 2 | **二进制算子模式** | `torch.npu.set_compile_mode(jit_compile=False)` | 使用 CANN 预编译二进制算子库，消除在线编译 |
| 3 | **预热排除编译时间** | `warmup()` 对全部 (桶, batch) 组合先跑 N 轮，计时从 warmup 后开始（warmup 耗时单独上报） | 公平计量稳态吞吐 |
| 4 | **固定 batch + 尾批补齐** | batch 维固定（默认 8/32，可用 `BATCHES` 扩展扫参）；分桶模式下尾批补齐到固定 batch 后只返回真实样本结果 | 提升 Cube 单元利用率，并避免尾批/单条调用触发新的 batch 维图编译 |
| 5 | **推理上下文瘦身** | 缓存 torch device 字符串；推理区间使用 `torch.inference_mode()` | 降低 Python 分发与 autograd 元数据开销，不改变输出 |
| 6 | **fast tokenizer 并行保护** | NPU/GPU 后端默认 `TOKENIZERS_PARALLELISM=true`，仍尊重用户显式环境变量 | 避免 CPU 侧 tokenization 因继承关闭并行的环境变量而成为瓶颈 |
| 7 | **混合精度** | `--half` 开关（`model.half()`） | 910B FP16 峰值算力约为 FP32 两倍，NER 精度损失可忽略 |
| 8 | 绑核（可选） | 复跑手册中用 `taskset -c` 固定 CPU 亲和 | 降低 host 侧预处理抖动 |

CPU 对照组走**完全相同**的分桶/批处理逻辑，保证对比公平。

## 3. 与 DataMate 平台的 NPU 链路衔接（原生钩子，零侵入）

DataMate 从算子声明到调度已内置 NPU 通道，本项目算子完整接入：

1. **资源声明**：算子包 `metadata.yml` 的 `runtime.npu` 字段（本算子 bert 模式建议设 1）；
2. **平台调度**：`datamate/runtime/python-executor/datamate/core/dataset.py` 的
   `_run_single_op` 读取 `init_kwargs["npu"] > 0` 后设置 Ray `resources={"npu": n}`；
3. **算子内判定**：`datamate/core/base_op.py` 的 `BaseOp.use_npu()`
   （`accelerator=="npu"` 且 `torch_npu.npu.is_available()`）；
4. **本项目对接**：`MedicalEntityExtractor._extract_bert` 调用 `self.use_npu()`
   决定 `BertNerBackend(device="npu"|"cpu")`——即平台下发 `accelerator: npu`
   参数即自动走 NPU，无需改代码。

## 4. 复跑手册（昇腾环境）

环境参考：Atlas 800T A2（Ascend 910B4）+ CANN 8.x + torch 2.7.1 + torch_npu 2.7.1（PTA 7.3.0）。

**一键复跑**（推荐；产出 CPU 基线 / NPU FP32 / NPU FP16 / 分桶消融 / 功耗采样 / 两份对比报告）：

```bash
cd dki-agents && bash scripts/run_npu_all.sh 2>&1 | tee outputs/bench/npu/run_all.log
# 扩展扫参以寻找吞吐平台期：
BATCHES=8,16,32,64,128 ITERS=50 bash scripts/run_npu_all.sh 2>&1 | tee outputs/bench/npu/run_all_sweep.log
```

batch sweep 中若原始病历数不足 `batch*4`，基准脚本会循环复用病历文本以保持满批，
避免大 batch 被尾批补齐拖低，从而更接近稳态吞吐上限。

分步命令（等价）：

```bash
# 1) 环境（ModelArts notebook 已预装 torch_npu；否则按 CANN 版本安装）
source /usr/local/Ascend/cann-8.x/set_env.sh    # 或用户目录安装的 set_env.sh
pip install transformers python-dotenv            # torch/torch_npu 已随环境提供

# 2) CPU 基线（同机 host CPU，绑核降低抖动）
taskset -c 0-15 python -m dki.bench.npu.run_npu_bench --device cpu --batch 8,32 --iters 50

# 3) NPU 实测（FP32 与 FP16 各一轮）
python -m dki.bench.npu.run_npu_bench --device npu --batch 8,32 --iters 50
python -m dki.bench.npu.run_npu_bench --device npu --batch 8,32 --iters 50 --half

# 4) 消融实验：关闭静态分桶（动态 shape），量化"分桶优化"本身的收益
python -m dki.bench.npu.run_npu_bench --device npu --batch 8,32 --iters 50 --half --dynamic-shape
python -m dki.bench.npu.run_npu_bench --compare-ablation \
    outputs/bench/npu/ner_bench_npu.json outputs/bench/npu/ner_bench_npu_dynamic.json

# 5) 生成 CPU vs NPU 对比报告
python -m dki.bench.npu.run_npu_bench --compare \
    outputs/bench/npu/ner_bench_cpu.json outputs/bench/npu/ner_bench_npu.json
# 产出 outputs/bench/npu/ner_cpu_vs_npu.md（吞吐/加速比/延迟/预热时间）

# 6) 能效：并行功耗采样（1s 间隔解析 npu-smi），吞吐 ÷ 平均功耗 = 样本/焦耳
python scripts/sample_npu_power.py --out outputs/bench/npu/power.csv --duration 90 &
python -m dki.bench.npu.run_npu_bench --device npu --batch 32 --iters 50 --half
```

## 5. 结果记录

### 5.1 词典后端 CPU 实测（本仓库已含，`outputs/bench/operator_bench.md`）

| 算子 | 吞吐(docs/s) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| MedicalEntityExtractor(dict, DataMate AC 自动机) | 7746.83 | 0.113 | 0.168 |
| MedicalRelationExtractor | 7488.73 | 0.121 | 0.184 |
| TripleBuilderValidator | 10198.88 | 0.081 | 0.131 |

### 5.2 BERT 后端 CPU vs NPU（Ascend 910B2C 实测，bert-base-chinese，iters=50）

| batch | 精度 | CPU 吞吐(样本/s) | NPU 吞吐(样本/s) | 加速比 | CPU p50(ms) | NPU p50(ms) |
|---:|---|---:|---:|---:|---:|---:|
| 8 | FP32 | 48.85 | 387.86 | 7.9x | 642.86 | 80.30 |
| 32 | FP32 | 46.27 | 465.94 | 10.1x | 2752.96 | 275.13 |
| 8 | FP16 | 48.85 | 555.96 | 11.4x | — | 55.94 |
| 32 | FP16 | 46.27 | 771.46 | 16.7x | — | 165.80 |

**消融：静态分桶收益（NPU FP16，同数据同精度）**

| batch | 动态 shape 吞吐 | 静态分桶吞吐 | 分桶收益 |
|---:|---:|---:|---:|
| 8 | 543.77 | 555.96 | 1.02x |
| 32 | 762.49 | 771.46 | 1.01x |

**能效**：FP16 bs32 平均功耗 98.5 W，吞吐 771.46 样本/s
→ **7.8 样本/焦耳**（采样方法：`scripts/sample_npu_power.py` 1s 间隔并行采样）。

> 实测环境：Ascend 910B2C（`npu-smi 26.0.rc1`）、CANN 8.5.1、
> torch 2.7.1+cpu + torch_npu 2.7.1；完整日志 `outputs/bench/npu/run_all.log`。

### 5.3 满批 batch sweep（FP16，Ascend 910B2C）

为逼近稳态吞吐上限，`run_npu_bench` 在 sweep 时按 `batch*4` 构造满批样本；
原始 200 份病历不足时循环复用文本，避免大 batch 被尾批补齐拖低。

| batch | 每轮样本数 | NPU 吞吐(样本/s) | p50(ms) | p95(ms) |
|---:|---:|---:|---:|---:|
| 8 | 32 | 704.66 | 44.30 | 44.64 |
| 16 | 64 | 876.26 | 72.79 | 75.71 |
| 32 | 128 | 990.91 | 128.81 | 133.19 |
| 64 | 256 | 1068.73 | 237.86 | 255.72 |
| 128 | 512 | **1105.72** | 461.21 | 473.30 |

继续上探 batch 后吞吐回落，说明当前 PyTorch+torch_npu BERT 路径在本数据形态下的
实测甜点为 bs128：

| batch | 每轮样本数 | NPU 吞吐(样本/s) | p50(ms) | p95(ms) |
|---:|---:|---:|---:|---:|
| 256 | 1024 | 841.86 | 1208.22 | 1348.77 |
| 512 | 2048 | 884.61 | 2296.65 | 2465.06 |

CPU bs128 对照（iters=10）吞吐 34.65 样本/s；NPU FP16 bs128 最优吞吐
1105.72 样本/s，端到端加速比 **31.9x**。原始数据：
`outputs/bench/npu/ner_bench_npu_fp16_sweep_opt_fullbatch.json` 与
`outputs/bench/npu/ner_bench_cpu_bs128_opt.json`；大 batch 上探数据：
`outputs/bench/npu/ner_bench_npu_fp16_largebatch_opt.json`。

bs128 功耗复测（iters=180）吞吐 1098.70 样本/s，p50 461.04ms，p95 482.35ms；
全窗口平均功耗 163.2W，对应 **6.7 样本/焦耳**；按 `npu-smi` 采样中
大于 100W 的活跃窗口计算，平均功耗 198.0W，对应 **5.6 样本/焦耳**。
原始数据：`outputs/bench/npu/ner_bench_npu_fp16_bs128_power_opt.json`、
`outputs/bench/npu/power_fp16_bs128_opt.csv` 与
`outputs/bench/npu/power_fp16_bs128_opt.log`。

CPU 侧 tokenizer 并行 A/B（bs128，iters=30）显示关闭并行会让端到端吞吐从
1104.01 降到 960.19 样本/s（约 0.87x），因此后端和一键脚本均把
`TOKENIZERS_PARALLELISM` 的默认值设为 `true`，但仍尊重用户显式环境变量。
原始数据：`outputs/bench/npu_opt/ner_bench_npu_fp16_bs128_tokenizers_true.json` 与
`outputs/bench/npu_opt/ner_bench_npu_fp16_bs128_tokenizers_false.json`。

### 5.4 ACL/OM 上限分层探测（非主链路成绩）

为判断 PyTorch+torch_npu 端到端路径距离更底层 NPU 执行上限还有多少空间，补充
ONNX → ATC → OM → Python ACL 的静态模型探测。这里把结果分成三层，并扫
bs64/128/256 三档固定 batch：

| batch | decode probe(样本/s) | argmax probe(样本/s) | forward-only(样本/s) |
|---:|---:|---:|---:|
| 64 | **1328.08** | **1817.98** | **2530.67** |
| 128 | 1281.59 | 1759.79 | 2414.31 |
| 256 | 1218.75 | 1633.22 | 2192.00 |

- decode probe：tokenizer + H2D + OM execute + D2H + argmax + BIO span loop；
- argmax probe：tokenizer + H2D + OM execute + D2H + argmax；
- forward-only：静态输入常驻设备，仅 OM execute，是纯模型前向上限。

这些探测**不能替代** 5.2/5.3 的端到端 NER 主成绩；主成绩仍采用
`BertNerBackend.extract()` 全流程。它们用于证明继续优化的方向：若要突破
1105.72 样本/s，应做完整 ACL 后端或 tokenizer/decode 流水化，而不是继续盲目增大 batch。
（该方向已在 §5.5 落地实测：三阶段流水线把端到端探针推到纯前向上限的 99%。）

复跑命令（昇腾环境，若未预装 ONNX 导出依赖，可先临时安装 `onnx` 与
`onnxscript`）：

```bash
python3 scripts/export_bert_ner_onnx.py \
  --batch 64 --seq-len 256 --half \
  --output outputs/bench/npu/bert_ner_bs64_seq256_fp16.onnx

export PYTHONPATH=/usr/local/python3.11/site-packages:$PYTHONPATH
INPUT_SHAPE=input_ids:64,256\;attention_mask:64,256\;token_type_ids:64,256
atc --framework=5 \
  --model=outputs/bench/npu/bert_ner_bs64_seq256_fp16.onnx \
  --output=outputs/bench/npu/bert_ner_bs64_seq256_fp16 \
  --input_format=ND --input_shape=$INPUT_SHAPE --soc_version=Ascend910B2C

python3 scripts/acl_om_forward_bench.py \
  --model outputs/bench/npu/bert_ner_bs64_seq256_fp16.om \
  --output outputs/bench/npu/acl_om_forward_bs64_fp16.json \
  --batch 64 --seq-len 256 --warmup 10 --iters 100

python3 scripts/acl_om_argmax_bench.py \
  --om outputs/bench/npu/bert_ner_bs64_seq256_fp16.om \
  --output outputs/bench/npu/acl_om_argmax_bs64_fp16.json \
  --batch 64 --seq-len 256 --warmup 3 --iters 30

python3 scripts/acl_om_argmax_bench.py \
  --om outputs/bench/npu/bert_ner_bs64_seq256_fp16.om \
  --output outputs/bench/npu/acl_om_decode_bs64_fp16.json \
  --batch 64 --seq-len 256 --warmup 2 --iters 20 --decode
```

原始数据：`outputs/bench/npu/acl_om_*_bs64_fp16.json`、
`outputs/bench/npu/acl_om_*_bs128_fp16.json` 与
`outputs/bench/npu/acl_om_*_bs256_fp16.json`。bs64 是当前 ACL/OM probe 的
实测甜点；bs256 三层吞吐均回落，说明更大 batch 已进入 host 侧与模型执行长尾区。

### 5.5 ACL/OM 端到端流水线实测（tokenize / execute / decode 三阶段并行）

按 5.4 指出的方向落地：`scripts/acl_om_pipeline_bench.py` 在 OM 静态模型上
实现完整端到端 NER 探针，两项优化叠加：

1. **向量化 BIO 解码**：offsets 有效位掩码 + B/I 标签查表 +
   `np.maximum.accumulate` 跨度锚定，替代逐 token Python 循环
   （旧探针 BIO 循环约 13ms/批@bs64，向量化后 <0.2ms/批）；
   同时改从 `acl.mdl.get_output_dims` 读取真实输出形状——OM 输出 host
   缓冲区存在对齐填充，按字节数反推 num_labels 会得到越界预测 id
   （旧探针的 `id2label.get(pid, "O")` 把该错误静默吞掉了）。
2. **多进程三阶段流水线**：tokenizer 进程池 → 主进程独占 ACL
   （H2D/execute/D2H + argmax）→ decode 进程池，有界队列背压；
   fork 先于 `acl.init()`（CANN 与 fork 不兼容），全部 worker 就绪后才开始计时。
   线程版流水线无效——pyACL 调用与 fp16 numpy 运算持有 GIL，实测 1476.76
   样本/s（≈串行），各阶段"忙时"因 GIL 等待膨胀 3~265 倍；进程版把设备占用率
   从 52% 推到 96%。

| 配置（FP16，seq256，iters=50） | 吞吐(样本/s) | 设备占用率 | vs 串行探针 | 达纯前向上限 |
|---|---:|---:|---:|---:|
| 串行（向量化解码，bs64） | 1325.74 | 0.52 | 1.00x | 52.4% |
| 线程流水线 bs64（GIL 对照） | 1476.76 | — | 1.11x | 58.4% |
| 进程流水线 bs64（tokw=2） | 2481.44 | 0.96 | 1.87x | 98.1% |
| 进程流水线 bs64（tokw=6） | **2506.87** | 0.96 | **1.89x** | **99.1%** |
| 进程流水线 bs128（tokw=4） | 2362.27 | 0.96 | — | 97.8%（上限 2414.31） |
| 进程流水线 bs256（tokw=4） | 2128.99 | 0.96 | — | 97.1%（上限 2192.00） |

61 秒持续长跑（bs64，tokw=6，153,600 样本）吞吐 **2508.72 样本/s**；并行功耗
采样活跃窗口（>100W）平均 305.8W → **8.2 样本/焦耳**（PyTorch 主链路 bs128
活跃能效为 5.6）。相对 PyTorch+torch_npu 主链路最优 1105.72 样本/s，端到端
探针吞吐提升 **2.27x**；相对 CPU bs128 基线 34.65 样本/s，加速比 **72.4x**。

同 5.4 的定位声明：探针使用未微调分类头，验证吞吐与链路可用性、不声明抽取
精度；主链路成绩（5.2/5.3）不变。复跑：

```bash
python3 scripts/acl_om_pipeline_bench.py \
  --om outputs/bench/npu/bert_ner_bs64_seq256_fp16.om \
  --output outputs/bench/npu_opt/acl_om_pipeline_bs64_mp_t6.json \
  --batch 64 --iters 50 --stages mp --tokenizer-workers 6 --decode-workers 2
# 消融：--stages serial（同解码逻辑串行）；--stages pipeline（线程版，GIL 对照）
```

原始数据：`outputs/bench/npu_opt/acl_om_pipeline_*.json`、
`outputs/bench/npu_opt/power_pipeline_bs64.csv`。

### 5.6 INT8 量化冲极限（微调模型，带精度验证）

5.5 的 FP16 端到端流水线已达纯前向上限 99.1%——继续调流水线无意义。**唯一能实质
突破的是降低前向本身的算力**：INT8 量化让 GEMM 走 Cube 的 INT8 通路（峰值约 FP16 的
2×）。本节在**微调后的真实医疗 NER 模型**（见 `docs/ner_finetuning.md`）上落地 INT8，
并**同时验证吞吐与实体级精度**——补齐 5.4/5.5 "只验吞吐、不验精度"的软肋。

**工具链（CANN 8.5.1 / 910B2C 实测）**：

1. **导出 FP32 ONNX**：`scripts/export_bert_ner_onnx.py --model models/medical_ner_bert`。
   量化必须从 FP32 导出——`model.half()` 的 FP16 ONNX 会产生 LayerNormalization 的
   float/float16 类型混用，OM 前向结果全乱。
2. **W8A8 PTQ 校准**：`scripts/quantize_ner_int8.py`，用 `msmodelslim` 的
   `OnnxCalibrator`，校准集为真实病历 tokenize 到定长（与 OM 静态 shape 一致）。
   CANN 自带 `tools/msmodelslim` 是编译 stub 无 Python API，须隔离 venv 装
   `onnx==1.16.2 onnxruntime msmodelslim==8.2.1`（onnx≥1.18 删了 msmodelslim 依赖的
   `onnx.mapping`）。
3. **atc 转 INT8 OM**：`--soc_version=Ascend910B2C`。FP16 对照用 `--precision_mode
   force_fp16`（其 IO 仍为 FP32）。
4. **精度验证**：`scripts/eval_om_ner.py` 用 `VectorizedBioDecoder.decode_typed` 从 OM
   输出还原带类型实体，经词典归一到规范名 → 实体级 F1 vs 金标。

> 关键修复：OM 输出 host 缓冲区的 dtype 随精度模式变化（force_fp16 的 OM 其 IO 常为
> FP32）。`StaticOmRunner` 改为用 `acl.mdl.get_output_data_type` **动态读取输出 dtype**，
> 而非硬编码 FP16——否则按错误 itemsize 解析缓冲区，argmax 全乱（旧未微调探针因
> num_labels=2 且只验吞吐，未暴露该隐患）。

**成绩（bs64/seq256，910B2C，微调 NER 模型）**：

| 精度模式 | 端到端吞吐(样本/s) | 设备占用率 | 能效(样本/焦耳) | 实体级 F1（vs 金标） |
|---|---:|---:|---:|---:|
| FP16 | 2515.91 | 0.964 | 8.2 | 0.9893 |
| **INT8** | **3168.22** | 0.954 | **12.0** | **0.9812** |
| INT8 相对 FP16 | **1.26×** | — | **1.46×** | −0.81 点 |

INT8 把端到端吞吐从 2516 推到 **3168 样本/s（1.26×）**、能效从 8.2 提到 **12.0 样本/焦耳
（1.46×，活跃功耗 265.6W）**，而实体级 F1 仅从 0.9893 降到 **0.9812（掉 0.81 点）**——
加速真实挂到医疗抽取精度。61 秒长跑（153,600 样本）吞吐 3117.66 样本/s，稳定无衰减。

复跑（服务器）：

```bash
# 隔离量化 venv（一次）
python3 -m venv --system-site-packages /root/quant_venv
/root/quant_venv/bin/pip install 'onnx==1.16.2' onnxruntime 'msmodelslim==8.2.1'
# atc 前置：pip install scipy; export PYTHONPATH=$PYTHONPATH:<系统 site-packages>
export F=outputs/bench/npu_int8
python3 scripts/export_bert_ner_onnx.py --model models/medical_ner_bert --output $F/ner_ft_bs64_seq256_fp32.onnx --batch 64 --seq-len 256
/root/quant_venv/bin/python scripts/quantize_ner_int8.py --onnx $F/ner_ft_bs64_seq256_fp32.onnx --output $F/ner_ft_bs64_seq256_int8.onnx --model-name models/medical_ner_bert --batch 64 --seq-len 256 --calib-batches 4
atc --framework=5 --model=$F/ner_ft_bs64_seq256_int8.onnx --output=$F/ner_ft_bs64_seq256_int8 --input_format=ND --input_shape="input_ids:64,256;attention_mask:64,256;token_type_ids:64,256" --soc_version=Ascend910B2C
python3 scripts/acl_om_pipeline_bench.py --om $F/ner_ft_bs64_seq256_int8.om --output $F/pipeline_ft_int8_bs64.json --model-name models/medical_ner_bert --batch 64 --iters 50 --stages mp --tokenizer-workers 6
python3 scripts/eval_om_ner.py --om $F/ner_ft_bs64_seq256_int8.om --model-name models/medical_ner_bert --output $F/eval_om_int8.json --label int8
```

原始数据：`outputs/bench/npu_int8/`（`pipeline_ft_{fp16,int8}_bs64.json`、
`eval_om_{fp16b,int8}.json`、`power_int8_bs64.csv`）。

### 5.7 更小模型 + fast-tokenize：吞吐阶梯冲到 5961 样本/s

5.6 之后，INT8 让设备执行(execute)变快，流水线的 **tokenize 阶段开始成为共瓶颈**
（INT8 下 tokenize 3.82s ≈ execute 3.85s）。继续压性能有两条正交路径：

**(a) 更小的模型（层数即算力）**。本 NER 任务对判别能力要求不高（dict 教师 F1≈0.997），
`bert-base-chinese`（12 层）明显过配。换 **hfl/rbt6（6 层蒸馏，权重≈一半）**：
微调 eval F1=0.9794（vs bert 0.9936，仅掉 1.4 点），前向算力≈减半。rbt3（3 层）
F1=0.9253 掉太多，不采用——**层数与精度的甜点在 rbt6**。

**(b) fast-tokenize：把 offset 计算移出热路径**。`return_offsets_mapping` 占 tokenize
总耗时的 **53%**（18.4ms 里 9.7ms），但 offset 只用于把 token 预测还原成字符 span。
`--fast-tokenize` 让 tokenizer worker 不算 offset（吞吐翻倍：3470→7337 样本/s/worker），
把 offset 补算下沉到**空闲的 decode worker**（decode 仅占 0.1s，有大量余量）。
tokenizer worker 受限时增益明显（2 workers：4768 vs 4456 样本/s，占用率 0.947 vs 0.889）。

**完整吞吐阶梯（910B2C，bs64/seq256，微调模型，端到端流水线）**：

| 配置 | 吞吐(样本/s) | 相对 bert-FP16 | OM 实体 F1 | 设备占用率 |
|---|---:|---:|---:|---:|
| bert-FP16 | 2516 | 1.00× | 0.9893 | 0.964 |
| bert-INT8 | 3168 | 1.26× | 0.9812 | 0.954 |
| rbt6-FP16 | 4687 | 1.86× | 0.9721 | 0.930 |
| rbt6-INT8 | 5774 | 2.29× | 0.9490 | 0.912 |
| **rbt6-INT8-fast（极限吞吐历史点）** | **5961** | **2.37×** | 0.9490（**Treatment F1=0：TP=0/FN=81，不得作为质量安全默认**） | 0.934 |

从 bert-FP16 到 rbt6-INT8-fast，端到端吞吐提升 **2.37×**（2516→5961 样本/s），
实体 F1 从 0.9893 降到 0.9490。逐类型复核进一步发现该历史极限点的 **Treatment
TP=0、FN=81、F1=0**，所以它只用于展示理论吞吐上限，不能作为质量安全的生产默认；
当前可辩护的平衡点仍是 bert-INT8（3168 样本/s、实体 F1 0.9812）。新候选必须同时通过
Treatment recall、macro-F1、worst-type F1、吞吐和能效门，失败即保留 `promotion=false`。

> RBT 系模型 INT8 量化坑：msmodelslim 直接量化 rbt6 的 ONNX 会报 ORT
> "Attempt to replace the existing tensor"——先过 `onnxsim.simplify` 再量化即可。

复跑：`scripts/train_ner.py --base-model hfl/rbt6 --output models/medical_ner_rbt6`；
导出/量化/转 OM 同 5.6（量化前加 onnxsim）；bench 加 `--fast-tokenize --decode-workers 6`。
原始数据：`outputs/bench/npu_rbt6/`。

### 5.8 过拟合与分布外鲁棒性（诚实边界）

上述所有 F1 都是**同分布**评测（train/eval 同源于合成生成器 + 同一词典，弱标签也来自
该词典）——本质在蒸馏词典，held-out F1 会高估真实鲁棒性。为诚实量化这一点，用外部
LLM（小米 mimo-v2.5）生成**表述迥异的分布外病历**，并由 LLM **独立标注**实体（不依赖
我们的词典），构成 OOD 评测集（`scripts/gen_ood_eval.py` + `scripts/eval_ood.py`，
alias 归一后比较）。

**OOD 实体级 F1（分布外 34 篇 / 229 实体，span-overlap 匹配）**：

> 匹配口径很关键：分布外文本里同一实体大量用非规范表述（"没睡好"=失眠、"高压"=高血压），
> 若按"(规范名,类型)精确匹配"评测，会把"抽对了但写法不同"误判为漏报，严重低估召回
> （早期用精确匹配时 bert 召回被压到 0.53，实为口径假象）。这里用标准 NER 的
> **span-overlap + 同类型**宽松匹配：LLM 金标提供原文逐字 mention 与规范名两者，
> 预测实体的字符跨度与同类型金标跨度有重叠即算命中（`scripts/eval_ood.py`）。

| 后端 | 同分布 F1 | OOD F1 | OOD 精确率 | OOD 召回 | 说明 |
|---|---:|---:|---:|---:|---|
| dict | 0.9974 | 0.621 | 1.00 | 0.45 | 精确率满分但只认词典词，漏掉一半 OOD 新表述 |
| 微调 bert | 0.9936 | **0.785** | 0.77 | **0.79** | **OOD 召回 0.79 ≈ dict 的 1.8 倍**：泛化到词典外口语化表述 |
| rbt6 | 0.9794 | 0.722 | 0.72 | 0.72 | 小模型鲁棒性接近 bert |
| rbt3 | 0.9253 | 0.620 | 0.59 | 0.65 | 过拟合最重（同分布→OOD 落差最大） |
| **可选 quality 模式（既有 OOD 校准开发集）** | **0.9974** | **0.950** | **0.948** | **0.952** | 隔离 OOD 鲁棒微调 + 医学规则校准；不替换任何性能模型/OM |

三点结论：① 确实存在分布偏移，同分布 0.99 → OOD 0.78（微调 bert），差 0.2 是诚实边界；
② **微调的价值被 OOD 硬证**：BERT 的 OOD 召回 **0.79 vs dict 0.45**——纯词典法精确率满分
但只认词表内词，漏掉近一半口语化/变体表述；神经模型能识别这些新表述，这正是"为什么要微调"
（生产可 dict 保精确 + BERT 补召回集成）；③ 模型越大 OOD 越稳（bert > rbt6 > rbt3），
rbt3 过拟合最重，印证不宜低于 rbt6。针对召回 ≥0.9 的质量目标，先以 80 篇隔离数据
混合原 ID 数据做鲁棒微调，并在显式 quality 模式加入稳定医学边界校准；既有 34 篇 OOD
P/R/F1=0.948/0.952/0.950，ID F1=0.9974，但该集合后续用于规则诊断，数字只属于校准开发成绩。
冻结后独立生成和标注的多主题盲测-2 上，旧单 quality 为 0.694/0.819/0.751，高召回双模型为
0.628/0.855/0.725。为同时提高召回和准确度，又新增十主题、类型均衡的困难样本流程，按
train/dev/internal-holdout 隔离选模并训练独立 `medical_ner_quality_v2`；在同一盲测-2 上达到
**0.702/0.880/0.781**，相对旧单模型精确率 **+0.80 点**、召回 **+6.02 点**、F1 **+2.93 点**，
且 ID F1 保持 0.9974。模型/规则预先冻结后新生成的 final3（15篇/173可定位实体、历史文本 hash
重合0）一次性评测中，单 v2 为 **0.821/0.740/0.778**，可选 recall mode 为
**0.769/0.809/0.789**；后者相对单 v2 补回12个 TP并提高 F1，但精确率下降。未知主题召回仍未稳定达到
0.9，因此不把 0.952 开发数字冒充盲测能力，也不再用 final3 调参。
质量 v2 前后对冻结 BERT/RBT6 权重与五个 FP16/INT8 OM 再次逐文件 SHA-256 复核，9/9 完全一致；
此前完整质量优化审计的 15 个冻结产物也全部不变，因此 5961 samples/s 默认性能路径没有回归。
方法、切分和防泄漏说明见 `docs/ner_finetuning.md` §9。原始证据：
`outputs/ner_finetune/metrics_quality_v2.json`、`outputs/ner_finetune/ood_final2_quality_v2.json`、
`outputs/ner_finetune/ood_final3_quality_v2.json`。

## 5.9 Release A：严格门禁与独立 OM Shadow（新增能力，默认路径不变）

在冻结历史成绩之后，新增了不覆盖任何 BERT/RBT6/ONNX/OM/benchmark JSON 的生产候选层：

- `dki/nlp/bio_decode.py` 统一 canonical BIO 状态机；孤立 `I-`、类型切换等畸形序列在 PyTorch/OM 路径语义一致；
- `scripts/acl_om_pipeline_bench.py` 修复此前 `--stages pipeline` 引用不存在函数的问题，ACL 未知 dtype 改为 fail closed，进程/线程队列全部有界；
- `dki/bench/npu/om_manifest.py` 绑定 SoC/CANN/OM/tokenizer/labels 与完整 IO signature；`om_runtime.py` 保证单 owner、整 worker 最多重建一次；
- `dki/nlp/windowing.py` 提供全局字符坐标、overlap/core ownership 和稳定去重，超过 seq256 不再由新路径静默截断；
- `dki/bench/npu/medical_ner_om_adapter.py` 是**生产级医疗 NER ACL 适配器**：把 text→tokenize→OM execute→argmax→canonical BIO `decode_typed`→全局 span 还原→词典归一→entities 收敛为实现 `open()/execute(text)/close()` 的对象，可直接作为 `MedicalEntityExtractorOmShadow` 的 `shadowRuntimeFactory`（`create_process_runtime`），长文本自动走 windowing 分窗合并、加载前经 manifest 强制门；重依赖（acl/transformers）惰性导入，CPU 机可导入与打包；
- `dki/bench/npu/om_scheduler.py` 是**服务级微批调度器**（本地语义）：绝对 deadline（排队等待计入端到端、超时先丢弃不占 NPU）、`maxBatchSize`/`maxWaitMs` 双触发成批、批内稳定重排与顺序无关回填、执行前 cancellation、静态定长 batch 的 tail padding+mask、kill switch、脱敏遥测（queue_wait/exec/fill，无原文）。**这是本地服务语义，`formal_microbatch=false`，不等于真实 DataMate 跨文档微批**；
- 真实 DataMate 平台 smoke：`scripts/datamate_smoke.py` 从 6 个算子市场 ZIP 出发按 `ops.user.<pkg>` 布局 upload→import→用真实 `datamate.core.base_op.OPERATORS` instantiate→invoke→shadow no-impact + 故障隔离，回执 `outputs/verification/datamate_smoke.json`；
- `MedicalEntityExtractorOmShadow` 是独立候选 NPU 算子（`runtime.npu=1`），默认 `pipeline_kg_build.yaml` 仍为 dict/NPU=0。候选配置 `configs/pipeline_kg_build_shadow_om.yaml` 也默认 `shadowEnabled=false`；只有目标机提供已验收的 `shadowRuntimeFactory` 且 OM/tokenizer/labels hash 与 ABI 全通过、并经环境变量 `DKI_SHADOW_HASH_KEY` 提供 sidecar HMAC 密钥后才能开启，启用但缺 runtime 或密钥会 fail closed，绝不静默 no-op。queue full、无 NPU、坏 hash、timeout、worker crash 和 sidecar 写失败均不改变权威输出；
- **manifest 强制门**：`om_runtime.py` 的进程/单 owner 加载路径在 `adapter.open()` 之前强制消费 `OmManifest.verify_artifacts + verify_runtime`（OM/tokenizer/labels hash + SoC/CANN/IO signature），校验失败直接 fail closed，不进入 ACL 加载。

当前 Release A 的本地契约/fake runtime/no-impact 测试已通过；CPU 发布验证 CLI `scripts/verify_innovation_release.py` 连续通过主输出等价、section/span_base 兼容、故障隔离、严格 exact/relaxed 分栏、长文本全局坐标、受限恢复、校准路由与 prospective receipt，产出 `outputs/verification/innovation_release_cpu.json`。这些是 **CPU 契约证据，不是新 NPU 成绩**；在目标 DataMate 未证明实际调用 `map_batches/execute_batch` 前，证据字段保持 `formal_microbatch=false`，不把单样本 shadow 或离线 probe 称作正式跨文档微批。

## 5.10 Treatment-sentinel 与后续晋级门

RBT6-INT8-fast 的 **5961 samples/s 仍是历史 extreme-throughput 点，不是质量安全默认**：其聚合 F1=0.9490，但 Treatment 为 TP=0/FN=81/F1=0。新增 `dki/nlp/treatment_sentinel.py` 将 Treatment recall、worst-type F1、macro-F1、overall F1、吞吐保留率共同设为 promotion gate；`quantize_ner_int8.py` 在工具链不支持 `disable_names` 时不再静默退化成全 INT8，并输出量化审计 manifest。

选择性混合精度、hot-shape lattice、校准路由都必须产生新命名工件并通过目标 NPU 实测后才可 promotion。目前冻结 BERT-INT8（3168 samples/s、F1 0.9812）继续作为最强可辩护平衡点；没有新增实机数字时不改写历史性能表。

## 5.11 目标机晋级门（尚未通过，晋级前不改写历史性能表）

§5.9/§5.10 描述的 shadow 算子、manifest 门与 Treatment-sentinel 均已有 CPU 契约证据；
以下四项必须在目标 Ascend 环境**另行**实测通过，任一失败即保持 `promotion=false`：

1. pyACL/CANN/SoC/OM manifest 兼容、100 次 init-execute-close 生命周期、60 分钟 soak 无残留进程；
2. PyTorch/OM exact parity 与 Treatment-sentinel 选择性混合精度（失败即 `promotion=false`）；
3. DataMate 真实 `map_batches` 平均 fill>1，方可称正式跨文档微批；否则仅称 shadow batch=1/local probe；
4. 真实到达 hot-shape lattice 的 p95/p99、吞吐与 monotonic marker 同步 joule 积分。

> 2026-07-20 目标机（910B2C）已实测通过第 1 项（100/100 生命周期、3600s soak）与第 2 项的
> exact parity（FP16b F1=1.0、INT8≈0.986 非 exact-equivalent）；并**在真实 INT8 OM 上验证了生产级
> 医疗 NER 适配器 text→实体全链路**（3 段病历抽出 [7,8,4] 个带类型实体，进程独占 runtime 与直连
> 逐条一致，manifest 篡改在 ACL 加载前 fail-closed；`outputs/remote_acceptance_20260720_medical_ner_adapter.json`）。
> 第 3 项：目标机安装 `ray[data]+datamate+nexent` 全栈后，真实 `RayDataset.process` 在 200 份 EMR 上
> 跑通（1569 三元组，`outputs/remote_acceptance_20260720_datamate_ray.json`），且全量 pytest 在该真实栈上
> `195 passed / 0 failed`。但实测确认 DataMate 对 Mapper 用 `ray.data.map` **逐样本派发（fill=1）**，
> 同机 `map_batches` 虽可 fill=16 却非 Mapper 执行路径，故正式跨文档微批仍 `formal_microbatch=false`（now measured, not assumed）；
> 第 4 项 lattice 仅 CPU 仿真。故整体保持 `Release B not promoted`。

## 5.12 证据闭环：typed claims + 托管清单 + 统计合同

- **Typed claims**：`scripts/build_evidence_claims.py` 从既有产物生成枚举化、带工件 SHA-256 的
  headline claims（`outputs/claims/headline_claims.json`），`scripts/validate_evidence.py`
  逐条校验；claim 明确区分 `dataset_role`(development/blind)、`metric_kind`(exact/relaxed)、
  `evidence_level`(probe/formal)、`runtime_kind`(cpu/npu/shadow)、`promotion_state`。Notebook §10
  用 `load_validated_claims` 只读校验通过的 claims，显示数字与磁盘证据强绑定。
- **性能统计合同**：`dki/evidence/stats.py` 对 5 次正式重复给出 mean/std/median/p95 +
  确定性 bootstrap 95% CI（910B2C INT8 吞吐 3048 样本/s，CI95≈[3019, 3073]）；并区分
  request-latency（调度器排队+执行）与 batch-execution-latency（单次 OM execute）。
- **证据托管**：`scripts/build_evidence_custody.py` 产出 `outputs/verification/evidence_custody_*.json`：
  内容指纹 + 来源/环境 provenance + 逐指标复跑命令 + 大 OM 受控 URI/指纹 + HMAC 签名 + ISO 时间戳
  （`DKI_EVIDENCE_HMAC_KEY`），做到本地可审计、防篡改；未做外部 WORM/第三方公证，不作过度声称。
- **final3 内容门**：`dki/evidence/final3_guard.py` 以内容哈希（重命名也拒绝）把 final3/blind 挡在
  training/calibration/selection/routing/profiling/threshold/quantization/error-analysis 之外，
  已接入 `scripts/train_ner.py`、`scripts/train_ner_quality.py`、`scripts/quantize_ner_int8.py`、
  `scripts/profile_ner_workload.py`。
- **hot-shape lattice（CPU 仿真，非实机成绩）**：`dki/nlp/hot_shape.py` 规划 2–4 个静态 shape 节点
  （每节点 shape 签名 + padding 浪费统计 + overflow 节点），`simulate_arrivals` 做 open/closed-loop
  到达回放，`compare_fixed_vs_lattice` 给同 trace 下 fixed-seq256 vs lattice 的 padding/等待/吞吐对照
  （`scripts/simulate_hot_shape_lattice.py` → `outputs/verification/hot_shape_lattice_sim.json`）。
  `promotion_state=cpu_simulation_only_not_npu_measured`——真实 p95/p99 与能效 Pareto、每节点 exact
  parity 仍须目标机实测（见 Release B 状态 `outputs/verification/phase4_release_b_status_20260720.md`）。

## 5.13 完美态 fail-closed 降级声明（2026-07-25）

在目标机未完成下列任一项前，**禁止**把对应数字写入提交成绩表或 README headline：

| 门 | 状态 | 对外表述 |
|----|------|----------|
| OM shadow 生产开启 | 默认 `shadowEnabled=false` | 仅候选算子，非默认路径 |
| DataMate Mapper `formal_microbatch` | 实测 fill=1 → **false** | 「本地服务微批 / probe」，非跨文档正式微批 |
| hot-shape lattice 实机 p95/p99 | 仅 CPU 仿真 | `cpu_simulation_only_not_npu_measured` |
| CMeEE GlobalPointer 上 NPU/OM | 训练可用 torch_npu；OM 未绑 CBLUE ckpt | 不以 CBLUE F1 冒充已 OM 加速 |

一句话：**有 CPU 契约 ≠ 有正式 NPU 成绩**；缺门就降级写明，不虚报。

## 5.14 CMeEE schema-v3 OM 正式验收（2026-07-25）

§5.13 中“CMeEE OM 未绑定”的缺口已在同一 Ascend 目标机补齐；该行现更新为：

| 门 | 状态 | 可引用结论 |
|---|---|---|
| CMeEE checkpoint→ONNX→ATC→OM 绑定 | **通过** | checkpoint/ONNX/OM/manifest/ATC/IO 全部 SHA-256 绑定 |
| 官方 CMeEE dev 质量差 | **通过** | FP16 OM F1 0.613020，FP32 ONNX 0.613214，绝对差 0.000194 |
| NPU 串行/微批一致 | **通过** | 同 200 EMR，200/200 文档完全一致 |
| 三次中位数性能 | **通过** | NPU 微批 326.098 docs/s；CPU ONNX 18.393；17.7295x |
| 同步能耗 | **通过** | 覆盖 0.936；0.664121 J/doc；扣 173.1 W idle 后 0.157099 J/doc |
| DataMate Mapper 原生跨文档微批 | **未通过** | 平台 Mapper 仍 fill=1；上述为真实生产适配器 `execute_batch`，不改写该事实 |

FP16 与 FP32 在 200 EMR 上不是逐文档完全相同（190/200），但实体级 F1
0.997604；最终质量门使用更强的 5,000 条官方 dev 主指标差值，而 NPU 串行与 NPU
微批本身保持 200/200 完全一致。完整环境、失败尝试、哈希与复跑口径见
`docs/remote_npu_acceptance_20260725.md`。

## 6. 可解释性小结

- 选型可解释：只把 GEMM 密集的 BERT 前向放上 NPU；串处理保持 CPU（词典后端本身已 0.1ms 级）；
  模型层数按任务难度选（rbt6 甜点）；
- 优化可解释：每一项优化都对应昇腾架构的具体约束（图编译缓存/二进制算子/Cube 利用率/
  INT8 通路/tokenize 瓶颈下沉）；
- 数据可解释：warmup 单列、CPU 对照同逻辑、能效采集方法给出、量化后精度实测、
  **OOD 鲁棒性诚实披露**——对比分析可复现、可审计。
