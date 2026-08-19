# 本地小模型微调与验证（实验过程说明）

本文件记录三个任务的「本地微调小模型」链路的完整实验过程、复现步骤、问题与对策，以及
验证结果，供 NPU 适配参考作为 CPU/GPU 基线。

本地小模型和 LLM API 是同一类可选增强层的两种实现：有本地 adapter 时可用 `--local-model`
离线增强；没有 adapter 或需要在线能力时，可用 `--llm-config` 读取 LLM API key，完成同类
规划、抽取或 NL2SQL 增强。默认调度顺序仍是 `本地模型 > LLM > 规则/模板`，任一层缺失都会
回退，不影响主链路运行。

---

## 1. 验证环境

| 项 | 值 |
| --- | --- |
| 操作系统 | Windows 10 |
| GPU | NVIDIA GeForce RTX 5070 Laptop（8 GB，可用约 6.4 GB） |
| CUDA | 12.8 |
| PyTorch | `2.11.0+cu128` |
| 关键依赖 | `transformers`、`peft`、`datasets`、`accelerate`（`trl` 缺失时自动回退 `transformers.Trainer`） |
| 基座模型 | `Qwen/Qwen2.5-0.5B-Instruct` |

> HuggingFace 在本环境不可达。基座模型通过 **ModelScope** 下载：
> ```bash
> python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))"
> ```
> 下载路径示例：`~/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct`。
> 训练/推理时建议设 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，避免联网卡顿。
> （备选：设 `HF_ENDPOINT=https://hf-mirror.com` 走 HF 镜像。）

---

## 2. 各任务的本地模型链路

| 任务 | 微调目标 | 训练脚本 | 推理模块 | 共享提示词（统一规范） |
| --- | --- | --- | --- | --- |
| 任务一 | 数据处理算子规划 | `src/training/finetune_small_model.py` | `src/agents/data_processing_agent/local_model_planner.py` (`predict_plan`) | `local_model_planner._SYSTEM_PROMPT` |
| 任务二 | 医疗实体/关系抽取 (NER) | `src/training/finetune_kg_model.py` | `src/operators/kg_ops/local_model_ner.py` (`predict_kg_entities`) | `src/operators/kg_ops/kg_prompts.py` |
| 任务三 | 分析规划 | `src/training/finetune_analysis_model.py --task planning` | `src/operators/analysis_ops/local_model_planning.py` (`predict_plan`) | `src/operators/analysis_ops/analysis_prompts.py` |
| 任务三 | NL2SQL | `src/training/finetune_analysis_model.py --task nl2sql` | `src/operators/analysis_ops/local_model_nl2sql.py` (`predict_sql`) | `src/operators/analysis_ops/analysis_prompts.py` |

**统一规范原则**：训练数据生成、微调脚本、推理三个环节共用同一份 system prompt 与 instruction。训练阶段用什么提示词，推理阶段就必须用同一套提示词（见第 5 节常见问题）。

---

## 3. 复现步骤

以下命令在仓库根目录 `nexent-dkm-agent/` 下执行。`<BASE>` 指向第 1 节下载的
基座模型目录。

```bash
# 0) 设离线环境变量（PowerShell）
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"

# 1) 生成训练数据（输出在 data/training/，作为本地复现产物保留）
python data/training/generate_training_data.py            # 任务一: 1600 train + 400 val
python data/training/generate_kg_training_data.py          # 任务二: 1800 train + 200 val
python data/training/generate_analysis_training_data.py    # 任务三: 各 600 (planning / nl2sql)

# 2) 微调（QLoRA / LoRA, 产物在 data/training/*_model_output/final）
# 任务一（脚本内置 4-bit，可选 bitsandbytes）
python src/training/finetune_small_model.py --model-name "<BASE>" \
    --output-dir data/training/model_output --epochs 3 --batch-size 2

# 任务二（KG NER）
python -m src.training.finetune_kg_model \
    --train-data data/training/kg_extraction_train.jsonl \
    --val-data data/training/kg_extraction_val.jsonl \
    --model-path "<BASE>" --output-dir data/training/kg_model_output --epochs 3 --batch-size 2

# 任务三（planning 与 nl2sql 各训一个 adapter）
python -m src.training.finetune_analysis_model --task planning \
    --train-data data/training/analysis_planning_train.jsonl \
    --val-data data/training/analysis_planning_val.jsonl \
    --model-path "<BASE>" --output-dir data/training/analysis_planning_model_output --epochs 3 --batch-size 2
python -m src.training.finetune_analysis_model --task nl2sql \
    --train-data data/training/analysis_nl2sql_train.jsonl \
    --val-data data/training/analysis_nl2sql_val.jsonl \
    --model-path "<BASE>" --output-dir data/training/analysis_nl2sql_model_output --epochs 3 --batch-size 2

# 3) 端到端验证（把 adapter 目录传给 --local-model）
python demos/task3_demo.py \
    --local-model data/training/analysis_planning_model_output/final \
    --task-request "分析核心枢纽节点，找出关键节点和社区结构" --question "各类实体的数量分布"
```

> adapter 目录会记录 `base_model_name_or_path`（即 `<BASE>`），推理时据此本地加载
> 基座；务必保证该路径在推理机上仍然存在。

---

## 4. 验证结果（真实 GPU 运行）

下表 `eval_loss` 与超参均回读自各 adapter 输出目录中的
`trainer_state.json` / `adapter_config.json`（即训练器写下的真实记录，非手填）。
所有 adapter 均为 LoRA，`r=16 / alpha=32`，target modules 为
`q_proj, k_proj, v_proj, o_proj`。最新一轮在 NVIDIA RTX 5070 Laptop GPU
（CUDA 12.8 / torch 2.11）上于 2026-06-24 重新训练并验证。

| 任务 / adapter | adapter 目录 | epochs | train_loss | eval_loss | 验证输出 |
| --- | --- | --- | --- | --- | --- |
| 任务一 规划 | `model_output` | 3 | 0.2228 | 0.0005123 | `predict_plan` 输出合法算子计划（load_csv→…→export_clean_dataset） |
| 任务二 NER | `kg_model_output/final` | 3 | 0.06095 | 0.005754 | 样例文本完整抽出 Disease/Symptom/Drug/Examination/Treatment，`local_model=active` |
| 任务三 规划 | `analysis_planning_model_output/final` | 3 | 0.8325 | 0.06268 | planning adapter 加载成功，`quality=passed`，按意图选出 graph_analytics 算子 |
| 任务三 NL2SQL | `analysis_nl2sql_model_output/final` | 3 | 0.3679 | 0.001397 | 生成与模板一致的 canonical SQL，`NL2SQL=local_model_generated`，4 rows |

> 上述 loss 为修复后结果。修复前（见第 5 节）任务二/三 train_loss 高达 6+，模型几乎无法收敛、推理输出为空。本轮四个 adapter 均使用默认超参（3 epochs，
> LoRA r=16/alpha=32）在同一 GPU 上连续训练，验证集 loss 均收敛。

---

## 5. 常见问题与修复记录（重要）

### 问题 1：QLoRA 标签错位导致模型无法收敛
`finetune_*` 早期的 `tokenize_fn` 把 `input_ids` 设为「纯 prompt」、`labels` 设为
「纯 output」，二者各自 padding 到 `max_length`，导致监督信号完全错位（模型被迫在
prompt 的每个位置去预测 output 的 token）。表现为 `train_loss` 居高不下、推理输出空字符串。

**修复**：改为标准 SFT 拼接——`input_ids = prompt + response`，对 prompt 与 padding
位置打 `-100`，只对 `response + EOS` 计算损失。三个训练脚本均已统一为此写法。

### 问题 2：训练与推理 prompt 不一致
任务三规划链路早期复用了**任务一**的 `predict_plan`（英文 system + `Task:/Input:`
格式），与任务三训练所用的中文 instruction 完全不同；NL2SQL 推理也用了与训练不同的
prompt。结果：微调 adapter 在推理时拿到陌生 prompt，退化成基座行为甚至输出为空。

**修复**：抽出 `analysis_prompts.py` / `kg_prompts.py` 作为统一规范来源，生成器、微调、推理三个环节共用；任务三新增独立的 `local_model_planning.predict_plan`；任务一让微调脚本直接 `import _SYSTEM_PROMPT` 与推理对齐。并加了回归测试，确保「训练 instruction 与推理 prompt 一致」。

### 问题 3：基座模型下载路径
HuggingFace 直连不可达。改用 ModelScope 下载（见第 1 节）。注意 ModelScope 缓存目录名
含 `___`（如 `Qwen2___5-0___5B-Instruct`），传 `--model-path` 时用实际目录名。

---

## 6. NPU 适配参考

### 当前 device adapter 状态

- 共享适配器：`src.common.device.get_device()` 已实现 `npu` / `cuda` / `cpu` 选择，
  且不会在模块导入时强依赖 accelerator 包。
- 推理接入：任务一 local planning、任务二 local-model NER、任务三 local-model
  planning/NL2SQL 已通过共享适配器加载模型。
- Ascend 行为：检测到 `torch_npu` 与 `torch.npu` 可用时，Hugging Face 模型加载不再使用
  CUDA 的 `device_map="auto"`，加载后统一移动到 `npu:0`。
- 训练脚本仍以 CUDA/CPU 为主，建议先在 Ascend 服务器验证本地小模型推理，再迁移训练。

本链路即为 NPU 适配的 **CPU/GPU 基线**。移植要点：

1. **依赖**：`requirements-npu.txt` 只安装非 torch Python 依赖；Ascend 服务器需先按
   CANN/torch_npu 专用流程激活匹配运行时。当前 910B3 快照为 CANN 8.5.0、`torch 2.9.0+cpu`、
   `torch_npu 2.9.0`；历史 910B2C 为 CANN 8.5.1、`torch_npu 2.9.0rc1`。不要混装 CUDA torch。
2. **设备选择**：四个推理模块与三个训练脚本已集中使用 `src.common.device.get_device()`
   或相关加载辅助函数处理 cuda/npu/cpu。NPU 上检测到 `torch_npu` 与 `torch.npu`
   可用时，模型加载后统一移动到 `npu:0`。
3. **量化**：任务一的 4-bit（`bitsandbytes`）仅支持 CUDA；NPU 上请关闭量化（脚本在
   缺包时已自动降级为全精度），或换用昇腾支持的量化方案。
4. **基线对照**：保留本文件第 4 节的 CPU/GPU loss 与输出作为正确性对照；NPU 复现后
   应得到同量级 loss 与等价的推理输出。
5. **算子级 NPU 优化**：见 [npu_optimization.md](npu_optimization.md)（已有 CPU 基线对照
   与 `torch_npu` 运行时探测，未检测到 NPU 时记录 `unavailable`，不伪造结果）。

> Ascend 910B3 服务器 NPU 验证：**770/770 pytest passed**（NPU 专项 **43/43**，2026-06-24 最新 NPU 快照，`final-version`；2026-06-16 历史为 380/380；2026-06-14 历史为 289/289）。本文第 4 节仍是本地 CPU/GPU 小模型训练基线；NPU 性能结论以 [npu_optimization.md](npu_optimization.md) 和 [server_environment.md](server_environment.md) 为准。Windows 主链路 pytest 见根目录 [README.md](../README.md#工程校验)（**437/437**，2026-07-03 证据包）。
