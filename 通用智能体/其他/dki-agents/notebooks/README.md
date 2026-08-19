# dki-agents 端到端演示 Notebook

`dki_end_to_end_demo.ipynb` 用一个 Notebook 串起赛题三任务，形成「数据 → 知识 → 洞察」完整闭环，评审可交互复现。

## 运行方式

```powershell
cd C:\Users\shenh\Desktop\比赛专用\ModelEngine开源项目贡献赛\dki-agents
$env:PYTHONUTF8 = 1

# 安装 Notebook 可选依赖（首次）
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
# 或直接： .\.venv\Scripts\python.exe -m pip install jupyter matplotlib

# 方式一：交互式打开
.\.venv\Scripts\jupyter.exe lab notebooks\dki_end_to_end_demo.ipynb

# 方式二：命令行一键执行（无界面，显式使用项目 venv 内核）
.\.venv\Scripts\python.exe -m ipykernel install --user `
    --name dki-agents-verify --display-name "Python (dki-agents verify)"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute `
    --ExecutePreprocessor.timeout=600 `
    --ExecutePreprocessor.kernel_name=dki-agents-verify `
    --output executed_demo.ipynb notebooks\dki_end_to_end_demo.ipynb
```

## 内核与依赖

- **内核**：项目 `.venv`（已含 Nexent/DataMate 依赖）。
- **可选依赖**：`jupyter`、`matplotlib`（数值图表）。BI 报告用 `IPython.display.IFrame`
  内嵌现成 ECharts HTML，CJK 走浏览器渲染，无字形问题。
- **PYTHONUTF8=1**：Windows 下必须，否则中文读写乱码。

## 章节

| # | 章节 | 说明 |
|---|---|---|
| 0 | 环境自检 | 项目路径、数据目录 |
| 1 | 生成/加载数据 | 200 份含噪合成病历 |
| 2 | 任务一 清洗 | DataMate 算子流水线，清洗前后对比 |
| 3 | 任务二 图谱 + 评测 | 抽取三元组 vs 金标 **F1≈99.97%** |
| 4 | 任务二 图谱问答 | 模板引擎，答案带证据句 |
| 5 | NER 微调（任务一加分）| dict 远程监督微调 bert-base-chinese，三后端 F1 对比柱状 |
| 6 | 任务三 NL2SQL | 自然语言 → SQL 查询 |
| 7 | 任务三 BI 报告 | ECharts 图文报告内联 |
| 8 | NPU 优化（任务二加分）| FP16/INT8/rbt6/fast-tokenize 吞吐/F1 阶梯 |
| 9 | OOD 鲁棒性 | 原 BERT 与可选 quality 模式 P/R/F1（开发集 quality: 0.948/0.952/0.950；quality v2 在盲测-2 达 0.702/0.880/0.781；预先冻结后生成的 final3 上单 v2=0.821/0.740/0.778、recall mode=0.769/0.809/0.789） |

## 说明

- 第 5、8、9 章读取服务器训练/benchmark 产物（`outputs/ner_finetune/metrics.json`、
  `outputs/ner_finetune/ood_eval_quality_calibrated.json`、`outputs/bench/npu_int8/*.json`）。这些由服务器上的 `scripts/train_ner.py`、
  `scripts/quantize_ner_int8.py`、`scripts/acl_om_pipeline_bench.py` 生成后回传；
  本机无 torch/NPU，故 Notebook 读产物展示而非现场推理。
- 其余章节全部本机现算，与 CLI（`python -m dki ...`）同源，零网络依赖。
