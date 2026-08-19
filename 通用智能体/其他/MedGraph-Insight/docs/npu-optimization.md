# NER 算子 NPU 优化实现与性能对比分析

> 对应任务二加分项「完成算子的 NPU 优化实现，并验证性能与能效提升效果」。
> 复现命令：`mgi bench --repeat 5 --scale 20 --markdown output/benchmark.md`

## 1. 优化目标与思路

医疗实体识别（NER）的核心是**多模式字符串匹配**（词典 N 个词条在文档中定位）。
CPU 上的最优解是 Aho-Corasick 自动机（亚线性、跳转驱动、含数据依赖分支）。
但**分支跳转 + 不规则内存访问**不适配 NPU（昇腾 Ascend）的向量核（Vector Core）与达芬奇架构——
NPU 擅长的是**规则、无分支、大批量的稠密张量运算**。

因此本项目把 NER **重写为稠密张量算子**（`mgi/operators/npu/ner_npu.py`）：

1. 文本批量编码为 Unicode 码点矩阵 `T ∈ int32[B, L]`（padding=0）；
2. 词典编码为码点矩阵 `P ∈ int32[N, K]`（padding=0）；
3. 对每个模式内偏移 `k∈[0,K)`：广播比较 `T[:, l+k] == P[:, k]` 得到 `bool[B, N, L]`，
   再沿 `k` 做逻辑与归约——**一次张量运算完成全部「文档×词条×位置」的命中判定**，
   全程无数据依赖分支，天然映射到 NPU 的 SIMD 向量核并行。

设备三级后端（同一套张量代码）：`torch_npu`（昇腾 CANN）→ `torch` CPU → NumPy 模拟，
保证 CI/无 NPU 环境可复现，且结果逐位一致（`tests/test_operators.py::test_npu_backend_matches_cpu`）。

## 2. 本机实测（NumPy 模拟后端，800 文档 × 253 词条）

| 后端 | 计算类型 | 耗时(ms) | 吞吐(字符/秒) | 延迟(ms/文档) | 字符比较次数 | 峰值张量内存(MB) |
| --- | --- | --- | --- | --- | --- | --- |
| cpu-naive | 亚线性扫描 | 72.4 | 999,830 | 0.0905 | —(亚线性) | — |
| cpu-aho-corasick | 亚线性自动机 | 35.4 | 2,042,037 | 0.0443 | —(亚线性) | — |
| npu-tensor[numpy] | 稠密并行 | 192.0 | 376,868 | 0.240 | 271,418,400 | 28.76 |

> 数字随机器波动；以 `mgi bench` 实测为准。

## 3. 对比分析（关键结论）

**为什么 CPU 上稠密张量后端更慢，却仍是正确的 NPU 优化方向？**

- **工作量差异**：稠密算子做了 `B·N·L·K ≈ 2.7 亿` 次字符比较（把稀疏匹配问题「稠密化」以换取规则并行）；
  Aho-Corasick 是亚线性的，在**串行标量处理器（CPU）**上自然更省。
- **硬件适配差异**：稠密算子的 2.7 亿次比较**彼此独立、无分支**，可被 NPU 的数千个向量通道
  一次性并行；而 Aho-Corasick 的状态跳转是串行数据依赖，**无法向量化**，在 NPU 上反而低效。
- **因此**：CPU 场景选 Aho-Corasick（`mgi build` 默认）；NPU 场景选张量算子（`mgi build --npu`）。
  本仓库同时提供两条路径并给出量化对比，而非二选一。

**能效 / 资源代理指标**：稠密算子峰值中间张量 `bool[B,N,L]` 约 28.8 MB、比较吞吐约 141 万次/ms。
在昇腾 NPU 上，这些独立比较由向量核并行摊薄，预期吞吐随 batch 规模近线性提升、单位能耗显著下降
（真机数据见 §4）。

## 4. 昇腾真机验证（Ascend 910B2C 实测）

已在 Ascend 910B2C（HBM 64 GB，CANN + `torch_npu 2.8.0`）真机完成验证，
完整数据见 [benchmark-ascend.md](benchmark-ascend.md)。要点：

- **设备端核函数 ≈ 7,402 万次字符比较/ms**（`torch.npu.synchronize()` 计时），
  较 CPU 侧比较吞吐高约 50 倍，验证「稠密化换并行」设计；
- 三后端（naive / Aho-Corasick / NPU 张量）在 800~80,000 文档全部规模下**实体结果逐位一致**；
- 端到端瓶颈在主机侧解码与传输。针对性完成三项工程优化：
  1. **设备端稀疏化**：`torch.nonzero` 后仅回传命中索引 `[M,3]`，不再搬运整块 `bool[B,N,L]`；
  2. **批维分块**：单块中间张量 ≤256 MB，80,000 文档规模无 OOM；
  3. **解码向量化**：NumPy 过滤 + `searchsorted` 分组替代逐命中 Python 循环。
  三项合计将 40,000 文档端到端耗时 6.0 s → 2.7 s（2.2×）。

复现命令（CANN 环境自动切换 `npu` 设备）：

```bash
pip install torch_npu
mgi bench --repeat 5 --scale 50 --markdown output/benchmark_npu.md
```
