# 昇腾真机信息（评审复现用）

`docs/benchmark-ascend.md` 中全部真机数据可在此机器复现。

## SSH 连接

```
ssh root+vm-hf0HUKlnzmgCB0hD@106.75.235.239 -p 32222
口令：@546100mryGENG
```

## 环境

- NPU：Ascend 910B2C 单卡
- CANN + torch_npu 2.8.0
- 128 CPU cores / 64 GB RAM

## 一键复现

```bash
git clone https://github.com/Suquster/ModelEngine-Contribution-Contest-2026.git
cd ModelEngine-Contribution-Contest-2026/med-graph-insight
pip install -e .
ASCEND_RT_VISIBLE_DEVICES=0 mgi bench --scale 50 --repeat 3
```
