> 英文简要说明；完整中文流程见 [npu_optimization.md](../../docs/npu_optimization.md) 与 [preparation.md](../../docs/preparation.md)。

# Benchmark Scripts

Shell helpers for Ascend NPU verification and environment collection.

| Script | Where to run | Purpose |
| --- | --- | --- |
| `run_npu_full_verify.sh` | **Ascend Linux only** | One-shot 16-step NPU regression (Demos, pytest, benchmarks, env snapshot) |
| `run_full_verify.sh` | Ascend Linux | Compatibility alias → `run_npu_full_verify.sh` |
| `collect_env.sh` | Ascend Linux | Print CANN/torch/npu-smi versions for evidence |

Windows/WSL does **not** run NPU scripts. For Nexent/DataMate/Neo4j on Windows, use WSL — see [docs/preparation.md](../../docs/preparation.md#windows--wsl-开发环境重要).

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
source /data/npu_env.sh
python -m pip install -r requirements-npu.txt
bash benchmarks/scripts/run_npu_full_verify.sh
```

Optional: `SKIP_XLARGE=1` · `SKIP_REACHABILITY=1` · `SKIP_ENV_SNAPSHOT=1`
