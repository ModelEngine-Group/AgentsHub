# Remote CMeEE NPU acceptance - 2026-07-25

## Environment

- Accelerator: Ascend 910 (`Ascend910_9382`)
- Driver / `npu-smi`: 25.5.1
- CANN: 9.1.0 beta 3
- PyTorch / torch_npu: 2.7.1 / 2.7.1.post6
- CPU reference: ONNX Runtime 1.20.1, aarch64 CPUExecutionProvider
- Corpus: the same 200 generated EMR documents for CPU and NPU

## Bound artifacts

| Artifact | SHA-256 |
|---|---|
| CMeEE checkpoint tree | `f0420fdcf86516cf126e0e9055d22c6f0c9c80d7feedddab5f72767e59d2c67f` |
| ONNX | `48a32f43e87183ee4ea331fea8e46fd72638d137a6c4a33f1cde458d396ba597` |
| FP16 OM | `ea4b3095c5295d24c71ba1a10fda9b704ea14a81e017fbb9b3c17652c3d89568` |
| Schema-v3 bound manifest | `8cb277cf164dfc3918ea899e51c89f214331c683bdf4ed65a3c176066d974fab` |
| ONNX export receipt | `4b522cdd5595a88c25689cf39364098bfc795478e16698675c7c532312a4f8bb` |

The strict preflight passed checkpoint, artifact, runtime, CBLUE and IO checks.
The OM output name differs from the ONNX friendly name, so the manifest binds
the output by tensor order, dtype and shape and records the explicit name map.

## Official quality

The deployment backends were evaluated on all 5,000 official CMeEE-V2 dev
documents using exact span and type:

| Backend | Precision | Recall | Micro-F1 | Throughput |
|---|---:|---:|---:|---:|
| FP32 ONNX CPU | 0.584110 | 0.645369 | 0.613214 | 19.783 docs/s |
| FP16 OM NPU | 0.583679 | 0.645468 | 0.613020 | 225.791 docs/s |

The absolute Micro-F1 delta is **0.000194**, below the predeclared 0.001
quality gate. The OM logit threshold `-1.005` was selected on the first 100
project EMRs and independently checked on the second 100; official CMeEE dev
was evaluation-only.

This candidate is not an unconditional replacement for the frozen 0.6031
soft-logit ensemble. Its overall F1 is higher, but weak types remain:
`ite` F1 0.3581, `dep` F1 0.4353 and `sym` F1 0.4664.

## Performance and energy

Formal v3 uses three post-warmup repeats and reports the median:

| Metric | Result |
|---|---:|
| NPU serial | 43.033 docs/s |
| NPU cross-document microbatch | **326.098 docs/s** |
| Serial-to-microbatch speedup | **7.5778x** |
| CPU ONNX reference | 18.393 docs/s |
| NPU-to-CPU speedup | **17.7295x** |
| NPU serial vs microbatch parity | 200/200 exact documents |
| CPU vs NPU entity parity on the 200 EMRs | F1 0.997604 (4 FP / 6 FN) |
| Power-window throughput | 319.543 docs/s |
| Power coverage | 0.9360 |
| Energy | 0.664121 J/document |
| Idle-subtracted energy | 0.157099 J/document |

Idle power was measured separately for 30 seconds: median 173.1 W, coverage
0.9826. The benchmark power window repeated the same 200-document corpus for
10 seconds (16 passes, 3,200 document inferences), because one pass took less
than one `npu-smi` sampling interval.

## Failed attempts retained

- v1 used PyTorch CPU batch 8. It was stopped after 27 minutes because the
  already-completed power window had only one sample and could not pass.
- v2 introduced a 10-second power window and ONNX CPU reference. Performance
  and power passed, but the old exact-document CPU/NPU gate failed at 187/200.
- A separate official-dev quality run then proved the FP16 quality delta.
  v3 accepts the bounded FP16 difference while retaining exact parity between
  NPU serial and NPU microbatch paths.

## Evidence

All small receipts are under
`outputs/remote_runs/plan_v1/cmeee_om_s42_bs16/`:

- `preflight_strict.json`
- `official_dev_om_vs_onnx_v1.json`
- `om_threshold_calibration_first100_v1.json`
- `idle_power_v1.summary.json`
- `acceptance_v3/cmeee_cpu_npu_acceptance.json`
- `acceptance_v3/cmeee_npu_power.summary.json`
- `acceptance_v3/cmeee_npu_release_receipt.json`

The 405 MB ONNX and compiled OM remain on the controlled target host. Their
hashes, receipts and compile log are stored locally; they are not silently
embedded in the source submission ZIP.
