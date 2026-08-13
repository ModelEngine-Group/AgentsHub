# Remote NPU acceptance - 2026-07-23

## Environment

- Architecture: AArch64
- Accelerator: two Ascend 910 devices
- CANN: 9.1.0 beta 3
- PyTorch: 2.7.1
- torch_npu: 2.7.1.post6
- Source bundle SHA-256:
  `ca50925ec6f03be8e90bd89efa8dda4e2b0cfa50ddfc03a36ffbb6bbd03b4efb`
- Remote targeted tests: 60 passed
- Local full suite: 270 passed, 2 skipped

The final remote preflight was run in strict mode with the versioned CMeEE
checkpoint and the complete hash-bound OM bundle. It reported
`ready_for_explicit_npu_runs=true`, including
`cmeee_checkpoint_ok=true`, `om_manifest_ok=true`, and `runtime_ok=true`.

## CBLUE results

| Candidate | Selection boundary | Official dev result | Best epoch | Train time |
| --- | --- | --- | ---: | ---: |
| CMeEE seed 42 | train-derived, text-disjoint internal split; checkpoint and threshold selected before official dev | F1 0.5935, P 0.5634, R 0.6269, threshold -1.0 | 8 | 3036.3 s |
| CMeIE seed 42 | fit-only predicate vocabulary and union of all fit-only type-signature constraints | F1 0.6560, P 0.6561, R 0.6558 | 2 | 3175.2 s |
| CMeIE seed 43 | same protocol, independent seed | F1 0.6559, P 0.6558, R 0.6559 | 3 | 2427.7 s |

CMeIE is a gold-pair, gold-type-signature predicate-classification result. It
is not an end-to-end SPO extraction score. The absolute difference between the
two official-dev seed results is 0.0001.

## Rebuilt FP16 OM

- Checkpoint: `bert-base-chinese`, seed 42, five epochs, 160 fit documents.
- Checkpoint eval: entity F1 0.9936, P 0.9923, R 0.9948 on the isolated
  40-document eval set; Treatment F1 1.0.
- ONNX SHA-256:
  `433f0289b26d11090246c3ca5a47a7566bb11e28cb76e93e7fb7dec530299986`.
- OM SHA-256:
  `bcb1f6716a3e9ad3ddcf5717c7cb47e018ed356be360612179d1899d411ea253`.
- Compile target: `Ascend910_9382`, CANN `9.1.0-beta.3`.
- Static contract: three `int64 [64,256]` inputs and one
  `float32 [64,256,13]` output.
- Real pyACL quality on 200 documents: F1 0.9866, P 0.9823, R 0.9908.
- Serial versus cross-document adapter batching: 200/200 exact document
  matches, 2364 entities on both paths, 22.096 versus 372.799 documents/s,
  a 16.8719x speedup.

The manifest binds the OM, `tokenizer.json`, `config.json`, SoC, CANN, and the
loaded model's actual IO signature. The strict preflight verifies all artifact
hashes before any later runtime load.

## Evidence

Credential-free receipts are stored under `dist/remote_npu_v3/evidence/`:

- `cmeee_roberta_gp_freeze_s42_cann91_20260723.json`
- `cmeie_multisig_fitonly_s42_20260723.json`
- `cmeie_multisig_fitonly_s43_20260723.json`
- `medical_ner_bert_s42_eval.json`
- `eval_om_fp16_200.json`
- `adapter_serial_vs_batch_200.json`
- `medical_ner_bert_s42_bs64_seq256_fp16.manifest.json`
- `preflight_strict_with_om.json`

Models remain on the controlled remote host in versioned candidate
directories. They were not copied into the repository or submission archive.

## Claim boundary

This is a rebuilt FP16 acceptance candidate. It does not reassert the old INT8
power, energy, or extreme-throughput results, and no synchronized joule
measurement was collected in this run. The adapter test proves a real
cross-document `execute_batch` surface; it does not change the earlier finding
that the stock DataMate Mapper dispatches one sample at a time.
