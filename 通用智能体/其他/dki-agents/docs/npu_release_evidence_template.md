# NPU Release Evidence Template

Do not promote a new NPU result until every item below is filled with measured
evidence from the same input snapshot and target environment.

| Gate | Required evidence |
|---|---|
| Artifact binding | checkpoint tree, ONNX receipt, OM, tokenizer, labels and ATC-log SHA-256 values; schema-v3 manifest verifies the model family and real ACL IO contract |
| Parity | identical input corpus, serial/batch output comparison, exact-document count and quality delta |
| CPU/NPU comparison | warmed median of three runs, throughput, p50/p95 latency and batch configuration |
| Energy | synchronized power/energy capture, joules/document, sampling interval and device state |
| DataMate | evidence that production path invokes formal cross-document `map_batches`/equivalent, not a local probe |
| Promotion | explicit quality non-regression decision and links to immutable receipts |

Until this evidence exists, use the existing validated result as historical
context only and keep `formal_microbatch=false` / Release B not promoted.

The historical 372.799 docs/s OM is `medical_ner_bert`. It must not be
attributed to or metadata-bound to the current CMeEE GlobalPointer checkpoint.

## Schema-v3 artifact sequence

Run these only on an explicitly authorized target workflow:

1. `export_cmeee_globalpointer_onnx.py`: load the frozen checkpoint, export a
   static ONNX, and seal the checkpoint tree plus ONNX receipt.
2. `compile_cmeee_om.py`: verify the receipt, invoke ATC without a shell, and
   preserve the full create-once ATC log.
3. `create_om_base_manifest.py`: load the OM once, perform no inference, and
   record the real ACL input/output tensor contract.
4. `bind_om_checkpoint.py`: require exact model ID, model family, IO and all
   hashes, then create a non-overwritable schema-v3 manifest.
5. `remote_npu_preflight.py`: verify the complete chain without loading the OM.

Source preparation is not NPU performance evidence. Promotion still requires
same-input parity, three warmed runs, p50/p95 latency, quality delta and
synchronized joules/document.
