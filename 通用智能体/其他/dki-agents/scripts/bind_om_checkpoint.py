"""Bind an ATC-built OM to the exact exported CMeEE checkpoint.

This is a fail-closed release step, not a benchmark. It verifies the complete
checkpoint -> ONNX receipt -> OM chain and creates a schema-v3 manifest once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.bench.npu.om_manifest import OmManifest, TensorSpec, sha256_tree
from dki.evidence import sha256_file


def _specs(values: list[dict]) -> tuple[TensorSpec, ...]:
    return tuple(TensorSpec.from_dict(value) for value in values)


def _layouts(values: tuple[TensorSpec, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((value.dtype, value.shape) for value in values)


def _write_create_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite bound manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a schema-v3 CMeEE checkpoint-bound OM manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--atc-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = {
        "base manifest": args.manifest,
        "export receipt": args.export_receipt,
        "ONNX": args.onnx,
        "OM": args.om,
        "tokenizer": args.tokenizer,
        "labels": args.labels,
        "ATC log": args.atc_log,
    }
    missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
    if missing:
        parser.error(f"release artifact missing: {missing}")
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory missing: {args.checkpoint}")
    if args.output.exists():
        parser.error("output is a create-once release artifact")
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    if checkpoint == output or checkpoint in output.parents:
        parser.error("bound manifest output must be outside the checkpoint directory")
    if args.atc_log.stat().st_size == 0:
        parser.error("ATC log is empty")

    try:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        receipt = json.loads(args.export_receipt.read_text(encoding="utf-8"))
        atc_log = json.loads(args.atc_log.read_text(encoding="utf-8"))
        base = OmManifest.load(args.manifest)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid JSON release artifact: {exc}")
    if base.schema_version != 1:
        parser.error("base OM manifest must be an unbound schema-v1 manifest")
    if base.checkpoint_sha256 or base.export_receipt_sha256:
        parser.error("base OM manifest is already checkpoint-bound")
    if receipt.get("schema_version") != 1:
        parser.error("unsupported ONNX export receipt schema")
    if receipt.get("kind") != "cmeee_globalpointer_onnx_export":
        parser.error("receipt is not a CMeEE GlobalPointer ONNX export")
    if receipt.get("model_family") != "cmeee_globalpointer":
        parser.error("receipt model family is not CMeEE GlobalPointer")
    if receipt.get("checkpoint_kind") != "directory_tree_v1":
        parser.error("receipt does not bind a directory_tree_v1 checkpoint")
    if receipt.get("model_id") != base.model_id:
        parser.error("base manifest model_id does not match the export receipt")

    checks = {
        "checkpoint_sha256": sha256_tree(args.checkpoint),
        "onnx_sha256": sha256_file(args.onnx),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "labels_sha256": sha256_file(args.labels),
    }
    for field, actual in checks.items():
        if str(receipt.get(field, "")) != actual:
            parser.error(f"{field} does not match the ONNX export receipt")
    if base.om_sha256 != sha256_file(args.om):
        parser.error("OM hash does not match the base manifest")
    if base.tokenizer_sha256 != checks["tokenizer_sha256"]:
        parser.error("tokenizer hash does not match the base manifest")
    if base.labels_sha256 != checks["labels_sha256"]:
        parser.error("labels hash does not match the base manifest")
    try:
        receipt_inputs = _specs(receipt.get("inputs", []))
        receipt_outputs = _specs(receipt.get("outputs", []))
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid tensor signature in export receipt: {exc}")
    if base.inputs != receipt_inputs:
        parser.error("base manifest input signature does not match the ONNX export")
    if _layouts(base.outputs) != _layouts(receipt_outputs):
        parser.error("base manifest output layout does not match the ONNX export")
    expected_output_mapping = [
        {
            "index": index,
            "onnx_name": exported.name,
            "om_name": actual.name,
        }
        for index, (exported, actual) in enumerate(
            zip(receipt_outputs, base.outputs)
        )
    ]
    if payload.get("output_binding") != "tensor_order_dtype_shape":
        parser.error("base manifest does not declare the output binding rule")
    if payload.get("output_name_mapping") != expected_output_mapping:
        parser.error("base manifest ONNX-to-OM output name mapping is invalid")

    returncode = atc_log.get("returncode")
    if atc_log.get("schema_version") != 1:
        parser.error("unsupported ATC log schema")
    if atc_log.get("kind") != "atc_compile_log":
        parser.error("ATC log kind is invalid")
    if (
        not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or returncode != 0
        or atc_log.get("validation_error")
    ):
        parser.error("ATC log does not prove a successful, validated compilation")
    if atc_log.get("onnx_sha256") != checks["onnx_sha256"]:
        parser.error("ATC log ONNX hash does not match the bound ONNX")
    if atc_log.get("om_sha256") != base.om_sha256:
        parser.error("ATC log OM hash does not match the inspected OM")
    if atc_log.get("export_receipt_sha256") != sha256_file(args.export_receipt):
        parser.error("ATC log export receipt hash does not match")
    if atc_log.get("soc_version") != base.soc_version:
        parser.error("ATC log SoC does not match the base OM manifest")
    try:
        atc_output = Path(str(atc_log["output_path"])).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"ATC log output path is invalid: {exc}")
    if atc_output != args.om.resolve():
        parser.error("ATC log output path does not identify the bound OM")

    payload.update(
        {
            "schema_version": 3,
            "model_family": receipt["model_family"],
            "checkpoint_kind": "directory_tree_v1",
            "checkpoint_sha256": checks["checkpoint_sha256"],
            "checkpoint_path_hint": args.checkpoint.name,
            "export_receipt_sha256": sha256_file(args.export_receipt),
            "export_receipt_path_hint": args.export_receipt.name,
            "onnx_sha256": checks["onnx_sha256"],
            "onnx_path_hint": args.onnx.name,
            "atc_log_sha256": sha256_file(args.atc_log),
            "atc_log_path_hint": args.atc_log.name,
        }
    )
    _write_create_once(args.output, payload)

    bound = OmManifest.load(args.output)
    bound.verify_artifacts(
        args.om,
        args.tokenizer,
        args.labels,
        args.checkpoint,
        args.export_receipt,
        args.atc_log,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "manifest_sha256": sha256_file(args.output),
                "checkpoint_sha256": bound.checkpoint_sha256,
                "om_sha256": bound.om_sha256,
                "export_receipt_sha256": bound.export_receipt_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
