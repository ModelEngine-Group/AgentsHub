"""Inspect a target-host OM and create an unbound schema-v1 IO manifest.

This command loads the OM only to read its real ACL tensor contract. The
result remains unpromoted until ``bind_om_checkpoint.py`` creates schema v3.
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

from dki.bench.npu.om_manifest import TensorSpec
from dki.bench.npu.static_om_runner import StaticOmRunner
from dki.evidence import sha256_file


def _spec_dict(spec: TensorSpec) -> dict:
    return {"name": spec.name, "dtype": spec.dtype, "shape": list(spec.shape)}


def _specs(values: list[dict]) -> tuple[TensorSpec, ...]:
    return tuple(TensorSpec.from_dict(value) for value in values)


def _layouts(values: tuple[TensorSpec, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((value.dtype, value.shape) for value in values)


def _write_create_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OM manifest: {path}")
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
        description="Load an OM once and record its exact ACL IO contract"
    )
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--soc-version", required=True)
    parser.add_argument("--cann-version", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for label, path in (
        ("OM", args.om),
        ("export receipt", args.export_receipt),
        ("tokenizer", args.tokenizer),
        ("labels", args.labels),
    ):
        if not path.is_file():
            parser.error(f"{label} artifact missing: {path}")
    if args.output.exists():
        parser.error("output is a create-once release artifact")

    try:
        receipt = json.loads(args.export_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid export receipt: {exc}")
    if receipt.get("kind") != "cmeee_globalpointer_onnx_export":
        parser.error("export receipt is not for CMeEE GlobalPointer")
    if receipt.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        parser.error("tokenizer hash does not match export receipt")
    if receipt.get("labels_sha256") != sha256_file(args.labels):
        parser.error("labels hash does not match export receipt")

    runner = StaticOmRunner(args.om, device_id=args.device_id)
    try:
        inputs = runner.input_specs
        outputs = runner.output_specs
    finally:
        runner.close()
    receipt_inputs = _specs(receipt.get("inputs", []))
    receipt_outputs = _specs(receipt.get("outputs", []))
    if inputs != receipt_inputs:
        parser.error(
            "loaded OM inputs do not match exported ONNX inputs: "
            f"actual={[_spec_dict(spec) for spec in inputs]!r}, "
            f"expected={[_spec_dict(spec) for spec in receipt_inputs]!r}"
        )
    if _layouts(outputs) != _layouts(receipt_outputs):
        parser.error(
            "loaded OM output layouts do not match exported ONNX outputs: "
            f"actual={[_spec_dict(spec) for spec in outputs]!r}, "
            f"expected={[_spec_dict(spec) for spec in receipt_outputs]!r}"
        )
    output_name_mapping = [
        {
            "index": index,
            "onnx_name": exported.name,
            "om_name": actual.name,
        }
        for index, (exported, actual) in enumerate(zip(receipt_outputs, outputs))
    ]

    payload = {
        "schema_version": 1,
        "model_id": receipt["model_id"],
        "om_sha256": sha256_file(args.om),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "labels_sha256": sha256_file(args.labels),
        "soc_version": args.soc_version,
        "cann_version": args.cann_version,
        "inputs": [_spec_dict(spec) for spec in inputs],
        "outputs": [_spec_dict(spec) for spec in outputs],
        "output_name_mapping": output_name_mapping,
        "output_binding": "tensor_order_dtype_shape",
        "inspection": "acl_model_loaded_for_io_only_no_inference",
    }
    _write_create_once(args.output, payload)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "manifest_sha256": sha256_file(args.output),
                "inputs": payload["inputs"],
                "outputs": payload["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
