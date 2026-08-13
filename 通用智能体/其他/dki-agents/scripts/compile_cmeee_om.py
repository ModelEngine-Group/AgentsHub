"""Compile a receipt-bound CMeEE ONNX with ATC and preserve the full log."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence import sha256_file


def _write_json_create_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ATC log: {path}")
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


def _input_shape(specs: list[dict]) -> str:
    values = []
    for spec in specs:
        name = str(spec.get("name", ""))
        shape = spec.get("shape")
        if not name or not isinstance(shape, list) or not shape:
            raise ValueError("export receipt contains an invalid input tensor")
        dims = [int(value) for value in shape]
        if any(value <= 0 for value in dims):
            raise ValueError("ATC release compilation requires static positive dimensions")
        values.append(f"{name}:{','.join(str(value) for value in dims)}")
    if not values:
        raise ValueError("export receipt contains no input tensors")
    return ";".join(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile a CMeEE GlobalPointer ONNX")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="target .om path")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--soc-version", required=True)
    parser.add_argument("--precision-mode", default="allow_fp32_to_fp16")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".om":
        parser.error("output must use the .om suffix")
    if args.output.exists() or args.log.exists():
        parser.error("OM and ATC log are create-once release artifacts")
    if args.output.resolve() == args.log.resolve():
        parser.error("OM output and ATC log must be different files")
    if not args.onnx.is_file() or not args.export_receipt.is_file():
        parser.error("ONNX and export receipt must exist")
    executable = shutil.which("atc")
    if executable is None:
        parser.error("atc is not available on PATH")

    try:
        receipt = json.loads(args.export_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid export receipt: {exc}")
    if receipt.get("schema_version") != 1:
        parser.error("unsupported export receipt schema")
    if receipt.get("kind") != "cmeee_globalpointer_onnx_export":
        parser.error("receipt is not a CMeEE GlobalPointer ONNX export")
    if receipt.get("model_family") != "cmeee_globalpointer":
        parser.error("receipt model family is not CMeEE GlobalPointer")
    if receipt.get("checkpoint_kind") != "directory_tree_v1":
        parser.error("receipt does not bind a directory_tree_v1 checkpoint")
    if receipt.get("onnx_sha256") != sha256_file(args.onnx):
        parser.error("ONNX hash does not match export receipt")
    try:
        input_shape = _input_shape(receipt.get("inputs", []))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output.with_suffix("")
    command = [
        executable,
        "--framework=5",
        f"--model={args.onnx.resolve()}",
        f"--output={output_prefix.resolve()}",
        "--input_format=ND",
        f"--input_shape={input_shape}",
        f"--soc_version={args.soc_version}",
        f"--precision_mode={args.precision_mode}",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log = {
        "schema_version": 1,
        "kind": "atc_compile_log",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "onnx_sha256": sha256_file(args.onnx),
        "export_receipt_sha256": sha256_file(args.export_receipt),
        "output_path": str(args.output.resolve()),
        "soc_version": args.soc_version,
        "precision_mode": args.precision_mode,
    }
    failure = None
    if completed.returncode != 0:
        failure = f"ATC failed with return code {completed.returncode}"
    elif not args.output.is_file():
        failure = f"ATC returned success but OM is missing: {args.output}"
    else:
        log["om_sha256"] = sha256_file(args.output)
    if args.output.is_file() and "om_sha256" not in log:
        log["unpromoted_om_sha256"] = sha256_file(args.output)
    if failure:
        log["validation_error"] = failure
    _write_json_create_once(args.log, log)
    if failure:
        raise SystemExit(
            completed.returncode if completed.returncode != 0 else failure
        )
    print(
        json.dumps(
            {
                "om": str(args.output),
                "om_sha256": log["om_sha256"],
                "atc_log": str(args.log),
                "atc_log_sha256": sha256_file(args.log),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
