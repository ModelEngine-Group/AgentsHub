"""Export a frozen CMeEE GlobalPointer checkpoint and seal its ONNX receipt.

The receipt binds the complete checkpoint tree, tokenizer, labels, model
family, tensor contract, and exported ONNX. It is consumed by
``bind_om_checkpoint.py`` after target-host ATC conversion.
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

from dki.bench.npu.om_manifest import sha256_tree
from dki.evidence import sha256_file
from scripts.train_ner_cmeee import TYPES

MODEL_FAMILY = "cmeee_globalpointer"


def _load_state(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_json_create_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite export receipt: {path}")
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


def _required_checkpoint_files(model_dir: Path, config: dict) -> list[Path]:
    required = [
        model_dir / "config.json",
        model_dir / "globalpointer.pt",
        model_dir / "globalpointer_config.json",
        model_dir / "types.json",
        model_dir / "tokenizer.json",
    ]
    if config.get("use_bilstm"):
        required.append(model_dir / "bilstm.pt")
    softlex_enabled = bool(config.get("softlexicon")) or int(
        config.get("softlex_dim", 0) or 0
    ) > 0
    if softlex_enabled:
        required.append(model_dir / "softlexicon_proj.pt")
        if str(config.get("softlex_mode") or "gated").lower() == "gated":
            required.append(model_dir / "softlexicon_gate.pt")
    return required


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a static CMeEE GlobalPointer ONNX with an immutable receipt"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default="medical_ner_cmeee_globalpointer",
        help="must exactly match the target OM manifest model_id",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.opset < 17:
        parser.error("batch-size must be positive and opset must be at least 17")

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if not checkpoint.is_dir():
        parser.error(f"checkpoint directory does not exist: {checkpoint}")
    if output.exists() or receipt.exists():
        parser.error("output and receipt are create-once release artifacts")
    if output == receipt:
        parser.error("ONNX output and export receipt must be different files")
    for label, path in (("ONNX output", output), ("export receipt", receipt)):
        if checkpoint == path or checkpoint in path.parents:
            parser.error(f"{label} must be outside the checkpoint directory")
    partial = output.with_name(f".{output.name}.partial")
    if partial.exists():
        parser.error(f"stale partial ONNX exists; inspect and remove it first: {partial}")

    try:
        config = json.loads(
            (checkpoint / "globalpointer_config.json").read_text(encoding="utf-8")
        )
        types = json.loads((checkpoint / "types.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid checkpoint metadata: {exc}")
    if types != TYPES:
        parser.error("types.json does not match the CMeEE-V2 label contract")
    required = _required_checkpoint_files(checkpoint, config)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"checkpoint is incomplete: {missing}")
    checkpoint_digest = sha256_tree(checkpoint)

    import torch
    from transformers import AutoModel

    from dki.nlp.globalpointer import EncoderGlobalPointer

    max_len = int(config["max_len"])
    inner_dim = int(config["inner_dim"])
    softlex_dim = int(config.get("softlex_dim", 0) or 0)
    if max_len <= 0 or inner_dim <= 0:
        parser.error("checkpoint max_len and inner_dim must be positive")

    encoder = AutoModel.from_pretrained(checkpoint, local_files_only=True)
    model = EncoderGlobalPointer(
        encoder,
        encoder.config.hidden_size,
        len(types),
        inner_dim,
        pointer=str(config.get("pointer", "classic")),
        softlex_dim=softlex_dim,
        softlex_mode=str(config.get("softlex_mode") or "gated"),
        use_bilstm=bool(config.get("use_bilstm", False)),
    )
    model.gp.load_state_dict(_load_state(torch, checkpoint / "globalpointer.pt"))
    if model.bilstm is not None:
        model.bilstm.load_state_dict(_load_state(torch, checkpoint / "bilstm.pt"))
    if model.softlex_proj is not None:
        model.softlex_proj.load_state_dict(
            _load_state(torch, checkpoint / "softlexicon_proj.pt")
        )
        if model.softlex_gate is not None:
            gate = _load_state(torch, checkpoint / "softlexicon_gate.pt")
            model.softlex_gate.data.copy_(gate["gate"])
    model.eval()

    shape = (args.batch_size, max_len)
    inputs = [
        torch.zeros(shape, dtype=torch.long),
        torch.ones(shape, dtype=torch.long),
        torch.zeros(shape, dtype=torch.long),
    ]
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    input_specs = [
        {"name": name, "dtype": "int64", "shape": list(shape)}
        for name in input_names
    ]
    if model.softlex_proj is not None:
        inputs.append(torch.zeros((*shape, softlex_dim), dtype=torch.float32))
        input_names.append("softlex")
        input_specs.append(
            {
                "name": "softlex",
                "dtype": "float32",
                "shape": [args.batch_size, max_len, softlex_dim],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(inputs),
                partial,
                input_names=input_names,
                output_names=["logits"],
                opset_version=args.opset,
                do_constant_folding=True,
            )
        import onnx

        onnx.checker.check_model(onnx.load(str(partial)))
        if sha256_tree(checkpoint) != checkpoint_digest:
            raise RuntimeError("checkpoint changed while ONNX export was in progress")
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)

    tokenizer = checkpoint / "tokenizer.json"
    labels = checkpoint / "types.json"
    payload = {
        "schema_version": 1,
        "kind": "cmeee_globalpointer_onnx_export",
        "model_family": MODEL_FAMILY,
        "model_id": args.model_id,
        "checkpoint_path_hint": checkpoint.name,
        "checkpoint_kind": "directory_tree_v1",
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_file_count": sum(1 for item in checkpoint.rglob("*") if item.is_file()),
        "tokenizer_path_hint": tokenizer.name,
        "tokenizer_sha256": sha256_file(tokenizer),
        "labels_path_hint": labels.name,
        "labels_sha256": sha256_file(labels),
        "onnx_path_hint": output.name,
        "onnx_sha256": sha256_file(output),
        "opset": args.opset,
        "inputs": input_specs,
        "outputs": [
            {
                "name": "logits",
                "dtype": "float32",
                "shape": [args.batch_size, len(types), max_len, max_len],
            }
        ],
        "checkpoint_config": {
            key: config.get(key)
            for key in (
                "pointer",
                "max_len",
                "inner_dim",
                "selected_threshold",
                "softlexicon",
                "softlex_dim",
                "use_bilstm",
            )
        },
    }
    _write_json_create_once(receipt, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "onnx_sha256": payload["onnx_sha256"],
                "checkpoint_sha256": checkpoint_digest,
                "receipt": str(receipt),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
