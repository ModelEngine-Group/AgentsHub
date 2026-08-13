"""Seal, verify and consume an internally audited prospective evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=True)


def _write_new_atomic(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication gives create-if-absent semantics on one volume;
        # unlike check-then-write, two creators cannot both claim success.
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise RuntimeError(f"seal already exists: {output}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def seal(protocol: Path, artifacts: list[Path], output: Path) -> dict:
    protocol = _resolved(protocol)
    resolved_artifacts = [_resolved(path) for path in artifacts]
    value = {
        "schema_version": 1,
        "threat_model": "internally_audited_create_once_not_external_WORM",
        "protocol": {"path": str(protocol), "sha256": digest(protocol)},
        "artifacts": [
            {"path": str(path), "sha256": digest(path)} for path in resolved_artifacts
        ],
        "consumed": False,
    }
    _write_new_atomic(output.resolve(), value)
    return value


def _load_and_verify(seal_path: Path) -> tuple[dict, str]:
    raw = seal_path.read_bytes()
    seal_digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw.decode("utf-8"))
    entries = [value["protocol"], *value["artifacts"]]
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise RuntimeError(f"sealed artifact mismatch: {path}")
    return value, seal_digest


def verify(seal_path: Path) -> dict:
    seal_path = seal_path.resolve(strict=True)
    value, current_digest = _load_and_verify(seal_path)
    marker = seal_path.with_name(f".{current_digest}.consumed")
    marker_exists = marker.exists()
    if value.get("consumed"):
        receipt_info = value.get("consumption_receipt") or {}
        expected = receipt_info.get("preconsume_seal_sha256")
        if not isinstance(expected, str):
            raise RuntimeError("consumed seal is missing pre-consumption hash")
        marker = seal_path.with_name(f".{expected}.consumed")
        if not marker.is_file() or marker.read_text(encoding="utf-8") != expected:
            raise RuntimeError("consumed seal marker is missing or inconsistent")
        receipt_path = Path(str(receipt_info.get("path", "")))
        if not receipt_path.is_file():
            raise RuntimeError("consumption receipt is missing")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt != {"schema_version": 1, "seal_sha256": expected, "status": "consumed"}:
            raise RuntimeError("consumption receipt is invalid")
    elif marker_exists:
        raise RuntimeError("unconsumed seal has a forged consumption marker")
    return value


def consume(seal_path: Path, receipt: Path) -> dict:
    seal_path = seal_path.resolve(strict=True)
    if seal_path.stat().st_nlink != 1:
        raise RuntimeError("seal has filesystem aliases; one-shot consumption requires one hard link")
    receipt = receipt.resolve()
    value, original_seal_sha256 = _load_and_verify(seal_path)
    if value.get("consumed"):
        raise RuntimeError("seal is already marked consumed")

    marker = seal_path.with_name(f".{original_seal_sha256}.consumed")
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(original_seal_sha256)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError("seal is already marked consumed") from exc

    # Recheck the same physical path/version after claiming the digest marker.
    try:
        if digest(seal_path) != original_seal_sha256:
            raise RuntimeError("seal changed concurrently during consumption")
        receipt_value = {
            "schema_version": 1,
            "seal_sha256": original_seal_sha256,
            "status": "consumed",
        }
        _write_new_atomic(receipt, receipt_value)
        consumed_value = dict(value)
        consumed_value["consumed"] = True
        consumed_value["consumption_receipt"] = {
            "path": str(receipt),
            "preconsume_seal_sha256": original_seal_sha256,
        }
        temporary = seal_path.with_name(f".{seal_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(consumed_value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(seal_path)
        return receipt_value
    except BaseException:
        # Keep the marker on ambiguous failure: fail closed rather than silently
        # permitting a second one-shot evaluation.
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    seal_cmd = sub.add_parser("seal")
    seal_cmd.add_argument("--protocol", type=Path, required=True)
    seal_cmd.add_argument("--artifacts", type=Path, nargs="+", required=True)
    seal_cmd.add_argument("--output", type=Path, required=True)
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("seal", type=Path)
    consume_cmd = sub.add_parser("consume")
    consume_cmd.add_argument("seal", type=Path)
    consume_cmd.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal(args.protocol, args.artifacts, args.output)
    elif args.command == "verify":
        result = verify(args.seal)
    else:
        result = consume(args.seal, args.receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
