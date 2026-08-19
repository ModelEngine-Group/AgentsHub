"""Immutable OM artifact contract and runtime compatibility checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from dki.evidence import sha256_file


class OmCompatibilityError(RuntimeError):
    pass


def sha256_tree(path: Path) -> str:
    """Hash a directory as a sorted relative-path/file-hash tree."""
    if not path.is_dir():
        raise OmCompatibilityError(f"checkpoint directory missing: {path}")
    if path.is_symlink():
        raise OmCompatibilityError(f"checkpoint directory must not be a symlink: {path}")
    entries = sorted(path.rglob("*"))
    symlinks = [item for item in entries if item.is_symlink()]
    if symlinks:
        relative = [item.relative_to(path).as_posix() for item in symlinks[:10]]
        raise OmCompatibilityError(
            f"checkpoint tree contains symlink(s), refusing ambiguous hash: {relative}"
        )
    files = [item for item in entries if item.is_file()]
    if not files:
        raise OmCompatibilityError(f"checkpoint directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TensorSpec":
        return cls(str(value["name"]), str(value["dtype"]), tuple(int(v) for v in value["shape"]))


@dataclass(frozen=True)
class OmManifest:
    schema_version: int
    model_id: str
    om_sha256: str
    tokenizer_sha256: str
    labels_sha256: str
    soc_version: str
    cann_version: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    checkpoint_sha256: str = ""
    checkpoint_kind: str = "file"
    export_receipt_sha256: str = ""
    onnx_sha256: str = ""
    model_family: str = ""
    atc_log_sha256: str = ""

    @classmethod
    def load(cls, path: Path) -> "OmManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            schema_version=int(value["schema_version"]),
            model_id=str(value["model_id"]),
            om_sha256=str(value["om_sha256"]),
            tokenizer_sha256=str(value["tokenizer_sha256"]),
            labels_sha256=str(value["labels_sha256"]),
            soc_version=str(value["soc_version"]),
            cann_version=str(value["cann_version"]),
            inputs=tuple(TensorSpec.from_dict(item) for item in value["inputs"]),
            outputs=tuple(TensorSpec.from_dict(item) for item in value["outputs"]),
            checkpoint_sha256=str(value.get("checkpoint_sha256", "")),
            checkpoint_kind=str(value.get("checkpoint_kind", "file")),
            export_receipt_sha256=str(value.get("export_receipt_sha256", "")),
            onnx_sha256=str(value.get("onnx_sha256", "")),
            model_family=str(value.get("model_family", "")),
            atc_log_sha256=str(value.get("atc_log_sha256", "")),
        )

    def verify_artifacts(
        self, om_path: Path, tokenizer_path: Path, labels_path: Path,
        checkpoint_path: Path | None = None,
        export_receipt_path: Path | None = None,
        atc_log_path: Path | None = None,
    ) -> None:
        artifacts = [
            ("OM", om_path, self.om_sha256),
            ("tokenizer", tokenizer_path, self.tokenizer_sha256),
            ("labels", labels_path, self.labels_sha256),
        ]
        if self.checkpoint_sha256:
            if checkpoint_path is None:
                raise OmCompatibilityError("manifest binds a checkpoint but checkpointPath is missing")
            if self.checkpoint_kind == "directory_tree_v1":
                actual = sha256_tree(checkpoint_path)
                if actual != self.checkpoint_sha256:
                    raise OmCompatibilityError(
                        f"checkpoint hash mismatch: {actual} != {self.checkpoint_sha256}"
                    )
            elif self.checkpoint_kind == "file":
                artifacts.append(("checkpoint", checkpoint_path, self.checkpoint_sha256))
            else:
                raise OmCompatibilityError(
                    f"unsupported checkpoint hash kind: {self.checkpoint_kind}"
                )
        if self.export_receipt_sha256:
            if export_receipt_path is None:
                raise OmCompatibilityError(
                    "manifest binds an export receipt but exportReceiptPath is missing"
                )
            artifacts.append(
                ("export receipt", export_receipt_path, self.export_receipt_sha256)
            )
        if self.atc_log_sha256:
            if atc_log_path is None:
                raise OmCompatibilityError(
                    "manifest binds an ATC log but atcLogPath is missing"
                )
            artifacts.append(("ATC log", atc_log_path, self.atc_log_sha256))
        for name, path, expected in artifacts:
            if not path.is_file():
                raise OmCompatibilityError(f"{name} artifact missing: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise OmCompatibilityError(f"{name} hash mismatch: {actual} != {expected}")
        if self.schema_version >= 3:
            self._verify_export_receipt(export_receipt_path)
            self._verify_atc_log(atc_log_path)

    def _verify_export_receipt(self, path: Path | None) -> None:
        required = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_family": self.model_family,
            "model_id": self.model_id,
            "onnx_sha256": self.onnx_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "labels_sha256": self.labels_sha256,
        }
        if not all(required.values()) or not self.atc_log_sha256:
            raise OmCompatibilityError("schema-v3 manifest evidence contract is incomplete")
        if path is None:
            raise OmCompatibilityError("schema-v3 manifest requires an export receipt")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OmCompatibilityError("export receipt is not valid JSON") from exc
        if receipt.get("schema_version") != 1:
            raise OmCompatibilityError("unsupported export receipt schema")
        if receipt.get("kind") != "cmeee_globalpointer_onnx_export":
            raise OmCompatibilityError("export receipt kind does not identify CMeEE GlobalPointer")
        if receipt.get("checkpoint_kind") != "directory_tree_v1":
            raise OmCompatibilityError(
                "export receipt does not bind a directory_tree_v1 checkpoint"
            )
        for field, expected in required.items():
            actual = str(receipt.get(field, ""))
            if actual != expected:
                raise OmCompatibilityError(
                    f"export receipt {field} mismatch: {actual} != {expected}"
                )

    def _verify_atc_log(self, path: Path | None) -> None:
        if path is None:
            raise OmCompatibilityError("schema-v3 manifest requires an ATC log")
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OmCompatibilityError("ATC log is not valid JSON") from exc
        returncode = log.get("returncode")
        if log.get("schema_version") != 1 or log.get("kind") != "atc_compile_log":
            raise OmCompatibilityError("ATC log schema or kind is invalid")
        if (
            not isinstance(returncode, int)
            or isinstance(returncode, bool)
            or returncode != 0
            or log.get("validation_error")
        ):
            raise OmCompatibilityError(
                "ATC log does not prove a successful, validated compilation"
            )
        required = {
            "onnx_sha256": self.onnx_sha256,
            "om_sha256": self.om_sha256,
            "export_receipt_sha256": self.export_receipt_sha256,
            "soc_version": self.soc_version,
        }
        for field, expected in required.items():
            actual = str(log.get(field, ""))
            if actual != expected:
                raise OmCompatibilityError(
                    f"ATC log {field} mismatch: {actual} != {expected}"
                )

    def verify_file(self, om_path: Path) -> None:
        """Backward-compatible OM-only check for historical probes.

        Production promotion must call :meth:`verify_artifacts` so tokenizer
        and label mappings are cryptographically bound too.
        """
        actual = sha256_file(om_path)
        if actual != self.om_sha256:
            raise OmCompatibilityError(f"OM hash mismatch: {actual} != {self.om_sha256}")

    def verify_io(
        self,
        *,
        inputs: Sequence[TensorSpec],
        outputs: Sequence[TensorSpec],
    ) -> None:
        """Verify only the tensor IO signature.

        Split from :meth:`verify_runtime` because the loaded model's real IO
        specs can only be read *after* the ACL model is loaded, whereas artifact
        hashes and SoC/CANN must be checked *before* the untrusted OM is touched.
        """
        if tuple(inputs) != self.inputs:
            raise OmCompatibilityError(f"input signature mismatch: {tuple(inputs)!r} != {self.inputs!r}")
        if tuple(outputs) != self.outputs:
            raise OmCompatibilityError(f"output signature mismatch: {tuple(outputs)!r} != {self.outputs!r}")

    def verify_runtime(
        self,
        *,
        soc_version: str,
        cann_version: str,
        inputs: Sequence[TensorSpec],
        outputs: Sequence[TensorSpec],
    ) -> None:
        if soc_version != self.soc_version:
            raise OmCompatibilityError(f"SoC mismatch: {soc_version} != {self.soc_version}")
        if cann_version != self.cann_version:
            raise OmCompatibilityError(f"CANN mismatch: {cann_version} != {self.cann_version}")
        self.verify_io(inputs=inputs, outputs=outputs)


def enforce_preload_manifest(config: Mapping[str, object]) -> "OmManifest | None":
    """Hard artifact + SoC/CANN gate consumed *before* any ACL model load.

    Opt-in by presence of ``manifestPath`` in the runtime/adapter config so
    fake-adapter unit tests and non-OM adapters remain unaffected. When present,
    every check is mandatory and any failure raises ``OmCompatibilityError``
    (fail closed) before the untrusted OM is loaded.

    Recognized keys: ``manifestPath`` (required to activate), ``omPath``,
    ``tokenizerPath``, ``labelsPath`` (all required when active), plus
    ``checkpointPath``, ``exportReceiptPath`` and ``atcLogPath`` when those
    artifacts are bound. ``socVersion`` and ``cannVersion`` are verified when
    provided and are also read from ``DKI_OM_SOC`` / ``DKI_OM_CANN``.
    """
    import os as _os

    manifest_path = config.get("manifestPath")
    if not manifest_path:
        return None
    manifest = OmManifest.load(Path(str(manifest_path)))
    missing = [key for key in ("omPath", "tokenizerPath", "labelsPath") if not config.get(key)]
    if manifest.checkpoint_sha256 and not config.get("checkpointPath"):
        missing.append("checkpointPath")
    if manifest.export_receipt_sha256 and not config.get("exportReceiptPath"):
        missing.append("exportReceiptPath")
    if manifest.atc_log_sha256 and not config.get("atcLogPath"):
        missing.append("atcLogPath")
    if missing:
        raise OmCompatibilityError(
            f"manifestPath set but required artifact path(s) missing: {missing}"
        )
    manifest.verify_artifacts(
        Path(str(config["omPath"])),
        Path(str(config["tokenizerPath"])),
        Path(str(config["labelsPath"])),
        Path(str(config["checkpointPath"])) if config.get("checkpointPath") else None,
        (
            Path(str(config["exportReceiptPath"]))
            if config.get("exportReceiptPath")
            else None
        ),
        Path(str(config["atcLogPath"])) if config.get("atcLogPath") else None,
    )
    soc = str(config.get("socVersion") or _os.environ.get("DKI_OM_SOC", "") or "")
    cann = str(config.get("cannVersion") or _os.environ.get("DKI_OM_CANN", "") or "")
    if soc and soc != manifest.soc_version:
        raise OmCompatibilityError(f"SoC mismatch: {soc} != {manifest.soc_version}")
    if cann and cann != manifest.cann_version:
        raise OmCompatibilityError(f"CANN mismatch: {cann} != {manifest.cann_version}")
    return manifest
