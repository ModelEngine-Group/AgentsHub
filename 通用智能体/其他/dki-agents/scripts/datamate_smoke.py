"""Strict, receipt-producing smoke test for the real DataMate runtime.

The smoke test deliberately goes through ``dki.paths`` and ``dki.dmcompat``.
It fails closed if the real runtime cannot be imported, loads the DataMate
built-ins, deploys the six standalone operator archives under the platform
``ops.user.<package>`` layout, invokes the medical chain, and checks shadow
failure isolation.

The JSON receipt is published exactly once.  It binds the canonical DataMate
runtime root and a compact deterministic tree digest produced by
``scan_bound_regular_tree``.  A pre-existing receipt is never overwritten.

Example:

    DKI_DATAMATE_RUNTIME=/opt/datamate/runtime \
      python scripts/datamate_smoke.py \
      --dist dist/operators \
      --output outputs/remote_runs/<RID>/datamate_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGES = [
    "medical_text_cleaner",
    "medical_section_splitter",
    "medical_entity_extractor",
    "medical_relation_extractor",
    "triple_builder_validator",
    "medical_entity_extractor_om_shadow",
]
CHAIN = [
    ("MedicalTextCleaner", {"maskPii": True}),
    ("MedicalSectionSplitter", {}),
    ("MedicalEntityExtractor", {"backend": "dict"}),
    ("MedicalRelationExtractor", {"useLlm": False}),
    ("TripleBuilderValidator", {"minConfidence": 0.55}),
]

_SAMPLE_TEXT = (
    "主诉：反复头痛、头晕3年。\n"
    "现病史：患者血压最高170/100mmHg，自觉头痛，并伴有头晕、心悸。\n"
    "既往史：否认糖尿病史。\n"
    "诊断：高血压病。\n"
    "治疗方案：给予硝苯地平缓释片口服，低盐饮食。\n"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _tree_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": str(report["algorithm"]),
        "sha256": str(report["sha256"]),
        "n_files": int(report["n_files"]),
        "bytes": int(report["total_bytes"]),
    }


def _package_version(distribution: str, module_name: str) -> str:
    try:
        value = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        value = getattr(module, "__version__", None)
    if value is None or not str(value).strip():
        raise RuntimeError(f"cannot determine the installed {distribution} version")
    return str(value)


def _publish_json_create_once(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish UTF-8 JSON without replacing an existing pathname."""

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o444)
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _make_sample(text: str, file_id: str) -> dict[str, Any]:
    return {
        "text": text,
        "data": b"",
        "fileName": f"{file_id}.txt",
        "fileType": "txt",
        "fileId": file_id,
        "filePath": "",
        "fileSize": str(len(text)),
        "export_path": "",
        "ext_params": {},
        "target_type": None,
        "instance_id": file_id,
    }


class _FailingRuntime:
    """Synthetic shadow runtime used to verify failure isolation."""

    def execute_batch(
        self,
        texts: list[str],
        timeout_s: float = 1.0,
    ) -> list[list[dict[str, Any]]]:
        del timeout_s
        first_text = texts[0] if texts else ""
        raise RuntimeError(
            "synthetic shadow failure with raw text: " + first_text[:10]
        )

    def close(self) -> None:
        return None


class _StubRuntime:
    """Deterministic shadow runtime used to verify the normal sidecar path."""

    def execute_batch(
        self,
        texts: list[str],
        timeout_s: float = 1.0,
    ) -> list[list[dict[str, Any]]]:
        del timeout_s
        return [
            [
                {
                    "name": "高血压",
                    "type": "Disease",
                    "span": [0, 3],
                    "source": "om",
                    "confidence": 0.8,
                }
            ]
            for _ in texts
        ]

    def close(self) -> None:
        return None


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    seen: set[str] = set()
    for member in archive.infolist():
        normalized = member.filename.replace("\\", "/")
        relative = PurePosixPath(normalized)
        if (
            not normalized
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or ":" in relative.parts[0]
        ):
            raise ValueError(f"unsafe archive member: {member.filename!r}")
        canonical_member = relative.as_posix().rstrip("/")
        if not canonical_member or canonical_member in seen:
            raise ValueError(f"duplicate archive member: {member.filename!r}")
        seen.add(canonical_member)

        unix_mode = (member.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in (0, stat.S_IFDIR, stat.S_IFREG):
            raise ValueError(f"non-regular archive member: {member.filename!r}")

        target = destination.joinpath(*relative.parts)
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=False)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, "r") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output)


def _deploy_zips(dist_dir: Path, stage: Path) -> dict[str, str]:
    """Deploy the six archives under ``ops/user/<package>``."""

    (stage / "ops").mkdir(parents=True, exist_ok=False)
    (stage / "ops" / "__init__.py").write_text("", encoding="utf-8")
    (stage / "ops" / "user").mkdir(parents=True, exist_ok=False)
    (stage / "ops" / "user" / "__init__.py").write_text("", encoding="utf-8")
    hashes: dict[str, str] = {}
    for package in PACKAGES:
        zip_path = dist_dir / f"{package}.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(
                f"missing operator archive: {zip_path}; "
                "run scripts/package_operators.py first"
            )
        hashes[package] = _sha256(zip_path)
        package_dir = stage / "ops" / "user" / package
        package_dir.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(zip_path) as archive:
            _safe_extract(archive, package_dir)
    return hashes


def _resolve_operator(registry: Any, name: str) -> Any:
    entry = registry.modules.get(name)
    if entry is None:
        raise KeyError(f"{name} is not registered in the real DataMate registry")
    if isinstance(entry, str):
        module = importlib.import_module(entry)
        return getattr(module, name)
    return entry


def _entities_of(entity_output: dict[str, Any]) -> list[dict[str, Any]]:
    return json.loads(entity_output["text"]).get("entities", [])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sidecar_leaks_text(records: list[dict[str, Any]]) -> bool:
    needles = ("高血压", "硝苯地平", "头痛", "现病史")
    return any(
        any(needle in json.dumps(record, ensure_ascii=False) for needle in needles)
        for record in records
    )


def _append_error(result: dict[str, Any], error: BaseException) -> None:
    text = f"{type(error).__name__}: {error}"
    previous = result.get("error")
    result["error"] = f"{previous}; {text}" if previous else text
    result["passed"] = False


def _run_operator_smoke(
    *,
    dmcompat: Any,
    dist_dir: Path,
    stage: Path,
    result: dict[str, Any],
) -> None:
    result["steps"]["upload_zip_sha256"] = _deploy_zips(dist_dir, stage)
    sys.path.insert(0, str(stage))

    ops_package = importlib.import_module("ops")
    ops_paths = getattr(ops_package, "__path__", None)
    if ops_paths is None:
        raise RuntimeError("the DataMate ops package has no package search path")
    stage_ops = str(stage / "ops")
    if stage_ops not in ops_paths:
        if hasattr(ops_paths, "insert"):
            ops_paths.insert(0, stage_ops)
        else:
            ops_package.__path__ = [stage_ops, *list(ops_paths)]

    imported: list[str] = []
    for package in PACKAGES:
        importlib.import_module(f"ops.user.{package}")
        imported.append(package)
    result["steps"]["import_ok"] = imported
    result["datamate"]["operator_count_after_package_import"] = len(
        dmcompat.OPERATORS.modules
    )
    expected_after_import = (
        result["datamate"]["operator_count"] + len(PACKAGES)
    )
    if result["datamate"]["operator_count_after_package_import"] != (
        expected_after_import
    ):
        raise RuntimeError(
            "DataMate operator registry count changed unexpectedly after "
            "package import: "
            f"expected={expected_after_import} actual="
            f"{result['datamate']['operator_count_after_package_import']}"
        )

    sample = _make_sample(_SAMPLE_TEXT, "smoke_0001")
    entity_count = 0
    for operator_name, parameters in CHAIN:
        operator_class = _resolve_operator(dmcompat.OPERATORS, operator_name)
        operator = operator_class(
            is_first_op=False,
            is_last_op=False,
            **parameters,
        )
        sample = operator.execute(sample)
        if operator_name == "MedicalEntityExtractor":
            entity_count = len(json.loads(sample["text"]).get("entities", []))
    triples = json.loads(sample["text"]).get("triples", [])
    result["steps"]["chain_invoke"] = {
        "ok": True,
        "n_entities": entity_count,
        "n_triples": len(triples),
    }
    if entity_count == 0 or not triples:
        raise AssertionError(
            f"operator chain returned entities={entity_count}, "
            f"triples={len(triples)}"
        )

    shadow_class = _resolve_operator(
        dmcompat.OPERATORS,
        "MedicalEntityExtractorOmShadow",
    )
    base_sample = _make_sample(_SAMPLE_TEXT, "smoke_shadow")
    disabled = shadow_class(
        shadowEnabled=False,
        is_first_op=False,
        is_last_op=False,
    )
    disabled_output = disabled.execute(dict(base_sample))
    entity_class = _resolve_operator(dmcompat.OPERATORS, "MedicalEntityExtractor")
    entity_only = entity_class(
        backend="dict",
        is_first_op=False,
        is_last_op=False,
    )
    entity_output = entity_only.execute(dict(base_sample))
    no_impact = disabled_output["text"] == entity_output["text"]
    disabled.close()
    result["steps"]["shadow_disabled_no_impact"] = bool(no_impact)
    if not no_impact:
        raise AssertionError("disabled shadow changed the primary output")

    os.environ.setdefault("DKI_SHADOW_HASH_KEY", "smoke-key")
    failure_sidecar = stage / "shadow_fail.jsonl"
    failing = shadow_class(
        shadowEnabled=True,
        shadow_runtime=_FailingRuntime(),
        sidecarPath=str(failure_sidecar),
        shadowTimeoutMs=500,
        is_first_op=False,
        is_last_op=False,
    )
    failure_output = failing.execute(dict(base_sample))
    failing.flush(timeout_s=3.0)
    failing.close()
    primary_ok = (
        json.loads(failure_output["text"]).get("entities")
        == _entities_of(entity_output)
    )
    failure_records = _read_jsonl(failure_sidecar)
    isolated = bool(failure_records) and all(
        record.get("error") for record in failure_records
    )
    leaks_text = _sidecar_leaks_text(failure_records)
    result["steps"]["shadow_failure_isolation"] = {
        "primary_unaffected": bool(primary_ok),
        "sidecar_error_recorded": bool(isolated),
        "sidecar_has_raw_text": bool(leaks_text),
    }
    if not primary_ok or not isolated or leaks_text:
        raise AssertionError(
            "failing shadow was not isolated or leaked raw medical text"
        )

    success_sidecar = stage / "shadow_ok.jsonl"
    stub = shadow_class(
        shadowEnabled=True,
        shadow_runtime=_StubRuntime(),
        sidecarPath=str(success_sidecar),
        shadowTimeoutMs=500,
        is_first_op=False,
        is_last_op=False,
    )
    stub.execute(dict(base_sample))
    stub.flush(timeout_s=3.0)
    stub.close()
    success_records = _read_jsonl(success_sidecar)
    success_recorded = bool(
        success_records and success_records[0].get("shadow_count") is not None
    )
    result["steps"]["shadow_normal_path"] = {
        "sidecar_written": bool(success_records),
        "shadow_count_recorded": success_recorded,
    }
    if not success_records or not success_recorded:
        raise AssertionError("successful shadow did not write its sidecar record")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict smoke test for a real DataMate runtime"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "verification" / "datamate_smoke.json",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=ROOT / "dist" / "operators",
    )
    args = parser.parse_args()

    output_path = _absolute_path(args.output)
    if os.path.lexists(output_path):
        print(
            f"refusing to overwrite existing DataMate receipt: {output_path}",
            file=sys.stderr,
        )
        return 2

    # Strict mode is part of this executable's acceptance contract.  Set it
    # before importing any module that can transitively import dki.dmcompat.
    os.environ["DKI_REQUIRE_REAL_DATAMATE"] = "1"
    os.environ["DKI_USE_DATAMATE_RUNTIME"] = "1"
    sys.dont_write_bytecode = True

    result: dict[str, Any] = {
        "schema_version": 2,
        "generated_utc": _utc_now(),
        "receipt_path": str(output_path),
        "strict_require_real": True,
        "deploy_layout": "ops.user.<package>",
        "formal_microbatch": False,
        "npu_runtime_exercised": False,
        "datamate_available": False,
        "datamate": {
            "version": None,
            "pytz_version": None,
            "mapper_module": None,
            "runtime_mode": "unavailable",
            "operator_count": 0,
        },
        "runtime": None,
        "steps": {},
        "passed": False,
    }
    runtime_dir: Path | None = None
    runtime_summary: dict[str, Any] | None = None
    stage: Path | None = None

    try:
        from dki.evidence_tree import scan_bound_regular_tree
        from dki.paths import find_datamate_runtime

        configured_runtime = os.getenv("DKI_DATAMATE_RUNTIME", "").strip()
        runtime_dir = find_datamate_runtime()
        if runtime_dir is None:
            raise FileNotFoundError(
                "no DataMate runtime root containing both ops/ and "
                "python-executor/ was found"
            )
        if configured_runtime:
            configured_canonical = Path(configured_runtime).expanduser().resolve()
            if runtime_dir != configured_canonical:
                raise FileNotFoundError(
                    "DKI_DATAMATE_RUNTIME is invalid; refusing fallback to "
                    f"{runtime_dir}"
                )
        os.environ["DKI_DATAMATE_RUNTIME"] = str(runtime_dir)

        runtime_report = scan_bound_regular_tree(
            runtime_dir,
            label="DataMate runtime",
        )
        runtime_dir = Path(runtime_report["root"])
        runtime_summary = _tree_summary(runtime_report)
        result["runtime"] = {
            "canonical_root": str(runtime_dir),
            "tree": runtime_summary,
        }

        from dki import dmcompat

        bootstrap_runtime = getattr(dmcompat, "_DATAMATE_RUNTIME_DIR", None)
        if bootstrap_runtime is None:
            raise RuntimeError("dki.dmcompat did not bind a DataMate runtime")
        bootstrap_runtime = Path(bootstrap_runtime).resolve()
        if bootstrap_runtime != runtime_dir:
            raise RuntimeError(
                "dki.dmcompat bound a different DataMate runtime: "
                f"{bootstrap_runtime}"
            )
        builtins_loaded = dmcompat.load_builtin_ops()
        mapper_module = str(dmcompat.Mapper.__module__)
        real_runtime = bool(
            dmcompat.DATAMATE_AVAILABLE
            and mapper_module.startswith("datamate.")
        )
        result["datamate_available"] = bool(dmcompat.DATAMATE_AVAILABLE)
        result["datamate"] = {
            "version": _package_version("datamate", "datamate"),
            "pytz_version": _package_version("pytz", "pytz"),
            "mapper_module": mapper_module,
            "runtime_mode": "real" if real_runtime else "shim",
            "builtins_loaded": bool(builtins_loaded),
            "operator_count": len(dmcompat.OPERATORS.modules),
        }
        if not real_runtime:
            raise RuntimeError("dki.dmcompat selected the DataMate shim")
        if not builtins_loaded:
            raise RuntimeError("DataMate built-in operator registration failed")
        if result["datamate"]["operator_count"] != 46:
            raise RuntimeError(
                "unexpected DataMate built-in operator count: "
                f"{result['datamate']['operator_count']} (expected 46)"
            )

        stage = Path(tempfile.mkdtemp(prefix="dm_smoke_"))
        _run_operator_smoke(
            dmcompat=dmcompat,
            dist_dir=_absolute_path(args.dist),
            stage=stage,
            result=result,
        )
        result["passed"] = True
    except BaseException as error:
        _append_error(result, error)
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)

        if runtime_dir is not None and runtime_summary is not None:
            try:
                from dki.evidence_tree import scan_bound_regular_tree

                post_report = scan_bound_regular_tree(
                    runtime_dir,
                    label="DataMate runtime after smoke",
                )
                post_summary = _tree_summary(post_report)
                unchanged = post_summary == runtime_summary
                result["steps"]["runtime_tree_unchanged"] = unchanged
                if not unchanged:
                    raise RuntimeError(
                        "DataMate runtime tree changed during the smoke test"
                    )
            except BaseException as error:
                _append_error(result, error)

    try:
        _publish_json_create_once(output_path, result)
    except FileExistsError:
        print(
            f"refusing to overwrite existing DataMate receipt: {output_path}",
            file=sys.stderr,
        )
        return 2
    except BaseException as error:
        print(
            f"failed to publish DataMate receipt create-once: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    print(f"wrote immutable DataMate receipt: {output_path}")
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
