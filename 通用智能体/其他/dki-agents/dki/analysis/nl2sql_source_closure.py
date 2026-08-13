"""Fail-closed static local-import closure for a sealed NL2SQL runtime.

This deliberately small analyzer is separate from the general execution-bundle
loader: the latter contains controlled dynamic loader primitives and therefore
must not become part of the runtime closure it is intended to audit.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTRY_MODULES = (
    "dki.analysis.nl2sql",
    "dki.analysis.nl2sql_eval",
    "dki.analysis.nl2sql_protocol",
    "dki.analysis.nl2sql_seal",
    "scripts.eval_nl2sql_blind",
    "scripts.freeze_nl2sql_blind",
    "scripts.seal_nl2sql_blind_candidate",
)
_LOCAL_TOP_LEVEL = frozenset({"dki", "scripts"})
_DYNAMIC_CALLS = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "importlib.import_module",
        "importlib.util.spec_from_file_location",
        "runpy.run_module",
    }
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_for_module(root: Path, module: str) -> tuple[Path, bool] | None:
    if not module or module.split(".")[0] not in _LOCAL_TOP_LEVEL:
        return None
    parts = module.split(".")
    package = root.joinpath(*parts, "__init__.py")
    if package.is_file():
        return package, True
    source = root.joinpath(*parts).with_suffix(".py")
    if source.is_file():
        return source, False
    return None


def _module_package(module: str, is_package: bool) -> str:
    return module if is_package else module.rpartition(".")[0]


def _resolve_from_import(
    node: ast.ImportFrom,
    *,
    current_module: str,
    current_is_package: bool,
) -> str | None:
    if node.level == 0:
        return node.module
    package = _module_package(current_module, current_is_package)
    if not package:
        raise ValueError(f"relative import has no package context: {current_module}")
    parts = package.split(".")
    if node.level > len(parts):
        raise ValueError(f"relative import escapes package: {current_module}")
    base = parts[: len(parts) - node.level + 1]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _assert_no_dynamic_loading(tree: ast.AST, module: str) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_name(node.func)
        if name in _DYNAMIC_CALLS or (name and name.rsplit(".", 1)[-1] in {"eval", "exec"}):
            raise ValueError(f"dynamic code loading is forbidden in sealed closure: {module}")


def discover_source_closure(
    source_root: Path = ROOT,
    entry_modules: Iterable[str] = DEFAULT_ENTRY_MODULES,
) -> list[dict[str, Any]]:
    """Return a sorted static local closure, rejecting missing/dynamic modules."""

    root = Path(source_root).resolve()
    pending = list(dict.fromkeys(str(item) for item in entry_modules))
    if not pending:
        raise ValueError("source closure requires at least one entry module")
    seen: dict[str, dict[str, Any]] = {}
    while pending:
        module = pending.pop(0)
        if module in seen:
            continue
        resolved = _source_for_module(root, module)
        if resolved is None:
            if module.split(".")[0] in _LOCAL_TOP_LEVEL:
                raise ValueError(f"local runtime import is missing: {module}")
            continue
        path, is_package = resolved
        data = path.read_bytes()
        try:
            tree = ast.parse(data, filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"runtime source is not parseable: {module}") from exc
        _assert_no_dynamic_loading(tree, module)
        relative = path.relative_to(root).as_posix()
        seen[module] = {
            "module": module,
            "path": relative,
            "size": len(data),
            "sha256": _sha256_bytes(data),
            "is_package": is_package,
        }
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from_import(
                    node,
                    current_module=module,
                    current_is_package=is_package,
                )
                if base:
                    targets.add(base)
                    for alias in node.names:
                        candidate = f"{base}.{alias.name}"
                        if alias.name != "*" and _source_for_module(root, candidate) is not None:
                            targets.add(candidate)
        for target in sorted(targets):
            if target.split(".")[0] in _LOCAL_TOP_LEVEL and target not in seen:
                pending.append(target)
    return sorted(seen.values(), key=lambda item: (str(item["path"]), str(item["module"])))


def closure_sha256(entries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["module"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_source_closure_manifest(
    output: Path,
    *,
    source_root: Path = ROOT,
    entry_modules: Iterable[str] = DEFAULT_ENTRY_MODULES,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"source closure output exists: {output}")
    entries = discover_source_closure(source_root, entry_modules)
    payload = {
        "schema_version": 1,
        "kind": "nl2sql_static_execution_closure",
        "created_once": True,
        "entry_modules": list(dict.fromkeys(str(item) for item in entry_modules)),
        "algorithm": "sha256 of sorted module NUL path NUL size NUL file_sha256 LF",
        "closure_sha256": closure_sha256(entries),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload
