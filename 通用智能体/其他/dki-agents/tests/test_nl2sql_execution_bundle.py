from __future__ import annotations

import collections
import hashlib
import json
import os
import stat
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

from dki.analysis import nl2sql_execution_bundle as BUNDLE


ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = ROOT / "scripts" / "build_nl2sql_execution_bundle.py"
INNER_SCRIPT = ROOT / "scripts" / "run_nl2sql_execution_bundle_inner.py"
RUNNER_ENVIRONMENT_MANIFEST_BYTES = b'{"kind":"synthetic-runner-environment"}\n'
RUNNER_ENVIRONMENT_MANIFEST_SHA256 = hashlib.sha256(
    RUNNER_ENVIRONMENT_MANIFEST_BYTES
).hexdigest()


def _runner_environment_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "synthetic-runner-environment.json"
    manifest.write_bytes(RUNNER_ENVIRONMENT_MANIFEST_BYTES)
    return manifest


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _synthetic_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "synthetic_source"
    _write(source_root / "demo" / "__init__.py", "\n")
    _write(
        source_root / "demo" / "entry.py",
        "from .helper import PREFIX\n"
        "from .nested import SUFFIX\n"
        "import json\n\n"
        "def answer() -> str:\n"
        "    return PREFIX + SUFFIX + json.dumps({'source': 'bundle'})\n",
    )
    _write(source_root / "demo" / "helper.py", "PREFIX = 'trusted-'\n")
    _write(
        source_root / "demo" / "nested" / "__init__.py",
        "from .util import SUFFIX\n",
    )
    _write(source_root / "demo" / "nested" / "util.py", "SUFFIX = 'snapshot-'\n")
    asset = source_root / "assets" / "settings.json"
    _write(asset, '{"mode":"synthetic"}\n')
    return source_root, asset


def _build(tmp_path: Path, *, name: str = "bundle.zip") -> tuple[
    Path,
    Path,
    BUNDLE.BuiltExecutionBundle,
]:
    source_root, asset = _synthetic_source_tree(tmp_path)
    archive = tmp_path / name
    bundle = BUNDLE.build_execution_bundle(
        archive,
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        runtime_assets={"runtime/settings.json": asset},
    )
    return source_root, archive, bundle


def _canonical_info(name: str, *, mode: int = stat.S_IFREG | 0o444) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=BUNDLE.ZIP_EPOCH)
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = mode << 16
    return info


def _single_module_manifest(
    module: str,
    source_bytes: bytes,
) -> tuple[dict[str, object], str]:
    relative = f"{module.replace('.', '/')}.py"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    file_entry = {
        "path": relative,
        "role": BUNDLE.SOURCE_ROLE,
        "sha256": source_digest,
        "size": len(source_bytes),
    }
    manifest: dict[str, object] = {
        "archive_format": BUNDLE.BUNDLE_ARCHIVE_FORMAT,
        "bundle_content_sha256": BUNDLE._canonical_content_sha256([file_entry]),
        "entry_module": module,
        "files": [file_entry],
        "kind": BUNDLE.BUNDLE_KIND,
        "module_registry": [
            {
                "is_package": False,
                "module": module,
                "path": relative,
                "sha256": source_digest,
            }
        ],
        "payload_root": BUNDLE.BUNDLE_PAYLOAD_ROOT,
        "runner_environment_manifest_sha256": RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        "runtime_assets": [],
        "schema_version": BUNDLE.BUNDLE_SCHEMA_VERSION,
    }
    manifest["manifest_sha256"] = BUNDLE._manifest_sha256(manifest)
    return manifest, relative


def _inner_with_constant_overrides(
    tmp_path: Path,
    name: str,
    replacements: dict[str, str],
) -> Path:
    source = INNER_SCRIPT.read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert old in source
        source = source.replace(old, new, 1)
    copied = tmp_path / name
    _write(copied, source)
    return copied


def test_bundle_is_deterministic_and_captures_transitive_closure(tmp_path: Path) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    first_path = tmp_path / "first.zip"
    second_path = tmp_path / "second.zip"
    first = BUNDLE.build_execution_bundle(
        first_path,
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        runtime_assets={"runtime/settings.json": asset},
    )
    second = BUNDLE.build_execution_bundle(
        second_path,
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        runtime_assets={"runtime/settings.json": asset},
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        BUNDLE.build_execution_bundle(
            first_path,
            source_root=source_root,
            entry_module="demo.entry",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            runtime_assets={"runtime/settings.json": asset},
        )
    assert first.archive_sha256 == second.archive_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert [entry["module"] for entry in first.manifest["module_registry"]] == [
        "demo",
        "demo.entry",
        "demo.helper",
        "demo.nested",
        "demo.nested.util",
    ]
    assert first.manifest["runtime_assets"] == ("runtime/settings.json",)
    assert first.manifest["runner_environment_manifest_sha256"] == (
        RUNNER_ENVIRONMENT_MANIFEST_SHA256
    )
    assert [entry["path"] for entry in first.manifest["files"]] == sorted(
        entry["path"] for entry in first.manifest["files"]
    )
    assert [module.name for module in BUNDLE.discover_local_import_closure(
        source_root,
        "demo.entry",
    )] == [
        "demo",
        "demo.entry",
        "demo.helper",
        "demo.nested",
        "demo.nested.util",
    ]


def test_literal_dynamic_import_is_resolved_into_the_snapshot(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "literal-dynamic-source"
    _write(source_root / "demo" / "__init__.py", "\n")
    _write(
        source_root / "demo" / "entry.py",
        "from importlib import import_module as load\n"
        "HELPER = load('.helper', package=__package__)\n"
        "VALUE = HELPER.VALUE\n",
    )
    _write(source_root / "demo" / "helper.py", "VALUE = 'captured'\n")
    built = BUNDLE.build_execution_bundle(
        tmp_path / "literal-dynamic.zip",
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
    )

    assert [entry["module"] for entry in built.manifest["module_registry"]] == [
        "demo",
        "demo.entry",
        "demo.helper",
    ]
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(built.archive_path),
        tmp_path / "literal-dynamic-stage",
    )
    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nname = 'demo.helper'\nimportlib.import_module(name)\n",
        "name = 'demo.helper'\n__import__(name)\n",
        "__import__('demo', fromlist=['helper'])\n",
        "import runpy\nrunpy.run_path('untrusted.py')\n",
        "from importlib.machinery import SourceFileLoader\n"
        "SourceFileLoader('untrusted', 'untrusted.py')\n",
        "exec(\"VALUE = 'untrusted'\")\n",
        "import importlib\nloader = importlib.import_module\n"
        "loader('demo.helper')\n",
        "globals()['__import__']('demo.helper')\n",
    ],
)
def test_unresolved_dynamic_loading_is_rejected(
    tmp_path: Path,
    source: str,
) -> None:
    source_root = tmp_path / "dynamic-source"
    _write(source_root / "demo" / "__init__.py", "\n")
    _write(source_root / "demo" / "entry.py", source)
    _write(source_root / "demo" / "helper.py", "VALUE = 'unused'\n")

    with pytest.raises(ValueError, match="(?:dynamic|unresolved|code loading|loader)"):
        BUNDLE.build_execution_bundle(
            tmp_path / "dynamic.zip",
            source_root=source_root,
            entry_module="demo.entry",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            extra_modules=("demo.helper",),
        )


def test_runner_environment_binding_survives_build_and_verification(
    tmp_path: Path,
) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    runner_digest = hashlib.sha256(b'{"kind":"alternate-runner"}\n').hexdigest()
    assert runner_digest != RUNNER_ENVIRONMENT_MANIFEST_SHA256

    built = BUNDLE.build_execution_bundle(
        tmp_path / "runner-bound.zip",
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=runner_digest,
        runtime_assets={"runtime/settings.json": asset},
    )
    verified = BUNDLE.verify_execution_bundle(built.archive_path)

    assert built.manifest["runner_environment_manifest_sha256"] == runner_digest
    assert verified.manifest["runner_environment_manifest_sha256"] == runner_digest


def test_builder_uses_one_snapshot_read_per_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    original_anchored_reader = BUNDLE._read_anchored_relative_bytes
    original_stable_reader = BUNDLE.read_stable_bytes
    source_reads: list[str] = []
    stable_reads: list[Path] = []

    def tracked_anchored_reader(
        root: object,
        relative: str,
        *,
        label: str,
        maximum_bytes: int | None = None,
    ) -> bytes:
        source_reads.append(relative)
        return original_anchored_reader(
            root,
            relative,
            label=label,
            maximum_bytes=maximum_bytes,
        )

    def tracked_stable_reader(
        path: Path,
        *,
        label: str = "file",
        maximum_bytes: int | None = None,
    ) -> bytes:
        stable_reads.append(BUNDLE._absolute_path(path))
        return original_stable_reader(
            path,
            label=label,
            maximum_bytes=maximum_bytes,
        )

    monkeypatch.setattr(
        BUNDLE,
        "_read_anchored_relative_bytes",
        tracked_anchored_reader,
    )
    monkeypatch.setattr(BUNDLE, "read_stable_bytes", tracked_stable_reader)
    BUNDLE.build_execution_bundle(
        tmp_path / "single-read.zip",
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        runtime_assets={"runtime/settings.json": asset},
    )

    expected_sources = {
        "demo/__init__.py",
        "demo/entry.py",
        "demo/helper.py",
        "demo/nested/__init__.py",
        "demo/nested/util.py",
    }
    assert collections.Counter(source_reads) == {
        relative: 1 for relative in expected_sources
    }
    stable_counts = collections.Counter(stable_reads)
    assert stable_counts[BUNDLE._absolute_path(asset)] == 1
    assert all(count == 1 for count in stable_counts.values())


def test_builder_applies_combined_payload_entry_and_byte_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "budget-source"
    source = source_root / "demo.py"
    asset = tmp_path / "asset.bin"
    _write(source, "VALUE = 1\n")
    asset.write_bytes(b"xy")
    source_size = source.stat().st_size

    monkeypatch.setattr(BUNDLE, "MAX_TOTAL_BYTES", source_size + 1)
    with pytest.raises(ValueError, match="runtime asset.*exceeds the byte limit"):
        BUNDLE.build_execution_bundle(
            tmp_path / "byte-budget.zip",
            source_root=source_root,
            entry_module="demo",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            runtime_assets={"runtime/asset.bin": asset},
        )

    monkeypatch.setattr(BUNDLE, "MAX_TOTAL_BYTES", 1024)
    monkeypatch.setattr(BUNDLE, "MAX_FILE_ENTRIES", 1)
    with pytest.raises(ValueError, match="runtime assets exceed the file-entry limit"):
        BUNDLE.build_execution_bundle(
            tmp_path / "entry-budget.zip",
            source_root=source_root,
            entry_module="demo",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            runtime_assets={"runtime/asset.bin": asset},
        )


def test_builder_applies_member_manifest_and_archive_byte_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "bounded-source"
    source = source_root / "demo.py"
    _write(source, "VALUE = 'too large'\n")

    monkeypatch.setattr(BUNDLE, "MAX_MEMBER_BYTES", 1)
    with pytest.raises(ValueError, match="local module.*exceeds the byte limit"):
        BUNDLE.build_execution_bundle(
            tmp_path / "member-budget.zip",
            source_root=source_root,
            entry_module="demo",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        )

    monkeypatch.setattr(BUNDLE, "MAX_MEMBER_BYTES", 1024)
    monkeypatch.setattr(BUNDLE, "MAX_MANIFEST_BYTES", 64)
    with pytest.raises(ValueError, match="manifest exceeds the byte limit"):
        BUNDLE.build_execution_bundle(
            tmp_path / "manifest-budget.zip",
            source_root=source_root,
            entry_module="demo",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        )

    monkeypatch.setattr(BUNDLE, "MAX_MANIFEST_BYTES", 1024 * 1024)
    monkeypatch.setattr(BUNDLE, "MAX_ARCHIVE_BYTES", 64)
    with pytest.raises(ValueError, match="archive exceeds the byte limit"):
        BUNDLE.build_execution_bundle(
            tmp_path / "archive-budget.zip",
            source_root=source_root,
            entry_module="demo",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        )


def test_stable_reader_rejects_a_descriptor_from_a_swapped_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "expected.txt"
    replacement = tmp_path / "replacement.txt"
    expected.write_bytes(b"expected")
    replacement.write_bytes(b"replacement")
    original_open = os.open

    def swapped_open(path: str | Path, flags: int) -> int:
        assert Path(path) == expected
        return original_open(replacement, flags)

    monkeypatch.setattr(BUNDLE.os, "open", swapped_open)
    with pytest.raises(ValueError, match="changed while being opened"):
        BUNDLE.read_stable_bytes(expected, label="synthetic source")


def test_stable_reader_rejects_oversize_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "oversize.bin"
    candidate.write_bytes(b"oversize")

    def unexpected_read(descriptor: int, size: int) -> bytes:
        raise AssertionError(f"oversize descriptor was read: {descriptor}, {size}")

    monkeypatch.setattr(BUNDLE.os, "read", unexpected_read)
    with pytest.raises(ValueError, match="exceeds the byte limit"):
        BUNDLE.read_stable_bytes(
            candidate,
            label="oversize synthetic file",
            maximum_bytes=1,
        )


def test_stable_reader_detects_growth_with_one_byte_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "growing.bin"
    candidate.write_bytes(b"abc")
    original_read = BUNDLE.os.read
    requested_sizes: list[int] = []
    grew = False

    def growing_read(descriptor: int, size: int) -> bytes:
        nonlocal grew
        requested_sizes.append(size)
        data = original_read(descriptor, size)
        if data and not grew:
            grew = True
            with candidate.open("ab") as stream:
                stream.write(b"unbounded-appended-data")
        return data

    monkeypatch.setattr(BUNDLE.os, "read", growing_read)
    with pytest.raises(ValueError, match="changed while being read"):
        BUNDLE.read_stable_bytes(candidate, label="growing synthetic file")
    assert requested_sizes == [3, 1]


def test_verify_extracts_only_verified_snapshot_into_fresh_directory(tmp_path: Path) -> None:
    source_root, archive, built = _build(tmp_path)
    verified = BUNDLE.verify_execution_bundle(
        archive,
        expected_archive_sha256=built.archive_sha256,
    )
    _write(
        source_root / "demo" / "helper.py",
        "raise RuntimeError('checkout source must not be used')\n",
    )

    stage_root = tmp_path / "stage"
    staged = BUNDLE.extract_execution_bundle(verified, stage_root)
    assert staged.archive_sha256 == built.archive_sha256
    assert staged.manifest_sha256 == built.manifest_sha256
    assert (staged.payload_root / "demo" / "helper.py").read_text(
        encoding="utf-8"
    ) == "PREFIX = 'trusted-'\n"
    checked = BUNDLE.verify_staged_execution_tree(stage_root)
    assert checked.manifest_sha256 == built.manifest_sha256

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        BUNDLE.extract_execution_bundle(verified, occupied)
    assert marker.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize(
    "relative",
    [
        "payload/demo/__pycache__/entry.pyc",
        "payload/demo/entry.pyo",
        "payload/injected.pth",
        "payload/sitecustomize.py",
    ],
)
def test_staged_verifier_rejects_bytecode_and_startup_artifacts(
    tmp_path: Path,
    relative: str,
) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    artifact = staged.root.joinpath(*relative.split("/"))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"unsafe")

    with pytest.raises(ValueError, match="forbidden Python (?:bytecode|startup) path"):
        BUNDLE.verify_staged_execution_tree(staged.root)


def test_verifiers_reject_linked_archive_and_stage_roots(tmp_path: Path) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    archive_link = tmp_path / "archive-link.zip"
    try:
        archive_link.symlink_to(archive)
    except OSError:
        archive_link = None
    if archive_link is not None:
        with pytest.raises(ValueError, match="(?:symlink|regular non-symlink)"):
            BUNDLE.verify_execution_bundle(archive_link)

    stage_link = tmp_path / "stage-link"
    try:
        stage_link.symlink_to(staged.root, target_is_directory=True)
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory link creation is unavailable: {symlink_error}")
        junction = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(stage_link), str(staged.root)],
            text=True,
            capture_output=True,
            check=False,
        )
        if junction.returncode != 0:
            pytest.skip(
                "directory link creation is unavailable: "
                f"{symlink_error}; {junction.stderr}"
            )

    with pytest.raises(ValueError, match="stage root must be a real directory"):
        BUNDLE.verify_staged_execution_tree(stage_link)

    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=stage_link,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "stage root must be a real directory" in completed.stderr


def test_inner_assertion_uses_staged_module_not_checkout(tmp_path: Path) -> None:
    source_root, archive, built = _build(tmp_path)
    verified = BUNDLE.verify_execution_bundle(archive)
    staged = BUNDLE.extract_execution_bundle(verified, tmp_path / "stage")
    _write(
        source_root / "demo" / "entry.py",
        "raise RuntimeError('ambient checkout was imported')\n",
    )
    runner_manifest = _runner_environment_manifest(tmp_path)
    startup_directory = tmp_path / "untrusted-startup"
    site_marker = tmp_path / "sitecustomize-ran.txt"
    pth_marker = tmp_path / "pth-ran.txt"
    _write(
        startup_directory / "sitecustomize.py",
        "from pathlib import Path\n"
        f"Path({str(site_marker)!r}).write_text('unexpected', encoding='utf-8')\n",
    )
    _write(
        startup_directory / "injected.pth",
        "import pathlib; "
        f"pathlib.Path({str(pth_marker)!r}).write_text('unexpected', encoding='utf-8')\n",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(startup_directory), str(source_root)))
    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=runner_manifest,
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        cwd=source_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not site_marker.exists()
    assert not pth_marker.exists()
    assert not list(staged.payload_root.rglob("__pycache__"))
    payload = json.loads(completed.stdout)
    assert payload["entry_module"] == "demo.entry"
    assert Path(payload["module_origin"]).resolve() == (
        staged.payload_root / "demo" / "entry.py"
    ).resolve()
    assert payload["stage_root"] == str(staged.root)
    assert payload["runner_environment_manifest_sha256"] == (
        RUNNER_ENVIRONMENT_MANIFEST_SHA256
    )
    assert payload["bundle_manifest_sha256"] == staged.manifest_sha256
    assert built.manifest["entry_module"] == "demo.entry"


def test_inner_rejects_unsafe_startup_before_shadowable_imports(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "inner-startup-fixture"
    copied_inner = fixture / INNER_SCRIPT.name
    _write(copied_inner, INNER_SCRIPT.read_text(encoding="utf-8"))
    marker = tmp_path / "shadow-hashlib-imported.txt"
    _write(
        fixture / "hashlib.py",
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
    )

    completed = subprocess.run(
        [sys.executable, "-S", "-B", str(copied_inner), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "requires python -I" in completed.stderr
    assert not marker.exists()


def test_inner_fails_closed_for_runner_identity_and_site_mode(tmp_path: Path) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    runner_manifest = _runner_environment_manifest(tmp_path)
    command = BUNDLE.isolated_inner_command(
        python_executable=sys.executable,
        inner_entrypoint=INNER_SCRIPT,
        stage_root=staged.root,
        runner_environment_manifest=runner_manifest,
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        bundle_manifest_sha256=staged.manifest_sha256,
    )
    runner_manifest.write_bytes(b"changed runner environment manifest\n")
    changed_runner = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert changed_runner.returncode != 0
    assert "runner environment manifest bytes differ from expected" in changed_runner.stderr

    runner_manifest.write_bytes(RUNNER_ENVIRONMENT_MANIFEST_BYTES)
    mismatched_bundle_digest = hashlib.sha256(b"different runner manifest\n").hexdigest()
    mismatched_bundle = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=runner_manifest,
            runner_environment_manifest_sha256=mismatched_bundle_digest,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert mismatched_bundle.returncode != 0
    assert "runner environment manifest SHA-256 differs from bundle" in (
        mismatched_bundle.stderr
    )

    no_site_command = [part for part in command if part != "-S"]
    missing_site_mode = subprocess.run(
        no_site_command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_site_mode.returncode != 0
    assert "requires python -S" in missing_site_mode.stderr


def test_inner_never_falls_back_to_pythonpath_dependencies(tmp_path: Path) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    _write(source_root / "demo" / "entry.py", "import ambient_dependency\n")
    archive = tmp_path / "bundle.zip"
    built = BUNDLE.build_execution_bundle(
        archive,
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        runtime_assets={"runtime/settings.json": asset},
    )
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(
            archive,
            expected_archive_sha256=built.archive_sha256,
        ),
        tmp_path / "stage",
    )
    runner_manifest = _runner_environment_manifest(tmp_path)
    ambient_directory = tmp_path / "ambient-dependencies"
    _write(ambient_directory / "ambient_dependency.py", "VALUE = 'ambient'\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ambient_directory)

    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=runner_manifest,
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "No module named 'ambient_dependency'" in completed.stderr


def test_inner_subprocess_rejects_unregistered_standard_library_injection(
    tmp_path: Path,
) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    marker = tmp_path / "fractions-imported.txt"
    _write(
        staged.payload_root / "fractions.py",
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
    )

    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "staged execution bundle" in completed.stderr
    assert not marker.exists()


def test_inner_subprocess_rejects_a_preloaded_registered_module(
    tmp_path: Path,
) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    runner_manifest = _runner_environment_manifest(tmp_path)
    inner_arguments = BUNDLE.isolated_inner_command(
        python_executable=sys.executable,
        inner_entrypoint=INNER_SCRIPT,
        stage_root=staged.root,
        runner_environment_manifest=runner_manifest,
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        bundle_manifest_sha256=staged.manifest_sha256,
    )[5:]
    bootstrap = (
        "import runpy, sys, types\n"
        "sys.modules['demo'] = types.ModuleType('demo')\n"
        f"sys.argv = {[str(INNER_SCRIPT), *inner_arguments]!r}\n"
        f"runpy.run_path({str(INNER_SCRIPT)!r}, run_name='__main__')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", bootstrap],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "staged registry module is already loaded: demo" in completed.stderr


def test_inner_rejects_a_grown_member_before_reading_beyond_declared_size(
    tmp_path: Path,
) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    with (staged.payload_root / "demo" / "helper.py").open("ab") as stream:
        stream.write(b"untrusted appended source")

    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "staged execution-bundle member exceeds the byte limit" in completed.stderr


@pytest.mark.parametrize(
    ("constant", "replacement", "message"),
    [
        ("MAX_MANIFEST_BYTES = 4 * 1024 * 1024", "MAX_MANIFEST_BYTES = 64", "manifest exceeds the byte limit"),
        ("MAX_TOTAL_BYTES = 512 * 1024 * 1024", "MAX_TOTAL_BYTES = 1", "total size limit"),
        ("MAX_FILE_ENTRIES = 10_000", "MAX_FILE_ENTRIES = 1", "file-entry limit"),
    ],
)
def test_inner_applies_manifest_total_and_entry_budgets(
    tmp_path: Path,
    constant: str,
    replacement: str,
    message: str,
) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    bounded_inner = _inner_with_constant_overrides(
        tmp_path,
        f"bounded-inner-{message.replace(' ', '-')}.py",
        {constant: replacement},
    )

    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=bounded_inner,
            stage_root=staged.root,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert message in completed.stderr


def test_inner_bounds_the_runner_manifest_before_reading(
    tmp_path: Path,
) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    runner_bytes = b"r" * (64 * 1024 + 1)
    runner_digest = hashlib.sha256(runner_bytes).hexdigest()
    runner_manifest = tmp_path / "oversize-runner-manifest.json"
    runner_manifest.write_bytes(runner_bytes)
    built = BUNDLE.build_execution_bundle(
        tmp_path / "runner-budget.zip",
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=runner_digest,
        runtime_assets={"runtime/settings.json": asset},
    )
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(built.archive_path),
        tmp_path / "runner-budget-stage",
    )
    bounded_inner = _inner_with_constant_overrides(
        tmp_path,
        "runner-bounded-inner.py",
        {"MAX_MANIFEST_BYTES = 4 * 1024 * 1024": "MAX_MANIFEST_BYTES = 64 * 1024"},
    )

    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=bounded_inner,
            stage_root=staged.root,
            runner_environment_manifest=runner_manifest,
            runner_environment_manifest_sha256=runner_digest,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "runner environment manifest exceeds the byte limit" in completed.stderr


def test_in_process_origin_assertion_requires_the_staged_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    for name in (
        "demo",
        "demo.entry",
        "demo.helper",
        "demo.nested",
        "demo.nested.util",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(staged.payload_root))

    entry = BUNDLE.assert_staged_entry_module_origin(staged)

    assert Path(entry.__file__).resolve() == (
        staged.payload_root / "demo" / "entry.py"
    ).resolve()


def test_in_process_origin_assertion_rechecks_staged_content(tmp_path: Path) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    _write(
        staged.payload_root / "demo" / "helper.py",
        "raise RuntimeError('changed staged source must not be imported')\n",
    )

    with pytest.raises(
        ValueError,
        match="(?:member binding failed|exceeds the byte limit)",
    ):
        BUNDLE.assert_staged_entry_module_origin(staged)


@pytest.mark.parametrize(
    "startup_flags",
    [
        ("-S", "-B"),
        ("-I", "-B"),
        ("-I", "-S"),
    ],
)
def test_builder_rejects_unsafe_startup_before_importing_project_code(
    tmp_path: Path,
    startup_flags: tuple[str, ...],
) -> None:
    fixture_root = tmp_path / "builder-fixture"
    copied_builder = fixture_root / "scripts" / BUILD_SCRIPT.name
    _write(copied_builder, BUILD_SCRIPT.read_text(encoding="utf-8"))
    marker = tmp_path / "dki-imported.txt"
    _write(
        fixture_root / "dki" / "__init__.py",
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
    )
    _write(fixture_root / "dki" / "analysis" / "__init__.py", "\n")
    _write(
        fixture_root / "dki" / "analysis" / "nl2sql_execution_bundle.py",
        "raise AssertionError('project import preceded startup verification')\n",
    )

    completed = subprocess.run(
        [sys.executable, *startup_flags, str(copied_builder), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "requires Python startup flags -I -S -B" in completed.stderr
    assert not marker.exists()


def test_cli_forwards_runner_manifest_digest_and_builds_only_synthetic_source(
    tmp_path: Path,
) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    runner_digest = hashlib.sha256(b'{"kind":"cli-runner"}\n').hexdigest()
    assert runner_digest != RUNNER_ENVIRONMENT_MANIFEST_SHA256
    archive = tmp_path / "cli.zip"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(BUILD_SCRIPT),
            "--output",
            str(archive),
            "--source-root",
            str(source_root),
            "--entry-module",
            "demo.entry",
            "--runner-environment-manifest-sha256",
            runner_digest,
            "--runtime-asset",
            f"runtime/settings.json={asset}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["entry_module"] == "demo.entry"
    assert payload["n_runtime_assets"] == 1
    assert payload["runner_environment_manifest_sha256"] == runner_digest
    verified = BUNDLE.verify_execution_bundle(archive)
    assert verified.manifest["runner_environment_manifest_sha256"] == runner_digest
    assert verified.archive_sha256 == payload["archive_sha256"]


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("../escape.py", stat.S_IFREG | 0o444),
        ("/absolute.py", stat.S_IFREG | 0o444),
        (BUNDLE.BUNDLE_MANIFEST_NAME, stat.S_IFREG | 0o444),
        ("payload/demo/link.py", stat.S_IFLNK | 0o777),
        ("payload/demo/directory", stat.S_IFDIR | 0o755),
        ("payload/demo/__pycache__/entry.pyc", stat.S_IFREG | 0o444),
        ("payload/demo/entry.pyo", stat.S_IFREG | 0o444),
        ("payload/injected.pth", stat.S_IFREG | 0o444),
        ("payload/sitecustomize.py", stat.S_IFREG | 0o444),
        ("payload/CON.py", stat.S_IFREG | 0o444),
    ],
)
def test_verifier_rejects_unsafe_or_duplicate_archive_members(
    tmp_path: Path,
    name: str,
    mode: int,
) -> None:
    _, archive, _ = _build(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED) as handle:
            handle.writestr(_canonical_info(name, mode=mode), b"unsafe")
    with pytest.raises(ValueError):
        BUNDLE.verify_execution_bundle(archive)


def test_verifier_rejects_member_content_not_bound_by_manifest(tmp_path: Path) -> None:
    _, archive, _ = _build(tmp_path)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive, "r") as source:
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
        }
    members["payload/demo/helper.py"] = b"PREFIX = 'tampered-'\n"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as target:
        for name in sorted(members):
            target.writestr(_canonical_info(name), members[name])
    with pytest.raises(ValueError, match="binding failed"):
        BUNDLE.verify_execution_bundle(tampered)


def test_verifier_applies_archive_manifest_member_total_and_entry_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, archive, _ = _build(tmp_path)
    with zipfile.ZipFile(archive, "r") as handle:
        infos = handle.infolist()
        manifest_size = handle.getinfo(BUNDLE.BUNDLE_MANIFEST_NAME).file_size
        payload_sizes = [
            info.file_size
            for info in infos
            if info.filename != BUNDLE.BUNDLE_MANIFEST_NAME
        ]
    original_archive_limit = BUNDLE.MAX_ARCHIVE_BYTES
    original_manifest_limit = BUNDLE.MAX_MANIFEST_BYTES
    original_member_limit = BUNDLE.MAX_MEMBER_BYTES
    original_total_limit = BUNDLE.MAX_TOTAL_BYTES
    original_entry_limit = BUNDLE.MAX_FILE_ENTRIES

    monkeypatch.setattr(BUNDLE, "MAX_ARCHIVE_BYTES", archive.stat().st_size - 1)
    with pytest.raises(ValueError, match="archive exceeds the byte limit"):
        BUNDLE.verify_execution_bundle(archive)
    monkeypatch.setattr(BUNDLE, "MAX_ARCHIVE_BYTES", original_archive_limit)

    monkeypatch.setattr(BUNDLE, "MAX_MANIFEST_BYTES", manifest_size - 1)
    with pytest.raises(ValueError, match="manifest.*(?:byte limit|size is unsafe)"):
        BUNDLE.verify_execution_bundle(archive)
    monkeypatch.setattr(BUNDLE, "MAX_MANIFEST_BYTES", original_manifest_limit)

    monkeypatch.setattr(BUNDLE, "MAX_MEMBER_BYTES", max(payload_sizes) - 1)
    with pytest.raises(ValueError, match="member size is unsafe"):
        BUNDLE.verify_execution_bundle(archive)
    monkeypatch.setattr(BUNDLE, "MAX_MEMBER_BYTES", original_member_limit)

    monkeypatch.setattr(BUNDLE, "MAX_TOTAL_BYTES", sum(payload_sizes) - 1)
    with pytest.raises(ValueError, match="total size limit"):
        BUNDLE.verify_execution_bundle(archive)
    monkeypatch.setattr(BUNDLE, "MAX_TOTAL_BYTES", original_total_limit)

    original_zip_file = BUNDLE.zipfile.ZipFile

    def unexpected_zip_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("ZIP directory was parsed before its entry-count precheck")

    monkeypatch.setattr(BUNDLE, "MAX_FILE_ENTRIES", len(payload_sizes) - 1)
    monkeypatch.setattr(BUNDLE.zipfile, "ZipFile", unexpected_zip_parse)
    with pytest.raises(ValueError, match="file-entry limit"):
        BUNDLE.verify_execution_bundle(archive)
    monkeypatch.setattr(BUNDLE.zipfile, "ZipFile", original_zip_file)
    monkeypatch.setattr(BUNDLE, "MAX_FILE_ENTRIES", original_entry_limit)


def test_bundled_module_name_validator_covers_build_and_archive_verify(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "stdlib-shadow-source"
    _write(source_root / "json.py", "VALUE = 'untrusted'\n")
    with pytest.raises(ValueError, match="cannot shadow a standard-library module"):
        BUNDLE.build_execution_bundle(
            tmp_path / "stdlib-shadow-build.zip",
            source_root=source_root,
            entry_module="json",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        )

    source_bytes = b"VALUE = 'untrusted'\n"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    file_entry = {
        "path": "json.py",
        "role": BUNDLE.SOURCE_ROLE,
        "sha256": source_digest,
        "size": len(source_bytes),
    }
    manifest = {
        "archive_format": BUNDLE.BUNDLE_ARCHIVE_FORMAT,
        "bundle_content_sha256": BUNDLE._canonical_content_sha256([file_entry]),
        "entry_module": "json",
        "files": [file_entry],
        "kind": BUNDLE.BUNDLE_KIND,
        "module_registry": [
            {
                "is_package": False,
                "module": "json",
                "path": "json.py",
                "sha256": source_digest,
            }
        ],
        "payload_root": BUNDLE.BUNDLE_PAYLOAD_ROOT,
        "runner_environment_manifest_sha256": RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        "runtime_assets": [],
        "schema_version": BUNDLE.BUNDLE_SCHEMA_VERSION,
    }
    manifest["manifest_sha256"] = BUNDLE._manifest_sha256(manifest)
    archive = tmp_path / "stdlib-shadow-verify.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr(
            _canonical_info(BUNDLE.BUNDLE_MANIFEST_NAME),
            BUNDLE._canonical_json_bytes(manifest),
        )
        handle.writestr(_canonical_info("payload/json.py"), source_bytes)

    with pytest.raises(ValueError, match="cannot shadow a standard-library module"):
        BUNDLE.verify_execution_bundle(archive)


def test_main_module_is_rejected_across_every_execution_boundary(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "main-source"
    _write(source_root / "__main__.py", "VALUE = 'unusable'\n")
    with pytest.raises(ValueError, match="cannot use __main__"):
        BUNDLE.build_execution_bundle(
            tmp_path / "main-build.zip",
            source_root=source_root,
            entry_module="__main__",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        )

    source_bytes = b"VALUE = 'unusable'\n"
    manifest, relative = _single_module_manifest("__main__", source_bytes)
    manifest_bytes = BUNDLE._canonical_json_bytes(manifest)
    archive = tmp_path / "main-verify.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr(_canonical_info(BUNDLE.BUNDLE_MANIFEST_NAME), manifest_bytes)
        handle.writestr(
            _canonical_info(f"{BUNDLE.BUNDLE_PAYLOAD_ROOT}/{relative}"),
            source_bytes,
        )
    with pytest.raises(ValueError, match="cannot use __main__"):
        BUNDLE.verify_execution_bundle(archive)

    stage = tmp_path / "main-stage"
    (stage / BUNDLE.BUNDLE_PAYLOAD_ROOT).mkdir(parents=True)
    (stage / BUNDLE.BUNDLE_MANIFEST_NAME).write_bytes(manifest_bytes)
    (stage / BUNDLE.BUNDLE_PAYLOAD_ROOT / relative).write_bytes(source_bytes)
    with pytest.raises(ValueError, match="cannot use __main__"):
        BUNDLE.verify_staged_execution_tree(stage)

    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=stage,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=str(manifest["manifest_sha256"]),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "cannot use __main__" in completed.stderr


@pytest.mark.parametrize(
    "destination",
    [
        "runtime/evil.py",
        "runtime/evil.pyc",
        "runtime/evil.pyo",
        "runtime/__pycache__/evil.pyc",
        "runtime/injected.pth",
        "runtime/sitecustomize.py",
        "runtime/extension.pyd",
        "runtime/CON.txt",
        "runtime/trailing.",
    ],
)
def test_runtime_assets_cannot_add_unregistered_python_code(
    tmp_path: Path,
    destination: str,
) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    with pytest.raises(ValueError):
        BUNDLE.build_execution_bundle(
            tmp_path / "forbidden-asset.zip",
            source_root=source_root,
            entry_module="demo.entry",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            runtime_assets={destination: asset},
        )


@pytest.mark.parametrize(
    "destinations",
    [
        ("runtime/asset.json", "runtime/ASSET.json"),
        ("runtime/tree", "runtime/tree/leaf"),
    ],
)
def test_builder_rejects_portability_ambiguous_payload_layouts(
    tmp_path: Path,
    destinations: tuple[str, str],
) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    resolved_assets = {destination: asset for destination in destinations}

    with pytest.raises(ValueError, match="(?:case-colliding|file/directory)"):
        BUNDLE.build_execution_bundle(
            tmp_path / "ambiguous-layout.zip",
            source_root=source_root,
            entry_module="demo.entry",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            runtime_assets=resolved_assets,
        )


@pytest.mark.parametrize(
    "destination",
    [
        "runtime/" + "/".join(["a"] * BUNDLE.MAX_PATH_COMPONENTS),
        "runtime/" + ("a" * (BUNDLE.MAX_PATH_COMPONENT_CHARS + 1)),
        "runtime/" + "/".join(["a" * 100] * 6),
    ],
)
def test_builder_bounds_runtime_asset_path_allocation(
    tmp_path: Path,
    destination: str,
) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)

    with pytest.raises(ValueError, match="(?:length|depth|oversized)"):
        BUNDLE.build_execution_bundle(
            tmp_path / "oversized-path.zip",
            source_root=source_root,
            entry_module="demo.entry",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            runtime_assets={destination: asset},
        )


def test_snapshot_importer_uses_authenticated_bytes_after_stage_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    for name in (
        "demo",
        "demo.entry",
        "demo.helper",
        "demo.nested",
        "demo.nested.util",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    original_snapshot = BUNDLE._snapshot_module_sources
    helper = staged.payload_root / "demo" / "helper.py"

    def replace_after_verification(
        checked: BUNDLE.StagedExecutionBundle,
    ) -> tuple[dict[str, Path], dict[str, object]]:
        helper.write_text(
            "raise RuntimeError('mutable staged helper was executed')\n",
            encoding="utf-8",
        )
        return original_snapshot(checked)

    monkeypatch.setattr(BUNDLE, "_snapshot_module_sources", replace_after_verification)
    entry = BUNDLE.assert_staged_entry_module_origin(staged)

    assert "mutable staged helper" in helper.read_text(encoding="utf-8")
    assert entry.answer().startswith("trusted-snapshot-")


def test_inner_snapshot_importer_ignores_replacement_during_import(tmp_path: Path) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    _write(
        source_root / "demo" / "entry.py",
        "from pathlib import Path\n"
        "Path(__file__).with_name('helper.py').write_text(\n"
        "    \"raise RuntimeError('mutable helper was executed')\\n\",\n"
        "    encoding='utf-8',\n"
        ")\n"
        "from .helper import PREFIX\n"
        "RESULT = PREFIX\n",
    )
    built = BUNDLE.build_execution_bundle(
        tmp_path / "bundle.zip",
        source_root=source_root,
        entry_module="demo.entry",
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        runtime_assets={"runtime/settings.json": asset},
    )
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(
            built.archive_path,
            expected_archive_sha256=built.archive_sha256,
        ),
        tmp_path / "stage",
    )
    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=staged.manifest_sha256,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mutable helper was executed" in (
        staged.payload_root / "demo" / "helper.py"
    ).read_text(encoding="utf-8")
    assert json.loads(completed.stdout)["entry_module"] == "demo.entry"


def test_source_snapshot_rejects_an_ancestor_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, asset = _synthetic_source_tree(tmp_path)
    replacement = tmp_path / "replacement-source"
    _write(replacement / "demo" / "__init__.py", "\n")
    _write(replacement / "demo" / "entry.py", "VALUE = 'external'\n")
    original_root = tmp_path / "original-source"
    original_reader = BUNDLE._read_anchored_relative_bytes
    swapped = False

    def swap_then_read(
        root: object,
        relative: str,
        *,
        label: str,
        maximum_bytes: int | None = None,
    ) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            source_root.rename(original_root)
            replacement.rename(source_root)
        return original_reader(
            root,
            relative,
            label=label,
            maximum_bytes=maximum_bytes,
        )

    monkeypatch.setattr(BUNDLE, "_read_anchored_relative_bytes", swap_then_read)
    with pytest.raises(ValueError, match="component changed|cannot be opened safely"):
        BUNDLE.build_execution_bundle(
            tmp_path / "swapped-source.zip",
            source_root=source_root,
            entry_module="demo.entry",
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            runtime_assets={"runtime/settings.json": asset},
        )


def test_stable_reader_rejects_a_synthetic_ancestor_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "archive-parent"
    parent.mkdir()
    archive = parent / "synthetic.zip"
    archive.write_bytes(b"not opened")
    original_reject = BUNDLE._reject_link_or_reparse

    def reject_synthetic_ancestor(path: Path, metadata: object, *, label: str) -> None:
        if path == parent:
            raise ValueError("synthetic ancestor reparse")
        original_reject(path, metadata, label=label)

    monkeypatch.setattr(BUNDLE, "_reject_link_or_reparse", reject_synthetic_ancestor)
    with pytest.raises(ValueError, match="synthetic ancestor reparse"):
        BUNDLE.verify_execution_bundle(archive)


@pytest.mark.parametrize(
    ("files", "directories", "message"),
    [
        (
            {"bundle_manifest.json", "payload/demo/entry.py", "payload/DEMO/helper.py"},
            {"payload", "payload/demo", "payload/DEMO"},
            "case-colliding",
        ),
        (
            {"bundle_manifest.json", "payload/tree"},
            {"payload", "payload/tree/child"},
            "file/directory",
        ),
    ],
)
def test_stage_layout_rejects_portability_collisions(
    files: set[str],
    directories: set[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BUNDLE._validate_stage_layout(files, directories)


def test_inner_rejects_an_untrusted_bundle_manifest_digest(tmp_path: Path) -> None:
    _, archive, _ = _build(tmp_path)
    staged = BUNDLE.extract_execution_bundle(
        BUNDLE.verify_execution_bundle(archive),
        tmp_path / "stage",
    )
    completed = subprocess.run(
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=staged.root,
            runner_environment_manifest=_runner_environment_manifest(tmp_path),
            runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
            bundle_manifest_sha256=hashlib.sha256(b"wrong manifest\n").hexdigest(),
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "bundle manifest SHA-256 differs from expected" in completed.stderr


def test_outer_coordinator_boundary_is_documented_but_not_wired(tmp_path: Path) -> None:
    contract = BUNDLE.outer_coordinator_contract()
    assert contract["implemented"] is False
    assert contract["evaluator_wired"] is False
    assert contract["preverification_boundary"]["stdlib_only"] is True
    assert contract["external_attestation"]["required"] is True
    assert contract["external_attestation"]["implemented"] is False
    assert contract["external_attestation"]["caller_selected_authority_allowed"] is False
    assert contract["external_attestation"]["missing_attestation_allows_execution"] is False
    assert contract["runner_environment"]["required"] is True
    assert contract["runner_environment"]["version_only_is_insufficient"] is True
    assert contract["runner_environment"]["ambient_site_fallback_allowed"] is False
    assert "the isolated inner assertion entrypoint bytes" in contract[
        "runner_environment"
    ]["must_bind"]
    assert contract["inner_boundary"]["requires_no_site_mode"] is True
    assert contract["inner_boundary"]["allows_ambient_python_dependencies"] is False
    assert contract["required_sequence"][3].startswith("verify the archive")
    runner_manifest = _runner_environment_manifest(tmp_path)
    command = BUNDLE.isolated_inner_command(
        python_executable=sys.executable,
        inner_entrypoint=INNER_SCRIPT,
        stage_root=tmp_path / "future-stage",
        runner_environment_manifest=runner_manifest,
        runner_environment_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        bundle_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
    )
    assert command[1:4] == ["-I", "-S", "-B"]
    assert "PYTHONPATH" not in command
    assert command[-8:] == [
        "--stage-root",
        str(tmp_path / "future-stage"),
        "--runner-environment-manifest",
        str(runner_manifest),
        "--expected-runner-environment-manifest-sha256",
        RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        "--expected-bundle-manifest-sha256",
        RUNNER_ENVIRONMENT_MANIFEST_SHA256,
    ]
    with pytest.raises(ValueError, match="runner environment manifest SHA-256"):
        BUNDLE.isolated_inner_command(
            python_executable=sys.executable,
            inner_entrypoint=INNER_SCRIPT,
            stage_root=tmp_path / "future-stage",
            runner_environment_manifest=runner_manifest,
            runner_environment_manifest_sha256="not-a-digest",
            bundle_manifest_sha256=RUNNER_ENVIRONMENT_MANIFEST_SHA256,
        )
