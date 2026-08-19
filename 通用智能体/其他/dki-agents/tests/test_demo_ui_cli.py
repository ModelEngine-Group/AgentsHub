from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path

import pytest

from dki import cli


TREE_A = "a" * 64
TREE_B = "b" * 64
RID = "20260807T120000Z-g1234567-t000000000000"
ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"


def _write_manifest(
    path: Path,
    *,
    rid: str = RID,
    project_root: Path | None = None,
) -> tuple[Path, str]:
    root = project_root or path.parent
    files = {
        "dki/cli.py": b"# sealed cli fixture\n",
        "dki/demo/service.py": b"# sealed service fixture\n",
    }
    entries = []
    for relative, content in sorted(files.items()):
        target = root.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append({
            "path": relative,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    tree = digest.hexdigest()
    payload = {
        "schema_version": 1,
        "rid": rid,
        "git_head": "1234567" + "0" * 33,
        "algorithm": ALGORITHM,
        "included_tree_sha256": tree,
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, tree


def _args(**overrides):
    values = {
        "host": "127.0.0.1",
        "port": 7860,
        "rid": None,
        "included_tree_manifest": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _fake_device_snapshot(path: Path, *, streamlit_runtime: dict) -> Path:
    path.write_text(
        json.dumps({
            "schema_version": 2,
            "kind": "dki_demo_device_snapshot",
            "requested_logical_device": "npu:0",
            "detected_available": False,
            "detected_device_name": None,
            "environment_label": None,
            "detection_sources": [],
            "streamlit_runtime": streamlit_runtime,
        }),
        encoding="utf-8",
    )
    return path.resolve()


def _bind_manifest_environment(
    monkeypatch: pytest.MonkeyPatch,
    manifest: Path,
    tree: str,
) -> None:
    monkeypatch.setenv("DKI_INCLUDED_TREE_MANIFEST", str(manifest.resolve()))
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", tree)
    monkeypatch.setenv(
        "DKI_INCLUDED_TREE_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )


def _clear_source_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DKI_INCLUDED_TREE_SHA256",
        "DKI_INCLUDED_TREE_MANIFEST",
        "DKI_INCLUDED_TREE_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)


def _fake_streamlit_install(demo_root: Path):
    target = demo_root / ".demo_deps"
    target.mkdir(parents=True)
    module = target / "streamlit" / "__init__.py"
    dist = target / "streamlit-1.54.0.dist-info"
    module.parent.mkdir(parents=True)
    dist.mkdir()
    module.write_text("__version__='1.54.0'\n", encoding="utf-8")
    receipt = demo_root / "streamlit_runtime.json"
    runtime = {
        "version": "1.54.0",
        "module_path": str(module.resolve()),
        "distribution_path": str(dist.resolve()),
        "record_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
    }
    receipt.write_text(json.dumps(runtime), encoding="utf-8")
    return target.resolve(), receipt.resolve(), runtime, cli._streamlit_environment(target)


def test_demo_ui_source_tree_resolution_precedence_starts_with_valid_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_A.upper())
    args = _args(included_tree_manifest=tmp_path / "missing.json", rid="missing-rid")

    with pytest.raises(FileNotFoundError):
        cli._resolve_demo_source_tree(args)


def test_demo_ui_naked_environment_hash_is_never_source_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_A)
    monkeypatch.delenv("DKI_INCLUDED_TREE_MANIFEST", raising=False)

    with pytest.raises(RuntimeError, match="not source authority"):
        cli._resolve_demo_source_tree(_args())


def test_demo_ui_invalid_environment_does_not_silently_fall_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _tree = _write_manifest(tmp_path / "included_tree_manifest.json")
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", "unbound")

    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        cli._resolve_demo_source_tree(_args(included_tree_manifest=manifest))


def test_demo_ui_explicit_manifest_is_validated_and_precedes_rid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_environment(monkeypatch)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    explicit, expected = _write_manifest(tmp_path / "explicit.json")

    assert cli._resolve_demo_source_tree(
        _args(included_tree_manifest=explicit, rid="missing-rid")
    ) == expected


def test_demo_ui_rid_manifest_is_discovered_and_rid_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_environment(monkeypatch)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    manifest = tmp_path / "outputs" / "remote_runs" / RID / "included_tree_manifest.json"
    _path, expected = _write_manifest(manifest, rid=RID, project_root=tmp_path)

    assert cli._resolve_demo_source_tree(_args(rid=RID)) == expected

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rid"] = "20260807T120001Z-g1234567-t000000000000"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match --rid"):
        cli._resolve_demo_source_tree(_args(rid=RID))


def test_demo_ui_bare_command_discovers_only_a_live_matching_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_environment(monkeypatch)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    manifest = (
        tmp_path / "outputs" / "remote_runs" / RID / "included_tree_manifest.json"
    )
    _path, expected = _write_manifest(manifest, rid=RID, project_root=tmp_path)

    binding = cli._resolve_demo_source_binding(_args())
    assert binding["included_tree_sha256"] == expected
    assert binding["manifest_path"] == str(manifest.resolve())

    (tmp_path / "dki" / "cli.py").write_text(
        "# changed after discovery\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="requires a verified included-tree manifest"):
        cli._resolve_demo_source_binding(_args())


def test_demo_ui_manifest_recomputes_aggregate_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_environment(monkeypatch)
    manifest, _expected = _write_manifest(tmp_path / "included_tree_manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["included_tree_sha256"] = TREE_A
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate hash"):
        cli._resolve_demo_source_tree(_args(included_tree_manifest=manifest))


def test_demo_ui_manifest_rejects_live_source_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_environment(monkeypatch)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    manifest, _expected = _write_manifest(
        tmp_path / "included_tree_manifest.json",
        project_root=tmp_path,
    )
    (tmp_path / "dki" / "cli.py").write_text("# changed after sealing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from manifest"):
        cli._resolve_demo_source_tree(_args(included_tree_manifest=manifest))


def test_demo_ui_final_status_requires_one_explicit_consistent_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_environment(monkeypatch)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    final_status = tmp_path / "FINAL_STATUS.json"
    final_status.write_text(
        json.dumps({"source": {"included_tree_sha256": TREE_B.upper()}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="verified included-tree manifest"):
        cli._resolve_demo_source_tree(_args())


def test_demo_ui_missing_source_binding_fails_before_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_environment(monkeypatch)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "outputs")
    rid = "unsealed-ui"

    with pytest.raises(RuntimeError, match="requires a verified included-tree manifest"):
        cli.cmd_demo_ui(_args(rid=rid))

    assert not (tmp_path / "outputs" / "demo_runs" / rid).exists()


def test_demo_ui_occupied_port_fails_before_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "outputs")
    manifest, tree = _write_manifest(
        tmp_path / "included_tree_manifest.json",
        project_root=tmp_path,
    )
    _bind_manifest_environment(monkeypatch, manifest, tree)
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    port = occupied.getsockname()[1]
    try:
        with pytest.raises(OSError, match="occupied or unavailable"):
            cli.cmd_demo_ui(_args(rid="port-conflict-ui", port=port))
    finally:
        occupied.close()

    assert not (tmp_path / "outputs" / "demo_runs" / "port-conflict-ui").exists()


def test_demo_ui_passes_resolved_tree_to_streamlit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "outputs")
    manifest, tree = _write_manifest(
        tmp_path / "included_tree_manifest.json",
        project_root=tmp_path,
    )
    _bind_manifest_environment(monkeypatch, manifest, tree)
    monkeypatch.setattr(cli, "_preflight_demo_port", lambda host, port: None)
    monkeypatch.setattr(cli, "_capture_demo_device_snapshot", _fake_device_snapshot)
    monkeypatch.setattr(cli, "_install_and_verify_streamlit", _fake_streamlit_install)
    app = tmp_path / "dki" / "demo" / "app.py"
    app.parent.mkdir(parents=True, exist_ok=True)
    app.write_text("# test app\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_demo_ui(_args(rid="sealed-ui"))

    assert exc_info.value.code == 0
    assert captured["env"]["DKI_INCLUDED_TREE_SHA256"] == tree
    assert captured["env"]["DKI_INCLUDED_TREE_MANIFEST"] == str(manifest.resolve())
    assert captured["env"]["DKI_DEMO_SERVER_RID"] == "sealed-ui"
    assert captured["env"]["DKI_DEMO_DEVICE_SNAPSHOT"].endswith("device_snapshot.json")
    assert captured["env"]["PYTHONPATH"].split(cli.os.pathsep)[0].endswith(".demo_deps")
    assert (tmp_path / "outputs" / "demo_runs" / "sealed-ui").is_dir()


def test_demo_ui_wrong_global_streamlit_version_uses_rid_isolated_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_installed_streamlit_version", lambda: "1.53.0")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "pip" in command:
            target = Path(command[command.index("--target") + 1])
            module = target / "streamlit" / "__init__.py"
            dist = target / "streamlit-1.54.0.dist-info"
            module.parent.mkdir(parents=True)
            dist.mkdir()
            module.write_text("__version__='1.54.0'\n", encoding="utf-8")
            (dist / "RECORD").write_text("record\n", encoding="utf-8")
            (dist / "METADATA").write_text("metadata\n", encoding="utf-8")
            return argparse.Namespace(returncode=0)
        target = tmp_path / "run" / ".demo_deps"
        runtime = {
            "version": "1.54.0",
            "module_path": str((target / "streamlit" / "__init__.py").resolve()),
            "distribution_path": str((target / "streamlit-1.54.0.dist-info").resolve()),
            "record_sha256": hashlib.sha256(b"record\n").hexdigest(),
            "metadata_sha256": hashlib.sha256(b"metadata\n").hexdigest(),
        }
        return argparse.Namespace(
            returncode=0,
            stdout="DKI_STREAMLIT_PROBE=" + json.dumps(runtime) + "\n",
            stderr="",
        )

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    target, receipt, runtime, env = cli._install_and_verify_streamlit(tmp_path / "run")

    assert calls[0][0][-1] == "streamlit==1.54.0"
    assert "--target" in calls[0][0]
    assert calls[1][0][0:2] == [cli.sys.executable, "-c"]
    assert calls[1][1]["env"]["PYTHONPATH"].split(cli.os.pathsep)[0] == str(target)
    assert env["PYTHONPATH"].split(cli.os.pathsep)[0] == str(target)
    assert runtime["version"] == "1.54.0"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "verified"


@pytest.mark.parametrize(
    ("version", "module_in_target", "message"),
    [
        ("1.53.0", True, "version is not exactly"),
        ("1.54.0", False, "outside the RID-local target"),
    ],
)
def test_demo_ui_streamlit_probe_fails_closed_on_wrong_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    module_in_target: bool,
    message: str,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    def fake_run(command, **kwargs):
        target = tmp_path / "run" / ".demo_deps"
        if "pip" in command:
            local_module = target / "streamlit" / "__init__.py"
            local_dist = target / "streamlit-1.54.0.dist-info"
            local_module.parent.mkdir(parents=True)
            local_dist.mkdir()
            local_module.write_text("# local\n", encoding="utf-8")
            return argparse.Namespace(returncode=0)
        external = tmp_path / "global"
        external.mkdir(exist_ok=True)
        module = (
            target / "streamlit" / "__init__.py"
            if module_in_target else external / "streamlit.py"
        )
        if not module.is_file():
            module.write_text("# global\n", encoding="utf-8")
        runtime = {
            "version": version,
            "module_path": str(module.resolve()),
            "distribution_path": str(
                (target / "streamlit-1.54.0.dist-info").resolve()
                if module_in_target else external.resolve()
            ),
            "record_sha256": None,
            "metadata_sha256": None,
        }
        return argparse.Namespace(
            returncode=0,
            stdout="DKI_STREAMLIT_PROBE=" + json.dumps(runtime),
            stderr="",
        )

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=message):
        cli._install_and_verify_streamlit(tmp_path / "run")


def test_demo_device_snapshot_is_sanitised_and_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli,
        "_python_device_runtime",
        lambda: {
            "python": "3.11.6",
            "torch": "2.7.1",
            "torch_npu": "2.7.1.post6",
            "npu_available": True,
            "device_name": "Ascend910_9382",
            "error_type": None,
        },
    )
    streamlit = {
        "version": "1.54.0",
        "module_path": "RID/.demo_deps/streamlit/__init__.py",
        "distribution_path": "RID/.demo_deps/streamlit-1.54.0.dist-info",
        "record_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
    }
    path = cli._capture_demo_device_snapshot(
        tmp_path / "device_snapshot.json",
        streamlit_runtime=streamlit,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["requested_logical_device"] == "npu:0"
    assert payload["detected_available"] is True
    assert payload["detected_device_name"] == "Ascend910_9382"
    assert payload["environment_label"] == "910C 验收环境"
    assert payload["commands"]["npu_smi"]["binary_available"] is False
    assert payload["commands"]["python_runtime"]["device_name"] == "Ascend910_9382"
    assert payload["streamlit_runtime"] == streamlit
    assert payload["privacy"] == {
        "contains_environment": False,
        "contains_secrets": False,
    }
    with pytest.raises(FileExistsError, match="create-once"):
        cli._capture_demo_device_snapshot(path, streamlit_runtime=streamlit)


def test_demo_device_snapshot_does_not_claim_910c_when_detection_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli,
        "_python_device_runtime",
        lambda: {
            "python": "3.11.6", "torch": None, "torch_npu": None,
            "npu_available": False, "device_name": None, "error_type": "OSError",
        },
    )
    path = cli._capture_demo_device_snapshot(
        tmp_path / "device.json",
        streamlit_runtime={"version": "1.54.0"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["requested_logical_device"] == "npu:0"
    assert payload["detected_available"] is False
    assert payload["detected_device_name"] is None
    assert payload["environment_label"] is None
    assert payload["detection_sources"] == []


def test_demo_device_snapshot_hashes_truncates_and_redacts_npu_smi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "Ascend 910 detected\nAPI_KEY=must-not-leak\n" + "\n".join(
        f"row-{index}-" + "x" * 800 for index in range(100)
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/npu-smi")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout=stdout),
    )
    monkeypatch.setattr(
        cli,
        "_python_device_runtime",
        lambda: {
            "python": "3.11.6", "torch": None, "torch_npu": None,
            "npu_available": False, "device_name": None, "error_type": None,
        },
    )
    path = cli._capture_demo_device_snapshot(
        tmp_path / "device.json",
        streamlit_runtime={"version": "1.54.0"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    smi = payload["commands"]["npu_smi"]

    assert smi["stdout_sha256"] == hashlib.sha256(stdout.encode("utf-8")).hexdigest()
    assert len(smi["summary_lines"]) == 80
    assert max(map(len, smi["summary_lines"])) <= 500
    assert "must-not-leak" not in json.dumps(payload)
    assert "[redacted-sensitive-line]" in smi["summary_lines"]
    assert payload["detected_available"] is True
    assert payload["detection_sources"] == ["npu-smi"]


def test_demo_device_snapshot_npu_smi_failure_is_not_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/npu-smi")

    def fail(*args, **kwargs):
        raise OSError("driver unavailable")

    monkeypatch.setattr(cli.subprocess, "run", fail)
    monkeypatch.setattr(
        cli,
        "_python_device_runtime",
        lambda: {
            "python": "3.11.6", "torch": None, "torch_npu": None,
            "npu_available": False, "device_name": None, "error_type": "OSError",
        },
    )
    path = cli._capture_demo_device_snapshot(
        tmp_path / "device.json",
        streamlit_runtime={"version": "1.54.0"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["commands"]["npu_smi"]["error_type"] == "OSError"
    assert payload["detected_available"] is False
    assert payload["environment_label"] is None


def test_demo_ui_parser_keeps_basic_interface_and_adds_optional_manifest() -> None:
    manifest = Path("evidence/included_tree_manifest.json")
    args = cli.build_parser().parse_args([
        "demo-ui",
        "--host", "0.0.0.0",
        "--port", "7861",
        "--rid", RID,
        "--included-tree-manifest", str(manifest),
    ])
    assert args.host == "0.0.0.0"
    assert args.port == 7861
    assert args.rid == RID
    assert args.included_tree_manifest == manifest
