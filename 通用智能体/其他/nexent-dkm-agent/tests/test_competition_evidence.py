import argparse
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demos import collect_competition_evidence as evidence
from demos import task2_neo4j_live_smoke as neo4j_smoke


def test_neo4j_report_input_path_is_repository_relative():
    input_path = neo4j_smoke.ROOT / "data" / "samples" / "task2_medical_notes.txt"

    assert neo4j_smoke._portable_path(input_path) == (
        "data/samples/task2_medical_notes.txt"
    )


def _args(**overrides):
    values = {
        "datamate_url": "none",
        "nexent_url": "none",
        "neo4j_uri": "none",
        "neo4j_user": "neo4j",
        "neo4j_password": "nexent2024",
        "timeout": 3.0,
        "command_timeout": 180,
        "include_ruff": False,
        "include_pytest": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_evidence_collector_skips_neo4j_smoke_when_uri_is_none(monkeypatch, tmp_path):
    """--neo4j-uri none should skip live Neo4j commands, not fail the bundle."""
    commands = []

    def fake_run_command(
        name,
        command,
        log_dir,
        timeout,
        *,
        stdin_text=None,
        sensitive_values=(),
    ):
        commands.append(name)
        return {
            "name": name,
            "command": command,
            "returncode": 0,
            "duration_sec": 0,
            "log_path": str(log_dir / f"{name}.log"),
        }

    monkeypatch.setattr(evidence, "_run_command", fake_run_command)

    results = evidence._run_evidence_commands(
        _args(),
        {"generated": tmp_path / "generated", "logs": tmp_path / "logs"},
    )

    assert all(item["returncode"] == 0 for item in results)
    assert "task1_data_quality_benchmark" in commands
    assert "task2_relation_quality_benchmark" in commands
    assert "service_reachability_probe" in commands
    assert "task2_neo4j_live_smoke" not in commands
    assert "planner_comparison_demo" in commands
    assert "dkm_nexent_toolchain_demo" in commands
    assert "task2_oov_extraction_benchmark" in commands
    assert "planner_llm_evidence_demo" in commands

    service_command = next(
        item["command"] for item in results if item["name"] == "service_reachability_probe"
    )
    host_label_index = service_command.index("--host-label") + 1
    assert service_command[host_label_index] == "local_evidence_host"


def test_evidence_collector_runs_neo4j_smoke_when_uri_is_configured(monkeypatch, tmp_path):
    commands = []

    def fake_run_command(
        name,
        command,
        log_dir,
        timeout,
        *,
        stdin_text=None,
        sensitive_values=(),
    ):
        commands.append((name, command, stdin_text, sensitive_values))
        return {
            "name": name,
            "command": command,
            "returncode": 0,
            "duration_sec": 0,
            "log_path": str(log_dir / f"{name}.log"),
        }

    monkeypatch.setattr(evidence, "_run_command", fake_run_command)

    evidence._run_evidence_commands(
        _args(neo4j_uri="bolt://localhost:7687"),
        {"generated": tmp_path / "generated", "logs": tmp_path / "logs"},
    )

    task3_command = next(
        command for name, command, _, _ in commands if name == "task3_demo"
    )
    graph_file_index = task3_command.index("--graph-file") + 1
    assert task3_command[graph_file_index] == str(
        tmp_path / "generated" / "task2" / "medical_kg.json"
    )

    neo4j_commands = [
        (command, stdin_text, sensitive_values)
        for name, command, stdin_text, sensitive_values in commands
        if name == "task2_neo4j_live_smoke"
    ]
    assert len(neo4j_commands) == 1
    command, stdin_text, sensitive_values = neo4j_commands[0]
    assert "--uri" in command
    assert "--password-stdin" in command
    assert "--skip-pipeline" in command
    assert "nexent2024" not in command
    assert stdin_text == "nexent2024\n"
    assert sensitive_values == ("nexent2024",)


def test_run_command_forces_utf8_child_output(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="高血压", stderr="")

    monkeypatch.setattr(evidence.subprocess, "run", fake_run)
    tmp_path.mkdir(exist_ok=True)

    evidence._run_command("demo", [sys.executable, "-c", "print('高血压')"], tmp_path, 30)

    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert "高血压" in (tmp_path / "demo.log").read_text(encoding="utf-8")


def test_run_command_redacts_sensitive_values_from_results_and_logs(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(evidence.subprocess, "run", fake_run)
    secret = "private-neo4j-password"

    result = evidence._run_command(
        "neo4j",
        ["neo4j-smoke", "--password", secret],
        tmp_path,
        30,
        stdin_text=f"{secret}\n",
        sensitive_values=(secret,),
    )

    log = (tmp_path / "neo4j.log").read_text(encoding="utf-8")
    assert captured["command"][-1] == secret
    assert captured["input"] == f"{secret}\n"
    assert secret not in result["command"]
    assert secret not in log
    assert "<redacted>" in log


def test_neo4j_smoke_reads_password_from_stdin(monkeypatch):
    monkeypatch.setattr(neo4j_smoke.sys, "stdin", io.StringIO("private-password\n"))

    password = neo4j_smoke._resolve_password(
        SimpleNamespace(password=None, password_file=None, password_stdin=True)
    )

    assert password == "private-password"


def test_neo4j_smoke_requires_explicit_password():
    with pytest.raises(ValueError, match="Neo4j password is required"):
        neo4j_smoke._resolve_password(
            SimpleNamespace(
                password=None,
                password_file=None,
                password_stdin=False,
            )
        )


def test_neo4j_smoke_reads_password_from_ignored_file(tmp_path):
    password_file = tmp_path / "neo4j.password"
    password_file.write_text("private-password\n", encoding="utf-8")

    password = neo4j_smoke._resolve_password(
        SimpleNamespace(
            password=None,
            password_file=str(password_file),
            password_stdin=False,
        )
    )

    assert password == "private-password"


def test_neo4j_compose_requires_external_auth_configuration():
    compose = (ROOT / "docker-compose.neo4j.yml").read_text(encoding="utf-8")

    assert "nexent2024" not in compose
    assert "${NEO4J_AUTH:?" in compose
