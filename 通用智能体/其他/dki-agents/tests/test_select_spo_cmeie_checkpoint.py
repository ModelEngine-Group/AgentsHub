from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from dki.nlp.cmeie_selection_protocol import (
    load_and_validate_selection_receipt,
)
from tests.test_cmeie_selection_protocol import make_run_candidate


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_spo_cmeie_checkpoint.py"


def _command(
    fixture,
    *,
    candidate: Path | None = None,
    output: Path | None = None,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--run-root",
        str(fixture.root),
        "--candidate",
        str(candidate or fixture.candidate),
        "--output",
        str(output or fixture.selection_path),
    ]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_cli_publishes_only_canonical_schema_v3_selection_create_once(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    command = _command(fixture)
    first = _run(command)

    assert first.returncode == 0, first.stderr
    payload = json.loads(fixture.selection_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["selected_epoch"] == 3
    assert payload["candidate_directory"] == str(fixture.candidate)
    assert payload["run_root"] == str(fixture.root)
    assert payload["rid"] == fixture.binding.rid
    assert payload["training_receipt_sha256"]
    assert payload["semantic_contract_sha256"]
    assert (
        load_and_validate_selection_receipt(
            fixture.selection_path,
            run_binding=fixture.binding,
        )
        == payload
    )

    second = _run(command)
    assert second.returncode != 0
    assert "output exists" in second.stderr


def test_cli_requires_run_root_argument(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    command = _command(fixture)
    run_root_index = command.index("--run-root")
    del command[run_root_index : run_root_index + 2]

    result = _run(command)
    assert result.returncode != 0
    assert "--run-root" in result.stderr
    assert "required" in result.stderr


def test_cli_rejects_noncanonical_output_path(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    wrong = tmp_path / "selection.json"

    result = _run(_command(fixture, output=wrong))
    assert result.returncode != 0
    assert "canonical seed selection path" in result.stderr
    assert not wrong.exists()
    assert not fixture.selection_path.exists()


def test_cli_rejects_copied_candidate_path_replay(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    copied = tmp_path / "copied-candidate"
    shutil.copytree(fixture.candidate, copied)

    result = _run(_command(fixture, candidate=copied))
    assert result.returncode != 0
    assert "canonical seed candidate path" in result.stderr
    assert not fixture.selection_path.exists()


def test_cli_rejects_seed_path_and_training_receipt_disagreement(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    seed_43 = fixture.root / "models" / "cmeie" / "seed-43"
    shutil.copytree(fixture.candidate, seed_43)

    result = _run(_command(fixture, candidate=seed_43))
    assert result.returncode != 0
    assert "canonical seed selection path" in result.stderr
    assert not fixture.selection_path.exists()


def test_cli_rejects_training_receipt_before_schema_v7(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    receipt = fixture.receipt()
    receipt["schema_version"] = 6
    fixture.write_receipt(receipt)

    result = _run(_command(fixture))
    assert result.returncode != 0
    assert "schema v7" in result.stderr
    assert not fixture.selection_path.exists()


def test_cli_rejects_training_receipt_before_schema_v6(tmp_path: Path):
    """Preserve the v6-era node while exercising the stronger v7 floor."""

    test_cli_rejects_training_receipt_before_schema_v7(tmp_path)


def test_cli_names_missing_epoch_contract_field(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    receipt = fixture.receipt()
    del receipt["history"][0]["extraction_contract_sha256"]
    fixture.write_receipt(receipt)

    result = _run(_command(fixture))
    assert result.returncode != 0
    assert (
        "history lacks attested epoch fields: extraction_contract_sha256"
        in result.stderr
    )
    assert not fixture.selection_path.exists()
