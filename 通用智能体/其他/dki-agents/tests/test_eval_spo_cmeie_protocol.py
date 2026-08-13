import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_spo_cmeie.py"
SPEC = importlib.util.spec_from_file_location("eval_spo_cmeie", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_eval_resolves_only_epoch_directory_or_its_checkpoint(tmp_path: Path):
    candidate = tmp_path / "seed-42"
    directory = candidate / "epoch-01"
    directory.mkdir(parents=True)
    checkpoint = directory / "gplinker.pt"
    checkpoint.write_bytes(b"weights")
    receipt = candidate / "training_receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")

    assert MODULE.resolve_checkpoint_bundle(directory) == (
        directory,
        checkpoint,
        receipt,
    )
    assert MODULE.resolve_checkpoint_bundle(checkpoint) == (
        directory,
        checkpoint,
        receipt,
    )
    assert MODULE.SUPPORTED_TRAINING_RECEIPT_SCHEMAS == frozenset({6})

    wrong = candidate / "checkpoint"
    wrong.mkdir()
    (wrong / "gplinker.pt").write_bytes(b"weights")
    with pytest.raises(SystemExit, match="epoch-NN"):
        MODULE.resolve_checkpoint_bundle(wrong)


def _required_args(split: str) -> list[str]:
    return [
        "--run-root",
        "run",
        "--split",
        split,
        "--split-manifest",
        "manifest",
        "--calibration-shard",
        "calibration",
        "--checkpoint",
        "checkpoint",
        "--output",
        "output",
        "--search-declared-grid",
    ]


@pytest.mark.parametrize(
    "removed_role",
    ["outer", "official-dev-confirmation"],
)
def test_legacy_cli_explicitly_rejects_non_calibration_roles(removed_role):
    with pytest.raises(SystemExit):
        MODULE._parser().parse_args(_required_args(removed_role))


def test_legacy_cli_has_no_full_source_or_frozen_dataset_arguments():
    parser = MODULE._parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--train" not in option_strings
    assert "--selection-receipt" not in option_strings
    assert "--outer-gate-receipt" not in option_strings
    assert "--official-dev" not in option_strings
    assert "--calibration-shard" in option_strings
    parsed = parser.parse_args(_required_args("calibration"))
    assert parsed.split == "calibration"


def test_legacy_module_exposes_no_old_dataset_access_helpers():
    assert not hasattr(MODULE, "publish_outer_access_marker")
    assert not hasattr(MODULE, "load_official_dev_rows")
    assert not hasattr(MODULE, "validate_outer_gate_authorization")
