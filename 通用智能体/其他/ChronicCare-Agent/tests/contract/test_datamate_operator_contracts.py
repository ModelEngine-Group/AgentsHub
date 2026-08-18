from __future__ import annotations

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPERATORS_ROOT = PROJECT_ROOT / "integrations/datamate/operators"


def test_operator_catalog_contracts_and_source_directories_are_aligned() -> None:
    catalog = yaml.safe_load((PROJECT_ROOT / "integrations/datamate/operator_catalog.yml").read_text(encoding="utf-8"))
    contracts = json.loads((PROJECT_ROOT / "configs/operator_contracts/contracts.json").read_text(encoding="utf-8"))[
        "operators"
    ]
    catalog_names = {item["name"] for item in catalog["operators"]}
    catalog_names.update(item["name"] for item in catalog["npu_enhancement"]["operators"])
    source_names = {
        path.name for path in OPERATORS_ROOT.iterdir() if path.is_dir() and path.name.startswith("chronic_")
    }
    assert catalog_names == source_names
    assert catalog_names <= set(contracts)


def test_each_datamate_operator_has_complete_metadata_and_entrypoint() -> None:
    for operator_dir in sorted(OPERATORS_ROOT.glob("chronic_*")):
        metadata = yaml.safe_load((operator_dir / "metadata.yml").read_text(encoding="utf-8"))
        assert metadata["name"] == operator_dir.name
        assert metadata["raw_id"] == operator_dir.name
        assert (operator_dir / "process.py").is_file()
        assert (operator_dir / "README.md").is_file()


def test_npu_contracts_declare_hardware_or_fallback_precondition() -> None:
    contracts = json.loads((PROJECT_ROOT / "configs/operator_contracts/contracts.json").read_text(encoding="utf-8"))[
        "operators"
    ]
    for name in (
        "chronic_entity_extract_model_npu",
        "chronic_relation_extract_model_npu",
    ):
        contract = contracts[name]
        assert contract["resource_requirements"]["npu"] == 1
        assert "model_or_fallback" in contract["preconditions"]
