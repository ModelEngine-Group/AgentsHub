import importlib.util
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_cmeie_argument_overfit.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_cmeie_argument_overfit",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _point(threshold, f1, precision, recall, predicted=100, gold=100):
    return {
        "threshold": threshold,
        "metric": {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": 98,
            "fp": 2,
            "fn": 2,
        },
        "predicted_mentions": predicted,
        "gold_mentions": gold,
    }


def test_argument_overfit_selection_uses_declared_tie_break():
    selected = MODULE.select_operating_point(
        [
            _point(-0.5, 0.90, 0.95, 0.86),
            _point(0.0, 0.98, 0.98, 0.98),
            _point(0.5, 0.98, 0.98, 0.98),
        ]
    )
    assert selected["threshold"] == 0.5


def test_argument_overfit_gate_requires_quality_and_density():
    passed = MODULE.overfit_gate(
        _point(0.5, 0.98, 0.98, 0.98, predicted=100, gold=100)
    )
    assert passed["passed"] is True
    too_dense = MODULE.overfit_gate(
        _point(0.5, 0.98, 0.98, 0.98, predicted=201, gold=100)
    )
    assert too_dense["passed"] is False
    assert too_dense["checks"]["candidate_density"] is False


def test_argument_overfit_uses_run_binding_as_source_tree_verifier():
    """The executable must not depend on an API absent from evidence_tree."""

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "dki.evidence_tree"
        for alias in node.names
    }
    assert "validate_source_tree_manifest" not in imported
    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_run_binding(args.run_root)" in source
    assert "run_binding.source_manifest" in source
