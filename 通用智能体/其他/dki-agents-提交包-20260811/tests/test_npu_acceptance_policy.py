import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bench_medical_ner_adapter_batch.py"
)
SPEC = importlib.util.spec_from_file_location("bench_medical_ner_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _entity(name, start):
    return {
        "name": name,
        "mention": name,
        "type": "Disease",
        "span": [start, start + len(name)],
    }


def test_entity_parity_reports_small_boundary_differences():
    reference = [[_entity("甲", 0)], [_entity("乙", 0)]]
    candidate = [[_entity("甲", 0)], [_entity("丙", 0)]]
    result = MODULE._entity_parity(reference, candidate)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["micro_f1"] == 0.5
