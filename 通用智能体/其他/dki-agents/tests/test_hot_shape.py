import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_ner_workload import select_lattice


def test_lattice_is_bounded_and_reports_overflow():
    report = select_lattice([10, 60, 61, 100, 200, 300], max_nodes=2)
    assert len(report["sequence_nodes"]) <= 2
    assert report["overflow"] == 1
    assert report["total"] == 6
