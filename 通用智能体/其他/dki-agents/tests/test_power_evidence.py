import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sample_npu_power import integrate_joules, power_statistics


def test_trapezoid_energy_and_idle_subtraction():
    result = integrate_joules([(0.0, 100.0), (1.0, 200.0), (2.0, 100.0)], idle_watts=50.0)
    assert result["joules"] == 300.0
    assert result["idle_subtracted_joules"] == 200.0
    assert result["covered_s"] == 2.0


def test_marker_window_filters_samples():
    result = integrate_joules([(0.0, 10.0), (1.0, 100.0), (2.0, 100.0), (3.0, 10.0)], 1.0, 2.0)
    assert result["joules"] == 100.0
    assert result["samples"] == 2


def test_insufficient_samples_are_explicit():
    result = integrate_joules([(1.0, 100.0)], 0.0, 2.0)
    assert result == {"joules": 0.0, "idle_subtracted_joules": 0.0, "covered_s": 0.0, "samples": 1}


def test_power_statistics_reports_selected_device_distribution():
    result = power_statistics([100.0, 300.0, 200.0, 400.0])

    assert result == {
        "minimum": 100.0,
        "mean": 250.0,
        "median": 250.0,
        "p95": 400.0,
        "maximum": 400.0,
    }
