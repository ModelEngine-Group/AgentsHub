"""Local-first extraction building blocks.

The package contains the fast-path baseline, confidence calibration and entity
linking used by the neuro-symbolic cascade.  Heavy neural models are optional;
the deterministic lexicon path remains runnable on a CPU-only machine.
"""

from medigraph.extraction.calibration import TemperatureCalibrator, expected_calibration_error
from medigraph.extraction.entity_linker import EntityLinker
from medigraph.extraction.fast_path import FastSpanRelationExtractor

__all__ = [
    "EntityLinker",
    "FastSpanRelationExtractor",
    "TemperatureCalibrator",
    "expected_calibration_error",
]
