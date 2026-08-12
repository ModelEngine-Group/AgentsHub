"""Example DataMate operator package for task 1 migration.

This file mirrors DataMate's expected `process.py` shape. It is an example for
copying into DataMate `runtime/ops`; the demo tests use the lightweight mappers
in `adapters.datamate_task1_mappers` directly.
"""

from __future__ import annotations

from typing import Any

try:
    from datamate.core.base_op import Mapper
except ImportError:  # Keeps the demo importable without DataMate runtime.
    class Mapper:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

from adapters.datamate_task1_mappers import NormalizeMedicalRecordMapper


class NormalizeMedicalRecord(Mapper):
    """Normalize configured medical record fields in DataMate sample mode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        columns = str(kwargs.get("columns", "diagnosis,medicine,department"))
        self.mapper = NormalizeMedicalRecordMapper(
            columns=[column.strip() for column in columns.split(",") if column.strip()]
        )

    def execute(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self.mapper.execute(sample)
