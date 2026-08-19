"""DataMate-style mappers for task 1 medical data processing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


class NormalizeMedicalRecordMapper:
    """Normalize configured text fields in a single DataMate sample."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)

    def execute(self, sample: Mapping[str, object]) -> dict[str, object]:
        result = dict(sample)
        for column in self.columns:
            if column in result:
                result[column] = re.sub(r"\s+", " ", str(result[column])).strip()
        return result


class ExtractMedicalFieldsMapper:
    """Keep only configured fields from a single DataMate sample."""

    def __init__(self, fields: Sequence[str]) -> None:
        self.fields = list(fields)

    def execute(self, sample: Mapping[str, object]) -> dict[str, object]:
        missing = [field for field in self.fields if field not in sample]
        if missing:
            raise KeyError(f"DataMate sample missing fields: {missing}")
        return {field: sample[field] for field in self.fields}
