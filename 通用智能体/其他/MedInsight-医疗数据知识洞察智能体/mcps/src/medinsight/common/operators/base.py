from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class Operator(Protocol):
    spec: OperatorSpec

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

