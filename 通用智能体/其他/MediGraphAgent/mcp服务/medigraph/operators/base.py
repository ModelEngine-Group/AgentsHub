"""Unified operator contract -- the "Operator-as-MCP-Tool" abstraction layer.

A single operator implementation can be:
  1. called directly in-process by the agents (DataProc / KGGen),
  2. exposed as a Nexent MCP tool via `to_mcp_tool()` (see mcp_server/server.py),
  3. mirrored as a DataMate marketplace operator (see CCF/datamate_ops/).

Every operator declares an `OperatorMeta` (name, description, JSON schemas) so the
planning agent can discover and reason about it from `meta.description` alone --
no hard-coded operator list in the agent.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel, Field


class OperatorMeta(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str  # natural-language description shown to the LLM / MCP
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    backend: str = "cpu"  # "cpu" | "npu" (npu reserved for future acceleration)


class BaseOperator(ABC):
    """All operators implement run(inputs: dict) -> dict."""

    meta: OperatorMeta

    @abstractmethod
    def run(self, inputs: dict, **kwargs) -> dict:
        """Core execution. Subclasses implement this."""

    # -- timing wrapper used by the DAG executor / benchmarks ----------- #
    def run_timed(self, inputs: dict, **kwargs) -> tuple[dict, float]:
        start = time.time()
        out = self.run(inputs, **kwargs)
        return out, time.time() - start

    # -- Nexent MCP tool adapter --------------------------------------- #
    def to_mcp_tool(self) -> dict:
        """Return a descriptor used by mcp_server to register an MCP tool."""
        return {
            "name": self.meta.name,
            "description": self.meta.description,
            "input_schema": self.meta.input_schema,
            "func": self.run,
        }


# ---------------------------------------------------------------------- #
# Global operator registry. Operators register themselves at import time so the
# DataProc planning agent can enumerate the available toolbox.
# ---------------------------------------------------------------------- #
OP_REGISTRY: dict[str, BaseOperator] = {}


def register(operator: BaseOperator) -> BaseOperator:
    OP_REGISTRY[operator.meta.name] = operator
    return operator


def get_operator(name: str) -> BaseOperator:
    if name not in OP_REGISTRY:
        raise KeyError(f"operator '{name}' not registered. available: {list(OP_REGISTRY)}")
    return OP_REGISTRY[name]


def catalog() -> list[dict]:
    """Operator catalog (name + description + schema) for the planner prompt."""
    return [
        {
            "name": op.meta.name,
            "description": op.meta.description,
            "input_schema": op.meta.input_schema,
            "output_schema": op.meta.output_schema,
        }
        for op in OP_REGISTRY.values()
    ]


def load_default_operators(llm: Any | None = None) -> None:
    """Instantiate and register the built-in operators.

    Pass a shared LLMClient so the LLM-backed operators reuse one connection and
    aggregate latency stats. Import is local to avoid circular imports.
    """
    from medigraph.operators.text_clean import TextCleanOperator
    from medigraph.operators.chunker import ChunkerOperator
    from medigraph.operators.medical_ner import MedicalNEROperator
    from medigraph.operators.medical_re import MedicalREOperator
    from medigraph.operators.triple_validator import TripleValidatorOperator
    from medigraph.operators.document_loader import DocumentLoaderOperator
    from medigraph.operators.data_quality import DataQualityOperator
    from medigraph.operators.pii_redact import PIIRedactOperator
    from medigraph.operators.entity_linker import EntityLinkerOperator

    register(DocumentLoaderOperator())
    register(TextCleanOperator())
    register(DataQualityOperator())
    register(PIIRedactOperator())
    register(ChunkerOperator())
    register(MedicalNEROperator(llm=llm))
    register(EntityLinkerOperator())
    register(MedicalREOperator(llm=llm))
    register(TripleValidatorOperator(llm=llm))
