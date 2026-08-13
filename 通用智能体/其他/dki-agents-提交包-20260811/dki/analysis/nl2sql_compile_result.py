"""Neutral internal outcomes for deterministic SQL compilation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CompileDisposition(str, Enum):
    """Whether a compiler produced SQL, did not apply, or owns an abstention."""

    SQL = "sql"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class CompileResult:
    """A compiler outcome whose disposition must be inspected explicitly.

    ``fallback_fenced`` marks SQL from an exclusive owner.  It keeps execution
    failure from broadening that owner's request to an LLM fallback without
    introducing another compilation disposition.

    ``contract_self_validated`` is reserved for high-specificity compilers
    that parse and render their complete result shape themselves.  The root
    may use that provenance to recover from a false negative in the generic
    answer-shape extractor, but it must still apply the read-only AST gate and
    every row-qualifier obligation.  Ordinary and mocked candidates default to
    ``False`` and therefore cannot bypass the generic contract.
    """

    disposition: CompileDisposition
    sql: Optional[str] = None
    fallback_fenced: bool = False
    contract_self_validated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, CompileDisposition):
            raise TypeError("disposition must be a CompileDisposition")
        if not isinstance(self.fallback_fenced, bool):
            raise TypeError("fallback_fenced must be a bool")
        if not isinstance(self.contract_self_validated, bool):
            raise TypeError("contract_self_validated must be a bool")
        if self.disposition is CompileDisposition.SQL:
            if not isinstance(self.sql, str) or not self.sql.strip():
                raise ValueError("SQL disposition requires nonempty SQL")
        elif self.sql is not None:
            raise ValueError("non-SQL dispositions must not include SQL")
        elif self.fallback_fenced:
            raise ValueError("only SQL dispositions may fence fallback")
        elif self.contract_self_validated:
            raise ValueError("only SQL dispositions may self-validate a contract")

    def __bool__(self) -> bool:
        raise TypeError("inspect CompileResult.disposition explicitly")
