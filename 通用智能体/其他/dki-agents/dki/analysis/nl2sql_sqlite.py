"""Safe read-only SQLite connection helpers for NL2SQL development runs."""
from __future__ import annotations

import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
# ``Path.as_uri`` percent-encodes these characters, but SQLite URI decoding is
# not a suitable boundary for untrusted database names.  Development paths fail
# closed instead of relying on a parser-specific interpretation of escapes.
_URI_AMBIGUOUS_CHARACTERS = frozenset("?#%&=;")

# These are deliberately much lower than SQLite's permissive compile-time
# defaults.  NL2SQL statements are small analytical queries; accepting
# multi-megabyte SQL text, thousands of result columns, or hundreds of compound
# branches only increases denial-of-service exposure.
SQLITE_RUNTIME_LIMITS = {
    sqlite3.SQLITE_LIMIT_SQL_LENGTH: 64 * 1024,
    sqlite3.SQLITE_LIMIT_LENGTH: 8 * 1024 * 1024,
    sqlite3.SQLITE_LIMIT_COLUMN: 256,
    sqlite3.SQLITE_LIMIT_EXPR_DEPTH: 100,
    sqlite3.SQLITE_LIMIT_COMPOUND_SELECT: 16,
    sqlite3.SQLITE_LIMIT_FUNCTION_ARG: 32,
    sqlite3.SQLITE_LIMIT_ATTACHED: 0,
}

DEFAULT_VM_STEP_BUDGET = 1_000_000
DEFAULT_QUERY_TIMEOUT_SECONDS = 2.0
DEFAULT_PROGRESS_OPS = 1_000

_DENIED_AUTHORIZER_ACTION_NAMES = (
    "SQLITE_INSERT",
    "SQLITE_UPDATE",
    "SQLITE_DELETE",
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW",
    "SQLITE_CREATE_VTABLE",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_TRIGGER",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER",
    "SQLITE_DROP_VIEW",
    "SQLITE_DROP_VTABLE",
    "SQLITE_ALTER_TABLE",
    "SQLITE_REINDEX",
    "SQLITE_ANALYZE",
    "SQLITE_TRANSACTION",
    "SQLITE_SAVEPOINT",
    "SQLITE_ATTACH",
    "SQLITE_DETACH",
    "SQLITE_PRAGMA",
)
_DENIED_AUTHORIZER_ACTIONS = frozenset(
    getattr(sqlite3, name)
    for name in _DENIED_AUTHORIZER_ACTION_NAMES
    if hasattr(sqlite3, name)
)
_ALLOWED_DATABASE_NAMES = frozenset({"main", "temp"})

# SQLite's authorizer sees scalar, aggregate, and window functions through the
# same SQLITE_FUNCTION action.  Keep an explicit pure-function allowlist so an
# extension or newly registered function cannot silently add filesystem,
# process, or unbounded-allocation behavior.
_ALLOWED_SQLITE_FUNCTIONS = frozenset(
    {
        # Core scalar functions.
        "abs",
        "char",
        "coalesce",
        "concat",
        "concat_ws",
        "format",
        "glob",
        "hex",
        "if",
        "ifnull",
        "iif",
        "instr",
        "length",
        "like",
        "likelihood",
        "likely",
        "lower",
        "ltrim",
        "max",
        "min",
        "nullif",
        "octet_length",
        "printf",
        "quote",
        "replace",
        "round",
        "rtrim",
        "sign",
        "soundex",
        "substr",
        "substring",
        "trim",
        "typeof",
        "unhex",
        "unicode",
        "unlikely",
        "upper",
        # Aggregate functions.
        "avg",
        "count",
        "group_concat",
        "json_group_array",
        "json_group_object",
        "string_agg",
        "sum",
        "total",
        # Date and time functions.
        "date",
        "datetime",
        "julianday",
        "strftime",
        "time",
        "timediff",
        "unixepoch",
        # JSON scalar functions.
        "json",
        "json_array",
        "json_array_length",
        "json_error_position",
        "json_extract",
        "json_insert",
        "json_object",
        "json_patch",
        "json_pretty",
        "json_quote",
        "json_remove",
        "json_replace",
        "json_set",
        "json_type",
        "json_valid",
        # Optional built-in math functions.
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "ceiling",
        "cos",
        "cosh",
        "degrees",
        "exp",
        "floor",
        "ln",
        "log",
        "log10",
        "log2",
        "mod",
        "pi",
        "pow",
        "power",
        "radians",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
        # Built-in window functions.
        "cume_dist",
        "dense_rank",
        "first_value",
        "lag",
        "last_value",
        "lead",
        "nth_value",
        "ntile",
        "percent_rank",
        "rank",
        "row_number",
    }
)


def _uri_safe_database_path(path: Path) -> Path:
    """Return an absolute database path only when its URI spelling is unambiguous."""
    supplied = Path(path)
    if any(
        character in component
        for component in supplied.parts
        for character in _URI_AMBIGUOUS_CHARACTERS
    ):
        raise ValueError("SQLite database path contains URI-reserved characters")
    return supplied.resolve()


def sqlite_readonly_uri(path: Path) -> str:
    """Encode one filesystem path as a SQLite URI with only ``mode=ro`` set."""
    return f"{_uri_safe_database_path(path).as_uri()}?mode=ro"


def require_sqlite_sidecars_absent(path: Path) -> None:
    """Reject a SQLite database whose logical state may live in a sidecar."""
    database = _uri_safe_database_path(path)
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        sidecar = database.with_name(f"{database.name}{suffix}")
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError(
                "SQLite database has a prohibited WAL/SHM/journal sidecar: "
                f"{sidecar}"
            )


def _readonly_authorizer(
    action_code: int,
    parameter1: str | None,
    parameter2: str | None,
    database_name: str | None,
    trigger_or_view: str | None,
) -> int:
    """Allow only bounded, side-effect-free reads from ``main`` or ``temp``."""
    del trigger_or_view
    if action_code == sqlite3.SQLITE_PRAGMA:
        pragma_name = (parameter1 or "").casefold()
        return (
            sqlite3.SQLITE_OK
            if pragma_name in {"table_info", "table_xinfo"}
            else sqlite3.SQLITE_DENY
        )
    if action_code in _DENIED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_DENY

    if action_code == sqlite3.SQLITE_READ:
        if database_name is not None and database_name.casefold() not in _ALLOWED_DATABASE_NAMES:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    if action_code == sqlite3.SQLITE_FUNCTION:
        function_name = (parameter2 or parameter1 or "").casefold()
        if function_name not in _ALLOWED_SQLITE_FUNCTIONS:
            return sqlite3.SQLITE_DENY

    return sqlite3.SQLITE_OK


def _configure_readonly_connection(connection: sqlite3.Connection) -> None:
    """Install connection-level defenses before returning a database handle."""
    try:
        connection.enable_load_extension(False)
    except sqlite3.NotSupportedError:
        # A build without loadable-extension support is already fail-closed.
        pass

    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    for category, limit in SQLITE_RUNTIME_LIMITS.items():
        connection.setlimit(category, limit)
    connection.set_authorizer(_readonly_authorizer)


def connect_readonly(path: Path) -> sqlite3.Connection:
    """Open a hardened, read-only SQLite connection for NL2SQL evaluation."""
    require_sqlite_sidecars_absent(path)
    connection = sqlite3.connect(sqlite_readonly_uri(path), uri=True)
    try:
        _configure_readonly_connection(connection)
    except Exception:
        connection.close()
        raise
    return connection


@contextmanager
def bounded_execution(
    connection: sqlite3.Connection,
    *,
    max_vm_steps: int = DEFAULT_VM_STEP_BUDGET,
    timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    progress_ops: int = DEFAULT_PROGRESS_OPS,
) -> Iterator[None]:
    """Interrupt work that exceeds an approximate VM-step or wall-time budget.

    SQLite invokes a progress callback every ``progress_ops`` virtual-machine
    instructions, so the step budget is intentionally approximate to that
    granularity.  The caller must execute *and consume* the cursor inside the
    context; the handler is always removed when the context exits.
    """
    if isinstance(max_vm_steps, bool) or not isinstance(max_vm_steps, int):
        raise TypeError("max_vm_steps must be an integer")
    if isinstance(progress_ops, bool) or not isinstance(progress_ops, int):
        raise TypeError("progress_ops must be an integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    if max_vm_steps <= 0:
        raise ValueError("max_vm_steps must be positive")
    if progress_ops <= 0:
        raise ValueError("progress_ops must be positive")
    timeout_value = float(timeout_seconds)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout_seconds must be positive and finite")

    deadline = time.monotonic() + timeout_value
    vm_steps = 0

    def _progress() -> int:
        nonlocal vm_steps
        vm_steps += progress_ops
        return int(
            vm_steps >= max_vm_steps
            or time.monotonic() >= deadline
        )

    connection.set_progress_handler(_progress, progress_ops)
    try:
        yield
    finally:
        connection.set_progress_handler(None, 0)
