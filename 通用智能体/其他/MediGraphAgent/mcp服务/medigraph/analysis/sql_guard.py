"""AST-level read-only validation for generated SQL.

The previous check was a regex blacklist over the query string (``insert``,
``update``, ``drop`` ...). Blacklists enumerate what is forbidden, so they fail
open: anything the author did not think of gets through, and string-level matching
is confused by comments, casing, whitespace and nesting. It also produced false
*positives* -- a legitimate column named ``updated_at`` contains "update".

This module parses the statement instead and decides structurally:

1. exactly one statement (a trailing ``;`` is fine, a second statement is not);
2. the root must be a projection -- ``SELECT``/``WITH``, or a set operation over
   them (``UNION``/``INTERSECT``/``EXCEPT``);
3. no node anywhere in the tree may be DDL/DML or a session-level command;
4. no calls to SQLite functions that touch the filesystem or load code.

Rule 3 is the important one: it holds for arbitrarily nested constructs, so a
``DELETE`` hidden inside a CTE or a scalar subquery is rejected on structure rather
than on spelling.

This is the outer layer of a defence-in-depth pair. The executor additionally opens
the database read-only, sets ``PRAGMA query_only`` and installs a SQLite authorizer
that denies write actions -- so a parser gap still does not become a write.
"""
from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

DIALECT = "sqlite"

#: Statement/clause types that must never appear anywhere in the tree.
FORBIDDEN_NODES: tuple[type, ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Attach,
    exp.Detach,
    exp.Pragma,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    # `Command` is sqlglot's catch-all for statements it does not model (VACUUM,
    # REINDEX, ANALYZE ...). Unmodelled means unvalidatable, so refuse it.
    exp.Command,
)

#: Statement roots that are acceptable: a projection or a set operation over them.
ALLOWED_ROOTS: tuple[type, ...] = (exp.Select, exp.Union, exp.Intersect, exp.Except)

#: SQLite functions that read/write files or load native code.
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
)


def ensure_read_only(sql: str) -> tuple[bool, str]:
    """Return ``(ok, reason)``. `reason` is empty when the statement is accepted."""
    if not sql or not sql.strip():
        return False, "empty statement"

    try:
        statements = [statement for statement in sqlglot.parse(sql, dialect=DIALECT) if statement]
    except Exception as exc:  # noqa: BLE001 - sqlglot raises several parse error types
        return False, f"unparseable SQL: {exc}"

    if not statements:
        return False, "no statement parsed"
    if len(statements) > 1:
        kinds = ", ".join(type(statement).__name__ for statement in statements)
        return False, f"multiple statements are not allowed ({kinds})"

    root = statements[0]
    if not isinstance(root, ALLOWED_ROOTS):
        return False, f"statement root must be a SELECT/WITH, got {type(root).__name__}"

    for node in root.walk():
        # sqlglot versions differ on whether walk() yields nodes or (node, ...) tuples.
        current = node[0] if isinstance(node, tuple) else node
        if isinstance(current, FORBIDDEN_NODES):
            return False, f"forbidden construct: {type(current).__name__}"
        if isinstance(current, exp.Anonymous):
            name = str(current.this or "").lower()
            if name in FORBIDDEN_FUNCTIONS:
                return False, f"forbidden function: {name}"

    return True, ""


def is_read_only(sql: str) -> bool:
    """Boolean form of `ensure_read_only`."""
    return ensure_read_only(sql)[0]


def transpile(sql: str, write: str) -> str:
    """Re-render a validated statement for another dialect.

    Used to run the same generated query against SQLite and PostgreSQL without
    string-patching quoting or function names by hand.
    """
    return sqlglot.transpile(sql, read=DIALECT, write=write)[0]
