"""PostgreSQL analytics backend: pooled connections, real indexes, EXPLAIN.

SQLite (`relational.py`) stays the zero-config default -- CI, demos and the
deterministic evals never need a server. This module is the production-shaped
swap the SQLite docstring has always promised:

  * one shared `psycopg_pool.ConnectionPool` per process instead of a connection
    per query (connection setup is the dominant cost of naive PG usage);
  * composite indexes chosen for the query shapes the NL2SQL router actually
    emits (see `INDEX_DDL`), with `EXPLAIN ANALYZE` evidence rather than folklore;
  * read-only execution with a server-side `statement_timeout`, mirroring the
    SQLite executor's authorizer + progress-handler guard rails.

Both engines ingest *exactly* the same rows (`relational.generate_rows`), so a
query returning different results across engines indicates non-portable SQL, not
different data. `visit_date` stays TEXT on both engines for the same reason: the
generated SQL uses `substr(visit_date,1,7)`, which is portable, whereas DATE would
make the SQLite and PG type systems diverge.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from psycopg_pool import ConnectionPool

from config.settings import AnalyticsConfig, get_analytics_config
from medigraph.analysis.relational import generate_rows
from medigraph.graph.local_store import LocalGraphStore

PG_SCHEMA = """
DROP TABLE IF EXISTS prescriptions;
DROP TABLE IF EXISTS lab_tests;
DROP TABLE IF EXISTS patient_visits;
DROP TABLE IF EXISTS kg_entities;
DROP TABLE IF EXISTS kg_triples;
CREATE TABLE patient_visits (
    visit_id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    disease TEXT NOT NULL,
    department TEXT NOT NULL,
    visit_date TEXT NOT NULL,
    cost DOUBLE PRECISION NOT NULL
);
CREATE TABLE prescriptions (
    rx_id INTEGER PRIMARY KEY,
    visit_id INTEGER NOT NULL REFERENCES patient_visits(visit_id),
    drug TEXT NOT NULL,
    days INTEGER NOT NULL
);
CREATE TABLE lab_tests (
    test_id INTEGER PRIMARY KEY,
    visit_id INTEGER NOT NULL REFERENCES patient_visits(visit_id),
    test_name TEXT NOT NULL,
    abnormal INTEGER NOT NULL
);
CREATE TABLE kg_entities (name TEXT, type TEXT);
CREATE TABLE kg_triples (
    head TEXT, head_type TEXT, relation TEXT,
    tail TEXT, tail_type TEXT, confidence DOUBLE PRECISION, source TEXT
);
"""

#: Indexes matched to the router's actual query shapes:
#:   WHERE disease=? [AND age>?]        -> ix_visits_disease (age in the key keeps
#:                                         the age filter inside the index scan)
#:   GROUP BY department / disease      -> ix_visits_department
#:   substr(visit_date,1,7) trends      -> ix_visits_date
#:   WHERE drug=? / GROUP BY drug       -> ix_rx_drug
#:   lab JOIN + abnormal filter         -> ix_lab_visit
#:   kg_triples head/relation lookups   -> ix_kg_head_rel
INDEX_DDL: dict[str, str] = {
    "ix_visits_disease": "CREATE INDEX ix_visits_disease ON patient_visits (disease, age)",
    "ix_visits_department": "CREATE INDEX ix_visits_department ON patient_visits (department)",
    "ix_visits_date": "CREATE INDEX ix_visits_date ON patient_visits (visit_date)",
    "ix_rx_drug": "CREATE INDEX ix_rx_drug ON prescriptions (drug)",
    "ix_lab_visit": "CREATE INDEX ix_lab_visit ON lab_tests (visit_id, abnormal)",
    "ix_kg_head_rel": "CREATE INDEX ix_kg_head_rel ON kg_triples (head, relation)",
}

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool(config: AnalyticsConfig | None = None) -> ConnectionPool:
    """Process-wide connection pool (lazily created).

    `open=True` connects eagerly so a bad DSN fails at first use, not first query;
    `max_idle` trims connections the load spike no longer needs.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                config = config or get_analytics_config()
                _pool = ConnectionPool(
                    conninfo=config.pg_dsn,
                    min_size=config.pool_min,
                    max_size=config.pool_max,
                    max_idle=60.0,
                    open=True,
                    name="medigraph-analytics",
                )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_pg_db(
    store: LocalGraphStore,
    n_visits: int = 600,
    seed: int = 42,
    year: int = 2024,
    with_indexes: bool = True,
    config: AnalyticsConfig | None = None,
) -> dict:
    """(Re)create the analytics schema in PostgreSQL from the same row generator.

    Loads via ``COPY`` -- at 100k+ rows executemany is minutes, COPY is seconds.
    """
    rows = generate_rows(store, n_visits=n_visits, seed=seed, year=year)
    pool = get_pool(config)
    started = time.perf_counter()
    with pool.connection() as conn:
        conn.execute(PG_SCHEMA)
        with conn.cursor() as cur:
            with cur.copy("COPY patient_visits FROM STDIN") as copy:
                for row in rows["patient_visits"]:
                    copy.write_row(row)
            with cur.copy("COPY prescriptions FROM STDIN") as copy:
                for row in rows["prescriptions"]:
                    copy.write_row(row)
            with cur.copy("COPY lab_tests FROM STDIN") as copy:
                for row in rows["lab_tests"]:
                    copy.write_row(row)
            with cur.copy("COPY kg_entities FROM STDIN") as copy:
                for row in rows["kg_entities"]:
                    copy.write_row(row)
            with cur.copy("COPY kg_triples FROM STDIN") as copy:
                for row in rows["kg_triples"]:
                    copy.write_row(row)
        if with_indexes:
            for ddl in INDEX_DDL.values():
                conn.execute(ddl)
        conn.execute("ANALYZE")
        conn.commit()
    return {
        "backend": "postgres",
        "n_visits": n_visits,
        "n_prescriptions": len(rows["prescriptions"]),
        "n_lab_tests": len(rows["lab_tests"]),
        "n_kg_triples": len(rows["kg_triples"]),
        "indexes": list(INDEX_DDL) if with_indexes else [],
        "load_seconds": round(time.perf_counter() - started, 3),
    }


def drop_indexes(config: AnalyticsConfig | None = None) -> None:
    pool = get_pool(config)
    with pool.connection() as conn:
        for name in INDEX_DDL:
            conn.execute(f"DROP INDEX IF EXISTS {name}")
        conn.execute("ANALYZE")
        conn.commit()


def create_indexes(config: AnalyticsConfig | None = None) -> None:
    pool = get_pool(config)
    with pool.connection() as conn:
        for name, ddl in INDEX_DDL.items():
            conn.execute(f"DROP INDEX IF EXISTS {name}")
            conn.execute(ddl)
        conn.execute("ANALYZE")
        conn.commit()


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def execute_readonly_pg(
    sql: str,
    timeout_seconds: float = 5.0,
    max_rows: int = 5000,
    config: AnalyticsConfig | None = None,
) -> tuple[list[str], list[tuple], str]:
    """Run one read-only statement on the pool. Mirrors the SQLite executor's
    contract: ``(columns, rows, error)`` with the error string empty on success.

    Defence in depth, same shape as SQLite's (authorizer + query_only + progress
    handler): the AST guard has already vetted the SQL upstream, and here the
    transaction is forced read-only and time-boxed *server-side*, so even a guard
    gap cannot write or run forever.
    """
    pool = get_pool(config)
    try:
        with pool.connection() as conn:
            # `read_only` must be set while the connection is idle: psycopg begins a
            # transaction implicitly on the first execute, and session defaults like
            # `default_transaction_read_only` would only take effect for the *next*
            # transaction -- i.e. never for this statement. Setting the connection
            # property makes the very transaction that runs `sql` read-only, which
            # PostgreSQL then enforces server-side ("cannot execute ... in a
            # read-only transaction").
            conn.read_only = True
            try:
                conn.execute(f"SET statement_timeout = {int(timeout_seconds * 1000)}")
                with conn.cursor() as cur:
                    cur.execute(sql)  # type: ignore[arg-type]
                    rows = cur.fetchmany(max_rows + 1)
                    if len(rows) > max_rows:
                        return [], [], f"result exceeds safety limit ({max_rows} rows)"
                    columns = [d.name for d in cur.description] if cur.description else []
                conn.rollback()  # release the snapshot promptly
                return columns, rows, ""
            finally:
                # The pool reuses connections and build/index DDL shares this pool,
                # so the read-only characteristic must not leak past this call.
                conn.rollback()
                conn.read_only = False
    except Exception as exc:  # noqa: BLE001 - surfaced to the self-correction loop
        return [], [], str(exc)


def explain_analyze(sql: str, config: AnalyticsConfig | None = None) -> dict[str, Any]:
    """Structured EXPLAIN ANALYZE, for the index benchmark and for debugging."""
    pool = get_pool(config)
    with pool.connection() as conn:
        conn.execute("SET default_transaction_read_only = on")
        cur = conn.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")  # type: ignore[arg-type]
        plan = cur.fetchone()[0][0]
        conn.rollback()
    return {
        "node_type": plan["Plan"]["Node Type"],
        "execution_ms": plan["Execution Time"],
        "planning_ms": plan["Planning Time"],
        "plan": plan["Plan"],
    }


def ping(config: AnalyticsConfig | None = None) -> bool:
    try:
        columns, rows, error = execute_readonly_pg("SELECT 1", config=config)
        return not error and rows == [(1,)]
    except Exception:  # noqa: BLE001
        return False
