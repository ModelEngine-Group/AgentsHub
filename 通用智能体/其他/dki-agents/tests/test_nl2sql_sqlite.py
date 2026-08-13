import ast
from pathlib import Path

import pytest

import dki.analysis.nl2sql_sqlite as sqlite_helpers


def test_readonly_uri_contains_only_mode_ro(tmp_path):
    database = tmp_path / "synthetic.sqlite"

    uri = sqlite_helpers.sqlite_readonly_uri(database)

    assert uri.startswith("file:")
    assert uri.endswith("?mode=ro")
    assert uri.count("?") == 1


@pytest.mark.parametrize(
    "name",
    [
        "database?.sqlite",
        "database#.sqlite",
        "database%.sqlite",
        "database&.sqlite",
        "database=.sqlite",
        "database;.sqlite",
    ],
)
def test_readonly_uri_rejects_reserved_path_characters(tmp_path, name):
    with pytest.raises(ValueError, match="URI-reserved characters"):
        sqlite_helpers.sqlite_readonly_uri(tmp_path / name)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecar_guard_rejects_synthetic_sidecars(tmp_path, suffix):
    database = tmp_path / "synthetic.sqlite"
    sidecar = database.with_name(f"{database.name}{suffix}")
    sidecar.write_bytes(b"synthetic sidecar bytes")

    with pytest.raises(ValueError, match="prohibited WAL/SHM/journal sidecar"):
        sqlite_helpers.require_sqlite_sidecars_absent(database)


def test_sidecar_guard_checks_the_resolved_database_path(tmp_path, monkeypatch):
    supplied = tmp_path / "supplied.sqlite"
    resolved = tmp_path / "resolved.sqlite"
    resolved.with_name(f"{resolved.name}-wal").write_bytes(b"synthetic sidecar")
    original_resolve = Path.resolve

    def resolve_supplied_path(path, *args, **kwargs):
        if Path(path) == supplied:
            return resolved
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_supplied_path)

    with pytest.raises(ValueError, match="prohibited WAL/SHM/journal sidecar"):
        sqlite_helpers.require_sqlite_sidecars_absent(supplied)


def test_connect_readonly_uses_uri_and_enforces_query_only(tmp_path, monkeypatch):
    database = tmp_path / "synthetic.sqlite"
    calls = []

    class FakeConnection:
        def enable_load_extension(self, enabled):
            calls.append(("enable_load_extension", enabled))

        def execute(self, statement):
            calls.append(("execute", statement))

        def setlimit(self, category, limit):
            calls.append(("setlimit", category, limit))

        def set_authorizer(self, callback):
            calls.append(("set_authorizer", callback))

        def close(self):
            calls.append(("close",))

    connection = FakeConnection()

    def fake_connect(database_uri, *, uri):
        calls.append(("connect", database_uri, uri))
        return connection

    monkeypatch.setattr(sqlite_helpers.sqlite3, "connect", fake_connect)

    result = sqlite_helpers.connect_readonly(database)

    assert result is connection
    assert calls == [
        ("connect", sqlite_helpers.sqlite_readonly_uri(database), True),
        ("enable_load_extension", False),
        ("execute", "PRAGMA query_only = ON"),
        ("execute", "PRAGMA trusted_schema = OFF"),
        *[
            ("setlimit", category, limit)
            for category, limit in sqlite_helpers.SQLITE_RUNTIME_LIMITS.items()
        ],
        ("set_authorizer", sqlite_helpers._readonly_authorizer),
    ]


def test_connect_readonly_closes_fake_connection_on_query_only_failure(
    tmp_path,
    monkeypatch,
):
    database = tmp_path / "synthetic.sqlite"
    calls = []

    class FailingConnection:
        def enable_load_extension(self, enabled):
            calls.append(("enable_load_extension", enabled))

        def execute(self, statement):
            calls.append(("execute", statement))
            raise RuntimeError("synthetic query-only failure")

        def close(self):
            calls.append(("close",))

    connection = FailingConnection()
    monkeypatch.setattr(
        sqlite_helpers.sqlite3,
        "connect",
        lambda *args, **kwargs: connection,
    )

    with pytest.raises(RuntimeError, match="synthetic query-only failure"):
        sqlite_helpers.connect_readonly(database)

    assert calls == [
        ("enable_load_extension", False),
        ("execute", "PRAGMA query_only = ON"),
        ("close",),
    ]


def _create_synthetic_database(path):
    connection = sqlite_helpers.sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE patients (patient_id TEXT PRIMARY KEY, age INTEGER)"
        )
        connection.executemany(
            "INSERT INTO patients(patient_id, age) VALUES (?, ?)",
            [("p1", 18), ("p2", 41), ("p3", 67)],
        )
        connection.commit()
    finally:
        connection.close()


def test_real_readonly_connection_keeps_normal_analytical_queries_working(tmp_path):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)

    connection = sqlite_helpers.connect_readonly(database)
    try:
        rows = connection.execute(
            "SELECT COUNT(*), ROUND(AVG(age), 2) FROM patients"
        ).fetchall()
    finally:
        connection.close()

    assert rows == [(3, 42.0)]


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO patients(patient_id, age) VALUES ('p4', 25)",
        "CREATE TEMP TABLE scratch(value INTEGER)",
        "ATTACH DATABASE ':memory:' AS other",
        "DETACH DATABASE main",
        "PRAGMA journal_mode",
    ],
)
def test_authorizer_rejects_writes_attach_detach_and_runtime_pragmas(
    tmp_path,
    statement,
):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)
    connection = sqlite_helpers.connect_readonly(database)
    try:
        with pytest.raises(sqlite_helpers.sqlite3.DatabaseError):
            connection.execute(statement).fetchall()
    finally:
        connection.close()


def test_authorizer_allows_only_readonly_schema_introspection_pragma(tmp_path):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)
    connection = sqlite_helpers.connect_readonly(database)
    try:
        columns = connection.execute("PRAGMA table_info(patients)").fetchall()
        assert any(row[1] == "patient_id" for row in columns)
        with pytest.raises(sqlite_helpers.sqlite3.DatabaseError):
            connection.execute("PRAGMA user_version").fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize(
    "function_call",
    [
        "load_extension('synthetic')",
        "writefile('synthetic', 'content')",
        "readfile('synthetic')",
        "randomblob(1)",
        "random()",
        "zeroblob(1)",
    ],
)
def test_authorizer_rejects_extension_io_and_blob_allocation_functions(
    tmp_path,
    function_call,
):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)
    connection = sqlite_helpers.connect_readonly(database)
    try:
        # ``readfile`` and ``writefile`` are supplied by SQLite's optional
        # shell extension rather than every Python build.  Register inert
        # stand-ins so this test proves the authorizer rejects the names even
        # when an extension makes them resolvable.  The callbacks must never
        # execute.
        if function_call.startswith("readfile"):
            connection.create_function(
                "readfile",
                1,
                lambda *_args: pytest.fail("readfile callback executed"),
            )
        elif function_call.startswith("writefile"):
            connection.create_function(
                "writefile",
                2,
                lambda *_args: pytest.fail("writefile callback executed"),
            )
        with pytest.raises(
            sqlite_helpers.sqlite3.DatabaseError,
            match="not authorized",
        ):
            connection.execute(f"SELECT {function_call}").fetchall()
    finally:
        connection.close()


def test_authorizer_rejects_reads_from_non_main_or_temp_databases():
    assert (
        sqlite_helpers._readonly_authorizer(
            sqlite_helpers.sqlite3.SQLITE_READ,
            "patients",
            "patient_id",
            "other",
            None,
        )
        == sqlite_helpers.sqlite3.SQLITE_DENY
    )
    assert (
        sqlite_helpers._readonly_authorizer(
            sqlite_helpers.sqlite3.SQLITE_READ,
            "patients",
            "patient_id",
            "main",
            None,
        )
        == sqlite_helpers.sqlite3.SQLITE_OK
    )


def test_connection_installs_declared_sqlite_limits(tmp_path):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)
    connection = sqlite_helpers.connect_readonly(database)
    try:
        installed = {
            category: connection.getlimit(category)
            for category in sqlite_helpers.SQLITE_RUNTIME_LIMITS
        }
    finally:
        connection.close()

    assert installed == sqlite_helpers.SQLITE_RUNTIME_LIMITS


def test_bounded_execution_interrupts_excessive_vm_work_and_cleans_up(tmp_path):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)
    connection = sqlite_helpers.connect_readonly(database)
    try:
        with pytest.raises(
            sqlite_helpers.sqlite3.OperationalError,
            match="interrupted",
        ):
            with sqlite_helpers.bounded_execution(
                connection,
                max_vm_steps=1_000,
                timeout_seconds=1.0,
                progress_ops=100,
            ):
                connection.execute(
                    "WITH RECURSIVE sequence(value) AS ("
                    "SELECT 1 UNION ALL SELECT value + 1 FROM sequence"
                    ") SELECT SUM(value) FROM sequence"
                ).fetchall()

        assert connection.execute("SELECT COUNT(*) FROM patients").fetchone() == (3,)
    finally:
        connection.close()


def test_bounded_execution_applies_a_monotonic_deadline(tmp_path, monkeypatch):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)
    connection = sqlite_helpers.connect_readonly(database)
    readings = iter((100.0, 102.0))
    monkeypatch.setattr(sqlite_helpers.time, "monotonic", lambda: next(readings))
    try:
        with pytest.raises(
            sqlite_helpers.sqlite3.OperationalError,
            match="interrupted",
        ):
            with sqlite_helpers.bounded_execution(
                connection,
                max_vm_steps=1_000_000,
                timeout_seconds=1.0,
                progress_ops=1,
            ):
                connection.execute(
                    "WITH RECURSIVE sequence(value) AS ("
                    "SELECT 1 UNION ALL SELECT value + 1 FROM sequence"
                    ") SELECT SUM(value) FROM sequence"
                ).fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"max_vm_steps": 0}, ValueError, "max_vm_steps must be positive"),
        ({"max_vm_steps": True}, TypeError, "max_vm_steps must be an integer"),
        ({"progress_ops": 0}, ValueError, "progress_ops must be positive"),
        ({"progress_ops": True}, TypeError, "progress_ops must be an integer"),
        ({"timeout_seconds": True}, TypeError, "timeout_seconds must be a number"),
        (
            {"timeout_seconds": float("inf")},
            ValueError,
            "timeout_seconds must be positive and finite",
        ),
        (
            {"timeout_seconds": float("nan")},
            ValueError,
            "timeout_seconds must be positive and finite",
        ),
        (
            {"timeout_seconds": 0},
            ValueError,
            "timeout_seconds must be positive and finite",
        ),
    ],
)
def test_bounded_execution_rejects_invalid_budgets(tmp_path, kwargs, exception, message):
    database = tmp_path / "synthetic.sqlite"
    _create_synthetic_database(database)
    connection = sqlite_helpers.connect_readonly(database)
    try:
        with pytest.raises(exception, match=message):
            with sqlite_helpers.bounded_execution(connection, **kwargs):
                pass
    finally:
        connection.close()


EVALUATOR_REACHABLE_READONLY_OPENERS = (
    "dki/analysis/nl2sql.py",
    "dki/analysis/nl2sql_protocol.py",
    "dki/analysis/metrics.py",
    "dki/analysis/nl2sql_css_projection.py",
    "dki/analysis/nl2sql_ehrsql_projection.py",
    "scripts/audit_ehrsql_public_source.py",
    "scripts/build_nl2sql_curator_pack.py",
    "scripts/build_nl2sql_hard_eval.py",
    "scripts/project_nl2sql_public_donors.py",
)


def _is_raw_sqlite_connect(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sqlite3"
    )


def test_all_evaluator_reachable_readonly_openers_use_shared_helper():
    root = Path(__file__).resolve().parent.parent
    for relative_path in EVALUATOR_REACHABLE_READONLY_OPENERS:
        source_path = root / relative_path
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

        assert "from dki.analysis.nl2sql_sqlite import connect_readonly" in source
        assert any(
            isinstance(node.func, ast.Name) and node.func.id == "connect_readonly"
            for node in calls
        )
        assert not any(_is_raw_sqlite_connect(node) for node in calls)
        assert "mode=ro" not in source


def test_database_readonly_helpers_use_the_shared_opener():
    source_path = Path(__file__).resolve().parent.parent / "dki/analysis/database.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    assert "from dki.analysis.nl2sql_sqlite import connect_readonly" in source
    for function_name in ("get_schema_description", "run_sql"):
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        assert any(
            isinstance(node.func, ast.Name) and node.func.id == "connect_readonly"
            for node in calls
        )
        assert not any(_is_raw_sqlite_connect(node) for node in calls)
