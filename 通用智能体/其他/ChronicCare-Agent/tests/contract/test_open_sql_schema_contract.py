from __future__ import annotations

import sqlite3

from analysis.open_sql.schema_catalog import (
    CORE_TABLES,
    JOIN_RELATIONS,
    build_schema_catalog,
    db_path,
)


def _database_fields() -> dict[str, set[str]]:
    with sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True) as connection:
        available = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {
            table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for table in CORE_TABLES & available
        }


def test_open_sql_core_tables_exist_in_delivery_database() -> None:
    fields = _database_fields()
    assert fields
    assert CORE_TABLES <= set(fields)
    assert {"patient_id", "disease_tags"} <= fields["patient_profile"]
    catalog = build_schema_catalog(write_files=False)
    assert catalog["status"] == "success"
    assert catalog["db_path"] == "data/sqlite/chroniccare.db"
    assert not catalog["db_path"].startswith("/")


def test_open_sql_join_whitelist_references_real_columns() -> None:
    fields = _database_fields()
    for left_table, left_field, right_table, right_field in JOIN_RELATIONS:
        assert left_table in fields
        assert right_table in fields
        assert left_field in fields[left_table]
        assert right_field in fields[right_table]


def test_open_sql_schema_does_not_expose_direct_identity_fields() -> None:
    exposed = set().union(*_database_fields().values())
    assert {"id_card", "phone", "address", "api_key"}.isdisjoint(exposed)

