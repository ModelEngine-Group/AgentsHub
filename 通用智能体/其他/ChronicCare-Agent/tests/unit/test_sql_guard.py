import pytest

from analysis.open_nl2sql.sql_guard import validate_sql


@pytest.mark.parametrize("sql", [None, "", "   "])
def test_rejects_empty_sql(sql: str | None) -> None:
    safe, errors, normalized, warnings = validate_sql(sql)
    assert safe is False
    assert errors
    assert normalized is None
    assert warnings == []


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE patient_profile",
        "DELETE FROM patient_profile",
        "UPDATE patient_profile SET age = 1",
        "INSERT INTO patient_profile(patient_id) VALUES ('x')",
    ],
)
def test_rejects_non_select_statements(sql: str) -> None:
    safe, errors, _, _ = validate_sql(sql)
    assert safe is False
    assert any("Only a single SELECT" in item for item in errors)


def test_rejects_multiple_statements() -> None:
    safe, errors, _, _ = validate_sql("SELECT patient_id FROM patient_profile; SELECT patient_id FROM patient_profile")
    assert safe is False
    assert any("Exactly one SQL statement" in item for item in errors)


def test_rejects_comments() -> None:
    safe, errors, _, _ = validate_sql("SELECT patient_id FROM patient_profile -- bypass")
    assert safe is False
    assert any("comments are not allowed" in item for item in errors)


def test_rejects_unknown_table() -> None:
    safe, errors, _, _ = validate_sql("SELECT value FROM private_table")
    assert safe is False
    assert any("Table not allowed" in item for item in errors)


def test_accepts_whitelisted_select_and_adds_limit() -> None:
    safe, errors, normalized, warnings = validate_sql("SELECT patient_id, age FROM patient_profile")
    assert safe is True
    assert errors == []
    assert normalized is not None and "LIMIT 200" in normalized
    assert warnings


def test_preserves_explicit_limit() -> None:
    safe, errors, normalized, warnings = validate_sql("SELECT patient_id FROM patient_profile LIMIT 5")
    assert safe is True
    assert errors == []
    assert normalized is not None and "LIMIT 5" in normalized
    assert warnings == []
