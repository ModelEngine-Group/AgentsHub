from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from analysis.open_nl2sql import open_query_executor
from analysis.open_sql import result_formatter, sql_executor


def test_result_formatter_helper_rendering() -> None:
    rows = [{"patient_count": 2, "risk_level": "high"}]
    table = result_formatter._markdown_table(rows)
    assert "| 患者人数 | 风险等级 |" in table
    assert "| 2 | high |" in table
    assert result_formatter._markdown_table([]) == ""
    assert result_formatter._percent(0.125) == "12.50%"
    assert result_formatter._percent("invalid") is None
    assert result_formatter._disease_label([]) == "全部患者"
    assert result_formatter._disease_label(["hypertension", "unknown"]) == "高血压、unknown"


@pytest.mark.parametrize(
    ("time_range", "expected"),
    [
        (None, None),
        ({"type": "past_months", "value": 6}, None),
        ({"type": "future_days", "value": 7}, (7, 6)),
        ({"type": "future_days", "value": 0}, (1, 0)),
    ],
)
def test_future_days_window(time_range: dict | None, expected: tuple[int, int] | None) -> None:
    assert result_formatter._future_days_window(time_range) == expected


def test_line_svg_handles_single_and_multiple_points() -> None:
    single = result_formatter._line_svg(
        "单日趋势",
        [{"followup_date": "2026-07-28", "patient_count": 3}],
    )
    multiple = result_formatter._line_svg(
        "多日趋势",
        [{"followup_date": f"2026-08-{day:02d}", "patient_count": day % 4} for day in range(1, 17)],
    )
    assert "<svg" in single
    assert "单日趋势" in single
    assert ">3</text>" in single
    assert "<polyline" in multiple
    assert "08-16" in multiple


@pytest.mark.parametrize(
    ("row", "fragment"),
    [
        (
            {"control_rate": 0.8, "numerator": 8, "denominator": 10},
            "达标率为 0.8（80.00%）",
        ),
        (
            {"abnormal_rate": 0.25, "numerator": 1, "denominator": 4},
            "异常率为 0.25（25.00%）",
        ),
        ({"avg_value": 7.05, "patient_count": 56}, "平均值为 7.05"),
        ({"avg_bmi": 24.2, "patient_count": 80}, "平均 BMI 为 24.2"),
        ({"patient_count": 433}, "患者人数 433 人"),
    ],
)
def test_format_result_builds_single_row_summaries(row: dict, fragment: str) -> None:
    result = result_formatter.format_result(
        question="问题",
        query_spec={"aggregation": "count"},
        template={"explanation": "统计结果"},
        execution={"status": "success", "rows": [row]},
        allow_chart=False,
    )
    assert fragment in result["summary_text"]
    assert "|" in result["answer_markdown"]
    assert result["chart_url"] is None
    assert result["charts"] == []


def test_format_result_handles_failure_and_empty_rows() -> None:
    failure = result_formatter.format_result(
        question="问题",
        query_spec={},
        template={},
        execution={"status": "failed", "error": "denied"},
        allow_chart=False,
    )
    empty = result_formatter.format_result(
        question="问题",
        query_spec={},
        template={},
        execution={"status": "success", "rows": []},
        allow_chart=False,
    )
    assert failure["summary_text"] == "SQL 执行失败：denied"
    assert empty["summary_text"] == "当前数据未检索到符合条件的记录。"


def test_format_result_includes_chart_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(result_formatter, "_write_chart", lambda *args: "http://chart")
    monkeypatch.setattr(
        result_formatter,
        "_write_followup_chart",
        lambda *args: {
            "image_url": "http://image",
            "image_service_url": "http://service",
            "charts": [{"name": "随访趋势"}],
            "trend_rows": [{"followup_date": "2026-07-28", "patient_count": 2}],
        },
    )
    result = result_formatter.format_result(
        question="未来7天随访",
        query_spec={"intent": "followup_count", "aggregation": "trend"},
        template={"explanation": "随访统计"},
        execution={"status": "success", "rows": [{"patient_count": 2}]},
        allow_chart=True,
    )
    assert "![随访趋势图](http://image)" in result["answer_markdown"]
    assert "图表入口：http://chart" in result["answer_markdown"]
    assert result["image_service_url"] == "http://service"
    assert result["trend_rows"][0]["patient_count"] == 2


def _create_readonly_test_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE patient_profile (patient_id TEXT, disease_tags TEXT)")
        connection.executemany(
            "INSERT INTO patient_profile VALUES (?, ?)",
            [("p1", "hypertension"), ("p2", "diabetes"), ("p3", "hypertension")],
        )


def _patch_executor_catalog(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    monkeypatch.setattr(sql_executor, "db_path", lambda: db)
    monkeypatch.setattr(
        sql_executor,
        "get_schema_catalog",
        lambda: {
            "tables": {
                "patient_profile": {
                    "fields": [
                        {"name": "patient_id"},
                        {"name": "disease_tags"},
                    ]
                }
            }
        },
    )
    monkeypatch.setattr(
        sql_executor,
        "allowed_tables",
        lambda catalog: {"patient_profile"},
    )
    monkeypatch.setattr(
        sql_executor,
        "allowed_fields",
        lambda catalog: {
            "patient_profile": {"patient_id", "disease_tags"},
        },
    )


def test_sql_executor_reads_truncates_and_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db = tmp_path / "test.db"
    _create_readonly_test_database(db)
    _patch_executor_catalog(monkeypatch, db)
    result = sql_executor.execute_sql(
        "SELECT patient_id, disease_tags FROM patient_profile ORDER BY patient_id",
        max_rows=2,
    )
    assert result["status"] == "success"
    assert result["row_count"] == 2
    assert result["truncated"] is True
    assert result["empty"] is False
    assert result["resource_limits"]["max_rows"] == 2


@pytest.mark.parametrize(
    ("sql", "error_code"),
    [
        ("SELECT secret FROM patient_profile", "query_execution_failed"),
        ("DELETE FROM patient_profile", "sqlite_authorizer_denied"),
        ("SELECT * FROM missing_table", "query_execution_failed"),
    ],
)
def test_sql_executor_blocks_unsafe_or_invalid_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sql: str,
    error_code: str,
) -> None:
    db = tmp_path / "test.db"
    _create_readonly_test_database(db)
    _patch_executor_catalog(monkeypatch, db)
    result = sql_executor.execute_sql(sql)
    assert result["status"] == "failed"
    assert result["error_code"] == error_code
    assert result["rows"] == []


def test_open_query_executor_success_and_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        open_query_executor,
        "route_intent",
        lambda payload: {"intent": "nl2sql", "tool": "sql", "reason": "matched"},
    )
    monkeypatch.setattr(
        open_query_executor,
        "build_schema_links",
        lambda question: {"tables": ["patient_profile"]},
    )
    monkeypatch.setattr(
        open_query_executor,
        "build_sql_candidate",
        lambda *args: {"sql": "SELECT 1"},
    )
    monkeypatch.setattr(
        open_query_executor,
        "validate_sql",
        lambda sql: (True, [], "SELECT 1 LIMIT 500", []),
    )
    monkeypatch.setattr(
        open_query_executor,
        "fetch_rows",
        lambda sql: [{"value": 1}, {"value": 2}],
    )
    success = open_query_executor.execute_open_query("测试")
    assert success["status"] == "success"
    assert success["row_count"] == 2
    assert success["sql"].endswith("LIMIT 500")

    monkeypatch.setattr(
        open_query_executor,
        "build_sql_candidate",
        lambda *args: {"sql": None},
    )
    no_candidate = open_query_executor.execute_open_query("测试")
    assert no_candidate["status"] == "failed"
    assert no_candidate["errors"] == ["No SQL candidate generated."]

    monkeypatch.setattr(
        open_query_executor,
        "build_sql_candidate",
        lambda *args: {"sql": "SELECT 1"},
    )
    monkeypatch.setattr(
        open_query_executor,
        "validate_sql",
        lambda sql: (False, ["blocked"], None, ["warning"]),
    )
    blocked = open_query_executor.execute_open_query("测试")
    assert blocked["status"] == "failed"
    assert blocked["warnings"] == ["warning"]
    assert blocked["errors"] == ["blocked"]
