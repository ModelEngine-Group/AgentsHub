import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mcps"))

from server import (  # noqa: E402
    analyze_course_data_tool,
    inspect_dataset_tool,
    query_course_data_tool,
)


def test_inspect_dataset() -> None:
    result = inspect_dataset_tool()
    assert result["status"] == "success"
    assert result["row_count"] == 30
    assert result["column_count"] == 9
    assert all(count == 0 for count in result["missing_count"].values())
    json.dumps(result, ensure_ascii=False)


def test_analyze_course_data() -> None:
    result = analyze_course_data_tool()
    assert result["status"] == "success"
    assert result["sample_count"] == 30
    assert result["final_score"]["mean"] > 0
    assert result["correlations_with_final_score"]["attendance_rate"] > 0.8
    assert result["at_risk_count"] == 0
    json.dumps(result, ensure_ascii=False)


def test_analyze_course_data_rejects_invalid_threshold() -> None:
    result = analyze_course_data_tool(at_risk_threshold=101)
    assert result["status"] == "error"


def test_query_course_data() -> None:
    result = query_course_data_tool(min_final_score=85, gender="女", limit=10)
    assert result["status"] == "success"
    assert result["matched_count"] > 0
    assert all(
        int(record["final_score"]) >= 85 and record["gender"] == "女"
        for record in result["records"]
    )
    json.dumps(result, ensure_ascii=False)


def test_query_course_data_rejects_invalid_limit() -> None:
    result = query_course_data_tool(limit=0)
    assert result["status"] == "error"
