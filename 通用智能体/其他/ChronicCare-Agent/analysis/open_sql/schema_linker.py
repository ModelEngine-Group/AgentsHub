from __future__ import annotations

from typing import Any, Dict, List

from analysis.open_sql.alias_registry import INDICATOR_ITEM_NAMES
from analysis.open_sql.schema_catalog import get_schema_catalog


def build_schema_links(query_spec: Dict[str, Any], catalog: Dict[str, Any] | None = None) -> Dict[str, Any]:
    catalog = catalog or get_schema_catalog()
    tables = catalog.get("tables") or {}
    errors: List[str] = []
    required_tables = {"patient_profile"}
    fields: Dict[str, List[str]] = {"patient_profile": ["patient_id", "disease_tags"]}

    intent = str(query_spec.get("intent") or "")
    indicators = query_spec.get("indicators") or []
    if intent.startswith(("avg", "abnormal_rate", "trend")) or indicators:
        if any(indicator == "bmi" for indicator in indicators):
            fields.setdefault("patient_profile", []).append("bmi")
        if any(indicator != "bmi" for indicator in indicators):
            required_tables.add("lab_result")
            fields.setdefault("lab_result", []).extend(["patient_id", "item_name", "value", "item_value", "abnormal_flag", "test_date"])
    if query_spec.get("risk_level") or "risk" in intent:
        required_tables.add("patient_risk_score")
        fields.setdefault("patient_risk_score", []).extend(["patient_id", "risk_level", "risk_score", "created_at"])
    if intent.startswith("followup"):
        required_tables.add("followup_plan")
        fields.setdefault("followup_plan", []).extend(["patient_id", "followup_date", "priority", "status"])
    if query_spec.get("lifestyle"):
        required_tables.add("lifestyle_record")
        required_tables.add("visit_record")
        lifestyle_field = query_spec["lifestyle"].get("field")
        fields.setdefault("lifestyle_record", []).extend(["patient_id", "visit_id", lifestyle_field])
        fields.setdefault("visit_record", []).extend(["patient_id", "visit_id"])
    if query_spec.get("medication_category") or "medication" in intent:
        required_tables.add("medication_record")
        required_tables.add("visit_record")
        fields.setdefault("medication_record", []).extend(["patient_id", "visit_id", "drug_category", "drug_name"])
        fields.setdefault("visit_record", []).extend(["patient_id", "visit_id"])

    for table in sorted(required_tables):
        if table not in tables:
            errors.append(f"table_not_found:{table}")
            continue
        allowed = {field["name"] for field in tables[table].get("fields", [])}
        fields[table] = sorted({field for field in fields.get(table, []) if field})
        missing = [field for field in fields[table] if field not in allowed]
        if missing:
            errors.append(f"fields_not_found:{table}.{','.join(missing)}")

    indicator_items = []
    for indicator in indicators:
        indicator_items.extend(INDICATOR_ITEM_NAMES.get(indicator, [indicator]))
    return {
        "status": "schema_link_failed" if errors else "success",
        "tables": sorted(required_tables),
        "fields": fields,
        "indicator_items": sorted(set(indicator_items)),
        "joins": catalog.get("joins") or [],
        "errors": errors,
    }
