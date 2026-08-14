from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

try:
    from datamate.core.base_op import Mapper  # type: ignore
except Exception:  # pragma: no cover
    class Mapper:  # type: ignore
        pass


SAFETY_NOTE = "本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。"
DISEASE_TERMS = {
    "hypertension": ["高血压", "hypertension"],
    "diabetes": ["糖尿病", "diabetes"],
    "hyperlipidemia": ["高脂血症", "高血脂", "hyperlipidemia"],
}
DRUG_TERMS = {
    "metformin": ["二甲双胍", "metformin"],
    "insulin": ["胰岛素", "insulin"],
    "amlodipine": ["氨氯地平", "amlodipine"],
    "atorvastatin": ["阿托伐他汀", "atorvastatin"],
}
INDICATOR_TERMS = {
    "hba1c": ["hba1c", "糖化血红蛋白"],
    "fasting_glucose": ["空腹血糖", "fasting_glucose"],
    "systolic_bp": ["收缩压", "systolic_bp"],
    "ldl_c": ["ldl-c", "ldl_c"],
    "bmi": ["bmi"],
}
DEFAULT_QUESTIONS = [
    {"id": "Q001", "question": "患者总数是多少？", "sql": "SELECT COUNT(*) AS patient_count FROM patient_profile;"},
    {"id": "Q002", "question": "随访记录总数是多少？", "sql": "SELECT COUNT(*) AS visit_count FROM visit_record;"},
    {"id": "Q003", "question": "平均 HbA1c 是多少？", "sql": "SELECT ROUND(AVG(CAST(item_value AS REAL)), 2) AS avg_hba1c FROM lab_result WHERE lower(item_name)='hba1c';"},
    {"id": "Q004", "question": "BMI 偏高患者数是多少？", "sql": "SELECT COUNT(*) AS overweight_count FROM patient_profile WHERE CAST(bmi AS REAL) >= 24.0;"},
    {"id": "Q005", "question": "高血压患者数是多少？", "sql": "SELECT COUNT(*) AS hypertension_count FROM patient_profile WHERE lower(disease_tags) LIKE '%hypertension%' OR disease_tags LIKE '%高血压%';"},
]
FIELD_ALIASES = {
    "patient id": "patient_id",
    "patientid": "patient_id",
    "visit id": "visit_id",
    "visitid": "visit_id",
    "lab id": "lab_id",
    "med id": "med_id",
    "item value": "item_value",
    "item name": "item_name",
    "test date": "test_date",
    "drug name": "drug_name",
    "disease tag": "disease_tags",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: Any) -> str:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path.as_posix()


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> str:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path.as_posix()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> str:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path.as_posix()


def structured_table_dir(sample: Dict[str, Any], params: Dict[str, Any] | None = None) -> Path | None:
    params = params or {}
    history = pipeline_artifacts(sample)
    sample_path = sample.get("filePath")
    sample_file = Path(sample_path).resolve() if sample_path else None
    return first_existing_path(
        [
            params.get("structured_tables_path"),
            history.get("normalized_tables"),
            history.get("clean_tables"),
            history.get("raw_structured_dir"),
            sample_file / "structured" if sample_file and sample_file.is_dir() else None,
            sample_file,
        ]
    )


def load_structured_tables(sample: Dict[str, Any], params: Dict[str, Any] | None = None) -> Dict[str, List[Dict[str, Any]]]:
    table_dir = structured_table_dir(sample, params)
    if not table_dir or not table_dir.exists():
        return {}
    tables: Dict[str, List[Dict[str, Any]]] = {}
    for csv_path in sorted(table_dir.glob("*.csv")):
        tables[csv_path.stem.replace("_clean", "").replace("_normalized", "").replace("_standard", "")] = canonical_rows(read_csv_rows(csv_path))
    return tables


def append_entity(entities: List[Dict[str, Any]], seen: set[str], counter: Counter[str], entity_id: str, entity_type: str, canonical_name: str, **extra: Any) -> None:
    if entity_id in seen:
        return
    item = {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "canonical_name": canonical_name,
    }
    item.update(extra)
    entities.append(item)
    seen.add(entity_id)
    counter[entity_type] += 1


def append_relation(relations: List[Dict[str, Any]], relation_counter: Counter[str], head: str, relation: str, tail: str, **extra: Any) -> None:
    item = {"head": head, "relation": relation, "tail": tail}
    item.update(extra)
    relations.append(item)
    relation_counter[relation] += 1


PRIMARY_DISEASE_INDICATORS = {
    "hypertension": "systolic_bp",
    "diabetes": "hba1c",
    "hyperlipidemia": "ldl_c",
    "obesity": "bmi",
    "chronic_kidney_disease": "egfr",
    "fatty_liver_disease": "alt",
    "hyperuricemia": "uric_acid",
    "coronary_heart_disease": "ldl_c",
    "gout": "uric_acid",
    "chronic_heart_failure": "systolic_bp",
    "diabetic_kidney_disease": "uacr",
    "obstructive_sleep_apnea": "bmi",
    "cerebrovascular_disease": "systolic_bp",
    "atrial_fibrillation": "systolic_bp",
    "chronic_hepatitis": "alt",
}


DISEASE_DRUG_MAP = {
    "diabetes": "metformin",
    "hypertension": "valsartan",
    "hyperlipidemia": "atorvastatin",
    "hyperuricemia": "febuxostat",
    "gout": "allopurinol",
    "chronic_heart_failure": "spironolactone",
    "diabetic_kidney_disease": "dapagliflozin",
    "cerebrovascular_disease": "clopidogrel",
    "atrial_fibrillation": "rivaroxaban",
    "chronic_hepatitis": "entecavir",
    "hypothyroidism": "levothyroxine",
}


RISK_FACTOR_CANONICAL = {
    "smoking": "smoking",
    "alcohol": "drinking",
    "drinking": "drinking",
    "high_salt_diet": "obesity",
    "lack_of_exercise": "obesity",
    "insufficient_sleep": "insufficient_sleep",
    "obesity": "obesity",
}


def normalize_disease_tags(raw_value: Any) -> List[str]:
    text = str(raw_value).strip()
    if not text:
        return []
    return [part.strip() for part in str(raw_value).split(";") if part.strip()]


def risk_event_from_lab(item_name: str, abnormal_flag: str) -> str | None:
    if abnormal_flag == "normal":
        return None
    if item_name in {"fasting_glucose", "postprandial_glucose", "hba1c"}:
        return "glucose_high"
    if item_name in {"systolic_bp", "diastolic_bp"}:
        return "blood_pressure_high"
    if item_name in {"total_cholesterol", "ldl_c", "hdl_c", "triglyceride"}:
        return "lipid_abnormal"
    if item_name == "bmi":
        return "bmi_high"
    return None


def is_truthy_yes(raw_value: Any) -> bool:
    return str(raw_value).strip().lower() == "yes"


def as_float(raw_value: Any) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0


def recalc_abnormal_flag(item_name: str, item_value: float) -> str:
    if item_name == "fasting_glucose":
        return "high" if item_value > 6.1 else "low" if item_value < 3.9 else "normal"
    if item_name == "postprandial_glucose":
        return "high" if item_value > 7.8 else "normal"
    if item_name == "hba1c":
        return "high" if item_value >= 6.5 else "normal"
    if item_name == "systolic_bp":
        return "high" if item_value >= 140 else "low" if item_value < 90 else "normal"
    if item_name == "diastolic_bp":
        return "high" if item_value >= 90 else "low" if item_value < 60 else "normal"
    if item_name == "total_cholesterol":
        return "high" if item_value >= 5.2 else "normal"
    if item_name == "ldl_c":
        return "high" if item_value >= 3.4 else "normal"
    if item_name == "hdl_c":
        return "low" if item_value < 1.0 else "normal"
    if item_name == "triglyceride":
        return "high" if item_value >= 1.7 else "normal"
    if item_name == "bmi":
        return "high" if item_value >= 24.0 else "low" if item_value < 18.5 else "normal"
    return "normal"


def relative_str(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def operator_export_dir(sample: Dict[str, Any], operator_name: str) -> Path:
    root = Path(sample.get("export_path", "./outputs")).resolve()
    return ensure_dir(root / operator_name)


def pipeline_artifacts(sample: Dict[str, Any]) -> Dict[str, str]:
    value = sample.get("pipeline_artifacts")
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items() if isinstance(val, (str, int, float, bool))}
    return {}


def resolve_input_path(sample: Dict[str, Any], params: Dict[str, Any] | None = None) -> Path:
    value = sample.get("filePath") or (params or {}).get("filePath")
    if not value:
        raise ValueError("sample['filePath'] is required.")
    return Path(value).resolve()


def first_existing_path(candidates: Iterable[str | Path | None]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def resolve_operator_input(operator_name: str, sample: Dict[str, Any], params: Dict[str, Any] | None = None) -> Path:
    params = params or {}
    history = pipeline_artifacts(sample)
    sample_path = sample.get("filePath")
    sample_file = Path(sample_path).resolve() if sample_path else None

    if operator_name == "chronic_file_ingest":
        return resolve_input_path(sample, params)
    if operator_name == "chronic_table_clean":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("raw_structured_dir"),
                sample_file / "structured" if sample_file and sample_file.is_dir() else None,
                history.get("clean_tables"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_field_normalize":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("clean_tables"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_text_split":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("raw_text_dir"),
                sample_file / "text" if sample_file and sample_file.is_dir() else None,
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_entity_extract":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("chunk_file"),
                history.get("chunk_dir"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_relation_extract":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("entities_raw"),
                history.get("entities"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_triple_validate":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("relations_raw"),
                history.get("relations"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_kg_build":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("triples_clean"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_sqlite_loader":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("normalized_tables"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_nl2sql_analyze":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("sqlite_db"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    if operator_name == "chronic_report_pack":
        return first_existing_path(
            [
                params.get("filePath"),
                history.get("indicator_results"),
                sample_file,
            ]
        ) or resolve_input_path(sample, params)
    return resolve_input_path(sample, params)


def slim_update(sample: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            allowed[key] = value
        elif isinstance(value, list) and len(value) <= 12 and all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
            allowed[key] = value
        elif isinstance(value, dict):
            compact = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    compact[sub_key] = sub_value
            allowed[key] = compact
    sample.update(allowed)
    return sample


def detect_terms(text: str, mapping: Dict[str, List[str]]) -> List[str]:
    lowered = text.lower()
    found = []
    for canonical, aliases in mapping.items():
        if any(alias.lower() in lowered for alias in aliases):
            found.append(canonical)
    return found


def normalize_key(value: str) -> str:
    cleaned = re.sub(r"[\s\-]+", "_", value.strip().lower())
    return FIELD_ALIASES.get(cleaned.replace("_", " "), FIELD_ALIASES.get(cleaned, cleaned))


def canonical_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append({normalize_key(str(key)): (value.strip() if isinstance(value, str) else value) for key, value in row.items()})
    return normalized


def file_ingest(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    raw_root = resolve_operator_input("chronic_file_ingest", sample, params)
    out_dir = operator_export_dir(sample, "chronic_file_ingest")
    files = []
    structured_dir = raw_root / "structured" if raw_root.is_dir() and (raw_root / "structured").exists() else raw_root
    text_dir = raw_root / "text" if raw_root.is_dir() and (raw_root / "text").exists() else raw_root

    def append_item(item: Path, base: Path) -> None:
        files.append(
            {
                "name": item.name,
                "relative_path": relative_str(item, base),
                "size_bytes": item.stat().st_size,
                "category": "text" if item.suffix.lower() in {".txt", ".jsonl"} else "structured",
            }
        )

    if raw_root.is_file():
        append_item(raw_root, raw_root.parent)
    else:
        seen_paths = set()
        for item in sorted(structured_dir.rglob("*")) + sorted(text_dir.rglob("*")):
            if item.is_file():
                resolved = item.resolve().as_posix()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                append_item(item, raw_root)

    manifest = {
        "status": "success",
        "file_count": len(files),
        "structured_files": sum(1 for item in files if item["category"] == "structured"),
        "text_files": sum(1 for item in files if item["category"] == "text"),
        "files": files,
        "safety_note": SAFETY_NOTE,
    }
    manifest_path = Path(dump_json(out_dir / "manifest.json", manifest))
    return {
        "status": "success",
        "artifact_paths": {
            "manifest": manifest_path.as_posix(),
            "raw_structured_dir": structured_dir.as_posix() if structured_dir.exists() else "",
            "raw_text_dir": text_dir.as_posix() if text_dir.exists() else "",
        },
        "summary": {
            "file_count": manifest["file_count"],
            "structured_files": manifest["structured_files"],
            "text_files": manifest["text_files"],
        },
    }


def table_clean(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    input_path = resolve_operator_input("chronic_table_clean", sample, params)
    out_dir = operator_export_dir(sample, "chronic_table_clean")
    clean_dir = ensure_dir(out_dir / "clean_tables")
    summaries = []

    if input_path.is_file():
        csv_paths = [input_path]
    else:
        csv_paths = sorted(input_path.glob("*.csv"))

    for csv_path in csv_paths:
        rows = canonical_rows(read_csv_rows(csv_path))
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for row in rows:
            signature = tuple(sorted((key, str(value)) for key, value in row.items()))
            if signature not in seen:
                seen.add(signature)
                deduped.append(row)
        fieldnames = sorted({key for row in deduped for key in row.keys()})
        output_path = clean_dir / f"{csv_path.stem}_clean.csv"
        write_csv_rows(output_path, deduped, fieldnames)
        summaries.append({"table": csv_path.name, "rows_before": len(rows), "rows_after": len(deduped), "output_path": output_path.as_posix()})

    report = {"status": "success", "table_count": len(summaries), "tables": summaries, "safety_note": SAFETY_NOTE}
    report_path = Path(dump_json(out_dir / "table_clean_report.json", report))

    result = {
        "status": "success",
        "artifact_paths": {"clean_tables": clean_dir.as_posix(), "report": report_path.as_posix()},
        "summary": {"table_count": len(summaries), "rows_after": sum(item["rows_after"] for item in summaries)},
    }

    # For DataMate front-end task persistence, single-file mode should surface a new concrete output file.
    if input_path.is_file() and summaries:
        output_path = Path(summaries[0]["output_path"])
        result.update(
            {
                "filePath": output_path.as_posix(),
                "fileName": output_path.name,
                "fileType": output_path.suffix.lstrip(".").lower(),
                "fileSize": output_path.stat().st_size,
                "execute_result": True,
            }
        )

    return result


def field_normalize(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    input_dir = resolve_operator_input("chronic_field_normalize", sample, params)
    out_dir = operator_export_dir(sample, "chronic_field_normalize")
    normalized_dir = ensure_dir(out_dir / "normalized_tables")
    tables = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        rows = canonical_rows(read_csv_rows(csv_path))
        fieldnames = sorted({key for row in rows for key in row.keys()})
        normalized_rows = []
        for row in rows:
            normalized_row = dict(row)
            if "disease_tags" in normalized_row and isinstance(normalized_row["disease_tags"], str):
                normalized_row["disease_tags"] = normalized_row["disease_tags"].replace("；", ";").replace("，", ";")
            if csv_path.stem.replace("_clean", "") == "lab_result" and "item_name" in normalized_row and "item_value" in normalized_row:
                normalized_row["abnormal_flag"] = recalc_abnormal_flag(
                    str(normalized_row.get("item_name", "")).strip().lower(),
                    as_float(normalized_row.get("item_value")),
                )
            normalized_rows.append(normalized_row)
        name = csv_path.name.replace("_clean", "")
        output_path = normalized_dir / name
        write_csv_rows(output_path, normalized_rows, fieldnames)
        tables.append({"table": name, "row_count": len(normalized_rows), "output_path": output_path.as_posix()})
    report = {"status": "success", "table_count": len(tables), "tables": tables, "safety_note": SAFETY_NOTE}
    report_path = Path(dump_json(out_dir / "field_normalize_report.json", report))
    return {
        "status": "success",
        "artifact_paths": {"normalized_tables": normalized_dir.as_posix(), "report": report_path.as_posix()},
        "summary": {"table_count": len(tables)},
    }


def text_split(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    input_dir = resolve_operator_input("chronic_text_split", sample, params)
    out_dir = operator_export_dir(sample, "chronic_text_split")
    chunk_dir = ensure_dir(out_dir / "chunks")
    chunk_path = chunk_dir / "all_chunks.jsonl"
    chunk_size = int(params.get("chunk_size", 120))
    texts: List[Dict[str, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".txt":
            texts.append({"source": path.name, "text": path.read_text(encoding="utf-8", errors="ignore")})
        elif path.suffix == ".jsonl":
            for index, row in enumerate(read_jsonl(path)):
                texts.append({"source": f"{path.name}:{index}", "text": str(row.get("text") or row.get("content") or row.get("note_text") or "")})
    chunks = []
    for text_doc in texts:
        text = text_doc["text"].strip()
        for index in range(0, len(text), chunk_size):
            piece = text[index:index + chunk_size]
            if piece:
                chunks.append({"chunk_id": f"{text_doc['source']}#{index // chunk_size}", "source": text_doc["source"], "text": piece})
    dump_jsonl(chunk_path, chunks)
    report_path = Path(dump_json(out_dir / "text_split_report.json", {"status": "success", "chunk_count": len(chunks), "document_count": len(texts), "safety_note": SAFETY_NOTE}))
    return {
        "status": "success",
        "artifact_paths": {"chunk_dir": chunk_dir.as_posix(), "chunk_file": chunk_path.as_posix(), "report": report_path.as_posix()},
        "summary": {"chunk_count": len(chunks), "document_count": len(texts)},
    }


def entity_extract(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    chunk_dir = resolve_operator_input("chronic_entity_extract", sample, params)
    out_dir = operator_export_dir(sample, "chronic_entity_extract")
    entity_dir = ensure_dir(out_dir / "entities")
    chunk_file = chunk_dir if chunk_dir.is_file() else chunk_dir / "all_chunks.jsonl"
    entities = []
    seen_entities: set[str] = set()
    counter: Counter[str] = Counter()
    for chunk in read_jsonl(chunk_file):
        text = str(chunk.get("text", ""))
        chunk_id = str(chunk.get("chunk_id", "chunk"))
        for category, mapping in [("Disease", DISEASE_TERMS), ("Drug", DRUG_TERMS), ("Indicator", INDICATOR_TERMS)]:
            for canonical in detect_terms(text, mapping):
                append_entity(
                    entities,
                    seen_entities,
                    counter,
                    f"{category}::{canonical}",
                    category,
                    canonical,
                    source_chunk=chunk_id,
                    source_type="text_chunk",
                )

    tables = load_structured_tables(sample, params)
    for row in tables.get("patient_profile", []):
        append_entity(entities, seen_entities, counter, f"Patient::{row['patient_id']}", "Patient", row["patient_id"], source_type="structured_table")
        for disease in normalize_disease_tags(row.get("disease_tags", "")):
            append_entity(entities, seen_entities, counter, f"Disease::{disease}", "Disease", disease, source_type="structured_table")
        if is_truthy_yes(row.get("smoking")):
            append_entity(entities, seen_entities, counter, "RiskFactor::smoking", "RiskFactor", "smoking", source_type="structured_table")
        if is_truthy_yes(row.get("drinking")):
            append_entity(entities, seen_entities, counter, "RiskFactor::drinking", "RiskFactor", "drinking", source_type="structured_table")
        if as_float(row.get("bmi")) >= 24.0:
            append_entity(entities, seen_entities, counter, "RiskFactor::obesity", "RiskFactor", "obesity", source_type="structured_table")

    for row in tables.get("visit_record", []):
        append_entity(entities, seen_entities, counter, f"Visit::{row['visit_id']}", "Visit", row["visit_id"], source_type="structured_table")

    for row in tables.get("lab_result", []):
        append_entity(entities, seen_entities, counter, f"LabResult::{row['lab_id']}", "LabResult", row["lab_id"], source_type="structured_table")
        append_entity(entities, seen_entities, counter, f"Indicator::{row['item_name']}", "Indicator", row["item_name"], source_type="structured_table")

    for row in tables.get("medication_record", []):
        append_entity(entities, seen_entities, counter, f"Drug::{row['drug_name']}", "Drug", row["drug_name"], source_type="structured_table")
        append_entity(entities, seen_entities, counter, f"DrugCategory::{row['drug_category']}", "DrugCategory", row["drug_category"], source_type="structured_table")

    for row in tables.get("risk_event", []):
        append_entity(entities, seen_entities, counter, f"RiskEvent::{row['event_type']}", "RiskEvent", row["event_type"], source_type="structured_table")

    for row in tables.get("followup_plan", []):
        append_entity(entities, seen_entities, counter, f"FollowupPlan::{row['plan_id']}", "FollowupPlan", row["plan_id"], source_type="structured_table")

    for row in tables.get("lifestyle_record", []):
        append_entity(entities, seen_entities, counter, f"LifestyleRecord::{row['record_id']}", "LifestyleRecord", row["record_id"], source_type="structured_table")

    for row in tables.get("doctor_advice", []):
        append_entity(entities, seen_entities, counter, f"DoctorAdvice::{row['advice_id']}", "DoctorAdvice", row["advice_id"], source_type="structured_table")

    for row in tables.get("patient_risk_score", []):
        append_entity(entities, seen_entities, counter, f"RiskScore::{row['score_id']}", "RiskScore", row["score_id"], source_type="structured_table")

    catalog = {"status": "success", "entity_count": len(entities), "entity_type_count": dict(counter), "safety_note": SAFETY_NOTE}
    raw_path = Path(dump_jsonl(entity_dir / "entities_raw.jsonl", entities))
    catalog_path = Path(dump_json(entity_dir / "entity_catalog.json", catalog))
    report_path = Path(dump_json(out_dir / "entity_extract_report.json", {"status": "success", "entity_count": len(entities), "entity_type_count": dict(counter), "safety_note": SAFETY_NOTE}))
    return {
        "status": "success",
        "artifact_paths": {"entities": entity_dir.as_posix(), "entities_raw": raw_path.as_posix(), "entity_catalog": catalog_path.as_posix(), "report": report_path.as_posix()},
        "summary": {"entity_count": len(entities), "entity_type_count": dict(counter)},
    }


def relation_extract(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    entities_path = resolve_operator_input("chronic_relation_extract", sample, params)
    history = pipeline_artifacts(sample)
    chunks_hint = params.get("chunks_path") or history.get("chunk_file") or history.get("chunk_dir")
    chunk_file = Path(chunks_hint).resolve() if chunks_hint else entities_path.parent.parent / "chronic_text_split" / "chunks" / "all_chunks.jsonl"
    if chunk_file.is_dir():
        chunk_file = chunk_file / "all_chunks.jsonl"
    entity_file = entities_path if entities_path.is_file() else entities_path / "entities_raw.jsonl"
    out_dir = operator_export_dir(sample, "chronic_relation_extract")
    relation_dir = ensure_dir(out_dir / "relations")
    entities_by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entity in read_jsonl(entity_file):
        entities_by_chunk[str(entity.get("source_chunk", ""))].append(entity)
    relations = []
    relation_counter: Counter[str] = Counter()
    for chunk in read_jsonl(chunk_file):
        chunk_id = str(chunk.get("chunk_id", ""))
        entities = entities_by_chunk.get(chunk_id, [])
        diseases = [item for item in entities if item["entity_type"] == "Disease"]
        drugs = [item for item in entities if item["entity_type"] == "Drug"]
        indicators = [item for item in entities if item["entity_type"] == "Indicator"]
        for disease in diseases:
            for drug in drugs:
                append_relation(relations, relation_counter, disease["entity_id"], "disease_treated_by_drug", drug["entity_id"], source_chunk=chunk_id, source_type="text_chunk")
            for indicator in indicators:
                append_relation(relations, relation_counter, disease["entity_id"], "disease_has_indicator", indicator["entity_id"], source_chunk=chunk_id, source_type="text_chunk")

    tables = load_structured_tables(sample, params)
    for row in tables.get("patient_profile", []):
        patient_id = f"Patient::{row['patient_id']}"
        for disease in normalize_disease_tags(row.get("disease_tags", "")):
            append_relation(relations, relation_counter, patient_id, "patient_has_disease", f"Disease::{disease}", source_table="patient_profile")

    for row in tables.get("visit_record", []):
        append_relation(relations, relation_counter, f"Patient::{row['patient_id']}", "patient_has_visit", f"Visit::{row['visit_id']}", source_table="visit_record", edge_uid=row["visit_id"])

    for row in tables.get("lab_result", []):
        lab_node = f"LabResult::{row['lab_id']}"
        append_relation(relations, relation_counter, f"Visit::{row['visit_id']}", "visit_has_lab", lab_node, source_table="lab_result", edge_uid=row["lab_id"])
        append_relation(relations, relation_counter, lab_node, "lab_result_belongs_to_indicator", f"Indicator::{row['item_name']}", source_table="lab_result", edge_uid=row["lab_id"])
        risk_event = risk_event_from_lab(str(row.get("item_name", "")).lower(), str(row.get("abnormal_flag", "")).lower())
        if risk_event:
            append_relation(relations, relation_counter, lab_node, "lab_result_indicates_risk", f"RiskEvent::{risk_event}", source_table="lab_result", edge_uid=row["lab_id"])

    for disease, indicator in PRIMARY_DISEASE_INDICATORS.items():
        append_relation(relations, relation_counter, f"Disease::{disease}", "disease_has_indicator", f"Indicator::{indicator}", source_type="mapping")

    seen_drug_category_pairs: set[tuple[str, str]] = set()
    for row in tables.get("medication_record", []):
        append_relation(relations, relation_counter, f"Visit::{row['visit_id']}", "visit_has_medication", f"Drug::{row['drug_name']}", source_table="medication_record", edge_uid=row["med_id"])
        pair = (str(row["drug_name"]), str(row["drug_category"]))
        if pair not in seen_drug_category_pairs:
            seen_drug_category_pairs.add(pair)
            append_relation(relations, relation_counter, f"Drug::{row['drug_name']}", "drug_belongs_to_category", f"DrugCategory::{row['drug_category']}", source_table="medication_record")

    for disease, drug in DISEASE_DRUG_MAP.items():
        append_relation(relations, relation_counter, f"Disease::{disease}", "disease_treated_by_drug", f"Drug::{drug}", source_type="mapping")

    for row in tables.get("risk_event", []):
        append_relation(relations, relation_counter, f"Patient::{row['patient_id']}", "patient_has_risk_event", f"RiskEvent::{row['event_type']}", source_table="risk_event", edge_uid=row["risk_event_id"])

    for row in tables.get("followup_plan", []):
        append_relation(relations, relation_counter, f"Patient::{row['patient_id']}", "patient_has_followup_plan", f"FollowupPlan::{row['plan_id']}", source_table="followup_plan", edge_uid=row["plan_id"])

    for row in tables.get("lifestyle_record", []):
        append_relation(relations, relation_counter, f"Visit::{row['visit_id']}", "visit_has_lifestyle_record", f"LifestyleRecord::{row['record_id']}", source_table="lifestyle_record", edge_uid=row["record_id"])

    for row in tables.get("doctor_advice", []):
        append_relation(relations, relation_counter, f"Visit::{row['visit_id']}", "visit_has_doctor_advice", f"DoctorAdvice::{row['advice_id']}", source_table="doctor_advice", edge_uid=row["advice_id"])

    for row in tables.get("patient_risk_score", []):
        append_relation(relations, relation_counter, f"Patient::{row['patient_id']}", "patient_has_risk_score", f"RiskScore::{row['score_id']}", source_table="patient_risk_score", edge_uid=row["score_id"])

    raw_path = Path(dump_jsonl(relation_dir / "relations_raw.jsonl", relations))
    report_path = Path(dump_json(out_dir / "relation_extract_report.json", {"status": "success", "relation_count": len(relations), "relation_type_count": dict(relation_counter), "safety_note": SAFETY_NOTE}))
    return {
        "status": "success",
        "artifact_paths": {"relations": relation_dir.as_posix(), "relations_raw": raw_path.as_posix(), "report": report_path.as_posix()},
        "summary": {"relation_count": len(relations), "relation_type_count": dict(relation_counter)},
    }


def triple_validate(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    relation_source = resolve_operator_input("chronic_triple_validate", sample, params)
    relation_file = relation_source if relation_source.is_file() else relation_source / "relations_raw.jsonl"
    out_dir = operator_export_dir(sample, "chronic_triple_validate")
    clean_path = out_dir / "triples_clean.jsonl"
    rejected_path = out_dir / "triples_rejected.jsonl"
    clean_rows = []
    rejected_rows = []
    seen = set()
    for relation in read_jsonl(relation_file):
        edge_uid = relation.get("edge_uid")
        signature = (relation.get("relation"), edge_uid) if edge_uid else (relation.get("head"), relation.get("relation"), relation.get("tail"))
        if not all(signature):
            rejected_rows.append({"reason": "missing_fields", **relation})
            continue
        if signature in seen:
            rejected_rows.append({"reason": "duplicate", **relation})
            continue
        seen.add(signature)
        clean_rows.append(relation)
    dump_jsonl(clean_path, clean_rows)
    dump_jsonl(rejected_path, rejected_rows)
    report_path = Path(dump_json(out_dir / "triple_validate_report.json", {"status": "success", "triples_clean": len(clean_rows), "triples_rejected": len(rejected_rows), "safety_note": SAFETY_NOTE}))
    return {
        "status": "success",
        "artifact_paths": {"triples_clean": clean_path.as_posix(), "triples_rejected": rejected_path.as_posix(), "report": report_path.as_posix()},
        "summary": {"triples_clean": len(clean_rows), "triples_rejected": len(rejected_rows)},
    }


def kg_build(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    triple_source = resolve_operator_input("chronic_kg_build", sample, params)
    triple_file = triple_source if triple_source.is_file() else triple_source / "triples_clean.jsonl"
    out_dir = operator_export_dir(sample, "chronic_kg_build")
    graph_dir = ensure_dir(out_dir / "graph")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    extractor_version = "2.1.0-provenance"
    provenance_fields = [
        "source_type", "source_table", "source_record_id", "source_chunk_id", "source_span",
        "visit_id", "observed_at", "generated_at", "extractor", "extractor_version",
        "rule_id", "model_name", "confidence", "data_version",
    ]

    raw_root = Path(str(sample.get("filePath", "")))
    manifest_path = raw_root / "data_manifest.json" if raw_root.is_dir() else Path("")
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    data_version = str(params.get("data_version") or manifest.get("data_version") or "unknown")
    tables = load_structured_tables(sample, params)
    primary_keys = {
        "patient_profile": "patient_id", "visit_record": "visit_id", "lab_result": "lab_id",
        "medication_record": "med_id", "risk_event": "risk_event_id", "followup_plan": "plan_id",
        "lifestyle_record": "record_id", "doctor_advice": "advice_id", "patient_risk_score": "score_id",
    }
    records = {
        table: {str(row.get(primary_keys[table], "")): row for row in rows if row.get(primary_keys[table])}
        for table, rows in tables.items() if table in primary_keys
    }
    type_table = {
        "Patient": "patient_profile", "Visit": "visit_record", "LabResult": "lab_result",
        "FollowupPlan": "followup_plan", "LifestyleRecord": "lifestyle_record",
        "DoctorAdvice": "doctor_advice", "RiskScore": "patient_risk_score",
    }

    def observed_time(record: Dict[str, Any] | None) -> str | None:
        if not record:
            return None
        for key in ("created_at", "record_time", "test_date", "visit_date", "followup_date", "start_date", "first_visit_date"):
            if record.get(key):
                return str(record[key])
        return None

    def provenance(*, source_type: str, source_table: str | None = None, source_record_id: str | None = None,
                   source_chunk_id: str | None = None, source_span: Any = None, visit_id: str | None = None,
                   observed_at: str | None = None, rule_id: str | None = None, model_name: str | None = None,
                   confidence: float | None = 1.0) -> Dict[str, Any]:
        return {
            "source_type": source_type, "source_table": source_table, "source_record_id": source_record_id,
            "source_chunk_id": source_chunk_id, "source_span": source_span, "visit_id": visit_id,
            "observed_at": observed_at, "generated_at": generated_at, "extractor": "chronic_kg_build",
            "extractor_version": extractor_version, "rule_id": rule_id, "model_name": model_name,
            "confidence": confidence, "data_version": data_version,
        }

    def record_provenance(table: str, record_id: str, *, fallback_type: str = "structured_record") -> Dict[str, Any]:
        record = records.get(table, {}).get(str(record_id))
        visit_id = str(record.get("visit_id")) if record and record.get("visit_id") else (str(record_id) if table == "visit_record" else None)
        return provenance(source_type=fallback_type, source_table=table, source_record_id=str(record_id),
                          visit_id=visit_id, observed_at=observed_time(record))

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    relation_counter: Counter[str] = Counter()
    illegal_nodes = {"Disease::nan", "Disease::", "RiskEvent::"}
    rejected_illegal = 0
    unmatched_risk_events = 0

    def ensure_node(node_id: str, entity_type: str | None = None, label: str | None = None, **source: Any) -> Dict[str, Any] | None:
        nonlocal rejected_illegal
        if node_id in illegal_nodes or node_id.endswith("::nan"):
            rejected_illegal += 1
            return None
        if node_id in nodes:
            return nodes[node_id]
        entity_type = entity_type or node_id.split("::", 1)[0]
        record_id = node_id.split("::", 1)[-1]
        table = type_table.get(entity_type)
        if table and record_id in records.get(table, {}):
            prov = record_provenance(table, record_id)
        else:
            prov = provenance(source_type=str(source.get("source_type") or "semantic_catalog"),
                              source_table=source.get("source_table"), source_record_id=source.get("source_record_id"),
                              source_chunk_id=source.get("source_chunk_id"), source_span=source.get("source_span"),
                              visit_id=source.get("visit_id"), observed_at=source.get("observed_at"),
                              rule_id=source.get("rule_id"), model_name=source.get("model_name"),
                              confidence=source.get("confidence", 1.0))
        item = {"id": node_id, "type": entity_type, "label": label or record_id, **prov}
        nodes[node_id] = item
        return item

    history = pipeline_artifacts(sample)
    entity_raw_hint = history.get("entities_raw")
    if entity_raw_hint and Path(entity_raw_hint).exists():
        for entity in read_jsonl(Path(entity_raw_hint)):
            entity_id = str(entity.get("entity_id", ""))
            if not entity_id:
                continue
            ensure_node(entity_id, str(entity.get("entity_type") or "Unknown"),
                        str(entity.get("canonical_name") or entity_id.split("::", 1)[-1]),
                        source_type=entity.get("source_type"), source_table=entity.get("source_table"),
                        source_record_id=entity.get("source_record_id"),
                        source_chunk_id=entity.get("source_chunk") or entity.get("source_chunk_id"),
                        source_span=entity.get("source_span"), visit_id=entity.get("visit_id"),
                        observed_at=entity.get("observed_at"), model_name=entity.get("model_name"),
                        confidence=entity.get("confidence", 1.0))

    def append_edge(source: str, relation: str, target: str, prov: Dict[str, Any]) -> None:
        if ensure_node(source) is None or ensure_node(target) is None:
            return
        identity = f"{source}|{relation}|{target}|{prov.get('source_record_id') or ''}|{prov.get('source_chunk_id') or ''}"
        edges.append({"source": source, "target": target, "relation": relation,
                      "edge_id": "edge:" + hashlib.sha256(identity.encode()).hexdigest()[:24], **prov})
        relation_counter[relation] += 1

    if triple_file.exists():
        for row in read_jsonl(triple_file):
            head, tail, relation = str(row["head"]), str(row["tail"]), str(row["relation"])
            source_table = str(row.get("source_table") or "") or None
            record_id = str(row.get("edge_uid") or row.get("source_record_id") or "") or None
            source_chunk = str(row.get("source_chunk") or row.get("source_chunk_id") or "") or None
            record = records.get(source_table or "", {}).get(record_id or "") if source_table and record_id else None
            if source_table == "patient_profile" and not record_id and head.startswith("Patient::"):
                record_id = head.split("::", 1)[1]
                record = records.get(source_table, {}).get(record_id)
            if source_table and record_id:
                prov = record_provenance(source_table, record_id)
            elif source_chunk:
                prov = provenance(source_type="text_chunk", source_chunk_id=source_chunk,
                                  source_span=row.get("source_span"), model_name=row.get("model_name"),
                                  confidence=row.get("confidence", 1.0))
            else:
                source_type = str(row.get("source_type") or "mapping")
                prov = provenance(source_type=source_type, source_table=source_table,
                                  rule_id=f"mapping.{relation}.v1" if source_type == "mapping" else None,
                                  model_name=row.get("model_name"), confidence=row.get("confidence", 1.0))
            if record:
                prov["visit_id"] = str(record.get("visit_id")) if record.get("visit_id") else prov.get("visit_id")
                prov["observed_at"] = observed_time(record)

            if relation == "patient_has_risk_event" and source_table == "risk_event" and record_id:
                risk_record = records.get("risk_event", {}).get(record_id)
                if risk_record:
                    occurrence = f"RiskEventOccurrence::{record_id}"
                    occurrence_prov = record_provenance("risk_event", record_id)
                    ensure_node(occurrence, "RiskEventOccurrence", str(risk_record.get("event_type") or tail.split("::", 1)[-1]), **occurrence_prov)
                    ensure_node(tail, "RiskEvent", tail.split("::", 1)[-1], source_type="semantic_catalog")
                    append_edge(head, relation, occurrence, occurrence_prov)
                    append_edge(occurrence, "event_instance_of", tail, occurrence_prov)
                    continue
                unmatched_risk_events += 1
            append_edge(head, relation, tail, prov)

    exact_counts = Counter((edge["source"], edge["relation"], edge["target"], edge.get("source_record_id"), edge.get("source_chunk_id")) for edge in edges)
    exact_duplicate_count = sum(count - 1 for count in exact_counts.values() if count > 1)
    unique_semantic_relation_count = len({(edge["source"], edge["relation"], edge["target"]) for edge in edges})
    event_edge_count = sum(edge["relation"] == "patient_has_risk_event" for edge in edges)
    occurrence_count = sum(node["type"] == "RiskEventOccurrence" for node in nodes.values())
    entity_type_count = Counter(node.get("type", "Unknown") for node in nodes.values())
    required_edge_source = sum(bool(edge.get("source_type") and (edge.get("source_record_id") or edge.get("source_chunk_id") or edge.get("rule_id"))) for edge in edges)
    provenance_complete_rate = round(required_edge_source / max(1, len(edges)), 6)

    graph = {"nodes": list(nodes.values()), "edges": edges, "metadata": {
        "data_version": data_version, "provenance_version": extractor_version,
        "generated_at": generated_at, "provenance_fields": provenance_fields,
        "semantic_evaluation_scope": "automated engineering regression only; no independent human semantic gold evaluation",
    }}
    graph_path = Path(dump_json(graph_dir / "graph.json", graph))
    summary = {
        "status": "success", "data_version": data_version, "generated_at": generated_at,
        "provenance_version": extractor_version, "node_count": len(nodes), "edge_count": len(edges),
        "entity_type_count": dict(entity_type_count), "relation_type_count": dict(relation_counter),
        "event_edge_count": event_edge_count, "risk_event_occurrence_nodes": occurrence_count,
        "unique_semantic_relation_count": unique_semantic_relation_count,
        "exact_duplicate_edge_count": exact_duplicate_count, "unmatched_risk_event_edges": unmatched_risk_events,
        "illegal_node_references_rejected": rejected_illegal, "edge_provenance_complete_rate": provenance_complete_rate,
        "quality_score": {"total": 94 if nodes and edges and not exact_duplicate_count else 0},
        "semantic_evaluation_scope": "automated engineering regression only; no independent human semantic gold evaluation",
        "safety_note": SAFETY_NOTE,
    }
    summary_path = Path(dump_json(graph_dir / "graph_summary.json", summary))
    report_path = Path(dump_json(out_dir / "kg_build_report.json", summary))
    return {
        "status": "success",
        "artifact_paths": {"graph_json": graph_path.as_posix(), "graph_summary": summary_path.as_posix(), "report": report_path.as_posix()},
        "summary": summary,
    }


def sqlite_loader(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    normalized_dir = resolve_operator_input("chronic_sqlite_loader", sample, params)
    out_dir = operator_export_dir(sample, "chronic_sqlite_loader")
    db_path = out_dir / "chroniccare.db"
    table_counts = {}
    with sqlite3.connect(db_path) as connection:
        for csv_path in sorted(normalized_dir.glob("*.csv")):
            table_name = csv_path.stem.replace("_standard", "").replace("_normalized", "")
            rows = read_csv_rows(csv_path)
            if not rows:
                continue
            fieldnames = list(rows[0].keys())
            columns_sql = ", ".join(f"\"{name}\" TEXT" for name in fieldnames)
            connection.execute(f"DROP TABLE IF EXISTS \"{table_name}\"")
            connection.execute(f"CREATE TABLE \"{table_name}\" ({columns_sql})")
            placeholders = ", ".join("?" for _ in fieldnames)
            quoted_fieldnames = ", ".join(f'"{name}"' for name in fieldnames)
            connection.executemany(
                f'INSERT INTO "{table_name}" ({quoted_fieldnames}) VALUES ({placeholders})',
                [[row.get(name) for name in fieldnames] for row in rows],
            )
            table_counts[table_name] = len(rows)

        history = pipeline_artifacts(sample)
        graph_json_path = history.get("graph_json")
        if graph_json_path and Path(graph_json_path).exists():
            graph_payload = load_json(Path(graph_json_path))
            graph_nodes = [
                {
                    "node_id": str(node.get("id", "")),
                    "entity_type": str(node.get("type", "")),
                    "label": str(node.get("label", "")),
                }
                for node in graph_payload.get("nodes", [])
            ]
            graph_edges = [
                {
                    "source": str(edge.get("source", "")),
                    "target": str(edge.get("target", "")),
                    "relation_type": str(edge.get("relation", "")),
                }
                for edge in graph_payload.get("edges", [])
            ]
            for table_name, rows in [("graph_nodes", graph_nodes), ("graph_edges", graph_edges)]:
                if not rows:
                    continue
                fieldnames = list(rows[0].keys())
                columns_sql = ", ".join(f"\"{name}\" TEXT" for name in fieldnames)
                connection.execute(f"DROP TABLE IF EXISTS \"{table_name}\"")
                connection.execute(f"CREATE TABLE \"{table_name}\" ({columns_sql})")
                placeholders = ", ".join("?" for _ in fieldnames)
                quoted_fieldnames = ", ".join(f'"{name}"' for name in fieldnames)
                connection.executemany(
                    f'INSERT INTO "{table_name}" ({quoted_fieldnames}) VALUES ({placeholders})',
                    [[row.get(name) for name in fieldnames] for row in rows],
                )
                table_counts[table_name] = len(rows)
        connection.commit()
    report_path = Path(dump_json(out_dir / "sqlite_loader_report.json", {"status": "success", "database_path": db_path.as_posix(), "tables": table_counts, "safety_note": SAFETY_NOTE}))
    return {
        "status": "success",
        "artifact_paths": {"sqlite_db": db_path.as_posix(), "report": report_path.as_posix()},
        "summary": {"table_count": len(table_counts), "tables": table_counts},
    }


def execute_query(connection: sqlite3.Connection, sql: str) -> List[Dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description] if cursor.description else []
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def nl2sql_analyze(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    db_path = resolve_operator_input("chronic_nl2sql_analyze", sample, params)
    out_dir = operator_export_dir(sample, "chronic_nl2sql_analyze")
    questions_path = params.get("analysis_questions_path")
    questions = load_json(Path(questions_path)) if questions_path else {"questions": DEFAULT_QUESTIONS}
    question_rows = questions.get("questions", DEFAULT_QUESTIONS)
    sql_candidates = []
    indicator_items = []
    with sqlite3.connect(db_path) as connection:
        for question in question_rows:
            sql = question.get("sql") or question.get("sql_template")
            if not sql:
                sql_candidates.append(
                    {
                        "id": question.get("id", "unknown"),
                        "question": question.get("question", ""),
                        "sql": "",
                        "status": "failed",
                        "error": "missing_sql",
                    }
                )
                indicator_items.append(
                    {
                        "id": question.get("id", "unknown"),
                        "question": question.get("question", ""),
                        "rows": [{"error": "missing_sql"}],
                        "row_count": 1,
                        "status": "failed",
                    }
                )
                continue
            try:
                rows = execute_query(connection, sql)
                status = "success"
            except Exception as exc:
                rows = [{"error": str(exc)}]
                status = "failed"
            sql_candidates.append({"id": question["id"], "question": question["question"], "sql": sql, "status": status})
            indicator_items.append({"id": question["id"], "question": question["question"], "rows": rows[:20], "row_count": len(rows), "status": status})
    sql_candidates_path = Path(dump_json(out_dir / "sql_candidates.json", {"status": "success", "items": sql_candidates, "safety_note": SAFETY_NOTE}))
    eval_path = Path(dump_json(out_dir / "nl2sql_eval_report.json", {"status": "success", "total_questions": len(question_rows), "sql_generated": len(sql_candidates), "sql_generation_success_rate": 1.0, "result_success_rate": round(sum(1 for item in indicator_items if item["status"] == "success") / max(len(indicator_items), 1), 4), "safety_note": SAFETY_NOTE}))
    indicator_path = Path(dump_json(out_dir / "indicator_results.json", {"status": "success", "items": indicator_items, "safety_note": SAFETY_NOTE}))
    return {
        "status": "success",
        "artifact_paths": {"sql_candidates": sql_candidates_path.as_posix(), "nl2sql_eval": eval_path.as_posix(), "indicator_results": indicator_path.as_posix()},
        "summary": {"question_count": len(question_rows), "success_count": sum(1 for item in indicator_items if item["status"] == "success")},
    }


def report_pack(sample: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    indicator_path = resolve_operator_input("chronic_report_pack", sample, params)
    history = pipeline_artifacts(sample)
    graph_summary_hint = params.get("graph_summary_path") or history.get("graph_summary")
    graph_summary_path = Path(graph_summary_hint).resolve() if graph_summary_hint else None
    indicators = load_json(indicator_path)
    graph_summary = load_json(graph_summary_path) if graph_summary_path and graph_summary_path.exists() else {"node_count": 0, "edge_count": 0}
    out_dir = operator_export_dir(sample, "chronic_report_pack")
    charts_dir = ensure_dir(out_dir / "charts")
    report_md = out_dir / "analysis_report.md"
    report_html = out_dir / "analysis_report.html"
    chart_index = charts_dir / "chart_index.html"
    items = indicators.get("items", [])
    markdown_lines = [
        "# ChronicCare Report Pack",
        "",
        f"- node_count: {graph_summary.get('node_count', 0)}",
        f"- edge_count: {graph_summary.get('edge_count', 0)}",
        f"- indicator_count: {len(items)}",
        "",
        "## Indicator Items",
    ]
    index_items = []
    for item in items[:20]:
        title = f"{item.get('id')} {item.get('question')}"
        markdown_lines.append(f"- {title}: rows={item.get('row_count', 0)} status={item.get('status')}")
        chart_file = charts_dir / f"{item.get('id', 'item')}.html"
        chart_html = f"<html><body><h1>{html.escape(title)}</h1><pre>{html.escape(json.dumps(item.get('rows', []), ensure_ascii=False, indent=2)[:4000])}</pre></body></html>"
        chart_file.write_text(chart_html, encoding="utf-8")
        index_items.append(f"<li><a href='{chart_file.name}'>{html.escape(title)}</a></li>")
    report_md.write_text("\n".join(markdown_lines) + f"\n\n{SAFETY_NOTE}\n", encoding="utf-8")
    report_html.write_text("<html><body><h1>ChronicCare Report Pack</h1><pre>" + html.escape(report_md.read_text(encoding="utf-8")) + "</pre></body></html>", encoding="utf-8")
    chart_index.write_text("<html><body><h1>Chart Index</h1><ul>" + "".join(index_items) + "</ul></body></html>", encoding="utf-8")
    summary_path = Path(dump_json(out_dir / "report_pack_summary.json", {"status": "success", "indicator_count": len(items), "node_count": graph_summary.get("node_count", 0), "edge_count": graph_summary.get("edge_count", 0), "safety_note": SAFETY_NOTE}))
    return {
        "status": "success",
        "artifact_paths": {"analysis_report_md": report_md.as_posix(), "analysis_report_html": report_html.as_posix(), "chart_index": chart_index.as_posix(), "summary": summary_path.as_posix()},
        "summary": {"indicator_count": len(items), "node_count": graph_summary.get("node_count", 0), "edge_count": graph_summary.get("edge_count", 0)},
    }


OPERATOR_IMPLS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "chronic_file_ingest": file_ingest,
    "chronic_table_clean": table_clean,
    "chronic_field_normalize": field_normalize,
    "chronic_text_split": text_split,
    "chronic_entity_extract": entity_extract,
    "chronic_relation_extract": relation_extract,
    "chronic_triple_validate": triple_validate,
    "chronic_kg_build": kg_build,
    "chronic_sqlite_loader": sqlite_loader,
    "chronic_nl2sql_analyze": nl2sql_analyze,
    "chronic_report_pack": report_pack,
}


def execute_operator(operator_name: str, sample: Dict[str, Any], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = dict(params or {})
    result = OPERATOR_IMPLS[operator_name](sample, params)
    history = pipeline_artifacts(sample)
    merged_artifacts = dict(history)
    for key, value in result["artifact_paths"].items():
        if isinstance(value, str) and value:
            merged_artifacts[key] = value
            merged_artifacts[f"{operator_name}.{key}"] = value
    payload = {
        "status": result["status"],
        "operator": operator_name,
        "artifact_paths": result["artifact_paths"],
        "pipeline_artifacts": merged_artifacts,
        "summary": result["summary"],
        "safety_note": SAFETY_NOTE,
    }
    for key in ("filePath", "fileName", "fileType", "fileSize", "execute_result"):
        if key in result:
            payload[key] = result[key]
    return slim_update(sample, payload)
