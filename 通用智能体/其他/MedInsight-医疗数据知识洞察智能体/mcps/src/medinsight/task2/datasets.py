from __future__ import annotations

import json
import csv
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


TASK2_RAW_DIR = Path("data/task2/raw/cblue")
LABELED_SPLITS = ["train", "dev", "test"]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at {path}:{line_number}: {exc}") from exc
    return rows


def read_json_with_comment_header(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    return json.loads("\n".join(lines))


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def resolve_splits(split: str, allowed: list[str] | None = None) -> list[str]:
    allowed = allowed or LABELED_SPLITS
    normalized = split.strip().lower().replace("_", "+")
    if normalized in {"all", "labeled", "labeled+all"}:
        return list(allowed)
    parts = [item.strip() for item in normalized.replace(",", "+").split("+") if item.strip()]
    if not parts:
        return ["train"]
    invalid = [item for item in parts if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported split expression: {split}; invalid parts: {invalid}")
    return parts


def load_json_splits(raw_dir: Path, folder: str, filename_template: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in resolve_splits(split):
        rows.extend(read_json(raw_dir / folder / filename_template.format(split=item)))
    return limit_rows(rows, limit)


def load_jsonl_splits(raw_dir: Path, folder: str, filename_template: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in resolve_splits(split):
        rows.extend(read_jsonl(raw_dir / folder / filename_template.format(split=item)))
    return limit_rows(rows, limit)


def load_cmeee_split(raw_dir: Path = TASK2_RAW_DIR, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    return load_json_splits(raw_dir, "CMeEE-V2", "CMeEE-V2_{split}.json", split, limit)


def load_cmeie_split(raw_dir: Path = TASK2_RAW_DIR, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    return load_jsonl_splits(raw_dir, "CMeIE-V2", "CMeIE-V2_{split}.jsonl", split, limit)


def load_cmeie_schema(raw_dir: Path = TASK2_RAW_DIR) -> list[dict[str, Any]]:
    return read_json_with_comment_header(raw_dir / "CMeIE-V2" / "53_schemas.json")


def load_chip_cdn_split(raw_dir: Path = TASK2_RAW_DIR, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    return load_json_splits(raw_dir, "CHIP-CDN", "CHIP-CDN_{split}.json", split, limit)


def load_cmedcausal_split(raw_dir: Path = TASK2_RAW_DIR, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    return load_json_splits(raw_dir, "CMedCausal", "CMedCausal_{split}.json", split, limit)


def load_icd10_terms(raw_dir: Path = TASK2_RAW_DIR, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    path = raw_dir / "CHIP-CDN" / "国际疾病分类 ICD-10北京临床版v601.xlsx"
    for row in read_xlsx_rows(path):
        code = clean_cell(row[0]) if len(row) > 0 else ""
        name = clean_cell(row[1]) if len(row) > 1 else ""
        if code and name:
            rows.append({"code": code, "name": name})
    return limit_rows(rows, limit)


def load_surgery_lexicon(raw_dir: Path = TASK2_RAW_DIR, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    path = raw_dir / "CHIP-CDN" / "手术标注词2500.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            raw_name, code, standard_name = [clean_cell(item) for item in row[:3]]
            if raw_name and code and standard_name:
                rows.append({"raw_name": raw_name, "code": code, "standard_name": standard_name})
    return limit_rows(rows, limit)


def clean_cell(value: Any) -> str:
    return str(value or "").strip()


def read_xlsx_rows(path: Path) -> list[list[str]]:
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        shared_strings = read_xlsx_shared_strings(workbook, namespace)
        sheet_names = [name for name in workbook.namelist() if name.startswith("xl/worksheets/sheet")]
        if not sheet_names:
            return []
        sheet_xml = workbook.read(sorted(sheet_names)[0])
    root = ET.fromstring(sheet_xml)
    rows: list[list[str]] = []
    for row in root.findall(".//main:sheetData/main:row", namespace):
        values: list[str] = []
        for cell in row.findall("main:c", namespace):
            cell_type = cell.attrib.get("t")
            value_node = cell.find("main:v", namespace)
            value = value_node.text if value_node is not None else ""
            if cell_type == "s" and value:
                value = shared_strings[int(value)]
            values.append(clean_cell(value))
        rows.append(values)
    return rows


def read_xlsx_shared_strings(workbook: zipfile.ZipFile, namespace: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("main:si", namespace):
        parts = [text.text or "" for text in item.findall(".//main:t", namespace)]
        strings.append("".join(parts))
    return strings


def dataset_manifest(raw_dir: Path = TASK2_RAW_DIR) -> dict[str, Any]:
    return {
        "root": str(raw_dir),
        "datasets": {
            "CMeEE-V2": {
                "task": "medical named entity recognition",
                "files": {
                    "train": str(raw_dir / "CMeEE-V2" / "CMeEE-V2_train.json"),
                    "dev": str(raw_dir / "CMeEE-V2" / "CMeEE-V2_dev.json"),
                    "test": str(raw_dir / "CMeEE-V2" / "CMeEE-V2_test.json"),
                },
            },
            "CMeIE-V2": {
                "task": "medical entity relation extraction",
                "files": {
                    "schema": str(raw_dir / "CMeIE-V2" / "53_schemas.json"),
                    "train": str(raw_dir / "CMeIE-V2" / "CMeIE-V2_train.jsonl"),
                    "dev": str(raw_dir / "CMeIE-V2" / "CMeIE-V2_dev.jsonl"),
                    "test": str(raw_dir / "CMeIE-V2" / "CMeIE-V2_test.jsonl"),
                },
            },
            "CHIP-CDN": {
                "task": "clinical terminology normalization",
                "role": "entity normalization, alias enhancement, ICD-10 code grounding and surgery lexicon grounding",
                "files": {
                    "train": str(raw_dir / "CHIP-CDN" / "CHIP-CDN_train.json"),
                    "dev": str(raw_dir / "CHIP-CDN" / "CHIP-CDN_dev.json"),
                    "test": str(raw_dir / "CHIP-CDN" / "CHIP-CDN_test.json"),
                    "icd10": str(raw_dir / "CHIP-CDN" / "国际疾病分类 ICD-10北京临床版v601.xlsx"),
                    "surgery_lexicon": str(raw_dir / "CHIP-CDN" / "手术标注词2500.tsv"),
                },
            },
            "CMedCausal": {
                "task": "medical causal relation extraction",
                "role": "causal relation enhancement; original relation_code is preserved in graph edge properties",
                "files": {
                    "train": str(raw_dir / "CMedCausal" / "CMedCausal_train.json"),
                    "dev": str(raw_dir / "CMedCausal" / "CMedCausal_dev.json"),
                    "test": str(raw_dir / "CMedCausal" / "CMedCausal_test.json"),
                    "unlabel": str(raw_dir / "CMedCausal" / "CMedCausal_unlabel_1k.json"),
                },
            },
        },
    }
