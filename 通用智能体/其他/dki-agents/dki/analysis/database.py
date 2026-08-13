"""分析数据库构建（SQLite）：任务一结构化产出 + 任务二图谱 → 统一分析底座。"""

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from dki.analysis.nl2sql_sqlite import bounded_execution
from dki.analysis.nl2sql_sqlite import connect_readonly
from dki.kg.schema import relation_zh
from dki.kg.store import KnowledgeGraph

_SCHEMA = """
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS visits;
DROP TABLE IF EXISTS kg_entities;
DROP TABLE IF EXISTS kg_triples;
CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,   -- 患者ID
    name TEXT,                     -- 姓名
    gender TEXT,                   -- 性别（男/女）
    age INTEGER,                   -- 年龄
    region TEXT                    -- 地区
);
CREATE TABLE visits (
    visit_id TEXT PRIMARY KEY,     -- 就诊ID（=病历文档ID）
    patient_id TEXT,               -- 患者ID
    visit_date TEXT,               -- 就诊日期 YYYY-MM-DD
    department TEXT,               -- 科室
    disease TEXT,                  -- 主诊断疾病
    drug_prescribed TEXT,          -- 主要处方药物
    exam TEXT,                     -- 主要检查
    cost REAL,                     -- 费用（元）
    los_days INTEGER               -- 住院天数（0=门诊）
);
CREATE TABLE kg_entities (
    name TEXT,                     -- 实体名
    type TEXT,                     -- 实体类型 Disease/Symptom/Drug/Exam/Treatment/Department
    mention_count INTEGER          -- 语料中提及次数
);
CREATE TABLE kg_triples (
    head TEXT, head_type TEXT,
    relation TEXT,                 -- 关系（英文标识）
    relation_zh TEXT,              -- 关系中文名
    tail TEXT, tail_type TEXT,
    confidence REAL,               -- 置信度
    count INTEGER                  -- 观察次数
);
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_path(db_path: Path) -> Path:
    return Path(db_path).with_suffix(".provenance.json")


def database_matches_manifest(
    db_path: Path,
    manifest_path: Path,
    kg_path: Optional[Path] = None,
) -> bool:
    """Return true only when SQLite matches the exact Task 1/Task 2 inputs."""
    receipt = _provenance_path(db_path)
    if not Path(db_path).is_file() or not receipt.is_file() or not Path(manifest_path).is_file():
        return False
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("database_sha256") != _sha256(Path(db_path)):
            return False
        if payload.get("task1_manifest_sha256") != _sha256(Path(manifest_path)):
            return False
        if kg_path is not None:
            kg_path = Path(kg_path)
            if not kg_path.is_file():
                return False
            if payload.get("task2_kg_sha256") != _sha256(kg_path):
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def build_database(
    db_path: Path,
    visits_csv: Optional[Path] = None,
    patients_csv: Optional[Path] = None,
    kg: Optional[KnowledgeGraph] = None,
    manifest_path: Optional[Path] = None,
    kg_path: Optional[Path] = None,
) -> Path:
    """幂等重建分析数据库。

    New production calls must provide ``manifest_path`` so Task 3 consumes the
    Task 1 snapshot.  Explicit CSV arguments remain for backwards-compatible
    library callers and fixtures.
    """
    if manifest_path is not None:
        from dki.pipeline.artifacts import load_task1_manifest, task1_table_paths

        tables = task1_table_paths(load_task1_manifest(manifest_path))
        visits_csv = tables["visits"]
        patients_csv = tables["patients"]
    if visits_csv is None or patients_csv is None:
        raise ValueError("build_database 需要 Task 1 manifest 或 patients/visits CSV")
    if kg_path is not None and not Path(kg_path).is_file():
        raise FileNotFoundError(f"任务二知识图谱不存在: {kg_path}")
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        with open(patients_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        conn.executemany(
            "INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?)",
            [(r["patient_id"], r["name"], r["gender"], int(r["age"]), r["region"]) for r in rows],
        )
        with open(visits_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        conn.executemany(
            "INSERT OR REPLACE INTO visits VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (r["visit_id"], r["patient_id"], r["visit_date"], r["department"],
                 r["disease"], r["drug_prescribed"], r["exam"],
                 float(r["cost"]), int(r["los_days"]))
                for r in rows
            ],
        )
        if kg is not None:
            conn.executemany(
                "INSERT INTO kg_entities VALUES (?,?,?)",
                [(n, e["type"], e["mention_count"]) for n, e in kg.entities.items()],
            )
            conn.executemany(
                "INSERT INTO kg_triples VALUES (?,?,?,?,?,?,?,?)",
                [
                    (t["head"], t["head_type"], t["relation"], relation_zh(t["relation"]),
                     t["tail"], t["tail_type"], t["confidence"], t["count"])
                    for t in kg.triples
                ],
            )
        conn.commit()
    finally:
        conn.close()
    if manifest_path is not None:
        manifest = Path(manifest_path)
        receipt = {
            "schema_version": 1,
            "task1_manifest_path": str(manifest),
            "task1_manifest_sha256": _sha256(manifest),
            "database_path": str(db_path),
            "database_sha256": _sha256(db_path),
        }
        if kg_path is not None:
            kg_path = Path(kg_path)
            receipt["task2_kg_path"] = str(kg_path)
            receipt["task2_kg_sha256"] = _sha256(kg_path)
        provenance = _provenance_path(db_path)
        provenance.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=provenance.parent,
            prefix=f".{provenance.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.replace(temporary, provenance)
        finally:
            temporary.unlink(missing_ok=True)
    return db_path


def get_schema_description(db_path: Path) -> str:
    """供 NL2SQL LLM 提示使用的 schema 说明（建表语句+中文注释+示例行）。"""
    conn = connect_readonly(Path(db_path))
    try:
        lines = [_SCHEMA.strip(), "", "-- 示例数据："]
        for table in ("patients", "visits", "kg_triples"):
            cur = conn.execute(f"SELECT * FROM {table} LIMIT 2")
            cols = [d[0] for d in cur.description]
            lines.append(f"-- {table}({', '.join(cols)})")
            for row in cur.fetchall():
                lines.append(f"--   {row}")
        lines.append("-- kg_triples.relation 枚举：HAS_SYMPTOM临床表现 / TREATED_BY_DRUG治疗药物 / "
                     "TREATED_BY治疗方式 / DIAGNOSED_BY诊断检查 / HAS_COMPLICATION并发症 / "
                     "CONTRAINDICATED_FOR用药禁忌 / BELONGS_TO_DEPT所属科室")
        return "\n".join(lines)
    finally:
        conn.close()


_FORBIDDEN = ("pragma", "attach", "detach", "insert", "update", "delete", "drop",
              "create", "alter", "replace", "vacuum")

MAX_ROWS = 200


def run_sql(sql: str, db_path: Path) -> Tuple[List[str], List[Tuple]]:
    """只读执行 SELECT（白名单校验 + 行数上限）。"""
    stripped = sql.strip().rstrip(";").strip()
    low = stripped.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise PermissionError("仅允许 SELECT/WITH 查询")
    if ";" in stripped:
        raise PermissionError("不允许多语句")
    for kw in _FORBIDDEN:
        # 词边界级检查，避免误伤列名（如 created）
        import re

        if re.search(rf"\b{kw}\b", low):
            raise PermissionError(f"不允许的关键字: {kw}")
    conn = connect_readonly(Path(db_path))
    try:
        with bounded_execution(conn):
            cur = conn.execute(stripped)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(MAX_ROWS + 1)
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        if truncated:
            rows.append(tuple(["...（已截断，仅显示前200行）"] + [""] * (len(columns) - 1)))
        return columns, [tuple(r) for r in rows]
    finally:
        conn.close()
