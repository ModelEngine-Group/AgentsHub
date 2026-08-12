# -*- coding: utf-8 -*-
"""DataMate 数据集注册与文件同步辅助函数。"""
import argparse
import csv
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from scripts.runtime_env import get_runtime_secret, load_runtime_env


ROOT = Path(__file__).resolve().parents[3]


def _runtime_value(name: str, default: str = "") -> str:
    """读取运行配置，避免把服务器路径写死在代码中。"""

    if name not in os.environ:
        load_runtime_env(root=ROOT, keys=[name])
    return os.environ.get(name, default)


def dataset_volume() -> str:
    """返回 DataMate 数据集文件目录。"""

    value = _runtime_value("CCF_DATASET_VOLUME")
    if not value:
        raise RuntimeError("CCF_DATASET_VOLUME is required. Set it in .env.runtime or the process environment.")
    return value

SOURCE_ROOT_CANDIDATES = [
    os.environ.get("CCF_DATA_ROOT"),
    str(ROOT / "data" / "input"),
]

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
EXPECTED_ABSENT = ["由HIS系统自动导出", "系统自动生成", "填表时间", "填表人", "@护士小王", "图片链接已失效"]


def resolve_source_root(explicit: Optional[str] = None) -> Path:
    candidates = [explicit] if explicit else []
    candidates.extend(x for x in SOURCE_ROOT_CANDIDATES if x)
    for item in candidates:
        path = Path(item)
        if path.exists():
            return path
    raise FileNotFoundError(
        "No dataset source root found. Set CCF_DATA_ROOT or pass --source-root. "
        "Set CCF_DATA_ROOT to a directory containing TXT, CSV, JSON, or JSONL files."
    )


def first_existing(root: Path, relatives: Iterable[str]) -> Optional[Path]:
    for rel in relatives:
        path = root / rel
        if path.exists():
            return path
    return None


def read_text_with_fallback(path: Path) -> str:
    last = None
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Cannot read {path}: {last}")


def read_csv_rows(path: Path, limit: int) -> List[Dict[str, str]]:
    last = None
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    clean = {str(k or "").strip(): str(v or "").strip() for k, v in row.items()}
                    if any(clean.values()):
                        rows.append(clean)
                    if len(rows) >= limit:
                        return rows
                return rows
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Cannot parse CSV {path}: {last}")


def load_json_texts(path: Path, limit: int) -> List[str]:
    text = read_text_with_fallback(path)
    stripped = text.strip()
    records = []

    def extract(item) -> str:
        if not isinstance(item, dict):
            return str(item or "").strip()
        for key in ("text", "content", "query", "original_text", "sentence", "normalized_result"):
            value = str(item.get(key, "") or "").strip()
            if value:
                return value
        values = []
        for value in item.values():
            if isinstance(value, (str, int, float)):
                text_value = str(value).strip()
                if text_value:
                    values.append(text_value)
        return "；".join(values[:4])

    if stripped.startswith("["):
        data = json.loads(stripped)
        for item in data:
            value = extract(item)
            if value:
                records.append(value)
            if len(records) >= limit:
                break
    else:
        for line in stripped.splitlines():
            item = json.loads(line)
            value = extract(item)
            if value:
                records.append(value)
            if len(records) >= limit:
                break
    return records


def load_clinical_text_rows(root: Path, limit: int) -> Tuple[List[Dict[str, str]], Optional[str]]:
    clinical_dir = root / "RuijinDiabetes"
    if not clinical_dir.exists():
        return [], None
    rows: List[Dict[str, str]] = []
    for path in sorted(clinical_dir.glob("*.txt")):
        raw = read_text_with_fallback(path)
        compact = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
        if len(compact) < 80:
            continue
        rows.append({
            "department": "糖尿病专病",
            "title": f"瑞金糖尿病病历 {path.stem}",
            "ask": compact[:2600],
            "answer": compact[2600:4200],
        })
        if len(rows) >= limit:
            break
    return rows, str(clinical_dir) if rows else None


def fallback_rows() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str], Dict[str, str]]:
    dialogue = [{
        "department": "内科",
        "title": "高血压伴头晕如何处理",
        "ask": "患者反复头晕3个月，血压最高180/110mmHg，担心是否需要调整用药。",
        "answer": "建议规律监测血压，完善肾功能和动态血压检查，在医生指导下调整降压方案。",
    }]
    qa = [{
        "question_id": "fallback_q1",
        "content": "空腹血糖一直在9-11mmol/L之间，二甲双胍治疗效果不佳怎么办？",
    }]
    cmeee = ["2型糖尿病患者应关注糖化血红蛋白、空腹血糖和心血管危险因素。"]
    sources = {"fallback": "built-in fallback text"}
    return dialogue, qa, cmeee, sources


def load_source_material(
    root: Path,
    dialogue_limit: int = 64,
    qa_limit: int = 256,
    cmeee_limit: int = 128,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str], Dict[str, str]]:
    sources: Dict[str, str] = {}
    dialogue_path = first_existing(root, [
        "ChineseMedicalDialogue/Data_数据/IM_内科/内科5000-33000.csv",
        "ChineseMedicalDialogue/Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv",
        "ChineseMedicalDialogue/样例_内科5000-6000.csv",
        "ChineseMedicalDialogue/Data_数据/IM_内科/内科.txt",
    ])
    question_path = first_existing(root, ["cMedQA2/question.csv", "cMedQA2/questions.csv"])
    cmeee_path = first_existing(root, [
        "CBLUE_data/CMeEE-V2/CMeEE-V2_dev.json",
        "CBLUE_data/CMeIE/CMeIE_dev.jsonl",
        "CBLUE_data/CDN/CHIP-CDN_dev_1007.json",
    ])

    if not dialogue_path and not question_path and not cmeee_path:
        return fallback_rows()

    dialogue = []
    if dialogue_path and dialogue_path.suffix.lower() == ".csv":
        dialogue = read_csv_rows(dialogue_path, dialogue_limit)
        sources["dialogue"] = str(dialogue_path)
    elif dialogue_path:
        raw = read_text_with_fallback(dialogue_path)
        dialogue = [{"department": "内科", "title": "真实内科文本", "ask": raw[:500], "answer": raw[500:1000]}]
        sources["dialogue"] = str(dialogue_path)

    clinical_rows, clinical_source = load_clinical_text_rows(root, dialogue_limit)
    if clinical_rows:
        dialogue = clinical_rows + dialogue
        sources["ruijin_diabetes"] = clinical_source or "RuijinDiabetes"

    qa = []
    if question_path:
        qa = read_csv_rows(question_path, qa_limit)
        sources["cMedQA2_question"] = str(question_path)

    cmeee = []
    if cmeee_path:
        cmeee = load_json_texts(cmeee_path, cmeee_limit)
        sources["cblue_entity_or_relation"] = str(cmeee_path)

    if not dialogue and not qa and not cmeee:
        return fallback_rows()
    return dialogue, qa, cmeee, sources


def inject_noise(text: str, idx: int, include_html: bool = False) -> str:
    fixed = [
        "由HIS系统自动导出 @护士小王",
        "填表时间：2026-06-07 填表人：实习护士陈某",
        "图片链接已失效",
    ]
    extra = "<p>系统自动生成</p> https://hospital.example.invalid/record/123" if include_html else "系统自动生成"
    return f"{text}\n{fixed[idx % len(fixed)]}\n{extra}"


def _cycle(items: List, index: int):
    if not items:
        raise ValueError("Cannot cycle empty source list")
    return items[index % len(items)]


def build_files(
    output_dir: Path,
    source_root: Optional[str] = None,
    text_file_count: int = 8,
    csv_file_count: int = 4,
    json_file_count: int = 4,
    records_per_text_file: int = 2,
    rows_per_structured_file: int = 3,
) -> Dict:
    root = resolve_source_root(source_root)
    needed_dialogue = max(8, text_file_count * records_per_text_file)
    needed_qa = max(32, (csv_file_count + json_file_count) * rows_per_structured_file)
    needed_cmeee = max(16, json_file_count * rows_per_structured_file)
    dialogue, qa, cmeee, sources = load_source_material(
        root,
        dialogue_limit=needed_dialogue,
        qa_limit=needed_qa,
        cmeee_limit=needed_cmeee,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = []
    text_record_count = 0
    text_source = dialogue or [
        {"department": "内科", "title": f"真实问诊样本{i+1}", "ask": row.get("content", ""), "answer": ""}
        for i, row in enumerate(qa or [])
    ]
    for file_idx in range(text_file_count):
        chunks = []
        for inner_idx in range(records_per_text_file):
            source_index = file_idx * records_per_text_file + inner_idx
            row = _cycle(text_source, source_index)
            record_idx = source_index + 1
            ask = row.get("ask") or row.get("question") or row.get("content") or ""
            answer = row.get("answer") or row.get("回复") or ""
            title = row.get("title") or f"真实问诊样本{record_idx}"
            body = (
                f"病例{inner_idx + 1}\n"
                f"来源科室：{row.get('department', '未知科室')}\n"
                f"标题：{title}\n"
                f"主诉/问题：{ask}\n"
                f"医生建议：{answer}"
            )
            chunks.append(inject_noise(body, record_idx, include_html=True))
            text_record_count += 1
        path = output_dir / f"real_text_{file_idx + 1:03d}.txt"
        path.write_text("\n====\n".join(chunks), encoding="utf-8")
        text_files.append(path)

    csv_files = []
    csv_row_count = 0
    qa_source = qa or [{"question_id": "fallback", "content": "患者头痛伴恶心，担心高血压。"}]
    for file_idx in range(csv_file_count):
        rows = []
        for row_idx in range(rows_per_structured_file):
            source_index = file_idx * rows_per_structured_file + row_idx
            idx = source_index + 1
            row = _cycle(qa_source, source_index)
            content = row.get("content", "")
            rows.append({
                "record_id": f"qa_{idx:05d}",
                "title": f"真实问答样本{idx}",
                "content": inject_noise(content, idx, include_html=False),
                "diagnosis": "待医生结合检查判断",
                "note": "系统自动生成；填表时间：2026-06-07",
            })
            csv_row_count += 1
        path = output_dir / f"real_table_{file_idx + 1:03d}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["record_id", "title", "content", "diagnosis", "note"])
            writer.writeheader()
            writer.writerows(rows)
        csv_files.append(path)

    json_source = cmeee or [r.get("content", "") for r in qa_source[:6]]
    json_source_label = (
        sources.get("cblue_entity_or_relation")
        or sources.get("cMedQA2_question")
        or "built-in fallback text"
    )
    json_files = []
    json_record_count = 0
    for file_idx in range(json_file_count):
        records = []
        for row_idx in range(rows_per_structured_file):
            source_index = file_idx * rows_per_structured_file + row_idx
            idx = source_index + 1
            text = _cycle(json_source, source_index)
            records.append({
                "id": f"json_{idx:05d}",
                "content": inject_noise(text, idx, include_html=False),
                "source": json_source_label,
                "entity_hint": "保留下游任务二可抽取实体关系的医学句子",
            })
            json_record_count += 1
        path = output_dir / f"real_records_{file_idx + 1:03d}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        json_files.append(path)

    files_by_type = {
        "text": [str(p) for p in text_files],
        "csv": [str(p) for p in csv_files],
        "json": [str(p) for p in json_files],
    }
    files = {
        "text": str(text_files[0]) if text_files else "",
        "csv": str(csv_files[0]) if csv_files else "",
        "json": str(json_files[0]) if json_files else "",
    }

    manifest = {
        "source_root": str(root),
        "sources": sources,
        "output_dir": str(output_dir),
        "files": files,
        "files_by_type": files_by_type,
        "expected_absent": EXPECTED_ABSENT,
        "file_counts": {
            "text": len(text_files),
            "csv": len(csv_files),
            "json": len(json_files),
            "mixed_total": len(text_files) + len(csv_files) + len(json_files),
        },
        "record_counts": {
            "text_records": text_record_count,
            "csv_rows": csv_row_count,
            "json_records": json_record_count,
        },
        "note": "Real medical NLP samples with controlled fixed noise injected for Task 1 quality testing.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_sudo(cmd: List[str]) -> subprocess.CompletedProcess:
    sudo_pw = get_runtime_secret("CCF_SUDO_PW", "SUDO_PW")
    if sudo_pw:
        return subprocess.run(["sudo", "-S"] + cmd, input=f"{sudo_pw}\n", capture_output=True, text=True)
    return subprocess.run(["sudo", "-n"] + cmd, capture_output=True, text=True)


def run_psql(sql: str) -> None:
    result = subprocess.run(
        ["docker", "exec", "-i", "datamate-database", "psql", "-U", "postgres", "-d", "datamate"],
        input=sql,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def register_dataset(
    name: str,
    files: List[Tuple[Path, str]],
    fmt: str = "txt",
    description: str = "Task 1 mixed real-source benchmark",
) -> Dict:
    dataset_id = str(uuid.uuid4())
    target_dir = f"{dataset_volume()}/{dataset_id}"
    mkdir = run_sudo(["mkdir", "-p", target_dir])
    if mkdir.returncode != 0:
        raise RuntimeError(mkdir.stderr)

    rows = []
    total = 0
    for src, ftype in files:
        fname = src.name
        cp = run_sudo(["cp", str(src), f"{target_dir}/{fname}"])
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr)
        size = src.stat().st_size
        total += size
        rows.append(
            "("
            + ",".join([
                sql_quote(str(uuid.uuid4())),
                sql_quote(dataset_id),
                sql_quote(fname),
                sql_quote(f"/dataset/{dataset_id}/{fname}"),
                sql_quote(ftype),
                str(size),
                sql_quote("ACTIVE"),
            ])
            + ")"
        )

    sql = (
        "INSERT INTO t_dm_datasets "
        "(id,name,description,dataset_type,path,format,size_bytes,file_count,status,is_public,version) "
        "VALUES ("
        + ",".join([
            sql_quote(dataset_id),
            sql_quote(name),
            sql_quote(description),
            sql_quote("TEXT"),
            sql_quote(f"/dataset/{dataset_id}"),
            sql_quote(fmt),
            str(total),
            str(len(files)),
            sql_quote("ACTIVE"),
            "false",
            "0",
        ])
        + ");"
        "INSERT INTO t_dm_dataset_files "
        "(id,dataset_id,file_name,file_path,file_type,file_size,status) VALUES "
        + ",".join(rows)
        + ";"
    )
    run_psql(sql)
    return {"id": dataset_id, "name": name, "format": fmt, "files": [p.name for p, _ in files]}


def register_benchmark(manifest: Dict) -> Dict:
    if "files_by_type" in manifest:
        file_map = {k: [Path(v) for v in values] for k, values in manifest["files_by_type"].items()}
    else:
        file_map = {k: [Path(v)] for k, v in manifest["files"].items()}
    stamp = uuid.uuid4().hex[:8]
    text_files = [(path, "txt") for path in file_map.get("text", [])]
    csv_files = [(path, "csv") for path in file_map.get("csv", [])]
    json_files = [(path, "json") for path in file_map.get("json", [])]
    mixed_files = text_files + csv_files + json_files
    return {
        "mixed": register_dataset(f"任务一_混合源数据_真实样本_{stamp}", mixed_files, "txt"),
        "text": register_dataset(f"任务一_文本源数据_真实样本_{stamp}", text_files, "txt"),
        "csv": register_dataset(f"任务一_表格源数据_真实样本_{stamp}", csv_files, "csv"),
        "json": register_dataset(f"任务一_JSON源数据_真实样本_{stamp}", json_files, "json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output-dir", default="data/task1_mixed_benchmark")
    parser.add_argument("--text-files", type=int, default=8)
    parser.add_argument("--csv-files", type=int, default=4)
    parser.add_argument("--json-files", type=int, default=4)
    parser.add_argument("--records-per-text-file", type=int, default=2)
    parser.add_argument("--rows-per-structured-file", type=int, default=3)
    parser.add_argument("--register", action="store_true", help="Register generated files into DataMate DB")
    args = parser.parse_args()

    manifest = build_files(
        Path(args.output_dir),
        args.source_root,
        text_file_count=args.text_files,
        csv_file_count=args.csv_files,
        json_file_count=args.json_files,
        records_per_text_file=args.records_per_text_file,
        rows_per_structured_file=args.rows_per_structured_file,
    )
    if args.register:
        manifest["datamate_datasets"] = register_benchmark(manifest)
        Path(args.output_dir, "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
