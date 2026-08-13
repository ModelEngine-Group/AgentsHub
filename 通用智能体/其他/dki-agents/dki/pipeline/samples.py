"""样本构造与加载（sample dict 契约与 DataMate 完全一致）。"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def make_sample(
    file_path: Optional[Path] = None,
    text: str = "",
    file_id: str = "",
    file_name: str = "",
    export_path: str = "",
    instance_id: str = "",
) -> Dict[str, Any]:
    """构造符合 DataMate 契约的 sample。

    - 给定 file_path：由第一个算子的 read_file_first 负责读取（与平台一致）；
    - 给定 text：内容预置（内存态样本，运行器会关闭首算子的 read_file）。
    """
    if file_path is not None:
        p = Path(file_path)
        return {
            "text": text,
            "data": b"",
            "fileName": p.name,
            "fileType": p.suffix.lstrip(".").lower() or "txt",
            "fileId": file_id or p.stem,
            "filePath": str(p),
            "fileSize": str(p.stat().st_size if p.exists() else 0),
            "export_path": export_path,
            "ext_params": {},
            "target_type": None,
            "instance_id": instance_id,
        }
    return {
        "text": text,
        "data": b"",
        "fileName": file_name or f"{file_id or uuid.uuid4().hex[:8]}.txt",
        "fileType": "txt",
        "fileId": file_id or uuid.uuid4().hex[:8],
        "filePath": "",
        "fileSize": str(len(text.encode("utf-8"))),
        "export_path": export_path,
        "ext_params": {},
        "target_type": None,
        "instance_id": instance_id,
    }


def samples_from_dir(
    input_dir: Path, pattern: str = "*.txt", export_path: str = "", instance_id: str = ""
) -> List[Dict[str, Any]]:
    files = sorted(Path(input_dir).glob(pattern))
    return [
        make_sample(file_path=f, export_path=export_path, instance_id=instance_id)
        for f in files
    ]


def samples_from_jsonl(dataset_path: Path, export_path: str = "") -> List[Dict[str, Any]]:
    """加载 DataMate dataset.jsonl（每行一个文件描述 JSON）。"""
    out = []
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sample = make_sample(
            file_path=Path(row["filePath"]) if row.get("filePath") else None,
            text=row.get("text", ""),
            file_id=str(row.get("fileId", "")),
            export_path=export_path or row.get("export_path", ""),
        )
        if row.get("fileName"):
            sample["fileName"] = row["fileName"]
        if row.get("fileType"):
            sample["fileType"] = row["fileType"]
        out.append(sample)
    return out


def iter_output_texts(samples: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for s in samples:
        if s.get("execute_result") is not False and s.get("text"):
            yield s["text"]
