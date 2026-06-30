from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = ROOT / "KB_上传版"

ES_URL = "http://[::1]:9210"
ES_USER = "elastic"
ES_PASSWORD = "nexent@2025"

KB_TO_INDEX = {
    "KB_代码模板库": "1-c496916098bb4ee4bbc9de84aae54818",
    "KB_建模方法库": "2-c37bfcdab2ed46de952db90cf34ca972",
    "KB_论文评分库": "3-b73f084fafb445e6825e4a31676ee9c2",
    "KB_图表模板库": "4-30e786f4fb0b420b86462c14d87d8739",
    "KB_优秀论文库": "5-3c1ccf5367c244ea9e15b21b66c46ff2",
}

ALLOWED = {
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".html",
    ".htm",
    ".xml",
    ".yml",
    ".yaml",
}


def auth_header() -> str:
    token = base64.b64encode(f"{ES_USER}:{ES_PASSWORD}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def http_json(method: str, url: str, payload: bytes | None = None, content_type: str = "application/json") -> dict:
    cmd = [
        "curl.exe",
        "-sS",
        "-u",
        f"{ES_USER}:{ES_PASSWORD}",
        "-X",
        method,
    ]
    temp_name = None
    try:
        if payload is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(payload)
                temp_name = tmp.name
            cmd.extend(["-H", f"Content-Type: {content_type}", "--data-binary", f"@{temp_name}"])
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"{method} {url} failed: {result.stderr.strip()}")
        raw = result.stdout.strip()
        return json.loads(raw) if raw else {}
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except Exception:
                pass


def delete_existing_index_docs(index_name: str) -> None:
    payload = json.dumps({"query": {"match_all": {}}}).encode("utf-8")
    http_json("POST", f"{ES_URL}/{index_name}/_delete_by_query?refresh=true&conflicts=proceed", payload)


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def chunk_text(text: str, size: int = 1800, overlap: int = 180) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_docs(folder_name: str, folder: Path) -> list[dict]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    docs: list[dict] = []
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED)
    for file_path in files:
        text = read_text(file_path)
        chunks = chunk_text(text)
        rel = file_path.relative_to(UPLOAD_ROOT).as_posix()
        path_or_url = f"local://{rel}"
        for idx, chunk in enumerate(chunks, start=1):
            raw_id = f"{folder_name}|{rel}|{idx}"
            doc_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
            title = file_path.stem if len(chunks) == 1 else f"{file_path.stem} - chunk {idx}"
            docs.append(
                {
                    "_id": doc_id,
                    "id": doc_id,
                    "title": title,
                    "filename": file_path.name,
                    "path_or_url": path_or_url,
                    "source_type": "local",
                    "language": "zh",
                    "author": "Codex",
                    "date": now[:10],
                    "content": chunk,
                    "process_source": "ManualImport",
                    "file_size": file_path.stat().st_size,
                    "create_time": now,
                    "embedding_model_name": "",
                }
            )
    return docs


def bulk_insert(index_name: str, docs: list[dict]) -> int:
    if not docs:
        return 0

    lines: list[str] = []
    for doc in docs:
        doc_id = doc.pop("_id")
        lines.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}, ensure_ascii=False))
        lines.append(json.dumps(doc, ensure_ascii=False))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    result = http_json("POST", f"{ES_URL}/_bulk?refresh=true", payload, "application/x-ndjson")
    if result.get("errors"):
        failed = [item for item in result.get("items", []) if item.get("index", {}).get("error")]
        raise RuntimeError(f"Bulk insert had {len(failed)} errors. First: {failed[:1]}")
    return len(docs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import generated KB files directly into Nexent Elasticsearch indices.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear existing documents before importing.")
    args = parser.parse_args()

    if not UPLOAD_ROOT.exists():
        print(f"Upload root not found: {UPLOAD_ROOT}", file=sys.stderr)
        return 1

    total = 0
    for folder_name, index_name in KB_TO_INDEX.items():
        folder = UPLOAD_ROOT / folder_name
        if not folder.exists():
            print(f"Skip missing folder: {folder}")
            continue
        docs = build_docs(folder_name, folder)
        if not args.no_clear:
            delete_existing_index_docs(index_name)
        count = bulk_insert(index_name, docs)
        total += count
        print(f"{folder_name}: indexed {count} chunks into {index_name}")

    print(f"Completed. Total indexed chunks: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
