"""Small IO helpers: read markdown/txt documents, write jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def iter_documents(input_dir: str | Path, exts: tuple[str, ...] = (".md", ".txt", ".markdown")) -> list[dict]:
    """Return [{fileName, filePath, text}] for every document under input_dir."""
    root = Path(input_dir)
    docs: list[dict] = []
    if not root.exists():
        return docs
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                docs.append({"fileName": p.name, "filePath": str(p), "text": read_text(p)})
            except Exception as exc:  # noqa: BLE001
                print(f"[io] skip {p}: {exc}")
    return docs


def write_jsonl(records: Iterable[dict], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)


def write_json(obj, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)
