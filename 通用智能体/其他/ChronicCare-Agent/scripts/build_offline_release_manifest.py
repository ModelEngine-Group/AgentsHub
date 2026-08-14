#!/usr/bin/env python3
"""Build the sanitized, self-verifying final offline delivery archive."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
ARCHIVE = RELEASE / "ChronicCare-Agent-Final.tar.gz"
PACKAGE_ROOT = "ChronicCare-Agent"
PACKAGES = ["pandas", "PyYAML", "networkx", "pyvis", "plotly", "streamlit", "fastapi", "uvicorn", "pydantic", "Pillow", "requests", "httpx", "sqlglot", "mcp", "pytest", "pytest-cov", "ruff"]
DIRS = ["analysis", "app", "configs", "deploy", "docker", "docs", "integrations", "kg", "mcp_adapter", "orchestration", "runtime_common", "scripts", "tests", "tool_server", "visualization", "data/raw", "data/evaluation", "data/sqlite", "outputs/evaluation", "outputs/release"]
FILES = ["README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "RELEASE_NOTES.md", "MANIFEST.json", "requirements.txt", "pyproject.toml", "Makefile", ".dockerignore", ".gitignore", ".env.example", "docker-compose.yml", "docker-compose.server.yml", "data/graph/graph.json", "data/graph/graph.html", "data/graph/graph_summary.json"]
MODEL_PATH = "/models/MedCleanStd/bge-small-zh-v1.5"
MODEL_SHA256 = "688f9664eb65edea0f73f78464c767a759a230b60ae74001af9492be6a67e94c"
PRIVATE = re.compile(r"/mnt/" r"nvme0n1/home/[^/\s]+|10\.236\.(?:12\.5|2\.5)")
SECRET = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".xml", ".log", ".md", ".txt", ".sh", ".toml", ".ini", ".cfg", ".env"}
FORMAL_LOGS = {"outputs/evaluation/pytest_execution.log", "outputs/evaluation/ruff_execution.log"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    cache_or_trace = any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", "logs", "mcp_traces", "runtime_generated"} for part in path.parts)
    open_sql_trace = path.as_posix() == "outputs/open_sql/traces" or path.as_posix().startswith("outputs/open_sql/traces/")
    generated_log = path.suffix == ".log" and path.as_posix() not in FORMAL_LOGS
    return cache_or_trace or open_sql_trace or path.suffix == ".pyc" or generated_log or path.name == ".env"


def copy_item(source: Path, destination: Path) -> None:
    if not source.exists() or ignored(source.relative_to(ROOT)):
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=lambda d, names: [n for n in names if ignored((Path(d) / n).relative_to(ROOT))])
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def sanitize_tree(stage: Path) -> list[str]:
    changed = []
    for path in stage.rglob("*"):
        if not path.is_file() or (path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env.example"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        clean = PRIVATE.sub(lambda m: "<project-root>" if m.group(0).startswith("/") else "host.docker.internal", text)
        if clean != text:
            path.write_text(clean, encoding="utf-8")
            changed.append(path.relative_to(stage).as_posix())
    return changed


def scan(stage: Path) -> dict[str, list[str]]:
    result = {"secret_hits": [], "private_path_or_ip_hits": [], "forbidden_files": []}
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(stage).as_posix()
        if path.name == ".env" or "__pycache__" in path.parts or path.suffix == ".pyc":
            result["forbidden_files"].append(rel)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".env.example":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if SECRET.search(text): result["secret_hits"].append(rel)
            if PRIVATE.search(text): result["private_path_or_ip_hits"].append(rel)
    return result


def inventory(stage: Path) -> tuple[list[dict[str, object]], str]:
    files=[]; digest=hashlib.sha256()
    for path in sorted(x for x in stage.rglob("*") if x.is_file()):
        rel=path.relative_to(stage).as_posix(); value=sha(path); size=path.stat().st_size
        files.append({"path": rel, "size": size, "sha256": value})
        digest.update(rel.encode()); digest.update(b"\0"); digest.update(value.encode()); digest.update(b"\n")
    return files, digest.hexdigest()


def main() -> int:
    RELEASE.mkdir(parents=True, exist_ok=True)
    dependency_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    dependency_lines = [line.strip() for line in dependency_text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not dependency_lines or any("==" not in line for line in dependency_lines):
        print(json.dumps({"status": "failed", "reason": "requirements.txt must contain pinned dependencies"}, ensure_ascii=False))
        return 1
    with tempfile.TemporaryDirectory(prefix="chroniccare-final-") as temporary:
        stage=Path(temporary)/PACKAGE_ROOT
        for item in DIRS + FILES: copy_item(ROOT/item, stage/item)
        sanitized=sanitize_tree(stage)
        privacy=scan(stage)
        if any(privacy.values()):
            print(json.dumps({"status":"failed","privacy_scan":privacy}, ensure_ascii=False)); return 1
        file_list, contents_sha=inventory(stage)
        package_manifest={
            "schema_version":"1.0.0", "status":"success", "created_at":datetime.now().astimezone().isoformat(),
            "package_root":PACKAGE_ROOT, "data_classification":"synthetic competition data; no real patient data",
            "contents_sha256":contents_sha, "file_count":len(file_list), "files":file_list,
            "snapshots":{"sqlite":{"path":"data/sqlite/chroniccare.db","sha256":sha(stage/"data/sqlite/chroniccare.db")},"graph":{"path":"data/graph/graph.json","sha256":sha(stage/"data/graph/graph.json")}},
            "model":{"name":"bge-small-zh-v1.5","version":"1.5","bundled":False,"mount_path":MODEL_PATH,"directory_sha256":MODEL_SHA256,"checksum_algorithm":"SHA-256 over sorted relative_path\\0file_sha256\\n records","preparation":"Place the verified model directory at the mount path before NPU evaluation; CPU and NPU must use these identical weights."},
            "startup":["docker compose up -d --build chroniccare-runtime","python3 scripts/final_competition_acceptance.py"],
            "ports":{"tool_server":18088,"mcp_adapter":18188,"dashboard":18501}, "sanitized_files":sanitized,
            "privacy_scan":privacy, "excluded":[".env","credentials","model weights","runtime caches/logs/traces","historical worklogs","intermediate processed data"]
        }
        (stage/"release").mkdir(parents=True,exist_ok=True)
        (stage/"release/package_contents_manifest.json").write_text(json.dumps(package_manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        tmp_archive=RELEASE/(ARCHIVE.name+".tmp")
        with tarfile.open(tmp_archive,"w:gz",compresslevel=6) as tar: tar.add(stage,arcname=PACKAGE_ROOT)
        os.replace(tmp_archive,ARCHIVE)
    final={**package_manifest,"archive":{"path":ARCHIVE.relative_to(ROOT).as_posix(),"size":ARCHIVE.stat().st_size,"sha256":sha(ARCHIVE)},"verification":[f"sha256sum {ARCHIVE.name}",f"tar -xzf {ARCHIVE.name}","cd ChronicCare-Agent && python3 scripts/final_competition_acceptance.py"]}
    out=RELEASE/"offline_release_manifest.json"; out.write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","archive":str(ARCHIVE.relative_to(ROOT)),"archive_size":ARCHIVE.stat().st_size,"archive_sha256":final["archive"]["sha256"],"files":final["file_count"],"privacy_scan":privacy},ensure_ascii=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
