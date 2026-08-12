"""校验并展开 MediFlow AgentHub 归档中的运行知识资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile


PACKAGE_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_ROOT = PACKAGE_ROOT / "knowledge_base"
MCP_ROOT = PACKAGE_ROOT / "mcps" / "mediflow-mcp"


def _sha256(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest().upper()


def _load_manifest() -> list[dict]:
    manifest_path = KNOWLEDGE_ROOT / "MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("knowledge_base/MANIFEST.json 没有可用资产记录")
    return assets


def _verify_asset(asset: dict) -> None:
    archive_path = KNOWLEDGE_ROOT / str(asset["archive"])
    entry_name = str(asset["entry"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"缺少知识资产压缩包：{archive_path}")

    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if entry_name not in names:
            raise RuntimeError(f"{archive_path.name} 缺少条目 {entry_name}")
        info = archive.getinfo(entry_name)
        if info.file_size != int(asset["bytes"]):
            raise RuntimeError(
                f"{archive_path.name}/{entry_name} 大小不符："
                f"{info.file_size} != {asset['bytes']}"
            )
        with archive.open(info) as stream:
            actual = _sha256(stream)
    expected = str(asset["sha256"]).upper()
    if actual != expected:
        raise RuntimeError(
            f"{archive_path.name}/{entry_name} SHA-256 不符：{actual} != {expected}"
        )


def _extract_asset(asset: dict, target_root: Path) -> None:
    archive_path = KNOWLEDGE_ROOT / str(asset["archive"])
    entry_name = str(asset["entry"])
    target = target_root / str(asset["target"])
    target.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive, archive.open(entry_name) as source:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(source, temporary, length=1024 * 1024)

    try:
        with temporary_path.open("rb") as stream:
            actual = _sha256(stream)
        expected = str(asset["sha256"]).upper()
        if actual != expected:
            raise RuntimeError(f"展开后的 {target.name} SHA-256 不符")
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证压缩资产，不写入 MCP 运行目录",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=MCP_ROOT,
        help="资产展开目标，默认为 mcps/mediflow-mcp",
    )
    args = parser.parse_args()

    assets = _load_manifest()
    for asset in assets:
        _verify_asset(asset)
        if not args.verify_only:
            _extract_asset(asset, args.target_root.resolve())
        action = "已验证" if args.verify_only else "已验证并展开"
        print(f"{action}: {asset['entry']} -> {asset['target']}")

    print(f"知识资产处理完成，共 {len(assets)} 个文件。")


if __name__ == "__main__":
    main()
