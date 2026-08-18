"""Package the medical-kg-report skill into an uploadable zip for Nexent.

Matches the official skill layout (files nested under the skill folder name):
  medical-kg-report/SKILL.md
  medical-kg-report/scripts/*.py
Output: nexent_skill/dist/medical-kg-report.zip

Usage:
  python nexent_skill/build_skill_zip.py
Then in Nexent: 构建技能 -> 上传技能文件 -> 选择该 zip。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = "medical-kg-report"
SRC = HERE / SKILL
DIST = HERE / "dist"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    DIST.mkdir(exist_ok=True)
    zip_path = DIST / f"{SKILL}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(SRC.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                # archive name keeps the skill folder as the top-level entry
                zf.write(f, arcname=str(Path(SKILL) / f.relative_to(SRC)))
    print(f"packaged -> {zip_path}")
    print("Upload via Nexent: 构建技能 -> 上传技能文件")


if __name__ == "__main__":
    main()
