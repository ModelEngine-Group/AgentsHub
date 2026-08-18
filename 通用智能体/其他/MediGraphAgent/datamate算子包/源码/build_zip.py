"""Package each DataMate operator folder into a correctly named .zip.

DataMate requires the zip file name (without extension) to equal the package
directory name referenced in __init__.py's module_path. This script zips the
*contents* of each operator folder (files at the archive root, no parent dir)
into datamate_ops/dist/<name>.zip.

Usage:
  python datamate_ops/build_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parent
DIST = OPS_DIR / "dist"
OPERATORS = ["text_clean", "chunker", "medical_ner", "medical_re", "triple_validator"]
INCLUDE = {"__init__.py", "metadata.yml", "process.py", "requirements.txt", "README.md"}


def build(name: str) -> Path:
    src = OPS_DIR / name
    if not src.exists():
        raise FileNotFoundError(src)
    DIST.mkdir(exist_ok=True)
    zip_path = DIST / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.iterdir()):
            if f.is_file() and f.name in INCLUDE:
                zf.write(f, arcname=f.name)  # at archive root
    return zip_path


def main() -> None:
    for name in OPERATORS:
        try:
            p = build(name)
            print(f"  packaged {name} -> {p}")
        except FileNotFoundError as exc:
            print(f"  skip {name}: {exc}")
    print(f"\nDone. Upload the zips in {DIST} to the DataMate operator marketplace.")


if __name__ == "__main__":
    main()
