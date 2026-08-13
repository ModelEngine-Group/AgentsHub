"""算子市场打包脚本：生成不依赖仓库内 ``dki`` 包的上传 ZIP。

源码在仓库内统一通过 :mod:`dki.dmcompat` 兼容 Windows/离线环境；上传包则
必须直接绑定 DataMate，并将 OM shadow 所依赖的主抽取器与词典一同内置。
每个 ZIP 都经过全量 Python 源码依赖审计。
"""

import argparse
import ast
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPS_DIR = PROJECT_ROOT / "dki" / "operators"
DICT_PATH = PROJECT_ROOT / "dki" / "kg" / "dictionaries" / "medical_dict.json"
SCHEMA_PATH = PROJECT_ROOT / "dki" / "kg" / "schema.py"
CMEIE_PAIR_PATH = PROJECT_ROOT / "dki" / "nlp" / "cmeie_pair.py"
DIST_DIR = PROJECT_ROOT / "dist" / "operators"

PACKAGES = [
    "medical_text_cleaner",
    "medical_section_splitter",
    "medical_entity_extractor",
    "medical_relation_extractor",
    "triple_builder_validator",
    # Release A opt-in candidate. Ships disabled (shadowEnabled=false) and only
    # depends on the medical_entity_extractor package at runtime; its optional
    # NPU runtime factory is configured on the target host, never bundled here.
    "medical_entity_extractor_om_shadow",
]

# Repository compatibility imports become the real platform imports in archives.
_DMCOMPAT_PROCESS_RE = re.compile(
    r"^from dki\.dmcompat import (?P<names>[^\n]+)$", re.MULTILINE,
)
_DMCOMPAT_INIT_RE = re.compile(
    r"^from dki\.dmcompat import OPERATORS$", re.MULTILINE,
)

# Cross-operator imports must point at the platform deployment path on upload.
_CROSS_OP_IMPORT_RE = re.compile(r"^from dki\.operators\.([a-z_]+)\.process import", re.MULTILINE)

# try/except 双路径导入 -> 平台单路径导入
_TRY_IMPORT_RE = re.compile(
    r"try:\n(?P<imp>(?:    from datamate[^\n]+\n)+)"
    r"(?:    _DATAMATE = True\n)?"
    r"except ImportError:[^\n]*\n"
    r"(?:    from dki[^\n]+\n)+"
    r"(?:    _DATAMATE = False\n)?",
)


def _transform_process(src: str, pkg: str) -> str:
    def _repl(m: re.Match) -> str:
        imports = "".join(line[4:] + "\n" for line in m.group("imp").rstrip("\n").split("\n"))
        extra = "_DATAMATE = True\n" if "_DATAMATE" in m.group(0) else ""
        return imports + extra

    out = _TRY_IMPORT_RE.sub(_repl, src)

    def platform_import(match: re.Match) -> str:
        names = match.group("names")
        names = names.replace("DATAMATE_AVAILABLE as _DATAMATE, ", "")
        prefix = "_DATAMATE = True\n" if "DATAMATE_AVAILABLE" in match.group("names") else ""
        return prefix + f"from datamate.core.base_op import {names}"

    out = _DMCOMPAT_PROCESS_RE.sub(platform_import, out)
    out = _CROSS_OP_IMPORT_RE.sub(r"from ops.user.\1.process import", out)
    if pkg == "medical_text_cleaner":
        out = re.sub(
            r"\s*try:\n"
            r"\s*# 独立运行：通过 dki 集成层注册 DataMate 内置算子\n"
            r"\s*from dki\.dmcompat import load_builtin_ops\n\n"
            r"\s*load_builtin_ops\(\)\n"
            r"\s*except ImportError:\n"
            r"\s*pass[^\n]*",
            "\n            # DataMate 平台已完成内置算子注册。\n            pass",
            out,
        )
    if pkg == "triple_builder_validator":
        out = re.sub(
            r"try:\n    from dki\.kg\.schema import validate_triple as _validate\n"
            r"except ImportError:\n",
            "if True:  # standalone archive uses the bundled equivalent schema\n",
            out,
        )
    if pkg == "medical_relation_extractor":
        out = out.replace(
            "from dki.kg.schema import RELATION_TYPES, validate_triple",
            "from ._bundled_schema import RELATION_TYPES, validate_triple",
        )
        out = out.replace(
            "from dki.nlp.cmeie_pair import (",
            "from ._bundled_cmeie_pair import (",
        )
    if pkg == "medical_entity_extractor_om_shadow":
        out = out.replace(
            "from ops.user.medical_entity_extractor.process import MedicalEntityExtractor",
            "from ._bundled_medical_entity_extractor import MedicalEntityExtractor",
        )
    # Optional repository-only backends remain explicit, actionable failures in
    # standalone archives rather than hidden imports of unavailable dki modules.
    out = re.sub(
        r"from dki\.llm import chat, llm_available",
        "raise ImportError('LLM backend requires the full dki application')",
        out,
    )
    out = re.sub(
        r"from dki\.llm import chat",
        "raise ImportError('LLM backend requires the full dki application')",
        out,
    )
    out = re.sub(
        r"from dki\.bench\.npu\.bert_ner_infer import BertNerBackend",
        "raise ImportError('BERT backend requires the external dki NPU runtime')",
        out,
    )
    out = re.sub(
        r"from dki\.bench\.npu\.cmeee_ner_infer import "
        r"(?:CmeeeGlobalPointer(?:Om)?Backend|\([\s\S]*?\))",
        "raise ImportError('CMeEE backend requires the external dki NPU runtime')",
        out,
    )
    return out


def _transform_init(src: str, pkg: str) -> str:
    out = src.replace(
        f'module_path="dki.operators.{pkg}.process"',
        f'module_path="ops.user.{pkg}.process"',
    )
    out = _DMCOMPAT_INIT_RE.sub("from datamate.core.base_op import OPERATORS", out)
    out = _TRY_IMPORT_RE.sub(lambda m: "".join(
        line[4:] + "\n" for line in m.group("imp").rstrip("\n").split("\n")), out)
    # __init__ 的简单双路径（两行式）
    out = out.replace(
        "try:\n    from datamate.core.base_op import OPERATORS\nexcept ImportError:\n"
        "    from dki.compat.datamate_base import OPERATORS\n",
        "from datamate.core.base_op import OPERATORS\n",
    )
    return out


def _self_check(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            content = zf.read(name).decode("utf-8")
            tree = ast.parse(content, filename=f"{zip_path.name}:{name}")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (
                    node.module == "dki" or str(node.module).startswith("dki.")
                ):
                    raise AssertionError(
                        f"{zip_path.name}:{name}:{node.lineno} 存在 dki 依赖: {node.module}"
                    )
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "dki" or alias.name.startswith("dki."):
                            raise AssertionError(
                                f"{zip_path.name}:{name}:{node.lineno} 存在 dki 依赖: {alias.name}"
                            )


def _package_all_into(dist_dir: Path) -> list[Path]:
    dist_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for pkg in PACKAGES:
        pkg_dir = OPS_DIR / pkg
        zip_path = dist_dir / f"{pkg}.zip"
        with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.yml", (pkg_dir / "metadata.yml").read_text(encoding="utf-8"))
            zf.writestr("__init__.py", _transform_init(
                (pkg_dir / "__init__.py").read_text(encoding="utf-8"), pkg))
            process_source = _transform_process(
                (pkg_dir / "process.py").read_text(encoding="utf-8"), pkg,
            )
            zf.writestr("process.py", process_source)
            if pkg == "medical_entity_extractor":
                zf.writestr("resources/medical_dict.json", DICT_PATH.read_text(encoding="utf-8"))
            elif pkg == "medical_relation_extractor":
                zf.writestr("_bundled_schema.py", SCHEMA_PATH.read_text(encoding="utf-8"))
                zf.writestr(
                    "_bundled_cmeie_pair.py",
                    CMEIE_PAIR_PATH.read_text(encoding="utf-8"),
                )
            elif pkg == "medical_entity_extractor_om_shadow":
                primary_source = _transform_process(
                    (OPS_DIR / "medical_entity_extractor" / "process.py").read_text(encoding="utf-8"),
                    "medical_entity_extractor",
                )
                zf.writestr("_bundled_medical_entity_extractor.py", primary_source)
                zf.writestr("resources/medical_dict.json", DICT_PATH.read_text(encoding="utf-8"))
        _self_check(zip_path)
        outputs.append(zip_path)
        try:
            display_path = zip_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = zip_path
        print(f"[√] {display_path}")
    return outputs


def package_all(output_dir: Path | None = None) -> list[Path]:
    """Build, self-check, and publish an immutable operator ZIP directory.

    Accepted operator archives are evidence, so rebuilding must never replace
    them in place.  The complete set is prepared in a sibling staging
    directory and becomes visible with one directory rename.  Any existing
    final path is rejected before packaging starts.
    """

    dist_dir = Path(output_dir) if output_dir is not None else DIST_DIR
    dist_dir = dist_dir.absolute()
    if os.path.lexists(dist_dir):
        raise FileExistsError(f"refusing to overwrite operator output: {dist_dir}")
    dist_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{dist_dir.name}.", dir=dist_dir.parent)
    )
    published = False
    try:
        staged = _package_all_into(staging)
        expected_names = sorted(f"{package}.zip" for package in PACKAGES)
        observed_names = sorted(path.name for path in staging.iterdir())
        if observed_names != expected_names:
            raise RuntimeError(
                "operator staging directory does not match the declared package set"
            )
        os.rename(staging, dist_dir)
        published = True
        return [dist_dir / path.name for path in staged]
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DIST_DIR)
    cli_args = parser.parse_args()
    outs = package_all(cli_args.output_dir)
    print(f"\n共打包 {len(outs)} 个算子，可通过 DataMate 算子市场『上传算子』直接导入"
          f"（归档内 metadata.yml 位于根路径，module_path 已指向 ops.user.*）。")
