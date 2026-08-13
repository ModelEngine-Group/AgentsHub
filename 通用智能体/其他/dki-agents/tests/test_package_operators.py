"""Standalone DataMate operator archive packaging and import smoke tests."""

import json
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from scripts import package_operators


_BASE_OP_STUB = """
class Registry:
    def __init__(self):
        self.modules = {}

    def register_module(self, module_name=None, module_cls=None, module_path=None, force=False):
        self.modules[module_name] = module_cls if module_cls is not None else module_path
        return self.modules[module_name]


OPERATORS = Registry()


class Mapper:
    def __init__(self, *args, **kwargs):
        self.accelerator = kwargs.get("accelerator", "cpu")
        self.is_first_op = kwargs.get("is_first_op", False)
        self.is_last_op = kwargs.get("is_last_op", False)
        self.text_key = kwargs.get("text_key", "text")
        self.data_key = kwargs.get("data_key", "data")
        self.filename_key = kwargs.get("fileName_key", "fileName")
        self.filetype_key = kwargs.get("fileType_key", "fileType")
        self.fileid_key = kwargs.get("fileId_key", "fileId")
        self.filepath_key = kwargs.get("filePath_key", "filePath")
        self.filesize_key = kwargs.get("fileSize_key", "fileSize")
        self.ext_params_key = kwargs.get("ext_params_key", "ext_params")

    def read_file_first(self, sample):
        if self.is_first_op:
            raise AssertionError("standalone smoke passes in-memory text")

    def use_npu(self):
        return False
"""


_SMOKE_SCRIPT = r"""
import importlib.abc
import json
import sys
from pathlib import Path

stage = Path(sys.argv[1])
sys.path.insert(0, str(stage))


class BlockRepositoryDki(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dki" or fullname.startswith("dki."):
            raise ModuleNotFoundError(f"repository-local import blocked: {fullname}")
        return None


sys.meta_path.insert(0, BlockRepositoryDki())


def sample(text):
    return {
        "text": text,
        "data": b"",
        "fileName": "smoke.txt",
        "fileType": "txt",
        "fileId": "smoke-1",
        "filePath": "",
        "fileSize": "0",
        "ext_params": {},
    }


from ops.user.medical_text_cleaner.process import MedicalTextCleaner
cleaned = MedicalTextCleaner(maskPii=True).execute(sample("<p>冠心病 13912345678</p>"))
assert cleaned["text"] == "冠心病 <tel>"

from ops.user.medical_section_splitter.process import MedicalSectionSplitter
split = MedicalSectionSplitter().execute(sample("诊断：冠心病。\n主诉：胸痛。"))
assert json.loads(split["text"])["sections"]["诊断"] == "冠心病。"

from ops.user.medical_entity_extractor.process import MedicalEntityExtractor
entities = MedicalEntityExtractor(backend="dict").execute(sample("诊断：冠心病，伴胸痛。"))
assert {item["name"] for item in json.loads(entities["text"])["entities"]} >= {"冠心病", "胸痛"}

relation_payload = {
    "sections": {"诊断": "冠心病。", "主诉": "反复胸痛。"},
    "entities": [
        {"name": "冠心病", "mention": "冠心病", "type": "Disease", "section": "诊断", "span": [0, 3]},
        {"name": "胸痛", "mention": "胸痛", "type": "Symptom", "section": "主诉", "span": [2, 4]},
    ],
}
from ops.user.medical_relation_extractor.process import MedicalRelationExtractor
relations = MedicalRelationExtractor().execute(sample(json.dumps(relation_payload, ensure_ascii=False)))
relation_items = json.loads(relations["text"])["relations"]
assert any(item["relation"] == "HAS_SYMPTOM" for item in relation_items)

from ops.user.triple_builder_validator.process import TripleBuilderValidator
triples = TripleBuilderValidator().execute(sample(json.dumps({"relations": relation_items}, ensure_ascii=False)))
assert json.loads(triples["text"])["validation"]["n_valid"] == 1

from ops.user.medical_entity_extractor_om_shadow.process import MedicalEntityExtractorOmShadow
shadow = MedicalEntityExtractorOmShadow(primaryBackend="dict", shadowEnabled=False)
shadow_result = shadow.execute(sample("冠心病伴胸痛。"))
assert {item["name"] for item in json.loads(shadow_result["text"])["entities"]} >= {"冠心病", "胸痛"}


class BatchRuntime:
    def __init__(self):
        self.batches = []
        self.closed = False

    def execute_batch(self, texts, timeout_s=1.0):
        del timeout_s
        self.batches.append(list(texts))
        return [[{"span": [0, 3], "type": "Disease"}] for _ in texts]

    def close(self):
        self.closed = True


import os
os.environ["DKI_SHADOW_HASH_KEY"] = "standalone-package-smoke"
runtime = BatchRuntime()
sidecar = stage / "standalone_shadow.jsonl"
enabled_shadow = MedicalEntityExtractorOmShadow(
    primaryBackend="dict",
    shadowEnabled=True,
    shadow_runtime=runtime,
    sidecarPath=str(sidecar),
    shadowTimeoutMs=500,
)
try:
    enabled_result = enabled_shadow.execute(sample("冠心病伴胸痛。"))
    assert {
        item["name"] for item in json.loads(enabled_result["text"])["entities"]
    } >= {"冠心病", "胸痛"}
    assert enabled_shadow.flush(timeout_s=3.0)
finally:
    enabled_shadow.close()
assert runtime.batches == [["冠心病伴胸痛。"]]
assert runtime.closed is True
record = json.loads(sidecar.read_text(encoding="utf-8"))
assert record["error"] is None
assert record["shadow_count"] == 1
"""


def _write_package(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("", encoding="utf-8")


def test_operator_archives_are_standalone_and_invocable(tmp_path, monkeypatch):
    dist_dir = tmp_path / "dist"
    monkeypatch.setattr(package_operators, "DIST_DIR", dist_dir)
    outputs = package_operators.package_all()

    assert {path.stem for path in outputs} == set(package_operators.PACKAGES)
    stage = tmp_path / "isolated"
    _write_package(stage / "ops")
    _write_package(stage / "ops" / "user")
    _write_package(stage / "datamate")
    _write_package(stage / "datamate" / "core")
    (stage / "datamate" / "core" / "base_op.py").write_text(
        textwrap.dedent(_BASE_OP_STUB), encoding="utf-8"
    )

    for archive in outputs:
        package_name = archive.stem
        target = stage / "ops" / "user" / package_name
        target.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            assert {"metadata.yml", "__init__.py", "process.py"} <= names
            init_source = zf.read("__init__.py").decode("utf-8")
            assert f'module_path="ops.user.{package_name}.process"' in init_source
            assert "from dki.dmcompat import" not in init_source
            zf.extractall(target)

    with zipfile.ZipFile(dist_dir / "medical_entity_extractor_om_shadow.zip") as zf:
        names = set(zf.namelist())
        assert "_bundled_medical_entity_extractor.py" in names
        assert "resources/medical_dict.json" in names
        shadow_source = zf.read("process.py").decode("utf-8")
        assert "from ._bundled_medical_entity_extractor import MedicalEntityExtractor" in shadow_source
        assert "ops.user.medical_entity_extractor" not in shadow_source

    completed = subprocess.run(
        [sys.executable, "-I", "-c", _SMOKE_SCRIPT, str(stage)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_self_check_rejects_dki_import_in_importerror_fallback(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "process.py",
            "try:\n    import unavailable_platform\nexcept ImportError:\n    from dki.dmcompat import Mapper\n",
        )

    with pytest.raises(AssertionError, match="dki"):
        package_operators._self_check(archive)


def test_operator_output_directory_is_create_once(tmp_path):
    output = tmp_path / "rid-scoped-operators"
    archives = package_operators.package_all(output)
    before = {path.name: path.read_bytes() for path in archives}

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        package_operators.package_all(output)

    assert {path.name: path.read_bytes() for path in archives} == before
