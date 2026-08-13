"""DataMate `datamate.core.base_op` 的契约兼容轻量实现（shim）。

设计目标：
- 当环境中**装有** DataMate runtime（`datamate` 包）时，本模块不会被使用，
  所有算子直接继承真实的 `datamate.core.base_op.Mapper/Filter/Slicer/LLM`；
- 当环境中**没有** DataMate runtime（例如评审在任意机器快速体验）时，
  本模块提供与其 **完全同构的执行契约**，使本项目算子无需改动即可运行。

与原实现的一致性（逐条对齐 datamate/runtime/python-executor/datamate/core/base_op.py）：
- sample 数据契约：text / data / fileName / fileType / fileId / filePath /
  fileSize / export_path / ext_params / target_type / instance_id /
  execute_result / failed_reason；
- Mapper.__call__：前序失败样本跳过；execute 异常不抛出，
  转 create_failure_sample 并继续处理后续文件；
- Filter.__call__：返回 bool，text 与 data 同时为空视为被过滤；
- Slicer.__call__：execute 返回 List[Dict]，经 load_sample_to_sample
  合并回父样本，整体返回 [sample]；
- Registry：register_module 支持 module_cls / module_path（懒加载 dotted path）。

有意省略的部分（均为平台侧强依赖，不影响算子语义）：
- PostgreSQL 任务信息持久化（TaskInfoPersistence）；
- cv2 / unstructured 的重型文件解析（shim 的 read_file 仅支持纯文本类格式）。
"""

import json
import time
import traceback
from typing import Any, Dict, List, Tuple

FAILED_STATUS = "FAILED"
SUCCESS_STATUS = "COMPLETED"
UNKNOWN_ERROR_CODE = "99999"


class Fields:
    result = "execute_result"
    instance_id = "instance_id"
    export_path = "export_path"


def get_exception_info(e: BaseException) -> str:
    exc_type = type(e).__name__
    exc_msg = str(e)
    tb = traceback.extract_tb(e.__traceback__)
    if tb:
        return (
            f"错误类型: {exc_type}\n错误原因: {exc_msg}\n"
            f"文件名: {tb[-1].filename}\n行号: {tb[-1].lineno}\n代码行: {tb[-1].line}"
        )
    return f"错误类型: {exc_type}\n错误原因: {exc_msg}"


class Registry:
    """与 datamate.common.utils.registry.Registry 同构的注册器。"""

    def __init__(self, name: str):
        self._name = name
        self._modules: Dict[str, Any] = {}

    @property
    def name(self):
        return self._name

    @property
    def modules(self):
        return self._modules

    def get(self, module_key):
        return self._modules.get(module_key, None)

    def register_module(self, module_name: str = None, module_cls: type = None,
                        module_path: str = None, force=False):
        if module_cls is not None:
            self._register(module_name, module_cls=module_cls, force=force)
            return module_cls
        if module_cls is None and isinstance(module_path, str):
            self._register(module_name, module_path=module_path, force=force)
            return module_path

        def _decorator(cls):
            self._register(module_name, module_cls=cls, force=force)
            return cls

        return _decorator

    def _register(self, module_name=None, module_cls=None, module_path=None, force=False):
        if module_name is None and module_cls is not None:
            module_name = module_cls.__name__
        if module_name in self._modules:
            existing = self._modules[module_name]
            if module_cls is not None and existing == module_cls:
                return
            if module_path is not None and existing == module_path:
                return
            if not force:
                raise KeyError(f"{module_name} is already registered in {self._name}")
        self._modules[module_name] = module_cls if module_cls is not None else module_path


OPERATORS = Registry("Operators")


class BaseOp:
    """所有算子的父类（契约同 datamate.core.base_op.BaseOp）。"""

    use_model = False
    custom_ops = False

    def __init__(self, *args, **kwargs):
        self.accelerator = kwargs.get("accelerator", "cpu")
        self.is_last_op = kwargs.get("is_last_op", False)
        self.is_first_op = kwargs.get("is_first_op", False)
        self._name = kwargs.get("op_name", None)
        self.infer_model = None
        self.text_key = kwargs.get("text_key", "text")
        self.data_key = kwargs.get("data_key", "data")
        self.image_key = kwargs.get("image_key", "image")
        self.filename_key = kwargs.get("fileName_key", "fileName")
        self.filetype_key = kwargs.get("fileType_key", "fileType")
        self.fileid_key = kwargs.get("fileId_key", "fileId")
        self.filepath_key = kwargs.get("filePath_key", "filePath")
        self.filesize_key = kwargs.get("fileSize_key", "fileSize")
        self.export_path_key = kwargs.get("export_path_key", "export_path")
        self.ext_params_key = kwargs.get("ext_params_key", "ext_params")
        self.target_type_key = kwargs.get("target_type_key", "target_type")

    @property
    def name(self):
        return self._name if self._name else "UnknownOp"

    @staticmethod
    def is_npu_available():
        try:
            import torch_npu  # noqa: F401
            return torch_npu.npu.is_available()
        except ImportError:
            return False

    def use_npu(self):
        return self.accelerator == "npu" and self.is_npu_available()

    def init_model(self, *args, **kwargs):
        raise NotImplementedError

    def fill_sample_params(self, sample: Dict[str, Any], **kwargs):
        if not sample.get(self.text_key, None):
            sample[self.text_key] = ""
        if not sample.get(self.data_key, None):
            sample[self.data_key] = b""
        if not sample[self.data_key] and not sample[self.text_key]:
            sample.update(kwargs)

    @staticmethod
    def _get_error_info(e: BaseException) -> Tuple[str, str]:
        return UNKNOWN_ERROR_CODE, get_exception_info(e)

    def create_failure_sample(self, sample: Dict[str, Any], op_name, excp: BaseException):
        sample["execute_result"] = False
        error_code, exc_info = self._get_error_info(excp)
        sample["failed_reason"] = {
            "op_name": op_name,
            "error_code": error_code,
            "reason": exc_info,
        }

    def read_file(self, sample):
        """轻量版文件读取：仅支持纯文本类格式（够本项目医疗文本场景使用）。"""
        filepath = sample[self.filepath_key]
        filetype = str(sample.get(self.filetype_key, "")).lower()
        if filetype in ["txt", "md", "markdown", "xml", "html", "json", "jsonl", "csv"]:
            with open(filepath, "rb") as f:
                sample[self.text_key] = f.read().decode("utf-8-sig").replace("\r\n", "\n")
                sample[self.data_key] = b""
        else:
            raise ValueError(
                f"shim read_file 仅支持文本格式，收到: {filetype}。"
                f"如需 pdf/docx/图片解析请安装 DataMate runtime（datamate 包）。"
            )
        return sample

    def read_file_first(self, sample):
        if self.is_first_op:
            self.read_file(sample)


class Mapper(BaseOp):
    def __call__(self, sample: Dict[str, Any], **kwargs):
        if sample.get(Fields.result) is False:
            return sample
        self.fill_sample_params(sample, **kwargs)
        try:
            sample = self.execute(sample)
            sample["execute_status"] = SUCCESS_STATUS
        except Exception as e:  # 单文件失败不阻断流水线
            self.create_failure_sample(sample, self.name, e)
            sample["execute_status"] = FAILED_STATUS
            sample[self.filesize_key] = "0"
        return sample

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class Filter(BaseOp):
    def __call__(self, sample: Dict[str, Any], **kwargs):
        if sample.get(Fields.result) is False:
            return sample
        self.fill_sample_params(sample, **kwargs)
        try:
            sample = self.execute(sample)
        except Exception as e:
            self.create_failure_sample(sample, self.name, e)
            return True  # 失败样本保留，交由后续环节统一处理
        if sample[self.text_key] == "" and sample[self.data_key] == b"":
            return False
        return True

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class Slicer(BaseOp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_file_type = None

    def __call__(self, sample: Dict[str, Any], **kwargs):
        if sample.get(Fields.result) is False:
            return [sample]
        self.fill_sample_params(sample, **kwargs)
        try:
            sample_list = self.execute(sample)
        except Exception as e:
            self.create_failure_sample(sample, self.name, e)
            return [sample]
        self.load_sample_to_sample(sample, sample_list)
        return [sample]

    @staticmethod
    def load_sample_to_sample(sample: Dict, sample_list: List[Dict]):
        for sample_i in sample_list:
            for k, v in sample_i.items():
                sample[k] = v
        if not sample.get("fileNum", None):
            sample["fileNum"] = 1

    def execute(self, sample: Dict[str, Any]) -> List[Dict]:
        raise NotImplementedError


class LLM(Mapper):
    """LLM 算子基类（契约同 datamate base_op.LLM：self.llm(prompt) -> str）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = self.get_llm(**kwargs)

    def get_llm(self, **kwargs):
        url = kwargs.get("LLMUrl", "")
        headers = kwargs.get("LLMHeaders", {}) or {}
        body = kwargs.get("LLMBody", {}) or {}

        def _call(prompt: str) -> str:
            import requests

            payload = dict(body)
            payload["messages"] = [{"role": "user", "content": prompt}]
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        return _call

    def build_llm_prompt(self, *args, **kwargs):
        raise NotImplementedError

    def save_sample(self, result_list: List, sample: Dict[str, Any]):
        import os

        export_path = sample.get(self.export_path_key) or "."
        os.makedirs(export_path, exist_ok=True)
        base, _ = __import__("os").path.splitext(sample[self.filename_key])
        out = __import__("os").path.join(export_path, f"{sample[self.fileid_key]}_{base}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for item in result_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return out


class FileExporter(BaseOp):
    """文本导出算子（对齐 datamate base_op.FileExporter 的文本路径分支）。"""

    def execute(self, sample: Dict[str, Any]) -> bool:
        import os

        export_path = sample.get(self.export_path_key) or "."
        os.makedirs(export_path, exist_ok=True)
        target_type = sample.get(self.target_type_key) or "txt"
        filename = f"{sample[self.fileid_key]}.{target_type}"
        path = os.path.join(export_path, filename)
        content = (
            sample[self.text_key].encode("utf-8")
            if sample.get(self.text_key)
            else sample.get(self.data_key, b"")
        )
        with open(path, "wb") as f:
            f.write(content)
        return True
