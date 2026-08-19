"""DataMate 集成开关层（本项目所有算子/运行器统一从这里导入基类与注册表）。

优先使用真实的 DataMate runtime：
    from datamate.core.base_op import Mapper, ...      # pip install datamate/runtime/python-executor
不可用时自动退化到契约兼容 shim：
    from dki.compat.datamate_base import Mapper, ...
正式验收可设 DKI_REQUIRE_REAL_DATAMATE=1；此时任何引导或导入失败都会报错，
不会静默退化到 shim。

另外提供 load_builtin_ops()：把 DataMate 的 runtime 目录挂到 sys.path
（等价于 DataMate 官方镜像的 PYTHONPATH=/opt/runtime/datamate），
触发 ops.mapper / ops.filter / ops.slicer 的懒注册，使 DataMate 全部内置算子
（ExtraSpaceCleaner、HtmlTagCleaner、AnonymizedPhoneNumber 等 40+）可被本项目
流水线直接编排。
"""

import logging
import os
import sys

from dki.paths import find_datamate_runtime

logger = logging.getLogger(__name__)


def _require_real_datamate() -> bool:
    return os.getenv("DKI_REQUIRE_REAL_DATAMATE", "0").strip() == "1"


def _prepend_sys_path(path) -> str:
    """Put one resolved directory first without accumulating duplicates."""
    path_text = str(path.resolve())
    normalized_path = os.path.normcase(os.path.abspath(path_text))
    remaining = []
    for entry in sys.path:
        try:
            normalized_entry = os.path.normcase(os.path.abspath(os.fspath(entry)))
        except (OSError, TypeError, ValueError):
            remaining.append(entry)
            continue
        if normalized_entry != normalized_path:
            remaining.append(entry)
    sys.path[:] = [path_text, *remaining]
    return path_text


def _bootstrap_failure_message(error, runtime_dir=None) -> str:
    configured = os.getenv("DKI_DATAMATE_RUNTIME", "").strip() or "<unset>"
    resolved = str(runtime_dir) if runtime_dir is not None else "<not found>"
    return (
        "Real DataMate runtime is required "
        "(DKI_REQUIRE_REAL_DATAMATE=1), but bootstrap failed. "
        f"DKI_DATAMATE_RUNTIME={configured!r}; resolved runtime={resolved!r}. "
        "Expected a runtime root containing both 'ops' and 'python-executor', "
        "with the datamate package importable from <runtime>/python-executor. "
        f"Cause: {type(error).__name__}: {error}. "
        "Set DKI_DATAMATE_RUNTIME to the extracted runtime root, or unset "
        "DKI_REQUIRE_REAL_DATAMATE for shim-based development."
    )


# The published DataMate wheel imports python-magic/unstructured at module load;
# its native magic binding can access-violate on Windows before Python can catch
# an exception. Use the contract shim there; real platform verification runs on
# Linux. Strict acceptance mode explicitly opts in even on Windows.
_REQUIRE_REAL_DATAMATE = _require_real_datamate()
_RUNTIME_ENABLED = os.getenv("DKI_USE_DATAMATE_RUNTIME", "1").strip() != "0"
_USE_REAL_DATAMATE = _RUNTIME_ENABLED and (
    os.name != "nt" or _REQUIRE_REAL_DATAMATE
)
_DATAMATE_RUNTIME_DIR = None
_DATAMATE_BOOTSTRAP_ERROR = None

try:
    if _REQUIRE_REAL_DATAMATE and not _RUNTIME_ENABLED:
        raise RuntimeError(
            "DKI_USE_DATAMATE_RUNTIME=0 conflicts with "
            "DKI_REQUIRE_REAL_DATAMATE=1"
        )
    if not _USE_REAL_DATAMATE:
        raise ImportError("real DataMate runtime disabled on this platform")
    _DATAMATE_RUNTIME_DIR = find_datamate_runtime()
    if _DATAMATE_RUNTIME_DIR is not None:
        _prepend_sys_path(_DATAMATE_RUNTIME_DIR / "python-executor")
    elif _REQUIRE_REAL_DATAMATE:
        raise FileNotFoundError(
            "no valid DataMate runtime root was found; both ops/ and "
            "python-executor/ are required"
        )
    from datamate.core.base_op import (  # type: ignore  # noqa: F401
        OPERATORS,
        BaseOp,
        Filter,
        LLM,
        Mapper,
        Slicer,
    )
    from datamate.core.base_op import FileExporter as _RealFileExporter  # noqa: F401
    from datamate.core.constant import Fields  # type: ignore  # noqa: F401

    FileExporter = _RealFileExporter
    DATAMATE_AVAILABLE = True
except Exception as _e:  # ImportError 或其重型依赖缺失
    _DATAMATE_BOOTSTRAP_ERROR = _e
    if _REQUIRE_REAL_DATAMATE:
        raise RuntimeError(
            _bootstrap_failure_message(_e, _DATAMATE_RUNTIME_DIR)
        ) from _e
    from dki.compat.datamate_base import (  # noqa: F401
        OPERATORS,
        BaseOp,
        Fields,
        FileExporter,
        Filter,
        LLM,
        Mapper,
        Slicer,
    )

    DATAMATE_AVAILABLE = False
    logger.info("DataMate runtime 不可用（%s），使用 dki.compat 契约兼容层。", _e)

_BUILTIN_OPS_LOADED = False


def _register_shim_builtin_ops() -> bool:
    """Register the limited config-compatible built-in subset in shim mode."""
    if "FileWithShortOrLongLengthFilter" in OPERATORS.modules:
        return True

    class FileWithShortOrLongLengthFilter(Filter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            bounds = kwargs.get("fileLength", [0, sys.maxsize])
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError("fileLength must be [minimum, maximum]")
            self.minimum, self.maximum = int(bounds[0]), int(bounds[1])
            if self.minimum < 0 or self.maximum < self.minimum:
                raise ValueError("fileLength bounds are invalid")

        def execute(self, sample):
            self.read_file_first(sample)
            if not self.minimum <= len(sample.get(self.text_key, "")) <= self.maximum:
                sample[self.text_key] = ""
            return sample

    OPERATORS.register_module(
        module_name="FileWithShortOrLongLengthFilter",
        module_cls=FileWithShortOrLongLengthFilter,
    )
    logger.info("registered shim built-in FileWithShortOrLongLengthFilter")
    return True


def load_builtin_ops() -> bool:
    """注册 DataMate 内置算子到 OPERATORS（幂等；返回是否成功）。

    注册是懒加载的（Registry 中只存 dotted module path），只有真正被流水线
    使用的算子才会 import 其 process.py，因此不会拉起 paddle/mineru 等重依赖。
    """
    global _BUILTIN_OPS_LOADED
    if _require_real_datamate() and not DATAMATE_AVAILABLE:
        error = _DATAMATE_BOOTSTRAP_ERROR or RuntimeError(
            "dki.dmcompat is already loaded with the shim; restart the "
            "process after enabling the real runtime"
        )
        raise RuntimeError(
            _bootstrap_failure_message(error, _DATAMATE_RUNTIME_DIR)
        ) from error
    if _BUILTIN_OPS_LOADED:
        return True
    if not DATAMATE_AVAILABLE:
        _BUILTIN_OPS_LOADED = _register_shim_builtin_ops()
        return _BUILTIN_OPS_LOADED

    runtime_dir = find_datamate_runtime()
    if runtime_dir is None:
        error = FileNotFoundError(
            "no valid DataMate runtime root was found for built-in ops"
        )
        if _require_real_datamate():
            raise RuntimeError(_bootstrap_failure_message(error)) from error
        logger.warning("未找到 DataMate runtime 目录（可设 DKI_DATAMATE_RUNTIME）。")
        return False

    try:
        _prepend_sys_path(runtime_dir)
        import ops.filter  # noqa: F401
        import ops.mapper  # noqa: F401
        import ops.slicer  # noqa: F401
    except Exception as e:
        if _require_real_datamate():
            raise RuntimeError(
                _bootstrap_failure_message(e, runtime_dir)
            ) from e
        logger.warning("DataMate 内置算子注册失败：%s", e)
        return False
    _BUILTIN_OPS_LOADED = True
    logger.info("DataMate 内置算子已注册，共 %d 个。", len(OPERATORS.modules))
    return True


def register_dki_ops() -> bool:
    """注册本项目全部医疗算子（幂等；子包缺失时告警不阻断）。"""
    try:
        import dki.operators  # noqa: F401  触发各算子包的 register_module
        return True
    except Exception as e:
        logger.warning("dki 医疗算子注册不完整：%s", e)
        return False
