"""项目路径与环境配置的唯一入口（对齐 nexent 后端 consts/const.py 的做法：
环境变量只在此处读取，其余模块一律从这里 import）。"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：dki 包的上一层
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env（存在则加载，不存在静默跳过）
load_dotenv(PROJECT_ROOT / ".env")

# ---- LLM（与 nexent/sdk/benchmark 的环境变量命名保持一致） ----
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")

# ---- 目录 ----
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GOLD_DATA_DIR = DATA_DIR / "gold"
OUTPUT_DIR = Path(os.getenv("DKI_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
TASK_DIR = OUTPUT_DIR / "tasks"
PROCESSED_DIR = OUTPUT_DIR / "processed"


# ---- DataMate runtime 目录探测 ----
def find_datamate_runtime() -> Path | None:
    """定位 DataMate 的 runtime 目录（包含 ops/ 与 python-executor/）。

    优先级：环境变量 DKI_DATAMATE_RUNTIME > 本仓库同级 ../datamate/runtime。
    环境变量在每次调用时读取，便于测试、CLI 和长进程切换运行时。
    """
    candidates: list[Path] = []
    configured_runtime = os.getenv("DKI_DATAMATE_RUNTIME", "").strip()
    if configured_runtime:
        candidates.append(Path(configured_runtime).expanduser())
    candidates.append(PROJECT_ROOT.parent / "datamate" / "runtime")
    for candidate in candidates:
        try:
            if (candidate / "ops").is_dir() and (
                candidate / "python-executor"
            ).is_dir():
                return candidate.resolve()
        except OSError:
            continue
    return None


def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        RAW_DATA_DIR,
        GOLD_DATA_DIR,
        OUTPUT_DIR,
        TASK_DIR,
        PROCESSED_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
