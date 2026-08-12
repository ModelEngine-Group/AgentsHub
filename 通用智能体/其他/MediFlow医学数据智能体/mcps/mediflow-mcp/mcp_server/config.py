"""
MCP 服务配置加载模块。

该模块集中读取数据库路径、平台地址、可选前端入口和运行参数。
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = {}
_cfg_path = ROOT / 'config.yaml'
if _cfg_path.exists():
    with open(_cfg_path, 'r', encoding='utf-8') as fh:
        CONFIG = yaml.safe_load(fh) or {}

def _load_runtime_env() -> None:
    env_path = ROOT / '.env.runtime'
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")

_load_runtime_env()

def _secret_value(env_name: str, file_env_name: str) -> str:
    v = os.environ.get(env_name, '').strip()
    if v: return v
    key_file = os.environ.get(file_env_name, '').strip()
    if key_file:
        try: return Path(key_file).read_text(encoding='utf-8').strip()
        except OSError: pass
    return ''

def _project_db_path(
    config_value,
    fallback: str,
    env_name: str | tuple[str, ...] | None = None,
) -> str:
    """把相对数据库路径解析到项目根目录，并允许运行态覆盖。"""

    env_names = (env_name,) if isinstance(env_name, str) else (env_name or ())
    value = next(
        (os.environ.get(name, '').strip() for name in env_names if os.environ.get(name, '').strip()),
        '',
    )
    value = value or str(config_value or '').strip()
    path = Path(value).expanduser() if value else Path(fallback)
    return str(path if path.is_absolute() else ROOT / path)

# 大模型配置
LLM_API_KEY = _secret_value('CCF_LLM_API_KEY', 'CCF_LLM_API_KEY_FILE')
_LLM_CONFIG = CONFIG.get('llm') or CONFIG.get('ollama') or {}
LLM_BASE_URL = os.environ.get('CCF_LLM_BASE_URL', _LLM_CONFIG.get('base_url', 'https://api.deepseek.com/v1/chat/completions'))
LLM_MODEL = os.environ.get('CCF_LLM_MODEL', _LLM_CONFIG.get('model', 'deepseek-chat'))

# DataMate 服务地址
DATAMATE_BASE = os.environ.get('CCF_DATAMATE_BASE', CONFIG.get('datamate', {}).get('api_base', 'http://localhost:18000'))
DATAMATE_GATEWAY = os.environ.get('CCF_DATAMATE_GATEWAY', CONFIG.get('datamate', {}).get('gateway_base', 'http://localhost:8080'))
DATASET_VOLUME = os.environ.get('CCF_DATASET_VOLUME', CONFIG.get('datamate', {}).get('dataset_volume', ''))
MINERU_API = os.environ.get(
    'CCF_MINERU_API',
    CONFIG.get('datamate', {}).get('mineru_api', 'https://mineru.net/api/v1/agent'),
)
SUDO_PW = os.environ.get('CCF_SUDO_PW', '')

# 知识图谱与分析库路径
_KG_CONFIG = CONFIG.get('kg', {}) or {}
KG_DB = _project_db_path(
    _KG_CONFIG.get('sqlite_path'),
    'data/task2_medical_kg.db',
    env_name=('CCF_TASK2_KG_DB', 'CCF_MEDICAL_KG_DB'),
)

# 分析服务只有一个权威分析库。兼容旧配置中的两个字段，但不再把自然语言查询
# 指向另一份可能不存在或内容不同的数据库。
_ANALYTICS_CONFIG_VALUE = (
    _KG_CONFIG.get('analytics_db_path')
    or _KG_CONFIG.get('analytics_sqlite_path')
    or _KG_CONFIG.get('sql_db_path')
)
ANALYTICS_DB = _project_db_path(
    _ANALYTICS_CONFIG_VALUE,
    'data/task3_analytics.db',
    env_name=('CCF_TASK3_ANALYTICS_DB', 'CCF_ANALYTICS_DB'),
)
SQL_DB = ANALYTICS_DB
