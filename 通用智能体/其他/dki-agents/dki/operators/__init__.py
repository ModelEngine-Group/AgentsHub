"""dki.operators —— 本项目医疗算子包集合（DataMate 自定义算子规范）。

每个子目录都是一个完整的 DataMate 算子包（__init__.py + metadata.yml + process.py），
可通过 scripts/package_operators.py 打包为 zip 直接上传 DataMate 算子市场。

导入本包即完成全部算子到 OPERATORS 注册表的懒注册。
"""

import importlib
import logging

logger = logging.getLogger(__name__)

_OP_PACKAGES = [
    "medical_text_cleaner",
    "medical_section_splitter",
    "medical_entity_extractor",
    "medical_relation_extractor",
    "triple_builder_validator",
    # Release A opt-in candidate: registered so the runner can resolve it by its
    # formal name, but hidden from the agent catalog (metadata catalog_visible:
    # false) and inert unless shadowEnabled + a verified runtime factory.
    "medical_entity_extractor_om_shadow",
]

for _pkg in _OP_PACKAGES:
    try:
        importlib.import_module(f"dki.operators.{_pkg}")
    except ModuleNotFoundError as _e:
        # 并行开发/裁剪部署时允许缺包：只注册存在的算子
        logger.warning("算子包 %s 缺失，跳过注册（%s）", _pkg, _e)
