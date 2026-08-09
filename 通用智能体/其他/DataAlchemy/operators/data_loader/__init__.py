# -*- coding: utf-8 -*-
"""DataLoaderMapper 算子包 - 双模式兼容"""

from .process import DataLoaderMapper as _DataLoaderMapper

try:
    from datamate.core.base_op import Mapper, OPERATORS

    class DataLoaderMapper(Mapper):
        """DataMate Mapper 兼容包装 - 透传模式"""
        name = 'DataLoaderMapper'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def execute(self, sample):
            return sample

    OPERATORS.register_module(
        module_name='DataLoaderMapper',
        module_path='datamate.ops.user.data_loader'
    )
except ImportError:
    DataLoaderMapper = _DataLoaderMapper

__all__ = ['DataLoaderMapper']
