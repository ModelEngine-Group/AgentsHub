# -*- coding: utf-8 -*-
"""DataTransformer 算子包 - 双模式兼容"""

from .process import DataTransformer as _DataTransformer, ValidationError, ProcessingError, DataTransformerError

try:
    from datamate.core.base_op import Mapper, OPERATORS

    class DataTransformer(Mapper):
        """DataMate Mapper 兼容包装"""
        name = 'DataTransformer'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._inner = _DataTransformer()
            self._params = {k: kwargs[k] for k in [
                'selectFields', 'dropFields', 'renameFields',
                'typeConversions', 'valueMappings', 'filterCondition', 'deriveColumns'
            ] if k in kwargs}
        def execute(self, sample):
            data = sample.get(self.data_key, [])
            if not isinstance(data, list) or not data:
                return sample
            inp = {
                'data': data,
                'schema': {'columns': list(data[0].keys()) if isinstance(data[0], dict) else [], 'row_count': len(data)}
            }
            result = self._inner.process(inp, self._params)
            sample[self.data_key] = result.get('data', [])
            return sample
        def process(self, input_data, params):
            return self._inner.process(input_data, params)

    OPERATORS.register_module(
        module_name='DataTransformer',
        module_cls=DataTransformer  # 直接注册类，不用路径字符串！
    )
except ImportError:
    DataTransformer = _DataTransformer

__all__ = ['DataTransformer', 'ValidationError', 'ProcessingError', 'DataTransformerError']
__version__ = '1.0.0'
