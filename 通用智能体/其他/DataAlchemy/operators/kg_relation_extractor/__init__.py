# -*- coding: utf-8 -*-
"""KGRelationExtractor 算子包 - 双模式兼容"""

from .process import KGRelationExtractor as _KGRelationExtractor, \
    RelationExtractorError, ValidationError, ProcessingError

try:
    from datamate.core.base_op import Mapper, OPERATORS

    class KGRelationExtractor(Mapper):
        """DataMate Mapper 兼容包装"""
        name = 'KGRelationExtractor'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._inner = _KGRelationExtractor()

        def execute(self, sample):
            data = sample.get(self.data_key, {})
            if not data:
                return sample
            result = self._inner.process(data)
            sample[self.data_key] = result
            return sample

        def process(self, input_data, params):
            return self._inner.process(input_data, params)

    OPERATORS.register_module(
        module_name='KGRelationExtractor',
        module_path='datamate.ops.user.kg_relation_extractor'
    )
except ImportError:
    KGRelationExtractor = _KGRelationExtractor

__all__ = ['KGRelationExtractor', 'RelationExtractorError', 'ValidationError', 'ProcessingError']
__version__ = '1.0.0'
