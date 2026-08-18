# -*- coding: utf-8 -*-
"""KGTripleGenerator 算子包 - 双模式兼容"""

from .process import KGTripleGenerator as _KGTripleGenerator, \
    TripleGeneratorError, ValidationError, ProcessingError

try:
    from datamate.core.base_op import Mapper, OPERATORS

    class KGTripleGenerator(Mapper):
        """DataMate Mapper 兼容包装"""
        name = 'KGTripleGenerator'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._inner = _KGTripleGenerator()

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
        module_name='KGTripleGenerator',
        module_path='datamate.ops.user.kg_triple_generator'
    )
except ImportError:
    KGTripleGenerator = _KGTripleGenerator

__all__ = ['KGTripleGenerator', 'TripleGeneratorError', 'ValidationError', 'ProcessingError']
__version__ = '1.0.0'
