# -*- coding: utf-8 -*-
"""KGEntityRecognizer 算子包 - 双模式兼容"""

from .process import KGEntityRecognizer as _KGEntityRecognizer, \
    EntityRecognizerError, ValidationError, ProcessingError

try:
    from datamate.core.base_op import Mapper, OPERATORS

    class KGEntityRecognizer(Mapper):
        """DataMate Mapper 兼容包装"""
        name = 'KGEntityRecognizer'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._inner = _KGEntityRecognizer()

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
        module_name='KGEntityRecognizer',
        module_path='datamate.ops.user.kg_entity_recognizer'
    )
except ImportError:
    KGEntityRecognizer = _KGEntityRecognizer

__all__ = ['KGEntityRecognizer', 'EntityRecognizerError', 'ValidationError', 'ProcessingError']
__version__ = '1.0.0'
