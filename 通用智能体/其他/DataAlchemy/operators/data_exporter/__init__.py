# -*- coding: utf-8 -*-
"""DataExporter 算子包 - 双模式兼容"""

from .process import DataExporter as _DataExporter, ValidationError, ProcessingError, DataExporterError

try:
    from datamate.core.base_op import Mapper, OPERATORS

    class DataExporter(Mapper):
        """DataMate Mapper 兼容包装"""
        name = 'DataExporter'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._inner = _DataExporter()
            self._params = {k: kwargs[k] for k in [
                'outputFormat', 'outputDir', 'outputFileName',
                'encoding', 'includeHeader', 'indexColumn', 'overwrite'
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
        module_name='DataExporter',
        module_path='datamate.ops.user.data_exporter'
    )
except ImportError:
    DataExporter = _DataExporter

__all__ = ['DataExporter', 'ValidationError', 'ProcessingError', 'DataExporterError']
__version__ = '1.0.0'
