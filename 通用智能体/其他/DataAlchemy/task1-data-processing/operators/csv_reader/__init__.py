# -*- coding: utf-8 -*-
"""
CSV读取算子 - DataMate Operator
从CSV文件中读取医疗数据
"""

from datamate.core.base_op import OPERATORS

OPERATORS.register_module(
    module_name='CsvReader',
    module_path="task1_data_processing.operators.csv_reader.process"
)
