# -*- coding: utf-8 -*-
"""
CSV读取算子
从CSV文件中读取医疗数据
"""

from typing import Dict, Any
import csv
import io

from datamate.core.base_op import Mapper


class CsvReader(Mapper):
    """
    CSV文件读取算子
    
    支持多种编码格式和分隔符，自动处理表头。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 读取配置参数
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.delimiter = kwargs.get('delimiter', ',')
        self.has_header = kwargs.get('hasHeader', 'true') == 'true'
        
        # 验证参数
        if self.delimiter == r'\t':
            self.delimiter = '\t'
    
    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行CSV文件读取
        
        Args:
            sample: 输入样本，包含以下字段：
                - text: CSV文件内容（文本模式）
                - fileName: 文件名
                - filePath: 文件路径
                
        Returns:
            处理后的样本，包含：
                - data: 解析后的数据列表
                - row_count: 行数
                - column_names: 列名列表
                - metadata: 元数据信息
        """
        try:
            # 从sample中获取数据
            csv_content = sample.get('text', '')
            file_name = sample.get('fileName', 'unknown.csv')
            
            if not csv_content:
                raise ValueError("CSV内容为空")
            
            # 解析CSV
            reader = csv.reader(
                io.StringIO(csv_content),
                delimiter=self.delimiter
            )
            
            rows = list(reader)
            
            if not rows:
                raise ValueError("CSV文件为空")
            
            # 分离表头和数据
            if self.has_header and rows:
                headers = rows[0]
                data_rows = rows[1:]
            else:
                headers = [f"column_{i}" for i in range(len(rows[0]))]
                data_rows = rows
            
            # 构建结果
            result = {
                'data': data_rows,
                'headers': headers,
                'row_count': len(data_rows),
                'column_count': len(headers),
                'file_name': file_name,
                'encoding': self.encoding,
                'delimiter': self.delimiter,
                'has_header': self.has_header,
                'status': 'success'
            }
            
            # 添加到sample
            sample['csv_data'] = result
            sample['processed'] = True
            
            return sample
            
        except Exception as e:
            # 错误处理
            sample['error'] = str(e)
            sample['status'] = 'error'
            sample['processed'] = False
            return sample
