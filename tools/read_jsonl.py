#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL文件处理工具
用于加载和遍历JSONL文件中的每一行数据
"""

import json
import sys
from typing import Dict, List, Any, Generator
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JSONLProcessor:
    """JSONL文件处理器"""
    
    def __init__(self, file_path: str):
        """
        初始化JSONL处理器
        
        Args:
            file_path: JSONL文件路径
        """
        self.file_path = file_path
        self.total_lines = 0
        self.valid_lines = 0
        self.invalid_lines = 0
    
    def read_jsonl_file(self) -> Generator[Dict[str, Any], None, None]:
        """
        读取JSONL文件并逐行解析
        
        Yields:
            每一行的JSON对象
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    self.total_lines += 1
                    line = line.strip()
                    
                    # 跳过空行
                    if not line:
                        continue
                    
                    try:
                        # 解析JSON行
                        data = json.loads(line)
                        self.valid_lines += 1
                        yield data
                    except json.JSONDecodeError as e:
                        self.invalid_lines += 1
                        logger.warning(f"第{line_num}行解析失败: {e}")
                        logger.warning(f"错误行内容: {line[:100]}...")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"文件不存在: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"读取文件时发生错误: {e}")
            raise
    
    def process_file(self, callback=None):
        """
        处理JSONL文件的每一行
        
        Args:
            callback: 处理每一行数据的回调函数
        """
        logger.info(f"开始处理文件: {self.file_path}")
        
        for data in self.read_jsonl_file():
            if callback:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"处理数据时发生错误: {e}")
                    logger.error(f"错误数据: {data}")
                    continue
        
        logger.info(f"文件处理完成")
        logger.info(f"总行数: {self.total_lines}, 有效行: {self.valid_lines}, 无效行: {self.invalid_lines}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取文件处理统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_lines': self.total_lines,
            'valid_lines': self.valid_lines,
            'invalid_lines': self.invalid_lines
        }
    
    def filter_data(self, filter_func) -> List[Dict[str, Any]]:
        """
        根据过滤函数筛选数据
        
        Args:
            filter_func: 过滤函数
            
        Returns:
            过滤后的数据列表
        """
        filtered_data = []
        
        def filter_callback(data):
            if filter_func(data):
                filtered_data.append(data)
        
        self.process_file(filter_callback)
        return filtered_data
    
    def count_by_field(self, field_name: str) -> Dict[Any, int]:
        """
        统计指定字段的值出现次数
        
        Args:
            field_name: 要统计的字段名
            
        Returns:
            字段值及其出现次数的字典
        """
        counter = {}
        
        def count_callback(data):
            if field_name in data:
                value = data[field_name]
                counter[value] = counter.get(value, 0) + 1
        
        self.process_file(count_callback)
        return counter


def print_jsonl_data(data: Dict[str, Any]):
    """
    打印JSON数据的示例回调函数
    
    Args:
        data: JSON数据字典
    """
    # 打印数据的基本信息
    print(f"\n{'='*50}")
    
    # 打印所有字段
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 100:
            # 对于长字符串，只显示前100个字符
            print(f"{key}: {value[:100]}... (长度: {len(value)})")
        else:
            print(f"{key}: {value}")


def main():
    """主函数"""
    print("JSONL文件处理工具")
    print("=" * 50)
    
    # 示例用法
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 默认使用sample.jsonl文件
        file_path = "sample.jsonl"
    
    try:
        # 创建处理器实例
        processor = JSONLProcessor(file_path)
        
        print(f"\n1. 遍历所有数据:")
        processor.process_file(print_jsonl_data)
        
        print(f"\n2. 统计信息:")
        stats = processor.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n3. 按字段统计 - query字段:")
        query_counts = processor.count_by_field("query")
        for query, count in query_counts.items():
            print(f"  {query}: {count}次")
        
        print(f"\n4. 过滤数据示例 - 查找query包含'olay'的数据:")
        filtered_data = processor.filter_data(lambda x: 'query' in x and 'olay' in x['query'].lower())
        print(f"  找到{len(filtered_data)}条包含'olay'的记录")
        for data in filtered_data:
            print(f"  - {data.get('query')} (uuid: {data.get('uuid')})")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()