import logging
import pandas as pd
import json
import csv
import re
import sys
import os
from datetime import datetime
from typing import Dict, Optional, Any, List

from xtask.benchmarks.base_benchmark import PerformanceMetrics, ResultSummary


class AisbenchResult:
    """Aisbench测试结果抽象类"""
    
    def __init__(self):
        self.parameters: Dict[str, Dict[str, PerformanceMetrics]] = {}
        self.summary: Optional[ResultSummary] = None
        self.metadata: Dict[str, Any] = {}
        self.raw_data: Optional[pd.DataFrame] = None
        
    def add_parameter(self, param_name: str, stage: str, metrics: PerformanceMetrics):
        """添加性能参数指标"""
        if param_name not in self.parameters:
            self.parameters[param_name] = {}
        self.parameters[param_name][stage] = metrics
    
    def calculate_summary(self):
        """计算结果摘要"""
        if not self.parameters:
            raise ValueError("没有可用的性能参数数据")
        
        # 获取total阶段的指标
        total_params = {}
        for param, stages in self.parameters.items():
            if 'total' in stages:
                total_params[param] = stages['total']
        
        # 提取关键指标
        input_tokens = total_params.get('InputTokens', PerformanceMetrics(0,0,0,0,0,0,0,0))
        output_tokens = total_params.get('OutputTokens', PerformanceMetrics(0,0,0,0,0,0,0,0))
        ttft = total_params.get('TTFT', PerformanceMetrics(0,0,0,0,0,0,0,0))
        tpot = total_params.get('TPOT', PerformanceMetrics(0,0,0,0,0,0,0,0))
        e2el = total_params.get('E2EL', PerformanceMetrics(0,0,0,0,0,0,0,0))
        output_throughput = total_params.get('OutputTokenThroughput', PerformanceMetrics(0,0,0,0,0,0,0,0))
        
        # 创建摘要
        self.summary = ResultSummary(
            total_requests=int(e2el.n if hasattr(e2el, 'n') else 0),
            concurrency=1,  # 需要从其他地方获取
            max_concurrency=1,  # 需要从其他地方获取
            request_throughput=float(e2el.n / (e2el.average / 1000)) if e2el.average > 0 else 0.0,
            input_tokens_avg=input_tokens.average,
            output_tokens_avg=output_tokens.average,
            output_token_throughput=output_throughput.average,
            ttft_avg=ttft.average,
            ttft_p90=ttft.p90,
            tpot_avg=tpot.average,
            tpot_p90=tpot.p90,
            e2el_avg=e2el.average,
            e2el_p90=e2el.p90
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'parameters': {},
            'summary': None,
            'metadata': self.metadata
        }
        
        # 转换参数
        for param_name, stages in self.parameters.items():
            result['parameters'][param_name] = {}
            for stage_name, metrics in stages.items():
                result['parameters'][param_name][stage_name] = {
                    'average': metrics.average,
                    'min': metrics.min,
                    'max': metrics.max,
                    'median': metrics.median,
                    'p75': metrics.p75,
                    'p90': metrics.p90,
                    'p99': metrics.p99,
                    'n': metrics.n
                }
        
        # 转换摘要
        if self.summary:
            result['summary'] = {
                'total_requests': self.summary.total_requests,
                'concurrency': self.summary.concurrency,
                'max_concurrency': self.summary.max_concurrency,
                'request_throughput': self.summary.request_throughput,
                'input_tokens_avg': self.summary.input_tokens_avg,
                'output_tokens_avg': self.summary.output_tokens_avg,
                'output_token_throughput': self.summary.output_token_throughput,
                'ttft_avg': self.summary.ttft_avg,
                'ttft_p90': self.summary.ttft_p90,
                'tpot_avg': self.summary.tpot_avg,
                'tpot_p90': self.summary.tpot_p90,
                'e2el_avg': self.summary.e2el_avg,
                'e2el_p90': self.summary.e2el_p90
            }
        
        return result
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """转换为JSON格式"""
        result_dict = self.to_dict()
        json_str = json.dumps(result_dict, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def print_summary(self, detailed: bool = False):
        """打印结果摘要"""
        if not self.summary:
            self.calculate_summary()
        
        logging.info("=" * 80)
        logging.info("AISBENCH TEST RESULT SUMMARY")
        logging.info("=" * 80)
        
        # 打印关键指标
        logging.info("  📊 KEY METRICS:")
        logging.info(f"{'Metric':<30} {'Value':<20} {'Unit':<10}")
        logging.info("-" * 80)
        logging.info(f"{'Total Requests':<30} {self.summary.total_requests:<20} {'count'}")
        logging.info(f"{'Request Throughput':<30} {self.summary.request_throughput:.2f} {'req/s'}")
        logging.info(f"{'Avg Input Tokens':<30} {self.summary.input_tokens_avg:.2f} {'tokens'}")
        logging.info(f"{'Avg Output Tokens':<30} {self.summary.output_tokens_avg:.2f} {'tokens'}")
        logging.info(f"{'Output Token Throughput':<30} {self.summary.output_token_throughput:.2f} {'tokens/s'}")
        logging.info(f"{'TTFT (Average)':<30} {self.summary.ttft_avg:.2f} {'ms'}")
        logging.info(f"{'TTFT (P90)':<30} {self.summary.ttft_p90:.2f} {'ms'}")
        logging.info(f"{'TPOT (Average)':<30} {self.summary.tpot_avg:.2f} {'ms'}")
        logging.info(f"{'TPOT (P90)':<30} {self.summary.tpot_p90:.2f} {'ms'}")
        logging.info(f"{'E2E Latency (Average)':<30} {self.summary.e2el_avg:.2f} {'ms'}")
        logging.info(f"{'E2E Latency (P90)':<30} {self.summary.e2el_p90:.2f} {'ms'}")
        
        if detailed and self.parameters:
            logging.info("  📈  DETAILED PERFORMANCE METRICS:")
            logging.info("-" * 60)
            for param_name, stages in self.parameters.items():
                logging.info(f"  {param_name}:")
                for stage_name, metrics in stages.items():
                    logging.info(f"  Stage: {stage_name}")
                    logging.info(f"    Average: {metrics.average:.2f} ms | Min: {metrics.min:.2f} ms | "
                          f"Max: {metrics.max:.2f} ms | N: {metrics.n}")
                    logging.info(f"    Median: {metrics.median:.2f} ms | P75: {metrics.p75:.2f} ms | "
                          f"P90: {metrics.p90:.2f} ms | P99: {metrics.p99:.2f} ms")
        
        logging.info("=" * 80)


class AisbenchResultParser:
    """Aisbench结果解析器"""
    
    @staticmethod
    def parse_csv(filepath: str) -> AisbenchResult:
        """
        从CSV文件解析结果
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            AisbenchResult对象
        """
        # 读取CSV文件
        df = pd.read_csv(filepath)
        
        # 创建结果对象
        result = AisbenchResult()
        result.raw_data = df
        result.metadata = {
            'source_file': filepath,
            'parse_time': datetime.now().isoformat(),
            'total_rows': len(df)
        }
        
        # 解析每一行
        for _, row in df.iterrows():
            param_name = str(row['Performance Parameters']).strip()
            stage = str(row['Stage']).strip().lower()
            
            # 解析数值（去掉单位）
            def parse_value(value):
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # 提取数字部分
                    match = re.search(r'[-]?\d*\.?\d+', value)
                    if match:
                        return float(match.group())
                return 0.0
            
            # 创建指标对象
            metrics = PerformanceMetrics(
                average=parse_value(row.get('Average', 0)),
                min=parse_value(row.get('Min', 0)),
                max=parse_value(row.get('Max', 0)),
                median=parse_value(row.get('Median', 0)),
                p75=parse_value(row.get('P75', 0)),
                p90=parse_value(row.get('P90', 0)),
                p99=parse_value(row.get('P99', 0)),
                n=int(parse_value(row.get('N', 0)))
            )
            
            # 添加到结果
            result.add_parameter(param_name, stage, metrics)
        
        # 计算摘要
        result.calculate_summary()
        
        return result
    
    @staticmethod
    def parse_directory(directory_path: str) -> Dict[str, AisbenchResult]:
        """
        解析目录中的所有CSV文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            文件名到结果对象的映射
        """
        results = {}
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory_path, filename)
                try:
                    result = AisbenchResultParser.parse_csv(filepath)
                    results[filename] = result
                except Exception as e:
                    logging.info(f"解析文件 {filename} 时出错: {e}")
        
        return results


class EnhancedResult(AisbenchResult):
    """增强版结果类，支持更多分析功能"""
    
    def __init__(self):
        super().__init__()
        self.percentiles: Dict[str, List[float]] = {}
        self.confidence_intervals: Dict[str, Dict[str, float]] = {}
        
    def calculate_percentiles(self):
        """计算百分位数分布"""
        for param_name, stages in self.parameters.items():
            for stage_name, metrics in stages.items():
                key = f"{param_name}_{stage_name}"
                self.percentiles[key] = [
                    metrics.min,
                    metrics.p75,
                    metrics.median,
                    metrics.p90,
                    metrics.p99,
                    metrics.max
                ]
    
    def calculate_confidence_intervals(self, confidence_level: float = 0.95):
        """计算置信区间"""
        z_score = 1.96  # 95%置信水平对应的z分数
        
        for param_name, stages in self.parameters.items():
            for stage_name, metrics in stages.items():
                if metrics.n > 1:
                    key = f"{param_name}_{stage_name}"
                    # 简化计算：假设正态分布
                    std_error = (metrics.max - metrics.min) / (2 * z_score)
                    margin = z_score * std_error
                    
                    self.confidence_intervals[key] = {
                        'lower': metrics.average - margin,
                        'upper': metrics.average + margin,
                        'confidence_level': confidence_level
                    }
    
    def print_statistical_report(self):
        """打印统计报告"""
        logging.info("  📊  STATISTICAL ANALYSIS REPORT")
        logging.info("=" * 80)
        
        if not self.summary:
            self.calculate_summary()
        
        # 打印统计显著性
        logging.info("1. 数据质量检查:")
        logging.info(f"   总样本数: {self.summary.total_requests}")
        
        # 检查数据分布
        ttft_data = self.parameters.get('TTFT', {}).get('total')
        if ttft_data:
            cv = (ttft_data.max - ttft_data.min) / ttft_data.average if ttft_data.average > 0 else 0
            logging.info(f"   TTFT变异系数: {cv:.2%}")
        
        # 打印百分位数报告
        if not self.percentiles:
            self.calculate_percentiles()
        
        logging.info("2. 百分位数分布:")
        for key, percentiles in self.percentiles.items():
            if 'total' in key:
                logging.info(f"   {key}:")
                logging.info(f"     Min={percentiles[0]:.2f} | P75={percentiles[1]:.2f} | "
                      f"Median={percentiles[2]:.2f} | P90={percentiles[3]:.2f} | "
                      f"P99={percentiles[4]:.2f} | Max={percentiles[5]:.2f}")
        
        # 打印建议
        logging.info("3. 性能优化建议:")
        ttft = self.parameters.get('TTFT', {}).get('total')
        tpot = self.parameters.get('TPOT', {}).get('total')
        
        if ttft and tpot:
            if ttft.p90 > 1000:  # TTFT P90 > 1秒
                logging.info("   ⚠️  TTFT P90超过1秒，建议检查模型加载和预热")
            if tpot.p90 > 50:  # TPOT P90 > 50ms
                logging.info("   ⚠️  TPOT P90较高，建议优化推理计算")
        
        # 吞吐量分析
        if self.summary.request_throughput < 10:
            logging.info("   ⚠️  请求吞吐量较低，考虑增加并发")
        
        logging.info("\n" + "=" * 80)


def aisbench_post_process(task_config, output_path, init_entries):
    aisbench_output_dir = find_aisbench_directory(output_path, init_entries)
    logging.info(f"aisbench perf的结果目录: {aisbench_output_dir}")
    if aisbench_output_dir and os.path.exists(os.path.join(output_path, aisbench_output_dir)):
        aisbench_csv = get_aisbench_csv(os.path.join(output_path, aisbench_output_dir), 
                                        task_config.get("benchmark").get("perf").get("api_type").replace("_", "-"))
    else:
        logging.error("get aisbench output path failed.")
        return
    
    # 解析结果
    if aisbench_csv and os.path.exists(aisbench_csv):
        parser = AisbenchResultParser()
        result = parser.parse_csv(aisbench_csv)
        
        # 生成摘要报告
        logging.info("  📄  生成测试报告...")
        
        # 保存JSON结果
        json_filename = task_config.get("case_name").lower() + ".json"
        json_path = os.path.join(output_path, json_filename)
        result.to_json(json_path)
        
        # 保存摘要文本
        summary_filename = task_config.get("case_name").lower() + '_summary.txt'
        summary_path = os.path.join(output_path, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            import sys
            from io import StringIO
            
            # 重定向输出到文件
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                result.print_summary(detailed=True)
                summary_text = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            f.write(summary_text)
        
        # 使用增强版结果类
        if task_config.get("enhanced_result"):
            logging.info("  🔬 增强分析:")
            enhanced_result = EnhancedResult()
            enhanced_result.parameters = result.parameters
            enhanced_result.calculate_summary()
            enhanced_result.print_statistical_report()
        
        logging.info(f"  ✅  结果已保存:")
        logging.info(f"   Result JSON file: {json_path}")
        logging.info(f"   Result Summary file: {summary_path}")
    else:
        logging.error(f"aisbench csv: {aisbench_csv} is not exist. aisbench perf run failed.")
        result = None
    
    return result
    

def aisbench_acc_post_process(task_config, output_path, init_entries):
    aisbench_output_dir = find_aisbench_directory(output_path, init_entries, aisbench_type="acc")
    logging.info(f"aisbench acc的结果目录: {aisbench_output_dir}")
    api_type= task_config.get("benchmark").get("acc").get("api_type").replace("_", "-")
    if aisbench_output_dir and os.path.exists(os.path.join(output_path, aisbench_output_dir)):
        aisbench_csv = get_aisbench_csv(os.path.join(output_path, aisbench_output_dir), api_type, output_type="accaurcy")
    else:
        logging.error("get aisbench output path failed.")
        return
    
    if os.path.exists(aisbench_csv):
        result_value = read_csv_value(aisbench_csv, api_type)
        if result_value and len(result_value) == 1:
            return result_value[0]
        else:
            logging.error("")
            return 
    else:
        logging.error(f"aisbench csv: {aisbench_csv} is not exist.")
        result = None
    return result


def find_aisbench_directory(directory_path=".", init_entries=list(), aisbench_type="perf"):
    """
    查找指定目录下最新生成的日期格式目录
    
    Args:
        directory_path (str): 要搜索的目录路径，默认为当前目录
    
    Returns:
        str or False: 最新日期目录名称，如果没有找到返回False
    """
    
    # 日期目录的正则表达式模式 (格式: YYYYMMDD_HHMMSS)
    date_pattern = re.compile(r'^\d{8}_\d{6}$')
    
    # 获取目录下的所有条目
    try:
        entries = os.listdir(directory_path)
    except FileNotFoundError:
        logging.info(f"错误: 目录 '{directory_path}' 不存在")
        return False
    except PermissionError:
        logging.info(f"错误: 没有权限访问目录 '{directory_path}'")
        return False
    
    # 过滤出符合日期格式的目录
    date_dirs = []
    for entry in entries:
        entry_path = os.path.join(directory_path, entry)
        if os.path.isdir(entry_path) and date_pattern.match(entry) and entry not in init_entries:
            date_dirs.append(entry)
    
    # 如果没有找到日期目录
    if not date_dirs:
        logging.info("未找到日期格式的目录 (格式: YYYYMMDD_HHMMSS)")
        return False
    
    # 按日期排序（从新到旧）
    def extract_datetime(dir_name):
        try:
            # 解析目录名中的日期时间
            return datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
        except ValueError:
            # 如果解析失败，返回一个很旧的日期
            return datetime.min
    
    # 按日期排序，最新的在前
    date_dirs.sort(key=extract_datetime, reverse=True)
    
    # 返回最新的目录
    for data_dir in date_dirs:
        if "summary" in os.listdir(os.path.join(directory_path, data_dir)) and aisbench_type == "acc":
            return data_dir
        elif "performances" in os.listdir(os.path.join(directory_path, data_dir)) and aisbench_type == "perf":
            return data_dir
    
    logging.error("can not match aisbech result dir.")


def get_aisbench_csv(aisbench_output_path, api_type, output_type="performances"):
    if output_type == "performances":
        csv_path = os.path.join(aisbench_output_path, output_type, api_type.replace("_", "-"))
    elif output_type == "accaurcy":
        csv_path = os.path.join(aisbench_output_path, "summary")
    else:
        logging.error(f"output_type: {output_type} is not supported.")
        return
    
    if os.path.exists(aisbench_output_path) and os.path.exists(csv_path):
        entries = os.listdir(csv_path)
        for entry in entries:
            entry_path = os.path.join(csv_path, entry)
            if entry_path.endswith(".csv"):
                return entry_path
        logging.error(f"Can't find aisbench csv output. {output_type} run failed.")
        return
    else:
        logging.error(f"aisbench output path: {csv_path} is not exist.")
        return


def read_csv_value(file_path, target_column):
    """
    读取 CSV 文件并返回特定列的值。
    
    Args:
        file_path (str): CSV 文件路径。
        target_column (str): 要读取的列名。
    
    Returns:
        list: 包含目标列所有值的列表。
    """
    values = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file) # 每一行是一个 OrderedDict
        for row in reader:
            # 通过列名（键）获取对应单元格的值
            if target_column in row:
                values.append(row[target_column])
            else:
                logging.error(f"警告：文件中未找到列名 '{target_column}'")
                return []
    return values
