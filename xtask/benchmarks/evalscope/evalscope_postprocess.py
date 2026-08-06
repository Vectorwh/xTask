import json
import logging
import csv
import re
import sys
import os
import shutil
from datetime import datetime
from typing import Dict, Optional, Any, List
from xtask.benchmarks.base_benchmark import PerformanceMetrics, ResultSummary

class EvalscopeResult:
    """Evalscope测试结果抽象类"""
    
    def __init__(self):
        self.parameters: Dict[str, Dict[str, PerformanceMetrics]] = {}
        self.summary: Optional[ResultSummary] = None
        self.metadata: Dict[str, Any] = {}
        self.raw_data: Optional = None
        
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
            failed_requests=self.metadata.get('Failed requests', 0),
            concurrency=self.metadata.get('Number of concurrency', 1),  # evalscope 当前无实际并发数统计
            max_concurrency=self.metadata.get('Number of concurrency', 1),
            #request_throughput=float(e2el.n / (e2el.average / 1000)) if e2el.average > 0 else 0.0,
            request_throughput=self.metadata.get('Request throughput (req/s)', 0),
            input_tokens_avg=input_tokens.average,
            output_tokens_avg=output_tokens.average,
            output_tps_avg=output_throughput.average,
            output_tps_total=output_throughput.average,
            summary_tps_total=self.metadata.get('Total token throughput (tok/s)', 1),
            ttft_avg=ttft.average,
            ttft_p90=ttft.p90,
            ttft_p99=ttft.p99,
            tpot_avg=tpot.average,
            tpot_p90=tpot.p90,
            tpot_p99=tpot.p99,
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
                'failed_requests': self.summary.failed_requests,
                'concurrency': self.summary.concurrency,
                'max_concurrency': self.summary.max_concurrency,
                'request_throughput': self.summary.request_throughput,
                'input_tokens_avg': self.summary.input_tokens_avg,
                'output_tokens_avg': self.summary.output_tokens_avg,
                'output_tps_avg': self.summary.output_tps_avg,
                'output_tps_total': self.summary.output_tps_total,
                'summary_tps_total': self.summary.summary_tps_total,
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
        logging.info("EvalScope TEST RESULT SUMMARY")
        logging.info("=" * 80)
        
        # 打印关键指标
        logging.info("  📊 KEY METRICS:")
        logging.info(f"{'Metric':<30} {'Value':<20} {'Unit':<10}")
        logging.info("-" * 80)
        logging.info(f"{'Total Requests':<30} {self.summary.total_requests:<20} {'count'}")
        logging.info(f"{'Failed Requests':<30} {self.summary.failed_requests:<20} {'count'}")
        logging.info(f"{'Concurrency':<30} {self.summary.concurrency:<20} {'count'}")
        logging.info(f"{'Max Concurrency':<30} {self.summary.max_concurrency:<20} {'count'}")
        logging.info(f"{'Request Throughput':<30} {self.summary.request_throughput:.2f} {'req/s'}")
        logging.info(f"{'Avg Input Tokens':<30} {self.summary.input_tokens_avg:.2f} {'tokens'}")
        logging.info(f"{'Avg Output Tokens':<30} {self.summary.output_tokens_avg:.2f} {'tokens'}")
        logging.info(f"{'Output Token Throughput':<30} {self.summary.output_tps_avg:.2f} {'tokens/s'}")
        logging.info(f"{'Output Token Throughput Total':<30} {self.summary.output_tps_total:.2f} {'tokens/s'}")
        logging.info(f"{'Summary Token Throughput Total':<30} {self.summary.summary_tps_total:.2f} {'tokens/s'}")
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


class EvalscopeResultParser:
    """Evalscope结果解析器"""
    
    @staticmethod
    def parse_perf_json(result_file: list[str], n_count=None) -> EvalscopeResult:
        """
        从JSON文件解析结果
        
        Args:
            result_file: JSON文件路径列表
            
        Returns:
            EvalscopeResult对象
        """

        result = EvalscopeResult()

        p75_metrics = {}
        p90_metrics = {}
        p99_metrics = {}
        summary_metrics = {}

        # 读取JSON文件
        for file in result_file:
            if "benchmark_percentile.json" in file:
                with open(file, 'r', encoding='utf-8') as f:
                    data_per = json.load(f)
               
                p75_metrics = data_per[4]
                p90_metrics = data_per[6]
                p99_metrics = data_per[-1]
    
            if "benchmark_summary.json" in file:
                with open(file, 'r', encoding='utf-8') as f:
                    summary_metrics = json.load(f)

                #summary_metrics = data_sum

        # 构建metrics data
        if p75_metrics and p90_metrics and p99_metrics and summary_metrics:
            TIME_MS = 1000
            # n_bs = float(summary_metrics.get('Number of concurrency', 1))
            # div_bs = float(1 / n_bs) if n_bs > 0.0 else 0.0
            n_count = summary_metrics.get('Total requests', 1)
            rawdata_fields = {
                'E2EL': [('Latency (s)', TIME_MS), ('Average latency (s)', TIME_MS)],
                'TTFT': [('TTFT (s)', TIME_MS), ('Average time to first token (s)', TIME_MS)], 
                'TPOT': [('TPOT (s)', TIME_MS), ('Average time per output token (s)', TIME_MS)],
                'ITL': [('ITL (s)', TIME_MS), ('Average inter-token latency (s)', TIME_MS)],
                'InputTokens': [('Input tokens', 1), ('Average input tokens per request', 1)], 
                'OutputTokens': [('Output tokens', 1), ('Average output tokens per request', 1)],
                'OutputTokenThroughput': [('Output (tok/s)', 1), ('Output token throughput (tok/s)', 1)]
                # 'OutputTokenThroughput': [('Output (tok/s)', div_bs), ('Output token throughput (tok/s)', div_bs)]
            }

            for key, value in rawdata_fields.items():
                param_name = key.strip()
                stage = "total"
                
                # 创建指标对象
                metrics = PerformanceMetrics(
                    average=float(summary_metrics.get(value[1][0])) * value[1][1],
                    min=0.0,
                    max=0.0,
                    median=0.0,
                    p75=float(p75_metrics.get(value[0][0])) * value[0][1],
                    p90=float(p90_metrics.get(value[0][0])) * value[0][1],
                    p99=float(p99_metrics.get(value[0][0])) * value[0][1],
                    n=n_count
                )

                # 添加到结果
                result.add_parameter(param_name, stage, metrics)
            
            result.metadata = dict(summary_metrics)
            
        # 计算摘要
        result.calculate_summary()
        
        return result
    
    @staticmethod
    def parse_directory(directory_path: str) -> Dict[str, Any]:
       pass


def find_evalscope_result_file(directory_path=".", init_entries=list(), evalscope_type="perf", dataset_name="mmmu"):
    """
    查找指定目录下最新生成的日期格式目录
    
    Args:
        directory_path (str): 要搜索的目录路径，默认为当前目录
    
    Returns:
        str or False: 最新日期目录名称，如果没有找到返回False
    """
    
    result_file = []

    # 日期目录的正则表达式模式 (格式: YYYYMMDD_HHMMSS)
    date_pattern = re.compile(r'^\d{8}_\d{6}$')
    
    # 获取目录下的所有条目
    try:
        entries = os.listdir(directory_path)
    except FileNotFoundError:
        logging.info(f"错误: 目录 '{directory_path}' 不存在")
        return False, ""
    except PermissionError:
        logging.info(f"错误: 没有权限访问目录 '{directory_path}'")
        return False, ""
    
    # 过滤出符合日期格式的目录
    date_dirs = []
    for entry in entries:
        entry_path = os.path.join(directory_path, entry)
        if os.path.isdir(entry_path) and date_pattern.match(entry) and entry not in init_entries:
            date_dirs.append(entry)
    
    # 如果没有找到日期目录
    if not date_dirs:
        logging.info("未找到日期格式的目录 (格式: YYYYMMDD_HHMMSS)")
        return False, ""
    
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
        if evalscope_type == "acc":
            # acc类型先待定
            target_dir = os.path.join(directory_path, data_dir)
            for root, dirs, files in os.walk(target_dir):
                if f"{dataset_name}.json" in files:
                    result_file.append(os.path.join(root, f"{dataset_name}.json"))
                    return result_file, target_dir
            
        elif evalscope_type == "perf":
            # 递归搜索文件夹下所有子目录和文件
            target_dir = os.path.join(directory_path, data_dir)
            has_percentile = False
            has_summary = False
            
            for root, dirs, files in os.walk(target_dir):
                if "benchmark_percentile.json" in files:
                    result_file.append(os.path.join(root, "benchmark_percentile.json"))
                    has_percentile = True
                if "benchmark_summary.json" in files:
                    result_file.append(os.path.join(root, "benchmark_summary.json"))
                    has_summary = True
                # 如果两个文件都找到了，可以提前结束搜索
                if has_percentile and has_summary:
                    break
            
            if has_percentile and has_summary:
                return result_file, target_dir
    
    logging.error("can not match evalscope result dir.")
    return [], os.path.join(directory_path, date_dirs[0])

def evalscope_acc_post_process(task_config, output_path, init_entries):
    evalscope_output_file, benchmark_outputdir = find_evalscope_result_file(output_path, init_entries, evalscope_type="acc", dataset_name=task_config.get("dataset_name").lower())
    if not evalscope_output_file:
        logging.error("evalscope acc的结果文件为空.")
        return 0, benchmark_outputdir
    
    if evalscope_output_file[0] and os.path.exists(evalscope_output_file[0]):
        logging.info(f"evalscope acc的结果文件: {evalscope_output_file[0]}")

        # 读取acc结果文件
        with open(evalscope_output_file[0], 'r') as f:
            data = json.load(f) 
            if data.get("score"):
                result_value = data["score"]
            else:
                logging.error("evalscope acc的结果文件中没有score字段.")
                result_value = 0

        # 将acc结果json文件拷贝一份到benchmark_outputdir目录
        try:
            if benchmark_outputdir:
                # dst_path = os.path.join(benchmark_outputdir, (task_config.get("case_name").lower() + ".json"))
                dst_path = os.path.join(output_path, (task_config.get("case_name").lower() + ".json"))
                shutil.copy2(evalscope_output_file[0], dst_path)
                logging.info(f"evalscope acc结果文件已拷贝到: {dst_path}")
        except Exception as e:
            logging.error(f"拷贝evalscope acc结果文件失败: {e}")

    else:
        logging.error(f"evalscope acc的结果文件: {evalscope_output_file[0]} is not exist.")
        result_value = 0
    return result_value, benchmark_outputdir

def evalscope_perf_post_process(task_config, output_path, init_entries):
    evalscope_output_file, benchmark_outputdir = find_evalscope_result_file(output_path, init_entries, evalscope_type="perf", dataset_name="")
    if not evalscope_output_file:
        logging.error("evalscope perf的结果文件为空.")
        return None, benchmark_outputdir

    logging.info(f"evalscope perf的结果文件: {evalscope_output_file[0]}, {evalscope_output_file[1]}")

    parser = EvalscopeResultParser()
    result = parser.parse_perf_json(evalscope_output_file)
     
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
        
    logging.info(f"  ✅  结果已保存:")
    logging.info(f"   Result JSON file: {json_path}")
    logging.info(f"   Result Summary file: {summary_path}")        

    
    return result, benchmark_outputdir
