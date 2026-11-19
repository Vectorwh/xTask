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
        
        # 创建摘要（metadata 键做新旧双兼容：1.14.0 改为 'Failed Requests'/'Req Throughput (req/s)'/
        # 'Total Throughput (tok/s)'/'Concurrency'，1.5.0 为 'Failed requests'/'Request throughput (req/s)'/
        # 'Total token throughput (tok/s)'/'Number of concurrency'）
        self.summary = ResultSummary(
            total_requests=int(e2el.n if hasattr(e2el, 'n') else 0),
            failed_requests=_get_first(self.metadata, [('Failed requests', None), ('Failed Requests', None)], default=0),
            # 类型语义沿用远端 bd97207（concurrency float / max_concurrency int）；
            # 键名用 _get_first 新旧双兼容（evalscope 1.14.0 'Concurrency' 与 1.5.0 'Number of concurrency'）。
            concurrency=float(_get_first(self.metadata, [('Number of concurrency', None), ('Concurrency', None)], default=1)),
            max_concurrency=int(_get_first(self.metadata, [('Number of concurrency', None), ('Concurrency', None)], default=1)),
            #request_throughput=float(e2el.n / (e2el.average / 1000)) if e2el.average > 0 else 0.0,
            request_throughput=_get_first(self.metadata, [('Request throughput (req/s)', None), ('Req Throughput (req/s)', None)], default=0),
            input_tokens_avg=input_tokens.average,
            output_tokens_avg=output_tokens.average,
            output_tps_avg=output_throughput.average,
            output_tps_total=output_throughput.average,
            summary_tps_total=_get_first(self.metadata, [('Total token throughput (tok/s)', None), ('Total Throughput (tok/s)', None)], default=1),
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
    
    def print_summary(self, detailed: bool = False) -> str:
        """打印结果摘要"""
        if not self.summary:
            self.calculate_summary()

        lines = []

        def emit(message):
            lines.append(message)
            logging.info(message)

        emit("=" * 80)
        emit("EvalScope TEST RESULT SUMMARY")
        emit("=" * 80)

        # 打印关键指标
        emit("  📊 KEY METRICS:")
        emit(f"{'Metric':<30} {'Value':<20} {'Unit':<10}")
        emit("-" * 80)
        emit(f"{'Total Requests':<30} {self.summary.total_requests:<20} {'count'}")
        emit(f"{'Failed Requests':<30} {self.summary.failed_requests:<20} {'count'}")
        emit(f"{'Concurrency':<30} {self.summary.concurrency:<20} {'count'}")
        emit(f"{'Max Concurrency':<30} {self.summary.max_concurrency:<20} {'count'}")
        emit(f"{'Request Throughput':<30} {self.summary.request_throughput:.2f} {'req/s'}")
        emit(f"{'Avg Input Tokens':<30} {self.summary.input_tokens_avg:.2f} {'tokens'}")
        emit(f"{'Avg Output Tokens':<30} {self.summary.output_tokens_avg:.2f} {'tokens'}")
        emit(f"{'Output Token Throughput':<30} {self.summary.output_tps_avg:.2f} {'tokens/s'}")
        emit(f"{'Output Token Throughput Total':<30} {self.summary.output_tps_total:.2f} {'tokens/s'}")
        emit(f"{'Summary Token Throughput Total':<30} {self.summary.summary_tps_total:.2f} {'tokens/s'}")
        emit(f"{'TTFT (Average)':<30} {self.summary.ttft_avg:.2f} {'ms'}")
        emit(f"{'TTFT (P90)':<30} {self.summary.ttft_p90:.2f} {'ms'}")
        emit(f"{'TPOT (Average)':<30} {self.summary.tpot_avg:.2f} {'ms'}")
        emit(f"{'TPOT (P90)':<30} {self.summary.tpot_p90:.2f} {'ms'}")
        emit(f"{'E2E Latency (Average)':<30} {self.summary.e2el_avg:.2f} {'ms'}")
        emit(f"{'E2E Latency (P90)':<30} {self.summary.e2el_p90:.2f} {'ms'}")

        if detailed and self.parameters:
            emit("  📈  DETAILED PERFORMANCE METRICS:")
            emit("-" * 60)
            for param_name, stages in self.parameters.items():
                emit(f"  {param_name}:")
                for stage_name, metrics in stages.items():
                    emit(f"  Stage: {stage_name}")
                    emit(f"    Average: {metrics.average:.2f} ms | Min: {metrics.min:.2f} ms | "
                         f"Max: {metrics.max:.2f} ms | N: {metrics.n}")
                    emit(f"    Median: {metrics.median:.2f} ms | P75: {metrics.p75:.2f} ms | "
                         f"P90: {metrics.p90:.2f} ms | P99: {metrics.p99:.2f} ms")

        emit("=" * 80)
        return "\n".join(lines) + "\n"


def _get_first(data, candidates, default=0.0):
    """按候选键顺序取第一个存在且非 None 的值，用于新旧版本键名双兼容。

    Args:
        data: 字典（如 benchmark_summary.json 内容）
        candidates: 候选 (键名, 默认乘数) 列表，乘数在此函数中不使用
        default: 所有候选键都不存在时的返回值

    Returns:
        命中键的原始值（未做乘法），或 default
    """
    for field, _mult in candidates:
        if field in data and data[field] is not None:
            return data[field]
    return default


def _lookup(data, candidates, default=0.0):
    """按候选 (键, 乘数) 顺序查找并应用单位换算，返回 float。

    用于新旧版本双兼容：不同版本键名不同、单位不同（旧版秒 → ×1000 转 ms，
    新版原生 ms → ×1），乘数跟随实际命中的键。
    注意：1.14.0 的 percentile 文件在无采样数据时可能写 NaN（如流式单 chunk
    请求的 ITL），此处将 NaN 归一为 0，避免下游 JSON 序列化出现非法值。
    """
    import math
    for field, multiplier in candidates:
        if field in data and data[field] is not None:
            value = float(data[field]) * multiplier
            return 0.0 if math.isnan(value) else value
    return float(default)


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

                # 按 'Percentiles' 字段的值匹配取行，兼容新旧两个版本：
                # - evalscope 1.5.0:   [10%,25%,50%,66%,75%,80%,90%,95%,98%,99%]
                # - evalscope 1.14.0:  [min,1%,5%,10%,25%,50%,75%,90%,95%,99%,max]
                # 旧代码按下标取 data_per[4]/[6]/[-1]，在 1.14.0 下会错位，故改为按值匹配
                def _match_percentile_row(rows, label):
                    for row in rows:
                        if isinstance(row, dict) and row.get('Percentiles') == label:
                            return row
                    return {}

                p75_metrics = _match_percentile_row(data_per, '75%')
                p90_metrics = _match_percentile_row(data_per, '90%')
                p99_metrics = _match_percentile_row(data_per, '99%')
    
            if "benchmark_summary.json" in file:
                with open(file, 'r', encoding='utf-8') as f:
                    summary_metrics = json.load(f)

                #summary_metrics = data_sum

        # 构建metrics data
        if p75_metrics and p90_metrics and p99_metrics and summary_metrics:
            TIME_MS = 1000
            # n_bs = float(summary_metrics.get('Number of concurrency', 1))
            # div_bs = float(1 / n_bs) if n_bs > 0.0 else 0.0
            n_count = _get_first(summary_metrics, [('Total Requests', None), ('Total requests', None)], default=1)
            # rawdata_fields 新旧版本双兼容（evalscope 1.5.0 → 1.14.0）：
            # - summary 键名变化: 'Average time to first token (s)' → 'Avg TTFT (ms)' 等，
            #   且新版 TTFT/TPOT/ITL 单位原生为 ms（旧版为 s）——乘数跟随实际命中的键
            # - percentile 键名变化: 'TTFT (s)' → 'TTFT (ms)' 等（Latency (s)/Input tokens/
            #   Output tokens/Output (tok/s) 不变）
            # 每项为 {'summary': [(键, 乘数)...], 'percentile': [(键, 乘数)...]}，
            # 候选键按优先级排列，取第一个存在的键，兼容历史 outputs 重放解析。
            rawdata_fields = {
                'E2EL': {
                    'summary': [('Avg Latency (s)', TIME_MS), ('Average latency (s)', TIME_MS)],
                    'percentile': [('Latency (s)', TIME_MS)],
                },
                'TTFT': {
                    'summary': [('Avg TTFT (ms)', 1), ('Average time to first token (s)', TIME_MS)],
                    'percentile': [('TTFT (ms)', 1), ('TTFT (s)', TIME_MS)],
                },
                'TPOT': {
                    'summary': [('Avg TPOT (ms)', 1), ('Average time per output token (s)', TIME_MS)],
                    'percentile': [('TPOT (ms)', 1), ('TPOT (s)', TIME_MS)],
                },
                'ITL': {
                    'summary': [('Avg ITL (ms)', 1), ('Average inter-token latency (s)', TIME_MS)],
                    'percentile': [('ITL (ms)', 1), ('ITL (s)', TIME_MS)],
                },
                'InputTokens': {
                    'summary': [('Avg Input Tokens', 1), ('Average input tokens per request', 1)],
                    'percentile': [('Input tokens', 1)],
                },
                'OutputTokens': {
                    'summary': [('Avg Output Tokens', 1), ('Average output tokens per request', 1)],
                    'percentile': [('Output tokens', 1)],
                },
                'OutputTokenThroughput': {
                    'summary': [('Output Throughput (tok/s)', 1), ('Output token throughput (tok/s)', 1)],
                    'percentile': [('Output (tok/s)', 1)],
                },
            }

            for key, value in rawdata_fields.items():
                param_name = key.strip()
                stage = "total"

                # 创建指标对象
                metrics = PerformanceMetrics(
                    average=_lookup(summary_metrics, value['summary']),
                    min=0.0,
                    max=0.0,
                    median=0.0,
                    p75=_lookup(p75_metrics, value['percentile']),
                    p90=_lookup(p90_metrics, value['percentile']),
                    p99=_lookup(p99_metrics, value['percentile']),
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

def _extract_report_score(data):
    """从 evalscope 报告 JSON 中提取精度分数，兼容新旧两个 schema。

    - 旧版 (<=1.5.0): 顶层 {'score': <float>, ...}
    - 新版 (1.14.0, schema_version=2): 顶层无 'score'，分数在 'metrics' 列表中，
      每个 metric 形如 {'identity': {'name': ..., 'aggregation': ..., 'dimensions': {...}},
                        'score': <float>, 'macro_score': <float>, 'categories': [...], ...}，
      并有 'primary_metric_identity' 指向主 metric。

    取值逻辑（对齐 1.14.0 Report._find_primary_metric / Report.score 的语义）：
      0) 顶层 'score' 存在则直接返回（旧版兼容）
      1) primary_metric_identity 存在时，取 metrics 中 identity.name 与之匹配的 metric 的 score
      2) 否则若 metrics 只有一个，取该 metric 的 score
      3) 多 metric 无 primary 时，取第一个 metric 的 score 并告警（假设主 metric 排在前；
         xTask 用例（mmmu 等）通常单 metric，此分支仅为兜底）

    Returns:
        float 分数；无法提取时返回 None
    """
    if not isinstance(data, dict):
        return None

    # 旧版 (<=1.5.0) 顶层 'score'
    if data.get('score') is not None:
        return data['score']

    metrics = data.get('metrics')
    if not isinstance(metrics, list) or not metrics:
        return None

    primary_identity = data.get('primary_metric_identity')
    if isinstance(primary_identity, dict) and primary_identity.get('name'):
        primary_name = primary_identity['name']
        for metric in metrics:
            identity = metric.get('identity') if isinstance(metric, dict) else None
            if isinstance(identity, dict) and identity.get('name') == primary_name:
                return metric.get('score')
        # primary_identity 与 metrics 不匹配（理论上 1.14.0 校验会拦截，此处兜底）
        logging.warning("evalscope 报告 primary_metric_identity=%s 未在 metrics 中匹配到", primary_name)

    if len(metrics) == 1:
        return metrics[0].get('score') if isinstance(metrics[0], dict) else None

    # 多 metric 且无 primary：取第一个 metric 的 score 作为兜底
    logging.warning("evalscope 报告含多个 metric 且无 primary_metric_identity，取第一个 metric 的 score")
    first = metrics[0]
    return first.get('score') if isinstance(first, dict) else None


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
            if data.get("score") is not None:
                # 旧版 evalscope (<=1.5.0) 报告顶层直接带 'score' 字段
                result_value = data["score"]
            else:
                # 新版 evalscope 1.14.0 报告 schema v2（report/report.py class Report）：
                # 顶层无 'score' 键，分数在 'metrics' 列表中，每个 metric 含
                # identity/num/score/macro_score/categories 字段；'primary_metric_identity'
                # 指向该报告的主 metric（Report.score 属性即取主 metric 的 score，但不序列化）。
                # 取值逻辑（对齐 1.14.0 Report._find_primary_metric 语义）：
                #   1) 有 primary_metric_identity 时，取 metrics 中 identity.name 匹配的 metric 的 score
                #   2) 无 primary_metric_identity 时：仅一个 metric 则取之（mmmu 等数据集的常见情况）；
                #      多 metric 时取第一个非 diagnostic metric 的 score，并打印告警
                result_value = _extract_report_score(data)
                if result_value is None:
                    logging.error("evalscope acc的结果文件中没有可用的score字段.")
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

    summary_text = result.print_summary(detailed=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        
    logging.info(f"  ✅  结果已保存:")
    logging.info(f"   Result JSON file: {json_path}")
    logging.info(f"   Result Summary file: {summary_path}")        

    
    return result, benchmark_outputdir
