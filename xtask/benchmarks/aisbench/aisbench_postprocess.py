import logging
import pandas as pd
import json
import csv
import re
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

        json_data = self.metadata.get('json_data') or {}

        def get_json_metric(name, default=0):
            value = json_data.get(name, default)
            if isinstance(value, dict):
                value = value.get('total', default)
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                match = re.search(r'[-]?\d*\.?\d+', value)
                if match:
                    return float(match.group())
            return default
        
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
            failed_requests=int(get_json_metric('Failed Requests')),
            concurrency=float(get_json_metric('Concurrency', 1)),
            max_concurrency=int(get_json_metric('Max Concurrency', 1)),
            request_throughput=float(get_json_metric('Request Throughput')),
            input_tokens_avg=input_tokens.average,
            output_tokens_avg=output_tokens.average,
            output_tps_avg=output_throughput.average,
            output_tps_total=float(get_json_metric('Output Token Throughput')),
            summary_tps_total=float(get_json_metric('Total Token Throughput')),
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
        emit("AISBENCH TEST RESULT SUMMARY")
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


class AisbenchResultParser:
    """Aisbench结果解析器"""
    
    @staticmethod
    def parse_csv_and_json(csv_filepath: str, json_filepath: str) -> AisbenchResult:
        """
        从CSV文件解析结果
        
        Args:
            csv_filepath: CSV文件路径
            json_filepath: JSON文件路径
            
        Returns:
            AisbenchResult对象
        """
        # 读取CSV文件
        df = pd.read_csv(csv_filepath)
        # 读取JSON文件
        json_data = None
        if json_filepath:
            with open(json_filepath, 'r') as f:
                json_data = json.load(f)
        
        # 创建结果对象
        result = AisbenchResult()
        result.raw_data = df
        result.metadata = {
            'source_file': csv_filepath,
            'parse_time': datetime.now().isoformat(),
            'total_rows': len(df),
            'json_data': json_data
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
                    result = AisbenchResultParser.parse_csv_and_json(filepath, "")
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
        aisbench_csv, aisbench_json = get_aisbench_csv_and_json(os.path.join(output_path, aisbench_output_dir), 
                                        task_config.get("benchmark").get("perf").get("api_type").replace("_", "-"))
    else:
        logging.error("get aisbench output path failed.")
        return None, ""
    
    # 解析结果
    if aisbench_csv and aisbench_json and os.path.exists(aisbench_csv) and os.path.exists(aisbench_json):
        parser = AisbenchResultParser()
        result = parser.parse_csv_and_json(aisbench_csv, aisbench_json)
        
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
    
    return result, (os.path.join(output_path, aisbench_output_dir))    

def convert_aisbench_csv_to_json(csv_rows, save_dir, case_name):
    """
    将aisbench的csv结果(已读取的行数据)转换为json并保存到指定目录下。

    Args:
        csv_rows (list): 已从csv读取的行数据(每行为dict)。
        save_dir (str): json文件保存目录。
        case_name (str): 用例名称，作为json文件名。
    """
    try:
        json_path = os.path.join(save_dir, (case_name + ".json"))
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(csv_rows, f, ensure_ascii=False, indent=4)
        logging.info(f"aisbench acc csv已转换为json并保存: {json_path}")
    except Exception as e:
        logging.error(f"aisbench acc csv转换为json失败: {e}")


def aisbench_acc_post_process(task_config, output_path, init_entries):
    aisbench_output_dir = find_aisbench_directory(output_path, init_entries, aisbench_type="acc")
    logging.info(f"aisbench acc的结果目录: {aisbench_output_dir}")
    api_type= task_config.get("benchmark").get("acc").get("api_type").replace("_", "-")
    if aisbench_output_dir and os.path.exists(os.path.join(output_path, aisbench_output_dir)):
        aisbench_csv, aisbench_json = get_aisbench_csv_and_json(os.path.join(output_path, aisbench_output_dir), api_type, output_type="accaurcy")
    else:
        logging.error("get aisbench output path failed.")
        return
    
    if os.path.exists(aisbench_csv):
        # 只打开一次csv文件，读取所有行数据，供转换json和取值复用
        try:
            with open(aisbench_csv, mode='r', encoding='utf-8') as f:
                csv_rows = [dict(row) for row in csv.DictReader(f)]
        except Exception as e:
            logging.error(f"读取aisbench csv失败: {e}")
            return

        # 将aisbench的csv结果转换为json并保存到结果目录下
        # save_dir = os.path.join(output_path, aisbench_output_dir)
        convert_aisbench_csv_to_json(csv_rows, output_path, task_config.get("case_name").lower())

        # 从已读取的行数据中提取目标列的值
        result_value = get_column_values(csv_rows, api_type)
        if result_value and len(result_value) == 1:
            return result_value[0], (os.path.join(output_path, aisbench_output_dir))
        else:
            logging.error("")
            return
    else:
        logging.error(f"aisbench csv: {aisbench_csv} is not exist.")
        result = None
    return result, (os.path.join(output_path, aisbench_output_dir))


def _extract_protocol_scores(data):
    """Recursively search a JSON-like structure for protocol evaluation scores.

    ais_bench eval mode may nest the evaluator output under
    ``{dataset_abbr: {model_abbr: scores}}`` or write it flat. This helper
    finds the first dict containing ``Protocol_Pass_Rate``.
    """
    if not isinstance(data, dict):
        return None
    if "Protocol_Pass_Rate" in data:
        return data
    for value in data.values():
        found = _extract_protocol_scores(value)
        if found is not None:
            return found
    return None


def protocol_result_post_process(task_config, output_path, init_entries):
    """Post-process protocol test results from ais_bench eval output.

    ais_bench ``--mode eval`` writes results under
    ``<output_path>/protocol_results/<YYYYMMDD_HHMMSS>/``. This function
    locates the newest date directory, searches for JSON files containing
    protocol evaluation scores (Protocol_Pass_Rate, Syntax_Valid_Rate,
    Semantic_Accuracy), and returns the parsed score dict.

    Returns:
        (result_dict, output_dir) on success; (None, "") on failure.
    """
    protocol_results_root = os.path.join(output_path, "protocol_results")
    if not os.path.isdir(protocol_results_root):
        logging.error("protocol_results directory not found: %s", protocol_results_root)
        return None, ""

    date_pattern = re.compile(r'^\d{8}_\d{6}$')
    date_dirs = []
    try:
        for entry in os.listdir(protocol_results_root):
            entry_path = os.path.join(protocol_results_root, entry)
            if os.path.isdir(entry_path) and date_pattern.match(entry):
                date_dirs.append(entry_path)
    except OSError:
        logging.error("Failed to list protocol_results directory: %s", protocol_results_root)
        return None, ""

    if not date_dirs:
        logging.error("No date directory found under %s", protocol_results_root)
        return None, ""

    date_dirs.sort(reverse=True)
    result_dir = date_dirs[0]
    logging.info("protocol test result directory: %s", result_dir)

    protocol_result = None
    for root, _dirs, files in os.walk(result_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            found = _extract_protocol_scores(data)
            if found is not None:
                protocol_result = found
                logging.info("Protocol scores loaded from %s", fpath)
                break
        if protocol_result is not None:
            break

    if protocol_result is None:
        logging.error("No protocol evaluation scores (Protocol_Pass_Rate) found under %s", result_dir)
        return None, ""

    logging.info("Protocol test results: %s", json.dumps(protocol_result, ensure_ascii=False))
    return protocol_result, result_dir


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
            # 尝试根据指定的格式解析目录名中的日期时间
            return datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
        except ValueError:
            # 如果目录名格式不匹配导致解析失败，则返回一个很旧的日期
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

    return date_dirs[0]


def _get_paired_performance_files(result_dir):
    if not os.path.isdir(result_dir):
        return "", ""
    entries = os.listdir(result_dir)
    csv_files = {
        os.path.splitext(name)[0]: os.path.join(result_dir, name)
        for name in entries
        if name.endswith(".csv")
    }
    json_files = {
        os.path.splitext(name)[0]: os.path.join(result_dir, name)
        for name in entries
        if name.endswith(".json") and not name.endswith("_details.json")
    }
    matching_stems = sorted(set(csv_files) & set(json_files))
    if len(matching_stems) == 1:
        stem = matching_stems[0]
        return csv_files[stem], json_files[stem]
    if len(matching_stems) > 1:
        logging.error(
            f"Multiple AISBench performance result pairs found in {result_dir}: "
            f"{matching_stems}"
        )
    return "", ""


def get_aisbench_csv_and_json(
    aisbench_output_path, api_type, output_type="performances"
):
    if output_type == "performances":
        performances_dir = os.path.join(aisbench_output_path, output_type)
        result_dir = os.path.join(performances_dir, api_type.replace("_", "-"))
        csv_path, json_path = _get_paired_performance_files(result_dir)
        if (not csv_path or not json_path) and os.path.isdir(performances_dir):
            candidates = []
            for entry in sorted(os.listdir(performances_dir)):
                candidate = os.path.join(performances_dir, entry)
                if not os.path.isdir(candidate):
                    continue
                csv_path, json_path = _get_paired_performance_files(candidate)
                if csv_path and json_path:
                    candidates.append(candidate)
            if len(candidates) == 1:
                result_dir = candidates[0]
            elif len(candidates) > 1:
                logging.error(
                    "Can't uniquely identify AISBench performance output for "
                    f"api_type={api_type}: {candidates}"
                )
                return "", ""
        csv_path, json_path = _get_paired_performance_files(result_dir)
        if not csv_path or not json_path:
            logging.error(
                f"Can't find aisbench csv or json output. {output_type} run failed."
            )
            return "", ""
        return csv_path, json_path
    elif output_type == "accaurcy":
        result_dir = os.path.join(aisbench_output_path, "summary")
    else:
        logging.error(f"output_type: {output_type} is not supported.")
        return
    
    csv_path = ""
    json_path = ""

    if os.path.isdir(result_dir):
        entries = os.listdir(result_dir)
        for entry in entries:
            entry_path = os.path.join(result_dir, entry)
            if entry_path.endswith(".csv"):
                csv_path = entry_path
            if entry_path.endswith(".json"):
                json_path = entry_path
    
    if output_type == "accaurcy" and not csv_path:
        logging.error(f"Can't find aisbench csv output. {output_type} run failed.")
        return "", ""
    
    return csv_path, json_path


def get_column_values(csv_rows, target_column):
    """
    从已读取的csv行数据中提取指定列的值。

    Args:
        csv_rows (list): 已从csv读取的行数据(每行为dict)。
        target_column (str): 要读取的列名。

    Returns:
        list: 包含目标列所有值的列表；若某行缺少该列则返回空列表。
    """
    values = []
    for row in csv_rows:
        # 通过列名（键）获取对应单元格的值
        if target_column in row:
            values.append(row[target_column])
        else:
            logging.error(f"警告：文件中未找到列名 '{target_column}'")
            return []
    return values


def read_csv_value(file_path, target_column):
    """
    读取 CSV 文件并返回特定列的值。
    
    Args:
        file_path (str): CSV 文件路径。
        target_column (str): 要读取的列名。
    
    Returns:
        list: 包含目标列所有值的列表。
    """
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_rows = [dict(row) for row in csv.DictReader(file)]
    return get_column_values(csv_rows, target_column)
