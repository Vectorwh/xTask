import logging
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    average: float
    min: float
    max: float
    median: float
    p75: float
    p90: float
    p99: float
    n: int


@dataclass
class ResultSummary:
    """结果摘要数据类"""
    total_requests: int
    concurrency: int
    max_concurrency: int
    request_throughput: float
    input_tokens_avg: float
    output_tokens_avg: float
    output_token_throughput: float
    ttft_avg: float
    ttft_p90: float
    tpot_avg: float
    tpot_p90: float
    e2el_avg: float
    e2el_p90: float


def result_postprocess(benchmark_stdout, task_config, output_file_path, init_entries):
    post_processed_result = {}
    benchmark_type = task_config.get("benchmark").get("benchmark_type")
    if benchmark_type == "aisbench":
        from xtask.benckmarks.aisbench.aisbench_postprocess import aisbench_post_process

        post_processed_result = aisbench_post_process(task_config, output_file_path, init_entries)

    elif benchmark_type == "vllm_bench":
        from xtask.benckmarks.vllm_bench.vllm_bench_postprocess import vllm_ascend_post_process

        post_processed_result = vllm_ascend_post_process(benchmark_type)

    else:
        logging.error("benchmark_type: %s is invalid."% benchmark_type)
    return post_processed_result


def acc_result_postprocess(task_config, output_file_path, init_entries):
    post_processed_result = ""
    benchmark_type = task_config.get("benchmark").get("benchmark_type")
    if benchmark_type == "aisbench":
        from xtask.benckmarks.aisbench.aisbench_postprocess import aisbench_acc_post_process

        post_processed_result = aisbench_acc_post_process(task_config, output_file_path, init_entries)

    elif benchmark_type == "vllm_bench":
        from xtask.benckmarks.vllm_bench.vllm_bench_postprocess import vllm_ascend_post_process

        post_processed_result = vllm_ascend_post_process(benchmark_type)

    else:
        logging.error("benchmark_type: %s is invalid."% benchmark_type)

    # 和base line 数据做对比，数据误差允许阈值是 threshold
    if task_config.get(""):
        pass
    
    logging.info("acc dataset name is: %s, acc result is: %s"% (task_config.get("dataset_name"), post_processed_result))
    return post_processed_result
