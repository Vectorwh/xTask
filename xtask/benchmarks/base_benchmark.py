import os
import logging
import shutil
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
    failed_requests: int
    concurrency: float
    max_concurrency: int
    request_throughput: float
    input_tokens_avg: float
    output_tokens_avg: float
    output_tps_avg: float
    output_tps_total: float
    summary_tps_total: float
    ttft_avg: float
    ttft_p90: float
    ttft_p99: float
    tpot_avg: float
    tpot_p90: float
    tpot_p99: float
    e2el_avg: float
    e2el_p90: float


def result_postprocess(benchmark_stdout, task_config, output_file_path, init_entries):
    post_processed_result = {}
    benchmark_type = task_config.get("benchmark").get("benchmark_type")
    if benchmark_type == "aisbench":
        from xtask.benchmarks.aisbench.aisbench_postprocess import aisbench_post_process

        post_processed_result, benchmark_outputdir = aisbench_post_process(task_config, output_file_path, init_entries)

    elif benchmark_type == "vllm_bench":
        from xtask.benchmarks.vllm_bench.vllm_bench_postprocess import vllm_ascend_post_process

        post_processed_result = vllm_ascend_post_process(benchmark_type)
    elif benchmark_type == "evalscope":
        from xtask.benchmarks.evalscope.evalscope_postprocess import evalscope_perf_post_process

        post_processed_result, benchmark_outputdir  = evalscope_perf_post_process(task_config, output_file_path, init_entries)
    elif benchmark_type == "image_edit":
        # image_edit perf shares the EvalScope output layout; prefer the dedicated
        # postprocess and fall back to the EvalScope one when the parallel
        # workstream module is not present yet.
        try:
            from xtask.benchmarks.evalscope.image_edit_postprocess import image_edit_perf_post_process

            post_processed_result, benchmark_outputdir = image_edit_perf_post_process(task_config, output_file_path, init_entries)
        except ImportError:
            from xtask.benchmarks.evalscope.evalscope_postprocess import evalscope_perf_post_process

            post_processed_result, benchmark_outputdir = evalscope_perf_post_process(task_config, output_file_path, init_entries)
    elif benchmark_type == "vbench":
        from xtask.benchmarks.vbench.vbench_postprocess import vbench_result_post_process

        post_processed_result, benchmark_outputdir = vbench_result_post_process(
            task_config, output_file_path, init_entries
        )
    else:
        logging.error("benchmark_type: %s is invalid."% benchmark_type)

    return post_processed_result, benchmark_outputdir

def acc_result_postprocess(task_config, output_file_path, init_entries):
    post_processed_result = ""
    benchmark_type = task_config.get("benchmark").get("benchmark_type")
    if benchmark_type == "aisbench":
        from xtask.benchmarks.aisbench.aisbench_postprocess import aisbench_acc_post_process

        post_processed_result, benchmark_outputdir = aisbench_acc_post_process(task_config, output_file_path, init_entries)

    elif benchmark_type == "vllm_bench":
        from xtask.benchmarks.vllm_bench.vllm_bench_postprocess import vllm_ascend_post_process

        post_processed_result = vllm_ascend_post_process(benchmark_type)
    elif benchmark_type == "evalscope":
        from xtask.benchmarks.evalscope.evalscope_postprocess import evalscope_acc_post_process

        post_processed_result, benchmark_outputdir = evalscope_acc_post_process(task_config, output_file_path, init_entries)
    elif benchmark_type == "vbench":
        from xtask.benchmarks.vbench.vbench_postprocess import vbench_result_post_process

        post_processed_result, benchmark_outputdir = vbench_result_post_process(
            task_config, output_file_path, init_entries
        )

    else:
        logging.error("benchmark_type: %s is invalid."% benchmark_type)

    # 和base line 数据做对比，数据误差允许阈值是 threshold
    if task_config.get(""):
        pass
    
    logging.info("acc dataset name is: %s, acc result is: %s"% (task_config.get("dataset_name"), post_processed_result))
    return post_processed_result, benchmark_outputdir


def get_acc_config(benchmark_type):
    if benchmark_type == "aisbench":
        aisbench_acc_config_file = os.path.join(os.getcwd(), "xtask/benchmarks/aisbench/aisbench_acc_config.json")
    elif benchmark_type == "vllm_bench":
        pass
    else:
        logging.error("benchmark_type: %s is invalid."% benchmark_type)
