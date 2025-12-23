import logging


def result_postprocess(benchmark_stdout, task_config, output_file_path):
    post_processed_result = {}
    benchmark_type = task_config.get("benchmark").get("benchmark_type")
    if benchmark_type == "aisbench":
        from xtask.benckmarks.aisbench.aisbench_postprocess import aisbench_post_process

        post_processed_result = aisbench_post_process(task_config, output_file_path)

    elif benchmark_type == "vllm_bench":
        from xtask.benckmarks.vllm_bench.vllm_bench_postprocess import vllm_ascend_post_process

        post_processed_result = vllm_ascend_post_process(benchmark_type)

    else:
        logging.error("benchmark_type: %s is invalid."% benchmark_type)
    return post_processed_result