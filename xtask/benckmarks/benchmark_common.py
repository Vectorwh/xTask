import logging


def result_postprocess(benchmark_stdout, benchmark_type):
    if benchmark_type == "aisbench":
        from xtask.benckmarks.aisbench.aisbench_postprocess import aisbench_post_process

        ret = aisbench_post_process(benchmark_stdout)

    elif benchmark_type == "vllm_bench":
        from xtask.benckmarks.vllm_bench.vllm_bench_postprocess import vllm_ascend_post_process

        ret = vllm_ascend_post_process(benchmark_type)
        
    else:
        logging.error("benchmark_type: %s is invalid."% benchmark_type)
    return ret