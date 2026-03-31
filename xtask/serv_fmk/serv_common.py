import logging


def wait_serv_start(engine, task_config, serving_wait_time, output_file_path: str):
    ret = False
    if engine == "xllm":
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import wait_xllm_serv_start

        ret = wait_xllm_serv_start(task_config, serving_wait_time, output_file_path)

    elif engine == "vllm-ascend":
        from xtask.serv_fmk.vllm_ascend_serv.generate_vllm_ascend_serv \
            import wait_vllm_ascend_serv_start

        ret = wait_vllm_ascend_serv_start(task_config)

    elif engine == "mindie":
        from xtask.serv_fmk.mindie_serv.generate_mindie_serv \
            import wait_mindie_serv_start

        ret = wait_mindie_serv_start(task_config)

    else:
        logging.error("engine: %s is invalid."% engine)
    return ret
