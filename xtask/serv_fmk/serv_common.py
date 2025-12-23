import logging


def wait_serv_start(framework, task_config, serving_wait_time):
    ret = False
    if framework == "xllm":
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import wait_xllm_serv_start

        ret = wait_xllm_serv_start(task_config, serving_wait_time)

    elif framework == "vllm-ascend":
        from xtask.serv_fmk.vllm_ascend_serv.generate_vllm_ascend_serv \
            import wait_vllm_ascend_serv_start

        ret = wait_vllm_ascend_serv_start(task_config)

    elif framework == "mindie":
        from xtask.serv_fmk.mindie_serv.generate_mindie_serv \
            import wait_mindie_serv_start

        ret = wait_mindie_serv_start(task_config)

    else:
        logging.error("framework: %s is invalid."% framework)
    return ret
