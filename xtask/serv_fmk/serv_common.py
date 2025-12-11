import logging


def wait_serv_start(framework):
    ret = False
    if framework == "xllm":
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import wait_xllm_serv_start

        ret = wait_xllm_serv_start(framework)

    elif framework == "vllm-ascend":
        from xtask.serv_fmk.vllm_ascend_serv.generate_vllm_ascend_serv \
            import wait_vllm_ascend_serv_start

        ret = wait_vllm_ascend_serv_start(framework)

    elif framework == "mindie":
        from xtask.serv_fmk.mindie_serv.generate_mindie_serv \
            import wait_mindie_serv_start

        ret = wait_mindie_serv_start(framework)

    else:
        logging.error("framework: %s is invalid."% framework)
    return ret
