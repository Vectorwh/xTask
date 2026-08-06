def generate_vllm_ascend_serv_commands(task_config, framework_config, output_file_path=None,
                                       instance=None):
    """Generate vLLM Ascend startup commands.

    TODO: multi-host deployment.instances support is not implemented yet.
    Keep single-host path as-is; when instance is provided, refuse clearly.
    """
    if instance is not None:
        raise NotImplementedError(
            "vLLM Ascend multi-host (deployment.instances) is not implemented yet"
        )
    # Existing single-host path remains a stub in this repository.
    return ""


def wait_vllm_ascend_serv_start(framework):
    pass
