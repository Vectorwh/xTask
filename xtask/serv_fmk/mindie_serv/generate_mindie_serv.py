def generate_mindie_serv_commands(task_config, framework_config, output_file_path=None,
                                  instance=None):
    """Generate MindIE startup commands.

    TODO: multi-host deployment.instances support is not implemented yet.
    """
    if instance is not None:
        raise NotImplementedError(
            "MindIE multi-host (deployment.instances) is not implemented yet"
        )
    return ""


def wait_mindie_serv_start(framework):
    pass
