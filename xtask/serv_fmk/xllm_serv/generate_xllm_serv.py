#!/usr/bin/env python3
import os
import yaml
import socket
import logging
import time
from xtask.utils import get_host_ip as get_local_ip
import re


def load_yaml_config(yaml_file):
    """加载 YAML 配置文件"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def _resolve_device_ids(config):
    """Resolve the physical device list used by an xLLM instance.

    ``device_ids`` is authoritative when supplied by deployment YAML.  The
    contiguous ``start_device_id``/``num_devices`` form remains supported for
    legacy task configurations.
    """
    configured_ids = config.get("device_ids")
    if configured_ids:
        device_ids = [int(device_id) for device_id in configured_ids]
    else:
        start_device = int(config.get("start_device_id", 0) or 0)
        device_count = int(config.get("num_devices", 1) or 1)
        device_ids = list(range(start_device, start_device + device_count))

    if not device_ids:
        raise ValueError("xLLM requires at least one device id")
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError(f"xLLM device ids must be non-negative: {device_ids}")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"xLLM device ids must be unique: {device_ids}")
    return device_ids


def _build_script_params(config):
    """Build template replacement params for one host/instance."""
    if config.get('host_addr') and config['host_addr'] != '0.0.0.0':
        local_ip = config['host_addr']
    else:
        local_ip = get_local_ip()

    device_ids = _resolve_device_ids(config)
    num_devices = len(device_ids)
    world_size = int(config.get('world_size', num_devices) or num_devices)
    # Single-host keeps historical --nnodes == num_devices behavior.
    # Multi-host sets global_nnodes / rank_start / master_host explicitly.
    local_nnodes = int(config.get('local_nnodes', num_devices) or num_devices)
    global_nnodes = int(config.get('global_nnodes', num_devices) or num_devices)
    master_host = config.get('master_host') or local_ip
    rank_start = int(config.get('rank_start', 0) or 0)
    pd_role = str(config.get("pd_role", config.get("role", ""))).upper()
    pd_enabled = bool(config.get("enable_disagg_pd", False))
    pd_master_host = config.get("pd_master_host") or master_host
    pd_etcd_port = int(config.get("pd_etcd_port", 3389) or 3389)

    return {
        'LOCAL_IP': local_ip,
        'MASTER_HOST': master_host,
        'HCCL_IF_PORT': config.get('hccl_if_port', 43439),
        'PROGRESS_CONN_PORT': config.get('progress_conn_port', 9590),
        'MODEL_PATH': config.get('model_path', '/models/Qwen3-4B'),
        'START_PORT': config.get('port', 14830),
        # Kept for compatibility with older generated scripts and diagnostics;
        # device selection itself is driven by VISIBLE_DEVICES below.
        'START_DEVICE': min(device_ids),
        'VISIBLE_DEVICES': ','.join(str(device_id) for device_id in device_ids),
        'NNODES': num_devices,
        'LOCAL_NNODES': local_nnodes,
        'GLOBAL_NNODES': global_nnodes,
        'RANK_START': rank_start,
        'WORLD_SIZE': world_size,
        'PD_ENABLED': "true" if pd_enabled else "false",
        'PD_ROLE': pd_role,
        'PD_MASTER_HOST': pd_master_host,
        'PD_ETCD_ADDR': f"{pd_master_host}:{pd_etcd_port}",
        'PD_NO_PROXY': config.get("pd_no_proxy", "localhost,127.0.0.1,0.0.0.0"),
        'LOG_DIR': 'log',
        'COMMUNICATION_BACKEND': config.get('communication_backend', 'hccl'),
        'DP_SIZE': config.get("dp_size", ""),
        'EP_SIZE': config.get("tp_size", ""),
        "USERID": config.get("userid", 0),
    }


def _select_template_path(local_nnodes, allow_large_world: bool = False, world_size: int = 1):
    if local_nnodes == 1 and not allow_large_world:
        return "xllm_single_card_template.sh"
    if local_nnodes >= 1 and (allow_large_world or world_size < 17):
        # Multi-card / multi-host uses the multi-process template.
        if local_nnodes == 1 and allow_large_world:
            return "xllm_mutil_cards_template.sh"
        if local_nnodes > 1:
            return "xllm_mutil_cards_template.sh"
        return "xllm_single_card_template.sh"
    return ""


def generate_xllm_script(config, output_file_path: str, output_file="xllm_serv_single_card.sh",
                         allow_large_world: bool = False):
    """生成 xLLM 启动脚本（单机或多机实例）。"""
    from xtask.utils.container_manager import path_mapping_in_container

    params = _build_script_params(config)
    local_nnodes = int(params["LOCAL_NNODES"])
    world_size = int(params["WORLD_SIZE"])

    template_name = _select_template_path(
        local_nnodes=local_nnodes,
        allow_large_world=allow_large_world,
        world_size=world_size,
    )
    if not template_name:
        logging.error(
            "WORLD_SIZE/LOCAL_NNODES unsupported: world_size=%s local_nnodes=%s",
            world_size,
            local_nnodes,
        )
        return "", ""

    template_path = os.path.join("xtask/serv_fmk/xllm_serv/", template_name)
    logging.info("template_path is: %s", template_path)
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    script_content = template
    for key, value in params.items():
        if key == "MODEL_PATH":
            value = path_mapping_in_container(value)
        script_content = script_content.replace(f'{{{key}}}', str(value))

    # 在模板中插入扩展变量（含 rank_tablefile 等透传参数）
    extense_str = ""
    extense_para = config.get("serving_extensiable_param", {}) or {}
    pd_enabled = bool(config.get("enable_disagg_pd", False))
    pd_role = str(config.get("pd_role", config.get("role", ""))).upper()
    pd_master_host = config.get("pd_master_host") or config.get("master_host")
    pd_etcd_port = int(config.get("pd_etcd_port", 3389) or 3389)
    transfer_port_base = int(config.get("transfer_port_base", 26000) or 26000)
    match_str = re.search(r'\n( *)\$SERVING_EXTENSIABLE_PARAM\n', script_content)
    space_str = match_str.group(1) if match_str else ""
    for key, value in extense_para.items():
        extense_str += f"{space_str}--{key}={value} \\\n"
    if pd_enabled:
        pd_parameters = [
            "--enable_disagg_pd=true",
            f"--instance_role={pd_role}",
            f"--etcd_addr={pd_master_host}:{pd_etcd_port}",
            (
                f"--transfer_listen_port=$(({transfer_port_base}+i))"
                if template_name == "xllm_mutil_cards_template.sh"
                else f"--transfer_listen_port={transfer_port_base}"
            ),
            "--disagg_pd_port=7777",
        ]
        for parameter in pd_parameters:
            extense_str += f"{space_str}{parameter} \\\n"
    script_content = script_content.replace(
        f"{space_str}$SERVING_EXTENSIABLE_PARAM\n", extense_str
    )

    os.makedirs(output_file_path, exist_ok=True)
    with open(os.path.join(output_file_path, output_file), 'w', encoding='utf-8') as f:
        f.write(script_content)
    os.chmod(os.path.join(output_file_path, output_file), 0o755)
    logging.debug("generate xllm script successfully: %s", output_file)

    return output_file, path_mapping_in_container(output_file_path)


def generate_xllm_serv_commands(task_config, framework_config, output_file_path: str,
                                instance=None):
    """Generate startup command for single-host or one multi-host instance.

    Args:
        task_config: task YAML dict (already instance-merged when instance is used)
        framework_config: engine container config
        output_file_path: host-side output directory for this script
        instance: optional deployment instance dict; when set, enables multi-host script mode
    """
    cmd_file = "start_xllm_serv.sh"
    if instance is not None:
        # Multi-host instance path: ignore legacy world_size<=16 gate.
        instance_name = instance.get("name") or task_config.get("instance_name") or "instance"
        cmd_file = f"start_xllm_serv_{instance_name}.sh"
        start_serv_script, script_path = generate_xllm_script(
            task_config,
            output_file_path,
            cmd_file,
            allow_large_world=True,
        )
        if not start_serv_script:
            logging.error("generate xllm multi-host serv script failed.")
            return ""
        return f"rm -rf {script_path}/log;cd {script_path}; bash {start_serv_script};"

    world_size = int(task_config.get("world_size") or 0)
    # 单机（保持原有分支语义）
    if 0 < world_size < 17:
        start_serv_script, script_path = generate_xllm_script(
            task_config, output_file_path, cmd_file
        )
        if start_serv_script:
            return f"rm -rf {script_path}/log;cd {script_path}; bash {start_serv_script};"
        logging.error("generate xllm serv script failed.")
        return ""

    # 多机但未走 deployment.instances：明确报错，避免静默 no-op
    if world_size > 16 and task_config.get("enable_disaggregated_pd") in (None, False, "false"):
        logging.error(
            "world_size=%s multi-host requires deployment.instances; "
            "legacy world_size-only multi-host branch is disabled.",
            world_size,
        )
        return ""

    # PD 分离：预留口子
    if task_config.get("enable_disaggregated_pd") in (True, "true"):
        logging.error(
            "PD-disaggregated xLLM startup is not implemented yet; "
            "use deployment.instances with role prefill/decode in a later iteration."
        )
        return ""

    logging.error("xllm param error.")
    return ""


def wait_xllm_serv_start(task_config, serving_wait_time, output_file_path: str,
                         log_file: str = None, success_keywords=None):
    from xtask.utils.watchdog import WatchdogConfig, WatchdogManager
    state = False

    current_pid = os.getpid()
    if log_file is None:
        log_file = os.path.join(output_file_path, "log/node_0.log")
    if success_keywords is None:
        success_keywords = ["Brpc Server started on port", "服务启动完成"]

    device_ids = _resolve_device_ids(task_config)
    # Large multi-host models (e.g. DeepSeek 16+16) need longer than default 300s.
    #ready_timeout = int(
    #    task_config.get("serving_ready_timeout")
    #    or task_config.get("serving_timeout")
    #    or 300
    #)
    ready_timeout = 320
    test_config = WatchdogConfig(
        log_file=log_file,
        success_keywords=success_keywords,
        npu_card_ids=device_ids,
        target_pids=[current_pid],
        timeout=ready_timeout,
        log_check_interval=1.0,
        npu_check_interval=2.0
    )

    watchdog_manager = WatchdogManager(test_config, serving_wait_time)
    try:
        watchdog_manager.start()
    except KeyboardInterrupt:
        logging.info("接收到键盘中断，停止Watchdog")
        watchdog_manager.stop()
    except Exception as e:
        logging.error(f"Watchdog运行异常：{str(e)}", exc_info=True)
        watchdog_manager.stop()

    return state
