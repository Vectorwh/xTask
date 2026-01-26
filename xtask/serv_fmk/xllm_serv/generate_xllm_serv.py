#!/usr/bin/env python3
import os
import yaml
import socket
import logging
import time
from xtask.utils import get_host_ip as get_local_ip

def load_yaml_config(yaml_file):
    """加载 YAML 配置文件"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_xllm_script(config, output_file="xllm_serv_single_card.sh"):
    """生成单卡启动脚本"""
    
    # 获取本地 IP
    local_ip = get_local_ip()
    if config.get('host_addr') and config['host_addr'] != '0.0.0.0':
        local_ip = config['host_addr']
    
    # 参数映射
    params = {
        'LOCAL_IP': local_ip,
        'MODEL_PATH': config.get('model_path', '/models/Qwen3-4B'),
        'START_PORT': config.get('port', 14830),
        'START_DEVICE': config.get('start_device_id', 0),
        'NNODES': config.get('num_devices', 1),
        'WORLD_SIZE': config.get('world_size', 1),
        'LOG_DIR': 'log',
        'BLOCK_SIZE': config.get('block_size', 128),
        'ENABLE_PREFIX_CACHE': config.get('enable_prefix_cache', False),
        'ENABLE_CHUNKED_PREFILL': config.get('enable_chunked_prefill', False),
        'COMMUNICATION_BACKEND': config.get('communication_backend', 'hccl'),
        'ENABLE_SCHEDULE_OVERLAP': config.get('enable_schedule_overlap', False),
        'ENABLE_MLA': config.get('enable_mla', False),
        'MAX_MEMORY_UTILIZATION': config.get('max_memory_utiliization', 0.9),
        'MAX_TOKENS_PER_BATCH': config.get('max_tokens_per_batch', 20000),
        'MAX_SEQS_PER_BATCH': config.get('max_seqs_per_batch', 256),
        'DP_SIZE': config.get("serving_extensiable_param", ""),  # 默认值
        'EP_SIZE': config.get("serving_extensiable_param", ""),  # 默认值
        "SERVING_EXTENSIABLE_PARAM": config.get("serving_extensiable_param", ""),
        "USERID": config.get("userid", 0)
    }
    
    # 读取模板
    if params.get("WORLD_SIZE") == 1:
        template_path = "xllm_single_card_template.sh"
    elif 1 < params.get("WORLD_SIZE") < 17:
        template_path = "xllm_mutil_cards_template.sh"
    else:
        template_path = ""
        logging.error("WORLD_SIZE is: %s is not supported yet."% params.get("WORLD_SIZE"))
    if template_path:
        template_path = os.path.join("xtask/serv_fmk/xllm_serv/", template_path)
        logging.info("template_path is: %s"% template_path)
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

    # 替换模板中的参数
    script_content = template
    if script_content:
        for key, value in params.items():
            if key == "MODEL_PATH":
                from xtask.utils.container_manager import path_mapping_in_container
                value = path_mapping_in_container(value)
            script_content = script_content.replace(f'{{{key}}}', str(value))
        
        output_file_path = os.path.join("./outputs", f"{config.get('case_name').lower()}")
        # 写入输出文件
        with open(os.path.join(output_file_path, output_file), 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(os.path.join(output_file_path, output_file), 0o755)
        
        logging.debug("generate single card script successfully.")
    else:
        output_file = ""
        output_file_path = ""
    
    return output_file, output_file_path

def generate_xllm_serv_commands(task_config, framework_config):
    cmd_file = "start_xllm_serv.sh"
    # 单卡
    if 0 < int(task_config.get("world_size")) < 17:
        start_serv_script, script_path = generate_xllm_script(task_config, cmd_file)
        if start_serv_script:
            startup_cmd = f"cp /xllm_test_060/vendors/xllm/op_api/lib/libcust_opapi.so /usr/lib64/;" \
                f"rm -rf {script_path}/log;cd {script_path}; bash {start_serv_script};"
        else:
            logging.error("generate xllm serv script failed.")
    # 多机
    elif int(task_config.get("world_size")) > 16 and \
        task_config.get("enable_disaggregated_pd") == "false":
        pass
    # PD分离
    elif task_config.get("enable_disaggregated_pd") == "true":
        pass
    else:
        logging.error("xllm param error.")
    
    return startup_cmd


def wait_xllm_serv_start(task_config, serving_wait_time):
    from xtask.utils.watchdog import WatchdogConfig, WatchdogManager
    state = False
    
    output_path = os.path.join(os.getcwd(), "outputs", task_config.get("case_name").lower())
    current_pid = os.getpid()
    test_config = WatchdogConfig(
        log_file=os.path.join(output_path, f"log/node_{task_config.get('start_device_id')}.log"),  # 推理服务日志路径
        success_keywords=["is serving on port", "服务启动完成"],  # 启动成功关键词
        npu_card_id=task_config.get("start_device_id"),  # 监控0号NPU卡
        target_pid=current_pid,  # 推理服务PID（可从容器启动时获取）
        timeout=300,  # 5分钟超时
        log_check_interval=1.0,
        npu_check_interval=2.0
    )

    # 2. 启动Watchdog
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

