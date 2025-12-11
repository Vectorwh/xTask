#!/usr/bin/env python3
import os
import yaml
import socket
import argparse
import logging
import time


def get_local_ip():
    """获取本地 IP 地址"""
    try:
        # 创建一个临时连接获取本地 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def load_yaml_config(yaml_file):
    """加载 YAML 配置文件"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_single_card_script(config, output_file="xllm_serv_single_card.sh"):
    """生成单卡启动脚本"""
    
    # 获取本地 IP
    local_ip = get_local_ip()
    if config.get('node_addr') and config['node_addr'] != '0.0.0.0':
        local_ip = config['node_addr']
    
    # 参数映射
    params = {
        'LOCAL_IP': local_ip,
        'MODEL_PATH': config.get('model_path', '/models/Qwen3-4B'),
        'START_PORT': config.get('node_port', 14830),
        'START_DEVICE': config.get('start_device_id', 0),
        'NNODES': config.get('nodes', 1),
        'WORLD_SIZE': config.get('world_size', 1),
        'LOG_DIR': 'log',
        'BLOCK_SIZE': config.get('block_size', 128),
        'ENABLE_PREFIX_CACHE': str(config.get('enable_prefix_cache', False)).lower(),
        'ENABLE_CHUNKED_PREFILL': str(config.get('enable_chunked_prefill', False)).lower(),
        'COMMUNICATION_BACKEND': config.get('communication_backend', 'hccl'),
        'ENABLE_SCHEDULE_OVERLAP': str(config.get('enable_schedule_overlap', False)).lower(),
        'ENABLE_MLA': str(config.get('enable_mla', False)).lower(),
        'MAX_MEMORY_UTILIZATION': config.get('max_memory_utiliization', 0.9),
        'MAX_TOKENS_PER_BATCH': config.get('max_tokens_per_batch', 20000),
        'MAX_SEQS_PER_BATCH': config.get('max_seqs_per_batch', 256),
        'DP_SIZE': 1,  # 默认值
        'EP_SIZE': 1,  # 默认值
    }
    
    # 读取模板
    template_path = "xllm_serv_template.sh"
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        # 如果模板不存在，使用硬编码的模板
        template = """#!/bin/bash
# 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"

rm -rf /root/atb/log/
rm -rf /root/ascend/log/
rm -rf core.*
mkdir -p log

# ============ 配置参数 ============
LOCAL_IP={LOCAL_IP}
MODEL_PATH="{MODEL_PATH}"
START_PORT={START_PORT}
START_DEVICE={START_DEVICE}
NNODES={NNODES}
WORLD_SIZE={WORLD_SIZE}
LOG_DIR="{LOG_DIR}"
BLOCK_SIZE={BLOCK_SIZE}
ENABLE_PREFIX_CACHE={ENABLE_PREFIX_CACHE}
ENABLE_CHUNKED_PREFILL={ENABLE_CHUNKED_PREFILL}
COMMUNICATION_BACKEND="{COMMUNICATION_BACKEND}"
ENABLE_SCHEDULE_OVERLAP={ENABLE_SCHEDULE_OVERLAP}
ENABLE_MLA={ENABLE_MLA}
MAX_MEMORY_UTILIZATION={MAX_MEMORY_UTILIZATION}
MAX_TOKENS_PER_BATCH={MAX_TOKENS_PER_BATCH}
MAX_SEQS_PER_BATCH={MAX_SEQS_PER_BATCH}
DP_SIZE={DP_SIZE}
EP_SIZE={EP_SIZE}
# =================================

export HCCL_IF_BASE_PORT=43439

MASTER_NODE_ADDR="$LOCAL_IP:$START_PORT"

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  
  /xllm_test_060/xllm \
    --model "$MODEL_PATH" \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$WORLD_SIZE \
    --node_rank=$i \
    --max_memory_utilization=$MAX_MEMORY_UTILIZATION \
    --max_tokens_per_batch=$MAX_TOKENS_PER_BATCH \
    --max_seqs_per_batch=$MAX_SEQS_PER_BATCH \
    --block_size=$BLOCK_SIZE \
    --enable_prefix_cache=$ENABLE_PREFIX_CACHE \
    --enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL \
    --communication_backend="$COMMUNICATION_BACKEND" \
    --enable_schedule_overlap=$ENABLE_SCHEDULE_OVERLAP \
    --enable_mla=$ENABLE_MLA \
    --dp_size=$DP_SIZE \
    --ep_size=$EP_SIZE \
    > "$LOG_FILE" 2>&1 &
    
  echo "启动节点 $i，设备 npu:$DEVICE，端口 $PORT，日志: $LOG_FILE"
done

echo "所有节点已启动，主节点地址: $MASTER_NODE_ADDR"
"""
    
    # 替换模板中的参数
    script_content = template
    for key, value in params.items():
        script_content = script_content.replace(f'{{{key}}}', str(value))
    
    output_file_path = os.path.join("./outputs", f"{config.get('case_name').lower()}")
    # 写入输出文件
    with open(os.path.join(output_file_path, output_file), 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(os.path.join(output_file_path, output_file), 0o755)
    
    logging.debug("generate single card script successfully.")
    
    return output_file, output_file_path

def generate_xllm_serv_commands(task_config, framework_config):
    cmd_file = "start_xllm_serv.sh"
    # 单卡
    if int(task_config.get("world_size")) == 1:
        start_serv_script, script_path = generate_single_card_script(task_config, cmd_file)
        if start_serv_script:
            startup_cmd = f"cd {script_path};bash {start_serv_script}"
        else:
            logging.error("generate xllm serv script failed.")
    # 多卡
    elif 1 < int(task_config.get("world_size")) < 17:
        pass
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


def main():
    parser = argparse.ArgumentParser(description='从 YAML 配置生成 XLLM 启动脚本')
    parser.add_argument('--config', type=str, default='case_qwen3.yaml', help='YAML 配置文件路径')
    parser.add_argument('--output', type=str, default='run_xllm_serv.sh', help='输出脚本路径')
    parser.add_argument('--template', type=str, default='xllm_serv.sh', help='模板脚本路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_yaml_config(args.config)
    
    # 生成脚本
    generate_single_card_script(config, args.output)


def wait_xllm_serv_start(framework):
    state = False
    
    time.sleep(20)

    return state


if __name__ == "__main__":
    main()
