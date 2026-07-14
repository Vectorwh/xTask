#!/bin/bash
# xllm_serv.sh - XLLM 服务启动脚本模板

# 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

# Ascend 环境设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source ${ASCEND_TOOLKIT_HOME}/opp/vendors/custom_xllm_math/bin/set_env.bash

# 日志和性能配置
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

# 并行配置
export OMP_NUM_THREADS=12

# HCCL 配置
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"

# 清理日志和核心文件
rm -rf /root/atb/log/
rm -rf /root/ascend/log/
rm -rf core.*

# ============ 从配置读取的参数 ============
# 以下变量将从 YAML 配置文件读取
LOCAL_IP={LOCAL_IP}
PROGRESS_CONN_PORT={PROGRESS_CONN_PORT}
MODEL_PATH={MODEL_PATH}
START_PORT={START_PORT}
START_DEVICE={START_DEVICE}
NNODES={NNODES}
WORLD_SIZE={WORLD_SIZE}
LOG_DIR={LOG_DIR}
COMMUNICATION_BACKEND={COMMUNICATION_BACKEND}
USERID={USERID}
SERVING_EXTENSIABLE_PARAM={SERVING_EXTENSIABLE_PARAM}
mkdir -p $LOG_DIR
# ========================================

# HCCL 端口配置
export HCCL_IF_BASE_PORT={HCCL_IF_PORT}

# 主节点地址
MASTER_NODE_ADDR="$LOCAL_IP:$PROGRESS_CONN_PORT"

# 启动服务节点
export ASCEBD_RT_VISIBLE_DEVICES=$START_DEVICE
LOG_DIR="log"
LOG_FILE="$LOG_DIR/node_0.log"

xllm_command=(
  xllm \
  --model "$MODEL_PATH" \
  --port "$START_PORT" \
  --device_id="$START_DEVICE" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --nnodes="$WORLD_SIZE" \
  --node_rank=0 \
  --communication_backend="$COMMUNICATION_BACKEND" \
  $SERVING_EXTENSIABLE_PARAM
)

echo "服务启动命令: ${xllm_command[*]}"

"${xllm_command[@]}" > "$LOG_FILE" 2>&1 &

echo "启动设备 npu:$DEVICE，端口 $PORT，日志: $LOG_FILE"

chown -R $USERID:$USERID $LOG_DIR