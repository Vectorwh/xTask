#!/bin/bash
# xllm_serv.sh - XLLM 服务启动脚本模板

# 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
#export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
TRITON_BINARY_PATH_CONFIG="{TRITON_BINARY_PATH}"
if [ -n "$TRITON_BINARY_PATH_CONFIG" ]; then
  export TRITON_BINARY_PATH="$TRITON_BINARY_PATH_CONFIG"
else
  export TRITON_BINARY_PATH="$(python3 -c 'import importlib.util, pathlib; spec=importlib.util.find_spec("xllm"); print(pathlib.Path(next(iter(spec.submodule_search_locations))) / "triton_npu" / "binary")')"
fi
#export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

export ASCEND_RT_VISIBLE_DEVICES={VISIBLE_DEVICES}
export ASCEND_VISIBLE_DEVICES={VISIBLE_DEVICES}

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
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export PROFILING_MODE="dynamic"

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
MASTER_HOST={MASTER_HOST}
PROGRESS_CONN_PORT={PROGRESS_CONN_PORT}
MODEL_PATH={MODEL_PATH}
START_PORT={START_PORT}
START_DEVICE={START_DEVICE}
LOCAL_NNODES={LOCAL_NNODES}
GLOBAL_NNODES={GLOBAL_NNODES}
RANK_START={RANK_START}
NNODES={NNODES}
WORLD_SIZE={WORLD_SIZE}
VISIBLE_DEVICES={VISIBLE_DEVICES}
LOG_DIR={LOG_DIR}
COMMUNICATION_BACKEND={COMMUNICATION_BACKEND}
SERVING_EXTENSIABLE_PARAM={SERVING_EXTENSIABLE_PARAM}
USERID={USERID}
XLLM_BINARY="{XLLM_BINARY}"
mkdir -p $LOG_DIR
if [ "{PD_ENABLED}" = "true" ]; then
  export no_proxy="{PD_NO_PROXY}"
  export NO_PROXY="$no_proxy"
elif [ "{SERVICE_ROUTING}" = "true" ]; then
  export no_proxy="{SERVICE_NO_PROXY}"
  export NO_PROXY="$no_proxy"
fi
# ========================================

# HCCL 端口配置
export HCCL_IF_BASE_PORT={HCCL_IF_PORT}

# 主节点地址：跨机时 MASTER_HOST 为主节点 IP，本机 IP 使用 LOCAL_IP
MASTER_NODE_ADDR="$MASTER_HOST:$PROGRESS_CONN_PORT"
LOG_DIR="log"
chown "$USERID" "$LOG_DIR"
{STARTUP_ENVIRONMENT}
IFS=',' read -r -a DEVICE_IDS <<< "$ASCEND_RT_VISIBLE_DEVICES"
if [ "${#DEVICE_IDS[@]}" -ne "$LOCAL_NNODES" ]; then
  echo "device_ids count (${#DEVICE_IDS[@]}) does not match LOCAL_NNODES ($LOCAL_NNODES)" >&2
  exit 1
fi

# 启动服务节点
for (( i=0; i<$LOCAL_NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE="${DEVICE_IDS[$i]}"
  NODE_RANK=$((RANK_START + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  touch "$LOG_FILE"
  chown "$USERID" "$LOG_FILE"
  xllm_command=(
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) "$XLLM_BINARY" \
#   xllm \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$GLOBAL_NNODES" \
    --node_rank="$NODE_RANK" \
    --communication_backend="$COMMUNICATION_BACKEND" \
    $SERVING_EXTENSIABLE_PARAM
  )
  echo '服务启动命令: ${xllm_command[*]}' >> "$LOG_FILE"
  echo "启动节点 $NODE_RANK，设备 npu:$DEVICE，端口 $PORT，日志: $LOG_FILE" >> "$LOG_FILE"
  "${xllm_command[@]}" >> "$LOG_FILE" 2>&1 &
done

echo "所有节点已启动，主节点地址: $MASTER_NODE_ADDR，本机 LOCAL_IP: $LOCAL_IP，RANK_START: $RANK_START"

chown -R "$USERID" "$LOG_DIR"
