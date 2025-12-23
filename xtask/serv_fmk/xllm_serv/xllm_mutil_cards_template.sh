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
MODEL_PATH={MODEL_PATH}
START_PORT={START_PORT}
START_DEVICE={START_DEVICE}
NNODES={NNODES}
WORLD_SIZE={WORLD_SIZE}
LOG_DIR={LOG_DIR}
BLOCK_SIZE={BLOCK_SIZE}
ENABLE_PREFIX_CACHE={ENABLE_PREFIX_CACHE}
ENABLE_CHUNKED_PREFILL={ENABLE_CHUNKED_PREFILL}
COMMUNICATION_BACKEND={COMMUNICATION_BACKEND}
ENABLE_SCHEDULE_OVERLAP={ENABLE_SCHEDULE_OVERLAP}
ENABLE_MLA={ENABLE_MLA}
MAX_MEMORY_UTILIZATION={MAX_MEMORY_UTILIZATION}
MAX_TOKENS_PER_BATCH={MAX_TOKENS_PER_BATCH}
MAX_SEQS_PER_BATCH={MAX_SEQS_PER_BATCH}
DP_SIZE={DP_SIZE}
EP_SIZE={EP_SIZE}
SERVING_EXTENSIABLE_PARAM={SERVING_EXTENSIABLE_PARAM}
USERID={USERID}
mkdir -p $LOG_DIR
# ========================================

# HCCL 端口配置
export HCCL_IF_BASE_PORT=43439

# 主节点地址
MASTER_NODE_ADDR="$LOCAL_IP:9590"
LOG_DIR="log"

# 启动服务节点
for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm_command=(
    xllm
    --model "$MODEL_PATH"
    --port "$PORT"
    --devices "npu:$DEVICE"
    --master_node_addr "$MASTER_NODE_ADDR"
    --nnodes "$i"
    --node_rank "$i"
    --max_memory_utilization "$MAX_MEMORY_UTILIZATION"
    --max_tokens_per_batch "$MAX_TOKENS_PER_BATCH"
    --max_seqs_per_batch "$MAX_SEQS_PER_BATCH"
    --block_size "$BLOCK_SIZE"
    --enable_prefix_cache "$ENABLE_PREFIX_CACHE"
    --enable_chunked_prefill "$ENABLE_CHUNKED_PREFILL"
    --communication_backend "$COMMUNICATION_BACKEND"
    --enable_schedule_overlap "$ENABLE_SCHEDULE_OVERLAP"
    --enable_mla "$ENABLE_MLA"
    $SERVING_EXTENSIABLE_PARAM
  )
  echo "服务启动命令: ${xllm_command[*]}"
  "${xllm_command[@]}" > "$LOG_FILE" 2>&1 &
  echo "启动节点 $i，设备 npu:$DEVICE，端口 $PORT，日志: $LOG_FILE"
done

echo "所有节点已启动，主节点地址: $MASTER_NODE_ADDR"

chown -R $USERID:$USERID $LOG_DIR