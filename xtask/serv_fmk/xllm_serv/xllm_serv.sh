export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# export ASCEND_RT_VISIBLE_DEVICES=0
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
# export ASDOPS_LOG_TO_FILE=1 
# export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"

# export LCCL_DETERMINISTIC=1 
# export HCCL_DETERMINISTIC=true
# export ATB_MATMUL_SHUFFLE_K_ENABLE=0
# export INF_NAN_MODE_ENABLE=0
# export PROFILING_MODE=dynamic

\rm -rf /root/atb/log/
\rm -rf /root/ascend/log/
\rm -rf core.*

MODEL_PATH="/export/home/models/DeepSeek-V3.1_W8A8_NEWQUANT_10_1205"


MASTER_NODE_ADDR="11.87.191.116:9590"
START_PORT=14830
START_DEVICE=0
LOG_DIR="log"
NNODES=16
WORLD_SIZE=16

export HCCL_IF_BASE_PORT=43439

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  /export/home/weinan5/xllm_test_060/xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$WORLD_SIZE \
    --node_rank=$i \
    --max_memory_utilization=0.9 \
    --max_tokens_per_batch=20000 \
    --max_seqs_per_batch=3000 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_mla=true \
    --ep_size=4 \
    --dp_size=4 \
    > $LOG_FILE 2>&1 &
done
