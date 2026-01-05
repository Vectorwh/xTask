#昇腾推理软件四件套环境变量
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
#设置自己的ranktable_file_path，machine_IP， master_IP
export RANK_TABLE_FILE="ranktable_file_path/hccl_2s_16p.json"
export MIES_CONTAINER_IP='machine_IP'
export MASTER_IP='master_IP'

#虚拟内存开启 bs、seq_len能大点
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#atb显存分配策略，3显存能多点
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
#AIV模式
export HCCL_OP_EXPANSION_MODE="AIV"
#显存因子
export NPU_MEMORY_FRACTION=0.92
#1是走hccl通信方式
export ATB_LLM_HCCL_ENABLE=1
#防止blabla溢出
export INF_NAN_MODE_ENABLE=1

#8244特性防止oom开关，atb的显存优化
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
#hccl通信超时时间
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0

#A3到15， A3双机到3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#atb异步执行
export ATB_OPERATION_EXECUTE_ASYNC=1
#matmul转置
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
#PCIE通信相关
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
#HCCL buff
export HCCL_BUFFSIZE=64

#异步发射
export MINDIE_ASYNC_SCHEDULING_ENABLE=1

#jemalloc优化
export LD_PRELOAD="path_to_file/libjemalloc.so:$LD_PRELOAD"

#队列优化特性
export TASK_QUEUE_ENABLE=2

#关闭所有日志打印和导出文件保证性能最优
for var in $(compgen -e | grep 'STDOUT$'); do
export "$var=0"
done

for var in $(compgen -e | grep 'LOG_TO_FILE$'); do
export "$var=0"
done

#遇到config.json文件权限问题
find /usr/local/lib/python3.11/site-packages/mindie* -name config.json |xargs chmod -R 640
chmod -R 640 "ranktable_file_path/hccl_2s_16p.json"

#mindie日志，测试性能时关闭
#export MINDIE_LOG_TO_STDOUT=1
#CPU线程数，可加快启服务时的权重加载
export OMP_NUM_THREADS=10
#HCCL显存分配策略，配置后不会跑着跑着继续申请显存。
export HCCL_ALGO="level0:NA;level1:pipeline"


curl https://pm8jb5oh5ebeckgaj9qjdqme59d85jga.openapi-sj.sii.edu.cn/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $INF_API_KEY" \
  -d '{
    "model": "deepseek_v32",
    "prompt": "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>Hello! I am DeepSeek.<｜end▁of▁sentence｜><｜User｜>自我介绍<｜Assistant｜><think>",
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 0.95,
    "stop": ["<｜end▁of▁sentence｜>"]
  }'


  export HCCL_IF_IP=141.61.81.134
export GLOO_SOCKET_IFNAME="enp48s3u1u2"
export TP_SOCKET_IFNAME="enp48s3u1u2"
export HCCL_SOCKET_IFNAME="enp48s3u1u2"
export HCCL_BUFFSIZE=1024
export VLLM_USE_V1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_VERSION=0.11.0
vllm serve /mnt/share/w00933241/Qwen3-235B-A22B-w8a8-rot \
        --served-model-name "qwen" \
        --host 0.0.0.0 \
        --port 20002 \
        --async-scheduling \
        --tensor-parallel-size 4 \
        --data-parallel-size 4 \
        --data-parallel-size-local 4 \
        --data-parallel-start-rank 0 \
        --data-parallel-address 141.61.81.134 \
        --data-parallel-rpc-port 2345 \
        --max-num-seqs 100 \
        --max-model-len 131072 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9 \
        --enable-expert-parallel \
        --no-enable-prefix-caching \
        --quantization "ascend" \
        --trust-remote-code \
        --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \ 
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"ascend_scheduler_config":{"enabled":false}}'




export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
#export CPU_AFFINITY_CONF=1
export CPU_AFFINITY_CONF=1,npu_affine:1
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
#export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
#export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_TORCH_PROFILER_DIR=/export/home/weinan/yjh/profiles/
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export MOONCAKE_CONFIG_PATH="/export/home/weinan/yjh/mooncake.json"
rm -rf *.core

#vllm serve /export/home/weinan/weights/DeepSeek-V3.2-Exp-W8A8/ \
#vllm serve /export/home/weinan/yjh/DeepSeek-V3.2-Exp-w4a8/ \
vllm serve /export/home/weinan/weights/DeepSeek-V3.2-Exp-W8A8/ \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 16 \
--seed 1024 \
--max-num-seqs 2 \
--quantization ascend \
--served-model-name deepseek_v32 \
--max-model-len 24576 \
--max-num-batched-tokens 24576 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.95  \
--compilation-config '{"cudagraph_capture_sizes": [2,4,8,16]}' \
--speculative-config '{"num_speculative_tokens": 2, "method": "mtp"}' 