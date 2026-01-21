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





#!/bin/bash
# 自动脚本，需要 root 权限在宿主机执行，独占推理/训练可用

# 可配置(给主进程、acl_thread分配几个CPU；main_process_range + acl_thread_range < cpu_suggest_allocation_range - 3)
main_process_range=5
acl_thread_range=1

# 初始化数组
npu_id=()
start_cpus=()
end_cpus=()
all_npu_id=()
all_start_cpus=()
all_end_cpus=()

DO_DEV=false
DO_IRQ=false
DO_THREAD=false
DO_ALL=false

# 通用平均分配函数
average_alloc() {
    local -n starts=$1
    local -n ends=$2
    local count=${#starts[@]}

    for ((i=0; i<count; i++)); do
        s=${starts[$i]}
        e=${ends[$i]}
        shared=()
        for ((j=0; j<count; j++)); do
            if [[ ${starts[$j]} -eq $s && ${ends[$j]} -eq $e ]]; then
                shared+=($j)
            fi
        done
        if (( ${#shared[@]} > 1 )); then
            len=$(( e - s + 1 ))
            chunk=$(( len / ${#shared[@]} ))
            for ((k=0; k<${#shared[@]}; k++)); do
                idx=${shared[$k]}
                new_start=$(( s + k * chunk ))
                new_end=$(( new_start + chunk - 1 ))
                starts[$idx]=$new_start
                ends[$idx]=$new_end
            done
        fi
    done
}

alloc_cpu() {
    declare -A npu_affinity
    RUNNING_NPUS=$(npu-smi info | awk '/NPU     Chip/{flag=1;next} flag' \
        | grep -v "No running processes" \
        | awk -F'|' '{print $2}' | awk '{print $1}' | sort -u)

    while read -r line; do
        if [[ $line =~ ^NPU[0-9] ]]; then
            npu=$(echo "$line" | awk '{print $1}' | grep -o '[0-9]\+')
            affinity=$(echo "$line" | awk '{print $NF}')
            [[ $affinity == "Affinity" ]] && continue
            IFS=',' read -ra ranges <<< "$affinity"
            npu_affinity[$npu]="${ranges[*]}"

            if echo "$RUNNING_NPUS" | grep -qw "${npu#NPU}"; then
                npu_id+=("$npu")
            fi
            all_npu_id+=("$npu")
        fi
    done < <(npu-smi info -t topo)

    # 先在 NUMA 层面分配
    for npu in "${npu_id[@]}"; do
        ranges=(${npu_affinity[$npu]})
        # 简单策略：轮流分配到第一个或第二个区间
        idx=$((npu % ${#ranges[@]}))
        start=$(echo "${ranges[$idx]}" | cut -d- -f1)
        end=$(echo "${ranges[$idx]}" | cut -d- -f2)
        start_cpus+=("$start")
        end_cpus+=("$end")
    done

    for npu in "${all_npu_id[@]}"; do
        ranges=(${npu_affinity[$npu]})
        idx=$((npu % ${#ranges[@]}))
        start=$(echo "${ranges[$idx]}" | cut -d- -f1)
        end=$(echo "${ranges[$idx]}" | cut -d- -f2)
        all_start_cpus+=("$start")
        all_end_cpus+=("$end")
    done

    average_alloc start_cpus end_cpus
    average_alloc all_start_cpus all_end_cpus

    # 输出结果（运行中的 NPU）
    echo "==== 正在运行的 NPU 建议分配 ===="
    for ((i=0; i<${#start_cpus[@]}; i++)); do
        echo "NPU${npu_id[$i]} -> CPU Suggest Allocation: start=${start_cpus[$i]}, end=${end_cpus[$i]}"
    done

    # 输出结果（所有 NPU）
    echo "==== 使用所有 NPU 卡时建议分配 ===="
    for ((i=0; i<${#all_start_cpus[@]}; i++)); do
        echo "NPU${all_npu_id[$i]} -> CPU Suggest Allocation: start=${all_start_cpus[$i]}, end=${all_end_cpus[$i]}"
    done
}

# 将10进制CPU号转换为16进制mask码
cpu_to_mask() {
    local cpu=$1
    local group=$((cpu / 32))
    local bit=$((cpu % 32))
    local value=$((1 << bit))
    local mask=$(printf "%08x" $value)

    # 拼接成逗号分隔的掩码字符串
    for ((i=1; i<=group; i++)); do
        mask="${mask},00000000"
    done
    echo $mask
}

check_and_stop_irqbalance() {
    # 检查 irqbalance 是否存在
    if systemctl list-unit-files | grep -q irqbalance.service; then
        if systemctl is-active --quiet irqbalance; then
            echo "检测到 irqbalance 服务正在运行，正在关闭..."
            systemctl stop irqbalance
        else
            echo "irqbalance 服务已安装，但当前未运行"
        fi
    elif service --status-all 2>/dev/null | grep -q irqbalance; then
        if service irqbalance status >/dev/null 2>&1; then
            echo "检测到 irqbalance 服务正在运行，正在关闭..."
            service irqbalance stop
            echo "irqbalance 已关闭"
        else
            echo "irqbalance 服务已安装，但当前未运行"
        fi
    else
        echo "当前环境未检测到 irqbalance 服务"
    fi
}

bind_irq() {
    echo "==== 开始对所有 NPU 卡的中断绑核 ===="
    check_and_stop_irqbalance
    SQ_IRQ_LIST=($(cat /proc/interrupts | grep sq_send_trigger_irq | cut -d: -f1))
    for i in "${!all_npu_id[@]}"; do
        CARD=${all_npu_id[$i]}
        SQ_CPU=${all_start_cpus[$i]}
        CQ_CPU=$((SQ_CPU+1))

        # 获取 PCI 地址
        PCI_ADDR=$(npu-smi info -t board -i $CARD | grep "PCIe Bus Info" | awk '{print $NF}' | tr '[:upper:]' '[:lower:]')
        if [ -z "$PCI_ADDR" ]; then
            echo "未找到 NPU 卡 $CARD 的 PCI 地址"
            continue
        fi

        # 获取该卡的 MSI 中断号列表
        NPU_IRQ_LIST=$(ls /sys/bus/pci/devices/$PCI_ADDR/msi_irqs/ | sort -n)
        for irq in "${SQ_IRQ_LIST[@]}"; do
            if echo "${NPU_IRQ_LIST[@]}" | grep -qw "$irq"; then
                SQ_IRQ=$irq
                CQ_IRQ=$((SQ_IRQ+1))
            fi
        done
        if [ -z "$SQ_IRQ" ]; then
            echo "未找到 NPU 卡 $CARD 的 SQ 中断"
            continue
        fi

        echo "NPU卡 $CARD (PCI $PCI_ADDR): SQ IRQ=$SQ_IRQ → CPU$SQ_CPU, CQ IRQ=$CQ_IRQ → CPU$CQ_CPU"

        # 绑 SQ
        echo $(cpu_to_mask $SQ_CPU) > /proc/irq/$SQ_IRQ/smp_affinity
        # 绑 CQ
        echo $(cpu_to_mask $CQ_CPU) > /proc/irq/$CQ_IRQ/smp_affinity
    done
}

bind_dev() {
    echo "==== 开始对所有 NPU 卡的dev进程绑核 ===="
    # 查询所有dev进程
    THREADS=($( ps aux | grep -E 'dev(1[0-5]|[0-9])_sq_task' | awk '{
        if (match($0, /dev([0-9]+)_sq_task/, arr)) {
            print arr[1] " " $0  # 输出数字和原始行
        }
    }' | sort -n | awk '{
        sub(/^[0-9]+ /, ""); print  # 移除数字，输出原始行
    }' |awk '{print $2}'))

    for i in "${!THREADS[@]}"; do
        THREAD_ID="${THREADS[i]}"
        CORE_ID="$((${all_start_cpus[$i]}+2))"

        echo "绑定dev线程 ${THREAD_ID} 到核心 ${CORE_ID}"
        taskset -acp "${CORE_ID}" "${THREAD_ID}"  
    done
}

bind_acl_release_thread() {
    echo "==== 开始对运行中 NPU 卡的主进程、acl_thread、release_thread 绑核 ===="
    
    THDNAME="acl_thread"

    # 将 THREADS 定义为数组：使用 ps 命令获取匹配 THDNAME 的线程 ID（去重）
    THREADS=($(ps -Te | grep "${THDNAME}" | grep -v grep | awk '!a[$1]++' | awk '{print $1}'))

    # 检查数组长度是否匹配
    if [ ${#THREADS[@]} -ne ${#start_cpus[@]} ]; then
        echo "错误: THREADS 数组长度（${#THREADS[@]}）与 CPUID 数组长度（${#start_cpus[@]}）不匹配"
        exit 1
    fi
    
    # 无需修改，适配CPU范围
    main_process_range=$((main_process_range - 1))
    acl_thread_range=$((acl_thread_range - 1))

    # 使用索引并行迭代两个数组
    for i in "${!THREADS[@]}"; do
        THREAD_ID="${THREADS[i]}"
        CORE_ID="${start_cpus[i]}"
        CORE_ID_BEGIN=`echo "${CORE_ID} + 3" | bc`
        CORE_ID_END=`echo "${CORE_ID} + 3 + ${main_process_range}" | bc`
        if [[ $main_process_range -eq 0 ]]; then
            echo "绑定线程 ${THREAD_ID} 到核心 ${CORE_ID_BEGIN}"
            taskset -acp "${CORE_ID_BEGIN}" "${THREAD_ID}"
        else
            echo "绑定线程 ${THREAD_ID} 到核心 ${CORE_ID_BEGIN}-${CORE_ID_END}"
            taskset -acp "${CORE_ID_BEGIN}-${CORE_ID_END}" "${THREAD_ID}"
        fi

        ACL_CORE_ID_BEGIN=`echo "${CORE_ID} + 4 + ${main_process_range}" | bc`
        ACL_CORE_ID_END=`echo "${CORE_ID} + 4 + ${main_process_range} + ${acl_thread_range}" | bc`
        ACL_ID=`ps -Te | grep ${THREAD_ID} | grep acl_thread | grep -v grep | awk '!a[$1]++' | awk '{print $2}'`
        if [[ $acl_thread_range -eq 0 ]]; then
            echo "绑定线程 ${ACL_ID} 到核心 ${ACL_CORE_ID_BEGIN}"
            taskset -cp "${ACL_CORE_ID_BEGIN}" "${ACL_ID}"
        else
            echo "绑定线程 ${ACL_ID} 到核心 ${ACL_CORE_ID_BEGIN}-${ACL_CORE_ID_END}"
            taskset -cp "${ACL_CORE_ID_BEGIN}-${ACL_CORE_ID_END}" "${ACL_ID}"
        fi

        RELEASE_CORE_ID=`echo "${ACL_CORE_ID_END} + 1" | bc`
        RELEASE_ID=`ps -Te | grep ${THREAD_ID} | grep release_thread | grep -v grep | awk '!a[$1]++' | awk '{print $2}'`
        echo "绑定线程 ${RELEASE_ID} 到核心 ${RELEASE_CORE_ID}"
        taskset -cp "${RELEASE_CORE_ID}" "${RELEASE_ID}"

    done
}

main() {
    alloc_cpu
    if $DO_ALL; then
        bind_irq
        bind_dev
        bind_acl_release_thread
    else
        $DO_IRQ && bind_irq
        $DO_DEV && bind_dev
        $DO_THREAD && bind_acl_release_thread
    fi
}

# 打印帮助信息
usage() {
    cat <<EOF
用法: $0 [选项]  （指定 -d/-i/-a/-r 参数时需要以 root 权限在宿主机执行，指定 -t 时可以在宿主机或 docker 容器中执行；dev_*_sq_task 进程绑定后永久生效，服务器重启时重置；sq_trigger_irq 与 cq_update_irq(首个) 中断，服务器重启或 irqbalance 服务重启时重置）

选项:
  -h, --help                    显示此帮助信息
  -d, --dev                     仅绑定所有 NPU 卡的 dev_*_sq_task 进程
  -i, --irq                     仅绑定所有 NPU 卡的 sq_trigger_irq 与 cq_update_irq(首个) 中断，会关闭 irqbalance 服务
  -t, --thread                  仅绑定正在运行 NPU 卡对应的主进程、acl_thread 与 release_thread 线程
  -a, --all                     进行以上所有绑核操作
  -m, --main_process_range <n>  设置每张 NPU 卡的主进程需要绑几个 CPU 核，默认为 5; -m与-l之和要小于min(cpu_suggest_allocation_range-3)
  -l, --acl_thread_range <n>    设置每张 NPU 卡的 acl_thread 需要绑几个 CPU 核，默认为 1; -m与-l之和要小于min(cpu_suggest_allocation_range-3)

示例:
  bash cpu_bind_all.sh -a
EOF
    exit 0
}

if [[ $# -eq 0 ]]; then
    usage
fi

# 参数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -d|--dev)
            DO_DEV=true
            shift
            ;;
        -i|--irq)
            DO_IRQ=true
            shift
            ;;
        -t|--thread)
            DO_THREAD=true
            shift
            ;;
        -a|--all)
            DO_ALL=true
            shift
            ;;
        -m|--main_process_range)
            main_process_range="$2"
            shift 2
            ;;
        -l|--acl_thread_range)
            acl_thread_range="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

main