#！！！！！！！！！！！！版本信息请先阅读README.md文件！！！！！！！！！！！！！！！
#编码格式设置
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓用户配置修改区↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# 从命令行参数获取配置
VERSION=""
Sever_NAME=""
SERVICE_IP=""
SERVICE_PORT=""
MODEL_PATH=""
#服务的D卡数和总卡数，用于计算性能测试结果中单卡的性能
D_NUM=1
ALL_NUM=1
# 获取当前日期
CURRENT_DATE=$(date +"%Y%m%d%H%M")
#数据集总数设置：并发数的倍数（参考4~10倍）
Concurrent_Multiplier=4
# 定义多组参数值 (格式: "请求频率 并发数 输入长度 输出长度 数据集类型 图片/视频文件路径/音频时长")[音频数据集为"音频时长.wav"的格式]
#数据集ID： 1、合成数据集：SYN ；2、GSM8K数据集：GSM；3、Textvqa图片数据集：VQA；4、Videobench数据集: VID; (5)Vocalsound数据集：VSD；
PARAM_SETS=(
  "0 2 64 1024 VQA ./dataset/Textvqa/1080.jpg" 
  "0 1 1024 1024 SYN 0"
  "0 2 1024 1024 GSM 0"
  "0 3 50 512 VID ./dataset/Videobench/test.mp4"
  "0 32 50 1024 VSD 30.wav"
)

# 创建基准测试根目录，更改路径为容器中aisbench工具的路径
AISBENCHMARK_PATH="/workspace/benchmark"
#性能采集开关，1表示开启，0表示关闭。路径需要自己手动记录并更改。建议手动采集，该功能后续优化或日落。
PROFILL_ENABLE=0  
PROFILL_DATA_PATH="/workspace/profi/"
#创建的log日志存放地址，输出结果在日志文件里，根据时间区分单次执行，再根据输入配置区分不同的运行结果;
#用户可更改保存路径，推荐更改“${AISBENCHMARK_PATH}/LOG_PLACE”
LOG_SAVE_PATH="${AISBENCHMARK_PATH}/LOG_PLACE/LOG_${VERSION}_${SERVICE_IP}_${CURRENT_DATE}"
CSV_SAVE_PATH="${AISBENCHMARK_PATH}"
#数据集测试PY文件，测试指令如果更改，需要到下方更改数据集对应的名称，目前数据集任务命令如下，请根据框架更改命令，命令集在下方数据集执行处更改，默认为vllm-ascend推理框架。
# 推理框架Mindie (1)SYN →→ synthetic_gen (2)GSM →→ gsm8k_gen_0_shot_cot_str_perf (3)VQA →→ Textvqa_gen (4)VID →→ Videobench_gen (5)VSD →→ vocalsound_gen
# 推理框架Vllm   (1)SYN →→ synthetic_gen (2)GSM →→ gsm8k_gen_0_shot_cot_str_perf (3)VQA →→ Textvqa_gen_base64 (4)VID →→ Videobench_gen_base64 (5)VSD →→ vocalsound_gen_base64
#让自动化脚本工具全部在基准路径下，减小配置改动
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑用户配置修改区↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


mkdir -p "$LOG_SAVE_PATH"
cd "$(dirname "$0")"
#cd "$AISBENCHMARK_PATH"
#Aisbench数据集配置文件默认路径
MODEL_NAME=$(basename "$MODEL_PATH")
Synthetic_config="${AISBENCHMARK_PATH}/ais_bench/datasets/synthetic"
gsm8k_dataset_path="${AISBENCHMARK_PATH}/ais_bench/datasets/gsm8k"
textQVA_dataset_path="${AISBENCHMARK_PATH}/ais_bench/datasets/textvqa/textvqa_json"
videobench_path="${AISBENCHMARK_PATH}/ais_bench/datasets/videobench"
vocalsound_path="${AISBENCHMARK_PATH}/ais_bench/datasets/vocalsound/"
Stream_chat="${AISBENCHMARK_PATH}/ais_bench/benchmark/configs/models/vllm_api"
#通过pid文件检查是否已有运行中的脚本，首先定义pid文件路径，当前固定在脚本目录下
PIDFILE="run.pid"

echo "[run_benchmark]: 自动化测试工具 维护团队【小巧灵算力组-0day适配团队】            "
echo "#################################################################################################################"
echo "------------------------------------------时间 $CURRENT_DATE 新一轮测试开始 --------------------------------------"
echo "#################################################################################################################"

#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓函数声明区↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

##############################################################
# 函数一：检查调用后的数据集文件是否存在，默认30次check，每次check间隔1s。
check_file_exists() {
    local file_path="$1"
    local max_attempts="${2:-30}"    
    local interval="${3:-1}"          
    local description="${4:-数据集文件}" 
    local attempt=1
    echo "[run_benchmark]: 开始校验 $description: $file_path"
    
    while [ $attempt -le $max_attempts ]; do
        if [ -e "$file_path" ]; then
            echo "[run_benchmark]: [Success] 校验文件成功生成: $file_path "
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            echo "[run_benchmark]:等待 $description 生成中... (尝试 $attempt/$max_attempts)"
            sleep $interval
        fi
        ((attempt++))
    done
    
    echo "[run_benchmark]: [ERROR] 校验失败: $file_path 在 ${max_attempts} 次尝试后仍未找到"
    echo "[run_benchmark]: 测试任务结束，请检查文件路径是否正确"
    exit 1
    return 1
}
##############################################################

##############################################################
# 函数二：检查pid文件是否存在并处理现有进程
check_existing_process() {
    if [ -f "$PIDFILE" ]; then
        # 读取PID文件中的进程ID
        pid=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$pid" ]; then
            # 检查该PID是否仍在运行（使用ps命令）
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "错误：另一个自动化测试脚本实例正在运行（PID: $pid）。请等待其结束再重试。"
                exit 1
            else
                echo "警告：PID文件存在，但进程（PID: $pid）未运行。删除旧的PID文件。"
                rm -f "$PIDFILE"
            fi
        else
            echo "警告：PID文件为空或无效。删除它。"
            rm -f "$PIDFILE"
        fi
    fi
}
##############################################################

##############################################################
# 函数三：创建PID文件并在脚本退出时清理
setup_pid_file() {
    # 将当前脚本的PID写入文件
    echo $$ > "$PIDFILE"
    # 设置陷阱：在脚本退出（正常退出、中断或错误）时删除PID文件
    trap 'rm -f "$PIDFILE"; exit' EXIT INT TERM
}
##############################################################

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑函数声明区↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

######################测试流程开始###########################
#检查现有进程
check_existing_process
#创建pid文件并设置陷阱
setup_pid_file

echo "[run_benchmark]: 创建本次测试csv文件"
python3 ./log_parsing.py -o ${CSV_SAVE_PATH} -a ${D_NUM} -d ${ALL_NUM}
# 遍历所有用户输入参数
for params in "${PARAM_SETS[@]}"; do
  # 拆分参数到变量
  read -r request_rate concurrency input_length output_length dataset datapath<<< "$params"
  
  #生成vllm_api_stream_chat配置文件
  CONFIG_FILE="${Stream_chat}/vllm_api_stream_chat.py"
  
  cat > "$CONFIG_FILE" <<EOF

from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="$MODEL_PATH",
        model="$Sever_NAME",
        request_rate = $request_rate,
        retry = 2,
        host_ip = "$SERVICE_IP",
        host_port = $SERVICE_PORT,
        max_out_len = $output_length,
        batch_size= $concurrency,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            top_k = -1,
            top_p = 1,
            seed = None,
            repetition_penalty = 1,
            ignore_eos = True
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]


EOF
  
  Multimodal_test_file=$(basename "$datapath")
  #判断当前单次测试的数据集类型
  if [ "$dataset" = "SYN" ]; then
    echo -e "\n################################################[run_benchmark]:新的一次测试开启#######################################################"
    echo "[run_benchmark]:数据集类型是SYN"
    chmod 640 $CONFIG_FILE
    Data_Command_Type="synthetic_gen"
    echo "[run_benchmark]:[Success] 已生成配置文件: $CONFIG_FILE"
    echo "[run_benchmark]:[Success] 数据集类型选择: $dataset"
    echo "[run_benchmark]:[Success] 数据集命令选择: $Data_Command_Type"
    # 生成Synthetic配置文件
    CONFIG_FILE2="${Synthetic_config}/synthetic_config.py"
    RequestCount=$(($Concurrent_Multiplier*$concurrency))
    cat > "$CONFIG_FILE2" <<EOF

synthetic_config = {
    "Type":"string",   # [tokenid/string]，生成的随机数据集类型，支持固定长度的随机tokenid，和随机长度的string，两种类型的数据集
    "RequestCount": $RequestCount, # 生成的请求条数，应与模型侧配置文件中的 decode_batch_size 一致
    "TrustRemoteCode": False, #是否信任远端代码，tokenid模式下需要加载tokenizer生成tokenid，默认为Fasle
    "StringConfig" : {  # string类型的随机数据集的配置相关项，请参考以上注释处："StringConfig中的随机生成方法参数说明"
        "Input" : {     # 每条请求的输入长度
            "Method": "uniform",
            "Params": {"MinValue": $input_length, "MaxValue": $input_length}
        },
        "Output" : {    # 每条请求的输出长度
            "Method": "uniform",
            "Params": {"MinValue": $output_length, "MaxValue": $output_length}
        }
    },
}



EOF


###### GSM8K数据集测试 #####
  elif [ "$dataset" = "GSM" ]; then
    echo -e "\n################################################[run_benchmark]:新的一次测试开启#######################################################"
    echo "[run_benchmark]:数据集类型是GSM"
    mkdir -p ${gsm8k_dataset_path}
    Data_Command_Type="gsm8k_gen_0_shot_cot_str_perf"
    echo "[run_benchmark]:[Success] 已生成配置文件: $CONFIG_FILE"
    echo "[run_benchmark]:[Success] 数据集类型选择: $dataset"
    echo "[run_benchmark]:[Success] 数据集命令选择: $Data_Command_Type"
    # 生成GSM8K数据集文件
    CONFIG_FILE2="${gsm8k_dataset_path}/test.jsonl"
    train_file="${gsm8k_dataset_path}/train.jsonl"
    RequestCount=$(($Concurrent_Multiplier*$concurrency))
    dest_file_2=GSM8K-in${input_length}-bs${RequestCount}.jsonl
    rm $dest_file_2
    echo "[run_benchmark]:开始监控GSM8K文件生成"
    #检查train.jsonl的空白文件是否生成
    touch ${train_file}
    check_file_exists "$train_file" 30 1 "GSM8K-train数据集"
    #检查test.jsonl的空白文件是否生成
    rm ${CONFIG_FILE2}
    python3 process_dataset.py --bs=$RequestCount --inputlen=$input_length --datasettype=$dataset --datapath=$datapath --modelpath=$MODEL_PATH
    check_file_exists "$dest_file_2" 30 1 "GSM8K-test数据集"
    cp  ${dest_file_2} ${CONFIG_FILE2}


######### VQA #########
  elif [ "$dataset" = "VQA" ]; then
    echo -e "\n###############################################[run_benchmark]:新的一次测试开启#######################################################"
    echo "[run_benchmark]:数据集类型是VQA"
    mkdir -p ${textQVA_dataset_path}
    Data_Command_Type="textvqa_gen_base64"
    echo "[run_benchmark]:[Success] 已生成配置文件: $CONFIG_FILE"
    echo "[run_benchmark]:[Success] 数据集类型选择: $dataset"
    echo "[run_benchmark]:[Success] 数据集命令选择: $Data_Command_Type"
    # 生成TEXTVQA数据集文件: textvqa_val.jsonl   textvqa_val_annotations.json
    CONFIG_FILE3="${textQVA_dataset_path}/textvqa_val.jsonl"
    CONFIG_FILE4="${textQVA_dataset_path}/textvqa_val_annotations.json"
    RequestCount=$(($Concurrent_Multiplier*$concurrency))
    dest_file_3="Textvqa-in${input_length}-bs${RequestCount}.jsonl"
    dest_file_4="Textvqa-in${input_length}-bs${RequestCount}-annotation.json"
    rm $dest_file_3
    rm $dest_file_4
    echo "[run_benchmark]:开始监控TEXTVQA文件生成"    
    python3 process_dataset.py --bs=$RequestCount --inputlen=$input_length --datasettype=$dataset --datapath=$datapath --modelpath=$MODEL_PATH
    
    check_file_exists "$dest_file_3" 30 1 "Textvqa-val数据集"
    check_file_exists "$dest_file_4" 30 1 "Textvqa-va-annotation数据集"
    cp ${dest_file_3} ${CONFIG_FILE3}
    cp ${dest_file_4} ${CONFIG_FILE4}
  
  
  ######### VID #########
  elif [ "$dataset" = "VID" ]; then
    echo -e "\n###############################################[run_benchmark]:新的一次测试开启#######################################################"
    echo "[run_benchmark]:数据集类型是VID"
    mkdir -p ${videobench_path}
    Data_Command_Type="videobench_gen_base64"
    echo "[run_benchmark]:[Success] 已生成配置文件: $CONFIG_FILE"
    echo "[run_benchmark]:[Success] 数据集类型选择: $dataset"
    echo "[run_benchmark]:[Success] 数据集命令选择: $Data_Command_Type"
    # 生成Videobench数据集文件: Test_QA_new.json   ANSWER.json
    mkdir -p ${videobench_path}
    mkdir -p ${videobench_path}/answer
    mkdir -p ${videobench_path}/TEST
    CONFIG_FILE5="${videobench_path}/TEST_QA_new.json"
    CONFIG_FILE6="${videobench_path}/answer/ANSWER.json"
    RequestCount=$(($Concurrent_Multiplier*$concurrency))
    dest_file_5=Videobench-in${input_length}-bs${RequestCount}-qa.json
    dest_file_6=Videobench-in${input_length}-bs${RequestCount}-answer.json
    dest_file_7=${videobench_path}/TEST/test.mp4
    rm $dest_file_5
    rm $dest_file_6
    echo "[run_benchmark]:开始监控Videobench文件生成"
    python3 process_dataset.py --bs=$RequestCount --inputlen=$input_length --datasettype=$dataset --datapath=$dest_file_7 --modelpath=$MODEL_PATH
    
    check_file_exists "$dest_file_5" 30 1 "Videobench-QA数据集"
    check_file_exists "$dest_file_6" 30 1 "Videobench-answer数据集" 
    cp ${dest_file_5} ${CONFIG_FILE5}
    cp ${dest_file_6} ${CONFIG_FILE6}
    cp ${datapath} ${dest_file_7}

  ######### VSD #########
  elif [ "$dataset" = "VSD" ]; then
    echo -e "\n###############################################[run_benchmark]:新的一次测试开启#######################################################"
    echo "[run_benchmark]:数据集类型是VSD"
    RequestCount=$(($Concurrent_Multiplier*$concurrency))
    mkdir -p ${vocalsound_path}
    Data_Command_Type="vocalsound_gen_base64"
    echo "[run_benchmark]:[Success] 已生成配置文件: $CONFIG_FILE"
    echo "[run_benchmark]:[Success] 数据集类型选择: $dataset"
    echo "[run_benchmark]:[Success] 数据集命令选择: $Data_Command_Type"
    rm -rf "${vocalsound_path}/*"
    filename="$datapath"
    wav_number="${filename%.wav}"
    echo "提取的数字: $wav_number"
    # 生成Vocalsound数据集文件: ./dataset/Vocalsound/{length}second/{number}文件夹
    echo "[run_benchmark]:开始监控Vocalsound文件生成"
    python3 process_dataset.py --bs=$RequestCount --inputlen=$input_length --datasettype=$dataset  --datapath=$wav_number --modelpath=$MODEL_PATH 

    dest_file_8="./dataset/Vocalsound/${wav_number}second/${RequestCount}/"
    check_file_exists "${dest_file_8}" 30 1 "Vocalsound数据集"
    
    cp ${dest_file_8}* "${vocalsound_path}/"
    


  ######### 非可支撑的数据集类型-Error #########
  else
    echo "[run_benchmark]: [ERROR] 本次测试的数据集类型不支持：$dataset"
    echo "[run_benchmark]: 请检查当前支持的数据集 1、合成数据集: SYN ; 2、GSM8K数据集: GSM; 3、Textvqa图片数据集: VQA; 4、Videobench数据集: VID"
    echo "[run_benchmark]: [ERROR] 跳过本次测试，参数如下：${input_length}_${output_length}_${RequestCount}_${concurrency}_${MODEL_NAME}_${dataset}_${request_rate}"
    continue

  fi

  # 执行基准测试
  echo "[run_benchmark]: 正在执行测试: 请求频率=$request_rate, 数据集类型=$dataset, 并发数=$concurrency, 输入长度=$input_length, 输出长度=$output_length, 多模态文件=$Multimodal_test_file"
  
  Tested_Results_LOG="${LOG_SAVE_PATH}/results_${input_length}_${output_length}_${RequestCount}_${concurrency}_${MODEL_NAME}_${dataset}_${Multimodal_test_file}_${request_rate}.log"
 
  # 设置日志环境变量
  #export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1"
  # 执行ais_benchmark命令
  ais_bench --models vllm_api_stream_chat --datasets $Data_Command_Type --debug --summarizer default_perf --mode perf > $Tested_Results_LOG 2>&1
  
  # 检查执行状态
  if [ $? -eq 0 ]; then
    echo "[run_benchmark]:[Success] 测试完成: $concurrency并发 | 输入${input_length} | 输出${output_length}"
    echo "[run_benchmark]:=== 测试完成，显示测试结果截图 ==="
    tail -n 47 $Tested_Results_LOG | head -n 46
    echo "[run_benchmark]:正在收集结果数据至summary.csv" 
    python3 ./log_parsing.py -i ${Tested_Results_LOG} -o ${CSV_SAVE_PATH} -a ${ALL_NUM} -d ${D_NUM} -f "${input_length};${output_length};${RequestCount};${concurrency};${MODEL_NAME};${dataset};${Multimodal_test_file};${request_rate}"
    echo "[run_benchmark]:本次测试结果已收集完毕, 存放至路径: $CSV_SAVE_PATH/summary.csv "
  else
    echo "[run_benchmark]:[ERROR] 测试失败: $concurrency并发 | 输入${input_length} | 输出${output_length}"
    echo "[run_benchmark]:自动化脚本关闭, 请检查ERROR日志"
    exit 1
  fi
  
  echo "[run_benchmark]:测试日志保存在：${Tested_Results_LOG}"

  #性能采集并且生成分析文件
  if [ $PROFILL_ENABLE = 1 ]; then
    mkdir -p ${LOG_SAVE_PATH}/${dataset}_results_${input_length}_${output_length}_${concurrency}_${MODEL_NAME}_${request_rate}
    python3 -m ms_service_profiler.parse --input-path ${PROFILL_DATA_PATH} --output-path ${LOG_SAVE_PATH}/${dataset}_results_${input_length}_${output_length}_${concurrency}_${MODEL_NAME}_${request_rate}/
    if [ $? -eq 0 ]; then
      echo "[run_benchmark]: 性能profiling采集数据存放位置为${LOG_SAVE_PATH}/${dataset}_results_${input_length}_${output_length}_${concurrency}_${MODEL_NAME}_${request_rate}"
    else
      echo "[run_benchmark]: [ERROR] profiling文件性能解析异常, 请检查是否正常启动profiling服务"
    fi
  fi


done  

echo "==================================================================="
echo "[run_benchmark]:所有测试任务执行完毕!"
echo "[run_benchmark]:测试结果保存在: $LOG_SAVE_PATH"

