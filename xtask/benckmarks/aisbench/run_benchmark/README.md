# Run_Benchmark工具
## 小巧灵突击队-0day适配项目组

# 一、最新版本进展

<span style="color:#e60000;">[2025.11.19] </span> 支持数据集：合成数据集、GSM8K数据集与多模态数据集：图片、视频。音频数据集的自动化测试；测试结果日志自动解析，13项性能指标通过csv文件存储，可在PC上一键式提取；提供多模态用例文件快速启动测试。


# 二、简介

**Run_Benchmark**，是基于Aisbench的自动化测试工具。结合Aisbench工具特性，从API命令配置文件、任意Question/Prompt输入长度任意请求数的数据集生成、多种数据集随意切换、性能测试结果数据解析收集等维度，赋能自动化测试。通过服务化参数、测试用例、PD卡数等参数填写，快速拉起自动化测试服务。

功能为：自动化动态配置任务、支持多种数据集生成、 自动化解析与收集性能测试结果。

```
run_benchmark_V1_4 工具树结构
├── dataset
│   ├── Textvqa
│   │   ├── 1080.jpg
│   │   ├── 1440.jpg
│   │   └── 9376.jpg
│   ├── Videobench
│   │   ├── target_1920_1080.mp4
│   │   ├── target_320_240.mp4
│   │   ├── target_736_448.mp4
│   └── Vocalsound
│       └── m15_0_throatclearing.wav
├── GSM8K.jsonl
├── log_parsing.py
├── process_dataset.py
└── run_benchmark.sh

```

# 三、使用指导

核心文件为：run_benchmark.sh。用户一般配置并运行该文件即可。

## 配置信息获取：
1、用户首先需要得到如下信息：在线服务的端口、服务名、权重路径、服务IP、PD卡数、版本信息、测试用例。

2、run_benchmark.sh脚本中服务参数设置

```sh
#从命令行参数获取配置
VERSION=""
Sever_NAME=""                                #服务化时使用的model-name
SERVICE_IP=""                                #一般为本地服务器的IP
SERVICE_PORT=""                              #拉起服务时设置的端口
MODEL_PATH=""                                #权重路径
#服务的D卡数和总卡数，用于计算性能测试结果中单卡的性能
D_NUM=1                                      #PD混部：D_NUM=P_NUM=ALL_NUM
ALL_NUM=1
```

3、数据集选择 && 输入参数 && 多模态文件路径   按照一行一次测试的要求，每个参数项通过空格区分。**检查数据集类型是否在范围内**
```sh
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
```
4、通过pip show ais-bench-benchmark查找；如果有更改，查看Editable project location。该路径为基准路径：
```sh
# 创建基准测试根目录，更改路径为容器中aisbench工具的路径
AISBENCHMARK_PATH="/workspace/benchmark"
```
5、CSV文件路径、测试结果日志路径可自定义修改，默认为Aisbench工具路径。
```sh
#创建的log日志存放地址，输出结果在日志文件里，根据时间区分单次执行，再根据输入配置区分不同的运行结果;
#用户可更改保存路径，推荐更改“${AISBENCHMARK_PATH}/LOG_PLACE”
LOG_SAVE_PATH="${AISBENCHMARK_PATH}/LOG_PLACE/LOG_${VERSION}_${SERVICE_IP}_${CURRENT_DATE}"
CSV_SAVE_PATH="${AISBENCHMARK_PATH}"
```
6、数据集指令请根据推理框架确认，本工具默认为vllm-ascend框架指令：
```sh
# 推理框架Mindie (1)SYN →→ synthetic_gen (2)GSM →→ gsm8k_gen_0_shot_cot_str_perf (3)VQA →→ Textvqa_gen (4)VID →→ Videobench_gen (5)VSD →→ vocalsound_gen
# 推理框架Vllm   (1)SYN →→ synthetic_gen (2)GSM →→ gsm8k_gen_0_shot_cot_str_perf (3)VQA →→ Textvqa_gen_base64 (4)VID →→ Videobench_gen_base64 (5)VSD →→ vocalsound_gen_base64
```
若需要修改，可检索对应数据集缩略词（如VSD等），查找Data_Command_Type，更改为所需要运行的数据集任务即可。

7、运行指令参考如下
```sh
bash run_benchmark.sh    #直接运行，日志打屏
```

8、音频数据集运行时需要安装相关依赖
```sh
pip install pydub
apt-get update
apt-get install -y ffmpeg
```