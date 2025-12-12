# xTask 测试框架
![XTask](https://img.shields.io/static/v1?label=dev&message=dev&color=red&style=flat&logo=Python) ![XTask](https://img.shields.io/static/v1?label=author&message=J&color=blue&style=flat&logo=gitee) ![XTask](https://img.shields.io/static/v1?label=email&message=jiji1206@163.com&color=orange&style=flat&logo=microsoft-outlook)


基于国产芯片推理框架任务流自动化测试框架
# <font face="Microsoft YaHei UI">


## 内容列表

- [参数说明](#一、参数说明)
- [工具说明](#二、工具说明)
- [目录结构说明](#三、目录结构说明)
- [执行样例](#四、执行样例)
- [配置说明](#五、配置说明)


## 参数说明

### Usage
'''
    usage: main.py [options] [case_name|feature_name]
    
    Formatter for Python code.
    
    positional arguments:
    files                 reads from stdin when no files are specified.
    
    optional arguments:
    help           show this help message and exit
    version        show program's version number and exit
    create         generate generalized cases based on the input case description
    design         generate the networking configuration information
    assign         assign yaml task to ctrlcpu,SSAPI 业务AW调用下发已生成的yml配置任务
    import         import ub_config.xlsx to ctrlcpu，基于确定组网一键导入系统默认配置：需要输入topo.json组网信息，ub_config所有配置信息
    install        install ubengine based on the input case description
    uninstall      uninstall ubengine 
    exec           run the input use case
    clean          clean the generated temporary files.
'''

## 工具说明


## 代码目录结构

本代码仓目录结构如下：

    xTask/
    ├── config/                         # 配置文件目录
    ├── templates/                      # 模板文件目录
    ├── tests/                          # 测试文件目录
    ├── tools/                          # 工具脚本目录
    ├── xtask/                          # 核心代码目录
    │   ├── benckmarks/                 # 基准测试相关
    │   ├── core/                       # 核心模块
    │   ├── datasets/                   # 数据集管理
    │   ├── distributed/                # 分布式相关
    │   ├── metrics/                    # 评估指标
    │   ├── models/                     # 模型定义
    │   ├── serv_fmk/                   # 服务框架
    │   ├── tasks/                      # 任务定义
    │   ├── utils/                      # 工具函数
    │   ├── workflows/                  # 工作流程
    │   ├── main.py                     # 主程序入口
    │   └── pipeline.py                 # 流水线处理
    ├── launch.py                       # 启动脚本
    ├── README.md                       # 项目说明文档
    └── requirement.txt                 # Python依赖库列表
## 执行样例

###  xLLM Qwen3-4b model_name&model_path
    python main.py --model_name QWEN3-4B --model_path qwen3/QWEN3-4B 

### xLLM Qwen3-4b case_name
    python main.py --case_name qwen3_4b_xllm_001

### xLLM Qwen3-4b template
    python main.py --template {template_path}