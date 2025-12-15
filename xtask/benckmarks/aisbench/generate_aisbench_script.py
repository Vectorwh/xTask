import os
import json
import logging
from typing import Dict, Optional, List

def generate_aisbench_script(
    task_config: Dict,
    output_shell_file: str = "run_aisbench_test.sh",
    python_exec: str = "python3"  # 指定python解释器路径，适配不同环境
) -> None:
    """
    生成ais-bench测试的shell脚本，包含配置修改、精度/性能测试命令生成
    
    Args:
        task_config: 测试任务配置字典，示例结构见下方
        output_shell_file: 生成的shell脚本输出脚本名称
        python_exec: Python解释器路径（如python/python3/anaconda3/bin/python）
    """
    output_file_path = os.path.join("./outputs", f"{task_config.get('case_name').lower()}")
    output_shell_path = os.path.join(output_file_path, output_shell_file)
    # ========== 1. 解析task_config，设置默认值 ==========
    benchmark_dict = task_config.get("benchmark", None)
    if benchmark_dict:
        accuracy_cfg = benchmark_dict.get("acc", None)
        perf_cfg = benchmark_dict.get("perf", None)

    # aisbench配置默认值
    api_address = task_config.get("node_addr", "127.0.0.1")
    api_port = task_config.get("node_port", 14830)
    model_name = task_config.get("served_model_name", "model")
    model_path = task_config.get("model_path", "/path/to/model")
    dataset_path = perf_cfg.get("dataset_path", "/path/to/dataset")
    max_new_tokens = perf_cfg.get("max_out_len", 512)
    batch_size = perf_cfg.get("batch_size", 1)

    # ========== 2. 构造shell脚本内容 ==========
    shell_content = []
    # 基础配置：shebang、出错退出、日志输出
    shell_content.append("#!/bin/bash")
    shell_content.append("set -euo pipefail")  # 增强健壮性：出错立即退出、未定义变量报错、管道失败报错
    shell_content.append('echo "===== 开始执行ais-bench测试脚本 ====="')

    # Step 1: 获取ais_bench的site-packages路径
    shell_content.append("\n# 1. 获取ais_bench安装路径")
    shell_content.append(f'export AIS_BENCH_PATH=$({python_exec} -c "import os;import ais_bench; print(os.path.dirname(ais_bench.__file__))" 2>/dev/null)')
    shell_content.append('if [ -z "$AIS_BENCH_PATH" ]; then')
    shell_content.append('    echo "错误：未找到ais_bench的安装路径，请确认已安装ais-bench"')
    shell_content.append('    exit 1')
    shell_content.append('fi')
    shell_content.append('echo "ais_bench安装路径：$AIS_BENCH_PATH"')

    # Step 2: 定位vllm_api_general_chat.py配置文件
    shell_content.append("\n# 2. 定位vllm_api配置文件")
    shell_content.append('CONFIG_FILE="$AIS_BENCH_PATH/benchmark/configs/models/vllm_api/vllm_api_general_chat.py"')
    shell_content.append('if [ ! -f "$CONFIG_FILE" ]; then')
    shell_content.append('    echo "错误：配置文件不存在 $CONFIG_FILE"')
    shell_content.append('    exit 1')
    shell_content.append('fi')
    shell_content.append('echo "目标配置文件：$CONFIG_FILE"')

    # Step 3: 修改配置文件中的核心参数（使用sed命令）
    shell_content.append("\n# 3. 修改vllm_api配置文件参数")
    # 替换API地址
    shell_content.append(f'sed -i.bak "s/^api_address = .*/api_address = \'{api_address}\'/g" "$CONFIG_FILE"')
    # 替换API端口
    shell_content.append(f'sed -i "s/^api_port = .*/api_port = {api_port}/g" "$CONFIG_FILE"')
    # 替换模型名称
    shell_content.append(f'sed -i "s/^model_name = .*/model_name = \'{model_name}\'/g" "$CONFIG_FILE"')
    # 替换模型路径
    shell_content.append(f'sed -i "s/^model_path = .*/model_path = \'{model_path}\'/g" "$CONFIG_FILE"')
    # 替换数据集路径
    shell_content.append(f'sed -i "s/^dataset_path = .*/dataset_path = \'{dataset_path}\'/g" "$CONFIG_FILE"')
    # 替换最大生成token数
    shell_content.append(f'sed -i "s/^max_new_tokens = .*/max_new_tokens = {max_new_tokens}/g" "$CONFIG_FILE"')
    # 替换批大小
    shell_content.append(f'sed -i "s/^batch_size = .*/batch_size = {batch_size}/g" "$CONFIG_FILE"')
    shell_content.append('echo "配置文件修改完成（已生成备份文件 ${CONFIG_FILE}.bak）"')

    # Step 4: 生成精度测试命令（若开启）
    if accuracy_cfg.get("enable", False):
        shell_content.append("\n# 4. 执行精度测试")
        accuracy_metrics = accuracy_cfg.get("metrics", ["bleu", "rouge"])
        metrics_str = ",".join(accuracy_metrics)
        accuracy_batch_size = accuracy_cfg.get("batch_size", batch_size)
        accuracy_max_new_tokens = accuracy_cfg.get("max_new_tokens", max_new_tokens)
        
        # 构造ais-bench精度测试命令（参考ais-bench官方文档）
        accuracy_cmd = (
            f'ais_bench run --config "$CONFIG_FILE" '
            f'--task accuracy '
            f'--batch_size {accuracy_batch_size} '
            f'--max_new_tokens {accuracy_max_new_tokens} '
            f'--metrics {metrics_str} '
            f'--output ./accuracy_test_result.json'
        )
        shell_content.append(f'echo "执行精度测试命令：{accuracy_cmd}"')
        shell_content.append(accuracy_cmd)
        shell_content.append('echo "精度测试完成，结果保存至 ./accuracy_test_result.json"')

    # Step 5: 生成性能测试命令（若开启）
    if perf_cfg.get("enable", False):
        shell_content.append("\n# 5. 执行性能测试")
        perf_concurrency = perf_cfg.get("concurrency", 10)
        perf_total_requests = perf_cfg.get("total_requests", 1000)
        perf_timeout = perf_cfg.get("timeout", 300)
        perf_warmup = perf_cfg.get("warmup_requests", 100)
        
        # 构造ais-bench性能测试命令（参考ais-bench官方文档）
        perf_cmd = (
            f'ais_bench run --config "$CONFIG_FILE" '
            f'--task performance '
            f'--concurrency {perf_concurrency} '
            f'--total_requests {perf_total_requests} '
            f'--warmup_requests {perf_warmup} '
            f'--timeout {perf_timeout} '
            f'--output ./perf_test_result.json'
        )
        shell_content.append(f'echo "执行性能测试命令：{perf_cmd}"')
        shell_content.append(perf_cmd)
        shell_content.append('echo "性能测试完成，结果保存至 ./perf_test_result.json"')

    # 收尾：提示脚本完成
    shell_content.append("\necho \"===== ais-bench测试脚本执行完成 =====")

    # ========== 3. 写入shell文件并添加执行权限 ==========
    with open(output_shell_path, "w", encoding="utf-8") as f:
        f.write("\n".join(shell_content))
    
    # 添加可执行权限
    os.chmod(output_shell_path, 0o755)
    logging.info(f"Shell脚本已生成：{os.path.abspath(output_shell_path)}")
    logging.info(f"脚本执行方式：bash {output_shell_path}")
    benchmark_cmd = f"bash {output_shell_path}"
    return benchmark_cmd

# ========== 示例调用 ==========
if __name__ == "__main__":
    # 示例task_config配置（可根据实际需求修改）
    sample_task_config = {
        "vllm_config": {
            "api_address": "127.0.0.1",
            "api_port": 8000,
            "model_name": "DeepSeek-V3",
            "model_path": "/data/models/DeepSeek-V3",
            "dataset_path": "/data/datasets/chat_eval_dataset",
            "max_new_tokens": 1024,
            "batch_size": 4
        },
        "accuracy_test": {
            "enable": True,
            "batch_size": 2,
            "max_new_tokens": 1024,
            "metrics": ["bleu", "rouge", "bertscore"]
        },
        "perf_test": {
            "enable": True,
            "concurrency": 20,
            "total_requests": 2000,
            "warmup_requests": 200,
            "timeout": 600
        }
    }

    # 生成shell脚本
    generate_aisbench_script(
        task_config=sample_task_config,
        output_shell_file="run_deepseek_aisbench_test.sh",
        python_exec="python3"
    )