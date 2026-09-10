import os
import json
import logging
from typing import Dict

from xtask.utils.container_manager import path_mapping_in_container
from xtask.datasets.dataset_downloader import download_dataset
from xtask.datasets.dataset_manger import preprocess_dataset


def append_checked_command(shell_content, command, failure_message):
    shell_content.append(f'if ! {command}; then')
    shell_content.append(f'    echo "{failure_message}" >&2')
    shell_content.append('    exit 1')
    shell_content.append('fi')


def generate_evalscope_script(
    task_config: Dict,
    output_file_path: str,
    output_shell_file: str = "run_evalscope_test.sh",
    python_exec: str = "python3"  # 指定python解释器路径，适配不同环境
) -> str:
    """
    生成EvalScope基准测试脚本
    
    Args:
        task_config: 任务配置
        output_file_path: 输出文件路径
        output_shell_file: 输出的Shell文件名
        python_exec: Python解释器路径
        
    Returns:
        基准测试脚本内容
    """

    output_file_path_incontainer = path_mapping_in_container(output_file_path)
    output_shell_path = os.path.join(output_file_path, output_shell_file)
    output_shell_path_incontainer = path_mapping_in_container(output_shell_path)

    benchmark_dict = task_config.get("benchmark", None)
    if benchmark_dict:
        accuracy_cfg = benchmark_dict.get("acc", None)
        perf_cfg = benchmark_dict.get("perf", None)

    # ========== 1. 获取EvalScope 基础配置 ==========
    api_address = task_config.get("host_addr", None)
    api_port = task_config.get("port", 14830)
    model_name = task_config.get("served_model_name", "model")
    dataset_name = task_config.get("dataset_name").lower()
    user_id = os.getuid()
    api_base = f"http://{api_address}:{api_port}"
    api_key = "EMPTY"
    from xtask.core.external_service import (
        is_external_service_enabled,
        resolve_external_location,
    )

    if is_external_service_enabled(task_config):
        location = resolve_external_location(task_config)
        api_base = location.origin + location.base_path
        if location.api_key:
            api_key = location.api_key
    
    # ========== 2. 构造shell脚本内容 ==========
    shell_content = []
    # 基础配置：shebang、出错退出、日志输出
    shell_content.append("#!/bin/bash")
    shell_content.append('echo "===== 开始执行evalscope测试脚本 ====="')

    # ========== 3. 构造精度测试脚本配置 ==========
    if "acc" in task_config.get("metrics"):        
        batch_size = accuracy_cfg.get("batch_size", 1)        
        # limit = accuracy_cfg.get("acc_dataset_len", 30)
        timeout = accuracy_cfg.get("timeout", 36000)
        endpoint = accuracy_cfg.get("endpoint", "/v1")
        generation_config = (f'{str(accuracy_cfg.get("generation_config", ""))}').replace("'", '"')
        dataset_args =  (f'{str(accuracy_cfg.get("dataset_args", ""))}').replace("'", '"')  

        accuracy_cmd = (
            f'evalscope eval --model {model_name} '
            f'--api-url {api_base}{endpoint} '
            f'--api-key {api_key} '
            f'--eval-type openai_api '
            f'--datasets {dataset_name} '
            f'--timeout {timeout} '
            f'--eval-batch-size {batch_size} '
            # f'--limit {limit} '
            f'--work-dir {output_file_path_incontainer} '
            f'--generation-config \'{generation_config}\' '
            f'--dataset-args \'{dataset_args}\''
        )

        limit = accuracy_cfg.get("acc_dataset_len")
        if limit is not None:
            accuracy_cmd = accuracy_cmd + f" --limit {limit}"
        
        shell_content.append(f'echo "执行精度测试命令：{accuracy_cmd}"')
        append_checked_command(
            shell_content, accuracy_cmd, "EvalScope精度测试执行失败"
        )
        shell_content.append(f'echo "精度测试完成，结果保存至 {output_file_path_incontainer}"')
    else:
        shell_content.append("\n# 精度测试，skip task...")
    
    # ========== 4. 构造性能测试脚本配置 ==========
    if "perf" in task_config.get("metrics"):
        request_rate = perf_cfg.get("request_rate", 0)
        number_requests = perf_cfg.get("num_prompts", 1)
        min_prompt_length = perf_cfg.get("min_prompt_length", 698)
        max_prompt_length = perf_cfg.get("max_prompt_length", 698)
        image_width = perf_cfg.get("image_width", 512)
        image_height = perf_cfg.get("image_height", 512)
        image_format = perf_cfg.get("image_format", "RGB")
        image_num = perf_cfg.get("image_num", 1)
        min_tokens = perf_cfg.get("output_len", 1024)
        max_tokens = perf_cfg.get("max_out_len", 1024)
        model_path = task_config.get("model_path", "")
        batch_size = perf_cfg.get("batch_size", 32)
        tokenizer_path = path_mapping_in_container(model_path)
        seed = perf_cfg.get("seed", 42)
        
        perf_cmd = (
            f'evalscope perf --model {model_name} '
            f'--url {api_base}/v1/chat/completions '
            f'--api openai '
            f'--dataset {dataset_name} '
            f'--rate {request_rate} '
            f'--number {number_requests} '
            f'--parallel {batch_size} '
            f'--min-prompt-length {min_prompt_length} '
            f'--max-prompt-length {max_prompt_length} '
            f'--image-width {image_width} '
            f'--image-height {image_height} '
            f'--image-format {image_format} '
            f'--image-num {image_num} '
            f'--min-tokens {min_tokens} '
            f'--max-tokens {max_tokens} '
            f'--tokenizer-path {tokenizer_path} '
            f'--seed {seed} '
            f'--outputs-dir {output_file_path_incontainer} '
            f'--extra-args \'{{"ignore_eos": true}}\''
        )

        shell_content.append(f'echo "执行性能测试命令：{perf_cmd}"')
        append_checked_command(
            shell_content, perf_cmd, "EvalScope性能测试执行失败"
        )
        shell_content.append(f'echo "性能测试完成，结果保存至 {output_file_path_incontainer}"')
    else:
        shell_content.append("\n# 性能测试，skip task...")

    shell_content.append(f'chown -R {user_id}:{user_id} {output_file_path_incontainer}/* ')
    # 收尾：提示脚本完成
    shell_content.append("\necho \"===== evalscope测试脚本执行完成 =====\"")

    # ========== 5. 写入shell文件并添加执行权限 ==========
    with open(output_shell_path, "w", encoding="utf-8") as f:
        f.write("\n".join(shell_content))
    
    # 添加可执行权限
    os.chmod(output_shell_path, 0o755)
    logging.info(f"Shell脚本已生成：{output_shell_path_incontainer}")
    logging.info(f"脚本执行方式：bash {output_shell_path_incontainer}")
    benchmark_cmd = f"bash {output_shell_path_incontainer}"
    # accuracy_cmd = ""
    return benchmark_cmd
