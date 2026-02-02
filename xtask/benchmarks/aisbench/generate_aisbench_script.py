import os
import json
import logging
from typing import Dict

from xtask.utils.container_manager import path_mapping_in_container
from xtask.datasets.dataset_downloader import download_dataset
from xtask.datasets.dataset_manger import preprocess_dataset


def get_api_type_file(api_type, shell_content):
    api_list = ["vllm_api_general_chat", "vllm_api_general_stream", "vllm_api_stream_chat"]
    if api_type not in api_list:
        logging.error(f"api_type: {api_type} is invalid")
    shell_content.append("\n# 2. 定位vllm_api配置文件")
    shell_content.append(f'CONFIG_FILE="$AIS_BENCH_PATH/benchmark/configs/models/vllm_api/{api_type}.py"')
    shell_content.append('if [ ! -f "$CONFIG_FILE" ]; then')
    shell_content.append('    echo "错误：配置文件不存在 $CONFIG_FILE"')
    shell_content.append('    exit 1')
    shell_content.append('fi')
    shell_content.append('echo "目标配置文件：$CONFIG_FILE"')

    return "$CONFIG_FILE", shell_content


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
    output_file_path = os.path.join(os.getcwd(), "outputs", f"{task_config.get('case_name').lower()}")
    output_file_path_incontainer = path_mapping_in_container(output_file_path)
    output_shell_path = os.path.join(output_file_path, output_shell_file)
    output_shell_path_incontainer = path_mapping_in_container(output_shell_path)
    # ========== 1. 解析task_config，设置默认值 ==========
    benchmark_dict = task_config.get("benchmark", None)
    if benchmark_dict:
        accuracy_cfg = benchmark_dict.get("acc", None)
        perf_cfg = benchmark_dict.get("perf", None)

    # aisbench配置默认值
    api_address = task_config.get("host_addr", "127.0.0.1")
    api_port = task_config.get("port", 14830)
    model_name = task_config.get("served_model_name", "model")
    model_path = task_config.get("model_path", "")
    dataset_name = task_config.get("dataset_name").lower()
    dataset_path = perf_cfg.get("dataset_path", "")
    output_len = perf_cfg.get("output_len", 1024)
    request_rate = perf_cfg.get("request_rate", 0)
    batch_size = perf_cfg.get("batch_size", 32)
    temperature = perf_cfg.get("temperature", 0)
    top_k = perf_cfg.get("top_k", -1)
    top_p = perf_cfg.get("top_p", 0.95)
    seed = perf_cfg.get("seed", 1024)
    num_warmups = perf_cfg.get("num_warmups", 0)
    repetition_penalty = perf_cfg.get("repetition_penalty", 1)
    model_path_inside = path_mapping_in_container(model_path)
    user_id = os.getuid()

    # ========== 2. 构造shell脚本内容 ==========
    shell_content = []
    # 基础配置：shebang、出错退出、日志输出
    shell_content.append("#!/bin/bash")
    shell_content.append('echo "===== 开始执行ais-bench测试脚本 ====="')

    # Step 1: 获取ais_bench的site-packages路径
    shell_content.append("\n# 1. 获取ais_bench安装路径")
    shell_content.append(f'export AIS_BENCH_PATH=$({python_exec} -c "import os;import ais_bench; print(os.path.dirname(ais_bench.__file__))" 2>/dev/null)')
    shell_content.append('if [ -z "$AIS_BENCH_PATH" ]; then')
    shell_content.append('    echo "错误：未找到ais_bench的安装路径，请确认已安装ais-bench"')
    shell_content.append('    exit 1')
    shell_content.append('fi')
    shell_content.append('echo "ais_bench安装路径：$AIS_BENCH_PATH"')

    download_cmd = download_dataset(task_config)
    if download_cmd:
        shell_content.extend(download_cmd)

    # Step 2: 生成精度测试命令, 如果--metrics配置了acc, 则读取benchmark xxx_acc_config.json配置我呢见中的参数
    if "acc" in task_config.get("metrics") and accuracy_cfg.get("dataset_conf"):
        api_type = accuracy_cfg.get("api_type", "vllm_api_general_chat")
        api_type_file, shell_content = get_api_type_file(api_type, shell_content)

        # Step 3: 修改配置文件中的核心参数（使用sed命令）
        shell_content.append("\n# 3. 修改vllm_api配置文件$CONFIG_FILE参数")
        # 替换API地址
        shell_content.append(f'sed -i.bak "s/host_ip\s*=\s*.*,/host_ip=\'{api_address}\',/g" {api_type_file}')
        # 替换API端口
        shell_content.append(f'sed -i "s/host_port\s*=\s*.*,/host_port={api_port},/g" {api_type_file}')
        # 替换模型名称
        shell_content.append(f'sed -i "s/model\s*=\s*.*,/model=\'{model_name}\',/g" {api_type_file}')
        # 替换模型路径
        shell_content.append(f'sed -i "s#path\s*=\s*\\"\\",#path=\'{model_path_inside}\',#g" {api_type_file}')

        shell_content.append(f'sed -i "s/temperature\s*=\s*.*,/temperature={temperature},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/request_rate\s*=\s*.*,/request_rate={request_rate},/g" {api_type_file}')

        shell_content.append(f'sed -i "s/top_k\s*=\s*.*,/top_k={top_k},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/top_p\s*=\s*.*,/top_p={top_p},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/seed\s*=\s*.*,/seed={seed},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/repetition_penalty\s*=\s*.*,/repetition_penalty={repetition_penalty},/g" {api_type_file}')

        shell_content.append("\n# 4. 执行精度测试")
        # metrics_str = ",".join(accuracy_metrics)
        dataset_conf = accuracy_cfg.get("dataset_conf")
        # accuracy_metrics = accuracy_cfg.get("metrics", ["bleu", "rouge"])
        accuracy_batch_size = accuracy_cfg.get("batch_size", batch_size)
        accuracy_max_out_len = accuracy_cfg.get("output_len", output_len)
        acc_concurrency = accuracy_cfg.get("concurrency", 0)
        dump_eval_details = task_config.get("dump_eval_details", "")

        # 替换最大生成token数
        shell_content.append(f'sed -i "s/max_out_len\s*=\s*.*,/max_out_len={accuracy_max_out_len},/g" {api_type_file}')
        # 替换batchsize
        shell_content.append(f'sed -i "s/batch_size\s*=\s*.*,/batch_size={accuracy_batch_size},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/request_rate\s*=\s*.*,/request_rate={acc_concurrency},/g" {api_type_file}')
        if dump_eval_details:
            dump_eval_details_str = "--dump-eval-details "
        else:
            dump_eval_details_str = ""

        # 构造ais-bench精度测试命令（参考ais-bench官方文档）
        accuracy_cmd = (
            f'ais_bench --models {api_type} '
            f'--datasets {dataset_conf} '
            f'{dump_eval_details_str} '
            f'--work-dir {output_file_path_incontainer}'
        )
        shell_content.append(f'echo "执行精度测试命令：{accuracy_cmd}"')
        shell_content.append(accuracy_cmd)
        shell_content.append(f'echo "精度测试完成，结果保存至 {output_file_path_incontainer}"')

        shell_content.append(f'cp {api_type_file} {output_file_path_incontainer}/ ')
        shell_content.append(f'echo "配置文件修改完成（已生成备份文件 ${api_type_file}）"')
    else:
        shell_content.append("\n# 4. 执行精度测试，skip task...")

    # Step 4: 生成性能测试命令（若开启）
    if "perf" in task_config.get("metrics"):
        api_type = perf_cfg.get("api_type", "vllm_api_general_stream")
        api_type_file, shell_content = get_api_type_file(api_type, shell_content)

        # Step 2: 修改配置文件中的核心参数（使用sed命令）
        shell_content.append("\n# 3. 修改vllm_api配置文件$CONFIG_FILE参数")
        # 替换API地址
        shell_content.append(f'sed -i.bak "s/host_ip\s*=\s*.*,/host_ip=\'{api_address}\',/g" {api_type_file}')
        # 替换API端口
        shell_content.append(f'sed -i "s/host_port\s*=\s*.*,/host_port={api_port},/g" {api_type_file}')
        # 替换模型名称
        shell_content.append(f'sed -i "s/model\s*=\s*.*,/model=\'{model_name}\',/g" {api_type_file}')
        # 替换模型路径
        shell_content.append(f'sed -i "s#path\s*=\s*\\"\\",#path=\'{model_path_inside}\',#g" {api_type_file}')

        shell_content.append("\n# 5. 执行性能测试")
        perf_concurrency = perf_cfg.get("concurrency", 10)
        perf_total_requests = perf_cfg.get("total_requests", 1000)
        perf_timeout = perf_cfg.get("timeout", 600)
        perf_warmup = perf_cfg.get("warmup_requests", 100)
        perf_batch_size = perf_cfg.get("batch_size", batch_size)
        perf_max_out_len = perf_cfg.get("output_len", output_len)
        
        mode = perf_cfg.get("mode", "perf") # 精度: all/infer/eval/viz  性能: perf/perf_viz
        summarizer = perf_cfg.get("summarizer", "default_perf")
        perf_temperature = perf_cfg.get("temperature", 0)
        perf_top_k = perf_cfg.get("top_k", -1)
        perf_top_p = perf_cfg.get("top_p", 10)
        perf_seed = perf_cfg.get("seed", 1024)

        process_dataset_cmd, dataset_conf_type = preprocess_dataset(dataset_name, task_config, output_file_path)
        shell_content.append(f'sed -i "s/request_rate\s*=\s*.*,/request_rate={perf_concurrency},/g" {api_type_file}')
        # 替换最大生成token数
        shell_content.append(f'sed -i "s/max_out_len\s*=\s*.*,/max_out_len={perf_max_out_len},/g" {api_type_file}')
        # 替换batchsize
        shell_content.append(f'sed -i "s/batch_size\s*=\s*.*,/batch_size={perf_batch_size},/g" {api_type_file}')

        shell_content.append(f'sed -i "s/temperature\s*=\s*.*,/temperature={perf_temperature},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/request_rate\s*=\s*.*,/request_rate={request_rate},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/top_k\s*=\s*.*,/top_k={perf_top_k},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/top_p\s*=\s*.*,/top_p={perf_top_p},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/seed\s*=\s*.*,/seed={perf_seed},/g" {api_type_file}')
        shell_content.append(f'sed -i "s/repetition_penalty\s*=\s*.*,/repetition_penalty={repetition_penalty},/g" {api_type_file}')

        if process_dataset_cmd:
            shell_content.extend(process_dataset_cmd)
        else:
            logging.error("process_dataset_cmd is null.")
        
        # 构造ais-bench性能测试命令（参考ais-bench官方文档）
        perf_cmd = (
            f'ais_bench --models {api_type} '
            f'--datasets {dataset_conf_type} '
            f'--num-warmups {num_warmups} '
            f'--mode {mode} --debug '
            f'--summarizer {summarizer} '
            f'--work-dir {output_file_path_incontainer}'
        )
        shell_content.append(f'echo "执行性能测试命令：{perf_cmd}"')
        shell_content.append(perf_cmd)
        shell_content.append(f'echo "性能测试完成，结果保存至 {output_file_path_incontainer}"')

        shell_content.append(f'cp {api_type_file} {output_file_path_incontainer}/ ')
        shell_content.append(f'echo "配置文件修改完成（已生成备份文件 ${api_type_file}）"')
    else:
        shell_content.append("\n# 5. 执行性能测试，skip task...")
    
    shell_content.append(f'chown -R {user_id}:{user_id} {output_file_path_incontainer}/* ')

    shell_content.append(f'cp $CHAT_CONFIG_FILE {output_file_path_incontainer}/ ')
    shell_content.append('echo "配置文件修改完成（已生成备份文件 ${CHAT_CONFIG_FILE}.bak）"')

    # 收尾：提示脚本完成
    shell_content.append("\necho \"===== ais-bench测试脚本执行完成 =====\"")

    # ========== 3. 写入shell文件并添加执行权限 ==========
    with open(output_shell_path, "w", encoding="utf-8") as f:
        f.write("\n".join(shell_content))
    
    # 添加可执行权限
    os.chmod(output_shell_path, 0o755)
    logging.info(f"Shell脚本已生成：{output_shell_path_incontainer}")
    logging.info(f"脚本执行方式：bash {output_shell_path_incontainer}")
    benchmark_cmd = f"bash {output_shell_path_incontainer}"
    # accuracy_cmd = ""
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