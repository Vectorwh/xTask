"""生成 Qwen-Image-Edit（gedit）性能评测 shell 脚本。

生成的脚本调用仓库内 wrapper 入口
``xtask/plugins/evalscope/run_gedit_perf.py``（该 wrapper 先注册
``gedit_image_gen`` API 插件再运行 evalscope perf），服务地址解析与
``generate_evalscope_script`` 同源（host_addr/port + external_service）。

task_config["benchmark"]["perf"] 字段映射（docstring 字段映射表）：

    YAML benchmark.perf.*         | 生成的 CLI 参数                | 缺省
    ------------------------------+--------------------------------+-----------
    dataset_path                  | --dataset-path                 | <DEPENDENCE_PACKAGE_PATH>/benchmark/
                                   |  ais_bench/datasets/gedit_perf/gedit_perf.jsonl
    num_prompts                   | --number                       | 1
    batch_size (int)              | --parallel                     | 1
    concurrency_sweep (list[int]) | --parallel v1 v2 ...           | 与 batch_size 二选一
    request_rate                  | --rate                         | -1（不限速）
    warmup_num                    | --warmup-num                   | 0
    seed                          | --seed                         | 42
    timeout                       | --total-timeout                | 21600
    size                          | 合并进 --extra-args JSON       | 无
    num_inference_steps           | 合并进 --extra-args JSON       | 无
    guidance_scale                | 合并进 --extra-args JSON       | 无

说明：
    - size / num_inference_steps / guidance_scale 是请求体参数而非
      evalscope Arguments 字段，因此通过 ``--extra-args`` JSON 注入
      ``dataset_args`` 无法生效（line_by_line 原样透传数据行），仅作为
      wrapper 可见的附加配置传递；真正的每请求参数以数据集行为准
      （见 tools/convert_gedit_to_perf.py 生成约定）。
    - 服务地址：非 external_service 用
      ``http://{host_addr}:{port}/v1/image/generation``；external_service
      用 ``location.origin + location.base_path + /image/generation``。
    - ``--name`` 用 case_name，便于 outputs 子目录与后处理对账。
"""

import json
import logging
import os
from typing import Any, Dict

from xtask.utils.config_parser import get_xtask_config_data
from xtask.utils.container_manager import path_mapping_in_container


# gedit perf 数据集默认根：环境变量优先、config/xTask_config.json 兜底
# （与 xtask/benchmarks/vbench/defaults.py 同一模式）。
_GEDIT_DATASET_FALLBACK_ROOT = "/export/home/models/xllm_test_060"


def _dependence_package_path() -> str:
    """Return the configured shared resource root (host path)."""
    env_value = os.environ.get("DEPENDENCE_PACKAGE_PATH", "").strip()
    if env_value:
        return env_value
    configured = (get_xtask_config_data().get("ENV_CONFIG") or {}).get(
        "DEPENDENCE_PACKAGE_PATH", ""
    )
    text = str(configured or "").strip()
    return text or _GEDIT_DATASET_FALLBACK_ROOT


def default_gedit_perf_dataset_path() -> str:
    """默认 gedit perf 数据集（host 路径，未映射容器侧）。

    由 tools/convert_gedit_to_perf.py 生成的 gedit_perf.jsonl，随共享
    DEPENDENCE_PACKAGE_PATH 布局放置；YAML benchmark.perf.dataset_path
    显式指定时始终优先。
    """
    return (
        _dependence_package_path()
        + "/benchmark/ais_bench/datasets/gedit_perf/gedit_perf.jsonl"
    )


def append_checked_command(shell_content, command, failure_message):
    shell_content.append(f'if ! {command}; then')
    shell_content.append(f'    echo "{failure_message}" >&2')
    shell_content.append('    exit 1')
    shell_content.append('fi')


def _image_edit_endpoint(task_config: Dict[str, Any]) -> str:
    """解析服务端点，与 evalscope 生成器同源（含 external_service）。"""
    from xtask.core.external_service import (
        is_external_service_enabled,
        resolve_external_location,
    )

    if is_external_service_enabled(task_config):
        location = resolve_external_location(task_config)
        base = location.origin + location.base_path
        return base.rstrip('/') + '/image/generation'

    api_address = task_config.get("host_addr", "127.0.0.1")
    api_port = task_config.get("port", 14830)
    return f"http://{api_address}:{api_port}/v1/image/generation"


def _image_edit_encoder_url(task_config: Dict[str, Any]) -> str:
    """两段式协议的 Text Encoder 地址。

    优先级：YAML benchmark.perf.encoder_url > 环境变量 GEDIT_ENCODER_URL >
    缺省 http://127.0.0.1:18007。
    """
    benchmark_dict = task_config.get("benchmark") or {}
    perf_cfg = benchmark_dict.get("perf") or {}
    configured = perf_cfg.get("encoder_url") or os.environ.get("GEDIT_ENCODER_URL")
    return str(configured or "http://127.0.0.1:18007")


def generate_image_edit_script(
    task_config: Dict,
    output_file_path: str,
    output_shell_file: str = "run_image_edit_test.sh",
    python_exec: str = "python3"
) -> str:
    """
    生成 Qwen-Image-Edit 基准测试脚本

    Args:
        task_config: 任务配置（字段映射见模块 docstring）
        output_file_path: 输出文件路径
        output_shell_file: 输出的Shell文件名
        python_exec: Python解释器路径，适配不同环境

    Returns:
        基准测试脚本内容（bash 执行命令）
    """
    output_file_path_incontainer = path_mapping_in_container(output_file_path)
    output_shell_path = os.path.join(output_file_path, output_shell_file)
    output_shell_path_incontainer = path_mapping_in_container(output_shell_path)

    benchmark_dict = task_config.get("benchmark", None)
    perf_cfg = {}
    if benchmark_dict:
        perf_cfg = benchmark_dict.get("perf", None) or {}

    # ========== 1. 基础配置（与 evalscope 生成器同源） ==========
    model_name = task_config.get("served_model_name", "model")
    user_id = os.getuid()
    endpoint = _image_edit_endpoint(task_config)
    case_name = task_config.get("case_name") or task_config.get("served_model_name") or "image_edit"

    # ========== 2. perf 参数 ==========
    # dataset_path 缺省：gedit perf 数据集随 DEPENDENCE_PACKAGE_PATH 共享
    # 布局放置（host 路径），容器侧经 path_mapping_in_container 映射。
    dataset_path = perf_cfg.get("dataset_path") or task_config.get(
        "dataset_path", ""
    ) or default_gedit_perf_dataset_path()
    dataset_path_inside = path_mapping_in_container(str(dataset_path)) if dataset_path else ""
    if not dataset_path_inside:
        raise ValueError(
            "image_edit perf failed to resolve dataset_path "
            "(expected benchmark.perf.dataset_path or the default "
            "<DEPENDENCE_PACKAGE_PATH>/benchmark/ais_bench/datasets/"
            "gedit_perf/gedit_perf.jsonl)"
        )

    number_requests = perf_cfg.get("num_prompts", 1)
    batch_size = perf_cfg.get("batch_size", 1)
    concurrency_sweep = perf_cfg.get("concurrency_sweep", None)
    request_rate = perf_cfg.get("request_rate", -1)
    warmup_num = perf_cfg.get("warmup_num", 0)
    seed = perf_cfg.get("seed", 42)
    timeout = perf_cfg.get("timeout", 21600)

    if concurrency_sweep:
        parallel_values = " ".join(str(int(v)) for v in concurrency_sweep)
    else:
        parallel_values = str(int(batch_size))

    # 请求体级参数：经 --extra-args 透传（wrapper 端可见，便于排障）
    extra = {}
    for key in ("size", "num_inference_steps", "guidance_scale"):
        if perf_cfg.get(key) is not None:
            extra[key] = perf_cfg[key]
    extra_args_json = json.dumps(extra, ensure_ascii=False)

    # wrapper 脚本随仓库版本化，容器内路径做 volumes 映射
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    run_gedit_perf_inside = path_mapping_in_container(
        os.path.join(workspace_root, "xtask", "plugins", "evalscope", "run_gedit_perf.py")
    )

    # ========== 3. 构造 shell 脚本 ==========
    shell_content = []
    shell_content.append("#!/bin/bash")
    shell_content.append('echo "===== 开始执行image_edit测试脚本 ====="')

    if "perf" in (task_config.get("metrics") or []):
        shell_content.append(f'if [ ! -f {dataset_path_inside} ]; then')
        shell_content.append(f'    echo "错误：数据集文件不存在 {dataset_path_inside}" >&2')
        shell_content.append('    exit 1')
        shell_content.append('fi')

        # 两段式协议：export encoder 地址，wrapper/插件经环境变量读取
        shell_content.append(f'export GEDIT_ENCODER_URL={_image_edit_encoder_url(task_config)}')

        perf_cmd = (
            f'{python_exec} {run_gedit_perf_inside} '
            f'--url {endpoint} '
            f'--encoder-url {_image_edit_encoder_url(task_config)} '
            f'--model {model_name} '
            f'--dataset-path {dataset_path_inside} '
            f'--number {number_requests} '
            f'--parallel {parallel_values} '
            f'--rate {request_rate} '
            f'--warmup-num {warmup_num} '
            f'--seed {seed} '
            f'--total-timeout {timeout} '
            f'--extra-args \'{extra_args_json}\' '
            f'--outputs-dir {output_file_path_incontainer} '
            f'--name {case_name}'
        )

        shell_content.append(f'echo "执行性能测试命令：{perf_cmd}"')
        append_checked_command(
            shell_content, perf_cmd, "image_edit性能测试执行失败"
        )
        shell_content.append(f'echo "性能测试完成，结果保存至 {output_file_path_incontainer}"')
    else:
        shell_content.append("\n# 性能测试，skip task...")

    shell_content.append(f'chown -R {user_id}:{user_id} {output_file_path_incontainer}/* ')
    shell_content.append("\necho \"===== image_edit测试脚本执行完成 =====\"")

    # ========== 4. 写入 shell 文件并添加执行权限 ==========
    with open(output_shell_path, "w", encoding="utf-8") as f:
        f.write("\n".join(shell_content))

    os.chmod(output_shell_path, 0o755)
    logging.info(f"Shell脚本已生成：{output_shell_path}")
    logging.info(f"脚本执行方式：bash {output_shell_path}")
    benchmark_cmd = f"bash {output_shell_path_incontainer}"
    return benchmark_cmd
