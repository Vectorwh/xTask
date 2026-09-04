"""Generate the case-local VBench I2V driver/evaluation shell script."""

from __future__ import annotations

import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List

from xtask.utils.container_manager import path_mapping_in_container
from xtask.benchmarks.vbench.defaults import (
    DEFAULT_FULL_INFO,
    DEFAULT_IMAGE_ROOT,
    default_cache_dir,
    default_evaluation_script,
)


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _safe_name(value: str, field: str) -> str:
    value = str(value or "")
    if not value or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"VBench {field} must contain only letters, numbers, '.', '_' or '-': {value!r}"
        )
    return value


def _vbench_config(task_config: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = task_config.get("benchmark") or {}
    config = benchmark.get("vbench") or {}
    if not isinstance(config, dict):
        raise ValueError("benchmark.vbench must be a mapping")
    return config


def _append_checked(shell: List[str], command: str, message: str) -> None:
    shell.extend(
        [
            f"if ! {command}; then",
            f"    echo {_quote(message)} >&2",
            "    exit 1",
            "fi",
        ]
    )


def generate_vbench_script(
    task_config: Dict[str, Any],
    output_file_path: str,
    output_shell_file: str = "run_vbench_test.sh",
    python_exec: str = "python3",
) -> str:
    """Create a reproducible VBench I2V generation/evaluation command.

    The generated script runs in the VBench runner container.  It creates one
    root-level YYYYMMDD_HHMMSS directory per invocation so xTask's existing
    result discovery and archive logic remains unchanged.
    """

    config = _vbench_config(task_config)
    mode = str(config.get("mode", "generate_and_evaluate")).lower()
    valid_modes = {"generate", "evaluate", "generate_and_evaluate"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported VBench mode: {mode!r}; expected one of {sorted(valid_modes)}")

    full_info = config.get("full_info") or config.get("dataset_path") or DEFAULT_FULL_INFO
    image_root = config.get("image_root") or DEFAULT_IMAGE_ROOT
    if mode in {"generate", "generate_and_evaluate"} and not image_root:
        raise ValueError("benchmark.vbench.image_root is required for generation")
    if not full_info:
        raise ValueError("benchmark.vbench.full_info is required")

    model_name = task_config.get("served_model_name") or config.get("model")
    if not model_name:
        raise ValueError("served_model_name or benchmark.vbench.model is required")
    name = _safe_name(config.get("name") or task_config.get("case_name") or "vbench", "name")

    output_path = Path(output_file_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_shell_path = output_path / output_shell_file
    output_dir = path_mapping_in_container(str(output_path))
    full_info_inside = path_mapping_in_container(str(full_info))
    image_root_inside = path_mapping_in_container(str(image_root)) if image_root else ""
    driver_inside = path_mapping_in_container(
        str(Path(__file__).resolve().with_name("run_i2v.py"))
    )
    evaluation_script = config.get("evaluation_script") or config.get(
        "evaluate_script", ""
    ) or default_evaluation_script()
    evaluation_script_inside = path_mapping_in_container(str(evaluation_script))
    cache_dir = config.get("cache_dir") or default_cache_dir()
    cache_dir_inside = path_mapping_in_container(str(cache_dir))
    endpoint = config.get("endpoint")
    if not endpoint:
        endpoint = (
            f"http://{task_config.get('host_addr', '127.0.0.1')}"
            f":{task_config.get('port', 14830)}/v1/video/generation"
        )
    elif str(endpoint).startswith("/"):
        endpoint = f"http://{task_config.get('host_addr', '127.0.0.1')}:{task_config.get('port', 14830)}{endpoint}"

    generation = config.get("generation") or {}
    generation = generation if isinstance(generation, dict) else {}
    # Keep generation settings under vbench, while accepting the flat spelling
    # used by the report for a simple migration path.
    for key in (
        "size",
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "num_videos_per_prompt",
        "max_sequence_length",
        "num_frames",
        "fps",
        "seconds",
        "boundary_ratio",
        "flow_shift",
        "true_cfg_scale",
        "seed",
        "request_timeout",
        "limit",
    ):
        if key not in generation and key in config:
            generation[key] = config[key]

    shell: List[str] = [
        "#!/bin/bash",
        "set -u",
        'echo "===== 开始执行 VBench I2V 测试 ====="',
        # 评测维度模型（CLIP/DINO/RAFT 等）默认跑 CPU，但 torch_npu 必须能干净
        # import（transformers 的 is_torch_npu_available 会探测）；不 source CANN
        # 环境时 libhccl.so 不在库路径，torch_npu 处于半初始化坏状态。runner
        # 镜像自带 toolkit，这里 source 之；纯 CPU 环境无此目录则跳过。
        # 注意 set_env.sh 内部引用了未定义变量，与 set -u 组合会静默退出，
        # 所以 source 前后临时切换 -u；必须在当前 shell source（子 shell 里
        # export 的 LD_LIBRARY_PATH 不会传出来）。
        "set +u",
        'for _env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do',
        '    [ -f "$_env" ] && . "$_env" >/dev/null 2>&1 || true',
        'done',
        'unset _env',
        'set -u',
        f"OUTPUT_DIR={_quote(output_dir)}",
        'RUN_DIR="$OUTPUT_DIR/$(date +%Y%m%d_%H%M%S)"',
        'VIDEO_DIR="$RUN_DIR/videos"',
        'if ! mkdir -p "$VIDEO_DIR"; then echo "VBench 输出目录创建失败" >&2; exit 1; fi',
        f"export VBENCH_CACHE_DIR={_quote(cache_dir_inside)}",
        'if ! mkdir -p "$VBENCH_CACHE_DIR"; then echo "VBench 缓存目录创建失败" >&2; exit 1; fi',
    ]

    if mode in {"generate", "generate_and_evaluate"}:
        shell.extend(
            [
                f"if [ ! -f {_quote(full_info_inside)} ]; then",
                f"    echo {_quote('VBench full_info 不存在: ' + str(full_info_inside))} >&2",
                "    exit 1",
                "fi",
                f"if [ ! -d {_quote(image_root_inside)} ]; then",
                f"    echo {_quote('VBench image_root 不存在: ' + str(image_root_inside))} >&2",
                "    exit 1",
                "fi",
            ]
        )
        output_full_info = f'$RUN_DIR/{name}_full_info_videos.json'
        output_full_info_arg = '"' + output_full_info + '"'
        generation_parts = [
            _quote(python_exec),
            _quote(driver_inside),
            "--full-info",
            _quote(full_info_inside),
            "--image-root",
            _quote(image_root_inside),
            "--save-path",
            '"$VIDEO_DIR"',
            "--output-full-info",
            output_full_info_arg,
            "--endpoint",
            _quote(endpoint),
            "--model",
            _quote(model_name),
        ]
        option_map = {
            "size": "--size",
            "num_inference_steps": "--num-inference-steps",
            "guidance_scale": "--guidance-scale",
            "guidance_scale_2": "--guidance-scale-2",
            "num_videos_per_prompt": "--num-videos-per-prompt",
            "max_sequence_length": "--max-sequence-length",
            "num_frames": "--num-frames",
            "fps": "--fps",
            "seconds": "--seconds",
            "boundary_ratio": "--boundary-ratio",
            "flow_shift": "--flow-shift",
            "true_cfg_scale": "--true-cfg-scale",
            "seed": "--seed",
            "request_timeout": "--request-timeout",
            "limit": "--limit",
        }
        for key, option in option_map.items():
            value = generation.get(key)
            if value is not None and value != "":
                generation_parts.extend([option, _quote(value)])
        _append_checked(
            shell,
            " ".join(generation_parts),
            "VBench 视频生成失败",
        )
    else:
        output_full_info = config.get("video_full_info") or full_info_inside
        output_full_info = path_mapping_in_container(str(output_full_info))
        output_full_info_arg = _quote(output_full_info)
        shell.extend(
            [
                f"if [ ! -f {_quote(output_full_info)} ]; then",
                f"    echo {_quote('VBench 视频 manifest 不存在: ' + str(output_full_info))} >&2",
                "    exit 1",
                "fi",
            ]
        )

    if mode in {"evaluate", "generate_and_evaluate"}:
        if not evaluation_script:
            raise ValueError("benchmark.vbench.evaluation_script is required for evaluation")
        # The default evaluator is framework source; a missing file means the
        # environment is incomplete.  Fail fast instead of a confusing
        # "command not found" deep inside the runner.
        if os.path.isfile(str(evaluation_script)):
            shell.extend(
                [
                    f"if [ ! -f {_quote(evaluation_script_inside)} ]; then",
                    f"    echo {_quote('VBench 评测脚本不存在: ' + str(evaluation_script_inside))} >&2",
                    "    exit 1",
                    "fi",
                ]
            )
        else:
            logging.warning(
                "VBench evaluation script not found on host: %s (container path: %s)",
                evaluation_script,
                evaluation_script_inside,
            )
        check_enabled = bool(config.get("check", True))
        # 评测结果（{name}_eval_results.json / {name}_final_score.json）必须写进
        # 本次 RUN_DIR，postprocess 才能在日期目录下找到分数文件。
        eval_parts = [
            _quote(python_exec),
            _quote(evaluation_script_inside),
            "--full_info",
            output_full_info_arg,
            "--resolution",
            _quote(config.get("resolution", "16-9")),
            "--name",
            _quote(name),
            "--output_dir",
            '"$RUN_DIR"',
        ]
        dimensions = config.get("dimensions") or config.get("dims")
        if dimensions:
            if not isinstance(dimensions, (list, tuple)):
                raise ValueError("benchmark.vbench.dimensions must be a list")
            # The evaluator defines --dims as a list-valued option.  Emit one
            # shell argument per dimension so a multi-dimension YAML config
            # behaves the same as its command-line form.
            eval_parts.append("--dims")
            eval_parts.extend(_quote(item) for item in dimensions)
        if check_enabled:
            check_parts = eval_parts + ["--check"]
            _append_checked(shell, " ".join(check_parts), "VBench 数据完整性检查失败")
        shell.append('echo "开始执行 VBench 质量评测"')
        _append_checked(
            shell,
            "(cd \"$RUN_DIR\" && " + " ".join(eval_parts) + ")",
            "VBench 质量评测失败",
        )

    shell.extend(
        [
            'echo "VBench 测试完成，结果保存至 $RUN_DIR"',
            'printf "%s\\n" "$RUN_DIR" > "$RUN_DIR/.xtask_vbench_output_dir"',
        ]
    )
    output_shell_path.write_text("\n".join(shell) + "\n", encoding="utf-8")
    os.chmod(output_shell_path, 0o755)
    logging.info("VBench shell script generated: %s", output_shell_path)
    return f"bash {path_mapping_in_container(str(output_shell_path))}"
