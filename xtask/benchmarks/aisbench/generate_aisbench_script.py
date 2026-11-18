import ast
import json
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Dict

from xtask.utils.container_manager import path_mapping_in_container
from xtask.utils.config_parser import get_xtask_config_data
from xtask.datasets.dataset_downloader import download_dataset
from xtask.datasets.dataset_manger import (
    get_custom_dataset_output_path,
    get_sharegpt_fixed_output_path,
    get_sharegpt_prefix_cache_output_path,
    is_custom_dataset_enabled,
    is_sharegpt_prefix_cache_enabled,
    is_sharegpt_fixed_length_enabled,
    preprocess_dataset,
)


CONFIG_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


# gedit 数据集默认根：环境变量优先、config/xTask_config.json 兜底
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


def default_gedit_acc_dataset_path() -> str:
    """gedit acc 默认数据集（GEdit-Bench，load_from_disk 格式，host 路径）。

    仅 api_type 以 gedit 开头的 aisbench 用例使用；YAML
    benchmark.acc.dataset_path 显式指定时始终优先。
    """
    return (
        _dependence_package_path()
        + "/benchmark/ais_bench/datasets/GEdit-Bench"
    )


def _validate_config_name(value, field_name):
    if not value or not CONFIG_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, '.', '_' or '-': {value!r}"
        )
    return value


def _split_dataset_config_reference(dataset_conf):
    parts = str(dataset_conf or "").split("/", 1)
    config_name = _validate_config_name(parts[0], "dataset_conf")
    config_suffix = ""
    if len(parts) == 2:
        config_suffix = _validate_config_name(parts[1], "dataset_conf suffix")
    return config_name, config_suffix


_URL_FIELD_SED = (
    '''sed -i -E "s#^([[:space:]]*)url[[:space:]]*=.*#\\1url='{url}',#" {config_file}'''
)
# grep -F: the URL is matched as a fixed string, so no regex escaping applies.
_URL_FIELD_CHECK = (
    '''if ! grep -qF "url='{url}'" {config_file}; then '''
    '''echo "错误：AISBench 模型配置缺少可用的 url 字段，无法使用端点 {url}" >&2; exit 1; fi'''
)


def _validate_url_literal(url: str) -> None:
    """Reject characters that would break the single-quoted sed/echo snippets."""
    if not url:
        raise ValueError("AISBench url override must not be empty")
    # '#': sed s### delimiter; '&': sed backreference in the replacement;
    # "'"/'"'/'$'/'`'/'\': shell and quoting hazards in the generated script.
    for bad in ("'", '"', "$", "`", "\\", "#", "&"):
        if bad in url:
            raise ValueError(
                f"AISBench url override contains unsupported character {bad!r}: {url!r}"
            )


def append_api_endpoint_override(
    shell_content, api_type_file, api_address, api_port, aisbench_url=""
):
    """Write the model-under-test endpoint into the case-local AISBench config.

    Two mutually exclusive modes:

    * ``aisbench_url`` empty -> historical behaviour: sed ``host_ip`` /
      ``host_port``. Byte-identical to the scripts xTask generated before
      external-service URL support existed.
    * ``aisbench_url`` set -> sed AISBench's ``url`` field and leave
      ``host_ip`` at the template default ``"localhost"``. AISBench ignores
      host_ip/host_port whenever ``url`` is non-empty, and ``localhost`` is
      the only value its config validator whitelists besides IP literals.

    Only the copied, case-local config (``$CONFIG_FILE``) is rewritten; the
    shared AISBench install is never touched.
    """
    if aisbench_url:
        _validate_url_literal(aisbench_url)
        shell_content.append(
            _URL_FIELD_SED.format(url=aisbench_url, config_file=api_type_file)
        )
        shell_content.append(
            _URL_FIELD_CHECK.format(url=aisbench_url, config_file=api_type_file)
        )
        return
    shell_content.append(
        f'sed -i.bak "s/host_ip\\s*=\\s*.*,/host_ip=\'{api_address}\',/g" {api_type_file}'
    )
    shell_content.append(
        f'sed -i "s/host_port\\s*=\\s*.*,/host_port={api_port},/g" {api_type_file}'
    )


def get_api_type_file(api_type, shell_content):
    _validate_config_name(api_type, "api_type")
    model_config_name = f"xtask_{api_type}"
    # xtask 内置 api_type 源（如 gedit_image_edit）：优先于 AISBench 部署目录，
    # 便于随仓库版本化而不污染共享的 DEPENDENCE_PACKAGE_PATH。
    from xtask.utils.container_manager import path_mapping_in_container

    builtin_source = Path(__file__).resolve().parent / "configs" / "models" / f"{api_type}.py"
    shell_content.append("\n# 2. 创建用例级vllm_api配置")
    if builtin_source.is_file():
        shell_content.append(
            f'SOURCE_CONFIG_FILE="{path_mapping_in_container(str(builtin_source))}"'
        )
    else:
        shell_content.append(
            f'SOURCE_CONFIG_FILE="/xllm_test_060/benchmark/'
            f'ais_bench/benchmark/configs/models/vllm_api/{api_type}.py"'
        )
        shell_content.append('if [ ! -f "$SOURCE_CONFIG_FILE" ]; then')
        shell_content.append(
            f'    SOURCE_CONFIG_FILE="$AIS_BENCH_PATH/benchmark/configs/models/vllm_api/{api_type}.py"'
        )
        shell_content.append('fi')
    shell_content.append(f'CONFIG_FILE="$AIS_BENCH_CONFIG_DIR/models/{model_config_name}.py"')
    shell_content.append('if [ ! -f "$SOURCE_CONFIG_FILE" ]; then')
    shell_content.append('    echo "错误：AISBench模型配置文件不存在 $SOURCE_CONFIG_FILE"')
    shell_content.append('    exit 1')
    shell_content.append('fi')
    shell_content.append('cp -f "$SOURCE_CONFIG_FILE" "$CONFIG_FILE"')
    shell_content.append('echo "用例级配置文件：$CONFIG_FILE"')

    return "$CONFIG_FILE", model_config_name, shell_content


def _aisbench_dataset_config_roots():
    dep_path = (get_xtask_config_data().get("ENV_CONFIG") or {}).get(
        "DEPENDENCE_PACKAGE_PATH", ""
    ).strip()
    if dep_path:
        yield Path(dep_path) / "benchmark"
    yield Path(__file__).resolve().parents[3] / "benchmark"


def _xtask_aisbench_dataset_config_root():
    return Path(__file__).resolve().parent / "configs" / "datasets"


def find_aisbench_dataset_config(dataset_conf):
    """Find one AISBench dataset config by its command-line name."""
    dataset_conf, _ = _split_dataset_config_reference(dataset_conf)
    relative_root = Path("ais_bench/benchmark/configs/datasets")
    for root in _aisbench_dataset_config_roots():
        config_root = root / relative_root
        # Why: candidate root 可能是 root 属主且无读/进权限(如公共 models 目录 750)，
        # is_dir()/rglob() 会抛 PermissionError 而非返回 False。捕获后跳到下一个 root，
        # 避免一个无权限的 candidate 把整个 generate_aisbench_script 搞崩。
        try:
            if not config_root.is_dir():
                continue
            matches = sorted(config_root.rglob(f"{dataset_conf}.py"))
        except PermissionError:
            continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"AISBench dataset config {dataset_conf!r} is ambiguous under {config_root}"
            )
    xtask_root = _xtask_aisbench_dataset_config_root()
    try:
        if xtask_root.is_dir():
            matches = sorted(xtask_root.rglob(f"{dataset_conf}.py"))
        else:
            matches = []
    except PermissionError:
        matches = []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"AISBench dataset config {dataset_conf!r} is ambiguous under {xtask_root}"
        )
    raise FileNotFoundError(
        f"AISBench dataset config {dataset_conf!r} was not found in the configured "
        "AISBench source, the bundled benchmark tree, or xTask AISBench dataset configs"
    )


class _DatasetPathOverride(ast.NodeTransformer):
    """Replace dataset dict path fields without importing AISBench on the host."""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.replacements = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "dict":
            return node
        keywords = {item.arg: item for item in node.keywords if item.arg}
        if {"abbr", "type", "path"}.issubset(keywords):
            keywords["path"].value = ast.Constant(value=self.dataset_path)
            self.replacements += 1
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        keys = {
            key.value: index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if {"abbr", "type", "path"}.issubset(keys):
            node.values[keys["path"]] = ast.Constant(value=self.dataset_path)
            self.replacements += 1
        return node


class _InferModeOverride(ast.NodeTransformer):
    """Update infer_mode in an isolated multi-turn dataset config."""

    def __init__(self, infer_mode):
        self.infer_mode = infer_mode
        self.replacements = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "dict":
            return node
        for keyword in node.keywords:
            if keyword.arg == "infer_mode":
                keyword.value = ast.Constant(value=self.infer_mode)
                self.replacements += 1
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        for index, key in enumerate(node.keys):
            if isinstance(key, ast.Constant) and key.value == "infer_mode":
                node.values[index] = ast.Constant(value=self.infer_mode)
                self.replacements += 1
        return node


class _LastInferencerOverride(ast.NodeTransformer):
    """Use xTask's isolated workaround for AISBench infer_mode=last."""

    def __init__(self):
        self.replacements = 0

    def _replace_type(self, type_node):
        if isinstance(type_node, ast.Name) and type_node.id == "MultiTurnGenInferencer":
            self.replacements += 1
            return ast.Name(id="XTaskShareGPTLastInferencer", ctx=ast.Load())
        return type_node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "dict":
            return node
        keywords = {item.arg: item for item in node.keywords if item.arg}
        if "infer_mode" in keywords and "type" in keywords:
            keywords["type"].value = self._replace_type(keywords["type"].value)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        keys = {
            key.value: index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "infer_mode" in keys and "type" in keys:
            type_index = keys["type"]
            node.values[type_index] = self._replace_type(node.values[type_index])
        return node


class _JudgeEndpointOverride(ast.NodeTransformer):
    """Point the LLM-judge model in a dataset config at a custom host:port.

    The gedit llmjudge configs build ``dict(judge_model=dict(...))`` with
    ``dict`` calls (not dict literals), so both ``ast.Call`` nodes whose
    keywords include ``judge_model`` and ``ast.Dict`` literals holding a
    ``judge_model`` key are handled. Other host_ip/host_port leaves are left
    untouched so the model-under-test endpoint keeps its sed replacement.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self.replacements = 0

    def _override_judge_value(self, judge):
        """Override host_ip/host_port inside the judge_model value node."""
        if isinstance(judge, ast.Call):
            for keyword in judge.keywords:
                if keyword.arg == "host_ip":
                    keyword.value = ast.Constant(value=self.host)
                    self.replacements += 1
                elif keyword.arg == "host_port":
                    keyword.value = ast.Constant(value=self.port)
                    self.replacements += 1
        elif isinstance(judge, ast.Dict):
            for index, key in enumerate(judge.keys):
                if isinstance(key, ast.Constant) and key.value == "host_ip":
                    judge.values[index] = ast.Constant(value=self.host)
                    self.replacements += 1
                elif isinstance(key, ast.Constant) and key.value == "host_port":
                    judge.values[index] = ast.Constant(value=self.port)
                    self.replacements += 1
        return judge

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "dict":
            return node
        for keyword in node.keywords:
            if keyword.arg == "judge_model":
                keyword.value = self._override_judge_value(keyword.value)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        keys = {
            key.value: index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "judge_model" in keys:
            node.values[keys["judge_model"]] = self._override_judge_value(
                node.values[keys["judge_model"]]
            )
        return node


def _add_last_inferencer_import(tree):
    import_node = ast.ImportFrom(
        module="xtask.benchmarks.aisbench.sharegpt_last_inferencer",
        names=[ast.alias(name="XTaskShareGPTLastInferencer")],
        level=0,
    )
    insert_index = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insert_index = 1
    while (
        insert_index < len(tree.body)
        and isinstance(tree.body[insert_index], ast.ImportFrom)
        and tree.body[insert_index].module == "__future__"
    ):
        insert_index += 1
    tree.body.insert(insert_index, import_node)


def create_dataset_config_override(
    dataset_conf,
    dataset_path,
    output_file_path,
    infer_mode=None,
    use_xtask_last_inferencer=False,
    config_dir_name="aisbench_configs",
    judge_endpoint=None,
):
    """Create an isolated AISBench dataset config for per-case overrides."""
    config_name, config_suffix = _split_dataset_config_reference(dataset_conf)
    if not dataset_path and not infer_mode and not judge_endpoint:
        return dataset_conf

    source_config = find_aisbench_dataset_config(config_name)
    tree = ast.parse(source_config.read_text(encoding="utf-8"), str(source_config))
    if dataset_path:
        dataset_path_inside = path_mapping_in_container(str(dataset_path))
        path_transformer = _DatasetPathOverride(dataset_path_inside)
        tree = path_transformer.visit(tree)
        if path_transformer.replacements == 0:
            raise ValueError(
                f"AISBench dataset config {source_config} has no dataset path field to override"
            )
    if infer_mode:
        infer_mode_transformer = _InferModeOverride(infer_mode)
        tree = infer_mode_transformer.visit(tree)
        if infer_mode_transformer.replacements == 0:
            raise ValueError(
                f"AISBench dataset config {source_config} has no infer_mode field to override"
            )
    if use_xtask_last_inferencer:
        last_inferencer_transformer = _LastInferencerOverride()
        tree = last_inferencer_transformer.visit(tree)
        if last_inferencer_transformer.replacements == 0:
            raise ValueError(
                f"AISBench dataset config {source_config} has no "
                "MultiTurnGenInferencer to override"
            )
        _add_last_inferencer_import(tree)
    if judge_endpoint:
        host, _, port_str = str(judge_endpoint).rpartition(":")
        if not host or not port_str.isdigit():
            raise ValueError(
                f"judge_endpoint must be host:port, got {judge_endpoint!r}"
            )
        judge_transformer = _JudgeEndpointOverride(host, int(port_str))
        tree = judge_transformer.visit(tree)
        if judge_transformer.replacements == 0:
            raise ValueError(
                f"AISBench dataset config {source_config} has no judge_model "
                "host_ip/host_port to override"
            )

    custom_name = f"xtask_{config_name}"
    config_dir_name = _validate_config_name(config_dir_name, "config_dir_name")
    custom_dir = Path(output_file_path) / config_dir_name / "datasets"
    custom_dir.mkdir(parents=True, exist_ok=True)
    custom_config = custom_dir / f"{custom_name}.py"
    custom_config.write_text(
        ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8"
    )
    return f"{custom_name}/{config_suffix}" if config_suffix else custom_name


def append_dataset_path_check(shell_content, dataset_path):
    if not dataset_path:
        return
    dataset_path_inside = path_mapping_in_container(str(dataset_path))
    if not os.path.isabs(dataset_path_inside):
        return
    quoted_path = shlex.quote(dataset_path_inside)
    shell_content.append(f"if [ ! -e {quoted_path} ]; then")
    error_message = shlex.quote(
        f"错误：AISBench数据集路径不存在 {dataset_path_inside}"
    )
    shell_content.append(f"    echo {error_message} >&2")
    shell_content.append("    exit 1")
    shell_content.append("fi")


def append_xtask_pythonpath(shell_content):
    """Make xTask compatibility modules importable in AISBench subprocesses.

    注意：调度器运行 tools/download_ais_bench.sh 时会把
    /xllm_test_060/benchmark rsync 到 /workspace/benchmark 并 pip install -e，
    因此 AISBench 的修复必须落在 rsync 源头（xllm_test_060 挂载卷），
    这里只负责注入 xTask 仓库根的 PYTHONPATH。
    """
    workspace_root = Path(__file__).resolve().parents[3]
    workspace_root_inside = path_mapping_in_container(str(workspace_root))
    quoted_root = shlex.quote(workspace_root_inside)
    shell_content.append(
        f'export PYTHONPATH={quoted_root}"${{PYTHONPATH:+:$PYTHONPATH}}"'
    )

def get_syn_config_file(shell_content):
    shell_content.append("\n# 2. 定位syn_config配置文件")
    shell_content.append(f'SYN_FILE="$AIS_BENCH_PATH/datasets/synthetic/synthetic_config.py"')
    shell_content.append('if [ ! -f "$SYN_FILE" ]; then')
    shell_content.append('    echo "错误：配置文件不存在 $SYN_FILE"')
    shell_content.append('    exit 1')
    shell_content.append('fi')
    shell_content.append('echo "目标配置文件：$SYN_FILE"')

    return "$SYN_FILE", shell_content

def build_kwargs_block(generation_kwargs):
    """根据 generation_kwargs 字典构建替换内容"""
    kwargs_lines = []
    for key, value in generation_kwargs.items():
        if value is None:
            kwargs_lines.append(f'            {key}=None,')
        elif isinstance(value, str):
            if value == 'None':
                kwargs_lines.append(f'            {key}=None,')
            elif value in ('True', 'False'):
                kwargs_lines.append(f'            {key}={value},')
            elif "dict" in value:
                kwargs_lines.append(f'            {key}={value},')
            else:
                kwargs_lines.append(f'            {key}="{value}",')
        elif isinstance(value, bool):
            kwargs_lines.append(f'            {key}={value},')
        else:
            kwargs_lines.append(f'            {key}={value},')
    return kwargs_lines

def update_generation_kwargs(generation_kwargs, api_type_file):
    update_cmd = list()
    kwargs_lines = build_kwargs_block(generation_kwargs)
    new_lines = ['        generation_kwargs=dict(']
    new_lines.extend(kwargs_lines)
    new_lines.append('        ),')
    new_block = '\\n'.join(new_lines)

    update_cmd.append(f'START=$(grep -n "generation_kwargs=dict(" "{api_type_file}" | head -1 | cut -d: -f1)')
    update_cmd.append(f'END=$(awk "NR>$START && /^[[:space:]]*\\),?[[:space:]]*$/ {{print NR; exit}}" "{api_type_file}")')
    update_cmd.append(f'echo "替换范围: $START - $END"')
    update_cmd.append(f'sed -i "${{START}},${{END}}c\\{new_block}" "{api_type_file}"')
    update_cmd.append(f'echo "替换完成"')

    return update_cmd


def append_checked_command(shell_content, command, failure_message):
    shell_content.append(f'if ! {command}; then')
    shell_content.append(f'    echo "{failure_message}" >&2')
    shell_content.append('    exit 1')
    shell_content.append('fi')

def generate_aisbench_script(
    task_config: Dict,
    output_file_path: str,
    output_shell_file: str = "run_aisbench_test.sh",
    python_exec: str = "python3",
    config_dir_name: str = "aisbench_configs",
) -> None:
    """
    生成ais-bench测试的shell脚本，包含配置修改、精度/性能测试命令生成
    
    Args:
        task_config: 测试任务配置字典，示例结构见下方
        output_shell_file: 生成的shell脚本输出脚本名称
        python_exec: Python解释器路径（如python/python3/anaconda3/bin/python）
    """
    #output_file_path = os.path.join(os.getcwd(), "outputs", f"{task_config.get('case_name').lower()}")
    config_dir_name = _validate_config_name(config_dir_name, "config_dir_name")
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
    model_path_inside = path_mapping_in_container(model_path)
    user_id = os.getuid()

    # external_service 的域名 / https / 带子路径端点无法用 AISBench 的 host_ip
    # 表达（AISBench 校验只放行 IP 字面量 + localhost）。这类端点改走 url 字段，
    # 纯 IP + http + 无子路径时返回 ""，脚本与历史行为逐字节一致。
    from xtask.core.external_service import resolve_aisbench_url_override

    aisbench_url = resolve_aisbench_url_override(task_config)
    if aisbench_url and not model_name:
        # url 模式下若 model 为空，AISBench 会去请求 v1/models 探测模型名；
        # 该探测走 requests(verify=True) 且路径拼接语义与 urljoin 不同，
        # 会引入额外的失败面，因此显式要求 served_model_name 非空。
        raise ValueError(
            "external_service endpoint requires the AISBench url field; "
            "served_model_name must be non-empty so AISBench does not probe v1/models"
        )

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
    shell_content.append(f'AIS_BENCH_CONFIG_DIR="{output_file_path_incontainer}/{config_dir_name}"')
    shell_content.append('mkdir -p "$AIS_BENCH_CONFIG_DIR/models" "$AIS_BENCH_CONFIG_DIR/datasets"')

    download_cmd = download_dataset(task_config)
    if download_cmd:
        shell_content.extend(download_cmd)

    # Step 2: 生成精度测试命令, 如果--metrics配置了acc, 则读取benchmark xxx_acc_config.json配置我呢见中的参数
    # 注意：AISBench SWE-bench 新格式用例（metrics: acc + benchmark.acc，块内为
    # mode/runner/dataset 等 SWE 字段，没有 dataset_conf）不会进入本分支——
    # 调度层在 _run_benchmark_and_process 中已将其路由到 SWE 专用脚本生成器。
    # VBench 用例同样不受影响：其 metrics 现在也是 acc，但 benchmark_type 为
    # vbench，调度层在 generate_benchmark_command 中按 benchmark_type 路由到
    # generate_vbench_script，根本不会进入本 AISBench 生成器。
    if "acc" in task_config.get("metrics") and accuracy_cfg.get("dataset_conf"):
        api_type = accuracy_cfg.get("api_type", "vllm_api_general_chat")
        # dataset_path：YAML 显式值优先；gedit 用例（api_type 以 gedit 开头）
        # 未指定时落到共享 DEPENDENCE_PACKAGE_PATH 下的 GEdit-Bench 默认路径
        # （host 路径，容器侧经 create_dataset_config_override 内的
        # path_mapping_in_container 映射）。非 gedit 的 aisbench 用例保持
        # 原有必填语义（空值由数据集配置自身的相对路径逻辑兜底）。
        dataset_path = accuracy_cfg.get("dataset_path") or task_config.get(
            "dataset_path", ""
        )
        if not dataset_path and str(api_type or "").startswith("gedit"):
            dataset_path = default_gedit_acc_dataset_path()
        output_len = accuracy_cfg.get("output_len", 1024)
        request_rate = accuracy_cfg.get("request_rate", 0)
        batch_size = accuracy_cfg.get("batch_size", 32)
        temperature = accuracy_cfg.get("temperature", 0)
        top_k = accuracy_cfg.get("top_k", -1)
        top_p = accuracy_cfg.get("top_p", 0.95)
        seed = accuracy_cfg.get("seed", 1024)
        num_warmups = accuracy_cfg.get("num_warmups", 0)
        repetition_penalty = accuracy_cfg.get("repetition_penalty", 1)

        api_type_file, model_config_name, shell_content = get_api_type_file(api_type, shell_content)
        append_xtask_pythonpath(shell_content)
        if accuracy_cfg.get("request_timeout") is not None:
            request_timeout = int(accuracy_cfg["request_timeout"])
            request_retries = int(accuracy_cfg.get("request_retries", 2))
            if request_timeout <= 0 or request_retries <= 0:
                raise ValueError("request_timeout and request_retries must be positive")
            if api_type != "vllm_api_general_chat":
                raise ValueError("request_timeout currently supports vllm_api_general_chat only")
            override = (
                "\nfrom xtask.benchmarks.aisbench.request_timeout import XTaskGPQAChat\n"
                "models[0]['type'] = XTaskGPQAChat\n"
                f"models[0]['request_timeout'] = {request_timeout}\n"
                f"models[0]['retry'] = {request_retries}\n"
            )
            shell_content.append("printf '%s' " + shlex.quote(override) + ' >> "$CONFIG_FILE"')


        # gedit 等 xtask 内置 api_type 的模型配置 import xtask 兼容模块
        # （xtask.benchmarks.aisbench.gedit_image_edit_api 等），AISBench 子进程
        # 需要 PYTHONPATH 含 xTask 仓库根才能 import 成功。
        if str(api_type or "").startswith("gedit"):
            append_xtask_pythonpath(shell_content)

        # Step 3: 修改配置文件中的核心参数（使用sed命令）
        is_gedit_api = str(api_type or "").startswith("gedit")
        shell_content.append("\n# 3. 修改vllm_api配置文件$CONFIG_FILE参数")
        # 替换API地址：external_service 的域名/https/带子路径端点无法用 host_ip
        # 表达（AISBench 只接受 IP 字面量 + localhost 白名单），此时改写 url 字段。
        append_api_endpoint_override(
            shell_content, api_type_file, api_address, api_port, aisbench_url
        )
        # 替换模型名称
        shell_content.append(f'sed -i "s/model\s*=\s*.*,/model=\'{model_name}\',/g" {api_type_file}')
        # 替换模型路径
        shell_content.append(
            f'''sed -i -E "s#^([[:space:]]*)path[[:space:]]*=.*,#\\1path='{model_path_inside}',#" {api_type_file}'''
        )

        # gedit（图片编辑两段式）配置中的 temperature/top_k/seed 等属于
        # DiT 请求体参数（generation_kwargs 透传），通用文本 LLM 的 sed
        # 替换会破坏 encoder_url/size/num_inference_steps 等关键参数，
        # 因此 gedit 只做地址/端口/模型名/路径替换，其余保持原配置。
        if not is_gedit_api:
            shell_content.append(f'sed -i "s/temperature\s*=\s*.*,/temperature={temperature},/g" {api_type_file}')
            shell_content.append(f'sed -i "s/request_rate\s*=\s*.*,/request_rate={request_rate},/g" {api_type_file}')

            shell_content.append(f'sed -i "s/top_k\s*=\s*.*,/top_k={top_k},/g" {api_type_file}')
            shell_content.append(f'sed -i "s/top_p\s*=\s*.*,/top_p={top_p},/g" {api_type_file}')
            shell_content.append(f'sed -i "s/seed\s*=\s*.*,/seed={seed},/g" {api_type_file}')
            shell_content.append(f'sed -i "s/repetition_penalty\s*=\s*.*,/repetition_penalty={repetition_penalty},/g" {api_type_file}')

        shell_content.append("\n# 4. 执行精度测试")
        # metrics_str = ",".join(accuracy_metrics)
        dataset_conf = accuracy_cfg.get("dataset_conf") or dataset_name
        dataset_conf = create_dataset_config_override(
            dataset_conf,
            dataset_path,
            output_file_path,
            config_dir_name=config_dir_name,
            judge_endpoint=accuracy_cfg.get("judge_endpoint"),
        )
        append_dataset_path_check(shell_content, dataset_path)
        # accuracy_metrics = accuracy_cfg.get("metrics", ["bleu", "rouge"])
        accuracy_batch_size = accuracy_cfg.get("batch_size", batch_size)
        accuracy_max_out_len = accuracy_cfg.get("output_len", output_len)
        acc_concurrency = accuracy_cfg.get("concurrency", 0)
        dump_eval_details = task_config.get("dump_eval_details", "")
        num_prompts = accuracy_cfg.get("num_prompts", "")

        # 替换最大生成token数
        # aisbench build.py 硬编码校验 max_out_len <= 131072，若配置值超限则自动放宽校验上限
        # (模型实际上下文远大于此，如 DSV4 支持 1M；131072 仅为 aisbench 保守默认值)
        if int(accuracy_max_out_len) > 131072:
            shell_content.append(
                'if [ -f "$AIS_BENCH_PATH/benchmark/utils/config/build.py" ]; then '
                'sed -i "s/0 < v <= 131072/0 < v <= 1048576/; s/(0, 131072\\\\]/(0, 1048576\\\\]/" '
                '"$AIS_BENCH_PATH/benchmark/utils/config/build.py" 2>/dev/null; fi'
            )
        shell_content.append(f'sed -i "s/max_out_len\s*=\s*.*,/max_out_len={accuracy_max_out_len},/g" {api_type_file}')
        # 替换batchsize
        shell_content.append(f'sed -i "s/batch_size\s*=\s*.*,/batch_size={accuracy_batch_size},/g" {api_type_file}')
        # shell_content.append(f'sed -i "s/request_rate\s*=\s*.*,/request_rate={acc_concurrency},/g" {api_type_file}')

        # 更新aisbench模型配置中generation_kwargs后端参数
        # gedit 的 generation_kwargs 承载两段式 DiT 参数（size/
        # num_inference_steps/encoder_url 等），YAML 里的通用文本 LLM
        # 参数（temperature 等）不得整块覆盖，保持模型配置文件原值。
        generation_kwargs = accuracy_cfg.get('generation_kwargs', {})
        if generation_kwargs and not str(api_type or "").startswith("gedit"):  # 如果有需要更新的参数
            update_cmd = update_generation_kwargs(generation_kwargs, api_type_file)
            shell_content.extend(update_cmd)

        if dump_eval_details:
            dump_eval_details_str = "--dump-eval-details "
        else:
            dump_eval_details_str = ""

        if num_prompts and isinstance(num_prompts, int): 
            num_prompts_str = f"--num-prompts {num_prompts} "
        else:
            num_prompts_str = ""

        # 构造ais-bench精度测试命令（参考ais-bench官方文档）
        accuracy_cmd = (
            f'ais_bench --config-dir "$AIS_BENCH_CONFIG_DIR" '
            f'--models {model_config_name} '
            f'--datasets {dataset_conf} '
            f'{dump_eval_details_str} '
            f'{num_prompts_str} '
            f'--work-dir {output_file_path_incontainer}'
        )

        shell_content.append(f'echo "执行精度测试命令：{accuracy_cmd}"')
        append_checked_command(shell_content, accuracy_cmd, "AISBench精度测试执行失败")
        shell_content.append(f'echo "精度测试完成，结果保存至 {output_file_path_incontainer}"')

        shell_content.append(f'cp {api_type_file} {output_file_path_incontainer}/ ')
        shell_content.append(f'echo "配置文件修改完成（已生成备份文件 ${api_type_file}）"')
    else:
        shell_content.append("\n# 4. 执行精度测试，skip task...")

    # Step 4: 生成性能测试命令（若开启）
    if "perf" in task_config.get("metrics"):
        dataset_path = perf_cfg.get("dataset_path") or task_config.get(
            "dataset_path", ""
        )
        output_len = perf_cfg.get("output_len", 1024)
        request_rate = perf_cfg.get("request_rate", 0)
        batch_size = perf_cfg.get("batch_size", 32)
        temperature = perf_cfg.get("temperature", 0)
        top_k = perf_cfg.get("top_k", -1)
        top_p = perf_cfg.get("top_p", 0.95)
        seed = perf_cfg.get("seed", 1024)
        num_warmups = perf_cfg.get("num_warmups", 0)
        repetition_penalty = perf_cfg.get("repetition_penalty", 1)
        ignore_eos = perf_cfg.get("ignore_eos", "True")

        api_type = perf_cfg.get("api_type", "vllm_api_general_stream")
        api_type_file, model_config_name, shell_content = get_api_type_file(api_type, shell_content)

        # Step 2: 修改配置文件中的核心参数（使用sed命令）
        shell_content.append("\n# 3. 修改vllm_api配置文件$CONFIG_FILE参数")
        # 替换API地址：同精度分支，external_service 端点无法用 host_ip 时改写 url。
        append_api_endpoint_override(
            shell_content, api_type_file, api_address, api_port, aisbench_url
        )
        # 替换模型名称
        shell_content.append(f'sed -i "s/model\s*=\s*.*,/model=\'{model_name}\',/g" {api_type_file}')
        # 替换模型路径
        shell_content.append(
            f'''sed -i -E "s#^([[:space:]]*)path[[:space:]]*=.*,#\\1path='{model_path_inside}',#" {api_type_file}'''
        )

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

        process_dataset_cmd, preprocessed_dataset_conf = preprocess_dataset(
            dataset_name, task_config, output_file_path
        )
        dataset_conf_type = (
            preprocessed_dataset_conf
            or perf_cfg.get("dataset_conf")
            or dataset_name
        )
        sharegpt_infer_mode = None
        if dataset_name == "sharegpt" and perf_cfg.get("infer_mode"):
            sharegpt_infer_mode = str(perf_cfg["infer_mode"]).lower()
            if sharegpt_infer_mode not in {"last", "every", "every_with_gt"}:
                raise ValueError(
                    "ShareGPT benchmark.perf.infer_mode must be one of: "
                    "last, every, every_with_gt"
                )
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
        # 替换ignore_eos配置
        shell_content.append(f'sed -i "s/ignore_eos\s*=\s*.*,/ignore_eos={ignore_eos},/g" {api_type_file}')

        # 更新aisbench模型配置中generation_kwargs后端参数
        generation_kwargs = perf_cfg.get('generation_kwargs', {})
        if generation_kwargs:  # 如果有需要更新的参数
            update_cmd = update_generation_kwargs(generation_kwargs, api_type_file)
            shell_content.extend(update_cmd)

        if dataset_name == "syn" and task_config.get("benchmark").get("benchmark_type") == "aisbench":
            syn_config_file, shell_content = get_syn_config_file(shell_content)
            num_prompts_cnt = task_config.get("benchmark").get("perf").get("num_prompts", batch_size)
            input_len = task_config.get("benchmark").get("perf").get("input_len")
            trust_remote_mode = "True"

            shell_content.append(f'sed -i \'s/"RequestCount"\s*:\s*.*,/"RequestCount": {num_prompts_cnt},/g\' {syn_config_file}')
            shell_content.append(f'sed -i \'s/"TrustRemoteCode"\s*:\s*.*,/"TrustRemoteCode": {trust_remote_mode},/g\' {syn_config_file}')
            shell_content.append(f'sed -i \'s/"RequestSize"\s*:\s*.*,/"RequestSize": {input_len},/g\' {syn_config_file}')

            #shell_content.append(f'sed -i "s/RequestCount\s*=\s*.*,/RequestCount={num_prompts_cnt},/g" {syn_config_file}')
            #shell_content.append(f'sed -i "s/TrustRemoteCode\s*=\s*.*,/TrustRemoteCode={trust_remote_mode},/g" {syn_config_file}')
            #shell_content.append(f'sed -i "s/RequestSize\s*=\s*.*,/RequestSize={input_len},/g" {syn_config_file}')

            # 随机合成数据集
            dataset_conf_type = "synthetic_gen"

        if is_sharegpt_prefix_cache_enabled(dataset_name, task_config):
            generated_dataset_path = get_sharegpt_prefix_cache_output_path(
                task_config, output_file_path
            )
            dataset_conf_type = create_dataset_config_override(
                dataset_conf_type,
                generated_dataset_path,
                output_file_path,
                infer_mode=sharegpt_infer_mode,
                use_xtask_last_inferencer=sharegpt_infer_mode == "last",
                config_dir_name=config_dir_name,
            )
            if sharegpt_infer_mode == "last":
                append_xtask_pythonpath(shell_content)
        elif is_sharegpt_fixed_length_enabled(dataset_name, task_config):
            generated_dataset_path = get_sharegpt_fixed_output_path(
                task_config, output_file_path
            )
            dataset_conf_type = create_dataset_config_override(
                dataset_conf_type,
                generated_dataset_path,
                output_file_path,
                infer_mode="last",
                use_xtask_last_inferencer=True,
                config_dir_name=config_dir_name,
            )
            append_xtask_pythonpath(shell_content)
        elif is_custom_dataset_enabled(dataset_name, task_config):
            generated_dataset_path = get_custom_dataset_output_path(
                task_config, output_file_path
            )
            dataset_conf_type = create_dataset_config_override(
                dataset_conf_type,
                generated_dataset_path,
                output_file_path,
                config_dir_name=config_dir_name,
            )
            append_dataset_path_check(shell_content, dataset_path)
        elif not preprocessed_dataset_conf and dataset_name != "syn":
            dataset_conf_type = create_dataset_config_override(
                dataset_conf_type,
                dataset_path,
                output_file_path,
                infer_mode=sharegpt_infer_mode,
                use_xtask_last_inferencer=sharegpt_infer_mode == "last",
                config_dir_name=config_dir_name,
            )
            if sharegpt_infer_mode == "last":
                append_xtask_pythonpath(shell_content)
            append_dataset_path_check(shell_content, dataset_path)

        if process_dataset_cmd:
            shell_content.extend(process_dataset_cmd)
        
        # 构造ais-bench性能测试命令（参考ais-bench官方文档）
        num_prompts = perf_cfg.get("num_prompts")
        num_prompts_arg = (
            f"--num-prompts {num_prompts} "
            if isinstance(num_prompts, int) and num_prompts > 0
            else ""
        )
        perf_cmd = (
            f'ais_bench --config-dir "$AIS_BENCH_CONFIG_DIR" '
            f'--models {model_config_name} '
            f'--datasets {dataset_conf_type} '
            f'{num_prompts_arg}'
            f'--num-warmups {num_warmups} '
            f'--mode {mode} --debug '
            f'--summarizer {summarizer} '
            f'--work-dir {output_file_path_incontainer}'
        )
        shell_content.append(f'echo "执行性能测试命令：{perf_cmd}"')
        append_checked_command(shell_content, perf_cmd, "AISBench性能测试执行失败")
        shell_content.append(f'echo "性能测试完成，结果保存至 {output_file_path_incontainer}"')

        shell_content.append(f'cp {api_type_file} {output_file_path_incontainer}/ ')
        shell_content.append(f'echo "配置文件修改完成（已生成备份文件 ${api_type_file}）"')
    else:
        shell_content.append("\n# 5. 执行性能测试，skip task...")
    
    shell_content.append(f'chown -R {user_id}:{user_id} {output_file_path_incontainer}/* ')

    shell_content.append('if [ -n "${CHAT_CONFIG_FILE:-}" ] && [ -f "$CHAT_CONFIG_FILE" ]; then')
    shell_content.append(f'    cp "$CHAT_CONFIG_FILE" {output_file_path_incontainer}/ ')
    shell_content.append('fi')

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
