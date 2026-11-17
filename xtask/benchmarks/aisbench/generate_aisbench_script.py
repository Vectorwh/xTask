import ast
import json
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Dict

from xtask.utils.container_manager import path_mapping_in_container
from xtask.utils.config_parser import get_xtask_config_data, is_protocol_test_enabled
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


def get_api_type_file(api_type, shell_content):
    _validate_config_name(api_type, "api_type")
    model_config_name = f"xtask_{api_type}"
    shell_content.append("\n# 2. 创建用例级vllm_api配置")
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


class _ProtocolModelTypeOverride(ast.NodeTransformer):
    """Replace VLLMCustomAPIChat/VLLMCustomAPIChatStream with ProtocolAwareAPIChat
    in an AISBench model config, inject stream_verify, bake connection params,
    and update generation_kwargs.

    Why overrides: ast.unparse() produces single-line output where sed's greedy
    ``.*`` patterns destroy fields after host_ip. Baking all connection params
    (host_ip, host_port, model, path, max_out_len, batch_size, request_rate,
    temperature) directly into the AST eliminates the need for sed entirely.
    """

    def __init__(self, stream_verify=False, generation_kwargs=None, overrides=None):
        self.stream_verify = stream_verify
        self.generation_kwargs = generation_kwargs or {}
        self.overrides = overrides or {}
        self.replacements = 0

    def _replace_type(self, type_node):
        target_names = {"VLLMCustomAPIChat", "VLLMCustomAPIChatStream"}
        if isinstance(type_node, ast.Name) and type_node.id in target_names:
            self.replacements += 1
            return ast.Name(id="ProtocolAwareAPIChat", ctx=ast.Load())
        return type_node

    @staticmethod
    def _value_to_ast(value):
        """Convert a Python value to an AST node."""
        if isinstance(value, (int, float, bool, str, type(None))):
            return ast.Constant(value=value)
        try:
            return ast.parse(repr(value), mode="eval").body
        except SyntaxError:
            return ast.Constant(value=str(value))

    @staticmethod
    def _set_keyword(node, arg, value_ast):
        """Set or replace a keyword in a dict(...) Call, preserving order."""
        for i, item in enumerate(node.keywords):
            if item.arg == arg:
                node.keywords[i] = ast.keyword(arg=arg, value=value_ast)
                return
        node.keywords.append(ast.keyword(arg=arg, value=value_ast))

    def _update_generation_kwargs(self, gk_call_node):
        """Update keywords inside generation_kwargs=dict(...) AST node."""
        if not self.generation_kwargs:
            return gk_call_node
        existing = {kw.arg: kw for kw in gk_call_node.keywords if kw.arg}
        for key, value in self.generation_kwargs.items():
            existing[key] = ast.keyword(arg=key, value=self._value_to_ast(value))
        gk_call_node.keywords = list(existing.values())
        return gk_call_node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "dict":
            return node
        keywords = {item.arg: item for item in node.keywords if item.arg}
        if "type" in keywords:
            keywords["type"].value = self._replace_type(keywords["type"].value)
        # Model dict: has both abbr and type keywords.
        if "abbr" in keywords and "type" in keywords:
            # Bake all connection params into the model dict.
            for key, value in self.overrides.items():
                self._set_keyword(node, key, self._value_to_ast(value))
            # Inject stream_verify kwarg into the model dict.
            self._set_keyword(
                node, "stream_verify", ast.Constant(value=self.stream_verify)
            )
        # Update generation_kwargs values baked into the AST.
        if "generation_kwargs" in keywords:
            gk_value = keywords["generation_kwargs"].value
            if (
                isinstance(gk_value, ast.Call)
                and isinstance(gk_value.func, ast.Name)
                and gk_value.func.id == "dict"
            ):
                self._update_generation_kwargs(gk_value)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        keys = {
            key.value: index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "type" in keys:
            node.values[keys["type"]] = self._replace_type(node.values[keys["type"]])
        return node


def _add_protocol_import(tree):
    """Insert the ProtocolAwareAPIChat import after existing imports.

    Imports the wrapper class directly (NOT the factory function) because
    mmengine's _parse_lazy_import forbids calling imported names at module
    level. ProtocolAwareAPIChat defers composition to __new__.
    """
    import_node = ast.ImportFrom(
        module="xtask.benchmarks.aisbench.protocol_api",
        names=[ast.alias(name="ProtocolAwareAPIChat")],
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
        and isinstance(tree.body[insert_index], (ast.Import, ast.ImportFrom))
    ):
        insert_index += 1
    tree.body.insert(insert_index, import_node)


def create_protocol_model_config(
    api_type,
    output_file_path,
    stream_verify=False,
    config_dir_name="aisbench_configs",
    generation_kwargs=None,
    overrides=None,
):
    """Create a protocol-aware AISBench model config from a base api_type config.

    Copies the base vllm_api config, replaces the model class with
    ProtocolAwareAPIChat, injects stream_verify + generation_kwargs, bakes
    connection params (host_ip, host_port, model, path, max_out_len, etc.),
    and adds the xtask import. All parameter overrides are baked into the AST
    so no sed-based post-editing is needed.
    """
    _validate_config_name(api_type, "api_type")
    # Model configs live under configs/models/vllm_api/, not configs/datasets/.
    relative_root = Path("ais_bench/benchmark/configs/models/vllm_api")
    source_path = None
    for root in _aisbench_dataset_config_roots():
        candidate = root / relative_root / f"{api_type}.py"
        try:
            if candidate.is_file():
                source_path = candidate
                break
        except PermissionError:
            continue
    if source_path is None:
        raise FileNotFoundError(
            f"AISBench model config vllm_api/{api_type}.py was not found"
        )

    tree = ast.parse(source_path.read_text(encoding="utf-8"), str(source_path))
    transformer = _ProtocolModelTypeOverride(
        stream_verify=stream_verify,
        generation_kwargs=generation_kwargs,
        overrides=overrides,
    )
    tree = transformer.visit(tree)
    if transformer.replacements == 0:
        raise ValueError(
            f"AISBench model config {source_path} has no VLLMCustomAPIChat type to override"
        )
    _add_protocol_import(tree)

    config_dir_name = _validate_config_name(config_dir_name, "config_dir_name")
    custom_dir = Path(output_file_path) / config_dir_name / "models"
    custom_dir.mkdir(parents=True, exist_ok=True)
    custom_config = custom_dir / f"xtask_protocol_{api_type}.py"
    custom_config.write_text(
        ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8"
    )
    return f"xtask_protocol_{api_type}"


def create_dataset_config_override(
    dataset_conf,
    dataset_path,
    output_file_path,
    infer_mode=None,
    use_xtask_last_inferencer=False,
    config_dir_name="aisbench_configs",
):
    """Create an isolated AISBench dataset config for per-case overrides."""
    config_name, config_suffix = _split_dataset_config_reference(dataset_conf)
    if not dataset_path and not infer_mode:
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
    """Make xTask compatibility modules importable in AISBench subprocesses."""
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
    if "acc" in task_config.get("metrics") and accuracy_cfg.get("dataset_conf"):
        dataset_path = accuracy_cfg.get("dataset_path") or task_config.get(
            "dataset_path", ""
        )
        output_len = accuracy_cfg.get("output_len", 1024)
        request_rate = accuracy_cfg.get("request_rate", 0)
        batch_size = accuracy_cfg.get("batch_size", 32)
        temperature = accuracy_cfg.get("temperature", 0)
        top_k = accuracy_cfg.get("top_k", -1)
        top_p = accuracy_cfg.get("top_p", 0.95)
        seed = accuracy_cfg.get("seed", 1024)
        num_warmups = accuracy_cfg.get("num_warmups", 0)
        repetition_penalty = accuracy_cfg.get("repetition_penalty", 1)

        api_type = accuracy_cfg.get("api_type", "vllm_api_general_chat")
        api_type_file, model_config_name, shell_content = get_api_type_file(api_type, shell_content)

        # Step 3: 修改配置文件中的核心参数（使用sed命令）
        shell_content.append("\n# 3. 修改vllm_api配置文件$CONFIG_FILE参数")
        # 替换API地址
        shell_content.append(f'sed -i.bak "s/host_ip\s*=\s*.*,/host_ip=\'{api_address}\',/g" {api_type_file}')
        # 替换API端口
        shell_content.append(f'sed -i "s/host_port\s*=\s*.*,/host_port={api_port},/g" {api_type_file}')
        # 替换模型名称
        shell_content.append(f'sed -i "s/model\s*=\s*.*,/model=\'{model_name}\',/g" {api_type_file}')
        # 替换模型路径
        shell_content.append(
            f'''sed -i -E "s#^([[:space:]]*)path[[:space:]]*=.*,#\\1path='{model_path_inside}',#" {api_type_file}'''
        )

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
        generation_kwargs = accuracy_cfg.get('generation_kwargs', {})
        if generation_kwargs:  # 如果有需要更新的参数
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
        # 替换API地址
        shell_content.append(f'sed -i.bak "s/host_ip\s*=\s*.*,/host_ip=\'{api_address}\',/g" {api_type_file}')
        # 替换API端口
        shell_content.append(f'sed -i "s/host_port\s*=\s*.*,/host_port={api_port},/g" {api_type_file}')
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

    # Step 6: 生成协议一致性/受限解码测试命令（受全局开关控制）
    if "protocol" in task_config.get("metrics", ""):
        protocol_cfg = (benchmark_dict or {}).get("protocol") or {}
        if not is_protocol_test_enabled():
            shell_content.append("\n# 6. 协议一致性测试：全局开关 enable_protocol_test 未开启，跳过")
            shell_content.append(
                'echo "警告：enable_protocol_test 全局开关未开启（config/xTask_config.json），跳过协议一致性测试"'
            )
        elif not protocol_cfg:
            shell_content.append("\n# 6. 协议一致性测试：未配置 benchmark.protocol 段，跳过")
        else:
            shell_content.append("\n# 6. 执行协议一致性/受限解码测试")
            append_xtask_pythonpath(shell_content)

            stream_verify = protocol_cfg.get("stream_verify", False)
            protocol_api_type = protocol_cfg.get("api_type", "vllm_api_general_chat")

            # 6.1 生成协议感知模型配置（替换 VLLMCustomAPIChat → ProtocolAwareAPIChat）
            # 所有参数（host_ip/host_port/model/path/max_out_len/batch_size/
            # request_rate/temperature/generation_kwargs/stream_verify）在 AST
            # 阶段直接烘焙进配置。ast.unparse() 生成单行格式，sed 的贪婪 .* 匹配
            # 会把 host_ip 之后的所有字段吃掉，因此协议模型配置完全不使用 sed。
            protocol_generation_kwargs = protocol_cfg.get("generation_kwargs", {})
            protocol_output_len = protocol_cfg.get("output_len", 4096)
            protocol_batch_size = protocol_cfg.get("batch_size", 8)
            protocol_request_rate = protocol_cfg.get("request_rate", 0)
            protocol_temperature = protocol_cfg.get("temperature", 0)
            protocol_overrides = {
                "host_ip": api_address,
                "host_port": api_port,
                "model": model_name,
                "path": model_path_inside,
                "max_out_len": protocol_output_len,
                "batch_size": protocol_batch_size,
                "request_rate": protocol_request_rate,
                "temperature": protocol_temperature,
            }
            try:
                protocol_model_name = create_protocol_model_config(
                    protocol_api_type,
                    output_file_path,
                    stream_verify=stream_verify,
                    config_dir_name=config_dir_name,
                    generation_kwargs=protocol_generation_kwargs,
                    overrides=protocol_overrides,
                )
            except (FileNotFoundError, ValueError) as exc:
                shell_content.append(f'echo "错误：生成协议模型配置失败 - {exc}" >&2')
                shell_content.append("exit 1")
                protocol_model_name = None

            if protocol_model_name:
                protocol_model_file = (
                    f"{output_file_path_incontainer}/{config_dir_name}/models/"
                    f"{protocol_model_name}.py"
                )
                # 6.2 连接参数已在 AST 阶段烘焙进配置文件，无需 sed 替换

                # 6.4 数据集配置（protocol_constrained_gen + 路径覆写）
                protocol_dataset_path = protocol_cfg.get("dataset_path") or task_config.get(
                    "dataset_path", ""
                )
                protocol_dataset_conf = protocol_cfg.get(
                    "dataset_conf", "protocol_constrained_gen"
                )
                protocol_dataset_conf = create_dataset_config_override(
                    protocol_dataset_conf,
                    protocol_dataset_path,
                    output_file_path,
                    config_dir_name=config_dir_name,
                )
                append_dataset_path_check(shell_content, protocol_dataset_path)

                # 6.5 执行协议测试
                protocol_work_dir = f"{output_file_path_incontainer}/protocol_results"
                num_prompts = protocol_cfg.get("num_prompts", "")
                if num_prompts and isinstance(num_prompts, int):
                    protocol_num_prompts_str = f"--num-prompts {num_prompts} "
                else:
                    protocol_num_prompts_str = ""
                protocol_cmd = (
                    f'ais_bench --config-dir "$AIS_BENCH_CONFIG_DIR" '
                    f'--models {protocol_model_name} '
                    f'--datasets {protocol_dataset_conf} '
                    f'{protocol_num_prompts_str}'
                    f'--mode all '
                    f'--work-dir {protocol_work_dir}'
                )
                shell_content.append(f'echo "执行协议测试命令：{protocol_cmd}"')
                append_checked_command(shell_content, protocol_cmd, "AISBench协议测试执行失败")
                shell_content.append(f'echo "协议测试完成，结果保存至 {protocol_work_dir}"')
                shell_content.append(f'cp {protocol_model_file} {output_file_path_incontainer}/')

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
