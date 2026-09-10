import os
import inspect
import logging
import copy
import re
from xtask.tasks.base_task import Task, get_case_name_by_args
from xtask.tasks.topology_search import (
    ASSESS_TOPOLOGY_PARAMETER,
    apply_assess_topology,
    normalize_assess_topologies,
    resolved_topology_metadata,
)

def init_task_path(args):
    if args.template:
        task_file_path = os.path.join(os.getcwd(), "outputs", args.template.lower())
    elif args.case_name:
        task_file_path = os.path.join(os.getcwd(), "outputs", args.case_name.lower())
    elif args.case_group:
        task_file_path = os.path.join(os.getcwd(), "outputs", args.case_group.lower())
    elif args.model_name:
        temp_case_name = get_case_name_by_args(args)
        task_file_path = os.path.join(os.getcwd(), "outputs", temp_case_name)
    else:
        logging.error("Must provide one of: --template, --case_name, or --model_name")
        return None
    os.makedirs(task_file_path, exist_ok=True)
    return task_file_path

def get_case_list_from_group(case_group):
    """Resolve case group members.

    If ``tests/case_group/<case_group>`` exists, treat it as a group file and
    return the list of case names inside. Otherwise, fall back to treating
    ``case_group`` as a single case name (the initial YAML), so users can run
    ``auto_assess`` / ``para_set`` without creating a separate group file.
    """
    case_name_list = list()
    case_group_file = os.path.join(os.getcwd(), f"tests/case_group/{case_group}")
    if os.path.exists(case_group_file):
        with open(case_group_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取所有行到列表
            for line in lines:
                if len(line.strip()):
                    case_name_list.append(line.strip('\n').strip())
        return case_name_list

    # Fallback: case_group is the initial case name (without .yaml extension).
    case_dir = os.path.join(os.getcwd(), "tests")
    try:
        case_file = find_case_file(case_group, case_dir)
    except Exception:
        case_file = None
    if case_file:
        logging.info(
            "Case group file not found; treating '%s' as a single-case group (case file: %s)",
            case_group,
            case_file,
        )
        return [case_group]

    logging.error(
        "case group file: %s is not exist and no case YAML found for '%s'.",
        case_group_file,
        case_group,
    )
    return case_name_list


def find_case_file(case_name, case_dir):
    """
    Find test case file in specified directory
    
    Args:
        case_name (str): Test case name (without .yaml extension)
        case_dir (str): Root directory to search
        
    Returns:
        str: Full path of the found test case file, returns None if not found
        
    Raises:
        FileExistsError: When multiple test case files with the same name are found
        FileNotFoundError: When the specified search directory does not exist
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"case_dir: {case_dir} is not exist.")
    
    if not os.path.isdir(case_dir):
        raise NotADirectoryError(f"case_dir: {case_dir} is not dir.")
    
    # case_name may include a directory prefix used to disambiguate cases.
    # Match the basename while requiring the normalized suffix path to match.
    normalized_case_name = os.path.normpath(case_name).lstrip(os.sep)
    target_filename = f"{os.path.basename(normalized_case_name)}.yaml"
    target_suffix = f"{normalized_case_name}.yaml"
    found_files = []
    
    # 递归遍历目录
    for root, dirs, files in os.walk(case_dir):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # 在当前目录查找目标文件
        if target_filename in files:
            file_path = os.path.join(root, target_filename)
            relative_path = os.path.relpath(file_path, case_dir)
            if os.sep not in normalized_case_name or relative_path.endswith(target_suffix):
                found_files.append(file_path)
    
    # 根据查找结果处理
    if len(found_files) == 0:
        return None
    elif len(found_files) == 1:
        return found_files[0]
    else:
        # 找到多个文件，抛出异常
        error_msg = f"the case name is not unique:\n"
        for file_path in found_files:
            error_msg += f"  - {file_path}\n"
        raise FileExistsError(error_msg)
    

def gen_task_file_by_template(template_name: str, task_file_path: str, xtask_config: dict):
    """根据模板生成任务文件"""
    try:
        config = Task.create_from_template(template_name)
        if config is None:
            return False, {}
        
        # 保存到任务目录
        output_file = os.path.join(task_file_path, "case.yaml")
        success = config.save_to_yaml(output_file)

        return success, {output_file:task_file_path} if success else {}

    except Exception as e:
        logging.error(f"Error generating task from template: {e}")
        return False, {}


def gen_task_file_by_casename(case_name: str, task_file_path: str, xtask_config: dict):
    """根据用例名称生成任务文件"""
    try:        
        # 从tests目录查找 case.yaml
        case_dir = os.path.join(os.getcwd(), "tests")
        case_file = find_case_file(case_name, case_dir)
        success = Task.check_case_yaml(case_file)

        return success, {case_file:task_file_path} if success else {}
    
    except Exception as e:
        logging.error(f"Error generating task from case: {e}")
        return False, {}


def dynamic_loops_from_dict(data_dict, key_patterns=None):
    """
    从字典中提取值进行动态循环

    参数:
        data_dict: 包含列表值的字典
        key_patterns: 要使用的键列表，如果为None则使用所有列表类型的值

    返回:
        生成器，产生包含各种组合的字典
    """
    import itertools

    # 参数验证
    if not isinstance(data_dict, dict):
        raise TypeError("data_dict 必须是字典类型")

    # 提取键和值
    if key_patterns is None:
        keys = [k for k, v in data_dict.items() if isinstance(v, (list, tuple, set))]
        ranges = [data_dict[k] for k in keys]
    else:
        # 验证指定的键都存在
        missing_keys = [k for k in key_patterns if k not in data_dict]
        if missing_keys:
            raise KeyError(f"字典中缺少以下键: {missing_keys}")

        keys = key_patterns
        ranges = [data_dict[key] for key in key_patterns]

    # 检查是否有可迭代对象
    if not ranges:
        print("警告: 字典中没有列表/元组/集合类型的值")
        return
        yield  # 空生成器

    # 集合没有稳定顺序，参数组合生成必须确定性排序。
    stable_ranges = [
        sorted(values, key=lambda item: repr(item)) if isinstance(values, set) else values
        for values in ranges
    ]

    # 动态生成组合
    for combo in itertools.product(*stable_ranges):
        yield dict(zip(keys, combo))

import yaml
import logging
from typing import Any, Dict, List, Tuple
from pathlib import Path

_EXTENSION_PARAM_KEYS = (
    "serving_extensiable_param",
    "serving_extensible_param",
)
_DEPLOYMENT_SELECTOR = re.compile(
    r"^(?P<collection>instances|pd_instances)"
    r"\[(?P<selector>\*|\d+|(?:name|role)=[^\]]+)\]$"
)
def _set_existing_mapping_path(mapping: Dict[str, Any], parts, value, key_path: str):
    """Set an existing mapping path and reject misspelled parameters."""
    if not parts:
        raise ValueError(f"Empty parameter path: {key_path}")

    current = mapping
    normalized_parts = list(parts)
    if (
        normalized_parts[0] == "serving_extensible_param"
        and "serving_extensible_param" not in current
        and "serving_extensiable_param" in current
    ):
        normalized_parts[0] = "serving_extensiable_param"

    for part in normalized_parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unsupported parameter path: {key_path}")
        current = current[part]

    final = normalized_parts[-1]
    if not isinstance(current, dict) or final not in current:
        raise KeyError(f"Unsupported parameter path: {key_path}")
    current[final] = value


def _select_deployment_items(
    config: Dict[str, Any], collection: str, selector: str, key_path: str
):
    deployment = config.get("deployment") or {}
    items = deployment.get(collection) or []
    if not items:
        raise KeyError(
            f"Parameter path requires deployment.{collection}: {key_path}"
        )

    if selector == "*":
        selected = list(items)
    elif selector.isdigit():
        index = int(selector)
        if index >= len(items):
            raise IndexError(f"Deployment item index out of range: {key_path}")
        selected = [items[index]]
    else:
        field, expected = selector.split("=", 1)
        selected = [
            item
            for item in items
            if str(item.get(field)) == expected
        ]
        if not selected:
            raise KeyError(
                f"No deployment.{collection} item matches {selector}: {key_path}"
            )
    return selected


def _update_explicit_parameter_path(
    config: Dict[str, Any], key_path: str, value: Any
) -> list:
    """Update an explicit dotted path, including instance selectors."""
    parts = key_path.split(".")
    if len(parts) < 2:
        raise KeyError(f"Unsupported parameter path: {key_path}")

    if parts[0] == "deployment" and len(parts) >= 3:
        selector_match = _DEPLOYMENT_SELECTOR.match(parts[1])
        if selector_match:
            collection = selector_match.group("collection")
            selector = selector_match.group("selector")
            items = _select_deployment_items(
                config, collection, selector, key_path
            )
            target_parts = parts[2:]
            changed = []
            for item in items:
                _set_existing_mapping_path(item, target_parts, value, key_path)
                changed.append(key_path)
            return changed

    _set_existing_mapping_path(config, parts, value, key_path)
    return [key_path]


def _apply_batch_size_defaults(config: Dict[str, Any], value: Any) -> None:
    """Keep the historical batch_size -> num_prompts coupling."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("batch_size must be numeric")
    benchmark = config.get("benchmark") or {}
    perf = benchmark.get("perf")
    if isinstance(perf, dict):
        perf["num_prompts"] = value * 4


def _update_config_by_path(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
    apply_derived: bool = True,
) -> list:
    """Update one assess parameter in ordinary or hierarchical deployments.

    Supported explicit forms include::

        benchmark.perf.batch_size
        serving_extensiable_param.max_seqs_per_batch
        deployment.instances[*].serving_extensiable_param.max_seqs_per_batch
        deployment.pd_instances[name=prefill-0].serving_extensiable_param.max_seqs_per_batch

    A bare name keeps the historical lookup order and updates matching
    instance-level serving parameters for backward compatibility.
    """
    if not isinstance(key_path, str) or not key_path.strip():
        raise ValueError("assess_test_param keys must be non-empty strings")
    key_path = key_path.strip()

    # Explicit dotted paths are strict: a typo must not silently generate an
    # unchanged candidate YAML.
    if "." in key_path or "instances[" in key_path:
        changed = _update_explicit_parameter_path(config, key_path, value)
        if (
            apply_derived
            and key_path == "benchmark.perf.batch_size"
        ):
            _apply_batch_size_defaults(config, value)
        return changed

    changed = []
    benchmark = config.get("benchmark") or {}
    perf = benchmark.get("perf")
    acc = benchmark.get("acc")

    # Preserve the old precedence: perf, then acc, then top-level serving
    # extension. Instance-level matching is intentionally applied afterward.
    if isinstance(perf, dict) and key_path in perf:
        perf[key_path] = value
        changed.append(f"benchmark.perf.{key_path}")
        if apply_derived and key_path == "batch_size":
            _apply_batch_size_defaults(config, value)
    elif isinstance(acc, dict) and key_path in acc:
        acc[key_path] = value
        changed.append(f"benchmark.acc.{key_path}")
    else:
        top_extension = None
        for extension_key in _EXTENSION_PARAM_KEYS:
            extension = config.get(extension_key)
            if isinstance(extension, dict) and key_path in extension:
                top_extension = extension
                break
        if top_extension is not None:
            top_extension[key_path] = value
            changed.append(f"{extension_key}.{key_path}")

    # A bare serving parameter applies to every flat non-PD instance or
    # hierarchical logical PD instance that already declares the key.
    deployment = config.get("deployment") or {}
    for collection in ("instances", "pd_instances"):
        for item in deployment.get(collection) or []:
            for extension_key in _EXTENSION_PARAM_KEYS:
                extension = item.get(extension_key)
                if isinstance(extension, dict) and key_path in extension:
                    extension[key_path] = value
                    changed.append(
                        f"deployment.{collection}[*].{extension_key}.{key_path}"
                    )
                    break

    if not changed:
        raise KeyError(f"Unsupported assess_test_param: {key_path}")
    return changed


def _normalize_assess_parameters(assess_test: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate ``assess_test_param`` for ``para_set``.

    The original format is a flat mapping whose values are lists, for
    example ``{"batch_size": [1, 2]}``.  Nested mappings are also accepted
    and flattened into dotted paths, which makes a multi-host configuration
    readable without weakening path validation::

        assess_test_param:
          deployment:
            pd_instances[name=prefill-0]:
              serving_extensiable_param:
                max_seqs_per_batch: [8, 16]

    Scalar leaves are rejected deliberately.  A scalar in this section is
    almost always a misplaced normal configuration value; silently ignoring
    it would produce fewer candidates than the user requested.
    """
    if assess_test is None:
        return {}
    if not isinstance(assess_test, dict):
        raise TypeError("assess_test_param must be a mapping")

    normalized: Dict[str, Any] = {}

    def visit(mapping: Dict[str, Any], prefix: str = "") -> None:
        for raw_key, raw_value in mapping.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ValueError("assess_test_param keys must be non-empty strings")

            key = raw_key.strip()
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(raw_value, dict):
                if not raw_value:
                    raise ValueError(f"Empty assess_test_param mapping: {path}")
                visit(raw_value, path)
                continue

            if not isinstance(raw_value, (list, tuple, set)):
                raise TypeError(
                    f"Values for assess_test_param '{path}' must be a list, tuple, or set"
                )

            values = list(raw_value)
            if not values:
                raise ValueError(f"Parameter value list is empty: {path}")
            if isinstance(raw_value, set):
                # Keep the generated YAMLs and directory names deterministic.
                values = sorted(values, key=lambda item: repr(item))

            if path in normalized:
                raise ValueError(f"Duplicate assess_test_param path: {path}")
            normalized[path] = values

    visit(assess_test)
    return normalized


def _safe_parameter_value(value: Any) -> str:
    """Return a deterministic, filesystem-safe representation of a value."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "none"
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
        if isinstance(value, set):
            values = sorted(values, key=lambda item: repr(item))
        text = "-".join(_safe_parameter_value(item) for item in values) or "empty"
    elif isinstance(value, dict):
        # Dict-valued candidates are unusual, but representing them via a
        # sorted repr keeps the helper total and avoids memory-address based
        # names from arbitrary object reprs in the common case.
        text = repr(sorted(value.items(), key=lambda item: str(item[0])))
    else:
        text = str(value)

    if text.startswith("-"):
        text = f"neg-{text[1:]}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-_.").lower() or "value"


def _parameter_label(parameter_name: str) -> str:
    """Build a readable label without putting path punctuation in a name."""
    # Keep the historical short name for the original flat syntax.  In
    # particular, ``batch_size`` continues to generate ``bs-<value>``.
    if "." not in parameter_name and "[" not in parameter_name:
        return get_specific_letters(parameter_name).lower()

    label = parameter_name
    label = label.replace("deployment.instances[*]", "all_instances")
    label = label.replace("deployment.pd_instances[*]", "all_pd_instances")
    label = re.sub(
        r"deployment\.instances\[(?:role|name)=([^\]]+)\]",
        r"instance_\1",
        label,
    )
    label = re.sub(
        r"deployment\.pd_instances\[(?:role|name)=([^\]]+)\]",
        r"pd_instance_\1",
        label,
    )
    label = re.sub(r"deployment\.instances\[(\d+)\]", r"instance_\1", label)
    label = re.sub(
        r"deployment\.pd_instances\[(\d+)\]", r"pd_instance_\1", label
    )
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return label or "param"


def _build_parameter_labels(param_dict: Dict[str, Any]) -> List[str]:
    """Build safe, deterministic ``key-value`` labels for one combination."""
    return [
        f"{_parameter_label(parameter_name)}-{_safe_parameter_value(parameter_value)}"
        for parameter_name, parameter_value in param_dict.items()
    ]

def get_specific_letters(para_name: str):
    # 使用 split('_') 按下划线分割字符串
    parts = para_name.split('_')

    # 检查分割后的部分数量是否足够（至少要有两部分）
    if len(parts) < 2:
        return para_name  # 如果没有下划线或格式不符，返回原字符串或做其他处理

    spec_letter = ""
    for part in parts:
        spec_letter += part[0]

    return spec_letter

def gen_group_paraset_task(
        case_name: str,
        task_file_path: str,
        tasks: Dict[str, str]
) -> bool:
    """Expand one case into a deterministic set of parameter combinations.

    This function only creates candidate YAML files.  Every candidate is
    later handed to the existing scheduler, so ordinary single-host,
    multi-host, and PD-disaggregated execution paths remain unchanged.
    """
    try:
        # 查找并加载基础配置文件
        basecase_dir = Path.cwd() / "tests"
        basecase_file = find_case_file(case_name, str(basecase_dir))

        if basecase_file is None:
            raise FileNotFoundError(f"case file not found: {case_name}")

        with open(basecase_file, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)

        if not isinstance(base_config, dict):
            logging.error(f"基础配置文件为空: {basecase_file}")
            return False

        from xtask.core.service_group import validate_pd_deployment_authoring

        validate_pd_deployment_authoring(base_config)

        from xtask.core.external_service import validate_external_service_config

        try:
            validate_external_service_config(base_config)
        except ValueError as exc:
            logging.error("%s", exc)
            return False

        # 获取评估测试配置。原始配置单独保留，用于写入每个参数组合的
        # 说明文件；生成组合时使用规范化后的路径映射。
        raw_assess_test = base_config.get("assess_test_param", {})
        assess_test = _normalize_assess_parameters(raw_assess_test)
        if not assess_test:
            logging.warning(f"用例 {case_name} 没有评估测试配置")
            return False
        assess_topologies = normalize_assess_topologies(base_config, assess_test)

        # 先完整构造候选，确保未知路径、重复 case_name 等问题在写盘
        # 前暴露，避免只生成半组 YAML。
        generated_candidates = []
        for param_dict in dynamic_loops_from_dict(assess_test):
            # 每个组合都必须从原始配置深拷贝，避免上一个组合的实例级
            # 参数、benchmark 联动字段污染下一个组合。
            config_copy = copy.deepcopy(base_config)
            changed_paths = []
            changed_paths_by_param = {}

            topology_name = param_dict.get(ASSESS_TOPOLOGY_PARAMETER)
            if topology_name is not None:
                changed_for_topology = apply_assess_topology(
                    config_copy, topology_name, assess_topologies
                )
                changed_paths.extend(changed_for_topology)
                changed_paths_by_param[ASSESS_TOPOLOGY_PARAMETER] = (
                    changed_for_topology
                )

            for param_name, param_value in param_dict.items():
                if param_name == ASSESS_TOPOLOGY_PARAMETER:
                    continue
                changed_for_param = _update_config_by_path(
                    config_copy,
                    param_name,
                    param_value,
                    apply_derived=False,
                )
                changed_paths.extend(changed_for_param)
                changed_paths_by_param[param_name] = changed_for_param

            # Preserve the historical batch_size -> num_prompts coupling, but
            # do it only after all explicit parameters have been applied.  An
            # explicit num_prompts in the same combination always wins.
            if "benchmark.perf.batch_size" in changed_paths:
                has_explicit_num_prompts = "benchmark.perf.num_prompts" in changed_paths
                batch_values = [
                    param_dict[param_name]
                    for param_name, changed_for_param in changed_paths_by_param.items()
                    if "benchmark.perf.batch_size" in changed_for_param
                ]
                if not has_explicit_num_prompts:
                    if len(batch_values) != 1:
                        raise ValueError(
                            "benchmark.perf.batch_size must be specified at most once"
                        )
                    _apply_batch_size_defaults(config_copy, batch_values[0])

            if topology_name is not None:
                from xtask.core.service_group import normalize_pd_deployment

                normalize_pd_deployment(copy.deepcopy(config_copy))

            # 参数组合使用短序号命名，避免把全部参数路径和值拼进目录名
            # 导致路径过长。序号从 1 开始，且与下面的说明文件一一对应。
            group_index = len(generated_candidates) + 1
            new_case_name = f"{case_name}_sg{group_index}".lower()
            config_copy.update({"case_name": new_case_name})
            generated_candidates.append(
                (
                    new_case_name,
                    config_copy,
                    copy.deepcopy(param_dict),
                )
            )

        generated_names = [name for name, _, _ in generated_candidates]
        if len(generated_names) != len(set(generated_names)):
            raise ValueError(
                f"Duplicate generated case_name(s) for {case_name}: {generated_names}"
            )

        existing_case_names = {
            Path(task_key).stem.lower() for task_key in tasks
        }
        conflicting_names = {
            name for name in generated_names if name in existing_case_names
        }
        if conflicting_names:
            raise ValueError(
                f"Generated case_name already exists in task list: {sorted(conflicting_names)}"
            )

        for new_case_name, config_copy, current_combination in generated_candidates:
            task_new_file_path = os.path.join(task_file_path, new_case_name)
            os.makedirs(task_new_file_path, exist_ok=True)
            new_case_dir = Path(task_new_file_path) / f"{new_case_name}.yaml"

            with open(new_case_dir, 'w', encoding='utf-8') as f:
                yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)

            # 只保存参数搜索相关信息，不把基础用例的模型、部署等配置
            # 复制进来。current_combination 与 sg 序号一一对应。
            assess_param_file = Path(task_new_file_path) / "assess_test_param.yaml"
            assess_param_data = {
                "current_combination": current_combination,
                "assess_test_param": copy.deepcopy(raw_assess_test),
            }
            resolved_topology = resolved_topology_metadata(
                current_combination, assess_topologies, config_copy
            )
            if resolved_topology is not None:
                assess_param_data["resolved_topology"] = resolved_topology
            with open(assess_param_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    assess_param_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            tasks[str(new_case_dir)] = task_new_file_path

            # 删除上一次的 case_name.json 文件（保留原有行为）。
            try:
                os.remove(os.path.join(str(task_new_file_path), (str(new_case_name) + ".json")))
                logging.info("remove last casename.json success.")
            except FileNotFoundError:
                logging.info("casename.json is not exist.")
            except PermissionError:
                logging.info("no right to remove casename.json.")            

        return True

    except FileNotFoundError as e:
        logging.error(f"找不到用例文件: {case_name} - {e}")
        return False
    except yaml.YAMLError as e:
        logging.error(f"YAML解析错误: {case_name} - {e}")
        return False
    except Exception as e:
        logging.error(f"处理评估用例时出错: {case_name} - {e}", exc_info=True)
        return False

def gen_group_autoassess_task(
        case_name: str,
        task_file_path: str,
        tasks: Dict[str, str],
        config_para: Dict
) -> bool:
    """
    处理评估测试用例：
    1. 加载基础配置。
    2. 根据 config_para 更新 batch_size (如果提供)。
    3. 生成新的配置文件并更新任务列表。
    """
    PARAM_NAME = "batch_size"

    try:
        # 1. 查找并加载基础配置文件
        basecase_dir = Path.cwd() / "tests"
        basecase_file = find_case_file(case_name, str(basecase_dir))

        with open(basecase_file, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)

        if not base_config:
            logging.error(f"基础配置文件为空或解析失败: {basecase_file}")
            return False

        # 2. 准备配置副本
        config_copy = base_config.copy()
        param_value = None

        # 3. 确定参数值并更新配置
        if config_para and PARAM_NAME in config_para:
            # 优先使用传入的参数
            param_value = config_para[PARAM_NAME]
            # 更新配置字典中的值
            _update_config_by_path(config_copy, PARAM_NAME, param_value)
        else:
            # 回退：从基础配置中获取默认值 (使用链式 get 防止 KeyError)
            param_value = base_config.get("benchmark", {}).get("perf", {}).get(PARAM_NAME)

            if param_value is None:
                logging.error(f"无法获取有效的 {PARAM_NAME}，请检查 config_para 或基础配置文件结构。")
                return False

        # 4. 公共逻辑：生成新用例名称、路径并保存
        new_case_name = f"{case_name}_bs-{param_value}".lower()
        config_copy["case_name"] = new_case_name

        # 使用 pathlib 规范化路径处理
        task_root_path = Path(task_file_path)
        new_task_dir = task_root_path / new_case_name
        new_task_dir.mkdir(parents=True, exist_ok=True)

        new_case_file = new_task_dir / f"{new_case_name}.yaml"

        # 保存配置文件
        with open(new_case_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)

        # 更新任务字典 (key: 文件路径, value: 目录路径)
        tasks[str(new_case_file)] = str(new_task_dir)

        #删除上一次的case_name.json文件
        try:
            os.remove(os.path.join(str(new_task_dir), (str(new_case_name) + ".json")))
            logging.info("remove last casename.json success.")
        except FileNotFoundError:
            logging.info("casename.json is not exist.")
        except PermissionError:
            logging.info("no right to remove casename.json.")

        logging.info(f"Successfully generated auto-assess task: {new_case_name}")
        return True

    except FileNotFoundError as e:
        logging.error(f"找不到用例文件: {case_name} - {e}")
    except yaml.YAMLError as e:
        logging.error(f"YAML解析错误: {case_name} - {e}")
    except Exception as e:
        logging.error(f"处理评估用例时出错: {case_name} - {e}", exc_info=True)

    return False

def gen_group_normal_task(
        case_name: str,
        task_file_path: str,
        xtask_config: Dict[str, Any],
        tasks: Dict[str, str]
) -> bool:
    """处理普通测试用例"""
    try:
        case_file_path = os.path.join(task_file_path, case_name)
        os.makedirs(case_file_path, exist_ok=True)

        ret, task = gen_task_file_by_casename(
            case_name,
            case_file_path,
            xtask_config
        )

        if not ret:
            return False

        # 合并任务字典
        if isinstance(task, dict):
            tasks.update(task)
        elif isinstance(task, list):
            # 假设任务列表中的每个元素是文件路径
            for task_item in task:
                tasks[task_item] = str(case_file_path)

        return True

    except Exception as e:
        logging.error(f"处理普通用例组出错: {case_name} - {e}", exc_info=True)
        return False

def gen_task_file_by_casegroup(
        case_group: str,
        task_file_path: str,
        xtask_config: Dict[str, Any],
        config_para: Dict,
        ass_test_type = "none"
) -> Tuple[bool, Dict[str, str]]:
    """
    根据用例组名称生成任务文件

    Args:
        case_group: 用例组名称
        task_file_path: 任务文件输出路径
        xtask_config: XTask配置字典
        ass_test: 是否进行评估测试

    Returns:
        Tuple[bool, Dict[str, str]]: (是否成功, 任务文件路径字典)
    """
    try:
        # 获取用例列表
        case_name_list = get_case_list_from_group(case_group)
        if not case_name_list:
            logging.error(f"获取用例组失败: {case_group}")
            return False, {}

        tasks = {}

        for case_name in case_name_list:
            if 'para_set' == ass_test_type:
                result = gen_group_paraset_task(
                    case_name,
                    task_file_path,
                    tasks
                )
            elif 'auto_assess' == ass_test_type:
                #当前自动评估测试，仅支持固定参数组合下并发摸高
                result = gen_group_autoassess_task(
                    case_name,
                    task_file_path,
                    tasks,
                    config_para
                )
            else:
                result = gen_group_normal_task(
                    case_name,
                    task_file_path,
                    xtask_config,
                    tasks
                )

            if not result:
                logging.error(f"处理用例失败: {case_name} (用例组: {case_group})")

        return True, tasks

    except Exception as e:
        logging.error(f"从用例组生成任务文件时出错: {e}", exc_info=True)
        return False, {}

def gen_task_file_by_args(args, task_file_path: str, xtask_config: dict):
    """根据命令行参数生成任务文件"""
    try:
        config = Task.create_from_args(args, xtask_config)
        
        # 保存到任务目录
        output_file = os.path.join(task_file_path, f"{config.case_name}.yaml")
        success = config.save_to_yaml(output_file)
        
        return success, {output_file:task_file_path} if success else {}
    
    except Exception as e:
        logging.error(f"Error generating task from args: {e}")
        return False, {}


def gen_task_file(args, task_output_path: str, xtask_config: dict, config_para: Dict):
    ret = False
    tasks = {}

    if args.template:
        ret, tasks = gen_task_file_by_template(args.template, task_output_path, xtask_config)
    elif args.case_name:
        ret, tasks = gen_task_file_by_casename(args.case_name, task_output_path, xtask_config)
    elif args.case_group:
        ret, tasks = gen_task_file_by_casegroup(args.case_group, task_output_path, xtask_config,
                                                config_para, args.assess_test_type)
    elif args.model_name:
        ret, tasks = gen_task_file_by_args(args, task_output_path, xtask_config)
    else:
        logging.error("Must provide one of: --template, --case_name, --case_group or --model_name")
        return None

    return ret, tasks


def get_tasks(args, xtask_config, config_para: Dict):
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)
    tasks = {}

    # init task file path
    task_output_path = init_task_path(args)
    logging.debug("task_file_path is: %s"% task_output_path)

    # gen task file
    ret, tasks = gen_task_file(args, task_output_path, xtask_config, config_para)
    if not ret:
        logging.error("gen task file failed.")
    elif getattr(args, "assess_test_type", None) == "para_set" and not tasks:
        logging.warning(
            "para_set produced no tasks; check that each case defines a non-empty "
            "assess_test_param mapping."
        )

    return tasks
