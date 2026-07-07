import os
import inspect
import logging
from xtask.tasks.base_task import Task, get_case_name_by_args

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
    case_name_list = list()
    case_group_file = os.path.join(os.getcwd(), f"tests/case_group/{case_group}")
    if os.path.exists(case_group_file):
        with open(case_group_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取所有行到列表
            for line in lines:
                if len(line.strip()):
                    case_name_list.append(line.strip('\n').strip())
    else:
        logging.error(f"case group file: {case_group_file} is not exist.")
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
    
    # 要查找的文件名
    target_filename = f"{case_name}.yaml"
    found_files = []
    
    # 递归遍历目录
    for root, dirs, files in os.walk(case_dir):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # 在当前目录查找目标文件
        if target_filename in files:
            file_path = os.path.join(root, target_filename)
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

        #删除上一次的case_name.json文件
        try:
            os.remove(os.path.join(task_file_path, (case_name + ".json")))
            logging.info("remove last casename.json success.")
        except FileNotFoundError:
            logging.info("casename.json is not exist.")
        except PermissionError:
            logging.info("no right to remove casename.json.")

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

    # 动态生成组合
    for combo in itertools.product(*ranges):
        yield dict(zip(keys, combo))

import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

def _update_config_by_path(
        config: Dict[str, Any],
        key_path: str,
        value: Any
) -> None:
    """
    根据键路径更新配置字典

    """
    if config.get("benchmark").get("perf") is not None and key_path in config.get("benchmark").get("perf"):
        config["benchmark"]["perf"].update({key_path: value})

        # 修改bs关联配置
        if key_path == "batch_size":
            config["benchmark"]["perf"].update({"num_prompts": (value * 4)})
            # config["benchmark"]["perf"].update({"request_rate": value})

    elif config.get("benchmark").get("acc") is not None and key_path in config.get("benchmark").get("acc"):
        config["benchmark"]["acc"].update({key_path: value})

    elif config.get("serving_extensiable_param") is not None and key_path in config.get("serving_extensiable_param"):
        config["serving_extensiable_param"].update({key_path: value})

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
    """处理评估测试用例"""
    try:
        # 查找并加载基础配置文件
        basecase_dir = Path.cwd() / "tests"
        basecase_file = find_case_file(case_name, str(basecase_dir))

        with open(basecase_file, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)

        if base_config is None:
            logging.error(f"基础配置文件为空: {basecase_file}")
            return False

        # 获取评估测试配置
        assess_test = base_config.get("assess_test_param", {})
        if not assess_test:
            logging.warning(f"用例 {case_name} 没有评估测试配置")
            return False

        # 生成动态参数组合
        for param_dict in dynamic_loops_from_dict(assess_test):
            config_copy = base_config.copy()
            param_parts = []

            for param_name, param_value in param_dict.items():
                # 更新配置
                _update_config_by_path(config_copy, param_name, param_value)
                spec_letters = get_specific_letters(param_name)
                param_parts.append(f"{spec_letters}-{param_value}")
                #param_parts.append(f"{param_name}-{param_value}")

            # 生成新用例名称
            param_suffix = "_".join(param_parts)
            new_case_name = (f"{case_name}_{param_suffix}").lower()
            config_copy.update({"case_name": new_case_name})
            task_new_file_path = os.path.join(task_file_path, new_case_name)
            os.makedirs(task_new_file_path, exist_ok=True)

            new_case_dir = Path(task_new_file_path) / f"{new_case_name}.yaml"
            #task_file_path = os.path.join(task_file_path, new_case_name.lower())
            #case_path = Path(task_file_path) / new_case_name

            # 保存配置文件
            with open(new_case_dir, 'w', encoding='utf-8') as f:
                yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)

            tasks[str(new_case_dir)] = task_new_file_path

            #删除上一次的case_name.json文件
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

    return tasks
