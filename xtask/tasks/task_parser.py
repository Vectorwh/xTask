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
        return None
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
            return False, []
        
        # 保存到任务目录
        output_file = os.path.join(task_file_path, "case.yaml")
        success = config.save_to_yaml(output_file)
        
        return success, [output_file] if success else []
    
    except Exception as e:
        logging.error(f"Error generating task from template: {e}")
        return False, []


def gen_task_file_by_casename(case_name: str, task_file_path: str, xtask_config: dict):
    """根据用例名称生成任务文件"""
    try:        
        # 从tests目录查找 case.yaml
        case_dir = os.path.join(os.getcwd(), "tests")
        case_file = find_case_file(case_name, case_dir)
        success = Task.check_case_yaml(case_file)
        
        return success, [case_file] if success else []
    
    except Exception as e:
        logging.error(f"Error generating task from case: {e}")
        return False, []

def gen_task_file_by_casegroup(case_group: str, task_file_path: str, xtask_config: dict):
    """根据用例名称生成任务文件"""
    try:
        case_name_list = list()
        tasks = list()
        case_name_list = get_case_list_from_group(case_group)

        if case_name_list:
            for case_name in case_name_list:
                case_file_path = os.path.join(os.getcwd(), "outputs", case_name)
                os.makedirs(case_file_path, exist_ok=True)
                ret, task = gen_task_file_by_casename(case_name, case_file_path, xtask_config)
                if ret:
                    tasks.extend(task)
                else:
                    logging.error(f"parser case name: {case_name} from case group: {case_group} failed.")
            return True, tasks
        else:
            logging.error(f"get case name: {case_name} from case group: {case_group} failed.")
    except Exception as e:
        logging.error(f"Error generating task from case group: {e}")
        return False, []


def gen_task_file_by_args(args, task_file_path: str, xtask_config: dict):
    """根据命令行参数生成任务文件"""
    try:
        config = Task.create_from_args(args, xtask_config)
        
        # 保存到任务目录
        output_file = os.path.join(task_file_path, f"{config.case_name}.yaml")
        success = config.save_to_yaml(output_file)
        
        return success, [output_file] if success else []
    
    except Exception as e:
        logging.error(f"Error generating task from args: {e}")
        return False, []


def gen_task_file(args, task_output_path: str, xtask_config: dict):
    ret = False
    tasks = list()

    if args.template:
        ret, tasks = gen_task_file_by_template(args.template, task_output_path, xtask_config)
    elif args.case_name:
        ret, tasks = gen_task_file_by_casename(args.case_name, task_output_path, xtask_config)
    elif args.case_group:
        ret, tasks = gen_task_file_by_casegroup(args.case_group, task_output_path, xtask_config)
    elif args.model_name:
        ret, tasks = gen_task_file_by_args(args, task_output_path, xtask_config)
    else:
        logging.error("Must provide one of: --template, --case_name, --case_group or --model_name")
        return None

    return ret, tasks


def get_tasks(args, xtask_config):
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)
    tasks = list()

    # init task file path
    task_output_path = init_task_path(args)
    logging.debug("task_file_path is: %s"% task_output_path)

    # gen task file
    ret, tasks = gen_task_file(args, task_output_path, xtask_config)
    if not ret:
        logging.error("gen task file failed.")

    return tasks
