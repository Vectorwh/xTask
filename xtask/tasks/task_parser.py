import os
import inspect
import logging

from xtask.tasks.base_task import Task


def init_task_path(args):
    if args.template:
        task_file_path = os.path.join(os.getcwd(), "outputs", f"{args.template.lower()}")
    elif args.case_name:
        task_file_path = os.path.join(os.getcwd(), "outputs", f"{args.case_name.lower()}")
    elif args.model_name:
        task_file_path = os.path.join(os.getcwd(), "outputs", 
                                      f"{args.model_name.lower()}_{args.framework.lower()}")
    else:
        logging.error("Must provide one of: --template, --case_name, or --model_name")
        return None
    os.makedirs(task_file_path, exist_ok=True)
    return task_file_path


def gen_task_file_by_template(template_name: str, task_file_path: str):
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


def gen_task_file_by_casename(case_name: str, task_file_path: str):
    """根据案例名称生成任务文件"""
    try:
        config = Task.create_from_case(case_name)
        if config is None:
            return False, []
        
        # 保存到任务目录
        output_file = os.path.join(task_file_path, "case.yaml")
        success = config.save_to_yaml(output_file)
        
        return success, [output_file] if success else []
    
    except Exception as e:
        logging.error(f"Error generating task from case: {e}")
        return False, []


def gen_task_file_by_args(args, task_file_path: str):
    """根据命令行参数生成任务文件"""
    try:
        config = Task.create_from_args(args)
        
        # 保存到任务目录
        output_file = os.path.join(task_file_path, f"{args.model_name}_{args.framework}.yaml")
        success = config.save_to_yaml(output_file)
        
        return success, [output_file] if success else []
    
    except Exception as e:
        logging.error(f"Error generating task from args: {e}")
        return False, []


def gen_task_file(args, task_file_path: str):
    ret = False
    tasks = list()

    if args.template:
        ret, tasks = gen_task_file_by_template(args.template, task_file_path)
    elif args.case_name:
        ret, tasks = gen_task_file_by_casename(args.case_name, task_file_path)
    elif args.model_name:
        ret, tasks = gen_task_file_by_args(args, task_file_path)
    else:
        logging.error("Must provide one of: --template, --case_name, or --model_name")
        return None

    return ret, tasks


def get_tasks(args):
    logging.info("entry function: %s"% inspect.currentframe().f_code.co_name)
    tasks = list()

    # init task file path
    task_file_path = init_task_path(args)
    logging.debug("task_file_path is: %s"% task_file_path)

    # gen task file
    ret, tasks = gen_task_file(args, task_file_path)
    if not ret:
        logging.error("gen task file failed.")

    return tasks
