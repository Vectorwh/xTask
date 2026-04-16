import sys
import logging
import traceback

from xtask.utils.logger import init_logger
from xtask.utils.arg_parser import ArgumentChecker
from xtask.utils.config_parser import get_xtask_config_data
from xtask.utils.openme import OpenMeClient
from xtask.tasks.task_parser import get_tasks
from xtask.core.scheduler import run_scheduler
from xtask.core.result_processor import verify_result, write_result


def main():
    """Main function for command line testing"""
    try:
        init_logger()
        checker = ArgumentChecker()
        openme_client = OpenMeClient("v0.9.0", "A3-ARM", "11.87.191.101")

        # Parse command line arguments
        args = checker.parse_arguments()
        args_result = checker.validate_arguments(args)
        checker.print_validation_result(args_result)
        xtask_config = get_xtask_config_data()
        logging.debug("args_result is: %s"% args_result)
        if args_result.is_valid:
            config_para = {}
            tasks = get_tasks(args, xtask_config, config_para)
            keys_to_process = list(tasks.keys())

            index = 0
            while index < len(keys_to_process):
                current_task_key = keys_to_process[index]
                index += 1
                logging.info(
                    f"======== Processing task [{index}/{len(keys_to_process)}]: {current_task_key} ========")

                try:
                    # 1. 检查任务是否存在
                    if current_task_key not in tasks:
                        logging.warning(f"Task key '{current_task_key}' not found in tasks dictionary. Skipping.")
                        continue

                    # 2. 执行任务
                    logging.info(f"Running scheduler for task: {current_task_key} ...")
                    result = run_scheduler(current_task_key, tasks[current_task_key], args)
                    openme_client.AddCaseResult(result)

                    # 调用封装好的自动评估函数, To do: 待合入

                except Exception as e:
                    logging.error(f"Failed to run scheduler for task '{current_task_key}': {str(e)}")
                    traceback.print_exc()
            # 发送京me通报
            # openme_client.SendJUEMsgCard()
        else:
            sys.exit(1)
    except Exception as e:
        logging.error(f"Argument validation failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
