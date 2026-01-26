import sys
import logging
import traceback

from xtask.utils.logger import init_logger
from xtask.utils.arg_parser import ArgumentChecker
from xtask.utils.config_parser import get_xtask_config_data
from xtask.tasks.task_parser import get_tasks
from xtask.core.scheduler import run_scheduler
from xtask.core.result_processor import verify_result, write_result


def main():
    """Main function for command line testing"""
    try:
        init_logger()
        checker = ArgumentChecker()
    
        # Parse command line arguments
        args = checker.parse_arguments()
        args_result = checker.validate_arguments(args)
        checker.print_validation_result(args_result)
        xtask_config = get_xtask_config_data()
        logging.debug("args_result is: %s"% args_result)
        if args_result.is_valid:
            tasks = get_tasks(args, xtask_config)
            for task in tasks:
                result = run_scheduler(task)
                verified_result = verify_result(result)
                write_result(verified_result)
        else:
            sys.exit(1)
    except Exception as e:
        logging.error(f"Argument validation failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
