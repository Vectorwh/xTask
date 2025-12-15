import sys
import logging

from xtask.utils.logger import init_logger
from xtask.utils.arg_parser import ArgumentChecker
from xtask.utils.config_parser import get_config_data
from xtask.tasks.task_parser import get_tasks
from xtask.core.scheduler import run_scheduler
from xtask.core.result_processor import verify_result, write_result


def main():
    """Main function for command line testing"""
    init_logger()
    checker = ArgumentChecker()
    
    try:
        # Parse command line arguments
        args = checker.parse_arguments()
        result = checker.validate_arguments(args)
        checker.print_validation_result(result)
        xtask_config = get_config_data()
        logging.info("result is: %s"% result)
        if result.is_valid:
            tasks = get_tasks(args, xtask_config)
            for task in tasks:
                result = run_scheduler(task)
                verified_result = verify_result(result)
                write_result(verified_result)
        else:
            sys.exit(1)
    except Exception as e:
        logging.error(f"Argument validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
