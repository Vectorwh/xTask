import sys
import logging
import traceback

from xtask.utils.logger import init_logger
from xtask.utils.arg_parser import ArgumentChecker
from xtask.utils.config_parser import get_xtask_config_data
from xtask.utils.openme import OpenMeClient
from xtask.tasks.task_parser import get_tasks
from xtask.tasks.parameter_search import (
    resolve_para_set_strategy,
    run_tpe_parameter_search,
)
from xtask.tasks.assessment_report import create_assessment_reporter
from xtask.tasks.auto_assessment import process_auto_assessment
from xtask.tasks.execution_workflow import TaskExecutionWorkflow
from xtask.core.scheduler import (
    close_scheduler_session,
    run_scheduler,
    run_scheduler_session_trial,
    start_scheduler_session,
)

def report_benchmark_if_available(result):
    """Invoke the optional bench-reporter without changing xTask behavior.

    Upload is opt-in via XTASK_BENCH_REPORT (default off) so routine debug
    runs never reach the dashboard. Enable with XTASK_BENCH_REPORT=1 for
    daily or release executions.
    """
    import os
    enabled = os.environ.get("XTASK_BENCH_REPORT", "").strip().lower()
    if enabled not in ("1", "true", "on", "yes"):
        return
    project_root = os.path.dirname(os.path.abspath(__file__))
    plugin_roots = [
        os.path.join(project_root, "third_party", "bench-reporter"),
        os.path.join(project_root, "bench-reporter"),
    ]
    plugin_root = next((p for p in plugin_roots if os.path.isdir(p)), None)
    if plugin_root is None:
        return
    if not os.path.isfile(os.path.join(plugin_root, "reporter.py")):
        logging.warning("bench-reporter directory exists but reporter.py is missing: %s", plugin_root)
        return
    try:
        if plugin_root not in sys.path:
            sys.path.insert(0, plugin_root)
        from bench_reporter.integration import report_xtask_result
        task_config = result.get("task_config", {}) if isinstance(result, dict) else {}
        report_xtask_result(result, task_config)
    except Exception as exc:
        logging.warning("Optional bench-reporter integration failed: %s", exc)

def run_task(current_task_key, output_path, args, scheduler_session=None):
    """Run one task, reusing the service lifecycle only for auto_assess."""
    if getattr(args, 'assess_test_type', None) == 'auto_assess':
        if scheduler_session is None:
            scheduler_session = start_scheduler_session(
                current_task_key, output_path, args
            )
        result = run_scheduler_session_trial(
            scheduler_session, current_task_key, output_path
        )
        return result, scheduler_session

    return run_scheduler(current_task_key, output_path, args), scheduler_session


def run_task_workflow(args, xtask_config, openme_client, para_strategy=None):
    """Build and execute the normal/exhaustive/auto-assess task workflow."""
    workflow = TaskExecutionWorkflow(
        args=args,
        xtask_config=xtask_config,
        openme_client=openme_client,
        para_strategy=para_strategy,
        task_loader=get_tasks,
        task_executor=run_task,
        auto_assessor=process_auto_assessment,
        reporter_factory=create_assessment_reporter,
        session_closer=close_scheduler_session,
        report_benchmark=report_benchmark_if_available,
    )
    return workflow.run()


def run_validated_workflow(args, xtask_config):
    """Dispatch validated CLI arguments to TPE or the task queue workflow."""
    openme_client = OpenMeClient(
        "main", "A3-ARM", getattr(args, "host_addr", "0.0.0.0")
    )
    assess_test_type = getattr(args, "assess_test_type", None)
    para_strategy = None
    if assess_test_type == "para_set":
        para_strategy = resolve_para_set_strategy(args)

    if assess_test_type == "para_set" and para_strategy == "tpe":
        run_tpe_parameter_search(
            args=args,
            openme_client=openme_client,
            run_trial=run_scheduler,
        )
        return

    workflow_success = run_task_workflow(
        args,
        xtask_config,
        openme_client,
        para_strategy=para_strategy,
    )

    # 发送京me通报
    #openme_client.SendJUEMsgCard(args=args)
    return workflow_success


def parse_and_validate_arguments():
    """Parse CLI input and return it together with validation and config."""
    checker = ArgumentChecker()
    args = checker.parse_arguments()
    args_result = checker.validate_arguments(args)
    checker.print_validation_result(args_result)
    logging.debug("args_result is: %s", args_result)
    return args, args_result, get_xtask_config_data()


def main():
    """CLI entry point."""
    try:
        init_logger()
        args, args_result, xtask_config = parse_and_validate_arguments()
        if not args_result.is_valid:
            sys.exit(1)
        workflow_success = run_validated_workflow(args, xtask_config)
        if workflow_success is False:
            logging.error("One or more tasks failed; exiting with status 1")
            sys.exit(1)
    except Exception as error:
        logging.error("Argument validation failed: %s", error)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
