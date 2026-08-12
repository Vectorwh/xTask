"""Task queue orchestration shared by normal and assessment executions."""

import logging
import traceback
from pathlib import Path


class TaskExecutionWorkflow:
    """Own the mutable state and lifecycle of one static/dynamic task queue."""

    def __init__(
        self,
        *,
        args,
        xtask_config,
        openme_client,
        para_strategy,
        task_loader,
        task_executor,
        auto_assessor,
        reporter_factory,
        session_closer,
    ):
        self.args = args
        self.xtask_config = xtask_config
        self.openme_client = openme_client
        self.para_strategy = para_strategy
        self.task_loader = task_loader
        self.task_executor = task_executor
        self.auto_assessor = auto_assessor
        self.reporter_factory = reporter_factory
        self.session_closer = session_closer

        self.config_para = {}
        self.auto_assess_threshold = {"lower": 0, "curr": 0, "upper": 0}
        self.target_ttft = getattr(args, "target_ttft", 600000)
        self.target_tpot = getattr(args, "target_tpot", 100)
        self.tasks = {}
        self.keys_to_process = []
        self.reporter = None
        self.scheduler_session = None
        self.active_auto_assess_case = None

    def run(self):
        """Generate the initial queue, execute it, and always close resources."""
        self._initialize_queue()
        try:
            index = 0
            while index < len(self.keys_to_process):
                current_task_key = self.keys_to_process[index]
                index += 1
                logging.info(
                    "======== Processing task [%s/%s]: %s ========",
                    index,
                    len(self.keys_to_process),
                    current_task_key,
                )
                action = self._execute_one(current_task_key)
                if action == "break":
                    break
        finally:
            self._close()

    def _initialize_queue(self):
        self.tasks = self.task_loader(
            self.args, self.xtask_config, self.config_para
        )
        self.keys_to_process = list(self.tasks.keys())

        assess_test_type = getattr(self.args, "assess_test_type", None)
        if assess_test_type in {"para_set", "auto_assess"}:
            self.reporter = self.reporter_factory(
                self.args,
                self.tasks,
                para_strategy=self.para_strategy,
            )

    def _execute_one(self, current_task_key):
        try:
            if current_task_key not in self.tasks:
                logging.warning(
                    "Task key '%s' not found in tasks dictionary. Skipping.",
                    current_task_key,
                )
                return None

            output_path = self.tasks[current_task_key]
            self._activate_auto_assess_case(current_task_key)
            logging.info("Running scheduler for task: %s ...", current_task_key)
            result, self.scheduler_session = self.task_executor(
                current_task_key,
                output_path,
                self.args,
                self.scheduler_session,
            )
            self.openme_client.AddCaseResult(
                result, task_path=current_task_key
            )

            action = self.auto_assessor(
                args=self.args,
                result=result,
                config_para=self.config_para,
                auto_assess_threshold=self.auto_assess_threshold,
                target_ttft=self.target_ttft,
                target_tpot=self.target_tpot,
                tasks=self.tasks,
                keys_to_process=self.keys_to_process,
                xtask_config=self.xtask_config,
                current_task_key=current_task_key,
                output_path=output_path,
            )
            self._record_result(current_task_key, output_path, result, action)
            if (
                getattr(self.args, "assess_test_type", None) == "auto_assess"
                and action == "break"
            ):
                # A completed search ends only its base case. The next base
                # case, if present, gets fresh state and a fresh service.
                return None
            return action
        except Exception as error:
            logging.error(
                "Failed to run scheduler for task '%s': %s",
                current_task_key,
                error,
            )
            traceback.print_exc()
            self._record_failure(current_task_key, error)
            return None

    @staticmethod
    def _auto_assess_case_name(task_key):
        generated_case = Path(task_key).stem
        base_case, separator, _ = generated_case.rpartition("_bs-")
        return base_case if separator and base_case else generated_case

    def _activate_auto_assess_case(self, task_key):
        if getattr(self.args, "assess_test_type", None) != "auto_assess":
            return
        base_case = self._auto_assess_case_name(task_key)
        if self.active_auto_assess_case is None:
            self.active_auto_assess_case = base_case
            return
        if base_case == self.active_auto_assess_case:
            return

        self.session_closer(self.scheduler_session)
        self.scheduler_session = None
        self.config_para = {}
        self.auto_assess_threshold = {"lower": 0, "curr": 0, "upper": 0}
        self.active_auto_assess_case = base_case

    def _record_result(self, task_key, output_path, result, action):
        if self.reporter is None:
            return
        try:
            self.reporter.record_result(
                task_key,
                output_path,
                result,
                search_state=self.auto_assess_threshold,
                action=action,
            )
        except Exception:
            logging.error(
                "Failed to update assessment report for %s",
                task_key,
                exc_info=True,
            )

    def _record_failure(self, task_key, error):
        if self.reporter is None:
            return
        try:
            self.reporter.record_failure(
                task_key,
                self.tasks.get(task_key, ""),
                str(error),
                search_state=self.auto_assess_threshold,
            )
        except Exception:
            logging.error(
                "Failed to record assessment failure for %s",
                task_key,
                exc_info=True,
            )

    def _close(self):
        self.session_closer(self.scheduler_session)
        if self.reporter is None:
            return
        try:
            self.reporter.finalize(
                self.auto_assess_threshold.get("stop_reason")
            )
        except Exception:
            logging.error(
                "Failed to finalize assessment report",
                exc_info=True,
            )
