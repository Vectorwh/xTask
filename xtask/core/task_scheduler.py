from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence

import logging
import time


@dataclass
class TaskExecutionContext:
    """Mutable state shared by one task's lifecycle stages."""

    task_yaml_path: str
    output_file_path: str
    args: Any = None
    result: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    complete: bool = False


Stage = Callable[[TaskExecutionContext], None]


@dataclass(frozen=True)
class SchedulerDependencies:
    """Real lifecycle stages for one local service instance."""

    load_task_and_engine: Stage
    prepare_container: Stage
    prepare_runtime_dependencies: Stage
    start_and_wait_service: Stage
    run_benchmark_and_process: Stage
    finalize_result: Stage
    cleanup: Stage


class TaskScheduler:
    """Run one task's stages, with controlled short-circuit and cleanup."""

    def __init__(self, dependencies: SchedulerDependencies, logger=None):
        self.dependencies = dependencies
        self.logger = logger or logging.getLogger(__name__)

    def prepare(self, context: TaskExecutionContext) -> None:
        """Prepare containers and services without running a benchmark."""
        phases: Sequence[Stage] = (
            self.dependencies.load_task_and_engine,
            self.dependencies.prepare_container,
            self.dependencies.prepare_runtime_dependencies,
            self.dependencies.start_and_wait_service,
        )
        self._run_phases(context, phases)

    def run_trial(self, context: TaskExecutionContext) -> None:
        """Run one benchmark against an already prepared service."""
        self._run_phases(context, (self.dependencies.run_benchmark_and_process,))

    def finalize_result(self, context: TaskExecutionContext) -> None:
        """Persist a result without allowing reporting failures to fail the task."""
        try:
            self.dependencies.finalize_result(context)
        except Exception:
            self.logger.error("Result finalization failed", exc_info=True)

    def cleanup(self, context: TaskExecutionContext) -> None:
        self.dependencies.cleanup(context)

    @staticmethod
    def _run_phases(context: TaskExecutionContext, phases: Sequence[Stage]) -> None:
        for phase in phases:
            if context.complete:
                break
            phase(context)

    def run(self, context: TaskExecutionContext) -> Dict[str, Any]:
        start_time = time.time()
        try:
            self.prepare(context)
            self.run_trial(context)
        except Exception as exc:
            context.result["success"] = False
            context.result["message"] = f"Error in scheduler: {exc}"
            self.logger.error(context.result["message"], exc_info=True)
        finally:
            try:
                self.cleanup(context)
            except Exception:
                self.logger.error("Scheduler cleanup failed", exc_info=True)
            context.result["execution_time"] = time.time() - start_time
            self.logger.info(
                "Scheduler execution completed in %.2f seconds",
                context.result["execution_time"],
            )
            self.finalize_result(context)
        return context.result
