"""Regression coverage for existing-container patches rebased onto main."""

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler
from xtask.core.service_group import build_service_plans, create_runtime_for_plan
from xtask.core.task_scheduler import TaskExecutionContext
from xtask.benchmarks.evalscope import generate_evalscope_script as generator


class ExistingContainerBenchmarkTests(unittest.TestCase):
    def test_instance_artifacts_and_startup_environment_reach_runtime(self):
        instance = {
            "name": "master-0", "host": "127.0.0.1", "role": "master",
            "device_ids": [0, 1], "local_port": 18045,
            "container_name": "existing-xllm", "reuse_existing_container": True,
            "xllm_binary": "/build/xllm", "triton_binary_path": "/build/triton",
            "serving_ready_timeout": 900,
            "startup_environment": ["export TEST_SETTING=1"],
        }
        task = {"case_name": "reuse", "deployment": {
            "master_host": "127.0.0.1", "instances": [instance],
        }}
        with tempfile.TemporaryDirectory() as output:
            plan = build_service_plans(task, output)[0]
            builder = mock.Mock()
            builder.create_container_config.return_value = SimpleNamespace(
                container_name="generated", ipc_mode=None, shm_size=None,
            )
            with mock.patch("xtask.core.service_group.create_host_executor"), \
                 mock.patch("xtask.core.service_group.ContainerManager"):
                runtime = create_runtime_for_plan(plan, {}, "case.yaml", builder)
        self.assertEqual(runtime.container_config.container_name, "existing-xllm")
        merged = builder.create_container_config.call_args.args[1]
        for key in ("xllm_binary", "triton_binary_path", "serving_ready_timeout",
                    "reuse_existing_container", "startup_environment"):
            self.assertEqual(merged[key], instance[key])

    def test_existing_runtime_checks_artifacts_without_downloading(self):
        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(success=True)
        runtime = SimpleNamespace(
            plan=SimpleNamespace(
                instance_name="master-0", host="127.0.0.1", is_benchmark_runner=True,
                instance_config={"reuse_existing_container": True,
                                 "xllm_binary": "/build/xllm",
                                 "triton_binary_path": "/build/triton"},
            ),
            container_config=SimpleNamespace(container_name="existing-xllm"),
            container_manager=manager,
        )
        with tempfile.TemporaryDirectory() as output:
            model = Path(output) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            context = TaskExecutionContext("case.yaml", output, result={})
            context.state.update(
                task_config={"model_path": str(model),
                             "benchmark": {"benchmark_type": "evalscope"}},
                engine_config={"startup_timeout": 30},
                service_runtimes=[runtime], service_plans=[],
            )
            with mock.patch("xtask.core.ranktable_manager.prepare_multihost_ranktables",
                            return_value={}), \
                 mock.patch.object(scheduler, "_clear_pre_benchmark_rst"):
                scheduler._prepare_runtime_dependencies_multi_host(context)
        commands = [call.args[1] for call in manager.exec_in_container.call_args_list]
        self.assertEqual(commands, ["test -x '/build/xllm' && test -d '/build/triton'"])

    def test_host_execution_preserves_case_timeout_and_separate_guard(self):
        manager, guard_manager = mock.Mock(), mock.Mock()
        expected = SimpleNamespace(success=True, stderr="", exit_code=0)
        manager.executor.run.return_value = expected
        guard = mock.Mock(aborted_reason=None, abort_event=threading.Event())
        with mock.patch("xtask.utils.watchdog.NpuProcessGuard", return_value=guard) as factory:
            result = scheduler._execute_benchmark_with_npu_guard(
                manager, SimpleNamespace(container_name="client"),
                {"benchmark_timeout": 123}, {"benchmark_timeout": 60}, "run-perf",
                guard_container_manager=guard_manager,
                guard_container_config=SimpleNamespace(container_name="service"),
                run_on_host=True,
            )
            factory.call_args.kwargs["command_executor"]("npu-check")
        self.assertIs(result, expected)
        manager.exec_in_container.assert_not_called()
        self.assertEqual(manager.executor.run.call_args.kwargs["timeout"], 123)
        guard_manager.exec_in_container.assert_called_once_with(
            "service", "npu-check", timeout=30, capture_output=True,
        )
        guard.start.assert_called_once()
        guard.stop.assert_called_once()

    def test_evalscope_generated_command_preserves_benchmark_settings(self):
        task = {
            "host_addr": "127.0.0.1", "port": 18045,
            "served_model_name": "test-model", "model_path": "/models/test",
            "dataset_name": "random", "metrics": "perf",
            "benchmark": {"perf": {
                "batch_size": 4, "num_prompts": 40, "warmup_num": 10,
                "temperature": 0, "stream": True, "tokenize_prompt": True,
            }},
        }
        with tempfile.TemporaryDirectory() as output:
            with mock.patch.object(generator, "path_mapping_in_container",
                                   side_effect=lambda path: path):
                generator.generate_evalscope_script(task, output)
            script = (Path(output) / "run_evalscope_test.sh").read_text()
        for argument in ("--parallel 4", "--number 40", "--rate -1",
                         "--warmup-num 10", "--temperature 0",
                         "--stream", "--tokenize-prompt"):
            self.assertIn(argument, script)


if __name__ == "__main__":
    unittest.main()
