import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.core.external_service import (
    apply_external_service_to_task_config,
    is_external_service_enabled,
    is_vbench_evaluate_only,
    resolve_external_location,
    resolve_health_path,
    topology_fields_ignored,
    validate_external_service_config,
)
from xtask.core.scheduler import (
    generate_commands,
    _probe_external_service,
    _prepare_runtime_dependencies,
    _start_and_wait_service,
)
from xtask.core.service_group import get_benchmark_endpoint
from xtask.core.task_scheduler import TaskExecutionContext


def _aisbench_task(**overrides):
    task = {
        "case_name": "external_case",
        "external_service": True,
        "served_model_name": "Qwen3-8B",
        "metrics": "perf",
        "dataset_name": "gsm8k",
        "benchmark": {"benchmark_type": "aisbench", "perf": {}},
        "deployment": {
            "benchmark": {
                "endpoint": "http://10.20.30.40:8800",
                "health_path": "/health",
            }
        },
    }
    task.update(overrides)
    return task


class ExternalServiceConfigTests(unittest.TestCase):
    def test_missing_or_false_is_disabled(self):
        self.assertFalse(is_external_service_enabled(None))
        self.assertFalse(is_external_service_enabled({}))
        self.assertFalse(is_external_service_enabled({"external_service": False}))
        self.assertFalse(is_external_service_enabled({"external_service": {"enabled": False}}))

    def test_true_and_mapping_enable(self):
        self.assertTrue(is_external_service_enabled({"external_service": True}))
        self.assertTrue(
            is_external_service_enabled(
                {"external_service": {"endpoint": "http://10.0.0.1:8800"}}
            )
        )
        self.assertTrue(
            is_external_service_enabled(
                {"external_service": {"enabled": True, "host": "10.0.0.1", "port": 9}}
            )
        )

    def test_endpoint_resolution_order(self):
        task = {
            "external_service": {
                "enabled": True,
                "endpoint": "https://gw.example:8443/inference/svc",
            },
            "deployment": {"benchmark": {"endpoint": "http://ignored:1"}},
            "host_addr": "1.1.1.1",
            "port": 2,
        }
        location = resolve_external_location(task)
        self.assertEqual(location.host, "gw.example")
        self.assertEqual(location.port, 8443)
        self.assertEqual(location.scheme, "https")
        self.assertEqual(location.base_path, "/inference/svc")
        self.assertEqual(location.origin, "https://gw.example:8443")

    def test_empty_health_path_skips_probe(self):
        task = {
            "external_service": {"enabled": True, "endpoint": "http://10.0.0.1:9", "health_path": ""},
        }
        self.assertIsNone(resolve_health_path(task))

    def test_vbench_evaluate_does_not_require_endpoint(self):
        task = {
            "external_service": True,
            "benchmark": {"benchmark_type": "vbench", "vbench": {"mode": "evaluate"}},
        }
        self.assertTrue(is_vbench_evaluate_only(task))
        validate_external_service_config(task)

    def test_missing_endpoint_raises(self):
        with self.assertRaises(ValueError):
            validate_external_service_config({"external_service": True, "benchmark": {}})

    def test_serving_search_is_rejected(self):
        task = {
            "external_service": True,
            "deployment": {"benchmark": {"endpoint": "http://10.0.0.1:8800"}},
            "assess_test_param": {
                "serving_extensiable_param": {"max_seqs_per_batch": [8, 16]}
            },
        }
        with self.assertRaises(ValueError) as ctx:
            validate_external_service_config(task)
        self.assertIn("serving", str(ctx.exception).lower())

    def test_topology_fields_are_listed_not_required(self):
        task = _aisbench_task()
        task["deployment"]["instances"] = [{"name": "master-0", "host": "10.0.0.1"}]
        self.assertEqual(topology_fields_ignored(task), ["deployment.instances"])
        validate_external_service_config(task)

    def test_apply_sets_host_and_vbench_full_url(self):
        task = {
            "external_service": True,
            "benchmark": {
                "benchmark_type": "vbench",
                "vbench": {},
            },
            "deployment": {
                "benchmark": {
                    "endpoint": "http://10.20.30.40:8800/v1/video/generation"
                }
            },
        }
        apply_external_service_to_task_config(task)
        self.assertEqual(task["host_addr"], "10.20.30.40")
        self.assertEqual(task["port"], 8800)
        self.assertEqual(
            task["benchmark"]["vbench"]["endpoint"],
            "http://10.20.30.40:8800/v1/video/generation",
        )

    def test_disabled_apply_is_noop(self):
        task = {"host_addr": "keep", "port": 1}
        apply_external_service_to_task_config(task)
        self.assertEqual(task["host_addr"], "keep")


class ExternalServiceEndpointTests(unittest.TestCase):
    def test_e2e_still_ignores_explicit_endpoint(self):
        task = {
            "deployment": {
                "master_host": "10.0.0.1",
                "benchmark": {"endpoint": "http://wrong-host:9999"},
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0],
                        "local_port": 8800,
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            from xtask.core.service_group import build_service_plans

            plans = build_service_plans(task, tmp)
            self.assertEqual(get_benchmark_endpoint(task, plans), "http://10.0.0.1:8800")

    def test_external_uses_explicit_endpoint_even_with_instances(self):
        task = {
            "external_service": True,
            "deployment": {
                "master_host": "10.0.0.1",
                "benchmark": {"endpoint": "http://10.9.8.7:1234"},
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0],
                        "local_port": 8800,
                    }
                ],
            },
        }
        self.assertEqual(get_benchmark_endpoint(task, []), "http://10.9.8.7:1234")


class ExternalServiceSchedulerTests(unittest.TestCase):
    def test_generate_commands_skips_xllm_startup(self):
        task = _aisbench_task()
        task["deployment"]["instances"] = [
            {
                "name": "master-0",
                "host": "10.0.0.1",
                "role": "master",
                "device_ids": [0],
                "local_port": 8800,
            }
        ]
        apply_external_service_to_task_config(task)
        with mock.patch(
            "xtask.core.scheduler.generate_benchmark_command",
            return_value="bash /tmp/run_aisbench_test.sh",
        ) as bench, mock.patch(
            "xtask.core.service_group.build_service_plans"
        ) as build:
            startup, command = generate_commands(
                {"engine_type": "xllm", "working_dir": "/workspace"},
                task,
                "case.yaml",
                "/tmp/out",
            )
        self.assertEqual(startup, "")
        self.assertEqual(command, "bash /tmp/run_aisbench_test.sh")
        bench.assert_called_once()
        build.assert_not_called()

    def test_runtime_dependencies_skip_download_xllm(self):
        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(
            success=True, stdout="", stderr=""
        )
        context = TaskExecutionContext("task.yaml", "output", result={})
        context.state.update(
            external_service=True,
            task_config={
                "case_name": "external_case",
                "benchmark": {"benchmark_type": "aisbench"},
            },
            engine_config={"startup_timeout": 30},
            container_config=SimpleNamespace(container_name="client"),
            container_manager=manager,
        )
        _prepare_runtime_dependencies(context)
        self.assertFalse(context.complete)
        commands = [call.args[1] for call in manager.exec_in_container.call_args_list]
        self.assertTrue(any("download_ais_bench.sh" in cmd for cmd in commands))
        self.assertFalse(any("download_xllm.sh" in cmd for cmd in commands))

    def test_start_stage_probes_instead_of_starting_xllm(self):
        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(
            success=True, stdout="200", stderr=""
        )
        context = TaskExecutionContext("task.yaml", "output", result={})
        task = _aisbench_task()
        apply_external_service_to_task_config(task)
        context.state.update(
            external_service=True,
            task_config=task,
            engine_config={},
            container_config=SimpleNamespace(container_name="client"),
            container_manager=manager,
            startup_cmd="should-not-run",
            service_plans=[],
        )
        _start_and_wait_service(context)
        self.assertTrue(context.state["service_healthy"])
        command = manager.exec_in_container.call_args.args[1]
        self.assertIn("10.20.30.40:8800/health", command)
        self.assertNotIn("should-not-run", command)

    def test_probe_accepts_http_2xx_without_ok_body(self):
        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(
            success=True, stdout="204", stderr=""
        )
        context = TaskExecutionContext("task.yaml", "output", result={})
        task = _aisbench_task()
        apply_external_service_to_task_config(task)
        context.state.update(
            external_service=True,
            task_config=task,
            container_config=SimpleNamespace(container_name="client"),
            container_manager=manager,
            service_plans=[],
        )
        self.assertTrue(_probe_external_service(context))

    def test_vbench_evaluate_skips_probe(self):
        context = TaskExecutionContext("task.yaml", "output", result={})
        context.state.update(
            external_service=True,
            task_config={
                "external_service": True,
                "benchmark": {
                    "benchmark_type": "vbench",
                    "vbench": {"mode": "evaluate"},
                },
            },
        )
        self.assertTrue(_probe_external_service(context))
        self.assertTrue(context.state["service_healthy"])


class ExternalEvalscopeScriptTests(unittest.TestCase):
    def test_e2e_url_shape_unchanged_when_switch_off(self):
        from xtask.benchmarks.evalscope.generate_evalscope_script import (
            generate_evalscope_script,
        )

        task = {
            "benchmark": {
                "benchmark_type": "evalscope",
                "perf": {
                    "batch_size": 1,
                    "max_out_len": 32,
                    "max_prompt_length": 16,
                    "min_prompt_length": 16,
                    "num_prompts": 1,
                    "output_len": 32,
                    "request_rate": 1,
                },
            },
            "dataset_name": "random",
            "host_addr": "127.0.0.1",
            "metrics": "perf",
            "model_path": "/models/test",
            "port": 8800,
            "served_model_name": "test-model",
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.evalscope.generate_evalscope_script.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            generate_evalscope_script(task, tmp)
            script = (Path(tmp) / "run_evalscope_test.sh").read_text(encoding="utf-8")
        self.assertIn("--url http://127.0.0.1:8800/v1/chat/completions", script)

    def test_external_gateway_path_is_prefixed(self):
        from xtask.benchmarks.evalscope.generate_evalscope_script import (
            generate_evalscope_script,
        )

        task = {
            "external_service": {
                "enabled": True,
                "endpoint": "http://gw.example:8080/inference/svc",
                "api_key": "secret",
            },
            "benchmark": {
                "benchmark_type": "evalscope",
                "acc": {
                    "batch_size": 1,
                    "dataset_args": {},
                    "endpoint": "/v1",
                    "generation_config": {},
                    "timeout": 60,
                },
            },
            "dataset_name": "gsm8k",
            "metrics": "acc",
            "served_model_name": "test-model",
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.evalscope.generate_evalscope_script.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            generate_evalscope_script(task, tmp)
            script = (Path(tmp) / "run_evalscope_test.sh").read_text(encoding="utf-8")
        self.assertIn("--api-url http://gw.example:8080/inference/svc/v1", script)
        self.assertIn("--api-key secret", script)


if __name__ == "__main__":
    unittest.main()
