import json
import os
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.benchmarks.aisbench import generate_aisbench_swe_script as swe_generator
from xtask.benchmarks.aisbench.aisbench_swe_postprocess import aisbench_swe_post_process
from xtask.benchmarks.aisbench import swebench_x86_setup as swe_setup
from xtask.benchmarks.aisbench.swebench import (
    is_aisbench_swebench_task,
    is_swebench_eval_only,
    validate_aisbench_swebench_task,
)
from xtask.core import result_processor, scheduler
from xtask.core.external_service import requires_external_endpoint
from xtask.core.task_scheduler import TaskExecutionContext
from xtask.utils.container_manager import CommandResult


_SETUP_RUNNER = "xtask.benchmarks.aisbench.swebench_x86_setup.setup_aisbench_swe_runner"


def _ok_setup(bin_path="/usr/local/bin/ais_bench"):
    return swe_setup.SweX86SetupResult(
        command=CommandResult(
            success=True,
            exit_code=0,
            stdout=f"AIS_BENCH_BIN={bin_path}",
            stderr="",
            execution_time=0.1,
        ),
        ais_bench_bin=bin_path,
    )


def _swe_task(**overrides):
    task = {
        "case_name": "swe_case",
        "external_service": True,
        "served_model_name": "Qwen3-8B",
        "metrics": "swebench",
        "host_addr": "11.87.191.75",
        "port": 8800,
        "benchmark": {
            "benchmark_type": "aisbench",
            "swebench": {
                "mode": "all",
                "dataset": "verified_mini",
                "reuse": True,
                "timeout": 86400,
                "runner": {"host": "127.0.0.1"},
            },
        },
        "deployment": {
            "benchmark": {
                "endpoint": "http://11.87.191.75:8800",
                "health_path": "/health",
            }
        },
    }
    swe = overrides.pop("swebench", None)
    if swe is not None:
        task["benchmark"]["swebench"].update(swe)
    task.update(overrides)
    return task


def _perf_task():
    return {
        "case_name": "perf_case",
        "metrics": "perf",
        "served_model_name": "Qwen3-8B",
        "benchmark": {
            "benchmark_type": "aisbench",
            "perf": {"baseline": None, "threshold": 0.1},
            "swebench": {"dataset": "verified_mini", "runner": {"host": "10.0.0.1"}},
        },
    }


class SwebenchGateTests(unittest.TestCase):
    def test_perf_yaml_is_not_a_swe_task_even_with_swebench_block(self):
        self.assertFalse(is_aisbench_swebench_task(_perf_task()))
        self.assertFalse(is_swebench_eval_only(_perf_task()))

    def test_acc_yaml_is_not_a_swe_task(self):
        task = {
            "metrics": "acc",
            "dataset_name": "swebench",
            "benchmark": {
                "benchmark_type": "aisbench",
                "acc": {"baseline": 90, "threshold": 0.04},
            },
        }
        self.assertFalse(is_aisbench_swebench_task(task))

    def test_swe_requires_metric_type_and_block(self):
        self.assertTrue(is_aisbench_swebench_task(_swe_task()))
        missing_block = _swe_task()
        missing_block["benchmark"].pop("swebench")
        self.assertFalse(is_aisbench_swebench_task(missing_block))
        wrong_type = _swe_task()
        wrong_type["benchmark"]["benchmark_type"] = "evalscope"
        self.assertFalse(is_aisbench_swebench_task(wrong_type))

    def test_eval_only_is_gated(self):
        self.assertTrue(is_swebench_eval_only(_swe_task(swebench={"mode": "eval"})))
        self.assertFalse(is_swebench_eval_only(_swe_task(swebench={"mode": "all"})))
        self.assertFalse(is_swebench_eval_only(_perf_task()))

    def test_runner_host_is_required(self):
        with self.assertRaisesRegex(ValueError, "runner.host"):
            validate_aisbench_swebench_task(
                _swe_task(swebench={"runner": {}})
            )

    def test_eval_only_does_not_require_external_endpoint(self):
        task = _swe_task(swebench={"mode": "eval"})
        self.assertFalse(requires_external_endpoint(task))
        self.assertTrue(requires_external_endpoint(_swe_task()))


class SwebenchScriptTests(unittest.TestCase):
    def test_generated_config_matches_aisbench_example_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = swe_generator.generate_aisbench_swe_script(
                _swe_task(), tmp
            )
            config = (Path(tmp) / "aisbench_swe_config.py").read_text(encoding="utf-8")
            script = (Path(tmp) / "run_aisbench_swe.sh").read_text(encoding="utf-8")

        self.assertIn("bash", command)
        self.assertIn("run_aisbench_swe.sh", command)
        self.assertIn("SWEBenchDataset", config)
        self.assertIn("SWEBenchInferTask", config)
        self.assertIn("SWEBenchEvalTask", config)
        self.assertIn('name=\'verified_mini\'', config)
        self.assertIn('model=\'Qwen3-8B\'', config)
        self.assertIn("http://11.87.191.75:8800/v1", config)
        self.assertIn("LiteLLMChat", config)
        self.assertNotIn("gsm8k", config)
        self.assertIn("-m all", script)
        self.assertIn("--reuse", script)
        self.assertIn("ais_bench", script)
        self.assertNotIn("/workspace/tools/download_ais_bench.sh", script)

    def test_perf_generate_benchmark_command_does_not_call_swe(self):
        with mock.patch.object(
            swe_generator, "generate_aisbench_swe_script"
        ) as swe, mock.patch(
            "xtask.benchmarks.aisbench.generate_aisbench_script.generate_aisbench_script",
            return_value="bash /tmp/run_aisbench_test.sh",
        ) as perf:
            command = scheduler.generate_benchmark_command(_perf_task(), "/tmp/out")
        self.assertEqual(command, "bash /tmp/run_aisbench_test.sh")
        perf.assert_called_once()
        swe.assert_not_called()

    def test_swe_generate_benchmark_command_does_not_call_gsm8k_generator(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.aisbench.generate_aisbench_script.generate_aisbench_script"
        ) as perf:
            command = scheduler.generate_benchmark_command(_swe_task(), tmp)
            perf.assert_not_called()
            self.assertIn("run_aisbench_swe.sh", command)
            self.assertTrue((Path(tmp) / "aisbench_swe_config.py").is_file())

    def test_unsafe_extra_args_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsafe token"):
                swe_generator.generate_aisbench_swe_script(
                    _swe_task(swebench={"extra_args": ["--foo; rm -rf /"]}),
                    tmp,
                )


class SwebenchExecuteTests(unittest.TestCase):
    def test_local_execute_uses_host_executor_not_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            swe_generator.generate_aisbench_swe_script(_swe_task(), tmp)
            fake = mock.Mock()
            fake.run.return_value = CommandResult(
                success=True, exit_code=0, stdout="ok", stderr="", execution_time=1.0
            )
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ) as factory, mock.patch.object(
                swe_generator, "is_local_host", return_value=True
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ):
                result = swe_generator.execute_aisbench_swe_benchmark(
                    _swe_task(), tmp, timeout=12
                )
        self.assertTrue(result.success)
        factory.assert_called_once()
        command = fake.run.call_args.args[0]
        self.assertIn("run_aisbench_swe.sh", command)
        self.assertIn("AIS_BENCH_BIN=/usr/local/bin/ais_bench", command)
        self.assertEqual(fake.run.call_args.kwargs["timeout"], 12)
        fake.upload.assert_not_called()

    def test_remote_execute_uploads_and_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _swe_task(swebench={"runner": {"host": "10.20.30.50", "work_dir": "/tmp/swe"}})
            swe_generator.generate_aisbench_swe_script(task, tmp)
            fake = mock.Mock()
            fake.run.side_effect = [
                CommandResult(
                    success=True, exit_code=0, stdout="/home/remote-user",
                    stderr="", execution_time=0.1,
                ),
                CommandResult(
                    success=True, exit_code=0, stdout="ok", stderr="",
                    execution_time=2.0,
                ),
            ]
            fake.upload.return_value = True
            fake.download.return_value = True
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ) as setup:
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)
        self.assertTrue(result.success)
        fake.upload.assert_called_once()
        fake.download.assert_called_once()
        self.assertEqual(setup.call_args.args[3], "/tmp/swe")
        self.assertEqual(
            setup.call_args.kwargs["install_dir"],
            "/home/remote-user/xtask_swe/runtime",
        )
        command = fake.run.call_args.args[0]
        self.assertIn("SWE_WORK_DIR=/tmp/swe", command)
        self.assertIn("AIS_BENCH_BIN=/usr/local/bin/ais_bench", command)

    def test_remote_execute_defaults_to_login_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _swe_task(swebench={"runner": {"host": "10.20.30.50"}})
            swe_generator.generate_aisbench_swe_script(task, tmp)
            fake = mock.Mock()
            fake.run.side_effect = [
                CommandResult(
                    success=True, exit_code=0, stdout="/home/remote-user",
                    stderr="", execution_time=0.1,
                ),
                CommandResult(
                    success=True, exit_code=0, stdout="ok", stderr="",
                    execution_time=2.0,
                ),
            ]
            fake.upload.return_value = True
            fake.download.return_value = True
            expected_work_dir = "/home/remote-user/xtask_swe/swe_case"
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ) as setup:
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)
        self.assertTrue(result.success)
        self.assertEqual(setup.call_args.args[3], expected_work_dir)
        self.assertEqual(
            setup.call_args.kwargs["install_dir"],
            "/home/remote-user/xtask_swe/runtime",
        )
        fake.upload.assert_called_once_with(tmp, expected_work_dir)
        fake.download.assert_called_once_with(expected_work_dir, tmp)
        command = fake.run.call_args.args[0]
        self.assertIn(f"SWE_WORK_DIR={expected_work_dir}", command)

    def test_execute_stops_when_x86_setup_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            swe_generator.generate_aisbench_swe_script(_swe_task(), tmp)
            fake = mock.Mock()
            failed = swe_setup.SweX86SetupResult(
                command=CommandResult(
                    success=False,
                    exit_code=1,
                    stdout="",
                    stderr="docker ps failed",
                    execution_time=0.2,
                )
            )
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch(
                _SETUP_RUNNER, return_value=failed
            ):
                result = swe_generator.execute_aisbench_swe_benchmark(_swe_task(), tmp)
        self.assertFalse(result.success)
        self.assertIn("docker ps failed", result.stderr)
        fake.run.assert_not_called()
        fake.upload.assert_not_called()


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _make_aisbench_tree(root: Path) -> Path:
    tree = root / "benchmark"
    (tree / "ais_bench").mkdir(parents=True)
    (tree / "setup.py").write_text("from setuptools import setup\nsetup(name='ais-bench')\n")
    return tree


class SwebenchX86SetupTests(unittest.TestCase):
    def test_skip_setup_does_not_touch_the_runner(self):
        fake = mock.Mock()
        result = swe_setup.setup_aisbench_swe_runner(
            _swe_task(swebench={"setup": {"skip": True}}),
            fake,
            "127.0.0.1",
            "/tmp/swe-work",
        )
        self.assertTrue(result.command.success)
        self.assertEqual(result.ais_bench_bin, "ais_bench")
        fake.run.assert_not_called()
        fake.upload.assert_not_called()

    def test_missing_aisbench_source_returns_failed_command(self):
        fake = mock.Mock()
        fake.run.return_value = CommandResult(
            success=True, exit_code=0, stdout="", stderr="", execution_time=0.0
        )
        task = _swe_task(
            swebench={"setup": {"aisbench_source": "/no/such/aisbench/tree"}}
        )
        with mock.patch.object(swe_setup, "is_local_host", return_value=True):
            result = swe_setup.setup_aisbench_swe_runner(
                task, fake, "127.0.0.1", "/tmp/swe-work"
            )
        self.assertFalse(result.command.success)
        self.assertIn("aisbench_source", result.command.stderr)

    def test_setup_runs_host_script_with_synced_aisbench_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ais = _make_aisbench_tree(root)
            install = root / "runtime"
            work = root / "work"
            work.mkdir()
            task = _swe_task(
                swebench={
                    "setup": {
                        "aisbench_source": str(ais),
                        "install_dir": str(install),
                    }
                }
            )
            fake = mock.Mock()
            fake.run.side_effect = [
                CommandResult(
                    success=True, exit_code=0, stdout="", stderr="", execution_time=0.0
                ),
                CommandResult(
                    success=True,
                    exit_code=0,
                    stdout="Installing AISBench from %s\nAIS_BENCH_BIN=/opt/ais_bench\n"
                    % ais,
                    stderr="",
                    execution_time=1.0,
                ),
            ]
            with mock.patch.object(swe_setup, "is_local_host", return_value=True):
                result = swe_setup.setup_aisbench_swe_runner(
                    task, fake, "127.0.0.1", str(work)
                )
        self.assertTrue(result.command.success)
        self.assertEqual(result.ais_bench_bin, "/opt/ais_bench")
        setup_cmd = fake.run.call_args_list[1].args[0]
        self.assertIn("AISBENCH_DIR=", setup_cmd)
        self.assertIn(str(ais), setup_cmd)
        self.assertIn("setup_aisbench_swe_x86.sh", setup_cmd)
        self.assertEqual(fake.run.call_args_list[1].kwargs["timeout"], 1800)

    def test_resolve_aisbench_from_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ais = _make_aisbench_tree(root)
            task = _swe_task(swebench={"setup": {"source_root": str(root)}})
            self.assertEqual(swe_setup.resolve_aisbench_source(task), ais)

    def test_setup_script_pips_aisbench_and_clones_swe_deps(self):
        script = swe_setup.setup_script_path()
        self.assertTrue(script.is_file())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            ais = _make_aisbench_tree(root)
            install = root / "runtime"
            log = root / "commands.log"
            _write_exec(
                bin_dir / "docker",
                "#!/bin/sh\necho docker \"$*\" >> %s\nexit 0\n" % shlex.quote(str(log)),
            )
            _write_exec(
                bin_dir / "git",
                """#!/bin/sh
echo git "$@" >> %s
dest=""
for arg in "$@"; do dest="$arg"; done
mkdir -p "$dest/src"
exit 0
"""
                % shlex.quote(str(log)),
            )
            _write_exec(
                bin_dir / "python3",
                """#!/bin/sh
echo python "$@" >> %s
if echo "$*" | grep -q shutil.which; then
  echo /tmp/fake-ais_bench
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  exit 0
fi
exit 1
"""
                % shlex.quote(str(log)),
            )
            env = os.environ.copy()
            env["PATH"] = "%s:%s" % (bin_dir, env.get("PATH", ""))
            env["PYTHON_BIN"] = str(bin_dir / "python3")
            env["INSTALL_ROOT"] = str(install)
            env["AISBENCH_DIR"] = str(ais)
            env["CLONE_IF_MISSING"] = "1"
            completed = subprocess.run(
                ["bash", str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("AIS_BENCH_BIN=/tmp/fake-ais_bench", completed.stdout)
            recorded = log.read_text(encoding="utf-8")
            self.assertIn("pip install -e", recorded)
            self.assertIn("clone", recorded)
            self.assertTrue((install / "mini-swe-agent" / "src").is_dir())
            self.assertTrue((install / "SWE-bench" / "src").is_dir())

    def test_setup_script_requires_working_docker(self):
        script = swe_setup.setup_script_path()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            _write_exec(bin_dir / "docker", "#!/bin/sh\nexit 1\n")
            _write_exec(bin_dir / "python3", "#!/bin/sh\nexit 0\n")
            env = os.environ.copy()
            env["PATH"] = "%s:%s" % (bin_dir, env.get("PATH", ""))
            env["PYTHON_BIN"] = str(bin_dir / "python3")
            env["INSTALL_ROOT"] = str(root / "runtime")
            env["AISBENCH_DIR"] = str(root / "ais")
            completed = subprocess.run(
                ["bash", str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("docker ps failed", completed.stderr)


class SwebenchPostprocessTests(unittest.TestCase):
    def test_eval_json_is_normalized_to_resolved_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results" / "swebench"
            result_dir.mkdir(parents=True)
            (result_dir / "swebench_verified_mini.json").write_text(
                json.dumps(
                    {
                        "accuracy": 0.42,
                        "submitted_accuracy": 0.5,
                        "resolved_instances": 21,
                        "unresolved_instances": 29,
                        "error_instances": 0,
                        "total_instances": 50,
                        "submitted_instances": 42,
                        "harness_exit_code": 0,
                    }
                ),
                encoding="utf-8",
            )
            result, output_dir = aisbench_swe_post_process(_swe_task(), tmp, [])
        self.assertEqual(result["resolved_rate"], 0.42)
        self.assertEqual(result["resolved_instances"], 21)
        self.assertEqual(result["total_instances"], 50)
        self.assertTrue(output_dir)

    def test_infer_only_accepts_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp) / "predictions" / "swebench"
            pred_dir.mkdir(parents=True)
            (pred_dir / "swebench_preds.json").write_text(
                json.dumps(
                    {
                        "django__django-123": {"instance_id": "django__django-123", "model_patch": "diff"}
                    }
                ),
                encoding="utf-8",
            )
            result, _ = aisbench_swe_post_process(
                _swe_task(swebench={"mode": "infer"}), tmp, []
            )
        self.assertEqual(result["mode"], "infer")
        self.assertIsNotNone(result["predictions_file"])
        self.assertGreaterEqual(result["submitted_instances"], 1)


class SwebenchVerifyTests(unittest.TestCase):
    def test_complete_result_passes_without_baseline(self):
        result = {
            "benchmark_success": True,
            "benchmark_swebench_result": {"resolved_rate": 0.4, "resolved_instances": 20},
        }
        self.assertTrue(result_processor.verify_result(result, _swe_task()))

    def test_missing_resolved_rate_fails_for_eval(self):
        result = {
            "benchmark_success": True,
            "benchmark_swebench_result": {"mode": "all"},
        }
        self.assertFalse(result_processor.verify_result(result, _swe_task()))

    def test_baseline_uses_resolved_rate(self):
        config = _swe_task(swebench={"baseline": 0.40, "threshold": 0.10})
        result = {
            "benchmark_success": True,
            "benchmark_swebench_result": {"resolved_rate": 0.38},
        }
        self.assertTrue(result_processor.verify_result(result, config))
        result["benchmark_swebench_result"]["resolved_rate"] = 0.20
        self.assertFalse(result_processor.verify_result(result, config))

    def test_perf_task_ignores_swebench_result_and_block(self):
        config = _perf_task()
        result = {
            "benchmark_success": True,
            "benchmark_swebench_result": {"resolved_rate": 1.0},
        }
        self.assertFalse(result_processor.verify_result(result, config))
        self.assertIsNone(result_processor._configured_baseline(config))

    def test_workbook_payload_is_bounded(self):
        result = {
            "task_config": _swe_task(),
            "benchmark_swebench_result": {
                "resolved_rate": 0.4,
                "resolved_instances": 20,
                "predictions": [{"patch": "x"} for _ in range(1000)],
            },
        }
        payload = result_processor._result_payload(result)
        self.assertEqual(payload["resolved_rate"], 0.4)
        self.assertNotIn("predictions", payload)


class SwebenchSchedulerIsolationTests(unittest.TestCase):
    def test_external_client_container_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = TaskExecutionContext("task.yaml", tmp, result={})
            context.state.update(
                task_config=_swe_task(),
                engine_config={},
            )
            with mock.patch.object(
                scheduler, "_prepare_single_host_client_container"
            ) as create_client:
                scheduler._prepare_external_client_container(context)
        create_client.assert_not_called()
        self.assertNotIn("container_manager", context.state)

    def test_eval_only_skips_serving_container_and_xllm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = TaskExecutionContext("task.yaml", tmp, result={})
            context.state.update(
                task_config=_swe_task(swebench={"mode": "eval"}),
                engine_config={},
            )
            with mock.patch.object(
                scheduler, "_prepare_single_host_client_container"
            ) as create_client:
                scheduler._prepare_container(context)
            create_client.assert_not_called()
            scheduler._start_and_wait_service(context)
        self.assertTrue(context.state["service_healthy"])

    def test_runtime_deps_skip_aisbench_install_for_swe(self):
        manager = mock.Mock()
        context = TaskExecutionContext("task.yaml", "/tmp/out", result={})
        context.state.update(
            external_service=True,
            task_config=_swe_task(),
            engine_config={"startup_timeout": 30},
            container_config=SimpleNamespace(container_name="client"),
            container_manager=manager,
        )
        scheduler._prepare_runtime_dependencies(context)
        manager.exec_in_container.assert_not_called()

    def test_run_benchmark_does_not_enter_npu_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = TaskExecutionContext("task.yaml", tmp, result={})
            context.state.update(
                service_healthy=True,
                task_config=_swe_task(),
                engine_config={"benchmark_timeout": 10},
            )
            fake_result = CommandResult(
                success=True, exit_code=0, stdout="ok", stderr="", execution_time=1.0
            )
            swe_result = {"resolved_rate": 0.4, "resolved_instances": 20}
            with mock.patch(
                "xtask.benchmarks.aisbench.generate_aisbench_swe_script.execute_aisbench_swe_benchmark",
                return_value=fake_result,
            ) as execute, mock.patch(
                "xtask.benchmarks.aisbench.aisbench_swe_postprocess.aisbench_swe_post_process",
                return_value=(swe_result, tmp),
            ), mock.patch.object(
                scheduler, "_execute_benchmark_with_npu_guard"
            ) as npu_guard:
                scheduler._run_benchmark_and_process(context)
        execute.assert_called_once()
        npu_guard.assert_not_called()
        self.assertTrue(context.result["success"])
        self.assertEqual(context.result["benchmark_swebench_result"]["resolved_rate"], 0.4)
        self.assertNotIn("container_config", context.state)

    def test_existing_perf_run_still_requires_container_state(self):
        context = TaskExecutionContext("task.yaml", "/tmp/out", result={})
        context.state.update(
            service_healthy=True,
            task_config=_perf_task(),
            engine_config={"benchmark_timeout": 10},
            benchmark_cmd="bash /tmp/run_aisbench_test.sh",
            init_entries=[],
        )
        with self.assertRaises(KeyError):
            scheduler._run_benchmark_and_process(context)


if __name__ == "__main__":
    unittest.main()
