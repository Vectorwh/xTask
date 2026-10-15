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
    swebench_agent_container_timeout,
    validate_aisbench_swebench_task,
)
from xtask.core import result_processor, scheduler
from xtask.core.external_service import requires_external_endpoint
from xtask.core.task_scheduler import TaskExecutionContext
from xtask.utils.container_manager import CommandResult
from xtask.utils.host_executor import SshHostExecutor


_SETUP_RUNNER = "xtask.benchmarks.aisbench.swebench_x86_setup.setup_aisbench_swe_runner"


def _fake_runner_download(_remote_path, local_path):
    Path(local_path).mkdir(parents=True)
    return True


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
    def test_agent_container_timeout_is_explicit_and_swe_only(self):
        task = _swe_task(swebench={"reuse": False, "runner": {
            "host": "127.0.0.1", "agent_container_timeout": "4h",
        }})
        self.assertEqual(swebench_agent_container_timeout(task), "4h")
        with tempfile.TemporaryDirectory() as tmp:
            swe_generator.generate_aisbench_swe_script(task, tmp)
            shell = (Path(tmp) / "run_aisbench_swe.sh").read_text()
            self.assertIn("XTASK_SWEBENCH_AGENT_CONTAINER_TIMEOUT=4h", shell)
            self.assertTrue((Path(tmp) / ".xtask_swe_copy_fallback/sitecustomize.py").is_file())
        for bad in ("0h", "4 hours", "1h; echo bad", True):
            task["benchmark"]["swebench"]["runner"]["agent_container_timeout"] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "positive duration"):
                swebench_agent_container_timeout(task)

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
        self.assertNotIn("import os", config)
        self.assertNotIn("os.path.abspath", config)
        self.assertIn("SWEBenchEvalTask", config)
        self.assertIn('name=\'verified_mini\'', config)
        self.assertIn('model=\'Qwen3-8B\'', config)
        self.assertIn("http://11.87.191.75:8800/v1", config)
        self.assertIn("LiteLLMChat", config)
        self.assertNotIn("gsm8k", config)
        self.assertIn("-m all", script)
        self.assertIn("--reuse", script)
        self.assertNotIn('--work-dir "$PWD"', script)
        self.assertIn("ais_bench", script)
        self.assertNotIn("/workspace/tools/download_ais_bench.sh", script)

    def test_home_placeholders_expand_for_remote_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _swe_task(
                swebench={
                    "dataset_path": "${HOME}/xtask_swe/runtime/datasets/swebench_verified_mini",
                    "ais_bench_bin": "${HOME}/xtask_swe/runtime/ais_bench-docker",
                    "setup": {
                        "python": "${HOME}/xtask_swe/runtime/python3.11-docker",
                    },
                    "runner": {
                        "host": "10.20.30.50",
                        "work_dir": "${HOME}/xtask_swe/work/swe_case",
                    },
                }
            )
            fake = mock.Mock()
            fake.run.side_effect = [
                CommandResult(
                    success=True, exit_code=0, stdout="/home/remote-user",
                    stderr="", execution_time=0.1,
                ),
                CommandResult(
                    success=True, exit_code=0, stdout="ok", stderr="",
                    execution_time=1.0,
                ),
            ]
            fake.upload.return_value = True
            fake.download.side_effect = _fake_runner_download
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ) as setup:
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)

            self.assertTrue(result.success)
            self.assertEqual(setup.call_args.args[3], "/home/remote-user/xtask_swe/work/swe_case")
            self.assertEqual(
                setup.call_args.kwargs["install_dir"],
                "/home/remote-user/xtask_swe/runtime",
            )
            config = (Path(tmp) / "aisbench_swe_config.py").read_text(encoding="utf-8")
            self.assertIn("/home/remote-user/xtask_swe/runtime/datasets/swebench_verified_mini", config)
            script = (Path(tmp) / "run_aisbench_swe.sh").read_text(encoding="utf-8")
            self.assertIn("/home/remote-user/xtask_swe/runtime/ais_bench-docker", script)

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

    def test_new_swe_run_writes_directly_under_case_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "swe_case"
            task = _swe_task(swebench={"reuse": False})
            swe_generator.generate_aisbench_swe_script(task, str(root))
            fake_bin = Path(tmp) / "fake_ais_bench"
            _write_exec(
                fake_bin,
                """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
assert args[args.index('--work-dir') + 1] == str(pathlib.Path.cwd())
run = pathlib.Path(args[args.index('--work-dir') + 1]) / '20261010_120000'
(run / 'configs').mkdir(parents=True)
(run / 'configs' / 'current.py').write_text('work_dir="current"')
(run / 'results' / 'swebench').mkdir(parents=True)
(run / 'results' / 'swebench' / 'swebench_verified_mini.json').write_text(
    json.dumps({'accuracy': 1.0, 'resolved_instances': 3, 'total_instances': 3}))
(run / 'predictions' / 'swebench').mkdir(parents=True)
(run / 'predictions' / 'swebench' / 'swebench_verified_mini.json').write_text(
    json.dumps({'django__django-11790': {'instance_id': 'django__django-11790', 'model_patch': 'diff'}}))
""",
            )
            legacy = root / "outputs/default/20990101_010101/results/swebench"
            legacy.mkdir(parents=True)
            (legacy / "swebench_verified_mini.json").write_text(
                json.dumps({"accuracy": 0.0, "resolved_instances": 0}), encoding="utf-8"
            )
            initial = [path.name for path in root.iterdir()]
            env = os.environ.copy()
            env.update(AIS_BENCH_BIN=str(fake_bin), SWE_WORK_DIR=str(root))
            completed = subprocess.run(
                ["bash", str(root / "run_aisbench_swe.sh")],
                env=env, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result, output_dir = aisbench_swe_post_process(
                task, str(root), initial, require_new_direct_run=True
            )
            self.assertEqual(output_dir, str(root / "20261010_120000"))
            self.assertEqual(result["resolved_rate"], 1.0)
            self.assertIn("20261010_120000", result["predictions_file"])
            self.assertFalse((root / "outputs/default/20261010_120000").exists())

    def test_explicit_workdir_and_reuse_keep_previous_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for swebench in (
                {"reuse": True},
                {"reuse": False, "extra_args": ["--work-dir=/tmp/custom"]},
            ):
                task = _swe_task(swebench=swebench)
                swe_generator.generate_aisbench_swe_script(task, tmp)
                script = (Path(tmp) / "run_aisbench_swe.sh").read_text(encoding="utf-8")
                self.assertNotIn('--work-dir "$PWD"', script)


class SwebenchExecuteTests(unittest.TestCase):
    def test_remote_result_merge_preserves_live_xllm_and_archived_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "swe_case"
            local_log = root / "xllm_log/master-0/log/node_0.log"
            local_log.parent.mkdir(parents=True)
            local_log.write_text("current NPU log", encoding="utf-8")
            old_trial_log = root / "20261009_120000/xllm_log/master-0/log/node_0.log"
            old_trial_log.parent.mkdir(parents=True)
            old_trial_log.write_text("old trial log", encoding="utf-8")
            legacy = root / "outputs/default/20261009_130000/results/swebench/old.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("old result", encoding="utf-8")
            config = root / "aisbench_swe_config.py"
            config.write_text("current config", encoding="utf-8")

            with tempfile.TemporaryDirectory() as staging:
                remote = Path(staging)
                remote_log = remote / "xllm_log/master-0/log/node_0.log"
                remote_log.parent.mkdir(parents=True)
                remote_log.write_text("stale remote log", encoding="utf-8")
                stale_trial_log = remote / "20261009_120000/xllm_log/master-0/log/node_0.log"
                stale_trial_log.parent.mkdir(parents=True)
                stale_trial_log.write_text("stale archived log", encoding="utf-8")
                new_result = remote / "20261010_120000/results/swebench/swebench_verified_mini.json"
                new_result.parent.mkdir(parents=True)
                new_result.write_text('{"accuracy": 1.0}', encoding="utf-8")
                new_legacy = remote / "outputs/default/20261009_130000/results/swebench/new.json"
                new_legacy.parent.mkdir(parents=True)
                new_legacy.write_text("new result", encoding="utf-8")
                (remote / "aisbench_swe_config.py").write_text("stale config", encoding="utf-8")
                swe_generator._merge_swebench_runner_outputs(
                    remote, root, "run_aisbench_swe.sh"
                )

            self.assertEqual(local_log.read_text(), "current NPU log")
            self.assertEqual(old_trial_log.read_text(), "old trial log")
            self.assertEqual(config.read_text(), "current config")
            self.assertTrue((root / "20261010_120000/results/swebench/swebench_verified_mini.json").exists())
            self.assertEqual(legacy.read_text(), "old result")
            self.assertEqual((legacy.parent / "new.json").read_text(), "new result")

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
            live_log = Path(tmp) / "xllm_log/master-0/log/node_0.log"
            live_log.parent.mkdir(parents=True)
            live_log.write_text("still being written by NPU", encoding="utf-8")
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

            def download_remote_results(_remote_path, local_path):
                result_dir = Path(local_path) / "20261010_120000/results/swebench"
                result_dir.mkdir(parents=True)
                (result_dir / "swebench_verified_mini.json").write_text(
                    '{"accuracy": 1.0}', encoding="utf-8"
                )
                return True

            fake.download.side_effect = download_remote_results
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ) as setup, mock.patch(
                "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
                return_value=None,
            ):
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)
            live_log_content = live_log.read_text(encoding="utf-8")
            merged_result_exists = (
                Path(tmp) / "20261010_120000/results/swebench/swebench_verified_mini.json"
            ).is_file()
        self.assertTrue(result.success)
        fake.upload.assert_called_once()
        fake.download.assert_called_once()
        self.assertEqual(live_log_content, "still being written by NPU")
        self.assertTrue(merged_result_exists)
        self.assertEqual(setup.call_args.args[3], "/tmp/swe")
        self.assertEqual(
            setup.call_args.kwargs["install_dir"],
            "/home/remote-user/xtask_swe/runtime",
        )
        command = fake.run.call_args.args[0]
        self.assertIn("SWE_WORK_DIR=/tmp/swe", command)
        self.assertIn("AIS_BENCH_BIN=/usr/local/bin/ais_bench", command)

    def test_remote_ssh_upload_uses_swe_only_exact_directory_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _swe_task(swebench={"runner": {"host": "10.20.30.50", "work_dir": "/tmp/swe"}})
            fake = mock.create_autospec(SshHostExecutor, instance=True)
            fake.run.side_effect = [
                CommandResult(True, 0, "/home/remote-user", "", 0.1),
                CommandResult(True, 0, "ok", "", 0.1),
            ]
            fake.upload_dir_to_exact_path.return_value = True
            fake.download.side_effect = _fake_runner_download
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ), mock.patch(
                "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
                return_value=None,
            ):
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)

            self.assertTrue(result.success)
            fake.upload_dir_to_exact_path.assert_called_once_with(tmp, "/tmp/swe")
            fake.upload.assert_not_called()

    def test_default_dataset_is_synced_and_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _swe_task(swebench={"runner": {"host": "10.20.30.50"}})
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
            fake.download.side_effect = _fake_runner_download
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ), mock.patch(
                "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
                return_value="/home/remote-user/xtask_swe/runtime/swebench_datasets/verified_mini",
            ) as prepare:
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)

            self.assertTrue(result.success)
            prepare.assert_called_once()
            self.assertEqual(
                prepare.call_args.args[3],
                "/home/remote-user/xtask_swe/runtime",
            )
            config = (Path(tmp) / "aisbench_swe_config.py").read_text(encoding="utf-8")
            self.assertIn(
                "/home/remote-user/xtask_swe/runtime/swebench_datasets/verified_mini",
                config,
            )

    def test_explicit_dataset_path_is_not_replaced_by_default_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _swe_task(
                swebench={
                    "dataset_path": "/tmp/custom-swebench-dataset",
                    "runner": {"host": "10.20.30.50"},
                }
            )
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
            fake.download.side_effect = _fake_runner_download
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ), mock.patch(
                "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
                return_value="/should/not/be/used",
            ) as prepare:
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)

            self.assertTrue(result.success)
            prepare.assert_not_called()
            config = (Path(tmp) / "aisbench_swe_config.py").read_text(encoding="utf-8")
            self.assertIn("/tmp/custom-swebench-dataset", config)
            self.assertNotIn("/should/not/be/used", config)

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
            fake.download.side_effect = _fake_runner_download
            expected_prefix = "/home/remote-user/xtask_swe/swe_case_"
            with mock.patch.object(
                swe_generator, "create_host_executor", return_value=fake
            ), mock.patch.object(
                swe_generator, "is_local_host", return_value=False
            ), mock.patch(
                _SETUP_RUNNER, return_value=_ok_setup()
            ) as setup, mock.patch(
                "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
                return_value=None,
            ):
                result = swe_generator.execute_aisbench_swe_benchmark(task, tmp)
        self.assertTrue(result.success)
        self.assertTrue(setup.call_args.args[3].startswith(expected_prefix))
        self.assertEqual(
            setup.call_args.kwargs["install_dir"],
            "/home/remote-user/xtask_swe/runtime",
        )
        work_dir = setup.call_args.args[3]
        fake.upload.assert_called_once_with(tmp, work_dir)
        fake.download.assert_called_once()
        self.assertEqual(fake.download.call_args.args[0], work_dir)
        self.assertNotEqual(fake.download.call_args.args[1], tmp)
        self.assertFalse(Path(fake.download.call_args.args[1]).exists())
        command = fake.run.call_args.args[0]
        self.assertIn(f"SWE_WORK_DIR={work_dir}", command)

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

    def test_setup_script_pips_aisbench_clones_mini_and_pins_swe(self):
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
if echo "$*" | grep -q 'import numpy, pyarrow, datasets'; then
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
            self.assertIn("swebench==4.1.0", recorded)
            self.assertIn("datasets==3.6.0", recorded)
            self.assertIn("numpy==1.26.4", recorded)
            self.assertIn("pyarrow==19.0.1", recorded)
            self.assertIn("import numpy, pyarrow, datasets", recorded)
            self.assertTrue((install / "mini-swe-agent" / "src").is_dir())
            self.assertFalse((install / "SWE-bench").exists())

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


class SwebenchDefaultDatasetTests(unittest.TestCase):
    def test_default_dataset_archive_syncs_to_x86_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "swebench" / "verified_mini"
            (dataset / "data").mkdir(parents=True)
            (dataset / "data" / "test-00000-of-00001.parquet").write_text(
                "parquet", encoding="utf-8"
            )
            fake = mock.Mock()
            fake.sync_path.return_value = True
            with mock.patch.object(
                swe_setup,
                "get_dependence_package_path",
                return_value=str(root),
            ), mock.patch.object(
                swe_setup, "is_local_host", return_value=False
            ):
                result = swe_setup.prepare_default_swebench_dataset(
                    _swe_task(), fake, "10.20.30.50", "/runtime"
                )

            self.assertEqual(result, "/runtime/swebench_datasets/verified_mini")
            fake.sync_path.assert_called_once_with(
                str(dataset),
                "/runtime/swebench_datasets/verified_mini",
                timeout=swe_setup.DEFAULT_SETUP_TIMEOUT_S,
            )

    def test_missing_default_dataset_falls_back_to_online(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = mock.Mock()
            with mock.patch.object(
                swe_setup,
                "get_dependence_package_path",
                return_value=tmp,
            ):
                result = swe_setup.prepare_default_swebench_dataset(
                    _swe_task(), fake, "10.20.30.50", "/runtime"
                )
            self.assertIsNone(result)
            fake.sync_path.assert_not_called()

    def test_local_runner_uses_default_archive_without_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "swebench" / "verified_mini"
            (dataset / "data").mkdir(parents=True)
            (dataset / "data" / "test-00000-of-00001.parquet").write_text(
                "parquet", encoding="utf-8"
            )
            fake = mock.Mock()
            with mock.patch.object(
                swe_setup,
                "get_dependence_package_path",
                return_value=str(root),
            ), mock.patch.object(
                swe_setup, "is_local_host", return_value=True
            ):
                result = swe_setup.prepare_default_swebench_dataset(
                    _swe_task(), fake, "127.0.0.1", "/runtime"
                )
            self.assertEqual(result, str(dataset))
            fake.sync_path.assert_not_called()


class SwebenchPostprocessTests(unittest.TestCase):
    @staticmethod
    def _write_eval(directory: Path, accuracy: float) -> None:
        target = directory / "results/swebench/swebench_verified_mini.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"accuracy": accuracy, "total_instances": 3,
                        "resolved_instances": round(accuracy * 3)}),
            encoding="utf-8",
        )

    def test_new_direct_run_is_selected_over_newer_legacy_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "outputs/default/20990101_010101"
            self._write_eval(legacy, 0.0)
            earlier_direct = root / "20261009_120000"
            (earlier_direct / "configs").mkdir(parents=True)
            self._write_eval(earlier_direct, 0.0)
            previous = [entry.name for entry in root.iterdir()]
            current = root / "20261010_120000"
            (current / "configs").mkdir(parents=True)
            self._write_eval(current, 1.0)
            result, output_dir = aisbench_swe_post_process(
                _swe_task(swebench={"reuse": False}), tmp, previous,
                require_new_direct_run=True,
            )
            self.assertEqual(output_dir, str(current))
            self.assertEqual(result["resolved_rate"], 1.0)
            self.assertEqual(result["resolved_instances"], 3)

    def test_missing_current_run_never_passes_using_legacy_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_eval(root / "outputs/default/20261009_120000", 1.0)
            old_date = root / "20261009_123000"
            self._write_eval(old_date, 1.0)
            (old_date / "configs").mkdir()
            previous = [entry.name for entry in root.iterdir()]
            with self.assertRaisesRegex(FileNotFoundError, "refusing to reuse historical"):
                aisbench_swe_post_process(
                    _swe_task(swebench={"reuse": False}), tmp, previous,
                    require_new_direct_run=True,
                )

    def test_infer_only_new_run_does_not_pick_old_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_eval(root / "outputs/default/20990101_010101", 1.0)
            previous = [entry.name for entry in root.iterdir()]
            current = root / "20261010_120000"
            (current / "configs").mkdir(parents=True)
            pred_dir = current / "predictions/swebench"
            pred_dir.mkdir(parents=True)
            (pred_dir / "swebench_verified_mini.json").write_text(
                json.dumps({"django__django-11790": {
                    "instance_id": "django__django-11790", "model_patch": "diff"
                }}), encoding="utf-8"
            )
            result, output_dir = aisbench_swe_post_process(
                _swe_task(swebench={"mode": "infer", "reuse": False}), tmp, previous,
                require_new_direct_run=True,
            )
            self.assertEqual(output_dir, str(current))
            self.assertIsNone(result["result_file"])
            self.assertEqual(result["submitted_instances"], 1)

    def test_legacy_reuse_result_still_loads_from_outputs_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "outputs/default/20261009_120000"
            self._write_eval(old, 1.0)
            result, output_dir = aisbench_swe_post_process(
                _swe_task(swebench={"reuse": True}), tmp, [],
            )
            self.assertEqual(output_dir, str(old))
            self.assertEqual(result["resolved_rate"], 1.0)

    def test_harness_exit_without_metrics_is_reported_as_a_harness_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "outputs/default/20261009_120000"
            self._write_eval(old, 1.0)
            previous = [entry.name for entry in root.iterdir()]
            current = root / "20261010_120000"
            (current / "configs").mkdir(parents=True)
            result_file = current / "results/swebench/swebench_verified_mini.json"
            result_file.parent.mkdir(parents=True)
            result_file.write_text(
                json.dumps({"harness_exit_code": 1, "run_id": "swebench"}),
                encoding="utf-8",
            )
            task = _swe_task(swebench={"reuse": False})
            result, output_dir = aisbench_swe_post_process(
                task, tmp, previous, require_new_direct_run=True
            )
            self.assertEqual(output_dir, str(current))
            self.assertEqual(result["result_file"], str(result_file))
            self.assertEqual(result["harness_exit_code"], 1)
            self.assertIsNone(result["resolved_rate"])
            self.assertFalse(result_processor.verify_result(
                {"benchmark_success": True, "benchmark_swebench_result": result}, task
            ))
            category, reason = result_processor._classify_failure(
                {"benchmark_success": True, "benchmark_swebench_result": result}, task
            )
            self.assertEqual(category, result_processor.FAILURE_BENCHMARK)
            self.assertIn("harness_exit_code=1", reason)

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

    def test_harness_failure_classification_requires_explicit_swe_route(self):
        task = _swe_task()
        task["benchmark"]["benchmark_type"] = "evalscope"
        result = {
            "success": False,
            "benchmark_success": True,
            "message": "unrelated failure after benchmark",
            "benchmark_swebench_result": {
                "resolved_rate": 0.5,
                "harness_exit_code": 1,
            },
        }
        category, reason = result_processor._classify_failure(result, task)
        self.assertEqual(category, result_processor.FAILURE_OTHER)
        self.assertNotIn("harness_exit_code", reason)

    def test_perf_failure_classification_ignores_leftover_swe_report(self):
        task = _perf_task()
        result = {
            "success": False,
            "benchmark_success": True,
            "message": "non-SWE postprocessing failed",
            "benchmark_swebench_result": {"harness_exit_code": 1},
        }
        category, reason = result_processor._classify_failure(result, task)
        self.assertNotEqual(category, result_processor.FAILURE_BENCHMARK)
        self.assertNotIn("harness_exit_code", reason)

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
            ) as post_process, mock.patch.object(
                scheduler, "_execute_benchmark_with_npu_guard"
            ) as npu_guard:
                scheduler._run_benchmark_and_process(context)
        execute.assert_called_once()
        self.assertFalse(post_process.call_args.kwargs["require_new_direct_run"])
        npu_guard.assert_not_called()
        self.assertTrue(context.result["success"])
        self.assertEqual(context.result["benchmark_swebench_result"]["resolved_rate"], 0.4)
        self.assertNotIn("container_config", context.state)

    def test_new_swe_run_requires_current_output_without_touching_perf(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _swe_task(swebench={"reuse": False})
            context = TaskExecutionContext("task.yaml", tmp, result={})
            runtimes = [mock.Mock()]
            context.state.update(
                service_healthy=True,
                task_config=task,
                engine_config={},
                service_runtimes=runtimes,
            )
            fake_result = CommandResult(True, 0, "ok", "", 1.0)
            with mock.patch(
                "xtask.benchmarks.aisbench.generate_aisbench_swe_script.execute_aisbench_swe_benchmark",
                return_value=fake_result,
            ), mock.patch(
                "xtask.benchmarks.aisbench.aisbench_swe_postprocess.aisbench_swe_post_process",
                return_value=({"resolved_rate": 1.0}, tmp),
            ) as post_process, mock.patch.object(
                scheduler, "_execute_benchmark_with_npu_guard"
            ) as npu_guard, mock.patch.object(
                scheduler, "_archive_service_group_logs"
            ) as archive_logs:
                scheduler._run_benchmark_and_process(context)
            self.assertTrue(post_process.call_args.kwargs["require_new_direct_run"])
            npu_guard.assert_not_called()
            archive_logs.assert_called_once_with(runtimes, tmp, tmp)
            self.assertTrue(context.result["success"])

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
