"""Opt-in x86 driver bootstrap tests; no Docker daemon or SSH needed."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.benchmarks.aisbench import generate_aisbench_swe_script as generator
from xtask.benchmarks.aisbench.swebench_x86_container import (
    SweRuntimeContainerState,
    cleanup_swebench_runtime_container,
    ensure_swebench_runtime_container,
)
from xtask.benchmarks.aisbench.swebench import swebench_setup
from xtask.utils.container_manager import CommandResult
from xtask.core import result_processor

ROOT = "/home/runner/xtask_swe"
IMAGE = "xllm:a2-x86-cann9-main-0920"


def _fake_runner_download(_remote_path, local_path):
    Path(local_path).mkdir(parents=True)
    return True


def task(container=True):
    swe = {"mode": "all", "runner": {"host": "10.1.2.3"}}
    if container:
        swe["setup"] = {
            "runtime_container": {
                "name": "xtask-swe-python",
                "image": IMAGE,
                "python": "/usr/local/python3.11.15/bin/python3",
                "ais_bench_bin": "/usr/local/python3.11.15/bin/ais_bench",
            }
        }
    return {"metrics": "acc", "served_model_name": "test", "case_name": "swe_case",
            "benchmark": {"benchmark_type": "aisbench", "acc": swe}}


def per_case_task():
    config = task()
    old = config["benchmark"]["acc"].pop("setup")
    old["runtime_container"].pop("name")
    config["benchmark"]["acc"]["runner"].update(old)
    return config


def ok(stdout=""):
    return CommandResult(success=True, exit_code=0, stdout=stdout, stderr="", execution_time=0)


class Executor:
    def __init__(self, inspect=None):
        self.inspected = inspect
        self.calls = []

    def run(self, command, timeout=60):
        self.calls.append(command)
        if command == "command -v docker":
            return ok("/usr/bin/docker\n")
        if command.startswith("docker run "):
            return ok("f" * 64)
        if command == "cat /etc/resolv.conf" or command.startswith("docker exec -i ") and command.endswith(" cat /etc/resolv.conf"):
            return ok("nameserver 172.16.16.16\nnameserver 10.16.16.16\n")
        if command.startswith("docker inspect --type container -f '{{.State.Running}}"):
            return self.inspected or CommandResult(
                success=False, exit_code=1, stdout="", stderr="Error: No such object: xtask-swe-python", execution_time=0
            )
        if command.startswith("docker inspect --type container -f '{{range .Mounts}}"):
            return ok(f"{ROOT}:{ROOT};/var/run/docker.sock:/var/run/docker.sock;")
        return ok("container-id")


class DriverTests(unittest.TestCase):
    def prepare(self, executor, config=None):
        return ensure_swebench_runtime_container(
            config or task(), executor, "/home/runner", ROOT + "/work", ROOT + "/runtime",
            state=SweRuntimeContainerState(),
        )

    def test_sample_yaml_uses_runner_without_shared_container_name(self):
        import yaml

        for name in ("28_single_aisbench_swebench", "31_single_aisbench_swebench_dsv4"):
            with self.subTest(case=name):
                path = Path(__file__).parent / "e2e/xllm/examples/yaml_authoring" / f"{name}.yaml"
                swe = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]["acc"]
                self.assertNotIn("setup", swe)
                self.assertNotIn("name", swe["runner"]["runtime_container"])
                self.assertEqual(swe["runner"]["runtime_container"]["image"], IMAGE)
        external = Path(__file__).parent / "e2e/xllm/examples/yaml_authoring/29_external_aisbench_swebench.yaml"
        swe = yaml.safe_load(external.read_text(encoding="utf-8"))["benchmark"]["acc"]
        self.assertNotIn("runtime_container", swe["runner"])

    def test_unconfigured_path_remains_host_only(self):
        executor = Executor()
        self.assertIsNone(self.prepare(executor, task(container=False)))
        self.assertEqual(executor.calls, [])

    def test_create_mounts_docker_and_writes_wrappers(self):
        executor = Executor()
        paths = self.prepare(executor)
        self.assertEqual(paths, (ROOT + "/runtime/xtask_xtask-swe-python_python",
                                 ROOT + "/runtime/xtask_xtask-swe-python_ais_bench"))
        command = next(item for item in executor.calls if item.startswith("docker run "))
        self.assertIn("--name xtask-swe-python", command)
        self.assertIn("-e TORCH_DEVICE_BACKEND_AUTOLOAD=0", command)
        self.assertIn("--dns 172.16.16.16 --dns 10.16.16.16", command)
        self.assertIn(IMAGE, command)
        self.assertIn(f"src={ROOT},dst={ROOT}", command)
        self.assertIn("src=/var/run/docker.sock,dst=/var/run/docker.sock", command)
        self.assertIn("src=/usr/bin/docker,dst=/usr/local/bin/docker,readonly", command)
        self.assertTrue(any(' -w "$PWD" ' in item for item in executor.calls))
        self.assertTrue(any('docker exec -i -e TORCH_DEVICE_BACKEND_AUTOLOAD=0' in item
                            for item in executor.calls if 'printf %s' in item))
        self.assertTrue(any("docker exec -i xtask-swe-python /usr/local/bin/docker ps" in item for item in executor.calls))

    def test_opted_in_driver_passes_shim_only_to_aisbench(self):
        cfg = per_case_task()
        cfg["benchmark"]["acc"]["runner"]["docker_copy_fallback"] = True
        cfg["benchmark"]["acc"]["runner"]["docker_run_retry"] = True
        cfg["benchmark"]["acc"]["runner"]["clean_infer_container"] = True
        cfg["benchmark"]["acc"]["runner"]["github_raw_retry"] = True
        cfg["benchmark"]["acc"]["runner"]["agent_container_timeout"] = "4h"
        executor = Executor()
        state = SweRuntimeContainerState()
        with mock.patch("xtask.benchmarks.aisbench.swebench_x86_container.time.time", return_value=1791500000):
            ensure_swebench_runtime_container(
                cfg, executor, "/home/runner", ROOT + "/work", ROOT + "/runtime", state=state
            )
        scripts = [__import__("shlex").split(command)[2] for command in executor.calls if command.startswith("printf %s ")]
        self.assertEqual(len(scripts), 2)
        self.assertIn("XTASK_SWEBENCH_DOCKER_COPY_FALLBACK=1", scripts[1])
        self.assertIn("XTASK_SWEBENCH_DOCKER_RUN_RETRY=1", scripts[1])
        self.assertIn("XTASK_SWEBENCH_CLEAN_INFER_CONTAINER=1", scripts[1])
        self.assertIn("XTASK_SWEBENCH_GITHUB_RAW_RETRY=1", scripts[1])
        self.assertIn("XTASK_SWEBENCH_AGENT_CONTAINER_TIMEOUT=4h", scripts[1])
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", scripts[1])
        self.assertIn("XTASK_SWE_PYTHONPATH", scripts[1])
        self.assertNotIn("XTASK_SWE_PYTHONPATH", scripts[0])
        self.assertNotIn("XTASK_SWEBENCH_AGENT_CONTAINER_TIMEOUT", scripts[0])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as wrapper:
            wrapper.write(scripts[1])
            wrapper.flush()
            subprocess.run(["sh", "-n", wrapper.name], check=True)
        self.assertTrue(cleanup_swebench_runtime_container(executor, state))

    def test_runtime_proxy_is_limited_to_benchmark_entrypoint(self):
        cfg = per_case_task()
        cfg["benchmark"]["acc"]["runner"]["docker_copy_fallback"] = True
        container = cfg["benchmark"]["acc"]["runner"]["runtime_container"]
        container["proxy"] = {
            "http": "http://bamboo-proxy.jd.com:80",
            "https": "http://bamboo-proxy.jd.com:80",
            "no_proxy": "localhost,127.0.0.1,11.87.191.75",
        }
        executor = Executor()
        state = SweRuntimeContainerState()
        with mock.patch("xtask.benchmarks.aisbench.swebench_x86_container.time.time", return_value=1791500000):
            ensure_swebench_runtime_container(
                cfg, executor, "/home/runner", ROOT + "/work", ROOT + "/runtime", state=state
            )
        scripts = [__import__("shlex").split(cmd)[2] for cmd in executor.calls if cmd.startswith("printf %s ")]
        self.assertEqual(len(scripts), 2)
        self.assertNotIn("HTTP_PROXY=", scripts[0])
        self.assertIn("-e HTTP_PROXY=http://bamboo-proxy.jd.com:80", scripts[1])
        self.assertIn("-e HTTPS_PROXY=http://bamboo-proxy.jd.com:80", scripts[1])
        self.assertIn("-e NO_PROXY=localhost,127.0.0.1,11.87.191.75", scripts[1])
        run = next(cmd for cmd in executor.calls if cmd.startswith("docker run "))
        self.assertNotIn("HTTP_PROXY=", run)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as wrapper:
            wrapper.write(scripts[1])
            wrapper.flush()
            subprocess.run(["sh", "-n", wrapper.name], check=True)
        self.assertTrue(cleanup_swebench_runtime_container(executor, state))

    def test_bad_runtime_proxy_is_rejected_before_docker_creation(self):
        cfg = per_case_task()
        cfg["benchmark"]["acc"]["runner"]["runtime_container"]["proxy"] = {
            "https": "not-a-url"
        }
        executor = Executor()
        with self.assertRaisesRegex(ValueError, "proxy.https"):
            self.prepare(executor, cfg)
        self.assertFalse(any(cmd.startswith("docker run ") for cmd in executor.calls))

    def test_reuse_existing_driver_and_start_if_stopped(self):
        executor = Executor(ok("true " + IMAGE))
        self.prepare(executor)
        self.assertFalse(any(item.startswith("docker run ") or item.startswith("docker start ") for item in executor.calls))
        stopped = Executor(ok("false " + IMAGE))
        self.prepare(stopped)
        self.assertIn("docker start xtask-swe-python", stopped.calls)
        self.assertFalse(any(item.startswith("docker run ") for item in stopped.calls))

    def test_per_case_driver_name_and_cleanup(self):
        cfg = per_case_task()
        state = SweRuntimeContainerState()
        executor = Executor()
        with mock.patch("xtask.benchmarks.aisbench.swebench_x86_container.time.time", return_value=1791500000):
            paths = ensure_swebench_runtime_container(
                cfg, executor, "/home/runner", ROOT + "/work", ROOT + "/runtime", state=state
            )
        self.assertEqual(state.name, "swe_case-swebench_rt-1791500000")
        self.assertEqual(state.container_id, "f" * 64)
        self.assertTrue(state.remove_on_exit)
        self.assertTrue(all(state.name in path for path in paths))
        create = next(command for command in executor.calls if command.startswith("docker run "))
        self.assertIn("--name swe_case-swebench_rt-1791500000", create)
        self.assertTrue(cleanup_swebench_runtime_container(executor, state))
        self.assertFalse(state.remove_on_exit)
        self.assertIn("docker rm -f " + "f" * 64, executor.calls)
        self.assertTrue(any(command.startswith("rm -f ") and paths[0] in command for command in executor.calls))
        self.assertTrue(cleanup_swebench_runtime_container(executor, state))
        self.assertEqual(sum(command.startswith("docker rm -f ") for command in executor.calls), 1)

    def test_new_runner_never_reuses_an_existing_container(self):
        existing = Executor(ok("true " + IMAGE))
        cfg = per_case_task()
        with mock.patch("xtask.benchmarks.aisbench.swebench_x86_container.time.time", return_value=1791500000):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                self.prepare(existing, cfg)
        self.assertFalse(any(command.startswith("docker run ") or command.startswith("docker rm ") for command in existing.calls))

    def test_legacy_setup_remains_reusable(self):
        cfg = task()
        executor = Executor(ok("true " + IMAGE))
        state = SweRuntimeContainerState()
        ensure_swebench_runtime_container(cfg, executor, "/home/runner", ROOT + "/work", ROOT + "/runtime", state=state)
        self.assertFalse(state.remove_on_exit)
        self.assertTrue(cleanup_swebench_runtime_container(executor, state))
        self.assertFalse(any(command.startswith("docker rm ") for command in executor.calls))

    def test_runner_setup_fields_override_legacy_without_mutating_config(self):
        cfg = per_case_task()
        cfg["benchmark"]["acc"]["setup"] = {"python": "/old/python", "skip": False}
        cfg["benchmark"]["acc"]["runner"]["python"] = "/new/python"
        cfg["benchmark"]["acc"]["runner"]["skip"] = True
        merged = swebench_setup(cfg)
        self.assertEqual(merged["python"], "/new/python")
        self.assertTrue(merged["skip"])
        self.assertIn("runtime_container", merged)
        self.assertEqual(cfg["benchmark"]["acc"]["setup"]["python"], "/old/python")

    def test_probe_failure_after_creation_is_cleanable_by_id(self):
        class BrokenProbe(Executor):
            def run(self, command, timeout=60):
                if "--version" in command and "docker exec" in command:
                    self.calls.append(command)
                    return CommandResult(False, 1, "", "Python probe failed", 0)
                return super().run(command, timeout=timeout)

        executor = BrokenProbe()
        state = SweRuntimeContainerState()
        with mock.patch("xtask.benchmarks.aisbench.swebench_x86_container.time.time", return_value=1791500000):
            with self.assertRaisesRegex(RuntimeError, "Python probe failed"):
                ensure_swebench_runtime_container(
                    per_case_task(), executor, "/home/runner", ROOT + "/work", ROOT + "/runtime", state=state
                )
        self.assertTrue(state.remove_on_exit)
        self.assertTrue(cleanup_swebench_runtime_container(executor, state))
        self.assertIn("docker rm -f " + "f" * 64, executor.calls)

    def test_cleanup_failure_does_not_delete_other_files(self):
        class FailedRemoval(Executor):
            def run(self, command, timeout=60):
                if command.startswith("docker rm -f "):
                    self.calls.append(command)
                    return CommandResult(False, 1, "", "daemon unavailable", 0)
                return super().run(command, timeout=timeout)

        state = SweRuntimeContainerState(
            container_id="f" * 64, name="swe_case-swebench_rt-1791500000",
            wrappers=(ROOT + "/runtime/py", ROOT + "/runtime/bench"), remove_on_exit=True,
        )
        executor = FailedRemoval()
        with self.assertLogs(level="WARNING"):
            self.assertFalse(cleanup_swebench_runtime_container(executor, state))
        self.assertTrue(state.remove_on_exit)
        self.assertFalse(any(command.startswith("rm -f ") for command in executor.calls))

    def test_repair_host_dns_only_for_managed_driver(self):
        class WrongDNS(Executor):
            def run(self, command, timeout=60):
                if command == "docker exec -i xtask-swe-python cat /etc/resolv.conf":
                    self.calls.append(command)
                    return ok("nameserver 8.8.8.8\n")
                if "index .Config.Labels" in command:
                    self.calls.append(command)
                    return ok(self.label)
                return super().run(command, timeout=timeout)

        existing = WrongDNS(ok("true " + IMAGE))
        existing.label = "true"
        self.prepare(existing)
        self.assertTrue(any("/etc/resolv.conf" in item and "sh -c" in item for item in existing.calls))
        unmanaged = WrongDNS(ok("true " + IMAGE))
        unmanaged.label = "<no value>"
        with self.assertRaisesRegex(RuntimeError, "unmanaged"):
            self.prepare(unmanaged)
        self.assertFalse(any("sh -c" in item for item in unmanaged.calls))

    def test_foreign_image_or_unmounted_path_fails_closed(self):
        wrong = Executor(ok("true other:image"))
        with self.assertRaisesRegex(RuntimeError, "unexpected image"):
            self.prepare(wrong)
        self.assertFalse(any(item.startswith("docker run ") for item in wrong.calls))
        cfg = task()
        cfg["benchmark"]["acc"]["dataset_path"] = "/other/datasets/test.parquet"
        with self.assertRaisesRegex(ValueError, "dataset_path"):
            self.prepare(Executor(), cfg)

    def test_failed_docker_probe_does_not_start_driver(self):
        class NoDocker(Executor):
            def run(self, command, timeout=60):
                if "docker ps >/dev/null" in command:
                    return CommandResult(False, 1, "", "permission denied", 0)
                return super().run(command, timeout=timeout)

        executor = NoDocker()
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            self.prepare(executor)
        self.assertFalse(any(item.startswith("docker run ") for item in executor.calls))

    def test_harness_instance_error_must_not_pass_without_baseline(self):
        cfg = task()
        result = {"benchmark_success": True, "success": False,
                  "benchmark_swebench_result": {
                      "resolved_rate": 0.0, "total_instances": 1,
                      "error_instances": 1, "harness_exit_code": 0,
                  }}
        self.assertFalse(result_processor.verify_result(result, cfg))
        category, reason = result_processor._classify_failure(result, cfg)
        self.assertEqual(category, result_processor.FAILURE_BENCHMARK)
        self.assertIn("error_instances=1", reason)
        result["benchmark_swebench_result"].update(error_instances=0, harness_exit_code=1)
        self.assertFalse(result_processor.verify_result(result, cfg))
        result["benchmark_swebench_result"].update(harness_exit_code=0, unresolved_instances=1)
        self.assertTrue(result_processor.verify_result(result, cfg))

    def test_new_runner_cleans_after_download_and_uses_runner_python(self):
        cfg = per_case_task()
        fake = mock.Mock()
        fake.run.side_effect = lambda command, **kwargs: ok(
            "/home/runner" if command == 'printf %s "$HOME"' else "done"
        )
        fake.upload.return_value = True
        fake.download.side_effect = _fake_runner_download
        wrappers = (ROOT + "/runtime/python-per-case", ROOT + "/runtime/bench-per-case")

        def provision(*args, **kwargs):
            state = kwargs["state"]
            state.name = "swe_case-swebench_rt-1791500000"
            state.container_id = "f" * 64
            state.wrappers = wrappers
            state.remove_on_exit = True
            return wrappers

        from xtask.benchmarks.aisbench.swebench_x86_setup import SweX86SetupResult
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            generator, "create_host_executor", return_value=fake
        ), mock.patch.object(generator, "is_local_host", return_value=False), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_container.ensure_swebench_runtime_container",
            side_effect=provision,
        ), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
            return_value=None,
        ), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_setup.setup_aisbench_swe_runner",
            return_value=SweX86SetupResult(ok(), ais_bench_bin=wrappers[1]),
        ) as install:
            result = generator.execute_aisbench_swe_benchmark(cfg, tmp)
            self.assertTrue(result.success)
            self.assertEqual(install.call_args.args[0]["benchmark"]["acc"]["runner"]["python"], wrappers[0])
            self.assertIn(wrappers[1], Path(tmp, "run_aisbench_swe.sh").read_text())
            calls = fake.mock_calls
            download = next(i for i, call in enumerate(calls) if call[0] == "download")
            removed = next(i for i, call in enumerate(calls)
                           if call[0] == "run" and str(call.args[0]).startswith("docker rm -f "))
            self.assertLess(download, removed)
            self.assertIn("docker rm -f " + "f" * 64, [call.args[0] for call in fake.run.call_args_list])
            self.assertTrue(any(str(call.args[0]).startswith("rm -f ") for call in fake.run.call_args_list))
        self.assertNotIn("python", cfg["benchmark"]["acc"]["runner"])

    def test_new_runner_does_not_hide_failed_cleanup(self):
        cfg = per_case_task()
        fake = mock.Mock()
        fake.run.side_effect = lambda command, **kwargs: (
            CommandResult(False, 1, "", "daemon down", 0)
            if command.startswith("docker rm -f ") else
            ok("/home/runner" if command == 'printf %s "$HOME"' else "done")
        )
        fake.upload.return_value = True
        fake.download.side_effect = _fake_runner_download
        wrappers = (ROOT + "/runtime/py", ROOT + "/runtime/bench")

        def provision(*args, **kwargs):
            state = kwargs["state"]
            state.name = "swe_case-swebench_rt-1791500000"
            state.container_id = "f" * 64
            state.wrappers = wrappers
            state.remove_on_exit = True
            return wrappers

        from xtask.benchmarks.aisbench.swebench_x86_setup import SweX86SetupResult
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            generator, "create_host_executor", return_value=fake
        ), mock.patch.object(generator, "is_local_host", return_value=False), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_container.ensure_swebench_runtime_container",
            side_effect=provision,
        ), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
            return_value=None,
        ), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_setup.setup_aisbench_swe_runner",
            return_value=SweX86SetupResult(ok(), ais_bench_bin=wrappers[1]),
        ):
            with self.assertLogs(level="WARNING"), self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                generator.execute_aisbench_swe_benchmark(cfg, tmp)
        fake.download.assert_called_once()

    def test_new_runner_cleans_after_setup_or_provision_failure(self):
        cfg = per_case_task()
        wrappers = (ROOT + "/runtime/py", ROOT + "/runtime/bench")
        from xtask.benchmarks.aisbench.swebench_x86_setup import SweX86SetupResult
        for fail_at in ("provision", "setup"):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                fake = mock.Mock()
                fake.run.side_effect = lambda command, **kwargs: ok(
                    "/home/runner" if command == 'printf %s "$HOME"' else "cleaned"
                )

                def provision(*args, **kwargs):
                    state = kwargs["state"]
                    state.name = "swe_case-swebench_rt-1791500000"
                    state.container_id = "f" * 64
                    state.wrappers = wrappers
                    state.remove_on_exit = True
                    if fail_at == "provision":
                        raise RuntimeError("probe failed after creation")
                    return wrappers

                failed_setup = SweX86SetupResult(CommandResult(False, 1, "", "pip failed", 0))
                with mock.patch.object(generator, "create_host_executor", return_value=fake), mock.patch.object(
                    generator, "is_local_host", return_value=False
                ), mock.patch(
                    "xtask.benchmarks.aisbench.swebench_x86_container.ensure_swebench_runtime_container",
                    side_effect=provision,
                ), mock.patch(
                    "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
                    return_value=None,
                ), mock.patch(
                    "xtask.benchmarks.aisbench.swebench_x86_setup.setup_aisbench_swe_runner",
                    return_value=failed_setup,
                ) as install:
                    result = generator.execute_aisbench_swe_benchmark(cfg, tmp)
                self.assertFalse(result.success)
                self.assertIn("probe failed" if fail_at == "provision" else "pip failed", result.stderr)
                self.assertEqual(install.called, fail_at == "setup")
                fake.upload.assert_not_called()
                self.assertTrue(any(str(call.args[0]).startswith("docker rm -f ")
                                    for call in fake.run.call_args_list))

    def test_new_runner_cleans_after_upload_or_download_failure(self):
        cfg = per_case_task()
        wrappers = (ROOT + "/runtime/py", ROOT + "/runtime/bench")
        from xtask.benchmarks.aisbench.swebench_x86_setup import SweX86SetupResult
        for fail_at in ("upload", "download"):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                fake = mock.Mock()
                fake.run.side_effect = lambda command, **kwargs: ok(
                    "/home/runner" if command == 'printf %s "$HOME"' else "done"
                )
                fake.upload.return_value = fail_at != "upload"
                fake.download.return_value = fail_at != "download"

                def provision(*args, **kwargs):
                    state = kwargs["state"]
                    state.name = "swe_case-swebench_rt-1791500000"
                    state.container_id = "f" * 64
                    state.wrappers = wrappers
                    state.remove_on_exit = True
                    return wrappers

                with mock.patch.object(generator, "create_host_executor", return_value=fake), mock.patch.object(
                    generator, "is_local_host", return_value=False
                ), mock.patch(
                    "xtask.benchmarks.aisbench.swebench_x86_container.ensure_swebench_runtime_container",
                    side_effect=provision,
                ), mock.patch(
                    "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
                    return_value=None,
                ), mock.patch(
                    "xtask.benchmarks.aisbench.swebench_x86_setup.setup_aisbench_swe_runner",
                    return_value=SweX86SetupResult(ok(), ais_bench_bin=wrappers[1]),
                ):
                    result = generator.execute_aisbench_swe_benchmark(cfg, tmp)
                self.assertFalse(result.success)
                self.assertIn("Failed to " + fail_at, result.stderr)
                self.assertTrue(any(str(call.args[0]).startswith("docker rm -f ")
                                    for call in fake.run.call_args_list))

    def test_execute_injects_managed_entrypoints_before_install(self):
        cfg = task()
        fake = mock.Mock()
        fake.run.side_effect = [ok("/home/runner"), ok("benchmark finished")]
        fake.upload.return_value = True
        fake.download.side_effect = _fake_runner_download
        wrappers = (ROOT + "/runtime/python-managed", ROOT + "/runtime/bench-managed")
        from xtask.benchmarks.aisbench.swebench_x86_setup import SweX86SetupResult
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            generator, "create_host_executor", return_value=fake
        ), mock.patch.object(generator, "is_local_host", return_value=False), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_container.ensure_swebench_runtime_container",
            return_value=wrappers,
        ) as driver, mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_setup.prepare_default_swebench_dataset",
            return_value=None,
        ), mock.patch(
            "xtask.benchmarks.aisbench.swebench_x86_setup.setup_aisbench_swe_runner",
            return_value=SweX86SetupResult(ok(), ais_bench_bin=wrappers[1]),
        ) as install:
            result = generator.execute_aisbench_swe_benchmark(cfg, tmp)
            self.assertTrue(result.success)
            driver.assert_called_once()
            assigned = install.call_args.args[0]["benchmark"]["acc"]
            self.assertEqual(assigned["setup"]["python"], wrappers[0])
            self.assertEqual(assigned["ais_bench_bin"], wrappers[1])
            self.assertIn(wrappers[1], Path(tmp, "run_aisbench_swe.sh").read_text())
        self.assertNotIn("ais_bench_bin", cfg["benchmark"]["acc"])


if __name__ == "__main__":
    unittest.main()
