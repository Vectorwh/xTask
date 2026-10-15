"""The Docker archive workaround is opt-in and only retries mount ENOSPC."""

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import requests

from xtask.benchmarks.aisbench import swebench_docker_copy_fallback as fallback
from xtask.benchmarks.aisbench.generate_aisbench_swe_script import (
    generate_aisbench_swe_script,
)


class CopyFallbackTests(unittest.TestCase):
    def test_opted_in_agent_container_timeout_changes_only_mini_docker_environment(self):
        module = types.ModuleType("minisweagent.environments.docker")

        class DockerEnvironment:
            def __init__(self, **kwargs):
                self.options = kwargs

        module.DockerEnvironment = DockerEnvironment
        original = DockerEnvironment.__init__
        packages = {
            "minisweagent": types.ModuleType("minisweagent"),
            "minisweagent.environments": types.ModuleType("minisweagent.environments"),
            "minisweagent.environments.docker": module,
        }
        try:
            with mock.patch.dict(sys.modules, packages):
                fallback.install_minisweagent_container_timeout("4h")
                fallback.install_minisweagent_container_timeout("4h")
                self.assertEqual(DockerEnvironment(image="test").options,
                                 {"image": "test", "container_timeout": "4h"})
                self.assertEqual(DockerEnvironment(image="test", container_timeout="2h").options[
                    "container_timeout"], "4h")
                with self.assertRaisesRegex(ValueError, "Invalid SWE-bench"):
                    fallback.install_minisweagent_container_timeout("4h;bad")
        finally:
            DockerEnvironment.__init__ = original

    def setUp(self):
        class APIError(Exception):
            pass

        class Container:
            def __init__(self, error=None):
                self.id = "f" * 64
                self.error = error

            def put_archive(self, path, data):
                if self.error is not None:
                    raise self.error
                return True

        self.APIError = APIError
        self.Container = Container
        self.original = Container.put_archive
        modules = {
            "docker": types.ModuleType("docker"),
            "docker.errors": types.ModuleType("docker.errors"),
            "docker.models": types.ModuleType("docker.models"),
            "docker.models.containers": types.ModuleType("docker.models.containers"),
        }
        modules["docker.errors"].APIError = APIError
        modules["docker.models.containers"].Container = Container
        self.modules = mock.patch.dict(sys.modules, modules)
        self.modules.start()

    def tearDown(self):
        self.Container.put_archive = self.original
        self.modules.stop()

    def test_successful_archive_uses_unmodified_docker_api(self):
        fallback.install()
        with mock.patch.object(fallback.subprocess, "run") as run:
            self.assertTrue(self.Container().put_archive("/tmp", b"tar-data"))
            run.assert_not_called()

    def test_only_mount_enospc_uses_exec_tar(self):
        fallback.install()
        fallback.install()  # Idempotent when sitecustomize is imported twice.
        error = self.APIError("500 mount /etc/hostname flags: 0x5000: no space left on device")
        with mock.patch.object(fallback.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, b"", b"")
            self.assertTrue(self.Container(error).put_archive("/tmp", b"tar-data"))
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["docker", "exec", "-i", "f" * 64, "tar", "-xf", "-", "-C", "/tmp"])
            self.assertEqual(kwargs["input"], b"tar-data")
            self.assertEqual(run.call_count, 1)

    def test_other_errors_and_unsafe_inputs_never_fallback(self):
        fallback.install()
        error = self.APIError("404 no such container")
        enospc = self.APIError("mount /etc/hostname: no space left on device")
        with mock.patch.object(fallback.subprocess, "run") as run:
            with self.assertRaises(self.APIError):
                self.Container(error).put_archive("/tmp", b"tar-data")
            with self.assertRaises(self.APIError):
                self.Container(enospc).put_archive("relative/path", b"tar-data")
            with self.assertRaises(self.APIError):
                self.Container(enospc).put_archive("/tmp", object())
            run.assert_not_called()

    def test_fallback_tar_failure_is_reported(self):
        fallback.install()
        error = self.APIError("mount /etc/hostname: no space left on device")
        with mock.patch.object(fallback.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, b"", b"tar: write failed")
            with self.assertRaisesRegex(RuntimeError, "tar: write failed"):
                self.Container(error).put_archive("/tmp", b"tar-data")

    def test_minisweagent_overlay_create_retries_only_matching_error(self):
        class Environment:
            calls = 0

            def _start_container(self):
                self.calls += 1
                if self.calls <= 2:
                    raise subprocess.CalledProcessError(
                        125, ["docker", "run"],
                        stderr="Error: error creating overlay mount: no space left on device",
                    )
                self.container_id = "instance-id"

        original = Environment._start_container
        modules = {
            "minisweagent": types.ModuleType("minisweagent"),
            "minisweagent.environments": types.ModuleType("minisweagent.environments"),
            "minisweagent.environments.docker": types.ModuleType("minisweagent.environments.docker"),
        }
        modules["minisweagent.environments.docker"].DockerEnvironment = Environment
        with mock.patch.dict(sys.modules, modules), mock.patch.object(fallback.time, "sleep") as sleep:
            try:
                fallback.install_minisweagent_docker_retry()
                fallback.install_minisweagent_docker_retry()
                env = Environment()
                env._start_container()
                self.assertEqual(env.calls, 3)
                self.assertEqual(env.container_id, "instance-id")
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
            finally:
                Environment._start_container = original

    def test_minisweagent_unrelated_run_error_does_not_retry(self):
        class Environment:
            calls = 0

            def _start_container(self):
                self.calls += 1
                raise subprocess.CalledProcessError(125, ["docker", "run"], stderr="name already in use")

        original = Environment._start_container
        modules = {
            "minisweagent": types.ModuleType("minisweagent"),
            "minisweagent.environments": types.ModuleType("minisweagent.environments"),
            "minisweagent.environments.docker": types.ModuleType("minisweagent.environments.docker"),
        }
        modules["minisweagent.environments.docker"].DockerEnvironment = Environment
        with mock.patch.dict(sys.modules, modules), mock.patch.object(fallback.time, "sleep") as sleep:
            try:
                fallback.install_minisweagent_docker_retry()
                with self.assertRaises(subprocess.CalledProcessError):
                    Environment()._start_container()
                sleep.assert_not_called()
            finally:
                Environment._start_container = original

    def test_completed_minisweagent_environment_is_cleaned_synchronously(self):
        class Environment:
            def __init__(self):
                self.container_id = "a" * 64

        env = Environment()
        module = types.ModuleType("minisweagent.run.benchmarks.swebench")
        module.get_sb_environment = lambda *args, **kwargs: env
        module.process_instance = lambda *args, **kwargs: module.get_sb_environment().container_id
        original_process = module.process_instance
        original_get = module.get_sb_environment
        parent = types.ModuleType("minisweagent.run.benchmarks")
        parent.swebench = module
        modules = {
            "minisweagent": types.ModuleType("minisweagent"),
            "minisweagent.run": types.ModuleType("minisweagent.run"),
            "minisweagent.run.benchmarks": parent,
            "minisweagent.run.benchmarks.swebench": module,
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(fallback.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            try:
                fallback.install_minisweagent_cleanup_each_instance()
                fallback.install_minisweagent_cleanup_each_instance()
                self.assertEqual(module.process_instance(), "a" * 64)
                self.assertIsNone(env.container_id)
                run.assert_called_once_with(
                    ["docker", "rm", "-f", "a" * 64],
                    capture_output=True, text=True, timeout=60, check=False,
                )
            finally:
                module.process_instance = original_process
                module.get_sb_environment = original_get

    def test_minisweagent_environment_creation_failure_does_not_clean_another_instance(self):
        module = types.ModuleType("minisweagent.run.benchmarks.swebench")
        def fail_get(*args, **kwargs):
            raise RuntimeError("docker run failed")
        module.get_sb_environment = fail_get
        module.process_instance = lambda *args, **kwargs: module.get_sb_environment()
        parent = types.ModuleType("minisweagent.run.benchmarks")
        parent.swebench = module
        modules = {
            "minisweagent": types.ModuleType("minisweagent"),
            "minisweagent.run": types.ModuleType("minisweagent.run"),
            "minisweagent.run.benchmarks": parent,
            "minisweagent.run.benchmarks.swebench": module,
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(fallback.subprocess, "run") as run:
            fallback.install_minisweagent_cleanup_each_instance()
            with self.assertRaisesRegex(RuntimeError, "docker run failed"):
                module.process_instance()
            run.assert_not_called()

    def test_script_shim_is_only_generated_for_opted_in_swe(self):
        cfg = {
            "case_name": "swe_case", "metrics": "swebench", "served_model_name": "model",
            "benchmark": {"benchmark_type": "aisbench", "swebench": {
                "runner": {"host": "10.1.2.3", "docker_copy_fallback": True},
            }},
        }
        with tempfile.TemporaryDirectory() as tmp:
            generate_aisbench_swe_script(cfg, tmp)
            shell = Path(tmp, "run_aisbench_swe.sh").read_text()
            site = Path(tmp, ".xtask_swe_copy_fallback/sitecustomize.py")
            self.assertTrue(site.is_file())
            self.assertIn("XTASK_SWEBENCH_DOCKER_COPY_FALLBACK=1", shell)
            self.assertNotIn("XTASK_SWEBENCH_DOCKER_RUN_RETRY=1", shell)
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", shell)
            cfg["benchmark"]["swebench"]["runner"]["docker_run_retry"] = True
            generate_aisbench_swe_script(cfg, tmp)
            self.assertIn("XTASK_SWEBENCH_DOCKER_RUN_RETRY=1", Path(tmp, "run_aisbench_swe.sh").read_text())
            cfg["benchmark"]["swebench"]["runner"]["clean_infer_container"] = True
            generate_aisbench_swe_script(cfg, tmp)
            self.assertIn("XTASK_SWEBENCH_CLEAN_INFER_CONTAINER=1", Path(tmp, "run_aisbench_swe.sh").read_text())
            cfg["benchmark"]["swebench"]["runner"].pop("clean_infer_container")
            cfg["benchmark"]["swebench"]["runner"].pop("docker_run_retry")
            cfg["benchmark"]["swebench"]["runner"]["github_raw_retry"] = True
            generate_aisbench_swe_script(cfg, tmp)
            self.assertIn("XTASK_SWEBENCH_GITHUB_RAW_RETRY=1", Path(tmp, "run_aisbench_swe.sh").read_text())
            subprocess.run(["bash", "-n", str(Path(tmp, "run_aisbench_swe.sh"))], check=True)
            cfg["benchmark"]["swebench"]["runner"].pop("docker_copy_fallback")
            with tempfile.TemporaryDirectory() as only_network:
                generate_aisbench_swe_script(cfg, only_network)
                self.assertTrue(Path(only_network, ".xtask_swe_copy_fallback/sitecustomize.py").is_file())
            cfg["benchmark"]["swebench"]["runner"].pop("github_raw_retry")
            with tempfile.TemporaryDirectory() as other:
                generate_aisbench_swe_script(cfg, other)
                self.assertFalse(Path(other, ".xtask_swe_copy_fallback").exists())
            cfg["benchmark"]["swebench"]["runner"]["github_raw_retry"] = "true"
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                generate_aisbench_swe_script(cfg, tmp)
            cfg["benchmark"]["swebench"]["runner"].pop("github_raw_retry")
            cfg["benchmark"]["swebench"]["runner"]["docker_copy_fallback"] = "true"
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                generate_aisbench_swe_script(cfg, tmp)

    def test_github_raw_get_retries_transient_proxy_and_http_errors_only(self):
        calls = []
        unavailable = mock.Mock(status_code=503)
        available = mock.Mock(status_code=200)

        def request(_session, method, url, **kwargs):
            calls.append((method, url, kwargs))
            if len(calls) == 1:
                raise requests.exceptions.ProxyError("temporary proxy failure")
            if len(calls) == 2:
                return unavailable
            return available

        with mock.patch.object(requests.sessions.Session, "request", request), mock.patch.object(
            fallback.time, "sleep"
        ) as sleep:
            fallback.install_github_raw_retry()
            fallback.install_github_raw_retry()  # Must not wrap twice.
            session = requests.Session()
            url = "https://raw.githubusercontent.com/django/django/rev/tests/requirements/py3.txt"
            self.assertIs(session.request("GET", url), available)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0][2]["timeout"], (5, 20))
            unavailable.close.assert_called_once()
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

            self.assertIs(session.request("POST", url), available)
            self.assertIs(session.request("GET", "http://11.87.191.75:18004/v1"), available)
            self.assertEqual(len(calls), 5)
            self.assertEqual(sleep.call_count, 2)

    def test_github_raw_retry_is_bounded_and_does_not_retry_bad_certificates(self):
        attempts = []

        def failed(_session, method, url, **kwargs):
            attempts.append((method, url))
            raise requests.exceptions.ReadTimeout("temporary timeout")

        url = "https://raw.githubusercontent.com/django/django/rev/py3.txt"
        with mock.patch.object(requests.sessions.Session, "request", failed), mock.patch.object(
            fallback.time, "sleep"
        ) as sleep:
            fallback.install_github_raw_retry()
            with self.assertRaises(requests.exceptions.ReadTimeout):
                requests.Session().request("GET", url)
            self.assertEqual(len(attempts), 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4])

        def bad_certificate(_session, method, url, **kwargs):
            raise requests.exceptions.SSLError("invalid certificate")

        with mock.patch.object(requests.sessions.Session, "request", bad_certificate), mock.patch.object(
            fallback.time, "sleep"
        ) as sleep:
            fallback.install_github_raw_retry()
            with self.assertRaises(requests.exceptions.SSLError):
                requests.Session().request("GET", url)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
