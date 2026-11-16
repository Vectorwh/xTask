"""CPU-only regression tests for pack, HCCN, credentials and ordered cleanup."""
import copy
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from xtask.core import scheduler
from xtask.core.pd_control_service import build_xllm_service_config, resolve_internal_api_token
from xtask.core.xllm_service_cluster import ManagedServiceProcess, XllmServiceCluster
from xtask.utils.hccn_config import prepare_runtime_hccn
from xtask.utils.secrets import register_secret, register_encoded_command, redact

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HCCN = load_tool("prepare_hccn_config")
PACK = load_tool("xllm_release_pack")


class HccnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "host.conf"
        self.output = self.root / "container.conf"
        self.source.write_text("tls_enable=1\ncertificate=/host/cert\n")
        self.original = self.source.read_bytes()

    def test_copy_preserves_tls_and_queries_all_physical_devices(self):
        query = mock.Mock(side_effect=lambda device: "10.1.1.%s" % (device + 1))
        result = HCCN.prepare(self.source, [0, 4], self.output, query=query)
        self.assertEqual(result["path"], str(self.output))
        self.assertIn("tls_enable=1", self.output.read_text())
        self.assertIn("address_4=10.1.1.5", self.output.read_text())
        self.assertEqual(query.call_args_list, [mock.call(0), mock.call(4)])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)

    def test_conflict_fails_without_touching_host_or_output(self):
        self.source.write_text("address_0=10.1.1.1\ntls_enable=1\n")
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            HCCN.prepare(self.source, [0], self.output, query=lambda _: "10.1.1.2")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(self.output.exists())

    def test_tls_only_unresolved_auto_does_not_mount_incomplete_copy(self):
        result = HCCN.prepare(self.source, [0, 4], self.output, query=lambda _: None)
        self.assertIsNone(result["path"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_strict_requires_every_visible_device(self):
        with self.assertRaisesRegex(ValueError, "devices \\[4\\]"):
            HCCN.prepare(self.source, [0, 4], self.output, "strict",
                         query=lambda device: "10.1.1.1" if device == 0 else None)

    def test_existing_valid_addresses_work_without_tool(self):
        self.source.write_text("address_4=10.1.1.5\ntls_enable=1\n")
        self.assertEqual(HCCN.prepare(self.source, [4], self.output, "strict",
                                     query=lambda _: None)["devices"], [4])

    def test_source_can_never_be_output(self):
        with self.assertRaisesRegex(ValueError, "host source"):
            HCCN.prepare(self.source, [0], self.source, query=lambda _: "10.1.1.1")
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_runtime_uses_executor_on_each_host_and_readonly_mount(self):
        plan = SimpleNamespace(instance_config={"deployment": {"mode": "pd_disaggregated"}},
                               device_ids=[0, 4], output_dir=str(self.root), host="10.0.0.2")
        config = SimpleNamespace(volumes={"/driver": "/driver"})
        executor = mock.Mock()
        def run(command, timeout):
            import shlex
            args = shlex.split(command)
            return SimpleNamespace(success=True, stdout=json.dumps({"path": args[args.index("--output") + 1]}))
        executor.run.side_effect = run
        prepare_runtime_hccn(plan, config, executor)
        self.assertNotIn("/etc/hccn.conf", config.volumes)
        self.assertIn("/etc/hccn.conf:ro", config.volumes.values())
        self.assertIn("--devices 0,4", executor.run.call_args.args[0])
        mount_source = next(path for path, target in config.volumes.items() if target == "/etc/hccn.conf:ro")
        self.assertNotIn(plan.output_dir + "/", mount_source)

    def test_non_pd_does_not_query_or_change_volumes(self):
        plan = SimpleNamespace(instance_config={}, host="unused")
        config = SimpleNamespace(volumes={})
        executor = mock.Mock()
        prepare_runtime_hccn(plan, config, executor)
        executor.run.assert_not_called()
        self.assertEqual(config.volumes, {})


class TokenTests(unittest.TestCase):
    def test_generated_token_is_private_stable_and_unique_per_mapping(self):
        with mock.patch.dict(os.environ, {"XLLM_INTERNAL_TOKEN": ""}):
            control = {}
            token = resolve_internal_api_token(control)
            self.assertEqual(token, resolve_internal_api_token(control))
            self.assertNotEqual(token, resolve_internal_api_token({}))
            self.assertGreaterEqual(len(token), 32)
            self.assertEqual(control, {})  # no credentials in saved YAML/config

    def test_new_run_rotates_only_generated_credentials(self):
        from xtask.core.pd_control_service import begin_service_run
        task = {"deployment": {"xllm_service": {"host": "10.0.0.1"}}}
        control = task["deployment"]["xllm_service"]
        with mock.patch.dict(os.environ, {"XLLM_INTERNAL_TOKEN": ""}):
            old = resolve_internal_api_token(control)
            begin_service_run(task)
            self.assertNotEqual(old, resolve_internal_api_token(control))
            control["internal_api_token"] = "explicit-stable-token"
            begin_service_run(task)
            self.assertEqual(resolve_internal_api_token(control), "explicit-stable-token")

    def test_explicit_yaml_and_environment_precedence(self):
        with mock.patch.dict(os.environ, {"XLLM_INTERNAL_TOKEN": "environment-token"}):
            self.assertEqual(resolve_internal_api_token({}), "environment-token")
            self.assertEqual(resolve_internal_api_token({"internal_api_token": "yaml-token"}), "yaml-token")

    def test_printable_explicit_token_with_spaces_keeps_legacy_contract(self):
        self.assertEqual(resolve_internal_api_token({"internal_api_token": "legacy token with spaces"}),
                         "legacy token with spaces")

    def test_invalid_token_is_rejected_without_echoing_it(self):
        for token in ("non-ascii-中文", "bad\nsecret", "bad\tsecret"):
            with self.assertRaises(ValueError) as error:
                resolve_internal_api_token({"internal_api_token": token})
            self.assertNotIn(token, str(error.exception))

    def test_registered_tokens_and_encoded_commands_are_redacted(self):
        import base64
        token = "sensitive-fixture-long-token"
        register_secret(token)
        command = "service --internal_api_token=" + token
        register_encoded_command(command)
        self.assertNotIn(token, redact(command))
        payload = base64.b64encode(command.encode())
        self.assertNotIn(payload.decode(), redact(payload.decode()))
        self.assertNotIn(repr(list(payload)), redact(repr(list(payload))))

    def test_handler_redacts_child_messages_and_tracebacks(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        register_secret("traceback-fixture-token")
        try:
            raise RuntimeError("traceback-fixture-token")
        except RuntimeError:
            logging.getLogger("child-test").error("traceback-fixture-token", exc_info=True)
        self.assertNotIn("traceback-fixture-token", stream.getvalue())
        self.assertIn("<redacted>", stream.getvalue())


class PackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = self.root / "xllm_pack"
        self.pack.mkdir()
        self.wheel = self.pack / ("xllm_npu-0.11.0-cp%s%s-cp%s%s-linux_%s.whl" %
                                 (*sys.version_info[:2], *sys.version_info[:2], platform.machine()))
        self.wheel.write_bytes(b"wheel fixture")
        self.vendors = self.root / "vendors"
        self.vendors.mkdir()
        (self.vendors / "custom_xllm_math").mkdir()
        (self.vendors / "custom_transformer").mkdir()
        (self.vendors / "config.ini").write_text("fixture")
        self.bundle(self.vendors, self.pack / "vendors-test.tar.gz")
        self.runtime = self.root / "mooncake-v1"
        (self.runtime / "bin").mkdir(parents=True)
        (self.runtime / "lib").mkdir()
        for name in ("mooncake_master", "mooncake_client"):
            binary = self.runtime / "bin" / name
            binary.write_text("#!/bin/bash\nexit 0\n")
            binary.chmod(0o755)
        (self.runtime / "lib/libasio.so").write_bytes(b"runtime lib")
        self.manifest = {"architecture": platform.machine(),
                         "source_wheel_sha256": PACK.sha256(self.wheel), "xllm_commit": "abc123"}
        self.write_runtime()
        self.archive = self.root / "xllm_pack.tar.gz"
        self.bundle(self.pack, self.archive)

    def bundle(self, directory, archive):
        with tarfile.open(archive, "w:gz") as target:
            target.add(directory, arcname=directory.name)

    def write_runtime(self):
        (self.runtime / "MANIFEST.txt").write_text(json.dumps(self.manifest))
        paths = sorted(path for path in self.runtime.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
        (self.runtime / "SHA256SUMS").write_text("".join(
            PACK.sha256(path) + "  " + str(path.relative_to(self.runtime)) + "\n" for path in paths))
        self.bundle(self.runtime, self.pack / "mooncake-runtime-v1.tar.gz")

    def test_new_pack_stages_complete_matched_components(self):
        (self.pack / "flash_linear_attention_npu-test.whl").write_bytes(b"FLA")
        self.bundle(self.pack, self.archive)
        out = self.root / "staged"
        metadata = PACK.prepare(self.archive, out)
        self.assertEqual(metadata["layout"], "release-v1")
        self.assertTrue((out / "xllm_pack/vendors/config.ini").exists())
        self.assertTrue((out / "xllm_pack/third_whl/flash_linear_attention_npu-test.whl").exists())
        self.assertFalse((out / "xllm_pack/etcd").exists())
        self.assertEqual(metadata["xllm_commit"], "abc123")

    def test_checksum_failure_keeps_previous_destination(self):
        Path(str(self.archive) + ".sha256").write_text("0" * 64 + "  xllm_pack.tar.gz\n")
        out = self.root / "staged"
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            PACK.prepare(self.archive, out)
        self.assertFalse(out.exists())
        self.assertFalse(list(self.root.glob(".release-pack-*")))

    def test_wrong_runtime_wheel_pair_rejected_before_install(self):
        self.manifest["source_wheel_sha256"] = "mismatched"
        self.write_runtime()
        self.bundle(self.pack, self.archive)
        with self.assertRaisesRegex(ValueError, "does not match the xLLM wheel"):
            PACK.prepare(self.archive, self.root / "staged")

    def test_publish_does_not_require_installer_python_or_architecture(self):
        self.manifest["architecture"] = "cross-host-build"
        self.write_runtime()
        self.bundle(self.pack, self.archive)
        published = PACK.publish(self.archive, self.root / "releases")
        self.assertTrue(Path(published).exists())
        with self.assertRaisesRegex(ValueError, "architecture"):
            PACK.prepare(Path(published) / "xllm_pack.tar.gz", self.root / "staged")

    def test_wrong_architecture_rejected(self):
        self.manifest["architecture"] = "wrong-architecture"
        self.write_runtime()
        self.bundle(self.pack, self.archive)
        with self.assertRaisesRegex(ValueError, "architecture"):
            PACK.prepare(self.archive, self.root / "staged")

    def test_tampered_runtime_member_rejected(self):
        (self.runtime / "lib/libasio.so").write_bytes(b"tampered")
        self.bundle(self.runtime, self.pack / "mooncake-runtime-v1.tar.gz")
        self.bundle(self.pack, self.archive)
        with self.assertRaisesRegex(ValueError, "member checksum"):
            PACK.prepare(self.archive, self.root / "staged")

    def test_legacy_layout_is_never_sent_to_new_installer(self):
        (self.pack / "xllm-lib-old.tar.gz").touch()
        self.bundle(self.pack, self.archive)
        with self.assertRaisesRegex(ValueError, "Legacy pack"):
            PACK.prepare(self.archive, self.root / "staged")

    def test_path_traversal_rejected(self):
        with tarfile.open(self.archive, "w:gz") as target:
            info = tarfile.TarInfo("../../escape")
            info.size = 1
            target.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            PACK.prepare(self.archive, self.root / "staged")
        self.assertFalse((self.root / "escape").exists())

    def test_publish_append_regenerates_checksum_and_preserves_old_current(self):
        releases = self.root / "releases"
        previous = PACK.publish(self.archive, releases)
        service = self.root / "service"
        service.mkdir()
        shutil.copy2("/bin/true", service / "xllm_master_serving")
        (service / "manifest.txt").write_text("service_commit=service123\n")
        current = PACK.publish(self.archive, releases, service)
        self.assertNotEqual(previous, current)
        self.assertTrue(Path(previous).exists())
        self.assertEqual((releases / "current").resolve(), Path(current).resolve())
        PACK.verify_sidecar(Path(current) / "xllm_pack.tar.gz")
        with tarfile.open(Path(current) / "xllm_pack.tar.gz") as bundle:
            self.assertIn("xllm_pack/xllm_master_serving", bundle.getnames())
        with self.assertRaisesRegex(ValueError, "Missing archived"):
            PACK.publish(self.archive, releases, self.root / "absent")
        self.assertEqual((releases / "current").resolve(), Path(current).resolve())

    def run_installer(self, service=False, invalid_etcd=False):
        workspace = self.root / "workspace"
        (workspace / "tools").mkdir(parents=True)
        shutil.copy2(ROOT / "tools/xllm_release_pack.py", workspace / "tools")
        bins = self.root / "fake-bin"
        bins.mkdir()
        log = self.root / "pip.log"
        def executable(name, content):
            path = bins / name
            path.write_text("#!/bin/bash\n" + content + "\n")
            path.chmod(0o755)
        executable("python3", 'if [[ "$1" == "-m" && "$2" == "pip" ]]; then echo "$*" >> "$PIP_LOG"; exit 0; fi; '
                   'if [[ "$1" == "-c" ]]; then echo "$XLLM_SCRIPTS_DIR_FIXTURE"; exit 0; fi; '
                   'exec ' + sys.executable + ' "$@"')
        executable("xllm", "exit 0")
        executable("numactl", "exit 0")
        executable("ldconfig", "exit 0")
        executable("wget", "echo unexpected-network-download >&2; exit 99")
        if service:
            shutil.copy2("/bin/true", self.pack / "xllm_master_serving")
            self.bundle(self.pack, self.archive)
            executable("etcd", "exit 0")
            executable("etcdctl", "exit 0")
        script = (ROOT / "tools/download_xllm.sh").read_text()
        for before, after in {
            "/workspace": str(workspace), "/usr/lib64": str(self.root / "lib64"),
            "/opt/xllm": str(self.root / "opt/xllm"), "/usr/local/bin": str(bins),
            "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors": str(self.root / "opp/vendors"),
        }.items():
            script = script.replace(before, after)
        (self.root / "lib64").mkdir()
        (self.root / "opp/vendors").mkdir(parents=True)
        installer = workspace / "installer.sh"
        installer.write_text(script)
        selected = workspace / "xllm_pack.tar.gz"
        shutil.copy2(self.archive, selected)
        environment = dict(os.environ, XLLM_RELEASE_PACK=str(selected),
                           XLLM_INSTALL_SERVICE="1" if service else "0",
                           PATH=str(bins) + ":" + os.environ["PATH"],
                           PIP_LOG=str(log), XLLM_SCRIPTS_DIR_FIXTURE=str(bins))
        environment.pop("XLLM_ETCD_BIN_DIR", None)
        if invalid_etcd:
            environment["XLLM_ETCD_BIN_DIR"] = str(self.root / "missing-etcd")
        result = subprocess.run(["bash", str(installer)], env=environment,
                                capture_output=True, text=True, timeout=20)
        return result, log, bins

    def test_complete_new_pack_installer_uses_no_legacy_pack_or_torch_copy(self):
        (self.pack / "flash_linear_attention_npu-fixture.whl").write_bytes(b"FLA")
        self.bundle(self.pack, self.archive)
        result, log, _ = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--no-deps --no-index", log.read_text())
        self.assertIn("third_whl", log.read_text())
        self.assertNotIn("unexpected-network", result.stderr)
        self.assertTrue((self.root / "lib64/libasio.so").exists())
        self.assertTrue((self.root / "opt/xllm/installed-pack.json").exists())
        self.assertFalse(list((self.root / "workspace").glob(".runtime-release.*")))
        self.assertTrue((self.root / "workspace/xllm_pack.tar.gz").is_file())

    def test_new_pack_static_service_with_independent_etcd(self):
        result, _, bins = self.run_installer(service=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((bins / "xllm_master_serving").exists())
        self.assertFalse((self.root / "lib64/librust_tokenizers.so").exists())

    def test_explicit_invalid_etcd_fails_before_engine_install(self):
        result, log, _ = self.run_installer(service=True, invalid_etcd=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("XLLM_ETCD_BIN_DIR", result.stderr)
        self.assertFalse(log.exists())
        self.assertFalse((self.root / "lib64/libasio.so").exists())



class ServiceBuildTests(unittest.TestCase):
    def fixture(self, root, dynamic=False, missing=False):
        source = root / "xllm-service"
        binary = source / "build/xllm_service/xllm_master_serving"
        binary.parent.mkdir(parents=True)
        shutil.copy2("/bin/true", binary)
        (source / "CMakeLists.txt").write_text("fixture")
        (source / "setup.py").write_text("fixture")
        (source / "third_party/xllm").mkdir(parents=True)
        fake = root / "bin"
        fake.mkdir()
        def executable(name, text):
            path = fake / name
            path.write_text("#!/bin/bash\n" + text + "\n")
            path.chmod(0o755)
        executable("python3", "echo \"$*\" >> \"$WORK_ROOT/python.log\"; if [[ \"$1\" != '-c' ]]; then echo --no-testing; fi; exit 0")
        if dynamic:
            executable("readelf", "echo 'Shared library: [librust_tokenizers.so]'")
            if not missing:
                lib = source / "build/xllm_service/tokenizer/release/librust_tokenizers.so"
                lib.parent.mkdir(parents=True)
                lib.write_bytes(b"dynamic tokenizer")
        executable("git", "if [[ \"$*\" == *'rev-parse HEAD'* ]]; then echo fixturecommit; fi; exit 0")
        env = dict(os.environ, WORK_ROOT=str(root), PATH=str(fake) + ":" + os.environ["PATH"],
                   XLLM_SERVICE_BUILD_TESTING="0")
        env.pop("HOST_UID", None)
        env.pop("HOST_GID", None)
        env.pop("XLLM_SERVICE_APPEND_PACK", None)
        return env

    def run_torch_prefix(self, root, discovered, override=None):
        import re
        script = (ROOT / "tools/build_xllm_service_in_container.sh").read_text()
        function = re.search(r"^ensure_torch_cmake_prefix\(\) \{\n.*?^\}", script,
                             re.MULTILINE | re.DOTALL).group(0)
        fake = root / "bin"
        fake.mkdir()
        python = fake / "python3"
        python.write_text("#!/bin/bash\nprintf '%s\\n' \"$DISCOVERED_TORCH_DIR\"\n")
        python.chmod(0o755)
        environment = dict(os.environ, PATH=str(fake)+":"+os.environ["PATH"],
                           DISCOVERED_TORCH_DIR=discovered, CMAKE_PREFIX_PATH="/existing/prefix")
        environment.pop("Torch_DIR", None)
        if override is not None:
            environment["Torch_DIR"] = str(override)
        return subprocess.run(["bash", "-c", "set -euo pipefail\n" + function
                               + '\nensure_torch_cmake_prefix\nprintf "prefix=%s\\n" "$CMAKE_PREFIX_PATH"\n'],
                              env=environment, capture_output=True, text=True, timeout=10)

    def test_torch_prefix_resolves_same_python_and_preserves_explicit_override(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                torch = root / "torch path/share/cmake/Torch"
                torch.mkdir(parents=True)
                (torch / "TorchConfig.cmake").touch()
                result = self.run_torch_prefix(root, str(torch) if not explicit else "/ignored/discovery",
                                              torch if explicit else None)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Torch_DIR="+str(torch), result.stdout)
                self.assertIn("prefix="+str(torch)+":/existing/prefix", result.stdout)

    def test_torch_prefix_missing_config_fails_and_no_torch_keeps_legacy_behavior(self):
        for discovered in ("", "/missing/Torch"):
            with self.subTest(discovered=discovered), tempfile.TemporaryDirectory() as temp:
                result = self.run_torch_prefix(Path(temp), discovered)
                if discovered:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Torch CMake config is missing", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("prefix=/existing/prefix", result.stdout)

    def test_static_build_archives_without_so_and_preserves_testing_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.fixture(root)
            result = subprocess.run(["bash", str(ROOT / "tools/build_xllm_service_in_container.sh")],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            current = root / "pkg/xllm/service_binary/current"
            self.assertTrue((current / "xllm_master_serving").exists())
            self.assertFalse((current / "librust_tokenizers.so").exists())
            self.assertIn("tokenizer_linkage=static", (current / "manifest.txt").read_text())
            self.assertIn("--no-testing", (root / "python.log").read_text())

    def test_dynamic_build_still_requires_and_archives_so(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                env = self.fixture(root, dynamic=True, missing=missing)
                result = subprocess.run(["bash", str(ROOT / "tools/build_xllm_service_in_container.sh")],
                                        env=env, capture_output=True, text=True, timeout=15)
                current = root / "pkg/xllm/service_binary/current"
                if missing:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(current.exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue((current / "librust_tokenizers.so").exists())


class LegacyOwnershipTests(unittest.TestCase):
    def test_legacy_start_and_stop_only_managed_children(self):
        from xtask.core.pd_control_service import XllmControlService, XllmControlConfig
        cfg = XllmControlConfig(host="10.0.0.1", service_binary_profile="legacy")
        manager = mock.Mock()
        runtime = SimpleNamespace(container_manager=manager,
                                  container_config=SimpleNamespace(container_name="fixture"))
        result = SimpleNamespace(success=True, stdout="", stderr="")
        with mock.patch("xtask.utils.container_manager.path_mapping_in_container", side_effect=lambda value: value), \
             mock.patch.object(ManagedServiceProcess, "execute", return_value=result), \
             mock.patch.object(ManagedServiceProcess, "start") as start, \
             mock.patch.object(ManagedServiceProcess, "stop") as stop, \
             mock.patch("xtask.core.pd_control_service.time.sleep"):
            service = XllmControlService(runtime, cfg, "/model", "/tmp/legacy")
            service._start_owned_processes()
            self.assertEqual(start.call_count, 2)
            self.assertIn("--name", start.call_args_list[0].args[0])
            self.assertIn("--internal_api_token=" + cfg.internal_api_token, start.call_args_list[1].args[0])
            self.assertNotIn("--service_deployment_role=COORDINATOR", start.call_args_list[1].args[0])
            service.stop()
            self.assertEqual(stop.call_count, 2)
            manager.exec_in_container.assert_not_called()

    def test_launch_exception_retains_cleanup_ownership(self):
        process = ManagedServiceProcess(SimpleNamespace(), "/tmp/owned", "service.log")
        with mock.patch.object(process, "execute", side_effect=RuntimeError("timeout")):
            with self.assertRaises(RuntimeError):
                process.start(["service"], {})
        self.assertTrue(process.started)

    def test_report_copy_masks_credentials_without_masking_token_metrics(self):
        from xtask.utils.secrets import redact_tree
        source = {"internal_api_token": "yaml-fixture-credential", "output_tokens": 123,
                  "benchmark": {"max_tokens": 1024}}
        copy = redact_tree(source)
        self.assertEqual(copy["internal_api_token"], "<redacted>")
        self.assertEqual(copy["output_tokens"], 123)
        self.assertEqual(copy["benchmark"]["max_tokens"], 1024)
        self.assertEqual(source["internal_api_token"], "yaml-fixture-credential")



class ShutdownTests(unittest.TestCase):
    def test_split_cleanup_orders_members_engine_then_etcd(self):
        cluster = object.__new__(XllmServiceCluster)
        cluster.legacy = None
        cluster.is_split = False
        events = []
        cluster.drain_after_benchmark = mock.Mock(side_effect=lambda: events.append("drain"))
        cluster.stop_members = mock.Mock(side_effect=lambda: events.append("members"))
        cluster.stop = mock.Mock(side_effect=lambda: events.append("etcd"))
        runtime = SimpleNamespace(plan=SimpleNamespace(instance_name="p"))
        context = SimpleNamespace(state={"multi_host": True, "service_runtimes": [runtime],
            "xllm_control_service": cluster, "task_config": {},
            "service_group_archive_paths": [], "engine_config": {}}, output_file_path="/tmp/out")
        with mock.patch.object(scheduler, "get_xtask_config_data", return_value={"ENV_CONFIG": {"CLEAN_RESULTS": True}}), \
             mock.patch.object(scheduler, "_freeze_service_group_logs", return_value={}), \
             mock.patch.object(scheduler, "_separate_shutdown_logs"), \
             mock.patch.object(scheduler, "_wait_xllm_processes_exit", side_effect=lambda *a, **k: events.append("engine")), \
             mock.patch.object(scheduler, "_wait_npu_hbm_released"), \
             mock.patch.object(scheduler, "_wait_hccl_listen_ports_released"), \
             mock.patch.object(scheduler, "_wait_tcpstore_listen_ports_released"), \
             mock.patch.object(scheduler, "_wait_pd_listen_ports_released"), \
             mock.patch.object(scheduler, "_cleanup_xllm_shm_in_container"), \
             mock.patch.object(scheduler, "_cleanup_xllm_shm"), \
             mock.patch("xtask.core.service_group.cleanup_service_group"):
            scheduler._cleanup_scheduler(context)
        self.assertEqual(events, ["drain", "members", "engine", "etcd"])

    def test_optional_drain_grace_is_bounded_wait_not_fabricated_api(self):
        cluster = object.__new__(XllmServiceCluster)
        cluster.shutdown_drain_seconds = 2
        with mock.patch("xtask.core.xllm_service_cluster.time.sleep") as sleep:
            cluster.drain_after_benchmark()
        sleep.assert_called_once_with(2)

    def test_download_command_passes_explicit_package_config(self):
        config = {"deployment": {"xllm_package": {
            "release_pack": "/workspace/pkg/new pack.tar.gz", "etcd_bin_dir": "/workspace/pkg/etcd"
        }}}
        with mock.patch.dict(os.environ, {"XLLM_RELEASE_PACK": "", "XLLM_ETCD_BIN_DIR": ""}):
            command = scheduler._download_xllm_command(True, config)
        self.assertIn("XLLM_RELEASE_PACK='/workspace/pkg/new pack.tar.gz'", command)
        self.assertIn("XLLM_ETCD_BIN_DIR=/workspace/pkg/etcd", command)
        self.assertIn("XLLM_INSTALL_SERVICE=1", command)



if __name__ == "__main__":
    unittest.main()
