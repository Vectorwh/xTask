# Copyright 2026 The xTask Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLA archiving and generic third-party wheel installs without builds or NPU use."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "tools/build_xllm_in_container.sh"
INSTALL_SCRIPT = ROOT / "tools/download_xllm.sh"


def _function(script: Path, name: str) -> str:
    source = script.read_text(encoding="utf-8")
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}$", source, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"Cannot locate {name} in {script}")
    return match.group(0)


class FlaWheelFixture(unittest.TestCase):
    temporary_prefix = "fla wheel tests "

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix=self.temporary_prefix)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dist = self.root / "source/dist"
        self.package = self.root / "pkg/xllm/third_whl"
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.calls = self.root / "python-calls.jsonl"
        self.env = os.environ.copy()
        self.env.update(
            FLA_NPU_DIST_DIR=str(self.dist),
            XLLM_THIRD_WHL_DIR=str(self.package),
            PATH=f"{self.fake_bin}:{self.env['PATH']}",
            FLA_TEST_PYTHON_LOG=str(self.calls),
            FLA_TEST_PIP_EXIT="0",
        )
        self._executable(
            "fake-python",
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['FLA_TEST_PYTHON_LOG'], 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "sys.exit(int(os.environ['FLA_TEST_PIP_EXIT']))\n",
        )
        self.env["XLLM_PYTHON"] = str(self.fake_bin / "fake-python")

    def _executable(self, name: str, source: str) -> Path:
        path = self.fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def _wheel(self, directory: Path, version: str = "26.9.1", tier: str = "a3") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"flash_linear_attention_npu_{tier}-{version}"
            "-py3-none-manylinux_2_34_aarch64.whl"
        )
        path.write_bytes(b"fixture wheel payload")
        return path

    def _run(self, body: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + body],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def _archive(self) -> subprocess.CompletedProcess:
        return self._run(
            _function(BUILD_SCRIPT, "archive_optional_fla_npu_wheel")
            + "\narchive_optional_fla_npu_wheel\n"
        )

    def _install(self) -> subprocess.CompletedProcess:
        return self._run(
            _function(INSTALL_SCRIPT, "install_optional_third_party_wheels")
            + "\ninstall_optional_third_party_wheels\n"
        )

    def _python_calls(self) -> list:
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]


class ArchiveFlaWheelTests(FlaWheelFixture):
    def test_missing_source_skips_without_creating_archive(self) -> None:
        result = self._archive()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No FLA wheel produced", result.stdout)
        self.assertFalse(self.package.exists())

    def test_no_fla_wheel_ignores_other_wheels(self) -> None:
        self.dist.mkdir(parents=True)
        (self.dist / "other_package-1.0.whl").write_bytes(b"unrelated")
        result = self._archive()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.package.exists())

    def test_no_new_wheel_removes_stale_archive_but_keeps_other_files(self) -> None:
        stale = self._wheel(self.package, "26.8.0")
        sentinel = self.package / "keep-me.txt"
        sentinel.write_text("unrelated", encoding="utf-8")
        other_wheel = self.package / "other_package-1.0-py3-none-any.whl"
        other_wheel.write_bytes(b"other third-party wheel")
        result = self._archive()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(stale.exists())
        self.assertTrue(sentinel.exists())
        self.assertEqual(other_wheel.read_bytes(), b"other third-party wheel")

    def test_single_wheel_is_archived_readable_with_the_same_filename(self) -> None:
        source = self._wheel(self.dist)
        result = self._archive()
        self.assertEqual(result.returncode, 0, result.stderr)
        archived = self.package / source.name
        self.assertEqual(archived.read_bytes(), source.read_bytes())
        self.assertEqual(archived.stat().st_mode & 0o777, 0o644)
        self.assertIn(str(archived), result.stdout)
        self.assertFalse(list(self.package.glob(".fla-wheel.*")))

    def test_new_wheel_replaces_old_version_only(self) -> None:
        stale = self._wheel(self.package, "26.8.0")
        sentinel = self.package / "keep-me.txt"
        sentinel.write_text("unrelated", encoding="utf-8")
        other_wheel = self.package / "other_package-1.0-py3-none-any.whl"
        other_wheel.write_bytes(b"other third-party wheel")
        source = self._wheel(self.dist)
        result = self._archive()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(stale.exists())
        self.assertTrue((self.package / source.name).is_file())
        self.assertTrue(sentinel.exists())
        self.assertEqual(other_wheel.read_bytes(), b"other third-party wheel")

    def test_rebuild_of_same_version_updates_the_payload(self) -> None:
        archived = self._wheel(self.package)
        archived.write_bytes(b"previous build")
        source = self._wheel(self.dist)
        result = self._archive()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(archived.read_bytes(), source.read_bytes())

    def test_multiple_versions_fail_without_modifying_previous_archive(self) -> None:
        previous = self._wheel(self.package, "26.8.0")
        self._wheel(self.dist)
        self._wheel(self.dist, "26.9.2")
        result = self._archive()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected at most one FLA wheel", result.stderr)
        self.assertTrue(previous.exists())
        self.assertEqual(len(list(self.package.glob("*.whl"))), 1)

    def test_copy_failure_keeps_previous_archive_and_cleans_staging(self) -> None:
        previous = self._wheel(self.package)
        previous.write_bytes(b"previous build")
        self._wheel(self.dist)
        self._executable("cp", "#!/bin/bash\nexit 7\n")
        result = self._archive()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed to stage FLA wheel", result.stderr)
        self.assertEqual(previous.read_bytes(), b"previous build")
        self.assertFalse(list(self.package.glob(".fla-wheel.*")))

    def test_move_failure_keeps_previous_archive_and_cleans_staging(self) -> None:
        previous = self._wheel(self.package)
        previous.write_bytes(b"previous build")
        self._wheel(self.dist)
        self._executable("mv", "#!/bin/bash\nexit 7\n")
        result = self._archive()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed to archive FLA wheel", result.stderr)
        self.assertEqual(previous.read_bytes(), b"previous build")
        self.assertFalse(list(self.package.glob(".fla-wheel.*")))


class InstallThirdPartyWheelTests(FlaWheelFixture):
    def test_missing_archive_skips_without_python(self) -> None:
        self.env.pop("XLLM_PYTHON")
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("跳过可选三方包安装", result.stdout)
        self.assertEqual(self._python_calls(), [])

    def test_empty_archive_skips_without_python(self) -> None:
        self.package.mkdir(parents=True)
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._python_calls(), [])

    def test_non_fla_wheel_is_installed_without_a_package_whitelist(self) -> None:
        self.package.mkdir(parents=True)
        wheel = self.package / "other_package-1.0-py3-none-any.whl"
        wheel.write_bytes(b"third-party wheel")
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._python_calls()[0][-1], str(wheel))

    def test_non_wheel_files_and_nested_wheels_are_ignored(self) -> None:
        self.package.mkdir(parents=True)
        (self.package / "README.txt").write_text("not a wheel", encoding="utf-8")
        (self.package / ".partial.whl.tmp").write_bytes(b"incomplete wheel")
        (self.package / "not-a-file.whl").mkdir()
        self._wheel(self.package / "nested")
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._python_calls(), [])

    def test_no_fla_output_still_installs_other_archived_packages(self) -> None:
        self.package.mkdir(parents=True)
        other = self.package / "other_package-1.0-py3-none-any.whl"
        other.write_bytes(b"other third-party wheel")
        archive_result = self._archive()
        self.assertEqual(archive_result.returncode, 0, archive_result.stderr)
        install_result = self._install()
        self.assertEqual(install_result.returncode, 0, install_result.stderr)
        self.assertEqual(self._python_calls()[0][-1], str(other))

    def test_single_wheel_installs_offline_without_upgrading_dependencies(self) -> None:
        wheel = self._wheel(self.package)
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._python_calls(), [[
            "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-index",
            str(wheel),
        ]])
        self.assertIn("三方 wheel 安装完成", result.stdout)

    def test_a2_wheel_is_also_detected(self) -> None:
        wheel = self._wheel(self.package, tier="a2")
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._python_calls()[0][-1], str(wheel))

    def test_multiple_third_party_wheels_are_passed_to_pip_together(self) -> None:
        fla = self._wheel(self.package)
        other = self.package / "other_package-1.0-py3-none-any.whl"
        other.write_bytes(b"other third-party wheel")
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._python_calls(), [[
            "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-index",
            *sorted([str(fla), str(other)]),
        ]])
        self.assertIn("三方 wheel 安装完成（2 个）", result.stdout)

    def test_install_failure_stops_the_runtime_setup(self) -> None:
        self._wheel(self.package)
        self.env["FLA_TEST_PIP_EXIT"] = "23"
        result = self._install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("三方 wheel 安装失败", result.stderr)
        self.assertNotIn("三方 wheel 安装完成", result.stdout)

    def test_archive_then_install_uses_the_delivered_wheel(self) -> None:
        source = self._wheel(self.dist)
        archive_result = self._archive()
        self.assertEqual(archive_result.returncode, 0, archive_result.stderr)
        install_result = self._install()
        self.assertEqual(install_result.returncode, 0, install_result.stderr)
        self.assertEqual(self._python_calls()[0][-1], str(self.package / source.name))


class FlaWheelPipelineTests(FlaWheelFixture):
    def test_source_cleanup_precedes_build_but_archive_cleanup_follows_it(self) -> None:
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        source_cleanup = 'rm -f "${FLA_NPU_DIST_DIR}"/flash_linear_attention_npu*.whl'
        build = "python setup.py bdist_wheel --device npu"
        archive_call = "\narchive_optional_fla_npu_wheel\n"
        self.assertLess(script.index(source_cleanup), script.index(build))
        self.assertLess(script.index(build), script.index(archive_call))
        self.assertLess(
            script.index(archive_call), script.index('# 收集编译环境中 xLLM 启动所需')
        )

    def test_cleanup_and_failed_build_preserve_previous_package(self) -> None:
        source = self._wheel(self.dist)
        previous = self._wheel(self.package)
        previous.write_bytes(b"previous successful build")
        self._executable("python", "#!/bin/bash\nexit 37\n")
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        cleanup = 'rm -f "${FLA_NPU_DIST_DIR}"/flash_linear_attention_npu*.whl'
        build = "python setup.py bdist_wheel --device npu"
        body = script[script.index(cleanup):script.index(build) + len(build)]
        result = self._run(body + "\n")
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertFalse(source.exists())
        self.assertEqual(previous.read_bytes(), b"previous successful build")

    def test_runtime_call_is_after_xllm_install_and_before_binary_lookup(self) -> None:
        script = INSTALL_SCRIPT.read_text(encoding="utf-8")
        xllm_install = (
            '"$XLLM_PYTHON" -m pip install "$XLLM_PKG_DIR"/xllm_npu*.whl '
            '--force-reinstall'
        )
        optional_call = "\n    install_optional_third_party_wheels\n"
        self.assertLess(script.index(xllm_install), script.index(optional_call))
        self.assertLess(script.index(optional_call), script.index("XLLM_SCRIPTS_DIR=$("))
        self.assertLess(
            script.index(optional_call), script.index('if [ "$XLLM_INSTALL_SERVICE" -eq 1 ]; then')
        )

    def test_build_and_runtime_use_the_same_package_directory(self) -> None:
        self.assertIn(
            'XLLM_THIRD_WHL_DIR="${XLLM_PKG_DIR}/third_whl"',
            BUILD_SCRIPT.read_text(encoding="utf-8"),
        )
        self.assertIn(
            'XLLM_THIRD_WHL_DIR="$XLLM_PKG_DIR/third_whl"',
            INSTALL_SCRIPT.read_text(encoding="utf-8"),
        )

    def test_shell_scripts_have_valid_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(BUILD_SCRIPT), str(INSTALL_SCRIPT)],
            text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class FlaRuntimeSetupTests(FlaWheelFixture):
    # The legacy installer has unquoted system paths; keep only helper fixtures spaced.
    temporary_prefix = "fla-runtime-tests-"

    def setUp(self) -> None:
        super().setUp()
        self.package.parent.mkdir(parents=True)
        (self.package.parent / "xllm_npu_test-0.11.0.whl").write_bytes(b"xllm wheel")
        vendors = self.package.parent / "vendors"
        vendors.mkdir()
        (vendors / "fixture-vendor.txt").write_text("vendor", encoding="utf-8")
        self.source = self.root / "source"
        self.source.mkdir(exist_ok=True)
        (self.source / "torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl").touch()
        self.scripts_dir = self.root / "scripts"
        self.scripts_dir.mkdir()
        binary = self.scripts_dir / "xllm"
        binary.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        (self.root / "lib64").mkdir()
        (self.root / "opp/vendors").mkdir(parents=True)
        pack = self.root / "pack-source/xllm_pack"
        libraries = pack / "xllm-lib"
        libraries.mkdir(parents=True)
        (libraries / "fixture.so").write_bytes(b"library")
        with tarfile.open(pack / "xllm-lib-test.tar.gz", "w:gz") as archive:
            archive.add(libraries, arcname="xllm-lib")
        local_pack = self.root / "local-pack"
        local_pack.mkdir()
        with tarfile.open(local_pack / "xllm_pack.tar.gz", "w:gz") as archive:
            archive.add(pack, arcname="xllm_pack")
        self.env.update(
            XLLM_INSTALL_SERVICE="0",
            XLLM_PACK_LOCAL_DIR=str(local_pack),
            FLA_TEST_SCRIPTS_DIR=str(self.scripts_dir),
        )
        self._executable(
            "python3",
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['FLA_TEST_PYTHON_LOG'], 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1] == '-c':\n"
            "    print(os.environ['FLA_TEST_SCRIPTS_DIR'])\n"
            "elif any(arg.endswith('.whl') and '/third_whl/' in arg for arg in sys.argv):\n"
            "    sys.exit(int(os.environ['FLA_TEST_PIP_EXIT']))\n",
        )
        self._executable("readelf", "#!/bin/bash\necho 'Shared library: [librust_tokenizers.so]'\n")
        self._executable("ldconfig", "#!/bin/bash\nexit 0\n")
        self._executable("numactl", "#!/bin/bash\nexit 0\n")
        self._executable("wget", "#!/bin/bash\necho 'Unexpected network download' >&2\nexit 99\n")

    def _runtime(self) -> subprocess.CompletedProcess:
        # Run the entire script, redirecting all system/workspace writes to this fixture.
        script = INSTALL_SCRIPT.read_text(encoding="utf-8")
        replacements = {
            "/workspace": str(self.root),
            "/xllm_test_060": str(self.source),
            "/export/home/models/xllm_test_060": str(self.source),
            "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors": str(self.root / "opp/vendors"),
            "/usr/lib64": str(self.root / "lib64"),
            "/usr/local/bin": str(self.scripts_dir),
        }
        for old, new in replacements.items():
            script = script.replace(old, new)
        path = self.root / "download_xllm.sh"
        path.write_text(script, encoding="utf-8")
        return subprocess.run(
            ["bash", str(path)], env=self.env, text=True, capture_output=True, timeout=15,
        )

    def _pip_calls(self) -> list:
        return [call for call in self._python_calls() if call[:3] == ["-m", "pip", "install"]]

    def test_full_runtime_without_fla_keeps_legacy_install_flow(self) -> None:
        result = self._runtime()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self._pip_calls()), 1)
        self.assertIn("xllm_npu_test", " ".join(self._pip_calls()[0]))
        self.assertIn("跳过可选三方包安装", result.stdout)
        self.assertIn("get xllm success", result.stdout)

    def test_full_runtime_installs_fla_after_xllm_with_same_interpreter(self) -> None:
        wheel = self._wheel(self.package)
        result = self._runtime()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self._pip_calls()
        self.assertEqual(len(calls), 2)
        self.assertIn("xllm_npu_test", " ".join(calls[0]))
        self.assertEqual(calls[1][-1], str(wheel))
        self.assertIn("--no-index", calls[1])
        self.assertIn("--no-deps", calls[1])
        self.assertIn("get xllm success", result.stdout)

    def test_full_runtime_installs_multiple_packages_in_one_pip_call(self) -> None:
        fla = self._wheel(self.package)
        other = self.package / "other_package-1.0-py3-none-any.whl"
        other.write_bytes(b"other third-party wheel")
        result = self._runtime()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self._pip_calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][6:], sorted([str(fla), str(other)]))
        self.assertIn("get xllm success", result.stdout)

    def test_non_fla_package_is_installed_in_full_runtime(self) -> None:
        self.package.mkdir()
        other = self.package / "other_package-1.0-py3-none-any.whl"
        other.write_bytes(b"other third-party wheel")
        result = self._runtime()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self._pip_calls()), 2)
        self.assertEqual(self._pip_calls()[1][-1], str(other))
        self.assertIn("get xllm success", result.stdout)

    def test_non_fla_install_failure_aborts_full_runtime(self) -> None:
        self.package.mkdir()
        (self.package / "other_package-1.0-py3-none-any.whl").write_bytes(b"other wheel")
        self.env["FLA_TEST_PIP_EXIT"] = "23"
        result = self._runtime()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("三方 wheel 安装失败", result.stderr)
        self.assertNotIn("get xllm success", result.stdout)

    def test_fla_install_failure_aborts_full_runtime(self) -> None:
        self._wheel(self.package)
        self.env["FLA_TEST_PIP_EXIT"] = "23"
        result = self._runtime()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("三方 wheel 安装失败", result.stderr)
        self.assertNotIn("xllm 命令准备完成", result.stdout)
        self.assertNotIn("get xllm success", result.stdout)

    def test_standalone_fla_archive_does_not_trigger_xllm_setup(self) -> None:
        self._wheel(self.package)
        (self.package.parent / "xllm_npu_test-0.11.0.whl").unlink()
        result = self._runtime()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._pip_calls(), [])
        self.assertIn("跳过 xLLM 依赖库下载与安装", result.stdout)
        self.assertIn("get xllm success", result.stdout)

    def test_service_setup_still_runs_after_optional_fla_install(self) -> None:
        self._wheel(self.package)
        service = self.package.parent / "service_binary"
        service.mkdir()
        for name in ("xllm_master_serving", "librust_tokenizers.so", "etcd", "etcdctl"):
            (service / name).write_bytes(b"service fixture")
        self.env["XLLM_INSTALL_SERVICE"] = "1"
        result = self._runtime()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self._pip_calls()), 2)
        self.assertTrue((self.scripts_dir / "xllm_master_serving").exists())
        self.assertTrue((self.scripts_dir / "etcd").exists())
        self.assertTrue((self.root / "lib64/librust_tokenizers.so").exists())
        self.assertIn("get xllm success", result.stdout)


if __name__ == "__main__":
    unittest.main()
