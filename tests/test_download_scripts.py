import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class DownloadScriptTests(unittest.TestCase):
    _tree_files = (
        "setup.py",
        "requirements.txt",
        "ais_bench/benchmark/cli/main.py",
        "ais_bench/datasets/__init__.py",
    )

    def _write_tree(self, root: Path, content: str) -> None:
        for relative_path in self._tree_files:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _run_script(self, source: Path, target: Path):
        fake_bin = source.parent / "fake-bin"
        fake_bin.mkdir(exist_ok=True)
        for command in ("pip", "pip3"):
            executable = fake_bin / command
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)

        env = os.environ.copy()
        env.update(
            AISBENCH_SOURCE_DIR=str(source),
            AISBENCH_TARGET_DIR=str(target),
            PATH=f"{fake_bin}:{env['PATH']}",
        )
        return subprocess.run(
            ["bash", "tools/download_ais_bench.sh"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_aisbench_source_tree_is_mirrored_to_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "workspace" / "benchmark"
            self._write_tree(source, "source")
            dataset = source / "ais_bench/datasets/future/dataset.json"
            dataset.parent.mkdir(parents=True)
            dataset.write_text("dataset", encoding="utf-8")
            self._write_tree(target, "stale-target")
            stale = target / "stale-file"
            stale.write_text("stale", encoding="utf-8")

            result = self._run_script(source, target)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Synchronizing AISBench directory", result.stdout)
            self.assertEqual((target / "setup.py").read_text(), "source")
            self.assertEqual(
                (target / dataset.relative_to(source)).read_text(), "dataset"
            )
            self.assertFalse(stale.exists())
            self.assertFalse((target / "benchmark").exists())

    def test_missing_aisbench_target_is_created_at_exact_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "workspace" / "benchmark"
            self._write_tree(source, "source")

            result = self._run_script(source, target)

            self.assertEqual(result.returncode, 0, result.stderr)
            for relative_path in self._tree_files:
                self.assertTrue((target / relative_path).is_file())
            self.assertFalse((target / "benchmark").exists())

    def test_empty_aisbench_source_does_not_remove_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "workspace" / "benchmark"
            source.mkdir()
            target.mkdir(parents=True)
            sentinel = target / "keep-me"
            sentinel.write_text("present", encoding="utf-8")

            result = self._run_script(source, target)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source directory is missing or empty", result.stdout)
            self.assertTrue(sentinel.is_file())


class DownloadXllmScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).resolve().parents[1] / "tools" / "download_xllm.sh"
        ).read_text(encoding="utf-8")

    def test_install_and_lookup_use_the_same_python(self):
        self.assertIn('XLLM_PYTHON=$(command -v python3)', self.script)
        self.assertIn('"$XLLM_PYTHON" -m pip install', self.script)
        self.assertIn('"$XLLM_PYTHON" -c \'import sysconfig;', self.script)

    def test_console_script_has_legacy_package_fallback(self):
        self.assertIn('XLLM_BIN_PATH="$XLLM_SCRIPTS_DIR/xllm"', self.script)
        self.assertIn('"$XLLM_PYTHON" -m pip show "$XLLM_PKG_NAME"', self.script)
        self.assertIn('find "$XLLM_INSTALL_DIR" -type f -name xllm', self.script)

    def test_link_is_self_link_safe_and_validated(self):
        self.assertIn(
            'if [ "$XLLM_BIN_PATH" = "$XLLM_LINK_PATH" ]; then', self.script
        )
        self.assertIn('ln -sfn "$XLLM_BIN_PATH" "$XLLM_LINK_PATH"', self.script)
        self.assertIn('if [ ! -x "$XLLM_LINK_PATH" ]; then', self.script)

    def test_runtime_libs_are_optional_for_legacy_package_layout(self):
        self.assertIn('if [ ! -d "$XLLM_RUNTIME_LIB_DIR" ]; then', self.script)
        self.assertIn('跳过额外 xLLM 运行时库安装', self.script)
        self.assertIn('return 0', self.script)

    def test_xllm_pack_download_uses_one_fixed_archive(self):
        self.assertIn('ensure_xllm_pack() {', self.script)
        self.assertIn(
            'wget --tries=3 -O "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_URL"',
            self.script,
        )
        self.assertIn(
            'rm -f "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_ARCHIVE".*',
            self.script,
        )

    def test_service_install_can_be_disabled_per_runtime(self):
        self.assertIn(
            'XLLM_INSTALL_SERVICE="${XLLM_INSTALL_SERVICE:-1}"', self.script
        )
        self.assertIn('case "$XLLM_INSTALL_SERVICE" in', self.script)
        self.assertIn('if [ "$XLLM_INSTALL_SERVICE" -eq 1 ]; then', self.script)
        self.assertIn(
            "跳过 xllm_master_serving 和 ETCD 安装", self.script
        )

    def test_service_binary_lookup_does_not_require_the_directory(self):
        self.assertNotIn('cd "$XLLM_PKG_DIR"/service_binary', self.script)
        self.assertIn(
            'XLLM_SERVICE_CURRENT_DIR="$XLLM_SERVICE_BINARY_DIR/current"',
            self.script,
        )
        self.assertIn(
            '[ -f "$XLLM_SERVICE_CURRENT_DIR/xllm_master_serving" ]',
            self.script,
        )
        self.assertIn(
            '[ -f "$XLLM_SERVICE_CURRENT_DIR/librust_tokenizers.so" ]',
            self.script,
        )
        self.assertIn(
            '[ -f "$XLLM_SERVICE_BINARY_DIR/etcd" ]', self.script
        )

    def test_service_binary_fallback_prepares_and_validates_pack(self):
        fallback_start = self.script.index(
            'echo "service_binary下未找到本地etcd/etcdctl。"'
        )
        fallback = self.script[fallback_start:]
        self.assertIn('ensure_xllm_pack', fallback)
        self.assertIn(
            'for binary in etcd etcdctl', fallback
        )
        self.assertIn('[ ! -f "$XLLM_PACK_DIR/$binary" ]', fallback)
        self.assertIn('LOCAL_ETCD_READY=0', self.script)
        self.assertIn('if [ -z "$SERVICE_RUNTIME_DIR" ]; then', fallback)


class PrepareXllmScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.script = (cls.root / "tools" / "prepare_xllm.sh").read_text(
            encoding="utf-8"
        )

    def _git(self, *args, cwd=None):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )

    def _create_remote(self, root: Path, name: str, content: str) -> Path:
        remote = root / f"{name}.git"
        source = root / f"{name}-source"
        self._git("init", "--bare", str(remote))
        self._git("init", "-b", "main", str(source))
        (source / "repository.txt").write_text(content, encoding="utf-8")
        self._git("add", "repository.txt", cwd=source)
        self._git(
            "-c",
            "user.name=xTask Test",
            "-c",
            "user.email=xtask@example.invalid",
            "commit",
            "-m",
            "initial",
            cwd=source,
        )
        self._git("remote", "add", "origin", str(remote), cwd=source)
        self._git("push", "-u", "origin", "main", cwd=source)
        self._git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
        return remote

    def _run_prepare(self, project: Path, repository: Path):
        tools_dir = project / "tools"
        tools_dir.mkdir(parents=True)
        shutil.copy2(self.root / "tools" / "prepare_xllm.sh", tools_dir)
        (tools_dir / "build_xllm_in_container.sh").write_text(
            "#!/bin/bash\nexit 0\n", encoding="utf-8"
        )
        fake_bin = project / "fake-bin"
        fake_bin.mkdir()
        docker = fake_bin / "docker"
        docker.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        docker.chmod(0o755)
        env = os.environ.copy()
        env.update(
            XLLM_REPO_URL=str(repository),
            XLLM_BUILD_IMAGE="xtask-test/xllm-build:latest",
            GIT_MAX_RETRIES="1",
            GIT_RETRY_DELAY="0",
            PATH=f"{fake_bin}:{env['PATH']}",
        )
        return subprocess.run(
            ["bash", "tools/prepare_xllm.sh", "main"],
            cwd=project,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_default_repository_is_internal_xllm_repository(self):
        self.assertIn(
            'XLLM_REPO_URL="${XLLM_REPO_URL:-git@coding.jd.com:xLLM_AI/xllm.git}"',
            self.script,
        )

    def test_compile_source_version_is_logged_after_submodules(self):
        commit = 'XLLM_COMPILE_COMMIT="$(git rev-parse HEAD)"'
        self.assertIn(commit, self.script)
        self.assertIn('echo "[source-version] xLLM commit: ${XLLM_COMPILE_COMMIT}"', self.script)
        self.assertLess(
            self.script.index('git submodule update --init --recursive'),
            self.script.index(commit),
        )

    def test_different_origin_is_removed_and_recloned(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_remote = self._create_remote(root, "old", "old")
            new_remote = self._create_remote(root, "new", "new")
            project = root / "project"
            project.mkdir()
            self._git("clone", str(old_remote), str(project / "xllm"))
            stale_file = project / "xllm" / "old-untracked-file"
            stale_file.write_text("stale", encoding="utf-8")

            result = self._run_prepare(project, new_remote)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("recloning source", result.stderr)
            self.assertEqual(
                (project / "xllm" / "repository.txt").read_text(), "new"
            )
            self.assertFalse(stale_file.exists())
            origin = self._git(
                "remote", "get-url", "origin", cwd=project / "xllm"
            ).stdout.strip()
            self.assertEqual(origin, str(new_remote))

    def test_same_origin_preserves_existing_checkout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            remote = self._create_remote(root, "same", "same")
            project = root / "project"
            project.mkdir()
            self._git("clone", str(remote), str(project / "xllm"))
            marker = project / "xllm" / "keep-untracked-file"
            marker.write_text("keep", encoding="utf-8")

            result = self._run_prepare(project, remote)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("recloning source", result.stderr)
            self.assertTrue(marker.is_file())

    def test_unreachable_new_origin_does_not_remove_existing_checkout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_remote = self._create_remote(root, "old", "old")
            missing_remote = root / "missing.git"
            project = root / "project"
            project.mkdir()
            self._git("clone", str(old_remote), str(project / "xllm"))
            marker = project / "xllm" / "keep-after-validation-error"
            marker.write_text("keep", encoding="utf-8")

            result = self._run_prepare(project, missing_remote)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validate xLLM repository", result.stderr)
            self.assertTrue(marker.is_file())
            origin = self._git(
                "remote", "get-url", "origin", cwd=project / "xllm"
            ).stdout.strip()
            self.assertEqual(origin, str(old_remote))


class BuildXllmInContainerScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "build_xllm_in_container.sh"
        ).read_text(encoding="utf-8")

    def test_stale_dist_wheels_are_removed_before_build(self):
        cleanup = 'rm -f "${XLLM_DIST_DIR}"/xllm_npu*.whl'
        build = "python setup.py bdist_wheel --device npu"

        self.assertIn(cleanup, self.script)
        self.assertLess(self.script.index(cleanup), self.script.index(build))

    def test_archive_requires_exactly_one_new_wheel(self):
        self.assertIn(
            'XLLM_WHEELS=("${XLLM_DIST_DIR}"/xllm_npu*.whl)',
            self.script,
        )
        self.assertIn('if [[ ${#XLLM_WHEELS[@]} -ne 1 ]]; then', self.script)

    def test_previous_archive_is_kept_until_new_wheel_is_staged(self):
        stage = 'cp -f "${XLLM_WHEELS[0]}" "${XLLM_WHEEL_TMP}"'
        cleanup = 'rm -f "${XLLM_PKG_DIR}"/xllm_npu*.whl'
        replace = (
            'mv -f "${XLLM_WHEEL_TMP}" '
            '"${XLLM_PKG_DIR}/${XLLM_WHEEL_NAME}"'
        )

        self.assertIn(stage, self.script)
        self.assertIn(cleanup, self.script)
        self.assertIn(replace, self.script)
        self.assertLess(self.script.index(stage), self.script.index(cleanup))
        self.assertLess(self.script.index(cleanup), self.script.index(replace))


class PrepareXllmServiceScriptTests(unittest.TestCase):
    _required_source_files = (
        "CMakeLists.txt",
        "prepare.sh",
        "third_party/brpc/CMakeLists.txt",
        "third_party/cpprestsdk/Release/CMakeLists.txt",
        "third_party/etcd_cpp_apiv3/CMakeLists.txt",
        "third_party/minja/CMakeLists.txt",
        "third_party/sentencepiece/CMakeLists.txt",
        "third_party/smhasher/src/CMakeLists.txt",
        "third_party/xllm/CMakeLists.txt",
        "third_party/xxHash/CMakeLists.txt",
    )

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.script = (cls.root / "tools" / "prepare_xllm_service.sh").read_text(
            encoding="utf-8"
        )

    def _git(self, *args, cwd=None):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )

    def _create_remote(self, root: Path, name: str, content: str) -> Path:
        remote = root / f"{name}.git"
        source = root / f"{name}-source"
        self._git("init", "--bare", str(remote))
        self._git("init", "-b", "main", str(source))
        for relative_path in self._required_source_files:
            path = source / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self._git("add", ".", cwd=source)
        self._git(
            "-c",
            "user.name=xTask Test",
            "-c",
            "user.email=xtask@example.invalid",
            "commit",
            "-m",
            "initial",
            cwd=source,
        )
        self._git("remote", "add", "origin", str(remote), cwd=source)
        self._git("push", "-u", "origin", "main", cwd=source)
        self._git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
        return remote

    def _run_prepare(
        self, project: Path, repository: Path, update_source: str = "1"
    ):
        tools_dir = project / "tools"
        tools_dir.mkdir(parents=True)
        shutil.copy2(self.root / "tools" / "prepare_xllm_service.sh", tools_dir)
        (tools_dir / "build_xllm_service_in_container.sh").write_text(
            "#!/bin/bash\nexit 0\n", encoding="utf-8"
        )
        fake_bin = project / "fake-bin"
        fake_bin.mkdir()
        docker = fake_bin / "docker"
        docker.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        docker.chmod(0o755)
        env = os.environ.copy()
        env.update(
            XLLM_SERVICE_REPO_URL=str(repository),
            XLLM_SERVICE_UPDATE_SOURCE=update_source,
            XLLM_SERVICE_BUILD_IMAGE="xtask-test/xllm-service-build:latest",
            GIT_MAX_RETRIES="1",
            GIT_RETRY_DELAY="0",
            PATH=f"{fake_bin}:{env['PATH']}",
        )
        return subprocess.run(
            ["bash", "tools/prepare_xllm_service.sh", "main"],
            cwd=project,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def _temporary_root(self):
        # The production script intentionally requires build sources under
        # /export/home because that exact tree is mounted into its container.
        return tempfile.TemporaryDirectory(
            prefix=".prepare-xllm-service-test-", dir=self.root
        )

    def test_default_repository_is_internal_service_repository(self):
        self.assertIn(
            'SERVICE_REPO_URL="${XLLM_SERVICE_REPO_URL:-git@coding.jd.com:xLLM_AI/xllm-service.git}"',
            self.script,
        )

    def test_compile_source_version_is_logged_after_source_validation(self):
        commit = 'SERVICE_COMPILE_COMMIT="$(git -C "${SERVICE_SOURCE_DIR}" rev-parse HEAD)"'
        self.assertIn(commit, self.script)
        self.assertIn(
            'echo "[source-version] xLLM-Service commit: ${SERVICE_COMPILE_COMMIT}"',
            self.script,
        )
        self.assertIn("third_party/xllm commit", self.script)
        self.assertLess(
            self.script.index("REQUIRED_SOURCE_FILES=("),
            self.script.index(commit),
        )

    def test_different_origin_is_removed_and_recloned(self):
        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            old_remote = self._create_remote(root, "old", "old")
            new_remote = self._create_remote(root, "new", "new")
            project = root / "project"
            project.mkdir()
            self._git("clone", str(old_remote), str(project / "xllm-service"))
            stale_file = project / "xllm-service" / "old-untracked-file"
            stale_file.write_text("stale", encoding="utf-8")

            result = self._run_prepare(project, new_remote)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("recloning source", result.stderr)
            self.assertEqual(
                (project / "xllm-service" / "CMakeLists.txt").read_text(), "new"
            )
            self.assertFalse(stale_file.exists())
            origin = self._git(
                "remote", "get-url", "origin", cwd=project / "xllm-service"
            ).stdout.strip()
            self.assertEqual(origin, str(new_remote))

    def test_same_origin_preserves_existing_checkout(self):
        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            remote = self._create_remote(root, "same", "same")
            project = root / "project"
            project.mkdir()
            self._git("clone", str(remote), str(project / "xllm-service"))
            marker = project / "xllm-service" / "keep-untracked-file"
            marker.write_text("keep", encoding="utf-8")

            result = self._run_prepare(project, remote)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("recloning source", result.stderr)
            self.assertTrue(marker.is_file())

    def test_unreachable_new_origin_does_not_remove_existing_checkout(self):
        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            old_remote = self._create_remote(root, "old", "old")
            missing_remote = root / "missing.git"
            project = root / "project"
            project.mkdir()
            self._git("clone", str(old_remote), str(project / "xllm-service"))
            marker = project / "xllm-service" / "keep-after-validation-error"
            marker.write_text("keep", encoding="utf-8")

            result = self._run_prepare(project, missing_remote)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validate xLLM-Service repository", result.stderr)
            self.assertTrue(marker.is_file())

    def test_update_source_disabled_does_not_replace_existing_checkout(self):
        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            old_remote = self._create_remote(root, "old", "old")
            new_remote = self._create_remote(root, "new", "new")
            project = root / "project"
            project.mkdir()
            self._git("clone", str(old_remote), str(project / "xllm-service"))
            marker = project / "xllm-service" / "keep-with-updates-disabled"
            marker.write_text("keep", encoding="utf-8")

            result = self._run_prepare(project, new_remote, update_source="0")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("XLLM_SERVICE_UPDATE_SOURCE=0", result.stderr)
            self.assertTrue(marker.is_file())


class BuildXllmServiceScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.outer_script = (root / "tools" / "prepare_xllm_service.sh").read_text(
            encoding="utf-8"
        )
        cls.inner_script = (
            root / "tools" / "build_xllm_service_in_container.sh"
        ).read_text(encoding="utf-8")

    def test_service_build_is_separate_from_xllm_build(self):
        self.assertIn("XLLM_SERVICE_SOURCE_DIR", self.outer_script)
        self.assertIn("docker run", self.outer_script)
        self.assertIn("submodule update --init", self.outer_script)
        self.assertNotIn("submodule update --init --recursive", self.outer_script)
        self.assertIn("REQUIRED_SOURCE_FILES", self.outer_script)
        self.assertIn("third_party/etcd_cpp_apiv3/CMakeLists.txt", self.outer_script)
        self.assertNotIn('bash "${SCRIPT_DIR}/prepare_xllm.sh"', self.outer_script)

    def test_service_build_applies_patch_and_builds_only_master_target(self):
        self.assertIn("bash prepare.sh", self.inner_script)
        self.assertIn("cmake -S", self.inner_script)
        self.assertIn("--target xllm_master_serving", self.inner_script)
        self.assertIn("librust_tokenizers.so", self.inner_script)
        self.assertIn("VCPKG_INSTALL_ROOT", self.inner_script)
        self.assertIn("boost-locale boost-interprocess zlib", self.inner_script)
        self.assertIn('CXX_STANDARD="${XLLM_SERVICE_CXX_STANDARD:-20}"', self.inner_script)
        self.assertIn("CMAKE_PROJECT_xllm_service_INCLUDE", self.inner_script)
        self.assertIn("-DWERROR=OFF", self.inner_script)
        self.assertIn("trap restore_host_ownership EXIT", self.inner_script)

    def test_service_archive_is_versioned_and_atomic(self):
        self.assertIn(
            'SERVICE_ARCHIVE_DIR="${XLLM_PKG_DIR}/service_binary"',
            self.inner_script,
        )
        self.assertIn(
            'SERVICE_ARCHIVE_DIR}/releases/${RELEASE_ID}',
            self.inner_script,
        )
        self.assertIn(
            'TMP_LINK="${SERVICE_ARCHIVE_DIR}/.current-${RELEASE_ID}.tmp"',
            self.inner_script,
        )
        self.assertIn('mv -Tf "${TMP_LINK}" "${CURRENT_LINK}"', self.inner_script)
        self.assertIn("xllm_submodule_commit", self.inner_script)

    def test_service_does_not_claim_to_build_etcd(self):
        self.assertIn("etcd/etcdctl", self.inner_script)
        self.assertIn("independent", self.inner_script)
        self.assertNotIn("make etcd", self.inner_script)


if __name__ == "__main__":
    unittest.main()
