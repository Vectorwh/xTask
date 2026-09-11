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

        # SOURCE_DIR is hardcoded to /xllm_test_060/benchmark in the script;
        # create a temp copy with SOURCE_DIR redirected to the test tree so
        # the sync logic can be exercised without touching system paths.
        original_script = Path(__file__).resolve().parents[1] / "tools" / "download_ais_bench.sh"
        script_text = original_script.read_text(encoding="utf-8")
        script_text = script_text.replace(
            'SOURCE_DIR="/xllm_test_060/benchmark"',
            f'SOURCE_DIR="{source}"',
        )
        temp_script = fake_bin / "download_ais_bench.sh"
        temp_script.write_text(script_text, encoding="utf-8")
        temp_script.chmod(0o755)

        env = os.environ.copy()
        env.update(
            AISBENCH_TARGET_DIR=str(target),
            PATH=f"{fake_bin}:{env['PATH']}",
        )
        return subprocess.run(
            ["bash", str(temp_script)],
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

    def test_xllm_pack_prefers_local_archive_before_wget(self):
        self.assertIn('XLLM_PACK_LOCAL_DIR="${XLLM_PACK_LOCAL_DIR:-}"', self.script)
        self.assertIn('XLLM_PACK_LOCAL_CANDIDATES=(', self.script)
        self.assertIn('"/export/home/models/xllm_test_060"', self.script)
        self.assertIn('"/xllm_test_060"', self.script)
        # 本地探测必须发生在 wget 之前。
        probe = self.script.index('XLLM_PACK_LOCAL_SOURCE="$candidate"')
        wget = self.script.index(
            'wget --tries=3 -O "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_URL"'
        )
        self.assertLess(probe, wget)
        # 命中本地归档且解压成功时直接返回，不触发外网下载。
        self.assertIn(
            'tar -zxf "$XLLM_PACK_LOCAL_SOURCE/xllm_pack.tar.gz" -C "$TARGET_DIR"',
            self.script,
        )
        self.assertIn(
            'echo "使用本地归档: $XLLM_PACK_LOCAL_SOURCE/xllm_pack.tar.gz"',
            self.script,
        )

    def test_xllm_pack_local_archive_failure_falls_back_to_wget(self):
        # 本地归档损坏（tar 校验失败）或解压后没有生成目录时回退外网下载。
        self.assertIn(
            'Error: 本地归档校验失败，回退到外网下载',
            self.script,
        )
        self.assertIn(
            'Error: 本地归档解压后未生成目录',
            self.script,
        )
        fallback = self.script.index('Error: 本地归档校验失败，回退到外网下载')
        wget = self.script.index(
            'wget --tries=3 -O "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_URL"'
        )
        self.assertLess(fallback, wget)

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
        "third_party/xllm/third_party/spdlog/CMakeLists.txt",
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

    def _git_commit(self, cwd: Path, message: str = "initial") -> None:
        self._git(
            "-c",
            "user.name=xTask Test",
            "-c",
            "user.email=xtask@example.invalid",
            "commit",
            "-m",
            message,
            cwd=cwd,
        )

    def _create_bare_remote_with_tree(
        self, root: Path, name: str, files: dict[str, str]
    ) -> Path:
        remote = root / f"{name}.git"
        source = root / f"{name}-source"
        self._git("init", "--bare", str(remote))
        self._git("init", "-b", "main", str(source))
        for relative_path, content in files.items():
            path = source / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self._git("add", ".", cwd=source)
        self._git_commit(source, f"add {name}")
        self._git("remote", "add", "origin", str(remote), cwd=source)
        self._git("push", "-u", "origin", "main", cwd=source)
        self._git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
        return remote

    def _create_remote_with_minja_submodule(self, root: Path) -> Path:
        minja_remote = root / "minja.git"
        minja_source = root / "minja-source"
        self._git("init", "--bare", str(minja_remote))
        self._git("init", "-b", "main", str(minja_source))
        (minja_source / "CMakeLists.txt").write_text("minja", encoding="utf-8")
        self._git("add", ".", cwd=minja_source)
        self._git_commit(minja_source, "add minja")
        self._git("remote", "add", "origin", str(minja_remote), cwd=minja_source)
        self._git("push", "-u", "origin", "main", cwd=minja_source)
        self._git("symbolic-ref", "HEAD", "refs/heads/main", cwd=minja_remote)

        service_remote = root / "service.git"
        service_source = root / "service-source"
        self._git("init", "--bare", str(service_remote))
        self._git("init", "-b", "main", str(service_source))
        for relative_path in self._required_source_files:
            if relative_path.startswith("third_party/minja/"):
                continue
            path = service_source / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub", encoding="utf-8")
        self._git("add", ".", cwd=service_source)
        self._git_commit(service_source, "add service skeleton")
        self._git(
            "submodule",
            "add",
            str(minja_remote),
            "third_party/minja",
            cwd=service_source,
        )
        self._git_commit(service_source, "add service with minja submodule")
        self._git("remote", "add", "origin", str(service_remote), cwd=service_source)
        self._git("push", "-u", "origin", "main", cwd=service_source)
        self._git("symbolic-ref", "HEAD", "refs/heads/main", cwd=service_remote)
        return service_remote

    def _simulate_half_pulled_submodule(self, submodule_dir: Path) -> None:
        git_pointer = (submodule_dir / ".git").read_text(encoding="utf-8")
        shutil.rmtree(submodule_dir)
        submodule_dir.mkdir()
        (submodule_dir / ".git").write_text(git_pointer, encoding="utf-8")

    def test_repair_half_pulled_top_level_submodule(self):
        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            remote = self._create_remote_with_minja_submodule(root)
            project = root / "project"
            project.mkdir()
            self._git(
                "clone",
                "--recurse-submodules",
                str(remote),
                str(project / "xllm-service"),
            )
            minja_dir = project / "xllm-service" / "third_party" / "minja"
            self.assertTrue((minja_dir / "CMakeLists.txt").is_file())
            self._simulate_half_pulled_submodule(minja_dir)
            self.assertFalse((minja_dir / "CMakeLists.txt").is_file())

            result = self._run_prepare(project, remote)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("repairing submodule 'third_party/minja'", result.stderr)
            self.assertEqual(
                (minja_dir / "CMakeLists.txt").read_text(encoding="utf-8"),
                "minja",
            )

    def test_repair_half_pulled_nested_xllm_submodule(self):
        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            spdlog_remote = self._create_bare_remote_with_tree(
                root,
                "spdlog",
                {"CMakeLists.txt": "spdlog"},
            )
            xllm_remote = root / "xllm.git"
            xllm_source = root / "xllm-source"
            self._git("init", "--bare", str(xllm_remote))
            self._git("init", "-b", "main", str(xllm_source))
            (xllm_source / "CMakeLists.txt").write_text("xllm", encoding="utf-8")
            self._git("add", ".", cwd=xllm_source)
            self._git_commit(xllm_source, "add xllm skeleton")
            self._git(
                "submodule",
                "add",
                str(spdlog_remote),
                "third_party/spdlog",
                cwd=xllm_source,
            )
            self._git_commit(xllm_source, "add xllm with spdlog submodule")
            self._git("remote", "add", "origin", str(xllm_remote), cwd=xllm_source)
            self._git("push", "-u", "origin", "main", cwd=xllm_source)
            self._git("symbolic-ref", "HEAD", "refs/heads/main", cwd=xllm_remote)

            service_remote = root / "service.git"
            service_source = root / "service-source"
            self._git("init", "--bare", str(service_remote))
            self._git("init", "-b", "main", str(service_source))
            for relative_path in self._required_source_files:
                if relative_path.startswith("third_party/xllm/"):
                    continue
                path = service_source / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stub", encoding="utf-8")
            self._git("add", ".", cwd=service_source)
            self._git_commit(service_source, "add service skeleton")
            self._git(
                "submodule",
                "add",
                str(xllm_remote),
                "third_party/xllm",
                cwd=service_source,
            )
            self._git_commit(service_source, "add service with xllm submodule")
            self._git("remote", "add", "origin", str(service_remote), cwd=service_source)
            self._git("push", "-u", "origin", "main", cwd=service_source)
            self._git("symbolic-ref", "HEAD", "refs/heads/main", cwd=service_remote)

            project = root / "project"
            project.mkdir()
            self._git(
                "clone",
                "--recurse-submodules",
                str(service_remote),
                str(project / "xllm-service"),
            )
            xllm_dir = project / "xllm-service" / "third_party" / "xllm"
            spdlog_dir = xllm_dir / "third_party" / "spdlog"
            self.assertTrue((xllm_dir / "CMakeLists.txt").is_file())
            self.assertTrue((spdlog_dir / "CMakeLists.txt").is_file())
            self._simulate_half_pulled_submodule(spdlog_dir)
            self.assertTrue((xllm_dir / "CMakeLists.txt").is_file())
            self.assertFalse((spdlog_dir / "CMakeLists.txt").is_file())

            result = self._run_prepare(project, service_remote)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "repairing nested submodule 'third_party/spdlog'",
                result.stderr,
            )
            self.assertEqual(
                (spdlog_dir / "CMakeLists.txt").read_text(encoding="utf-8"),
                "spdlog",
            )
            self.assertEqual(
                (xllm_dir / "CMakeLists.txt").read_text(encoding="utf-8"),
                "xllm",
            )

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
        self.assertIn(
            "submodule update --init --recursive third_party/xllm",
            self.outer_script,
        )
        self.assertNotRegex(
            self.outer_script,
            r"submodule update --init --recursive(?!\s+third_party/xllm)",
        )
        self.assertIn("third_party/xllm/third_party/spdlog/CMakeLists.txt", self.outer_script)
        self.assertIn("REQUIRED_SOURCE_FILES", self.outer_script)
        self.assertIn("repair_required_submodules", self.outer_script)
        self.assertIn("nested_submodule_within_xllm", self.outer_script)
        self.assertIn(
            'git -C "${SERVICE_SOURCE_DIR}/third_party/xllm" submodule update --init',
            self.outer_script,
        )
        self.assertIn("submodule_path_for_required_file", self.outer_script)
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
        self.assertIn("-DBUILD_TESTING=ON", self.inner_script)
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
