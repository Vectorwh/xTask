import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DownloadScriptTests(unittest.TestCase):
    _required_files = (
        "setup.py",
        "requirements.txt",
        "ais_bench/benchmark/cli/main.py",
        "ais_bench/datasets/__init__.py",
    )

    def _write_tree(self, root: Path, content: str) -> None:
        for relative_path in self._required_files:
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

    def test_complete_aisbench_target_skips_copy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "workspace" / "benchmark"
            self._write_tree(source, "source")
            self._write_tree(target, "target")
            sentinel = target / "keep-me"
            sentinel.write_text("present", encoding="utf-8")

            result = self._run_script(source, target)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("skip copy", result.stdout)
            self.assertEqual((target / "setup.py").read_text(), "target")
            self.assertTrue(sentinel.is_file())

    def test_incomplete_aisbench_target_is_replaced_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "workspace" / "benchmark"
            self._write_tree(source, "source")
            target.mkdir(parents=True)
            stale = target / "stale-file"
            stale.write_text("stale", encoding="utf-8")

            result = self._run_script(source, target)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("missing or incomplete", result.stdout)
            self.assertFalse(stale.exists())
            for relative_path in self._required_files:
                self.assertTrue((target / relative_path).is_file())


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

    def test_service_binary_lookup_does_not_require_the_directory(self):
        self.assertNotIn('cd "$XLLM_PKG_DIR"/service_binary', self.script)
        self.assertIn(
            '[ -f "$XLLM_SERVICE_BINARY_DIR/etcd" ]', self.script
        )
        self.assertIn(
            '"$XLLM_SERVICE_BINARY_DIR/xllm_master_serving"', self.script
        )

    def test_service_binary_fallback_prepares_and_validates_pack(self):
        fallback_start = self.script.index(
            'echo "service_binary下未找到二进制文件，使用wget的二进制。"'
        )
        fallback = self.script[fallback_start:]
        self.assertIn('ensure_xllm_pack', fallback)
        self.assertIn(
            'for binary in etcd etcdctl xllm_master_serving', fallback
        )
        self.assertIn('[ ! -f "$XLLM_PACK_DIR/$binary" ]', fallback)


if __name__ == "__main__":
    unittest.main()
