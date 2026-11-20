import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.benchmarks.vbench.generate_vbench_script import generate_vbench_script


class VBenchScriptTests(unittest.TestCase):
    def _config(self, mode="generate_and_evaluate"):
        return {
            "case_name": "vbench-case",
            "served_model_name": "wan2_2",
            "host_addr": "127.0.0.1",
            "port": 14830,
            "benchmark": {
                "benchmark_type": "vbench",
                "vbench": {
                    "mode": mode,
                    "full_info": "/data/full_info.json",
                    "image_root": "/data/images",
                    "evaluation_script": "/data/evaluate.py",
                    "dimensions": ["camera_motion", "i2v_subject"],
                    "generation": {"limit": 2, "seed": 7},
                },
            },
        }

    def test_generate_and_evaluate_script_has_runner_contract(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.vbench.generate_vbench_script.path_mapping_in_container",
            side_effect=lambda value: value,
        ):
            command = generate_vbench_script(self._config(), tmp)
            script = (Path(tmp) / "run_vbench_test.sh").read_text(encoding="utf-8")

        self.assertEqual(command, f"bash {tmp}/run_vbench_test.sh")
        self.assertIn('RUN_DIR="$OUTPUT_DIR/$(date +%Y%m%d_%H%M%S)"', script)
        self.assertIn("--full-info /data/full_info.json", script)
        self.assertIn("--image-root /data/images", script)
        self.assertIn("--limit 2", script)
        self.assertIn("--seed 7", script)
        self.assertIn("--dims camera_motion i2v_subject", script)
        self.assertIn('(cd "$RUN_DIR" &&', script)
        self.assertNotIn("$RUN_DIR'", script)

    def test_generate_mode_does_not_require_evaluation_script(self):
        config = self._config("generate")
        config["benchmark"]["vbench"].pop("evaluation_script")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.vbench.generate_vbench_script.path_mapping_in_container",
            side_effect=lambda value: value,
        ):
            generate_vbench_script(config, tmp)
            script = (Path(tmp) / "run_vbench_test.sh").read_text(encoding="utf-8")
        self.assertNotIn("evaluate.py", script)
        self.assertNotIn("开始执行 VBench 质量评测", script)

    def test_evaluate_mode_uses_existing_manifest(self):
        config = self._config("evaluate")
        config["benchmark"]["vbench"].pop("image_root")
        config["benchmark"]["vbench"]["video_full_info"] = "/data/videos.json"
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.vbench.generate_vbench_script.path_mapping_in_container",
            side_effect=lambda value: value,
        ):
            generate_vbench_script(config, tmp)
            script = (Path(tmp) / "run_vbench_test.sh").read_text(encoding="utf-8")
        self.assertIn("VBench 视频 manifest 不存在: /data/videos.json", script)
        self.assertIn("--full_info /data/videos.json", script)
        self.assertNotIn("--image-root", script)

    def test_omitted_paths_fall_back_to_shared_defaults(self):
        config = self._config()
        vbench = config["benchmark"]["vbench"]
        for key in ("full_info", "image_root", "cache_dir", "evaluation_script"):
            vbench.pop(key, None)
        # 默认路径跟随 config/xTask_config.json 的 DEPENDENCE_PACKAGE_PATH，
        # 而不是任何写死的目录，避免依赖包根目录迁移后测试失效。
        from xtask.benchmarks.vbench.defaults import (
            get_default_full_info,
            get_default_image_root,
        )

        expected_full_info = get_default_full_info()
        expected_image_root = get_default_image_root()
        # The default cache is the on-demand symlink view rooted inside the
        # canonical asset tree; the evaluation driver materializes it.
        from xtask.benchmarks.vbench.defaults import default_cache_dir

        expected_cache_dir = default_cache_dir()
        self.assertTrue(expected_cache_dir.endswith("/vbench_cache"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.vbench.generate_vbench_script.path_mapping_in_container",
            side_effect=lambda value: value,
        ):
            generate_vbench_script(config, tmp)
            script = (Path(tmp) / "run_vbench_test.sh").read_text(encoding="utf-8")
        self.assertIn(f"VBench full_info 不存在: {expected_full_info}", script)
        self.assertIn(f"VBench image_root 不存在: {expected_image_root}", script)
        # path_mapping_in_container is mocked to identity, so the shared
        # project paths appear with their host-side suffixes.
        self.assertIn(f"VBENCH_CACHE_DIR={expected_cache_dir}", script)
        self.assertRegex(
            script,
            r"\S*/xtask/benchmarks/vbench/evaluate_vbench_i2v_bench\.py\b",
        )

    def test_explicit_paths_override_defaults(self):
        config = self._config()
        config["benchmark"]["vbench"]["cache_dir"] = "/opt/vbench_cache"
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.vbench.generate_vbench_script.path_mapping_in_container",
            side_effect=lambda value: value,
        ):
            generate_vbench_script(config, tmp)
            script = (Path(tmp) / "run_vbench_test.sh").read_text(encoding="utf-8")
        self.assertIn("--full-info /data/full_info.json", script)
        self.assertIn("--image-root /data/images", script)
        self.assertIn("/data/evaluate.py", script)
        self.assertIn("VBENCH_CACHE_DIR=/opt/vbench_cache", script)
        # 显式路径生效后不应再出现依赖包默认根目录名。
        from xtask.benchmarks.vbench.defaults import _dependence_package_path

        self.assertNotIn(_dependence_package_path().rsplit("/", 1)[-1], script)
        self.assertNotIn("outputs/vbench_data", script)


if __name__ == "__main__":
    unittest.main()
