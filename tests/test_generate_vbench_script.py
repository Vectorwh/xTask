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
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.benchmarks.vbench.generate_vbench_script.path_mapping_in_container",
            side_effect=lambda value: value,
        ):
            generate_vbench_script(config, tmp)
            script = (Path(tmp) / "run_vbench_test.sh").read_text(encoding="utf-8")
        self.assertIn("VBench full_info 不存在: /export/home/models/xllm_test_060/vbench_data/full_info.json", script)
        self.assertIn("VBench image_root 不存在: /export/home/models/xllm_test_060/vbench_data/data", script)
        # path_mapping_in_container is mocked to identity, so the shared
        # project paths appear with their host-side suffixes.
        self.assertRegex(script, r"VBENCH_CACHE_DIR=\S*/outputs/vbench_data/cache\b")
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
        self.assertNotIn("xllm_test_060", script)
        self.assertNotIn("outputs/vbench_data", script)


if __name__ == "__main__":
    unittest.main()
