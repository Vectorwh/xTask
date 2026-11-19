"""generate_image_edit_script / image_edit_postprocess 单元测试。

与 tests/test_generate_evalscope_script.py 同风格：mock
path_mapping_in_container 后检查生成的 shell 脚本内容。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.benchmarks.evalscope import generate_image_edit_script as generator


class ImageEditScriptGenerationTests(unittest.TestCase):
    def _task_config(self, metrics="perf"):
        return {
            "benchmark": {
                "benchmark_type": "image_edit",
                "perf": {
                    "dataset_path": "/data/gedit_perf.jsonl",
                    "num_prompts": 10,
                    "batch_size": 2,
                    "request_rate": -1,
                    "warmup_num": 1,
                    "seed": 42,
                    "timeout": 7200,
                    "size": "1328*1328",
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                },
            },
            "case_name": "case_gedit",
            "dataset_name": "gedit",
            "host_addr": "127.0.0.1",
            "metrics": metrics,
            "model_path": "/models/qwen-image-edit",
            "port": 8800,
            "served_model_name": "qwen-image-edit",
        }

    def _generate(self, output_dir, metrics="perf"):
        with mock.patch.object(
            generator, "path_mapping_in_container", side_effect=lambda path: path
        ):
            generator.generate_image_edit_script(
                self._task_config(metrics), str(output_dir)
            )
        return output_dir / "run_image_edit_test.sh"

    def test_perf_command_uses_wrapper_and_full_url(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            script_path = self._generate(output_dir)
            script = script_path.read_text(encoding="utf-8")

        self.assertIn("run_gedit_perf.py", script)
        self.assertIn("--url http://127.0.0.1:8800/v1/image/generation", script)
        self.assertIn("--model qwen-image-edit", script)
        self.assertIn("--dataset-path /data/gedit_perf.jsonl", script)
        self.assertIn("--number 10", script)
        self.assertIn("--parallel 2", script)
        self.assertIn("--rate -1", script)
        self.assertIn("--warmup-num 1", script)
        self.assertIn("--seed 42", script)
        self.assertIn("--total-timeout 7200", script)
        self.assertIn("--name case_gedit", script)
        # 请求体级参数经 --extra-args JSON 注入
        extra = json.dumps(
            {"size": "1328*1328", "num_inference_steps": 50, "guidance_scale": 3.5}
        )
        self.assertIn(f"--extra-args '{extra}'", script)
        # 失败短路 + 数据集存在性检查
        self.assertIn("if ! ", script)
        self.assertIn("image_edit性能测试执行失败", script)
        self.assertIn("数据集文件不存在", script)

    def test_concurrency_sweep_renders_multiple_parallels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            config = self._task_config()
            config["benchmark"]["perf"]["concurrency_sweep"] = [1, 2, 4]
            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ):
                generator.generate_image_edit_script(config, str(output_dir))
            script = (output_dir / "run_image_edit_test.sh").read_text(encoding="utf-8")

        self.assertIn("--parallel 1 2 4", script)
        self.assertNotIn("--parallel 2\n", script.replace("--parallel 1 2 4", ""))

    def test_missing_dataset_path_falls_back_to_default(self):
        # 未显式指定 dataset_path 时落到 DEPENDENCE_PACKAGE_PATH 下的默认路径。
        config = self._task_config()
        config["benchmark"]["perf"]["dataset_path"] = ""
        config["dataset_path"] = ""
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(
                generator,
                "path_mapping_in_container",
                side_effect=lambda path: path,
            ), mock.patch.object(
                generator,
                "default_gedit_perf_dataset_path",
                return_value="/dep/pkg/benchmark/ais_bench/datasets/gedit_perf/gedit_perf.jsonl",
            ):
                generator.generate_image_edit_script(config, temporary_directory)
            script = (
                Path(temporary_directory) / "run_image_edit_test.sh"
            ).read_text(encoding="utf-8")

        self.assertIn(
            "--dataset-path /dep/pkg/benchmark/ais_bench/datasets/"
            "gedit_perf/gedit_perf.jsonl",
            script,
        )

    def test_default_dataset_path_resolves_dependence_package_root(self):
        # 环境变量 DEPENDENCE_PACKAGE_PATH 优先、config 兜底。
        with mock.patch.dict(
            os.environ, {"DEPENDENCE_PACKAGE_PATH": "/env/dep/root"}
        ):
            self.assertEqual(
                generator.default_gedit_perf_dataset_path(),
                "/env/dep/root/benchmark/ais_bench/datasets/gedit_perf/"
                "gedit_perf.jsonl",
            )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEPENDENCE_PACKAGE_PATH", None)
            with mock.patch.object(
                generator,
                "get_xtask_config_data",
                return_value={
                    "ENV_CONFIG": {
                        "DEPENDENCE_PACKAGE_PATH": "/cfg/dep/root"
                    }
                },
            ):
                self.assertEqual(
                    generator.default_gedit_perf_dataset_path(),
                    "/cfg/dep/root/benchmark/ais_bench/datasets/gedit_perf/"
                    "gedit_perf.jsonl",
                )

    def test_non_perf_metrics_skips_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            script = self._generate(Path(temporary_directory), metrics="acc").read_text(
                encoding="utf-8"
            )
        self.assertIn("skip task", script)
        self.assertNotIn("run_gedit_perf.py", script)

    def test_external_service_endpoint_is_used(self):
        config = self._task_config()
        config["external_service"] = {"endpoint": "https://api.example.com:9443/base"}
        location = mock.Mock(origin="https://api.example.com:9443", base_path="/base")
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch(
                "xtask.core.external_service.is_external_service_enabled",
                return_value=True,
            ), mock.patch(
                "xtask.core.external_service.resolve_external_location",
                return_value=location,
            ):
                generator.generate_image_edit_script(config, temporary_directory)
            script = (Path(temporary_directory) / "run_image_edit_test.sh").read_text(
                encoding="utf-8"
            )

        self.assertIn("--url https://api.example.com:9443/base/image/generation", script)


if __name__ == "__main__":
    unittest.main()
