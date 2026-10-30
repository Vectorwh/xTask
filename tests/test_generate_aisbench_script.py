import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.benchmarks.aisbench import generate_aisbench_script as generator


class AisbenchScriptGenerationTests(unittest.TestCase):
    def _task_config(self):
        return {
            "benchmark": {
                "benchmark_type": "aisbench",
                "perf": {
                    "api_type": "vllm_api_general_stream",
                    "batch_size": 1,
                    "dataset_conf": "gsm8k_gen_0_shot_cot_str_perf",
                    "num_warmups": 20,
                    "output_len": 1024,
                    "request_rate": 1,
                },
            },
            "dataset_name": "gsm8k",
            "host_addr": "127.0.0.1",
            "metrics": "perf",
            "model_path": "/models/deepseek",
            "port": 8800,
            "served_model_name": "deepseek",
        }

    def _generate(self, output_dir, python_exec):
        with mock.patch.object(
            generator, "path_mapping_in_container", side_effect=lambda path: path
        ), mock.patch.object(generator, "download_dataset", return_value=[]), mock.patch.object(
            generator,
            "preprocess_dataset",
            return_value=(["true"], "gsm8k_gen_0_shot_cot_str_perf"),
        ):
            generator.generate_aisbench_script(
                self._task_config(),
                str(output_dir),
                python_exec=str(python_exec),
            )

        return output_dir / "run_aisbench_test.sh"

    def test_perf_uses_isolated_model_config_and_checked_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            python_exec = root / "fake-python"
            python_exec.write_text("#!/bin/sh\necho /unused/ais_bench\n", encoding="utf-8")
            python_exec.chmod(0o755)

            script_path = self._generate(output_dir, python_exec)
            script = script_path.read_text(encoding="utf-8")

        self.assertIn('AIS_BENCH_CONFIG_DIR="', script)
        self.assertIn('cp -f "$SOURCE_CONFIG_FILE" "$CONFIG_FILE"', script)
        self.assertIn('--config-dir "$AIS_BENCH_CONFIG_DIR"', script)
        self.assertIn('--models xtask_vllm_api_general_stream', script)
        self.assertIn("path[[:space:]]*=.*", script)
        self.assertIn("path='/models/deepseek'", script)
        self.assertIn('if ! ais_bench ', script)
        self.assertIn('echo "AISBench性能测试执行失败" >&2', script)

    def test_sharegpt_uses_explicit_dataset_path_and_multiturn_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = root / "sharegpt_gen.py"
            source_config.write_text(
                "from example import ShareGPTDataset\n"
                "sharegpt_datasets = [\n"
                "    dict(\n"
                "        abbr='sharegpt',\n"
                "        type=ShareGPTDataset,\n"
                "        path='ais_bench/datasets/sharegpt/default.json',\n"
                "    )\n"
                "]\n",
                encoding="utf-8",
            )
            dataset_path = (
                "/xllm_test_060/datasets/sharegpt/"
                "ShareGPT_V3_unfiltered_cleaned_split.json"
            )
            task_config = self._task_config()
            task_config["dataset_name"] = "sharegpt"
            task_config["dataset_path"] = dataset_path
            task_config["benchmark"]["perf"].update(
                {
                    "api_type": "vllm_api_stream_chat_multiturn",
                    "dataset_conf": "sharegpt_gen",
                    "num_prompts": 8,
                }
            )

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "preprocess_dataset", return_value=([], "")
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config", return_value=source_config
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )
            dataset_config = (
                output_dir
                / "aisbench_configs"
                / "datasets"
                / "xtask_sharegpt_gen.py"
            ).read_text(encoding="utf-8")

        self.assertIn(
            "--models xtask_vllm_api_stream_chat_multiturn", script
        )
        self.assertIn("--datasets xtask_sharegpt_gen", script)
        self.assertIn("--num-prompts 8", script)
        self.assertIn(f"path='{dataset_path}'", dataset_config)
        self.assertNotIn("default.json", dataset_config)

    def test_sharegpt_defaults_to_aisbench_relative_dataset_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            output_dir.mkdir()
            task_config = self._task_config()
            task_config["dataset_name"] = "sharegpt"
            task_config["benchmark"]["perf"].update(
                {
                    "api_type": "vllm_api_stream_chat_multiturn",
                    "dataset_conf": "sharegpt_gen",
                    "num_prompts": 8,
                }
            )

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "preprocess_dataset", return_value=([], "")
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config"
            ) as find_config:
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )

        self.assertIn("--datasets sharegpt_gen", script)
        self.assertNotIn("--datasets xtask_sharegpt_gen", script)
        self.assertNotIn("AISBench数据集路径不存在", script)
        find_config.assert_not_called()

    def test_gedit_acc_defaults_dataset_path_when_missing(self):
        # gedit（api_type 以 gedit 开头）未指定 dataset_path 时落到
        # DEPENDENCE_PACKAGE_PATH 下的 GEdit-Bench 默认路径。
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = root / "gedit_gen_0_shot_llmjudge.py"
            source_config.write_text(
                "from example import GEditDataset\n"
                "gedit_datasets = [\n"
                "    dict(\n"
                "        abbr='gedit',\n"
                "        type=GEditDataset,\n"
                "        path='ais_bench/datasets/GEdit-Bench',\n"
                "    )\n"
                "]\n",
                encoding="utf-8",
            )
            task_config = {
                "benchmark": {
                    "benchmark_type": "aisbench",
                    "acc": {
                        "api_type": "gedit_image_edit",
                        "dataset_conf": "gedit_gen_0_shot_llmjudge",
                    },
                },
                "dataset_name": "gedit",
                "host_addr": "127.0.0.1",
                "metrics": ["acc"],
                "model_path": "/models/qwen-image-edit",
                "port": 19000,
                "served_model_name": "qwen-image-edit",
            }
            with mock.patch.dict(
                os.environ, {"DEPENDENCE_PACKAGE_PATH": "/xllm_test_060"}
            ), mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config",
                return_value=source_config,
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            dataset_config = (
                output_dir / "aisbench_configs" / "datasets"
                / "xtask_gedit_gen_0_shot_llmjudge.py"
            ).read_text(encoding="utf-8")

        default_path = (
            "/xllm_test_060/benchmark/ais_bench/datasets/GEdit-Bench"
        )
        self.assertIn(f"path='{default_path}'", dataset_config)

    def test_gedit_acc_explicit_dataset_path_wins(self):
        # 显式 dataset_path 始终优先于默认值。
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = root / "gedit_gen_0_shot_llmjudge.py"
            source_config.write_text(
                "from example import GEditDataset\n"
                "gedit_datasets = [\n"
                "    dict(\n"
                "        abbr='gedit',\n"
                "        type=GEditDataset,\n"
                "        path='ais_bench/datasets/GEdit-Bench',\n"
                "    )\n"
                "]\n",
                encoding="utf-8",
            )
            task_config = {
                "benchmark": {
                    "benchmark_type": "aisbench",
                    "acc": {
                        "api_type": "gedit_image_edit",
                        "dataset_conf": "gedit_gen_0_shot_llmjudge",
                        "dataset_path": "/custom/GEdit-Bench",
                    },
                },
                "dataset_name": "gedit",
                "host_addr": "127.0.0.1",
                "metrics": ["acc"],
                "model_path": "/models/qwen-image-edit",
                "port": 19000,
                "served_model_name": "qwen-image-edit",
            }
            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config",
                return_value=source_config,
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            dataset_config = (
                output_dir / "aisbench_configs" / "datasets"
                / "xtask_gedit_gen_0_shot_llmjudge.py"
            ).read_text(encoding="utf-8")

        self.assertIn("path='/custom/GEdit-Bench'", dataset_config)

    def test_non_gedit_acc_keeps_empty_dataset_path_semantics(self):
        # 非 gedit 的 aisbench acc 用例：dataset_path 未指定时不注入默认值，
        # 保持原有必填/相对路径语义不变。
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = root / "gsm8k_gen.py"
            source_config.write_text(
                "from example import GSM8KDataset\n"
                "gsm8k_datasets = [\n"
                "    dict(\n"
                "        abbr='gsm8k',\n"
                "        type=GSM8KDataset,\n"
                "        path='ais_bench/datasets/gsm8k',\n"
                "    )\n"
                "]\n",
                encoding="utf-8",
            )
            task_config = {
                "benchmark": {
                    "benchmark_type": "aisbench",
                    "acc": {
                        "api_type": "vllm_api_general_chat",
                        "dataset_conf": "gsm8k_gen",
                    },
                },
                "dataset_name": "gsm8k",
                "host_addr": "127.0.0.1",
                "metrics": ["acc"],
                "model_path": "/models/deepseek",
                "port": 8800,
                "served_model_name": "deepseek",
            }
            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config",
                return_value=source_config,
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            # create_dataset_config_override 没有 dataset_path/infer_mode/
            # judge_endpoint 时直接返回原名，不生成 xtask_ 覆盖配置。
            self.assertFalse(
                (
                    output_dir / "aisbench_configs" / "datasets"
                    / "xtask_gsm8k_gen.py"
                ).exists()
            )

    def test_sharegpt_last_fixed_length_uses_generated_dataset_and_case_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = root / "sharegpt_gen.py"
            source_config.write_text(
                "from example import MultiTurnGenInferencer, ShareGPTDataset\n"
                "sharegpt_infer_cfg = dict(\n"
                "    inferencer=dict(type=MultiTurnGenInferencer, infer_mode='every')\n"
                ")\n"
                "sharegpt_datasets = [dict(\n"
                "    abbr='sharegpt',\n"
                "    type=ShareGPTDataset,\n"
                "    path='ais_bench/datasets/sharegpt/default.json',\n"
                "    infer_cfg=sharegpt_infer_cfg,\n"
                ")]\n",
                encoding="utf-8",
            )
            dataset_path = "/datasets/sharegpt/source.json"
            task_config = self._task_config()
            task_config["dataset_name"] = "sharegpt"
            task_config["dataset_path"] = dataset_path
            task_config["benchmark"]["perf"].update(
                {
                    "api_type": "vllm_api_stream_chat_multiturn",
                    "dataset_conf": "sharegpt_gen",
                    "infer_mode": "last",
                    "input_len": 2048,
                    "num_prompts": 8,
                }
            )

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch(
                "xtask.utils.container_manager.path_mapping_in_container",
                side_effect=lambda path: path,
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config", return_value=source_config
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )
            dataset_config = (
                output_dir
                / "aisbench_configs"
                / "datasets"
                / "xtask_sharegpt_gen.py"
            ).read_text(encoding="utf-8")

        generated_path = output_dir / "ShareGPT-in2048-n8-last.json"
        self.assertIn("--datasettype=SHAREGPT", script)
        self.assertIn("--inputlen=2048", script)
        self.assertIn("--bs=8", script)
        self.assertIn(f'--sourcepath="$SHAREGPT_SOURCE"', script)
        self.assertIn(f"--outputpath={generated_path}", script)
        self.assertIn("if ! python ", script)
        self.assertIn("python -m pip install 'jinja2>=3.1,<4'", script)
        self.assertIn("export PYTHONPATH=", script)
        self.assertIn("--datasets xtask_sharegpt_gen", script)
        self.assertIn("infer_mode='last'", dataset_config)
        self.assertIn(
            "from xtask.benchmarks.aisbench.sharegpt_last_inferencer import "
            "XTaskShareGPTLastInferencer",
            dataset_config,
        )
        self.assertIn("type=XTaskShareGPTLastInferencer", dataset_config)
        self.assertIn(f"path='{generated_path}'", dataset_config)
        self.assertNotIn(
            "cp -f " + str(generated_path) + " $AIS_BENCH_PATH/datasets/sharegpt",
            script,
        )

    def test_sharegpt_last_without_input_len_keeps_source_data_unmodified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = root / "sharegpt_gen.py"
            source_config.write_text(
                "from example import MultiTurnGenInferencer, ShareGPTDataset\n"
                "sharegpt_infer_cfg = dict(\n"
                "    inferencer=dict(type=MultiTurnGenInferencer, infer_mode='every')\n"
                ")\n"
                "sharegpt_datasets = [dict(\n"
                "    abbr='sharegpt', type=ShareGPTDataset,\n"
                "    path='ais_bench/datasets/sharegpt/default.json',\n"
                "    infer_cfg=sharegpt_infer_cfg,\n"
                ")]\n",
                encoding="utf-8",
            )
            task_config = self._task_config()
            task_config["dataset_name"] = "sharegpt"
            task_config["benchmark"]["perf"].update(
                {
                    "api_type": "vllm_api_stream_chat_multiturn",
                    "dataset_conf": "sharegpt_gen",
                    "infer_mode": "last",
                    "num_prompts": 8,
                }
            )

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config", return_value=source_config
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )
            dataset_config = (
                output_dir
                / "aisbench_configs"
                / "datasets"
                / "xtask_sharegpt_gen.py"
            ).read_text(encoding="utf-8")

        self.assertNotIn("--datasettype=SHAREGPT", script)
        self.assertIn("export PYTHONPATH=", script)
        self.assertIn("--datasets xtask_sharegpt_gen", script)
        self.assertIn("infer_mode='last'", dataset_config)
        self.assertIn("type=XTaskShareGPTLastInferencer", dataset_config)
        self.assertIn(
            "path='ais_bench/datasets/sharegpt/default.json'", dataset_config
        )

    def test_generic_perf_dataset_uses_yaml_config_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            output_dir.mkdir()
            task_config = self._task_config()
            task_config["dataset_name"] = "future_dataset"
            task_config["benchmark"]["perf"]["dataset_conf"] = "future_gen"

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator, "preprocess_dataset", return_value=([], "")
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )

        self.assertIn("--datasets future_gen", script)
        self.assertNotIn("xtask_future_gen.py", script)

    def test_sharegpt_prefix_cache_uses_case_local_dataset_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = root / "sharegpt_gen.py"
            source_config.write_text(
                "from example import MultiTurnGenInferencer, ShareGPTDataset\n"
                "sharegpt_infer_cfg = dict(\n"
                "    inferencer=dict(type=MultiTurnGenInferencer, infer_mode='every')\n"
                ")\n"
                "sharegpt_datasets = [dict(\n"
                "    abbr='sharegpt', type=ShareGPTDataset,\n"
                "    path='ais_bench/datasets/sharegpt/default.json',\n"
                "    infer_cfg=sharegpt_infer_cfg,\n"
                ")]\n",
                encoding="utf-8",
            )
            task_config = self._task_config()
            task_config["dataset_name"] = "sharegpt"
            task_config["benchmark"]["perf"].update(
                {
                    "api_type": "vllm_api_stream_chat_multiturn",
                    "dataset_conf": "sharegpt_gen",
                    "infer_mode": "last",
                    "input_len": 2048,
                    "num_prompts": 8,
                    "num_warmups": 4,
                    "prefix_cache": {
                        "enabled": True,
                        "prefix_source": "dataset",
                        "prefix_len": 32,
                        "prefix_groups": 4,
                        "seed_requests": True,
                    },
                }
            )

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator,
                "preprocess_dataset",
                return_value=(
                    ["prepare-prefix-dataset"],
                    "",
                ),
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config", return_value=source_config
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )
            dataset_config = (
                output_dir
                / "aisbench_configs"
                / "datasets"
                / "xtask_sharegpt_gen.py"
            ).read_text(encoding="utf-8")

        generated_path = output_dir / "ShareGPT-prefix-cache-i2048-p32-g4-n8.json"
        self.assertIn("prepare-prefix-dataset", script)
        self.assertIn("--datasets xtask_sharegpt_gen", script)
        self.assertIn("--num-prompts 8", script)
        self.assertIn("infer_mode='last'", dataset_config)
        self.assertIn("type=XTaskShareGPTLastInferencer", dataset_config)
        self.assertIn(f"path='{generated_path}'", dataset_config)

    def test_prefix_cache_requires_warmup_for_each_seed_group(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._task_config()
        task_config["dataset_name"] = "sharegpt"
        task_config["benchmark"]["perf"].update(
            {
                "infer_mode": "last",
                "input_len": 2048,
                "num_prompts": 8,
                "num_warmups": 3,
                "prefix_cache": {
                    "enabled": True,
                    "prefix_source": "dataset",
                    "prefix_len": 512,
                    "prefix_groups": 4,
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "num_warmups >= prefix_groups"):
            preprocess_dataset("sharegpt", task_config, "/tmp/xtask-prefix-test")

    def test_prefix_cache_preprocessor_passes_fixed_input_and_dataset_source(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._task_config()
        task_config["dataset_name"] = "sharegpt"
        task_config["benchmark"]["perf"].update(
            {
                "infer_mode": "last",
                "input_len": 2048,
                "num_prompts": 8,
                "num_warmups": 4,
                "prefix_cache": {
                    "enabled": True,
                    "prefix_source": "dataset",
                    "prefix_len": 512,
                    "prefix_groups": 4,
                    "seed_requests": True,
                    "order": "grouped",
                },
            }
        )

        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            commands, dataset_conf = preprocess_dataset(
                "sharegpt", task_config, "/tmp/xtask-prefix-test"
            )

        command_text = "\n".join(commands)
        self.assertEqual(dataset_conf, "")
        self.assertIn("--datasettype=SHAREGPT_PREFIX_CACHE", command_text)
        self.assertIn("--infermode=last", command_text)
        self.assertIn("--inputlen=2048", command_text)
        self.assertIn("--prefix-len=512", command_text)
        self.assertIn("--prefix-source=dataset", command_text)
        self.assertIn("--prefix-assembly=single", command_text)
        self.assertNotIn("prefix-text", command_text)

        task_config["benchmark"]["perf"]["prefix_cache"][
            "prefix_assembly"
        ] = "packed"
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            packed_commands, _ = preprocess_dataset(
                "sharegpt", task_config, "/tmp/xtask-prefix-test"
            )
        packed_command_text = "\n".join(packed_commands)
        self.assertIn("--prefix-assembly=packed", packed_command_text)
        self.assertIn("-p512-apacked-g4-n8.json", packed_command_text)

        task_config["benchmark"]["perf"]["prefix_cache"][
            "prefix_assembly"
        ] = "invalid"
        with self.assertRaisesRegex(ValueError, "must be single or packed"):
            preprocess_dataset(
                "sharegpt", task_config, "/tmp/xtask-prefix-test"
            )

    def test_prefix_cache_rejects_non_last_mode(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._task_config()
        task_config["dataset_name"] = "sharegpt"
        task_config["benchmark"]["perf"].update(
            {
                "infer_mode": "every_with_gt",
                "input_len": 2048,
                "num_prompts": 8,
                "num_warmups": 4,
                "prefix_cache": {
                    "enabled": True,
                    "prefix_source": "dataset",
                    "prefix_len": 512,
                    "prefix_groups": 4,
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "only supports infer_mode=last"):
            preprocess_dataset("sharegpt", task_config, "/tmp/xtask-prefix-test")

    def test_preprocessed_dataset_keeps_specialized_path_handling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            output_dir.mkdir()
            task_config = self._task_config()
            task_config["benchmark"]["perf"]["dataset_path"] = "/custom/gsm8k"

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator,
                "preprocess_dataset",
                return_value=(["prepare-gsm8k"], "generated_gsm8k"),
            ), mock.patch.object(
                generator, "find_aisbench_dataset_config"
            ) as find_config:
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )

        self.assertIn("prepare-gsm8k", script)
        self.assertIn("--datasets generated_gsm8k", script)
        self.assertNotIn("AISBench数据集路径不存在", script)
        find_config.assert_not_called()

    def test_stale_source_path_is_overwritten_and_failure_is_propagated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            source_config = (
                root
                / "source"
                / "ais_bench"
                / "benchmark"
                / "configs"
                / "models"
                / "vllm_api"
                / "vllm_api_general_stream.py"
            )
            source_config.parent.mkdir(parents=True)
            source_config.write_text(
                "models=[dict(\n"
                "    path='/models/glm',\n"
                "    host_ip='old',\n"
                "    host_port=1,\n"
                "    model='old',\n"
                "    request_rate=1,\n"
                "    max_out_len=1,\n"
                "    batch_size=1,\n"
                "    temperature=1,\n"
                "    top_k=1,\n"
                "    top_p=1,\n"
                "    seed=1,\n"
                "    repetition_penalty=1,\n"
                "    ignore_eos=False,\n"
                ")]\n",
                encoding="utf-8",
            )

            fake_bin = root / "bin"
            fake_bin.mkdir()
            python_exec = fake_bin / "python"
            python_exec.write_text("#!/bin/sh\necho /unused/ais_bench\n", encoding="utf-8")
            python_exec.chmod(0o755)
            ais_bench = fake_bin / "ais_bench"
            ais_bench.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
            ais_bench.chmod(0o755)

            script_path = self._generate(output_dir, python_exec)
            # The generated script hardcodes /xllm_test_060/benchmark as the
            # AISBench source root; redirect it to the test tree so the
            # stale source config can be found without touching system paths.
            script_text = script_path.read_text(encoding="utf-8")
            script_text = script_text.replace(
                "/xllm_test_060/benchmark", str(root / "source")
            )
            script_path.write_text(script_text, encoding="utf-8")
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            result = subprocess.run(
                ["bash", str(script_path)],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

            ais_bench.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            success_result = subprocess.run(
                ["bash", str(script_path)],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

            isolated_config = (
                output_dir
                / "aisbench_configs"
                / "models"
                / "xtask_vllm_api_general_stream.py"
            )
            isolated_content = isolated_config.read_text(encoding="utf-8")
            source_content = source_config.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertIn("AISBench性能测试执行失败", result.stderr)
        self.assertEqual(success_result.returncode, 0, success_result.stderr)
        self.assertIn("性能测试完成", success_result.stdout)
        self.assertIn("path='/models/deepseek'", isolated_content)
        self.assertIn("path='/models/glm'", source_content)

    def test_centralized_execution_outputs_keep_separate_config_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            python_exec = root / "fake-python"
            python_exec.write_text(
                "#!/bin/sh\necho /unused/ais_bench\n", encoding="utf-8"
            )
            python_exec.chmod(0o755)
            output_names = [
                "single_case",
                "auto_case_bs-2",
                "para_case_sg1",
                "tpe_case_sg2",
            ]

            scripts = {}
            for output_name in output_names:
                output_dir = root / output_name
                output_dir.mkdir()
                script_path = self._generate(output_dir, python_exec)
                scripts[output_name] = script_path.read_text(encoding="utf-8")

        for output_name, script in scripts.items():
            expected = root / output_name / "aisbench_configs"
            self.assertIn(f'AIS_BENCH_CONFIG_DIR="{expected}"', script)
            for other_name in output_names:
                if other_name != output_name:
                    unexpected = root / other_name / "aisbench_configs"
                    self.assertNotIn(str(unexpected), script)

    def test_auto_assess_regenerates_aisbench_script_for_each_trial_output(self):
        from xtask.core import scheduler
        from xtask.core.task_scheduler import TaskExecutionContext

        base_config = self._task_config()
        base_config["case_name"] = "auto_case_bs-1"
        next_config = copy.deepcopy(base_config)
        next_config["case_name"] = "auto_case_bs-2"
        next_config["benchmark"]["perf"]["batch_size"] = 2

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_output = root / "auto_case_bs-1"
            second_output = root / "auto_case_bs-2"
            context = TaskExecutionContext(
                "base.yaml", str(first_output), result={}
            )
            context.state.update(
                source_task_config=base_config,
                service_plans=[],
            )
            session = scheduler.SchedulerSession(
                scheduler=mock.Mock(),
                context=context,
                base_output_file_path=str(first_output),
                prepared=True,
            )

            with mock.patch.object(
                scheduler.TaskConfig,
                "load_task_config",
                side_effect=[base_config, next_config],
            ), mock.patch.object(
                generator,
                "path_mapping_in_container",
                side_effect=lambda path: path,
            ), mock.patch.object(
                generator, "download_dataset", return_value=[]
            ), mock.patch.object(
                generator,
                "preprocess_dataset",
                return_value=(["true"], "gsm8k_gen_0_shot_cot_str_perf"),
            ):
                scheduler._reset_session_trial(
                    session, "base.yaml", str(first_output)
                )
                scheduler._reset_session_trial(
                    session, "next.yaml", str(second_output)
                )

            first_script = (first_output / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )
            second_script = (second_output / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )

        self.assertIn(
            f'AIS_BENCH_CONFIG_DIR="{first_output}/aisbench_configs"',
            first_script,
        )
        self.assertIn(
            f'AIS_BENCH_CONFIG_DIR="{second_output}/aisbench_configs"',
            second_script,
        )
        self.assertEqual(context.result["output_file_path"], str(second_output))

    def test_find_prefers_aisbench_dataset_config_over_xtask(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ais_root = root / "ais"
            xtask_root = root / "xtask"
            ais_config = (
                ais_root
                / "ais_bench"
                / "benchmark"
                / "configs"
                / "datasets"
                / "custom"
                / "custom_dataset_gen.py"
            )
            xtask_config = xtask_root / "custom_dataset_gen.py"
            ais_config.parent.mkdir(parents=True)
            xtask_root.mkdir()
            ais_config.write_text("# aisbench\n", encoding="utf-8")
            xtask_config.write_text("# xtask\n", encoding="utf-8")

            with mock.patch.object(
                generator, "_aisbench_dataset_config_roots", return_value=[ais_root]
            ), mock.patch.object(
                generator, "_xtask_aisbench_dataset_config_root", return_value=xtask_root
            ):
                found = generator.find_aisbench_dataset_config("custom_dataset_gen")

        self.assertEqual(found, ais_config)

    def test_find_falls_back_to_xtask_dataset_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            xtask_root = Path(temporary_directory) / "xtask"
            xtask_config = xtask_root / "custom_dataset_gen.py"
            xtask_root.mkdir()
            xtask_config.write_text("# xtask\n", encoding="utf-8")

            with mock.patch.object(
                generator, "_aisbench_dataset_config_roots", return_value=[]
            ), mock.patch.object(
                generator, "_xtask_aisbench_dataset_config_root", return_value=xtask_root
            ):
                found = generator.find_aisbench_dataset_config("custom_dataset_gen")

        self.assertEqual(found, xtask_config)

    def test_custom_dataset_perf_uses_xtask_dataset_config_and_stream_chat(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            output_dir.mkdir()
            dataset_path = "/workspace/xtask/datasets/custom/prompt_30.jsonl"
            task_config = self._task_config()
            task_config["dataset_name"] = "custom_dataset"
            task_config["dataset_path"] = dataset_path
            task_config["benchmark"]["perf"].update(
                {
                    "api_type": "vllm_api_stream_chat",
                    "dataset_conf": "custom_dataset_gen",
                    "batch_size": 1,
                    "request_rate": 0,
                    "num_prompts": 30,
                    "num_warmups": 5,
                    "output_len": 128,
                    "ignore_eos": False,
                    "generation_kwargs": {
                        "temperature": 0,
                        "ignore_eos": False,
                    },
                }
            )

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "_aisbench_dataset_config_roots", return_value=[]
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )
            dataset_config = (
                output_dir
                / "aisbench_configs"
                / "datasets"
                / "xtask_custom_dataset_gen.py"
            ).read_text(encoding="utf-8")

        generated_path = output_dir / "prompt_30-n30.jsonl"
        self.assertIn("--models xtask_vllm_api_stream_chat", script)
        self.assertIn("--datasets xtask_custom_dataset_gen", script)
        self.assertIn("--num-prompts 30", script)
        self.assertIn("--num-warmups 5", script)
        self.assertIn("ignore_eos=False", script)
        self.assertIn("repeat_jsonl.py", script)
        self.assertIn(
            f"错误：AISBench数据集路径不存在 {dataset_path}",
            script,
        )
        self.assertIn("按 num_prompts 重复拼接自定义数据集 jsonl", script)
        self.assertIn(f"path='{generated_path}'", dataset_config)
        self.assertNotIn(f"path='{dataset_path}'", dataset_config)
        self.assertIn("template='{question}'", dataset_config)
        self.assertNotIn("Question: {question}", dataset_config)
        self.assertNotIn("gsm8k_gen_0_shot_cot_str_perf", script)

    def test_custom_dataset_num_prompts_90_repeats_jsonl_in_generated_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            output_dir.mkdir()
            dataset_path = "/workspace/xtask/datasets/custom/prompt_30.jsonl"
            task_config = self._task_config()
            task_config["dataset_name"] = "custom_dataset"
            task_config["dataset_path"] = ""
            task_config["benchmark"]["perf"].update(
                {
                    "api_type": "vllm_api_stream_chat",
                    "dataset_conf": "custom_dataset_gen",
                    "dataset_path": dataset_path,
                    "batch_size": 1,
                    "request_rate": 0,
                    "num_prompts": 90,
                    "num_warmups": 5,
                    "output_len": 128,
                    "ignore_eos": False,
                }
            )

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "_aisbench_dataset_config_roots", return_value=[]
            ):
                generator.generate_aisbench_script(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_test.sh").read_text(
                encoding="utf-8"
            )
            dataset_config = (
                output_dir
                / "aisbench_configs"
                / "datasets"
                / "xtask_custom_dataset_gen.py"
            ).read_text(encoding="utf-8")

        generated_path = output_dir / "prompt_30-n90.jsonl"
        self.assertIn("--num-prompts 90", script)
        self.assertIn(f"--source {dataset_path}", script)
        self.assertIn(f"--output {generated_path}", script)
        self.assertIn(f"path='{generated_path}'", dataset_config)
        self.assertIn("repeat_jsonl.py", script)
        self.assertIn("/workspace/xtask/datasets/custom/repeat_jsonl.py", script)

    def test_custom_alias_preprocesses_like_custom_dataset(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._task_config()
        task_config["dataset_name"] = "custom"
        task_config["benchmark"]["perf"].update(
            {
                "dataset_path": "/workspace/xtask/datasets/custom/log_request.jsonl",
                "num_prompts": 200,
            }
        )

        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            commands, dataset_conf = preprocess_dataset(
                "custom", task_config, "/tmp/xtask-custom-test"
            )

        command_text = "\n".join(commands)
        self.assertEqual(dataset_conf, "custom_dataset_gen")
        self.assertIn("--num-prompts 200", command_text)
        self.assertIn("/tmp/xtask-custom-test/log_request-n200.jsonl", command_text)

    def test_custom_dataset_preprocess_repeats_source_jsonl(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._task_config()
        task_config["dataset_name"] = "custom_dataset"
        task_config["benchmark"]["perf"].update(
            {
                "dataset_conf": "custom_dataset_gen",
                "dataset_path": "/workspace/xtask/datasets/custom/prompt_30.jsonl",
                "num_prompts": 90,
            }
        )

        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            commands, dataset_conf = preprocess_dataset(
                "custom_dataset", task_config, "/tmp/xtask-custom-test"
            )

        command_text = "\n".join(commands)
        self.assertEqual(dataset_conf, "custom_dataset_gen")
        self.assertIn("repeat_jsonl.py", command_text)
        self.assertIn("--num-prompts 90", command_text)
        self.assertIn("/tmp/xtask-custom-test/prompt_30-n90.jsonl", command_text)
        self.assertIn(
            "/workspace/xtask/datasets/custom/prompt_30.jsonl",
            command_text,
        )

    def test_custom_dataset_preprocess_requires_positive_num_prompts(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._task_config()
        task_config["dataset_name"] = "custom_dataset"
        task_config["benchmark"]["perf"].update(
            {
                "dataset_path": "/workspace/xtask/datasets/custom/prompt_30.jsonl",
                "num_prompts": 0,
            }
        )
        with self.assertRaisesRegex(ValueError, "num_prompts must be a positive integer"):
            preprocess_dataset(
                "custom_dataset", task_config, "/tmp/xtask-custom-test"
            )

    def test_reserved_dataset_name_skips_custom_preprocess(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._task_config()
        task_config["dataset_name"] = "gsm8k"
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            commands, dataset_conf = preprocess_dataset(
                "gsm8k", task_config, "/tmp/xtask-gsm8k-test"
            )
        command_text = "\n".join(commands)
        self.assertEqual(dataset_conf, "gsm8k_gen_0_shot_cot_str_perf")
        self.assertNotIn("repeat_jsonl.py", command_text)

    def test_repeat_jsonl_records_pads_and_truncates(self):
        import importlib.util

        module_path = (
            Path(__file__).resolve().parents[1]
            / "xtask"
            / "datasets"
            / "custom"
            / "repeat_jsonl.py"
        )
        spec = importlib.util.spec_from_file_location("repeat_jsonl", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        records = ["a\n", "b\n", "c\n"]
        self.assertEqual(module.repeat_jsonl_records(records, 3), records)
        self.assertEqual(
            module.repeat_jsonl_records(records, 7),
            ["a\n", "b\n", "c\n", "a\n", "b\n", "c\n", "a\n"],
        )
        self.assertEqual(module.repeat_jsonl_records(records, 2), ["a\n", "b\n"])
        with self.assertRaisesRegex(ValueError, "jsonl dataset is empty"):
            module.repeat_jsonl_records([], 1)

    def _custom_warmup_task_config(self, warmup_paras, num_prompts=200):
        task_config = self._task_config()
        task_config["dataset_name"] = "custom_dataset"
        task_config["benchmark"]["perf"].update(
            {
                "dataset_conf": "custom_dataset_gen",
                "dataset_path": "/workspace/xtask/datasets/custom/log_request.jsonl",
                "num_prompts": num_prompts,
                "warmup_paras": warmup_paras,
            }
        )
        return task_config

    def test_custom_dataset_warmup_paras_is_optional(self):
        from xtask.datasets.dataset_manger import parse_custom_dataset_warmup_paras

        task_config = self._custom_warmup_task_config(warmup_paras=None)
        task_config["benchmark"]["perf"].pop("warmup_paras", None)
        self.assertIsNone(parse_custom_dataset_warmup_paras(task_config))

    def test_custom_dataset_warmup_paras_requires_mapping_and_num_prompts(self):
        from xtask.datasets.dataset_manger import parse_custom_dataset_warmup_paras

        with self.assertRaisesRegex(ValueError, "warmup_paras must be a mapping"):
            parse_custom_dataset_warmup_paras(
                self._custom_warmup_task_config(warmup_paras="10")
            )
        with self.assertRaisesRegex(ValueError, "warmup_paras.num_prompts"):
            parse_custom_dataset_warmup_paras(
                self._custom_warmup_task_config(warmup_paras={"batch_size": 1})
            )
        with self.assertRaisesRegex(ValueError, "warmup_paras.num_prompts"):
            parse_custom_dataset_warmup_paras(
                self._custom_warmup_task_config(warmup_paras={"num_prompts": 0})
            )
        with self.assertRaisesRegex(ValueError, "warmup_paras.dataset_path"):
            parse_custom_dataset_warmup_paras(
                self._custom_warmup_task_config(
                    warmup_paras={"num_prompts": 10, "dataset_path": "  "}
                )
            )

    def test_custom_dataset_warmup_paras_rejects_invalid_during_preprocess(self):
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._custom_warmup_task_config(
            warmup_paras={"num_prompts": 0}
        )
        with self.assertRaisesRegex(ValueError, "warmup_paras.num_prompts"):
            preprocess_dataset(
                "custom_dataset", task_config, "/tmp/xtask-custom-warmup"
            )

    def test_custom_dataset_warmup_overlay_repeats_jsonl_to_warmup_num_prompts(self):
        from xtask.core.scheduler import _build_warmup_task_config
        from xtask.datasets.dataset_manger import preprocess_dataset

        task_config = self._custom_warmup_task_config(
            warmup_paras={
                "dataset_path": "/workspace/xtask/datasets/custom/log_request_warmup.jsonl",
                "num_prompts": 10,
                "batch_size": 1,
                "request_rate": 0,
            }
        )
        warmup_task_config, _ = _build_warmup_task_config(task_config)
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            commands, dataset_conf = preprocess_dataset(
                "custom_dataset",
                warmup_task_config,
                "/tmp/xtask-custom-warmup",
            )

        command_text = "\n".join(commands)
        self.assertEqual(dataset_conf, "custom_dataset_gen")
        self.assertNotIn("warmup_paras", warmup_task_config["benchmark"]["perf"])
        self.assertIn("--num-prompts 10", command_text)
        self.assertIn(
            "/workspace/xtask/datasets/custom/log_request_warmup.jsonl",
            command_text,
        )
        self.assertIn(
            "/tmp/xtask-custom-warmup/log_request_warmup-n10.jsonl",
            command_text,
        )
        self.assertNotIn("log_request-n200.jsonl", command_text)

    def test_custom_dataset_warmup_script_uses_isolated_config_dir(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            output_dir.mkdir()
            task_config = self._custom_warmup_task_config(
                warmup_paras={
                    "dataset_path": "/workspace/xtask/datasets/custom/log_request_warmup.jsonl",
                    "num_prompts": 10,
                    "batch_size": 1,
                    "request_rate": 0,
                }
            )
            task_config["benchmark"]["perf"]["api_type"] = "vllm_api_stream_chat"
            task_config["benchmark"]["perf"]["num_warmups"] = 5

            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ), mock.patch.object(
                generator, "_aisbench_dataset_config_roots", return_value=[]
            ), mock.patch(
                "xtask.utils.container_manager.path_mapping_in_container",
                side_effect=lambda path: path,
            ):
                command = scheduler._build_warmup_command(task_config, str(output_dir))

            script = (output_dir / "run_aisbench_warmup.sh").read_text(
                encoding="utf-8"
            )
            dataset_config = (
                output_dir
                / "aisbench_configs_warmup"
                / "datasets"
                / "xtask_custom_dataset_gen.py"
            ).read_text(encoding="utf-8")

        self.assertIn("run_aisbench_warmup.sh", command)
        self.assertIn("aisbench_configs_warmup", script)
        self.assertIn("--num-prompts 10", script)
        self.assertIn("--num-warmups 0", script)
        self.assertIn("log_request_warmup-n10.jsonl", script)
        self.assertIn("log_request_warmup-n10.jsonl", dataset_config)
        self.assertFalse((output_dir / "aisbench_configs").exists())

    def test_should_run_perf_warmup_only_for_evalscope_or_custom_dataset(self):
        from xtask.core.scheduler import _should_run_perf_warmup

        evalscope = self._task_config()
        evalscope["benchmark"]["benchmark_type"] = "evalscope"
        evalscope["benchmark"]["perf"]["warmup_paras"] = {"num_prompts": 20}
        self.assertTrue(_should_run_perf_warmup(evalscope))

        gsm8k = self._task_config()
        gsm8k["benchmark"]["perf"]["warmup_paras"] = {"num_prompts": 10}
        self.assertFalse(_should_run_perf_warmup(gsm8k))

        custom = self._custom_warmup_task_config(
            warmup_paras={"num_prompts": 10}
        )
        self.assertTrue(_should_run_perf_warmup(custom))

        custom_without = self._custom_warmup_task_config(warmup_paras=None)
        custom_without["benchmark"]["perf"].pop("warmup_paras", None)
        self.assertFalse(_should_run_perf_warmup(custom_without))


if __name__ == "__main__":
    unittest.main()
