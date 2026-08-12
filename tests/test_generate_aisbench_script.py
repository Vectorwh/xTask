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
            env = os.environ.copy()
            env["AISBENCH_SOURCE_DIR"] = str(root / "source")
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


if __name__ == "__main__":
    unittest.main()
