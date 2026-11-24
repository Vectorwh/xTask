import os
import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.benchmarks.evalscope import generate_evalscope_script as generator


class EvalscopeScriptGenerationTests(unittest.TestCase):
    def test_accuracy_timeout_is_in_generation_config_and_explicit_value_wins(self):
        for explicit in (None, 17):
            with self.subTest(explicit_timeout=explicit), tempfile.TemporaryDirectory() as root:
                task = self._task_config(metrics="acc")
                if explicit is not None:
                    task["benchmark"]["acc"]["generation_config"]["timeout"] = explicit
                with mock.patch.object(generator, "path_mapping_in_container", side_effect=lambda path: path):
                    generator.generate_evalscope_script(task, root)
                script = (Path(root) / "run_evalscope_test.sh").read_text()
                command = next(line for line in script.splitlines() if line.startswith("if ! evalscope eval "))
                args = shlex.split(command.removeprefix("if ! ").removesuffix("; then"))
                generation = json.loads(args[args.index("--generation-config") + 1])
                self.assertEqual(generation["timeout"], 60 if explicit is None else explicit)
                self.assertNotIn("--timeout", args)

    def _task_config(self, metrics="perf"):
        return {
            "benchmark": {
                "benchmark_type": "evalscope",
                "acc": {
                    "batch_size": 1,
                    "dataset_args": {},
                    "endpoint": "/v1",
                    "generation_config": {},
                    "timeout": 60,
                },
                "perf": {
                    "batch_size": 1,
                    "max_out_len": 32,
                    "max_prompt_length": 16,
                    "min_prompt_length": 16,
                    "num_prompts": 1,
                    "output_len": 32,
                    "request_rate": 1,
                },
            },
            "dataset_name": "random",
            "host_addr": "127.0.0.1",
            "metrics": metrics,
            "model_path": "/models/test",
            "port": 8800,
            "served_model_name": "test-model",
        }

    def _generate(self, output_dir, metrics="perf"):
        with mock.patch.object(
            generator, "path_mapping_in_container", side_effect=lambda path: path
        ):
            generator.generate_evalscope_script(
                self._task_config(metrics), str(output_dir)
            )
        return output_dir / "run_evalscope_test.sh"

    def test_perf_failure_is_propagated_and_success_still_completes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            evalscope = fake_bin / "evalscope"
            evalscope.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
            evalscope.chmod(0o755)
            chown = fake_bin / "chown"
            chown.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            chown.chmod(0o755)

            script_path = self._generate(output_dir)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            failed = subprocess.run(
                ["bash", str(script_path)],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

            evalscope.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            succeeded = subprocess.run(
                ["bash", str(script_path)],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(failed.returncode, 1)
        self.assertIn("EvalScope性能测试执行失败", failed.stderr)
        self.assertNotIn("性能测试完成", failed.stdout)
        self.assertEqual(succeeded.returncode, 0, succeeded.stderr)
        self.assertIn("性能测试完成", succeeded.stdout)

    def test_accuracy_command_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            script = self._generate(output_dir, metrics="acc").read_text(
                encoding="utf-8"
            )

        self.assertIn("if ! evalscope eval ", script)
        self.assertIn('echo "EvalScope精度测试执行失败" >&2', script)

    def test_each_output_directory_is_embedded_in_its_own_script(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "case_sg1"
            second = root / "case_sg2"
            first.mkdir()
            second.mkdir()
            first_script = self._generate(first).read_text(encoding="utf-8")
            second_script = self._generate(second).read_text(encoding="utf-8")

        self.assertIn(f"--outputs-dir {first}", first_script)
        self.assertNotIn(f"--outputs-dir {second}", first_script)
        self.assertIn(f"--outputs-dir {second}", second_script)
        self.assertNotIn(f"--outputs-dir {first}", second_script)

    def test_evalscope_warmup_command_still_uses_overlayed_num_prompts(self):
        from xtask.core.scheduler import _build_warmup_command

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            task_config = self._task_config()
            task_config["benchmark"]["perf"]["warmup_paras"] = {
                "num_prompts": 20,
                "batch_size": 10,
                "request_rate": 10,
            }
            with mock.patch.object(
                generator, "path_mapping_in_container", side_effect=lambda path: path
            ):
                command = _build_warmup_command(task_config, str(output_dir))
            script = (output_dir / "run_evalscope_warmup.sh").read_text(
                encoding="utf-8"
            )

        self.assertIn("run_evalscope_warmup.sh", command)
        self.assertIn("--number 20", script)
        self.assertIn("--parallel 10", script)
        self.assertIn("--rate 10", script)


if __name__ == "__main__":
    unittest.main()
