# Copyright 2026 The xTask Authors.
"""Diffusers layout support is opt-in; legacy dependency preparation is unchanged."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler


class TextVideoDependencyTests(unittest.TestCase):
    def _context(self, root, benchmark, multi_host=False):
        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(success=True, stdout="", stderr="")
        container = SimpleNamespace(container_name="model-container")
        runtime = SimpleNamespace(
            plan=SimpleNamespace(instance_name="master-0", host="127.0.0.1", is_benchmark_runner=True),
            container_config=container, container_manager=manager,
        )
        return SimpleNamespace(
            state={
                "task_config": {"model_path": str(root), "benchmark": benchmark},
                "engine_config": {"startup_timeout": 30},
                "container_config": container, "container_manager": manager,
                "multi_host": multi_host, "service_runtimes": [runtime], "service_plans": [],
            }, result={}, complete=False, output_file_path=str(root / "outputs"),
        )

    def _prepare(self, context):
        with mock.patch("xtask.core.ranktable_manager.prepare_multihost_ranktables", return_value={}), mock.patch.object(
            scheduler, "_clear_pre_benchmark_rst"
        ):
            scheduler._prepare_runtime_dependencies(context)

    def _benchmark(self, task="t2av"):
        return {"benchmark_type": "vbench", "vbench": {"task": task}}

    def test_component_layout_supported_in_both_runtime_paths_without_pip_upgrade(self):
        for multi_host in (False, True):
            for task in ("t2v", "t2av"):
                with self.subTest(multi_host=multi_host, task=task), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    index = root / "model_index.json"
                    index.write_text('{"_class_name": "JoyOV2Pipeline"}', encoding="utf-8")
                    original = index.read_bytes()
                    context = self._context(root, self._benchmark(task), multi_host)
                    self._prepare(context)
                    self.assertFalse(context.complete)
                    commands = [call.args[1] for call in context.state["container_manager"].exec_in_container.call_args_list]
                    self.assertEqual(commands, ["XLLM_INSTALL_SERVICE=0 bash /workspace/tools/download_xllm.sh;"])
                    self.assertEqual(index.read_bytes(), original)
                    self.assertFalse((root / "config.json").exists())

    def test_root_config_pin_takes_precedence_for_all_models(self):
        benchmarks = [self._benchmark(), self._benchmark("i2v"), {"benchmark_type": "aisbench"}]
        for multi_host in (False, True):
            for benchmark in benchmarks:
                with self.subTest(multi_host=multi_host, benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "config.json").write_text('{"transformers_version": "4.51.0"}', encoding="utf-8")
                    (root / "model_index.json").write_text('invalid JSON', encoding="utf-8")
                    context = self._context(root, benchmark, multi_host)
                    self._prepare(context)
                    self.assertFalse(context.complete)
                    commands = [call.args[1] for call in context.state["container_manager"].exec_in_container.call_args_list]
                    self.assertIn("pip install transformers==4.51.0 --upgrade", commands)

    def test_legacy_layout_never_falls_back_to_component_index(self):
        benchmarks = [self._benchmark("i2v"), {"benchmark_type": "vbench", "vbench": {}},
                      {"benchmark_type": "aisbench", "vbench": {"task": "t2av"}}]
        for multi_host in (False, True):
            for benchmark in benchmarks:
                with self.subTest(multi_host=multi_host, benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "model_index.json").write_text('{"_class_name": "JoyOV2Pipeline"}', encoding="utf-8")
                    context = self._context(root, benchmark, multi_host)
                    self._prepare(context)
                    self.assertTrue(context.complete)
                    self.assertEqual(context.result, {})  # Preserve the legacy missing-config behavior.

    def test_missing_or_invalid_component_index_fails_with_reason(self):
        invalid = [None, 'broken', '[]', '{}', '{"_class_name": "  "}']
        for multi_host in (False, True):
            for value in invalid:
                with self.subTest(multi_host=multi_host, value=value), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    if value is not None:
                        (root / "model_index.json").write_text(value, encoding="utf-8")
                    context = self._context(root, self._benchmark(), multi_host)
                    self._prepare(context)
                    self.assertTrue(context.complete)
                    self.assertIn("Invalid text-video model index", context.result["message"])
                    self.assertIn("model_index.json", context.result["message"])


if __name__ == "__main__":
    unittest.main()
