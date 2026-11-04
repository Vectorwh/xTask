"""Ensure CLI display cannot mislabel YAML ASR cases as default AISBench."""

import unittest
from types import SimpleNamespace

from xtask.utils.arg_parser import ArgumentChecker, ConfigSource


class ArgumentSummaryTests(unittest.TestCase):
    def test_case_group_summary_does_not_use_direct_argument_defaults(self):
        checker = ArgumentChecker()
        args = SimpleNamespace(case_group="asr_real_repeats", benchmark_type="aisbench")
        config = checker._generate_config_data(args, ConfigSource.CASE_GROUP)
        self.assertEqual(config, {"source": "case_group", "case_group": "asr_real_repeats"})
        with self.assertLogs(level="INFO") as captured:
            checker._print_config_summary(config)
        output = "\n".join(captured.output)
        self.assertIn("Case Group: asr_real_repeats", output)
        self.assertNotIn("aisbench", output)

    def test_flat_yaml_summary_prints_actual_asr_backend(self):
        with self.assertLogs(level="INFO") as captured:
            ArgumentChecker()._print_config_summary({
                "served_model_name": "JoyaiASR", "engine_type": "xllm",
                "benchmark": {"benchmark_type": "asr"},
            })
        output = "\n".join(captured.output)
        self.assertIn("Model: JoyaiASR", output)
        self.assertIn("Benchmark: asr", output)
        self.assertNotIn("Benchmark: None", output)

    def test_direct_argument_summary_keeps_old_shape(self):
        with self.assertLogs(level="INFO") as captured:
            ArgumentChecker()._print_config_summary({
                "model": {"name": "model", "framework": "xllm", "path": "/model"},
                "benchmark": {"type": "aisbench"},
            })
        self.assertIn("Benchmark: aisbench", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
