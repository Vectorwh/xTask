"""Discrimination tests for the VBench task gate.

The gate must recognise both VBench YAML formats (new ``metrics: acc`` and
legacy ``metrics: vbench``) via ``benchmark.benchmark_type`` only, and must
reject every neighbouring scenario: plain accuracy (gpqa/gedit), AISBench
SWE-bench, AISBench perf and EvalScope perf.
"""

import unittest

from xtask.benchmarks.vbench.task_gate import is_vbench_task
from xtask.core import result_processor


def vbench_config(metrics="acc"):
    return {
        "metrics": metrics,
        "benchmark": {
            "benchmark_type": "vbench",
            "vbench": {
                "mode": "generate_and_evaluate",
                "evaluation": {"baseline": 0.9, "threshold": 0.1},
            },
        },
    }


def gpqa_config():
    return {
        "metrics": "acc",
        "benchmark": {
            "benchmark_type": "aisbench",
            "acc": {
                "dataset_conf": {"path": "gpqa"},
                "api_type": "vllm_api_general_chat",
                "baseline": 0.5,
                "threshold": 0.1,
            },
        },
    }


def gedit_config():
    return {
        "metrics": "acc",
        "benchmark": {
            "benchmark_type": "aisbench",
            "acc": {
                "dataset_conf": {"path": "gedit"},
                "api_type": "gedit_api_general",
                "generation_kwargs": {},
                "baseline": 0.5,
                "threshold": 0.1,
            },
        },
    }


def swebench_config():
    return {
        "metrics": "acc",
        "dataset_name": "swebench",
        "benchmark": {
            "benchmark_type": "aisbench",
            "acc": {
                "mode": "all",
                "runner": {"host": "10.0.0.1"},
                "baseline": 0.2,
                "threshold": 0.1,
            },
        },
    }


def aisbench_perf_config():
    return {
        "metrics": "perf",
        "benchmark": {
            "benchmark_type": "aisbench",
            "perf": {"baseline": 100, "threshold": 0.1},
            # Metadata-only leftover vbench block must not flip the gate.
            "vbench": {"evaluation": {"baseline": 0.9}},
        },
    }


def evalscope_perf_config():
    return {
        "metrics": "perf",
        "benchmark": {
            "benchmark_type": "evalscope",
            "perf": {"baseline": 100, "threshold": 0.1},
        },
    }


class VBenchTaskGateTests(unittest.TestCase):
    def test_vbench_new_format_is_recognised(self):
        self.assertTrue(is_vbench_task(vbench_config(metrics="acc")))

    def test_vbench_legacy_format_is_recognised(self):
        self.assertTrue(is_vbench_task(vbench_config(metrics="vbench")))

    def test_vbench_gate_ignores_metrics_value(self):
        self.assertTrue(is_vbench_task(vbench_config(metrics="perf")))

    def test_gpqa_is_not_vbench(self):
        self.assertFalse(is_vbench_task(gpqa_config()))

    def test_gedit_is_not_vbench(self):
        self.assertFalse(is_vbench_task(gedit_config()))

    def test_swebench_new_format_is_not_vbench(self):
        self.assertFalse(is_vbench_task(swebench_config()))

    def test_aisbench_perf_is_not_vbench(self):
        self.assertFalse(is_vbench_task(aisbench_perf_config()))

    def test_evalscope_perf_is_not_vbench(self):
        self.assertFalse(is_vbench_task(evalscope_perf_config()))

    def test_missing_vbench_block_is_not_vbench(self):
        config = vbench_config()
        config["benchmark"].pop("vbench")
        self.assertFalse(is_vbench_task(config))

    def test_non_vbench_task_ignores_metadata_vbench_block(self):
        # The perf baseline wins; the metadata-only vbench block is ignored.
        configured = result_processor._configured_baseline(aisbench_perf_config())
        self.assertEqual(configured[0], "perf")
        self.assertEqual(configured[1], 100)

    def test_new_format_vbench_baseline_is_selected(self):
        configured = result_processor._configured_baseline(vbench_config())
        self.assertEqual(configured[0], "vbench")
        self.assertEqual(configured[1], 0.9)

    def test_legacy_format_vbench_baseline_is_selected(self):
        configured = result_processor._configured_baseline(
            vbench_config(metrics="vbench")
        )
        self.assertEqual(configured[0], "vbench")
        self.assertEqual(configured[1], 0.9)


if __name__ == "__main__":
    unittest.main()