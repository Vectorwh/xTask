"""image_edit_postprocess 单元测试。

复用 evalscope 后处理的 benchmark_summary/benchmark_percentile 布局，
验证附加的 output_unit 字段与 TTFT≈E2EL 告警分支。
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtask.benchmarks.evalscope.image_edit_postprocess import (
    image_edit_perf_post_process,
)


def _write_result(root, summary, percentiles):
    run_dir = root / "20260101_000000" / "image_edit"
    run_dir.mkdir(parents=True)
    (run_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "benchmark_percentile.json").write_text(
        json.dumps(percentiles), encoding="utf-8"
    )
    return run_dir


def _summary_and_percentiles(latency_s=2.0, ttft_ms=2000.0):
    summary = {
        "Avg Latency (s)": latency_s,
        "Average latency (s)": latency_s,
        "Avg TTFT (ms)": ttft_ms,
        "Average time to first token (s)": ttft_ms / 1000,
        "Avg TPOT (ms)": 0.0,
        "Average time per output token (s)": 0.0,
        "Avg ITL (ms)": 0.0,
        "Average inter-token latency (s)": 0.0,
        "Avg Input Tokens": 1,
        "Average input tokens per request": 1,
        "Avg Output Tokens": 1.76,
        "Average output tokens per request": 1.76,
        "Output Throughput (tok/s)": 0.88,
        "Output token throughput (tok/s)": 0.88,
        "Number of concurrency": 1,
        "Succeed requests": 10,
        "Failed requests": 0,
        "Request throughput (req/s)": 0.5,
    }
    percentile_values = {
        "Latency (s)": latency_s,
        "TTFT (ms)": ttft_ms,
        "TTFT (s)": ttft_ms / 1000,
        "TPOT (ms)": 0.0,
        "TPOT (s)": 0.0,
        "ITL (ms)": 0.0,
        "ITL (s)": 0.0,
        "Input tokens": 1,
        "Output tokens": 1.76,
        "Output (tok/s)": 0.88,
    }
    percentiles = [{"Percentiles": f"unused-{index}"} for index in range(8)]
    percentiles[4] = {"Percentiles": "75%", **percentile_values}
    percentiles[6] = {"Percentiles": "90%", **percentile_values}
    percentiles[-1] = {"Percentiles": "99%", **percentile_values}
    return summary, percentiles


class ImageEditPostProcessTests(unittest.TestCase):
    def _task_config(self):
        return {"case_name": "Case_Gedit", "benchmark": {"benchmark_type": "image_edit"}}

    def test_output_unit_annotation_added(self):
        summary, percentiles = _summary_and_percentiles()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_result(root, summary, percentiles)
            result, output_dir = image_edit_perf_post_process(
                self._task_config(), str(root), init_entries=[]
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.metadata["output_unit"], "MPx/s")
            # 语义重定义字段
            self.assertAlmostEqual(result.summary.output_tps_avg, 0.88, places=4)
            # 落盘文件
            self.assertTrue((root / "case_gedit.json").exists())
            self.assertTrue((root / "case_gedit_summary.txt").exists())

    def test_ttft_close_to_e2el_is_info(self):
        summary, percentiles = _summary_and_percentiles(latency_s=2.0, ttft_ms=1990.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_result(root, summary, percentiles)
            with mock.patch.object(logging.Logger, "warning") as warning:
                image_edit_perf_post_process(self._task_config(), str(root), init_entries=[])
        warning.assert_not_called()

    def test_ttft_divergence_from_e2el_warns(self):
        summary, percentiles = _summary_and_percentiles(latency_s=2.0, ttft_ms=1000.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_result(root, summary, percentiles)
            with mock.patch.object(logging.Logger, "warning") as warning:
                image_edit_perf_post_process(self._task_config(), str(root), init_entries=[])
        warning.assert_called_once()

    def test_missing_result_returns_none(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result, output_dir = image_edit_perf_post_process(
                self._task_config(), temporary_directory, init_entries=[]
            )
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
