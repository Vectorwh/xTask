import unittest

from xtask.core import result_processor


class VBenchResultVerificationTests(unittest.TestCase):
    @staticmethod
    def _config(baseline=None, threshold=0.1):
        return {
            "metrics": "vbench",
            "benchmark": {
                "benchmark_type": "vbench",
                "vbench": {
                    "mode": "generate_and_evaluate",
                    "evaluation": {"baseline": baseline, "threshold": threshold},
                },
            },
        }

    @staticmethod
    def _result(score=0.88, failed=0):
        return {
            "benchmark_success": True,
            "benchmark_vbench_result": {
                "total_score": score,
                "failed": failed,
            },
        }

    def test_complete_vbench_result_passes_without_baseline(self):
        self.assertTrue(result_processor.verify_result(self._result(), self._config()))

    def test_missing_score_fails_for_evaluation(self):
        result = self._result()
        result["task_config"] = self._config()
        result["benchmark_vbench_result"]["total_score"] = None
        self.assertFalse(result_processor.verify_result(result, self._config()))

    def test_failed_video_fails_even_without_baseline(self):
        self.assertFalse(
            result_processor.verify_result(self._result(failed=1), self._config())
        )

    def test_vbench_baseline_uses_total_score(self):
        config = self._config(baseline=0.90, threshold=0.05)
        self.assertTrue(result_processor.verify_result(self._result(0.88), config))
        self.assertFalse(result_processor.verify_result(self._result(0.80), config))

    def test_non_vbench_task_ignores_unrelated_vbench_block(self):
        config = {
            "metrics": "perf",
            "benchmark": {
                "benchmark_type": "aisbench",
                "vbench": {"evaluation": {"baseline": 0.9, "threshold": 0.01}},
                "perf": {"baseline": None, "threshold": 0.1},
            },
        }
        self.assertIsNone(result_processor._configured_baseline(config))

    def test_workbook_payload_keeps_summary_but_excludes_per_video_details(self):
        result = self._result()
        result["task_config"] = self._config()
        result["benchmark_vbench_result"].update(
            {
                "generation": {"latency_seconds": [0.1] * 1000},
                "eval_results": [{"video": index} for index in range(1000)],
                "final_score_file": "/outputs/case/final_score.json",
            }
        )
        payload = result_processor._result_payload(result)
        self.assertEqual(payload["total_score"], 0.88)
        self.assertIn("final_score_file", payload)
        self.assertNotIn("generation", payload)
        self.assertNotIn("eval_results", payload)

    def test_workbook_payload_uses_explicit_task_config(self):
        result = self._result()
        result.pop("task_config", None)
        payload = result_processor._result_payload(result, self._config())
        self.assertEqual(payload["total_score"], 0.88)


if __name__ == "__main__":
    unittest.main()
