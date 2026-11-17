import json
import tempfile
import unittest
from pathlib import Path

from xtask.benchmarks.aisbench.aisbench_postprocess import (
    AisbenchResultParser,
    get_aisbench_csv_and_json,
)


class AisbenchOutputDiscoveryTests(unittest.TestCase):
    def test_perf_finds_csv_and_json(self):
        with tempfile.TemporaryDirectory() as root:
            result_dir = Path(root) / "performances" / "vllm-api-general-stream"
            result_dir.mkdir(parents=True)
            csv_path = result_dir / "gsm8k.csv"
            json_path = result_dir / "gsm8k.json"
            csv_path.touch()
            json_path.touch()

            actual_csv, actual_json = get_aisbench_csv_and_json(
                root, "vllm_api_general_stream"
            )

        self.assertEqual(actual_csv, str(csv_path))
        self.assertEqual(actual_json, str(json_path))

    def test_accuracy_finds_csv_without_json(self):
        with tempfile.TemporaryDirectory() as root:
            result_dir = Path(root) / "summary"
            result_dir.mkdir(parents=True)
            csv_path = result_dir / "accuracy.csv"
            csv_path.touch()

            actual_csv, actual_json = get_aisbench_csv_and_json(
                root, "vllm_api_general_stream", output_type="accaurcy"
            )

        self.assertEqual(actual_csv, str(csv_path))
        self.assertEqual(actual_json, "")

    def test_missing_required_perf_file_returns_empty_paths(self):
        with tempfile.TemporaryDirectory() as root:
            result_dir = Path(root) / "performances" / "vllm-api-general-stream"
            result_dir.mkdir(parents=True)
            (result_dir / "gsm8k.csv").touch()

            actual_csv, actual_json = get_aisbench_csv_and_json(
                root, "vllm_api_general_stream"
            )

        self.assertEqual((actual_csv, actual_json), ("", ""))

    def test_perf_falls_back_to_the_only_model_abbr_directory(self):
        with tempfile.TemporaryDirectory() as root:
            stale_expected_dir = (
                Path(root) / "performances" / "vllm-api-stream-chat-multiturn"
            )
            stale_expected_dir.mkdir(parents=True)
            (stale_expected_dir / "stale.csv").touch()
            result_dir = (
                Path(root) / "performances" / "vllm-multiturn-api-chat-stream"
            )
            result_dir.mkdir(parents=True)
            csv_path = result_dir / "sharegpt.csv"
            json_path = result_dir / "sharegpt.json"
            csv_path.touch()
            json_path.touch()
            (result_dir / "sharegpt_details.json").touch()

            actual_csv, actual_json = get_aisbench_csv_and_json(
                root, "vllm_api_stream_chat_multiturn"
            )

        self.assertEqual(actual_csv, str(csv_path))
        self.assertEqual(actual_json, str(json_path))

    def test_perf_does_not_guess_when_multiple_model_directories_exist(self):
        with tempfile.TemporaryDirectory() as root:
            for model_abbr in ("model-a", "model-b"):
                result_dir = Path(root) / "performances" / model_abbr
                result_dir.mkdir(parents=True)
                (result_dir / "dataset.csv").touch()
                (result_dir / "dataset.json").touch()

            actual_csv, actual_json = get_aisbench_csv_and_json(
                root, "unknown_api_type"
            )

        self.assertEqual((actual_csv, actual_json), ("", ""))


class AisbenchAccPostProcessTests(unittest.TestCase):
    def _build_task_config(self, api_type="gedit_image_edit"):
        return {
            "case_name": "demo_acc",
            "benchmark": {
                "benchmark_type": "aisbench",
                "acc": {"api_type": api_type},
            },
        }

    def test_multi_metric_rows_return_composite_average(self):
        from xtask.benchmarks.aisbench.aisbench_postprocess import (
            aisbench_acc_post_process,
        )

        with tempfile.TemporaryDirectory() as root:
            stamp_dir = Path(root) / "20261027_110902" / "summary"
            stamp_dir.mkdir(parents=True)
            (stamp_dir / "summary.csv").write_text(
                "dataset,version,metric,mode,gedit-image-edit\n"
                "gedit-SC_judge,6c769c,SC,gen,0.45\n"
                "gedit-PQ_judge,6c769c,PQ,gen,7.15\n",
                encoding="utf-8",
            )

            result, output_dir = aisbench_acc_post_process(
                self._build_task_config(), root, []
            )

        self.assertAlmostEqual(result, 3.8)
        self.assertEqual(output_dir, str(Path(root) / "20261027_110902"))

    def test_single_metric_row_keeps_original_value(self):
        from xtask.benchmarks.aisbench.aisbench_postprocess import (
            aisbench_acc_post_process,
        )

        with tempfile.TemporaryDirectory() as root:
            stamp_dir = Path(root) / "20261027_110902" / "summary"
            stamp_dir.mkdir(parents=True)
            (stamp_dir / "summary.csv").write_text(
                "dataset,version,metric,mode,gedit-image-edit\n"
                "gedit-SC_judge,6c769c,SC,gen,0.45\n",
                encoding="utf-8",
            )

            result, output_dir = aisbench_acc_post_process(
                self._build_task_config(), root, []
            )

        self.assertAlmostEqual(float(result), 0.45)
        self.assertEqual(output_dir, str(Path(root) / "20261027_110902"))


class AisbenchResultParserTests(unittest.TestCase):
    def test_summary_json_total_values_and_units_are_normalized(self):
        csv_content = """Performance Parameters,Stage,Average,Min,Max,Median,P75,P90,P99,N
E2EL,total,100 ms,90 ms,110 ms,100 ms,105 ms,108 ms,110 ms,6
InputTokens,total,1024,1024,1024,1024,1024,1024,1024,6
OutputTokens,total,1024,1024,1024,1024,1024,1024,1024,6
OutputTokenThroughput,total,112.7 token/s,110 token/s,115 token/s,112 token/s,113 token/s,114 token/s,115 token/s,6
"""
        json_content = {
            "Failed Requests": {"total": 0},
            "Concurrency": {"total": 1.0},
            "Max Concurrency": {"total": 2},
            "Request Throughput": {"total": "0.1101 req/s"},
            "Output Token Throughput": {"total": "112.7145 token/s"},
            "Total Token Throughput": {"total": "225.4289 token/s"},
        }

        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "result.csv"
            json_path = Path(root) / "result.json"
            csv_path.write_text(csv_content, encoding="utf-8")
            json_path.write_text(json.dumps(json_content), encoding="utf-8")
            result = AisbenchResultParser.parse_csv_and_json(
                str(csv_path), str(json_path)
            )

        self.assertEqual(result.summary.total_requests, 6)
        self.assertEqual(result.summary.failed_requests, 0)
        self.assertEqual(result.summary.concurrency, 1)
        self.assertEqual(result.summary.max_concurrency, 2)
        self.assertAlmostEqual(result.summary.request_throughput, 0.1101)
        self.assertAlmostEqual(result.summary.output_tps_avg, 112.7)
        self.assertAlmostEqual(result.summary.output_tps_total, 112.7145)
        self.assertAlmostEqual(result.summary.summary_tps_total, 225.4289)

        summary_text = result.print_summary(detailed=True)
        self.assertIn("AISBENCH TEST RESULT SUMMARY", summary_text)
        self.assertIn("Total Requests", summary_text)
        self.assertIn("6", summary_text)

        with tempfile.TemporaryDirectory() as root:
            summary_path = Path(root) / "summary.txt"
            summary_path.write_text(summary_text, encoding="utf-8")
            self.assertGreater(summary_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
