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


if __name__ == "__main__":
    unittest.main()
