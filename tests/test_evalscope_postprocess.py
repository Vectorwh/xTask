import json
import tempfile
import unittest
from pathlib import Path

from xtask.benchmarks.evalscope.evalscope_postprocess import EvalscopeResultParser


class EvalscopeResultParserTests(unittest.TestCase):
    def test_millisecond_summaries_and_labeled_percentiles_survive_main_merge(self):
        for prefix in ("", "Avg "):
            with self.subTest(summary_prefix=prefix), tempfile.TemporaryDirectory() as root:
                summary = {
                    "Total Requests": 21, "Failed Requests": 1, "Concurrency": 8,
                    "Req Throughput (req/s)": 4.5, "Total Throughput (tok/s)": 300,
                    "Avg Latency (s)": 2.0,
                    f"{prefix}TTFT (ms)": 100, f"{prefix}TPOT (ms)": 20,
                    f"{prefix}ITL (ms)": 30,
                    "Avg Input Tokens": 1024, "Avg Output Tokens": 512,
                    "Output Throughput (tok/s)": 256.5,
                }
                values = {
                    "Latency (s)": 2.2, "TTFT (ms)": 120, "TPOT (ms)": 25,
                    "ITL (ms)": 35, "Input tokens": 1024, "Output tokens": 512,
                    "Output (tok/s)": 250,
                }
                rows = [{"Percentiles": label, **values} for label in
                        ("min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%")]
                rows.append({"Percentiles": "max", **{key: 99999 for key in values}})
                root_path = Path(root)
                files = [root_path / "benchmark_summary.json", root_path / "benchmark_percentile.json"]
                files[0].write_text(json.dumps(summary), encoding="utf-8")
                files[1].write_text(json.dumps(rows), encoding="utf-8")
                result = EvalscopeResultParser.parse_perf_json([str(path) for path in files])
                self.assertEqual(result.summary.total_requests, 21)
                self.assertEqual(result.summary.failed_requests, 1)
                self.assertEqual(result.summary.request_throughput, 4.5)
                self.assertEqual(result.summary.e2el_avg, 2000)
                self.assertEqual(result.summary.ttft_avg, 100)
                self.assertEqual(result.summary.tpot_avg, 20)
                self.assertEqual(result.summary.ttft_p99, 120)
                self.assertEqual(result.parameters["ITL"]["total"].average, 30)
                self.assertEqual(result.parameters["E2EL"]["total"].p90, 2200)

    def test_summary_fields_are_preserved_in_metadata(self):
        summary = {
            "Average latency (s)": 2.0,
            "Average time to first token (s)": 0.1,
            "Average time per output token (s)": 0.02,
            "Average inter-token latency (s)": 0.03,
            "Average input tokens per request": 1024,
            "Average output tokens per request": 512,
            "Output token throughput (tok/s)": 256.5,
            "Number of concurrency": 8,
            "Succeed requests": 20,
            "Failed requests": 1,
            "Request throughput (req/s)": 4.5,
        }
        percentile_values = {
            "Latency (s)": 2.2,
            "TTFT (s)": 0.12,
            "TPOT (s)": 0.025,
            "ITL (s)": 0.035,
            "Input tokens": 1024,
            "Output tokens": 512,
            "Output (tok/s)": 250.0,
        }
        percentiles = [{"Percentiles": f"unused-{index}"} for index in range(8)]
        percentiles[4] = {"Percentiles": "75%", **percentile_values}
        percentiles[6] = {"Percentiles": "90%", **percentile_values}
        percentiles[-1] = {"Percentiles": "99%", **percentile_values}

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            summary_path = root_path / "benchmark_summary.json"
            percentile_path = root_path / "benchmark_percentile.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            percentile_path.write_text(json.dumps(percentiles), encoding="utf-8")

            result = EvalscopeResultParser.parse_perf_json(
                [str(percentile_path), str(summary_path)]
            )

        expected_metadata = dict(summary)
        self.assertEqual(result.metadata, expected_metadata)
        self.assertEqual(result.to_dict()["metadata"], expected_metadata)


if __name__ == "__main__":
    unittest.main()
