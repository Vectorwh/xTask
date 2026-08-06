import json
import tempfile
import unittest
from pathlib import Path

from xtask.benchmarks.evalscope.evalscope_postprocess import EvalscopeResultParser


class EvalscopeResultParserTests(unittest.TestCase):
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
