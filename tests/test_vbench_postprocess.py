import json
import tempfile
import unittest
from pathlib import Path

from xtask.benchmarks.vbench.vbench_postprocess import vbench_result_post_process


class VBenchPostprocessTests(unittest.TestCase):
    def test_explicit_scores_take_precedence_over_nested_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            date_dir = root / "20260831_120000"
            date_dir.mkdir()
            (date_dir / "generation_metrics.json").write_text(
                json.dumps({"total": 1, "generated": 1, "failed": 0}),
                encoding="utf-8",
            )
            (date_dir / "wan_eval_final_score.json").write_text(
                json.dumps(
                    {
                        "raw": {"total": 1},
                        "total_score": 0.88,
                        "quality_score": "0.80",
                        "i2v_score": 0.96,
                        "temporal_flickering": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            result, output_dir = vbench_result_post_process(
                {"benchmark": {"vbench": {"mode": "generate_and_evaluate"}}},
                str(root),
                [],
            )
        self.assertEqual(result["total_score"], 0.88)
        self.assertEqual(result["quality_score"], 0.80)
        self.assertEqual(result["i2v_score"], 0.96)
        self.assertEqual(result["temporal_flickering"], 0.75)
        self.assertEqual(output_dir, str(date_dir))

    def test_generation_only_allows_missing_final_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            date_dir = root / "20260831_120000"
            date_dir.mkdir()
            (date_dir / "generation_metrics.json").write_text(
                json.dumps({"total": 1, "generated": 1, "failed": 0}),
                encoding="utf-8",
            )
            result, _ = vbench_result_post_process(
                {"benchmark": {"vbench": {"mode": "generate"}}},
                str(root),
                [],
            )
        self.assertEqual(result["generated"], 1)
        self.assertIsNone(result["total_score"])

    def test_evaluation_requires_final_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            date_dir = root / "20260831_120000"
            date_dir.mkdir()
            (date_dir / "generation_metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                vbench_result_post_process(
                    {"benchmark": {"vbench": {"mode": "evaluate"}}},
                    str(root),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
