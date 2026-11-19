"""Calibration leakage/readiness fences; no pretrained downloads or NPU."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("semantic_calibration", Path(__file__).parents[1] / "tools/calibrate_video_semantic_scorers.py")
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class CalibrationTests(unittest.TestCase):
    def test_confusion_and_missing_predictions(self):
        rows = [{"present": True}, {"present": False}, {"present": True}, {"present": False}]
        result = calibration.confusion(rows, [True, False, False, True])
        self.assertEqual([result[k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 1, 1])
        self.assertEqual(result["balanced_accuracy"], .5)
        with self.assertRaises(ValueError):
            calibration.confusion(rows, [])

    def test_validation_never_selects_threshold(self):
        training = [{"split": "calibration", "label": "dog", "present": positive,
                     "scores_by_text_threshold": {"0.2": .31 if positive else .26, "0.3": .31 if positive else .26}}
                    for positive in [True] * 4 + [False] * 4]
        validation = [{**row, "split": "validation"} for row in training]
        a = calibration.select_threshold(training + validation, "vision")
        adversarial = [{**row, "present": not row["present"]} for row in validation]
        b = calibration.select_threshold(training + adversarial, "vision")
        self.assertEqual(a["thresholds"], b["thresholds"])
        self.assertTrue(a["operational_readiness"])
        self.assertFalse(b["operational_readiness"])

    def test_sparse_class_cannot_be_certified_by_micro_average(self):
        rows = [{"split": split, "label": label, "present": positive, "similarity": .4 if positive else .05,
                 "margin": .1 if positive else -.1}
                for split in ("calibration", "validation")
                for label, n in (("rain", 4), ("music", 1))
                for positive in [True] * n + [False] * n]
        result = calibration.select_threshold(rows, "audio")
        self.assertFalse(result["operational_readiness"])
        self.assertEqual(result["operationally_supported_labels"], ["rain"])

    def test_shared_source_between_splits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "x"
            path.write_text("data")
            digest = calibration.sha256(path)
            rows = [{"id": split, "split": split, "source": "same recording", "path": "x", "sha256": digest}
                    for split in ("calibration", "validation")]
            with self.assertRaisesRegex(ValueError, "leakage"):
                calibration.validate_samples(rows, root, "source")

    def test_same_split_recording_duplicates_cannot_inflate_independent_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x").write_text("data")
            rows = [{"id": str(i), "split": "calibration", "source": "same recording", "path": "x",
                     "sha256": calibration.sha256(root / "x")} for i in (1, 2)]
            with self.assertRaisesRegex(ValueError, "Repeated source"):
                calibration.validate_samples(rows, root, "source", require_unique_groups=True)

    def test_checksum_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x").write_text("data")
            rows = [{"id": "a", "split": "calibration", "source": "a", "path": "x", "sha256": "0" * 64}]
            with self.assertRaisesRegex(ValueError, "checksum"):
                calibration.validate_samples(rows, root, "source")


if __name__ == "__main__":
    unittest.main()
