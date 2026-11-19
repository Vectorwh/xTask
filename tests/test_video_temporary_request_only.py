# Copyright 2026 The xTask Authors.
"""Temporary CFG diagnostic uses existing opt-in settings; never fake judge pass."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from xtask.benchmarks.vbench.text_video import run as runner
from xtask.benchmarks.vbench.text_video.config import normalize_config, result_issues
from xtask.core import result_processor

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "tests/e2e/xllm/examples/yaml_authoring/55_single_joyai_video_negative_calibrated.yaml"
TEMPORARY = ROOT / "tests/e2e/xllm/diagnostics/joyai_video/joyai_video_cfg_pair_request_only_temporary.yaml"


class TemporaryVideoDiagnosticTests(unittest.TestCase):
    def test_only_new_case_acceptance_scope_changes(self):
        strict = yaml.safe_load(STRICT.read_text())
        temporary = yaml.safe_load(TEMPORARY.read_text())
        for key in strict.keys() - {"case_name", "benchmark"}:
            self.assertEqual(strict[key], temporary[key], key)
        old = strict["benchmark"]["vbench"]
        new = temporary["benchmark"]["vbench"]
        self.assertEqual(temporary["benchmark"]["benchmark_type"], strict["benchmark"]["benchmark_type"])
        for key in old.keys() - {"mode", "checks"}:
            self.assertEqual(old[key], new[key], key)
        self.assertEqual(new["mode"], "generate")
        self.assertEqual(new["checks"], {})
        self.assertEqual(new["provenance"]["acceptance_scope"], "request_and_media_only")
        self.assertIn("negative_control", old["checks"])

    def _settings(self):
        task = yaml.safe_load(TEMPORARY.read_text())
        raw = dict(task["benchmark"]["vbench"], metrics=task["metrics"], model=task["served_model_name"], endpoint="http://local")
        raw["manifest"] = str(ROOT / raw["manifest"])
        return task, normalize_config(raw)

    def _request(self, prepared, settings, directory, index):
        video = directory / "videos" / prepared["filename"]
        video.write_bytes(b"mocked video, not real NPU output")
        return {k: v for k, v in prepared.items() if k != "payload"} | {
            "request_index": index, "success": True, "video_path": str(video.relative_to(directory)),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "started": float(index + 1), "finished": float(index + 2), "e2el_ms": 1000.,
        }

    def test_twenty_four_requests_pass_only_request_media_scope(self):
        task, settings = self._settings()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner, "_request", side_effect=self._request) as request, mock.patch.object(
            runner, "validate_media", return_value={"fully_decoded": True}
        ), mock.patch("xtask.benchmarks.vbench.text_video.checks.run_checks") as judge:
            payload, archive = runner.run(settings, Path(temporary))
            self.assertEqual(request.call_count, 24)
            self.assertEqual(payload["expected_requests"], 24)
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["checks_status"]["negative_control"], "not_scored")
            self.assertNotIn("negative_control", payload)
            self.assertEqual(payload["video_quality"], {})
            self.assertEqual(payload["issues"], [])
            judge.assert_not_called()
            self.assertFalse((archive / "negative_control.json").exists())
            result = {"benchmark_vbench_result": payload, "benchmark_success": True}
            self.assertTrue(result_processor.verify_result(result, task))
            workbook = result_processor._result_payload(result, task)["video"]
            self.assertEqual(workbook["checks_status"]["negative_control"], "not_scored")

    def test_media_failure_still_fails_and_never_calls_judge(self):
        task, settings = self._settings()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner, "_request", side_effect=self._request), mock.patch.object(
            runner, "validate_media", side_effect=ValueError("invalid media")
        ), mock.patch("xtask.benchmarks.vbench.text_video.checks.run_checks") as judge:
            payload, _ = runner.run(settings, Path(temporary))
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["failed"], 24)
            self.assertFalse(result_processor.verify_result({"benchmark_vbench_result": payload}, task))
            judge.assert_not_called()

    def test_strict_fifty_five_still_requires_actual_semantic_assessment(self):
        strict = yaml.safe_load(STRICT.read_text())
        settings = normalize_config(dict(strict["benchmark"]["vbench"], metrics=strict["metrics"]))
        payload = {"task": "t2av", "scenario": "cfg_pair", "complete": True,
                   "expected_requests": 24, "completed_requests": 24, "failed": 0,
                   "checks_status": {"negative_control": "not_scored"}}
        self.assertTrue(result_issues(payload, settings))
        payload["negative_control"] = {
            "complete": True, "profile": "xtask_negative_v1", "assessment": {
                "status": "inconclusive", "passed": False, "provisional": True,
                "policy_profile": "xtask_negative_v1",
                "violations": [{"category": "processing", "reason": "insufficient informative controls"}],
            },
        }
        self.assertFalse(result_processor.verify_result({"benchmark_vbench_result": payload}, strict))


if __name__ == "__main__":
    unittest.main()
