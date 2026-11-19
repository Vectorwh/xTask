"""Opt-in text/video validation, without models, NPU or external endpoints."""
import copy
import unittest

from xtask.benchmarks.vbench.text_video.config import is_text_video_task, normalize_config, result_issues


def config(**overrides):
    value = {"task": "t2v", "manifest": "/data/prompts.jsonl", "mode": "generate",
             "generation": {"size": "512*288", "num_frames": 81, "fps": 16,
                            "num_inference_steps": 4, "guidance_scale": 1, "seed": 42}}
    value.update(overrides)
    return value


class TextVideoConfigurationTests(unittest.TestCase):
    def test_default_wan_metadata_cannot_enable_new_path(self):
        for kind in ("aisbench", "evalscope", "asr", "image_edit"):
            self.assertFalse(is_text_video_task({"benchmark": {"benchmark_type": kind, "vbench": {"task": "t2v"}}}))
        self.assertFalse(is_text_video_task({"benchmark": {"benchmark_type": "vbench", "vbench": {}}}))
        self.assertFalse(is_text_video_task({"benchmark": {"benchmark_type": "vbench", "vbench": {"task": "i2v"}}}))

    def test_explicit_tasks_only(self):
        for task in ("t2v", "t2av"):
            self.assertTrue(is_text_video_task({"benchmark": {"benchmark_type": "vbench", "vbench": {"task": task}}}))

    def test_requires_explicit_dataset_and_generation(self):
        for key in ("size", "num_frames", "fps", "num_inference_steps", "seed", "guidance_scale"):
            value = config()
            del value["generation"][key]
            with self.assertRaises(ValueError, msg=key):
                normalize_config(value)
        with self.assertRaises(ValueError):
            normalize_config({"task": "t2v"})

    def test_t2av_requires_matching_sample_rate_not_32000_default(self):
        with self.assertRaises(ValueError):
            normalize_config(config(task="t2av"))
        value = config(task="t2av")
        value["generation"]["sampling_rate"] = 44100
        self.assertEqual(normalize_config(value)["media"]["audio"], "required")

    def test_rejects_inactive_and_misleading_parameters(self):
        for key, val in (("seconds", 5), ("batch_size", 2), ("output_type", "binary")):
            value = config()
            value["generation"][key] = val
            with self.assertRaises(ValueError):
                normalize_config(value)
        value = config()
        value["generation"]["num_videos_per_prompt"] = 2
        with self.assertRaises(ValueError):
            normalize_config(value)

    def test_perf_never_executes_quality_or_reuses_outputs(self):
        value = config(metrics="perf", evaluation={"enabled": True})
        with self.assertRaises(ValueError):
            normalize_config(value)
        with self.assertRaises(ValueError):
            normalize_config(config(metrics="perf", mode="evaluate", source_run="/old/run"))
        self.assertFalse(normalize_config(config(metrics="perf"))["evaluation"]["enabled"])

    def test_no_fake_i2v_or_full_16_total(self):
        for evaluation in ({"baseline": .9}, {"dimensions": ["i2v_subject"]}, {"dimensions": ["object_class"]}):
            with self.assertRaises(ValueError):
                normalize_config(config(evaluation=evaluation))

    def test_invalid_media_and_nonfinite_values(self):
        for value in (float("nan"), float("inf"), -1, True):
            raw = config()
            raw["generation"]["fps"] = value
            with self.assertRaises(ValueError):
                normalize_config(raw)
        with self.assertRaises(ValueError):
            normalize_config(config(media={"audio": "whatever"}))

    def test_zero_error_style_gates_and_missing_values(self):
        settings = normalize_config(config(gates=[{"metric": "performance.e2el_avg_ms", "compare": "lte", "baseline": 1000}]))
        result = {"complete": True, "failed": 0, "expected_requests": 2, "completed_requests": 2,
                  "performance": {"e2el_avg_ms": 900}}
        self.assertEqual(result_issues(result, settings), [])
        result["performance"]["e2el_avg_ms"] = 1100
        self.assertEqual(result_issues(result, settings)[0]["category"], "baseline")
        result["performance"]["e2el_avg_ms"] = float("nan")
        self.assertEqual(result_issues(result, settings)[0]["category"], "processing")

    def test_incomplete_always_fails_without_baseline(self):
        result = {"complete": True, "failed": 0, "expected_requests": 2, "completed_requests": 1}
        self.assertTrue(result_issues(result, normalize_config(config())))

    def test_config_does_not_mutate_original_generation(self):
        raw = config()
        original = copy.deepcopy(raw)
        normalize_config(raw)
        self.assertEqual(raw, original)
