"""Semantic negative-control evidence/policy tests: no NPU or scorer downloads."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.benchmarks.vbench.text_video.negative import (
    DEFAULT_POLICY, normalize_spec, presence_summary, score_pairs, summarize, validate_constraints,
)
from xtask.benchmarks.vbench.text_video.config import normalize_config, result_issues
from xtask.benchmarks.vbench.text_video.checks import run_checks
from xtask.benchmarks.vbench.text_video.run import prepare_requests
from xtask.benchmarks.vbench.text_video.semantic_models import window_starts
from xtask.core.result_processor import _classify_failure, verify_result
from test_text_video_config import config


def spec(**kwargs):
    raw = {"backend": "benchmark_semantics", "vision": {"model_path": "/offline/grounding"}}
    raw.update(kwargs)
    return normalize_spec(raw)


def row(index, ref=.5, target=0, required_ref=1, required_target=1, modality="video", repeat=1):
    return {"condition_key": str(index), "sample_id": str(index), "repeat": repeat,
            modality: {"reference": {"required_rate": required_ref, "forbidden_rate": ref},
                       "target": {"required_rate": required_target, "forbidden_rate": target}}}


def pair(index=0, seed=42, annotations=None):
    entries = {}
    for arm in ("reference", "target"):
        entries[arm] = {"conditioning": "text", "parameters": {"seed": seed, "guidance_scale": 4},
                        "prompt": "A person on a street", "video_path": arm + ".mp4",
                        "checks": annotations or {"required_video": ["person"], "forbidden_video": ["car"]},
                        "request_input": {"prompt": "A person on a street", "negative_prompt": "artifacts" if arm == "reference" else "car"}}
    return {"sample_id": str(index), "repeat": 1, **entries}


class SpecificationTests(unittest.TestCase):
    def test_default_thresholds_are_named_provisional_not_benchmark_standard(self):
        settings = spec()
        self.assertEqual(settings["profile"], "xtask_negative_v1")
        self.assertEqual(settings["policy"], DEFAULT_POLICY)

    def test_no_scorer_or_bad_backend_rejected(self):
        for raw in ({"backend": "benchmark_semantics"}, {"backend": "magic"}, {"backend": "benchmark_semantics", "vision": {"model_path": ""}}):
            with self.assertRaises(ValueError):
                normalize_spec(raw)

    def test_thresholds_and_sample_counts_validated(self):
        for policy in ({"target_forbidden_max": -1}, {"target_forbidden_max": float("nan")}, {"min_conditions": True},
                       {"min_conditions": 1, "min_informative_conditions": 2}, {"reference_presence_min": 0}):
            with self.assertRaises(ValueError):
                spec(policy=policy)

    def test_visual_requires_positive_and_negative_annotations(self):
        for annotations in ({}, {"forbidden_video": ["car"]}, {"required_video": ["person"], "forbidden_video": ["person"]},
                            {"required": "person", "forbidden": "car"}):
            with self.assertRaises(ValueError):
                validate_constraints(annotations, spec())

    def test_no_negative_sentence_as_detector_query(self):
        with self.assertRaises(ValueError):
            validate_constraints({"required_video": ["person"], "forbidden_video": ["no cars."]}, spec())

    def test_audio_requires_explicit_competing_descriptions(self):
        settings = spec(audio={"model_path": "/offline/clap"})
        event = {"id": "music", "text": "Music is playing", "alternatives": ["Rain is falling", "Silence"]}
        validate_constraints({"required_audio": [{"id": "rain", "text": "Rain is falling", "alternatives": ["Music is playing"]}],
                              "forbidden_audio": [event]}, settings)
        event["alternatives"] = []
        with self.assertRaises(ValueError):
            validate_constraints({"required_audio": [event], "forbidden_audio": [event]}, settings)

    def test_audio_constraints_without_scorer_fail(self):
        with self.assertRaises(ValueError):
            validate_constraints({"required_audio": ["rain"], "forbidden_audio": ["music"]}, spec())

    def test_audio_config_never_hides_missing_track(self):
        raw = config(scenario="cfg_pair", checks={"negative_control": spec(audio={"model_path": "/offline/clap"})})
        raw["generation"]["guidance_scale"] = 4
        with self.assertRaises(ValueError):
            normalize_config(raw)

    def test_model_defaults_do_not_enable_scorers_for_existing_wan(self):
        from xtask.benchmarks.vbench.text_video.config import is_text_video_task
        self.assertFalse(is_text_video_task({"benchmark": {"benchmark_type": "vbench", "vbench": {"checks": {"negative_control": spec()}}}}))

    def test_spec_does_not_mutate_user_input(self):
        raw = {"backend": "benchmark_semantics", "vision": {"model_path": "/offline/model"}}
        before = copy.deepcopy(raw)
        normalize_spec(raw)
        self.assertEqual(raw, before)


class PresenceTests(unittest.TestCase):
    def test_required_conjunction_forbidden_union(self):
        r = presence_summary({"person": [True, True, False], "dog": [True, False, True]},
                             {"car": [False, True, False], "bus": [False, False, True]})
        self.assertAlmostEqual(r["required_rate"], 1 / 3)
        self.assertAlmostEqual(r["forbidden_rate"], 2 / 3)

    def test_missing_frames_or_nonboolean_cannot_be_absent(self):
        for required, forbidden in (({}, {"car": [False]}), ({"person": []}, {"car": []}),
                                    ({"person": [True, True]}, {"car": [False]}), ({"person": [1]}, {"car": [False]})):
            with self.assertRaises(ValueError):
                presence_summary(required, forbidden)

    def test_audio_windows_cover_end_without_random_crop(self):
        self.assertEqual(window_starts(20, 8, 4), [0, 4, 8, 12])
        self.assertEqual(window_starts(21, 8, 4), [0, 4, 8, 12, 13])
        self.assertEqual(window_starts(3, 8, 4), [0])
        with self.assertRaises(ValueError):
            window_starts(20, 4, 8)


class AssessmentTests(unittest.TestCase):
    def test_meaningful_improvement_and_positive_retention_pass(self):
        metrics, assessment = summarize([row(i) for i in range(6)], spec())
        self.assertEqual(assessment["status"], "pass")
        self.assertTrue(assessment["provisional"])
        self.assertEqual(metrics["video_relative_reduction"], 1)
        self.assertEqual(metrics["video_improvement_rate"], 1)

    def test_already_absent_reference_is_inconclusive_not_success(self):
        metrics, assessment = summarize([row(i, ref=0) for i in range(6)], spec())
        self.assertEqual(assessment["status"], "inconclusive")
        self.assertNotIn("video_relative_reduction", metrics)
        self.assertEqual(metrics["passed"], 0)

    def test_no_reduction_fails(self):
        _, assessment = summarize([row(i, target=.5) for i in range(6)], spec())
        self.assertEqual(assessment["status"], "fail")
        self.assertTrue(any("relative_reduction" in v["reason"] for v in assessment["violations"]))

    def test_positive_target_collapse_fails_even_exclusion_success(self):
        _, assessment = summarize([row(i, required_target=0) for i in range(6)], spec())
        self.assertEqual(assessment["status"], "fail")
        self.assertTrue(any("positive_drop" in v["reason"] for v in assessment["violations"]))

    def test_failed_positive_reference_cannot_prove_control(self):
        _, assessment = summarize([row(i, required_ref=0) for i in range(6)], spec())
        self.assertEqual(assessment["status"], "fail")

    def test_repeats_same_condition_do_not_inflate_independent_counts(self):
        metrics, assessment = summarize([row(0, repeat=i) for i in range(12)], spec())
        self.assertEqual(metrics["conditions"], 1)
        self.assertEqual(assessment["status"], "inconclusive")

    def test_partial_informative_sample_coverage_fails(self):
        _, assessment = summarize([row(i, ref=.5 if i < 2 else 0) for i in range(6)], spec())
        self.assertEqual(assessment["status"], "inconclusive")

    def test_enabled_audio_without_audio_results_cannot_pass(self):
        _, assessment = summarize([row(i) for i in range(6)], spec(audio={"model_path": "/offline/clap"}))
        self.assertEqual(assessment["status"], "inconclusive")

    def test_modalities_checked_separately(self):
        rows = [row(i) | {"audio": row(i, modality="audio", required_target=0)["audio"]} for i in range(6)]
        metrics, assessment = summarize(rows, spec(audio={"model_path": "/offline/clap"}))
        self.assertEqual(assessment["status"], "fail")
        self.assertEqual(metrics["video_positive_drop"], 0)
        self.assertEqual(metrics["audio_positive_drop"], 1)


class PairIntegrationTests(unittest.TestCase):
    def test_same_seed_params_and_text_required(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = [pair()]
            pairs[0]["target"]["parameters"]["seed"] = 43
            with self.assertRaises(ValueError):
                score_pairs(Path(directory), pairs, spec(), {"media": {}}, scorers={})

    def test_negative_prompt_must_actually_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = [pair()]
            pairs[0]["target"]["request_input"] = pairs[0]["reference"]["request_input"]
            with self.assertRaises(ValueError):
                score_pairs(Path(directory), pairs, spec(), {"media": {}}, scorers={})

    def test_fake_observer_fixture_is_labelled_and_policy_not_fake_pass(self):
        class Observer:
            provenance = {"fixture": True}
            def score(self, path, required, forbidden, media):
                return {"required": {x: [True] * 10 for x in required},
                        "forbidden": {x: [path.name == "reference.mp4"] * 10 for x in forbidden}}
        with tempfile.TemporaryDirectory() as directory:
            pairs = [pair(i, seed=42+i) for i in range(6)]
            result = score_pairs(Path(directory), pairs, spec(), {"media": {}}, scorers={"vision": Observer()})
            self.assertEqual(result["assessment"]["status"], "pass")
            self.assertEqual(result["metrics"]["conditions"], 6)
            self.assertTrue(result["provenance"]["vision"]["fixture"])
            self.assertTrue((Path(directory) / "negative_control.json").is_file())

    def test_assessment_violations_reach_xtask_verification(self):
        raw = config(scenario="cfg_pair", checks={"negative_control": spec()})
        raw["generation"]["guidance_scale"] = 4
        settings = normalize_config(raw)
        metrics, assessment = summarize([row(i, target=.5) for i in range(6)], spec())
        result = {"complete": True, "failed": 0, "expected_requests": 12, "completed_requests": 12,
                  "negative_control": {"complete": True, **metrics, "assessment": assessment}}
        self.assertTrue(any(i["category"] == "baseline" for i in result_issues(result, settings)))
        task = {"metrics": "acc", "benchmark": {"benchmark_type": "vbench", "vbench": raw}}
        scheduler_result = {"benchmark_success": False, "success": False, "benchmark_vbench_result": result}
        self.assertFalse(verify_result(scheduler_result, task))
        self.assertEqual(_classify_failure(scheduler_result, task)[0], "benchmark结果与基线比对失败")

    def test_unknown_semantic_gate_rejected(self):
        raw = config(scenario="cfg_pair", checks={"negative_control": spec()},
                     gates=[{"metric": "negative_control.magic", "compare": "gte", "baseline": .5}])
        raw["generation"]["guidance_scale"] = 4
        with self.assertRaises(ValueError):
            normalize_config(raw)

    def test_manifest_constraints_checked_before_model_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text(json.dumps([{"id": "x", "prompt": "person", "negative_prompt": "car", "neutral_negative_prompt": "artifacts"}]))
            raw = config(scenario="cfg_pair", manifest=str(path), model="Joy", endpoint="http://local", checks={"negative_control": spec()})
            raw["generation"]["guidance_scale"] = 4
            with self.assertRaises(ValueError):
                prepare_requests(normalize_config(raw), root)


class ScorerInvocationTests(unittest.TestCase):
    def test_semantic_result_is_executed_in_separate_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = normalize_config(dict(config(scenario="cfg_pair", checks={"negative_control": spec()}),
                                             generation=dict(config()["generation"], guidance_scale=4)))
            entries = []
            for arm in ("reference", "target"):
                entries.append(dict(pair()[arm], sample_id="x", repeat=1, arm=arm))
            metrics, assessment = summarize([row(i) for i in range(6)], spec())
            result = {"complete": True, "profile": "xtask_negative_v1", "metrics": metrics,
                      "assessment": assessment, "pairs": [{"sample_id": "x", "repeat": 1}]}
            def process(command, **kwargs):
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")
                self.assertEqual(kwargs["timeout"], 3600)
                self.assertTrue(command[1].endswith("negative.py"))
                (root / "negative_control.json").write_text(json.dumps(result))
                return SimpleNamespace(returncode=0, stdout="fixture", stderr="")
            with mock.patch("subprocess.run", side_effect=process):
                scored = run_checks(root, entries, settings)
            self.assertEqual(scored["negative_control"]["assessment"]["status"], "pass")
            self.assertTrue((root / "negative_control_command.json").is_file())

    def test_missing_weights_fail_before_torch_import_or_network(self):
        from xtask.benchmarks.vbench.text_video.semantic_models import preflight_checkpoints
        with mock.patch.dict("sys.modules", {"torch": None, "transformers": None}):
            with self.assertRaises(FileNotFoundError):
                preflight_checkpoints(spec())

    def test_wrong_scorer_checkpoint_type_fails(self):
        from xtask.benchmarks.vbench.text_video.semantic_models import preflight_checkpoints
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
            (path / "weights.safetensors").write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                preflight_checkpoints(spec(vision={"model_path": str(path)}))

    def test_timeout_does_not_produce_success(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [dict(pair()[arm], sample_id="x", repeat=1, arm=arm) for arm in ("reference", "target")]
            settings = {"checks": {"negative_control": spec()}, "media": {}}
            with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["fixture"], 1)):
                with self.assertRaises(RuntimeError):
                    run_checks(root, entries, settings)
            self.assertTrue((root / "negative_control_stderr.log").is_file())

    def test_missing_result_and_crashed_scorer_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [dict(pair()[arm], sample_id="x", repeat=1, arm=arm) for arm in ("reference", "target")]
            settings = {"checks": {"negative_control": spec()}, "media": {}}
            with mock.patch("subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="fixture fail")):
                with self.assertRaises(RuntimeError):
                    run_checks(root, entries, settings)
            self.assertEqual((root / "negative_control_stderr.log").read_text(), "fixture fail")

    def test_invalid_counts_or_assessment_do_not_pass(self):
        settings = normalize_config(dict(config(scenario="cfg_pair", checks={"negative_control": spec()}),
                                             generation=dict(config()["generation"], guidance_scale=4)))
        payload = {"complete": True, "failed": 0, "expected_requests": 12, "completed_requests": 12,
                   "negative_control": {"complete": True, "assessment": {"status": "fail", "violations": []}}}
        self.assertTrue(result_issues(payload, settings))


class EvidenceAndModelAPITests(unittest.TestCase):
    def test_duplicate_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                score_pairs(Path(directory), [pair(), pair()], spec(), {"media": {}}, scorers={})

    def test_reference_dilution_cannot_mask_residual_forbidden_content(self):
        data = [row(i, ref=.5, target=.15) if i < 3 else row(i, ref=0, target=0) for i in range(6)]
        metrics, assessment = summarize(data, spec())
        self.assertLess(metrics["video_target_forbidden_rate"], .1)
        self.assertEqual(assessment["status"], "fail")
        self.assertTrue(any("informative_target_forbidden_rate" in v["reason"] for v in assessment["violations"]))

    def test_repeat_sample_ids_do_not_create_more_evidence(self):
        class Observer:
            provenance = {"fixture": True}
            def score(self, path, required, forbidden, media):
                return {"required": {x: [True] for x in required}, "forbidden": {x: [path.name == "reference.mp4"] for x in forbidden}}
        with tempfile.TemporaryDirectory() as directory:
            result = score_pairs(Path(directory), [pair(i) for i in range(12)], spec(), {"media": {}}, scorers={"vision": Observer()})
            self.assertEqual(result["metrics"]["conditions"], 1)
            self.assertEqual(result["assessment"]["status"], "inconclusive")

    def test_visual_adapter_queries_one_label_at_a_time(self):
        from xtask.benchmarks.vbench.text_video.semantic_models import GroundingDinoScorer
        from contextlib import nullcontext
        scorer = GroundingDinoScorer.__new__(GroundingDinoScorer)
        scorer.spec = {"frames": 2, "max_side": 768, "box_threshold": .4, "text_threshold": .3}
        scorer.device = "cpu"
        scorer.torch = SimpleNamespace(inference_mode=nullcontext)
        scorer.model = mock.Mock(return_value=object())
        called = []
        class Inputs(dict):
            def to(self, device): return self
        class Tensor:
            def __init__(self, value): self.value=value
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.value
        class Processor:
            def __call__(self, **kwargs):
                called.append(kwargs["text"])
                return Inputs(input_ids=object())
            def post_process_grounded_object_detection(self, *args, **kwargs):
                return [{"scores": Tensor([.9] if called[-1] == "person." else []), "boxes": Tensor([]), "text_labels": []}]
        scorer.processor = Processor()
        image = SimpleNamespace(size=(64, 64))
        with mock.patch("xtask.benchmarks.vbench.text_video.semantic_models.sampled_frames", return_value=([image,image],[1,3])):
            evidence = scorer.score(Path("clip.mp4"), ["person"], ["car"], {})
        self.assertEqual(called, ["person.", "person.", "car.", "car."])
        self.assertEqual(evidence["required"]["person"], [True,True])
        self.assertEqual(evidence["forbidden"]["car"], [False,False])


class FailClosedAssessmentTests(unittest.TestCase):
    def test_inconsistent_assessment_passed_field_fails(self):
        settings = normalize_config(dict(config(scenario="cfg_pair", checks={"negative_control": spec()}), generation=dict(config()["generation"], guidance_scale=4)))
        metrics, assessment = summarize([row(i) for i in range(6)], spec())
        assessment["passed"] = False
        payload = {"complete": True, "failed": 0, "expected_requests": 12, "completed_requests": 12,
                   "negative_control": {"complete": True, **metrics, "assessment": assessment}}
        self.assertTrue(result_issues(payload, settings))

    def test_wrong_policy_profile_is_not_a_pass(self):
        settings = normalize_config(dict(config(scenario="cfg_pair", checks={"negative_control": spec()}), generation=dict(config()["generation"], guidance_scale=4)))
        metrics, assessment = summarize([row(i) for i in range(6)], spec())
        assessment["policy_profile"] = "wrong"
        payload = {"complete": True, "failed": 0, "expected_requests": 12, "completed_requests": 12,
                   "negative_control": {"complete": True, **metrics, "assessment": assessment}}
        self.assertTrue(result_issues(payload, settings))

    def test_preflight_clears_parent_npu_pythonpath(self):
        from xtask.benchmarks.vbench.text_video.semantic_models import preflight_environment
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"PYTHONPATH": "/npu/runtime"}), mock.patch(
            "subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="ready", stderr="")
        ) as command:
            preflight_environment(spec(), Path(directory))
            self.assertEqual(command.call_args.kwargs["env"]["PYTHONPATH"], "")
            self.assertEqual(command.call_args.kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")

    def test_preflight_bad_interpreter_returns_clear_error(self):
        from xtask.benchmarks.vbench.text_video.semantic_models import preflight_environment
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="wrong transformers")):
                with self.assertRaises(RuntimeError):
                    preflight_environment(spec(), Path(directory))
            self.assertIn("wrong transformers", (Path(directory)/"negative_environment_stderr.log").read_text())


class RunnerCLIRegressionTests(unittest.TestCase):
    def test_direct_runner_script_reports_missing_checkpoint_before_http(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"id": "x", "prompt": "A person on a street", "negative_prompt": "artifacts, cars", "neutral_negative_prompt": "artifacts", "checks": {"required_video": ["person"], "forbidden_video": ["car"]}}]))
            settings = config(scenario="cfg_pair", manifest=str(manifest), model="fixture", endpoint="http://127.0.0.1:1",
                              checks={"negative_control": spec()})
            settings["generation"]["guidance_scale"] = 4
            cfg = root / "config.json"
            cfg.write_text(json.dumps(settings))
            from xtask.benchmarks.vbench.text_video import run
            result = subprocess.run([sys.executable, str(Path(run.__file__).resolve()), "--config", str(cfg), "--output-dir", str(root / "output")], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            output = json.loads(next((root / "output").glob("*/text_video_result.json")).read_text())
            self.assertIn("Prepare local vision scorer checkpoint first", output["execution_error"])
            self.assertNotIn("relative import", output["execution_error"])
            self.assertEqual(output["completed_requests"], 0)
