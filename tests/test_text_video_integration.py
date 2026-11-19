"""Regression fences for Wan and native text-video result/report routing."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.benchmarks.vbench.generate_vbench_script import generate_vbench_script
from xtask.benchmarks.vbench.vbench_postprocess import vbench_result_post_process
from xtask.benchmarks.vbench.text_video.config import normalize_config
from xtask.benchmarks.vbench.text_video.evaluate import evaluate, evaluate_archive
from xtask.benchmarks.vbench.text_video.reporting import archive_execution
from xtask.benchmarks.vbench.text_video.checks import run_checks
from xtask.core import result_processor, scheduler
from test_text_video_config import config
from test_text_video_runner import manifest, tensor, ENCODER


def task(root, **overrides):
    source = overrides.pop("manifest", None) or str(manifest(root))
    raw = config(manifest=source, **overrides)
    return {"case_name": "joyvideo_case", "served_model_name": "JoyAI-Video", "metrics": "acc",
            "external_service": {"enabled": True, "endpoint": "http://localhost:18004"},
            "benchmark": {"benchmark_type": "vbench", "vbench": raw}}


def payload():
    return {"adapter": "text_video", "task": "t2v", "scenario": "text", "complete": True,
            "expected_requests": 1, "completed_requests": 1, "failed": 0,
            "video_quality": {}, "performance": {"e2el_avg_ms": 900}}


class ScriptTests(unittest.TestCase):
    def test_text_video_script_never_calls_i2v_or_requires_image_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            generate_vbench_script(value, str(root / "out"))
            script = (root / "out/run_vbench_test.sh").read_text()
            runtime = json.loads((root / "out/text_video_inputs/runner_config.json").read_text())
            self.assertNotIn("run_i2v", script)
            self.assertNotIn("image-root", script)
            self.assertIn("text_video/run.py", script)
            self.assertTrue(runtime["endpoint"].endswith("/v1/video/generation"))
            self.assertNotIn("full_info", runtime)

    def test_bad_task_rejected_no_fallback_to_wan(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_vbench_script({"benchmark": {"vbench": {"task": "t2av_typo"}}}, directory)

    def test_stage_embeddings_and_keep_source_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor = tensor(root)
            source = manifest(root, [{"id": "x", "prompt": "dog", "embedding": descriptor, "encoder": ENCODER}])
            before = source.read_bytes()
            value = task(root, manifest=str(source), scenario="embedding", embedding_dim=2)
            # task() writes a default manifest; restore the explicit fixture.
            manifest(root, [{"id": "x", "prompt": "dog", "embedding": descriptor, "encoder": ENCODER}])
            generate_vbench_script(value, str(root / "out"))
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue((root / "out/text_video_inputs/00000_embedding.json").is_file())


class ResultRoutingTests(unittest.TestCase):
    def test_no_total_score_required_and_perf_not_treated_as_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            result = {"benchmark_success": True, "benchmark_vbench_result": payload()}
            self.assertTrue(result_processor.verify_result(result, value))
            value["metrics"] = "perf"
            self.assertTrue(result_processor.verify_result(result, value))
            self.assertIsNone(result_processor._configured_baseline(value))
            row = result_processor._result_payload(result, value)
            self.assertIn("video", row)
            self.assertNotIn("total_score", row)
            self.assertNotIn("summary", row)

    def test_missing_dimension_and_baseline_correctly_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root, mode="generate_and_evaluate", evaluation={"enabled": True, "dimensions": ["temporal_flickering"]})
            result = {"benchmark_success": True, "benchmark_vbench_result": payload(), "success": False}
            self.assertFalse(result_processor.verify_result(result, value))
            self.assertEqual(result_processor._classify_failure(result, value)[0], result_processor.FAILURE_PROCESSING)
            value["benchmark"]["vbench"]["evaluation"]["enabled"] = False
            value["benchmark"]["vbench"]["gates"] = [{"metric": "performance.e2el_avg_ms", "compare": "lte", "baseline": 800}]
            result.pop("video_gate_issues")
            self.assertFalse(result_processor.verify_result(result, value))
            self.assertEqual(result_processor._classify_failure(result, value)[0], result_processor.FAILURE_BASELINE)
            self.assertIn("900", result_processor._baseline_difference(result, value))

    def test_postprocess_reads_new_date_not_old_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            old = root / "20261104_010101"
            old.mkdir()
            current = root / "20261105_010101"
            current.mkdir()
            (current / "text_video_result.json").write_text(json.dumps(payload()))
            (current / "clip.mp4").write_bytes(b"fixture")
            (current / "requests.json").write_text(json.dumps([{"sample_id": "x", "repeat": 1, "arm": "main", "success": True, "media": {"fully_decoded": True}, "video_path": "clip.mp4", "video_sha256": hashlib.sha256(b"fixture").hexdigest()}]))
            result, path = vbench_result_post_process(value, str(root), [old.name])
            self.assertEqual(path, str(current))
            self.assertEqual(result["issues"], [])
            self.assertNotIn("i2v_score", result)

    def test_non_vbench_metadata_never_triggers_dispatch(self):
        value = {"metrics": "perf", "benchmark": {"benchmark_type": "aisbench", "perf": {}, "vbench": {"task": "t2v"}}}
        self.assertFalse(scheduler._is_text_video_task(value))
        self.assertIsNone(result_processor._configured_baseline(value))

    def test_archive_preserves_real_service_command_and_launch_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            archived = root / "20261105_010101"
            archived.mkdir()
            launch_dir = root / "xllm_log/master-0"
            launch_dir.mkdir(parents=True)
            launch = launch_dir / "start_xllm_serv_master-0.sh"
            launch.write_text("#!/bin/bash\nexport HCCL_OP_EXPANSION_MODE=HOST\nxllm --model_id=Joy\n")
            plan = SimpleNamespace(instance_name="master-0", startup_cmd="bash start_xllm_serv_master-0.sh",
                                   output_dir=str(launch_dir))
            context = SimpleNamespace(output_file_path=str(root), state={"task_config": value,
                                      "service_runtimes": [SimpleNamespace(plan=plan)]},
                                      result={"task_config": value, "benchmark_output_path": str(archived)}, task_yaml_path="")
            archive_execution(context)
            commands = json.loads((archived / "commands.json").read_text())
            self.assertEqual(commands["service_instances"]["master-0"], plan.startup_cmd)
            snapshot = archived / "service_startup/master-0" / launch.name
            self.assertEqual(snapshot.read_bytes(), launch.read_bytes())
            launch.write_text("replacement from next run")
            self.assertIn("--model_id=Joy", snapshot.read_text())

    def test_execution_archive_final_report_failure_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            archived = root / "20261105_010101"
            archived.mkdir()
            context = SimpleNamespace(output_file_path=str(root), state={"task_config": value, "init_entries": []},
                                      result={"task_config": value, "success": False, "benchmark_output_path": str(archived),
                                              "report_generated": False, "archive_error": "workbook failed", "artifact_host": "controller"},
                                      task_yaml_path="")
            archive_execution(context)
            result = json.loads((archived / "execution_result.json").read_text())
            self.assertFalse(result["report_generated"])
            self.assertEqual(result["artifact_host"], "controller")
            self.assertEqual(result["archive_error"], "workbook failed")

    def test_finalize_workbook_failure_only_changes_new_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            archived = root / "20261105_010101"
            archived.mkdir()
            context = SimpleNamespace(output_file_path=str(root), state={"task_config": value}, result={"task_config": value,
                         "success": True, "benchmark_output_path": str(archived), "artifact_host": "controller"},
                         task_yaml_path="", args=SimpleNamespace())
            with mock.patch.object(scheduler, "write_result", return_value=False):
                scheduler._finalize_result(context)
            self.assertFalse(context.result["success"])
            self.assertFalse(context.result["report_generated"])


class EvaluatorContractTests(unittest.TestCase):
    def test_external_scorer_must_cover_all_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"sample_id": "x", "repeat": 1, "arm": arm} for arm in ("reference", "target")]
            config_value = {"checks": {"negative_control": {"command": ["python", "scorer.py"]}}}
            (root / "negative_control.json").write_text(json.dumps({"complete": True, "metrics": {"avoid_rate": .9}, "pairs": []}))
            with mock.patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")):
                with self.assertRaises(ValueError):
                    run_checks(root, records, config_value)

    def test_external_scorer_not_run_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_checks(Path(directory), [], {"checks": {}})["checks_status"], {})

    def test_per_video_dimension_coverage_required(self):
        torch = SimpleNamespace(device=lambda value: value)
        utils = SimpleNamespace(CACHE_DIR="/no-cache", init_submodules=lambda dimensions, **kwargs: {d: [] for d in dimensions})
        module = SimpleNamespace(compute_temporal_flickering=lambda *args: (.9, []))
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("sys.modules", {"torch": torch, "vbench": SimpleNamespace(), "vbench.utils": utils}), mock.patch("importlib.import_module", return_value=module), mock.patch("xtask.benchmarks.vbench.text_video.decoder.prepare_decoder", return_value="fixture"):
            with self.assertRaises(ValueError):
                evaluate(Path(directory), [{"prompt": "dog", "video_path": "clip.mp4"}], {"dimensions": ["temporal_flickering"], "device": "cpu", "local": True})

    def test_evaluate_does_not_synthesize_missing_source_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = config(manifest=str(manifest(root)), mode="evaluate", source_run=str(root / "missing"))
            result, _ = evaluate_archive(settings, root / "out")
            self.assertFalse(result["complete"])
            self.assertTrue(result["issues"])


class FailureAndSafetyTests(unittest.TestCase):
    def test_failed_command_with_native_gate_is_not_missing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root, gates=[{"metric": "performance.e2el_avg_ms", "compare": "lte", "baseline": 800}])
            result = {"benchmark_success": False, "benchmark_vbench_result": payload(), "success": False}
            self.assertFalse(result_processor.verify_result(result, value))
            category, reason = result_processor._classify_failure(result, value)
            self.assertEqual(category, result_processor.FAILURE_BASELINE)
            self.assertIn("900", reason)

    def test_offline_script_does_not_need_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root, mode="evaluate", source_run=str(root / "saved"), evaluation={"enabled": True, "dimensions": ["temporal_flickering"]})
            value["external_service"] = {"enabled": True, "health_path": ""}
            generate_vbench_script(value, str(root / "out"))
            self.assertTrue((root / "out/run_vbench_test.sh").exists())

    def test_unknown_task_configuration_fields_fail_closed(self):
        with self.assertRaises(ValueError):
            normalize_config(config(fake_encoder_mode="auto"))
        with self.assertRaises(ValueError):
            normalize_config(config(metrics="acc,perf"))

    def test_optional_audio_cannot_claim_pure_video_performance(self):
        with self.assertRaises(ValueError):
            normalize_config(config(metrics="perf", media={"audio": "optional"}))

    def test_running_summary_is_interrupted_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            archive = root / "20261105_010101"
            archive.mkdir()
            (archive / "text_video_result.json").write_text(json.dumps(dict(payload(), state="running")))
            result, _ = vbench_result_post_process(value, str(root), [])
            self.assertEqual(result["state"], "interrupted")
            self.assertFalse(result["complete"])

    def test_corrupt_summary_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            archive = root / "20261105_010101"
            archive.mkdir()
            (archive / "text_video_result.json").write_bytes(b'{"broken":')
            with self.assertRaises(ValueError):
                vbench_result_post_process(value, str(root), [])
            self.assertEqual((archive / "text_video_result.corrupt.json").read_bytes(), b'{"broken":')

    def test_quality_cfg_requires_explicit_judge(self):
        value = config(scenario="cfg_pair", mode="generate_and_evaluate")
        value["generation"]["guidance_scale"] = 4
        with self.assertRaises(ValueError):
            normalize_config(value)

    def test_offline_weight_preflight_blocks_implicit_download(self):
        torch = SimpleNamespace(device=lambda value: value)
        utils = SimpleNamespace(CACHE_DIR="/no-such-assets", init_submodules=mock.Mock())
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("sys.modules", {"torch": torch, "vbench": SimpleNamespace(), "vbench.utils": utils}):
            with self.assertRaises(FileNotFoundError):
                evaluate(Path(directory), [{"prompt": "dog", "video_path": "clip.mp4"}], {"dimensions": ["subject_consistency"], "device": "cpu", "local": True})
        utils.init_submodules.assert_not_called()


class ArchiveAndConfigAdditionalTests(unittest.TestCase):
    def test_wire_batch_dimension_rejected_before_http(self):
        from xtask.benchmarks.vbench.text_video.embedding import load_embedding
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tensor.json"
            path.write_text(json.dumps({"datatype": "FP32", "shape": [1, 2, 2], "contents": {"fp32_contents": [.1] * 4}}))
            descriptor = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaises(ValueError):
                load_embedding(descriptor, root, 2, ENCODER)

    def test_missing_archived_media_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            archive = root / "20261105_010101"
            archive.mkdir()
            (archive / "text_video_result.json").write_text(json.dumps(payload()))
            result, _ = vbench_result_post_process(value, str(root), [])
            self.assertTrue(result["issues"])

    def test_paired_concurrency_and_mixed_perf_gates_rejected(self):
        value = config(scenario="cfg_pair")
        value["generation"]["guidance_scale"] = 4
        value["generation"]["concurrency"] = 2
        with self.assertRaises(ValueError):
            normalize_config(value)
        value["generation"]["concurrency"] = 1
        value["gates"] = [{"metric": "performance.request_throughput", "compare": "gte", "baseline": 1}]
        with self.assertRaises(ValueError):
            normalize_config(value)

    def test_invalid_request_counts_not_zero_or_bool(self):
        from xtask.benchmarks.vbench.text_video.config import result_issues
        settings = normalize_config(config())
        for count in (None, "1", True, 0, -1):
            result = dict(payload(), expected_requests=count, completed_requests=count)
            self.assertTrue(result_issues(result, settings))

    def test_external_endpoint_preserves_gateway_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = task(root)
            value["external_service"]["endpoint"] = "http://localhost:18004/gateway"
            generate_vbench_script(value, str(root / "out"))
            runtime = json.loads((root / "out/text_video_inputs/runner_config.json").read_text())
            self.assertEqual(runtime["endpoint"], "http://localhost:18004/gateway/v1/video/generation")


class OfflinePortablePathsTests(unittest.TestCase):
    def test_offline_scoring_stores_relative_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "generation/20261105_010101"
            source.mkdir(parents=True)
            video = source / "clip.mp4"
            video.write_bytes(b"fixture")
            (source / "manifest.json").write_text("[]")
            (source / "text_video_result.json").write_text(json.dumps(dict(payload(), expected_requests=1, completed_requests=1)))
            (source / "requests.json").write_text(json.dumps([{"sample_id": "x", "repeat": 1, "arm": "main", "prompt": "dog", "success": True, "parameters": config()["generation"], "media": {"fully_decoded": True}, "video_path": "clip.mp4", "video_sha256": hashlib.sha256(b"fixture").hexdigest()}]))
            settings = config(manifest=str(manifest(root)), mode="evaluate", source_run=str(source),
                              evaluation={"enabled": True, "dimensions": ["temporal_flickering"]})
            with mock.patch("xtask.benchmarks.vbench.text_video.media.validate_media", return_value={}), mock.patch("xtask.benchmarks.vbench.text_video.evaluate.evaluate", return_value={"temporal_flickering": .9}):
                result, archive = evaluate_archive(settings, root / "scoring")
            self.assertFalse(Path(result["source_run"]).is_absolute())
            self.assertEqual((archive / result["source_run"]).resolve(), source)
            rows = json.loads((archive / "requests.json").read_text())
            self.assertFalse(Path(rows[0]["video_path"]).is_absolute())
            self.assertEqual((archive / rows[0]["video_path"]).resolve(), video)
            task_value = {"metrics": "acc", "benchmark": {"benchmark_type": "vbench", "vbench": settings}}
            parsed, _ = vbench_result_post_process(task_value, str(root / "scoring"), [])
            self.assertEqual(parsed["issues"], [])
