import io
import json
import subprocess
import tempfile
import threading
import unittest
import wave
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.benchmarks.asr.artifacts import archive_execution, asr_result_post_process, extract_archive, portable_result, score_legacy, write_json
from xtask.benchmarks.asr.config import normalize_config, result_issues
from xtask.benchmarks.asr.generate_asr_script import generate_asr_script
from xtask.benchmarks.asr.run_asr import run, summarize
from xtask.core import result_processor, scheduler

PROJECT = Path(__file__).resolve().parents[1]


@contextmanager
def fake_server(mode="success"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"test-asr"}]}')
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, body))
            if mode == "http_error":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error":"test failure"}')
                return
            stream = (json.loads(body).get("stream", False) if self.path.endswith("completions")
                      else b'name="stream"\r\n\r\ntrue' in body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.end_headers()
            text = "今天下雨" if mode != "wrong" else "明天下雪"
            if mode == "empty":
                text = ""
            if not stream:
                value = {"text": text} if self.path.endswith("transcriptions") else {"choices": [{"message": {"content": text}}]}
                self.wfile.write(json.dumps(value).encode())
                return
            events = [{"choices": [{"delta": {"role": "assistant"}}]},
                      {"choices": [{"delta": {"content": text}}]}]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            if mode == "invalid_sse":
                self.wfile.write(b'data: not-json\n\n')
            if mode != "truncated":
                reason = "length" if mode == "length" else "stop"
                event = {"choices": [{"delta": {}, "finish_reason": reason}]}
                self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def dataset_archive(root):
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000)
    path = root / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/ta3_test0925_quiet/audios/one.wav", audio.getvalue())
        archive.writestr("data/ta3_test0925_quiet/data.jsonl", json.dumps({
            "key": "sample-1", "source": "/data/ta3_test0925_quiet/audios/one.wav",
            "target": "今天下雨", "context": "天气相关内容",
        }, ensure_ascii=False) + "\n")
    return path


def config(root, endpoint, **overrides):
    value = {
        "infer_archive": str(PROJECT / "asr_doc/infer.zip"),
        "dataset_archive": str(dataset_archive(root)),
        "server_addr": endpoint, "model": "test-asr", "expected_samples": 1,
        "concurrency": [1], "warmup": {"synthetic_requests": 0},
    }
    value.update(overrides)
    return value


class ASRConfigTests(unittest.TestCase):
    def test_rejects_unsupported_or_invalid_config(self):
        for value in ({"backend": "aisbench"}, {"concurrency": [1, 1]}, {"concurrency": [0]},
                      {"dataset_name": "../outside"}, {"cache_condition": "cold"},
                      {"generation": {"seed": 4419}}, {"stream": "true"},
                      {"unknown_field": 1}, {"generation": {"top_k": 0}},
                      {"gates": [{"metric": "cer", "compare": "lte", "baseline": 1}]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_config(value)

    def test_multiple_metrics_are_and_gated_and_zero_is_valid(self):
        cfg = {"benchmark": {"benchmark_type": "asr", "asr": {
            "concurrency": [1], "gates": [
                {"metric": "error_rate_pct", "compare": "lte", "baseline": 4, "threshold": 0.05},
                {"metric": "e2el_p99_ms", "compare": "lte", "baseline": 100},
            ],
        }}}
        payload = {"runs": [{"run_id": "c1_r1", "concurrency": 1, "repeat": 1,
                    "complete": True, "failed_requests": 0,
                    "expected_requests": 1162, "completed_requests": 1162,
                    "score": {"N": 4, "C": 4, "S": 0, "D": 0, "I": 0, "profile": "joyai_legacy"},
                    "metrics": {"error_rate_pct": 3, "e2el_p99_ms": 90}}]}
        self.assertFalse(result_issues(payload, cfg))
        payload["runs"][0]["metrics"]["error_rate_pct"] = 0
        self.assertFalse(result_issues(payload, cfg))
        payload["runs"][0]["metrics"]["error_rate_pct"] = 5
        self.assertEqual(result_issues(payload, cfg)[0]["category"], "baseline")
        self.assertFalse(result_processor.verify_result({"benchmark_success": True, "benchmark_asr_result": payload}, cfg))

    def test_missing_run_or_score_fails_without_baseline(self):
        self.assertTrue(result_issues({"runs": []}, {"benchmark": {"asr": {"concurrency": [1]}}}))

    def test_combined_non_asr_quality_failure_cannot_pass_perf(self):
        task = {"metrics": "acc,perf", "benchmark": {"benchmark_type": "evalscope",
                "acc": {"baseline": 90.0, "threshold": 0.05}, "perf": {}}}
        result = {"benchmark_success": True, "benchmark_acc_result": 1,
                  "benchmark_perf_result": SimpleNamespace(summary=SimpleNamespace())}
        self.assertFalse(result_processor.verify_result(result, task))


class ASRArtifactTests(unittest.TestCase):
    def test_rejects_zip_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / "bad.zip", "w") as z:
                z.writestr("../escape.py", "test")
            with self.assertRaises(ValueError):
                extract_archive(root / "bad.zip", root / "extracted")
            self.assertFalse((root / "escape.py").exists())

    @unittest.skipUnless((PROJECT / "asr_doc/infer.zip").is_file(), "Requires the supplied infer.zip integration fixture")
    def test_original_scorer_and_missing_sample_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extract_archive(PROJECT / "asr_doc/infer.zip", root)
            ref, hyp = root / "ref", root / "hyp"
            ref.write_text("one 今天下雨\n", encoding="utf-8")
            hyp.write_text("one 今天下雪\n", encoding="utf-8")
            score = score_legacy(root / "infer/wer_tools/compute-wer.py", ref, hyp, root / "wer")
            self.assertEqual(score["N"], 4)
            self.assertEqual(score["error_rate_pct"], 25)
            hyp.write_text("other 今天下雪\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                score_legacy(root / "infer/wer_tools/compute-wer.py", ref, hyp, root / "wer")

    def test_latency_and_wall_throughput_use_distinct_denominators(self):
        records = [{"idx": i, "status": "ok", "elapsed_ms": 1000,
                    "first_text_ms": 100, "legacy_ttft_ms": 50,
                    "audio_duration_s": 10, "started_at_s": 0, "finished_at_s": 1}
                   for i in range(2)]
        summary = summarize(records, 1, 0)
        self.assertEqual(summary["legacy_rtf"], 0.1)
        self.assertEqual(summary["audio_throughput"], 20)
        self.assertEqual(summary["request_throughput"], 2)


class ASRDateArchiveTests(unittest.TestCase):
    def test_portable_references_survive_moving_the_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "20261103_120000"
            directory.mkdir()
            file = directory / "c2_r1" / "requests.jsonl"
            file.parent.mkdir()
            file.write_text("test evidence")
            payload = {"schema_version": 1, "state": "completed", "config_file": str(directory / "config.json"),
                       "runs": [{"run_id": "c2_r1", "files": {"requests.jsonl": str(file)}}]}
            write_json(directory / "asr_result.json", portable_result(payload, directory))
            stored = json.loads((directory / "asr_result.json").read_text())
            self.assertEqual(stored["runs"][0]["files"]["requests.jsonl"], "c2_r1/requests.jsonl")
            relocated = root / "moved"
            relocated.mkdir()
            directory.rename(relocated / directory.name)
            parsed, path = asr_result_post_process({}, str(relocated), [])
            self.assertTrue(Path(parsed["runs"][0]["files"]["requests.jsonl"]).is_file())
            self.assertEqual(parsed["config_file"], str(Path(path) / "config.json"))

    def test_interrupted_partial_run_has_a_failed_date_summary(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                directory = root / "20261103_120000"
                run_dir = directory / "c2_r1"
                run_dir.mkdir(parents=True)
                (run_dir / "requests.jsonl").write_text(
                    '{"sample_id":"one","status":"ok"}\n{"incomplete":', encoding="utf-8"
                )
                if existing:
                    write_json(directory / "asr_result.json", {
                        "schema_version": 1, "state": "running", "runs": [{
                            "run_id": "c2_r1", "concurrency": 2, "repeat": 1,
                            "expected_requests": 1162, "complete": False,
                        }],
                    })
                payload, _ = asr_result_post_process({}, str(root), [])
                self.assertFalse(payload["success"])
                self.assertEqual(payload["state"], "interrupted")
                self.assertEqual(payload["runs"][0]["completed_requests"], 1)
                self.assertIsNone(payload["runs"][0].get("score"))
                self.assertTrue((directory / "asr_result.json").is_file())

    def test_execution_archive_is_self_contained_and_redacts_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            date = root / "20261103_120000"
            inputs = root / "asr_inputs"
            inputs.mkdir()
            write_json(inputs / "runner_config.json", {"api_key": "private-key", "api": "audio"})
            (root / "run_asr_test.sh").write_text("#!/bin/bash\necho first-run\n")
            task = {"case_name": "test", "external_service": {"api_key": "private-key"}}
            source = root / "source.yaml"
            source.write_text("case_name: test\nexternal_service:\n  api_key: private-key\n")
            result = {"success": False, "benchmark_exit_code": 1,
                      "benchmark_stdout": "completed 2 requests", "benchmark_stderr": "timeout",
                      "benchmark_command": "bash run_asr_test.sh", "startup_command": "external"}
            archive_execution(date, root, task, result, str(source))
            for name in ("task.yaml", "source_task.yaml", "runner_config.json", "run_asr_test.sh",
                         "commands.json", "benchmark_stdout.log", "benchmark_stderr.log", "execution_result.json"):
                self.assertTrue((date / name).is_file(), name)
            self.assertNotIn("private-key", (date / "task.yaml").read_text())
            self.assertNotIn("private-key", (date / "runner_config.json").read_text())
            self.assertNotIn("private-key", (date / "source_task.yaml").read_text())
            self.assertFalse(json.loads((date / "execution_result.json").read_text())["success"])
            self.assertEqual((date / "benchmark_stderr.log").read_text(), "timeout")
            # A later invocation may rewrite case-root inputs. It must not
            # replace this run's configuration or script during finalization.
            (root / "run_asr_test.sh").write_text("#!/bin/bash\necho second-run\n")
            write_json(inputs / "runner_config.json", {"api": "chat"})
            archive_execution(date, root, {"case_name": "another-run"}, {"success": True})
            self.assertIn("first-run", (date / "run_asr_test.sh").read_text())
            self.assertEqual(json.loads((date / "runner_config.json").read_text())["api"], "audio")
            self.assertEqual((date / "benchmark_stderr.log").read_text(), "timeout")
            self.assertTrue(json.loads((date / "execution_result.json").read_text())["success"])

    def test_scheduler_date_archive_captures_logs_and_final_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            date = root / "20261103_120000"
            date.mkdir()
            write_json(date / "asr_result.json", {"schema_version": 1, "state": "completed", "runs": []})
            log_root = root / "xllm_log" / "master-0"
            (log_root / "log").mkdir(parents=True)
            (log_root / "log" / "node_0.log").write_text("inference output")
            manager = mock.Mock()
            manager.exec_in_container.return_value = SimpleNamespace(success=True)
            runtime = SimpleNamespace(
                plan=SimpleNamespace(instance_name="master-0", is_local=True, output_dir=str(log_root)),
                container_manager=manager, container_config=SimpleNamespace(container_name="owned-runtime"),
            )
            task = {"benchmark": {"benchmark_type": "asr", "asr": {"concurrency": [2]}}}
            result = {"task_config": task, "benchmark_output_path": str(date), "success": True,
                      "execution_time": 12.3, "benchmark_stdout": "batch finished", "benchmark_stderr": ""}
            context = SimpleNamespace(output_file_path=str(root), task_yaml_path="",
                                      state={"task_config": task, "service_runtimes": [runtime]}, result=result)
            with mock.patch("xtask.utils.get_host_ip", return_value="11.87.191.75"):
                scheduler._archive_asr_execution(context)
            self.assertEqual((date / "xllm_log/master-0/log/node_0.log").read_text(), "inference output")
            self.assertEqual(context.result["artifact_host"], "11.87.191.75")
            location = result_processor._result_location(context.result, "server")
            self.assertTrue(location.startswith("11.87.191.75/"))
            self.assertTrue(location.endswith("/20261103_120000"))
            result["success"], result["message"] = False, "final failure"
            scheduler._archive_asr_execution(context)
            saved = json.loads((date / "execution_result.json").read_text())
            self.assertFalse(saved["success"])
            self.assertEqual(saved["execution_time"], 12.3)
            # Finalize must not overwrite trial-specific inference logs.
            self.assertEqual((date / "xllm_log/master-0/log/node_0.log").read_text(), "inference output")

    def test_readiness_failure_creates_a_date_archive_without_running_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = {"benchmark": {"benchmark_type": "asr", "asr": {"concurrency": [2]}}}
            context = SimpleNamespace(output_file_path=str(root), task_yaml_path="",
                state={"task_config": task, "external_service": True, "init_entries": []},
                result={"task_config": task, "success": False, "message": "Service not ready"})
            with mock.patch("xtask.utils.get_host_ip", return_value="client"), mock.patch(
                "xtask.core.external_service.resolve_external_location", return_value=SimpleNamespace(url="http://server:8018")
            ):
                scheduler._archive_asr_execution(context)
            directory = Path(context.result["benchmark_output_path"])
            self.assertRegex(directory.name, r"^\d{8}_\d{6}$")
            self.assertTrue((directory / "execution_result.json").is_file())
            self.assertFalse(json.loads((directory / "asr_result.json").read_text())["success"])

    def test_atomic_json_update_retains_previous_document_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "asr_result.json"
            write_json(path, {"state": "running"})
            with mock.patch("xtask.benchmarks.asr.artifacts.os.replace", side_effect=OSError("test interruption")):
                with self.assertRaises(OSError):
                    write_json(path, {"state": "completed"})
            self.assertEqual(json.loads(path.read_text())["state"], "running")
            self.assertEqual(list(Path(temp).iterdir()), [path])


class ASRGeneratorTests(unittest.TestCase):
    def test_scheduler_generation_stages_archives_and_does_not_install(self):
        task = {"case_name": "asr-test", "host_addr": "127.0.0.1", "port": 8018,
                "served_model_name": "JoyaiASR", "metrics": "acc,perf",
                "benchmark": {"benchmark_type": "asr", "asr": {"concurrency": [2]}}}
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "xtask.benchmarks.asr.generate_asr_script.path_mapping_in_container", side_effect=lambda p: p
        ):
            # Generator unit tests do not depend on untracked business archives.
            for name in ("infer", "data"):
                archive = Path(temp) / (name + ".zip")
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr("placeholder", "unit test")
                task["benchmark"]["asr"]["infer_archive" if name == "infer" else "dataset_archive"] = str(archive)
            command = scheduler.generate_benchmark_command(task, temp)
            script = (Path(temp) / "run_asr_test.sh").read_text()
            snapshot = json.loads((Path(temp) / "asr_inputs/runner_config.json").read_text())
            self.assertEqual(command, "bash " + temp + "/run_asr_test.sh")
            self.assertEqual(snapshot["server_addr"], "http://127.0.0.1:8018/v1")
            self.assertEqual(snapshot["concurrency"], [2])
            self.assertNotIn("pip install", script)
            self.assertFalse(scheduler._should_run_perf_warmup(task))
            self.assertTrue((Path(temp) / "asr_inputs/infer.zip").is_file())

    def test_external_endpoint_preserves_gateway_and_api_key(self):
        task = {"external_service": {"enabled": True, "endpoint": "https://host:8443/gateway/v1", "api_key": "secret"},
                "benchmark": {"benchmark_type": "asr", "asr": {"concurrency": [1]}}}
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "xtask.benchmarks.asr.generate_asr_script.path_mapping_in_container", side_effect=lambda p: p
        ):
            for name in ("infer", "data"):
                archive = Path(temp) / (name + ".zip")
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr("placeholder", "unit test")
                task["benchmark"]["asr"]["infer_archive" if name == "infer" else "dataset_archive"] = str(archive)
            generate_asr_script(task, temp)
            cfg = json.loads((Path(temp) / "asr_inputs/runner_config.json").read_text())
            self.assertEqual(cfg["server_addr"], "https://host:8443/gateway/v1")
            self.assertEqual(cfg["api_key"], "secret")


class ASRSchedulerTests(unittest.TestCase):
    def _context(self, root):
        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(
            success=True, exit_code=0, stdout="", stderr="", execution_time=0.1
        )
        manager.get_container_logs.return_value = ""
        task = {"case_name": "asr-case", "metrics": "acc,perf", "engine_type": "xllm",
                "benchmark": {"benchmark_type": "asr", "asr": {"concurrency": [1]}}}
        context = SimpleNamespace(output_file_path=str(root), result={"container_id": "test"}, state={
            "service_healthy": True, "external_service": True, "task_config": task,
            "engine_config": {}, "container_manager": manager,
            "container_config": SimpleNamespace(container_name="test"),
            "benchmark_cmd": "run-asr", "init_entries": [],
        })
        return context, manager

    def test_failed_command_still_processes_and_archives_partial_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            context, manager = self._context(root)
            manager.exec_in_container.return_value.success = False
            manager.exec_in_container.return_value.exit_code = 1
            directory = root / "20261103_120000"
            def execute(*args, **kwargs):
                directory.mkdir()
                (directory / "asr_result.json").write_text(json.dumps({
                    "schema_version": 1, "backend": "legacy_infer", "runs": [],
                    "execution_error": "ASR request failed",
                }))
                return SimpleNamespace(success=False, exit_code=1, stdout="", stderr="", execution_time=0.1)
            manager.exec_in_container.side_effect = execute
            with mock.patch.object(scheduler, "_record_service_group_archive_path") as archive:
                scheduler._run_benchmark_and_process(context)
            self.assertFalse(context.result["success"])
            self.assertIn("benchmark_asr_result", context.result)
            self.assertEqual(context.result["benchmark_output_path"], str(directory))
            archive.assert_called_once()

    def test_pass_does_not_enter_generic_acc_or_perf_postprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            context, _ = self._context(root)
            def execute(*args, **kwargs):
                directory = root / "20261103_120000"
                directory.mkdir()
                (directory / "asr_result.json").write_text(json.dumps({
                    "schema_version": 1, "backend": "legacy_infer", "runs": [{
                        "run_id": "c1_r1", "concurrency": 1, "repeat": 1,
                        "complete": True, "expected_requests": 1162, "completed_requests": 1162,
                        "failed_requests": 0, "score": {"N": 4, "C": 4, "S": 0, "D": 0, "I": 0, "profile": "joyai_legacy"}, "metrics": {},
                    }],
                }))
                return SimpleNamespace(success=True, exit_code=0, stdout="", stderr="", execution_time=1)
            context.state["container_manager"].exec_in_container.side_effect = execute
            with mock.patch("xtask.benchmarks.base_benchmark.acc_result_postprocess") as acc, mock.patch(
                "xtask.benchmarks.base_benchmark.result_postprocess"
            ) as perf, mock.patch.object(scheduler, "_record_service_group_archive_path"):
                scheduler._run_benchmark_and_process(context)
            self.assertTrue(context.result["success"], context.result)
            acc.assert_not_called()
            perf.assert_not_called()


@unittest.skipUnless((PROJECT / "asr_doc/infer.zip").is_file(), "Requires the supplied original ASR client archive")
class ASREndToEndTests(unittest.TestCase):
    def test_original_client_streams_scores_and_archives_all_runs(self):
        with tempfile.TemporaryDirectory() as temp, fake_server() as (url, requests):
            root = Path(temp)
            cfg = config(root, url, concurrency=[1, 2], repeats=2)
            result = run(cfg, root / "outputs")
            self.assertTrue(result["success"], result)
            self.assertEqual(len(result["runs"]), 4)
            self.assertEqual(len(requests), 4)
            for item in result["runs"]:
                self.assertEqual(item["score"]["error_rate_pct"], 0)
                path = Path(item["files"]["requests.jsonl"])
                record = json.loads(path.read_text())
                self.assertEqual(record["request"]["form"]["prompt"], "天气相关内容")
                self.assertNotIn("seed", record["request"]["form"])
                self.assertIsNotNone(record["first_text_ms"])
            payload, directory = asr_result_post_process({}, str(root / "outputs"), [])
            self.assertEqual(payload["schema_version"], 1)
            self.assertTrue(Path(directory).is_dir())
            task = {"case_name": "asr-test", "metrics": "acc,perf", "benchmark": {"benchmark_type": "asr", "asr": cfg}}
            _, row = result_processor._build_result_row({"success": True, "benchmark_asr_result": payload}, task)
            self.assertEqual(row["accuracy"], "NA")
            self.assertEqual(row["c1_r1.score.error_rate_pct"], 0)
            self.assertFalse(any("prediction" in key for key in row))

    def test_http_failure_truncated_sse_and_empty_transcript_fail(self):
        for mode in ("http_error", "truncated", "empty", "invalid_sse", "length"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, fake_server(mode) as (url, _):
                root = Path(temp)
                result = run(config(root, url), root / "outputs")
                self.assertFalse(result["success"])
                self.assertEqual(result["runs"][0]["failed_requests"], 1)
                self.assertTrue(Path(result["runs"][0]["files"]["err.log"]).read_text())

    def test_quality_failure_survives_passing_performance_gate(self):
        with tempfile.TemporaryDirectory() as temp, fake_server("wrong") as (url, _):
            root = Path(temp)
            result = run(config(root, url, gates=[
                {"metric": "error_rate_pct", "compare": "lte", "baseline": 1},
                {"metric": "e2el_p99_ms", "compare": "lte", "baseline": 100000},
            ]), root / "outputs")
            self.assertFalse(result["success"])
            self.assertEqual(result["gate_issues"][0]["category"], "baseline")

    def test_sync_and_chat_paths(self):
        for api, stream in (("audio", False), ("chat", False), ("chat", True)):
            with self.subTest(api=api, stream=stream), tempfile.TemporaryDirectory() as temp, fake_server() as (url, _):
                root = Path(temp)
                result = run(config(root, url, api=api, stream=stream), root / "outputs")
                self.assertTrue(result["success"], result)

    def test_generated_shell_executes_original_client_and_scorer(self):
        with tempfile.TemporaryDirectory() as temp, fake_server() as (url, _):
            root = Path(temp)
            task = {"served_model_name": "test-asr", "metrics": "acc,perf",
                    "benchmark": {"benchmark_type": "asr", "asr": config(root, url)}}
            with mock.patch("xtask.benchmarks.asr.generate_asr_script.path_mapping_in_container", side_effect=lambda p: p):
                command = generate_asr_script(task, str(root / "case"))
            result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload, _ = asr_result_post_process(task, str(root / "case"), ["asr_inputs", "run_asr_test.sh"])
            self.assertTrue(payload["success"])
            self.assertEqual(payload["runs"][0]["score"]["error_count"], 0)

    def test_warmup_requests_are_separate_from_scored_dataset(self):
        with tempfile.TemporaryDirectory() as temp, fake_server() as (url, requests):
            root = Path(temp)
            result = run(config(root, url, warmup={"synthetic_requests": 3}), root / "outputs")
            self.assertTrue(result["success"], result)
            self.assertEqual(len(requests), 4)
            self.assertEqual(result["runs"][0]["completed_requests"], 1)
            self.assertEqual(result["runs"][0]["score"]["N"], 4)

    def test_repeated_invocations_have_distinct_dates_and_relative_references(self):
        with tempfile.TemporaryDirectory() as temp, fake_server() as (url, _):
            root = Path(temp)
            cfg = config(root, url)
            first = run(cfg, root / "outputs")
            second = run(cfg, root / "outputs")
            self.assertTrue(first["success"] and second["success"])
            directories = sorted((root / "outputs").iterdir())
            self.assertEqual(len(directories), 2)
            for directory in directories:
                payload = json.loads((directory / "asr_result.json").read_text())
                self.assertEqual(payload["config_file"], "config.json")
                reference = payload["runs"][0]["files"]["requests.jsonl"]
                self.assertFalse(Path(reference).is_absolute())
                self.assertTrue((directory / reference).is_file())

    def test_wrong_dataset_count_fails_before_network_requests(self):
        with tempfile.TemporaryDirectory() as temp, fake_server() as (url, requests):
            root = Path(temp)
            result = run(config(root, url, expected_samples=1162), root / "outputs")
            self.assertFalse(result["success"])
            self.assertEqual(requests, [])
            self.assertIn("Expected 1162", result["execution_error"])


if __name__ == "__main__":
    unittest.main()
