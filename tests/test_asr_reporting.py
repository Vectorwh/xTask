"""ASR archival/report/search regression tests, without model or network access."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from xtask.benchmarks.asr.artifacts import asr_result_post_process, write_json, write_run_reports
from xtask.benchmarks.asr.reporting import report_to_dashboard, search_metrics, summarize_runs
from xtask.core import result_processor, scheduler
from xtask.tasks import parameter_search, task_parser
from xtask.tasks.assessment_report import AssessmentReporter, _extract_metrics
from xtask.tasks.auto_assessment import process_auto_assessment


def run_record(concurrency=2, repeat=1, latency=200, throughput=10, failed=False):
    return {
        "run_id": f"c{concurrency}_r{repeat}", "concurrency": concurrency, "repeat": repeat,
        "expected_requests": 1162, "completed_requests": 1162,
        "complete": True, "failed_requests": 1 if failed else 0,
        "score": {"profile": "joyai_legacy", "N": 10, "C": 9, "S": 1, "D": 0, "I": 0,
                  "error_rate_pct": 10, "error_count": 1},
        "metrics": {"e2el_p99_ms": latency, "audio_throughput": throughput,
                    "error_rate_pct": 10, "failed_requests": 1 if failed else 0},
    }


def task_config(concurrency=None, auto=False):
    asr = {"concurrency": concurrency or [2]}
    if auto:
        asr["auto_assess"] = {"metric": "e2el_p99_ms", "max": 300, "low": 0, "high": 8}
    return {"case_name": "asr_case", "metrics": "acc,perf", "served_model_name": "JoyaiASR",
            "benchmark": {"benchmark_type": "asr", "asr": asr}}


def result(config=None, runs=None, success=True):
    return {
        "success": success, "task_config": config or task_config(),
        "benchmark_asr_result": {"schema_version": 1, "backend": "legacy_infer",
                                 "scoring_profile": "joyai_legacy", "runs": runs or [run_record()]},
    }


class ASRRunSummaryTests(unittest.TestCase):
    def test_stats_separate_loads_and_exclude_failed_repetitions(self):
        runs = [run_record(latency=100, throughput=10), run_record(repeat=2, latency=300, throughput=30),
                run_record(repeat=3, latency=9999, throughput=9999, failed=True),
                run_record(concurrency=5, latency=800, throughput=50)]
        groups = summarize_runs(runs)
        self.assertEqual(groups["c2"]["total_runs"], 3)
        self.assertEqual(groups["c2"]["valid_runs"], 2)
        self.assertEqual(groups["c2"]["metrics"]["e2el_p99_ms"]["median"], 200)
        self.assertEqual(groups["c2"]["metrics"]["e2el_p99_ms"]["stddev"], 100)
        metrics = search_metrics(result(task_config([2, 5]), runs))
        self.assertNotIn("audio_throughput", metrics)
        self.assertEqual(metrics["c2.audio_throughput"], 20)
        self.assertEqual(metrics["c5.audio_throughput"], 50)

    def test_reports_write_csv_and_summary_not_only_excel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run_reports(root, result()["benchmark_asr_result"])
            rows = list(csv.DictReader((root / "runs.csv").open(encoding="utf-8-sig")))
            self.assertEqual(rows[0]["run_id"], "c2_r1")
            self.assertEqual(rows[0]["e2el_p99_ms"], "200.0")
            self.assertEqual(json.loads((root / "summary_by_concurrency.json").read_text())["c2"]["valid_runs"], 1)

    def test_corrupt_aggregate_preserves_evidence_and_completed_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            date = root / "20261103_120000"
            run_dir = date / "c2_r1"
            run_dir.mkdir(parents=True)
            original = b'{"broken":'
            (date / "asr_result.json").write_bytes(original)
            summary = run_record()
            summary["files"] = {"requests": "requests.jsonl"}
            write_json(run_dir / "summary.json", summary)
            (run_dir / "requests.jsonl").write_text('{"status":"ok"}\n')
            payload, _ = asr_result_post_process(task_config(), str(root), [])
            self.assertFalse(payload["success"])
            self.assertEqual((date / "asr_result.corrupt.json").read_bytes(), original)
            self.assertEqual(payload["runs"][0]["score"]["N"], 10)
            self.assertEqual(payload["runs"][0]["files"]["requests"], str(run_dir / "requests.jsonl"))

    def test_malformed_run_schema_is_archived_as_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            date = root / "20261103_120000"
            date.mkdir()
            write_json(date / "asr_result.json", {"schema_version": 1, "runs": [None]})
            payload, _ = asr_result_post_process(task_config(), str(root), [])
            self.assertFalse(payload["success"])
            self.assertTrue((date / "asr_result.corrupt.json").is_file())
            write_run_reports(date, payload)
            self.assertTrue((date / "runs.csv").is_file())

    def test_archive_failure_cannot_be_overridden_by_metric_pass(self):
        value = result()
        value.update(benchmark_success=True, archive_error="disk unavailable")
        self.assertFalse(result_processor.verify_result(value, task_config()))
        self.assertIn("disk unavailable", value["asr_gate_issues"][-1]["reason"])


class ASRSearchReportTests(unittest.TestCase):
    def test_search_observation_and_native_default_objective(self):
        config = parameter_search.search_config_for_task(task_config())
        self.assertEqual(config["objective"], {"metric": "audio_throughput", "direction": "maximize"})
        observation = parameter_search.build_search_observation(result(), config)
        self.assertTrue(observation["feasible"])
        self.assertEqual(observation["objective_value"], 10)
        bad = parameter_search.build_search_observation(result(success=False), config)
        self.assertFalse(bad["feasible"])
        self.assertIsNone(bad["objective_value"])
        self.assertEqual(_extract_metrics(result())["e2el_p99_ms"], 200)
        self.assertNotIn("tpot_p90", _extract_metrics(result()))

    def test_search_config_preserves_mapping_validation(self):
        cfg = task_config()
        cfg["assess_search"] = ["not", "a", "mapping"]
        with self.assertRaises(TypeError):
            parameter_search.search_config_for_task(cfg)

    def test_auto_assessment_uses_asr_latency_and_generates_next_load(self):
        cfg = task_config(auto=True)
        cfg["case_name"] = "asr_case_bs-2"
        thresholds, parameters = {}, {}
        current = "/tmp/asr_case_bs-2/asr_case_bs-2.yaml"
        tasks, keys = {current: "/tmp/asr_case_bs-2"}, [current]
        def generate(base, root, added, params):
            self.assertEqual(base, "asr_case")
            self.assertEqual(params["batch_size"], 5)
            added["next.yaml"] = "next"
            return True
        with mock.patch("xtask.tasks.auto_assessment.gen_group_autoassess_task", side_effect=generate):
            action = process_auto_assessment(
                SimpleNamespace(assess_test_type="auto_assess"), result(cfg), parameters, thresholds,
                999999, 0, tasks, keys, {}, current, "/tmp/asr_case_bs-2",
            )
        self.assertIsNone(action)
        self.assertEqual(keys, [current, "next.yaml"])
        self.assertEqual(thresholds["lower"], 2)

    def test_missing_asr_auto_target_stops_explicitly(self):
        thresholds = {}
        action = process_auto_assessment(
            SimpleNamespace(assess_test_type="auto_assess"), result(), {}, thresholds,
            1000, 10, {}, [], {},
        )
        self.assertEqual(action, "break")
        self.assertEqual(thresholds["stop_reason"], "invalid_asr_auto_assess_config")

    def test_auto_yaml_changes_asr_concurrency_not_llm_batch_size(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "base.yaml"
            original.write_text(yaml.safe_dump(task_config(auto=True)))
            tasks = {}
            with mock.patch.object(task_parser, "find_case_file", return_value=str(original)):
                ok = task_parser.gen_group_autoassess_task("asr_case", str(root / "cases"), tasks, {"batch_size": 5})
            self.assertTrue(ok)
            value = yaml.safe_load(Path(next(iter(tasks))).read_text())
            self.assertEqual(value["benchmark"]["asr"]["concurrency"], [5])
            self.assertNotIn("perf", value["benchmark"])

    def test_assessment_report_serializes_real_metrics_and_rejects_failed_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = task_config()
            cfg.update(case_name="asr_case_sg1", assess_search={"strategy": "exhaustive"})
            task = root / "asr_case_sg1.yaml"
            task.write_text(yaml.safe_dump(cfg))
            reporter = AssessmentReporter(SimpleNamespace(assess_test_type="para_set"),
                {str(task): str(root)}, strategy="exhaustive", output_root=root / "reports")
            reporter.record_result(str(task), str(root), result(cfg, success=False))
            state = reporter._states["asr_case"]
            self.assertEqual(state["objective"]["metric"], "audio_throughput")
            self.assertFalse(state["trials"][0]["feasible"])
            self.assertEqual(state["trials"][0]["metrics"]["audio_throughput"], 10)

    def test_auto_report_has_no_fake_tpot_constraints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = task_config(auto=True)
            cfg["case_name"] = "asr_case_bs-2"
            task = root / "asr_case_bs-2.yaml"
            task.write_text(yaml.safe_dump(cfg))
            reporter = AssessmentReporter(SimpleNamespace(assess_test_type="auto_assess"),
                {str(task): str(root)}, strategy="auto_assess", output_root=root / "reports")
            reporter.record_result(str(task), str(root), result(cfg))
            state = reporter._states["asr_case"]
            self.assertEqual(state["constraints"], {"e2el_p99_ms": {"max": 300}})
            self.assertTrue(state["trials"][0]["feasible"])
            self.assertEqual(state["trials"][0]["objective_value"], 2)


class ASRRemoteArchiveTests(unittest.TestCase):
    def test_first_remote_run_download_preserves_control_host_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            logs = root / "xllm_log/other-node"
            logs.mkdir(parents=True)
            (logs / "startup.sh").write_text("keep me")
            (root / "20261102_120000").mkdir()
            manager = mock.Mock()
            manager.exec_in_container.return_value = SimpleNamespace(success=True,
                stdout='["20261102_120000", "20261103_120000", "asr_inputs"]')
            def download(remote, local):
                self.assertEqual(Path(remote).name, "20261103_120000")
                self.assertNotEqual(Path(local), root)
                Path(local).mkdir()
                return True
            manager.executor.download.side_effect = download
            runtime = SimpleNamespace(container_manager=manager,
                container_config=SimpleNamespace(container_name="remote"), plan=SimpleNamespace(host="remote"))
            context = SimpleNamespace(output_file_path=str(root), state={"init_entries": ["20261102_120000"]})
            with mock.patch.object(scheduler, "_chown_via_running_container", return_value=True):
                scheduler._download_asr_runner_results(context, runtime)
            self.assertEqual((logs / "startup.sh").read_text(), "keep me")
            self.assertTrue((root / "20261102_120000").is_dir())
            manager.executor.download.assert_called_once()

    def test_first_remote_run_returns_summary_through_scheduler_branch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "xllm_log/other-node").mkdir(parents=True)
            (root / "xllm_log/other-node/startup.sh").write_text("keep me")
            manager = mock.Mock()
            benchmark = SimpleNamespace(success=True, exit_code=0, stdout="batch complete", stderr="", execution_time=1)
            manager.exec_in_container.return_value = SimpleNamespace(success=True, stdout='["20261103_120000"]')
            runtime = SimpleNamespace(container_manager=manager,
                container_config=SimpleNamespace(container_name="remote"),
                plan=SimpleNamespace(is_local=False, is_benchmark_runner=True, host="remote"))
            def download(remote, local):
                date = Path(local)
                date.mkdir()
                value = result()["benchmark_asr_result"]
                value["state"] = "completed"
                write_json(date / "asr_result.json", value)
                return True
            manager.executor.download.side_effect = download
            context = SimpleNamespace(output_file_path=str(root), result={"container_id": "test"},
                state={"task_config": task_config(), "service_healthy": True,
                       "engine_config": {}, "container_config": runtime.container_config,
                       "container_manager": manager, "benchmark_cmd": "test", "init_entries": [],
                       "multi_host": True, "service_runtimes": [runtime]})
            with mock.patch.object(scheduler, "_execute_benchmark_with_npu_guard", return_value=benchmark), mock.patch.object(
                scheduler, "_chown_via_running_container", return_value=True
            ), mock.patch.object(scheduler, "_record_service_group_archive_path"):
                scheduler._run_benchmark_and_process(context)
            self.assertTrue(context.result["success"], context.result)
            self.assertEqual(context.result["benchmark_asr_result"]["runs"][0]["run_id"], "c2_r1")
            self.assertEqual((root / "xllm_log/other-node/startup.sh").read_text(), "keep me")
            manager.executor.download.assert_called_once()

    def test_initial_staging_failure_still_has_task_identity_for_date_archive(self):
        cfg = task_config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "task.yaml"
            path.write_text(yaml.safe_dump(cfg))
            context = SimpleNamespace(output_file_path=str(root / "case"), task_yaml_path=str(path),
                                      args=None, result={}, state={}, complete=False)
            with mock.patch.object(scheduler, "generate_commands", side_effect=FileNotFoundError("infer.zip")), mock.patch.object(
                scheduler.Json_Reader, "load_config", return_value={"XLLM_CONFIG": {}}
            ):
                with self.assertRaises(FileNotFoundError):
                    scheduler._load_task_and_engine(context)
            self.assertEqual(context.state["task_config"]["benchmark"]["benchmark_type"], "asr")
            context.result.update(success=False, message="missing infer.zip")
            with mock.patch("xtask.utils.get_host_ip", return_value="client"):
                scheduler._archive_asr_execution(context)
            archive = Path(context.result["benchmark_output_path"])
            self.assertTrue((archive / "execution_result.json").is_file())
            self.assertFalse(json.loads((archive / "execution_result.json").read_text())["success"])

    def test_repeated_remote_trials_do_not_upload_over_the_case_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "asr_inputs").mkdir()
            (root / "asr_inputs/runner_config.json").write_text("{}")
            (root / "run_asr_test.sh").write_text("run")
            old = root / "20261103_120000"
            old.mkdir()
            runtime = SimpleNamespace(container_manager=mock.Mock())
            runtime.container_manager.executor.run.return_value = SimpleNamespace(success=True)
            runtime.container_manager.executor.upload.return_value = True
            self.assertTrue(scheduler._sync_asr_trial_inputs(runtime, str(root)))
            targets = [call.args[1] for call in runtime.container_manager.executor.upload.call_args_list]
            self.assertEqual(targets, [str(root / "run_asr_test.sh"), str(root / "asr_inputs")])
            self.assertNotIn(str(root), targets)
            self.assertTrue(old.is_dir())

    def test_workbook_exception_is_archived_not_swallowed_as_success(self):
        cfg = task_config()
        context = SimpleNamespace(result={"success": True, "task_config": cfg}, state={}, task_yaml_path="task", args=None)
        with mock.patch.object(scheduler, "_try_archive_asr_execution"), mock.patch.object(
            scheduler, "_ensure_benchmark_date_dir"
        ), mock.patch.object(scheduler, "write_result", side_effect=OSError("disk full")):
            scheduler._finalize_result(context)
        self.assertFalse(context.result["success"])
        self.assertFalse(context.result["report_generated"])
        self.assertIn("disk full", context.result["archive_error"])

    def test_workbook_failure_is_saved_in_execution_status(self):
        cfg = task_config()
        context = SimpleNamespace(result={"success": True, "task_config": cfg}, state={}, task_yaml_path="task", args=None)
        def archive(ctx):
            return None
        with mock.patch.object(scheduler, "_try_archive_asr_execution", side_effect=archive), mock.patch.object(
            scheduler, "_ensure_benchmark_date_dir"
        ), mock.patch.object(scheduler, "write_result", return_value=False):
            scheduler._finalize_result(context)
        self.assertFalse(context.result["success"])
        self.assertFalse(context.result["report_generated"])
        self.assertIn("workbook", context.result["archive_error"])


class ASRDashboardTests(unittest.TestCase):
    def test_main_optional_hook_routes_asr_without_overwriting_unknown_version(self):
        import main
        root = Path(__file__).resolve().parents[1] / "third_party/bench-reporter"
        sys.path.insert(0, str(root))
        try:
            from bench_reporter import integration
            value = result()
            with mock.patch.dict("os.environ", {"XTASK_BENCH_REPORT": "1"}), mock.patch.object(
                integration, "report_xtask_result"
            ) as llm, mock.patch("xtask.benchmarks.asr.reporting.report_to_dashboard") as asr:
                main.report_benchmark_if_available(value)
            asr.assert_called_once()
            llm.assert_not_called()
            self.assertNotIn("engine_version", value["task_config"])
        finally:
            sys.path.remove(str(root))

    def test_opt_in_adapter_preserves_asr_units_and_has_no_fake_accuracy(self):
        root = Path(__file__).resolve().parents[1] / "third_party/bench-reporter"
        sys.path.insert(0, str(root))
        try:
            from bench_reporter import integration
            captured = []
            value = result(runs=[run_record(), run_record(repeat=2)])
            value["artifact_host"] = "client"
            with mock.patch.object(integration, "_post_or_spool", side_effect=lambda p: captured.append(p)):
                report_to_dashboard(value, task_config())
            self.assertEqual(len(captured), 2)
            for report in captured:
                self.assertEqual(report["modelType"], "asr")
                self.assertNotIn("accuracyResults", report)
                self.assertNotIn("sla", report["toolSpecificConfig"])
                self.assertIsNone(report["hardware"]["gpuCount"])
                self.assertNotIn("outputTokenPerSec", report["sweepResults"][0])
                self.assertNotIn("tpotAvgMs", report["sweepResults"][0])
                self.assertEqual(report["toolSpecificConfig"]["asr"]["metrics"]["error_rate_pct"], 10)
        finally:
            sys.path.remove(str(root))


if __name__ == "__main__":
    unittest.main()
