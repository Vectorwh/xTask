import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openpyxl import load_workbook

from xtask.core import result_processor
from xtask.core.result_workbook import (
    EXCLUDED_RESULT_FIELDS,
    LEADING_RESULT_FIELDS,
    PERF_SUMMARY_FIELDS,
    SUMMARY_SHEET,
)


class FakePerfResult:
    def __init__(self, throughput=100.0):
        self.parameters = {
            "TTFT": {
                "total": SimpleNamespace(
                    average=100.0,
                    p90=120.0,
                    p99=140.0,
                    n=8,
                )
            },
            "ITL": {
                "total": SimpleNamespace(
                    average=8.8,
                    p90=9.0,
                    n=8,
                )
            },
        }
        self.summary = SimpleNamespace(
            total_requests=8,
            failed_requests=1,
            concurrency=2,
            max_concurrency=4,
            request_throughput=0.5,
            output_tps_avg=throughput,
            output_tps_total=throughput * 2,
            summary_tps_total=throughput * 3,
            input_tokens_avg=1024,
            ttft_p90=120.0,
        )
        self.metadata = {"should_not_be_exported": True}
        self.throughput = throughput

    def to_dict(self):
        raise AssertionError("Excel extraction must not call to_dict()")

    def to_json(self):
        raise AssertionError("Excel extraction must not call to_json()")


class VerifyResultTests(unittest.TestCase):
    @staticmethod
    def _config(baseline=None):
        return {
            "metrics": "perf",
            "benchmark": {
                "perf": {"baseline": baseline, "threshold": 0.1},
            },
        }

    def test_failed_benchmark_command_is_not_overwritten_by_missing_baseline(self):
        result = {
            "benchmark_success": False,
            "benchmark_perf_result": FakePerfResult(),
        }

        self.assertFalse(result_processor.verify_result(result, self._config()))

    def test_missing_perf_summary_fails_without_baseline(self):
        result = {"benchmark_success": True}

        self.assertFalse(result_processor.verify_result(result, self._config()))

    def test_valid_perf_summary_passes_without_baseline(self):
        result = {
            "benchmark_success": True,
            "benchmark_perf_result": FakePerfResult(),
        }

        self.assertTrue(result_processor.verify_result(result, self._config()))


class WriteResultTests(unittest.TestCase):
    def _source_case(self, root, model="qwen3_8b", case_name="case-a"):
        source = (
            Path(root)
            / "tests"
            / "e2e"
            / "xllm"
            / "qwen"
            / model
            / f"{case_name}.yaml"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("case_name: test\n", encoding="utf-8")
        return source

    def _config(self, case_name, *, metrics="perf", baseline=50):
        metric_config = {"baseline": baseline, "threshold": 0.1}
        return {
            "case_name": case_name,
            "engine_type": "xllm",
            "metrics": metrics,
            "host_addr": "11.87.191.75",
            "benchmark": {
                "benchmark_type": "aisbench",
                metrics: metric_config,
            },
        }

    def _result(self, config, *, success=True, perf=None, execution_time=1):
        result = {
            "success": success,
            "execution_time": execution_time,
            "output_file_path": f"outputs/{config['case_name']}/run",
            "task_config": config,
            "benchmark_success": True,
            "message": "ok" if success else "Benchmark result verification failed",
        }
        if perf is not None:
            result["benchmark_perf_result"] = perf
        return result

    def _sheet_rows(self, workbook_path, sheet_name):
        workbook = load_workbook(workbook_path, data_only=True)
        try:
            worksheet = workbook[sheet_name]
            headers = [cell.value for cell in worksheet[1]]
            return [
                dict(zip(headers, values))
                for values in worksheet.iter_rows(min_row=2, values_only=True)
            ]
        finally:
            workbook.close()

    def _sheet_headers(self, workbook_path, sheet_name):
        workbook = load_workbook(workbook_path, data_only=True)
        try:
            return [cell.value for cell in workbook[sheet_name][1]]
        finally:
            workbook.close()

    def test_writes_model_sheet_and_selected_performance_fields(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            source = self._source_case(root)
            config = self._config("case-a")
            written = result_processor.write_result(
                self._result(config, perf=FakePerfResult(123.4), execution_time=12.345),
                config,
                task_path=str(source),
            )
            workbook_path = Path(root) / "outputs" / "result.xlsx"
            rows = self._sheet_rows(workbook_path, "qwen3_8b")
            headers = self._sheet_headers(workbook_path, "qwen3_8b")
            workbook = load_workbook(workbook_path, data_only=True)
            sheet_names = workbook.sheetnames
            model_sheet = workbook["qwen3_8b"]
            summary_sheet = workbook[SUMMARY_SHEET]
            model_cells_have_borders = all(
                all(side.style == "thin" for side in (
                    cell.border.left,
                    cell.border.right,
                    cell.border.top,
                    cell.border.bottom,
                ))
                for row in model_sheet.iter_rows()
                for cell in row
            )
            summary_cells_have_borders = all(
                all(side.style == "thin" for side in (
                    cell.border.left,
                    cell.border.right,
                    cell.border.top,
                    cell.border.bottom,
                ))
                for row in summary_sheet.iter_rows()
                for cell in row
            )
            workbook.close()

        self.assertTrue(written)
        self.assertEqual(sheet_names, [SUMMARY_SHEET, "qwen3_8b"])
        self.assertTrue(model_cells_have_borders)
        self.assertTrue(summary_cells_have_borders)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_name"], "case-a")
        self.assertEqual(rows[0]["execution_time_s"], 12.35)
        self.assertEqual(rows[0]["output_path"], "11.87.191.75/case-a/run")
        self.assertEqual(rows[0]["accuracy"], "NA")
        self.assertEqual(headers.index("accuracy"), headers.index("output_path") + 1)
        self.assertEqual(rows[0]["summary.output_tps_avg"], 123.4)
        self.assertEqual(rows[0]["summary.total_requests"], 8)
        self.assertEqual(rows[0]["summary.failed_requests"], 1)
        self.assertEqual(rows[0]["summary.concurrency"], 2)
        self.assertEqual(rows[0]["summary.max_concurrency"], 4)
        self.assertEqual(rows[0]["summary.request_throughput"], 0.5)
        self.assertEqual(rows[0]["summary.output_tps_total"], 246.8)
        self.assertEqual(rows[0]["summary.summary_tps_total"], 370.2)
        self.assertEqual(rows[0]["TTFT.p99"], 140.0)
        self.assertEqual(headers[:6], list(LEADING_RESULT_FIELDS))
        self.assertTrue(EXCLUDED_RESULT_FIELDS.isdisjoint(headers))
        self.assertFalse(any(header.startswith("metadata.") for header in headers))
        self.assertEqual(
            {header for header in headers if header.startswith("summary.")},
            {f"summary.{field}" for field in PERF_SUMMARY_FIELDS},
        )
        self.assertFalse(any(header.endswith(".n") for header in headers))
        self.assertFalse(
            any(header.startswith("parameters.ITL.") for header in headers)
        )
        self.assertFalse(any("parameters." in header for header in headers))
        self.assertFalse(any("total" in header.split(".") for header in headers))
        self.assertTrue(rows[0]["finished_at"].endswith("+08:00"))
        self.assertEqual(rows[0]["failure_category"], "NA")
        self.assertEqual(rows[0]["failure_reason"], "NA")
        self.assertEqual(rows[0]["baseline_difference"], "NA")
        self.assertFalse((Path(root) / "outputs" / "result.csv").exists())

    def test_same_case_history_is_kept_in_one_contiguous_block(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            source_a = self._source_case(root, case_name="case-a")
            source_b = self._source_case(root, case_name="case-b")
            config_a = self._config("case-a")
            config_b = self._config("case-b")
            result_processor.write_result(
                self._result(config_a, perf=FakePerfResult(100), execution_time=1),
                config_a,
                task_path=str(source_a),
            )
            result_processor.write_result(
                self._result(config_b, perf=FakePerfResult(200), execution_time=2),
                config_b,
                task_path=str(source_b),
            )
            result_processor.write_result(
                self._result(config_a, perf=FakePerfResult(300), execution_time=3),
                config_a,
                task_path=str(source_a),
            )
            rows = self._sheet_rows(
                Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
            )

        self.assertEqual(
            [row["case_name"] for row in rows], ["case-a", "case-a", "case-b"]
        )
        self.assertEqual(
            [row["summary.output_tps_avg"] for row in rows],
            [100, 300, 200],
        )

    def test_performance_cases_precede_grouped_accuracy_cases(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            cases = (
                ("case-acc-a", "acc", 0.81),
                ("case-perf-a", "perf", 100),
                ("case-acc-b", "acc", 0.82),
                ("case-perf-b", "perf", 200),
                ("case-acc-a", "acc", 0.83),
            )
            for case_name, metrics, value in cases:
                source = self._source_case(root, case_name=case_name)
                config = self._config(case_name, metrics=metrics)
                if metrics == "acc":
                    result = self._result(config)
                    result["benchmark_acc_result"] = value
                else:
                    result = self._result(config, perf=FakePerfResult(value))
                result_processor.write_result(result, config, task_path=str(source))

            rows = self._sheet_rows(
                Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
            )
            headers = self._sheet_headers(
                Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
            )

        self.assertEqual(
            [row["case_name"] for row in rows],
            [
                "case-perf-a",
                "case-perf-b",
                "case-acc-a",
                "case-acc-a",
                "case-acc-b",
            ],
        )
        self.assertEqual(
            [row["accuracy"] for row in rows[2:]], [0.81, 0.83, 0.82]
        )
        self.assertEqual([row["accuracy"] for row in rows[:2]], ["NA", "NA"])
        performance_headers = headers[headers.index("accuracy") + 1 :]
        for row in rows[2:]:
            self.assertTrue(performance_headers)
            self.assertTrue(all(row[field] == "NA" for field in performance_headers))
        self.assertTrue(all(row["baseline_difference"] == "NA" for row in rows))
        self.assertEqual(headers.index("accuracy"), headers.index("output_path") + 1)

    def test_different_models_use_separate_sheets(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            for model_name, case_name in (
                ("qwen3_8b", "qwen-case"),
                ("deepseek_v3.2", "deepseek-case"),
            ):
                source = self._source_case(
                    root, model=model_name, case_name=case_name
                )
                config = self._config(case_name)
                result_processor.write_result(
                    self._result(config, perf=FakePerfResult()),
                    config,
                    task_path=str(source),
                )

            workbook = load_workbook(Path(root) / "outputs" / "result.xlsx")
            sheet_names = workbook.sheetnames
            workbook.close()

        self.assertEqual(
            sheet_names, [SUMMARY_SHEET, "qwen3_8b", "deepseek_v3.2"]
        )

    def test_accuracy_case_only_adds_accuracy_result(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            source = self._source_case(root, case_name="case-acc")
            config = self._config("case-acc", metrics="acc", baseline=0.8)
            result = self._result(config)
            result["benchmark_acc_result"] = 0.85
            result["benchmark_perf_result"] = FakePerfResult(999)
            result_processor.write_result(result, config, task_path=str(source))
            rows = self._sheet_rows(
                Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
            )
            headers = self._sheet_headers(
                Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
            )

        self.assertEqual(rows[0]["accuracy"], 0.85)
        self.assertEqual(headers.index("accuracy"), headers.index("output_path") + 1)
        self.assertNotIn("case_type", rows[0])
        self.assertNotIn("summary.output_tps_avg", rows[0])

    def test_accuracy_case_without_result_uses_dash(self):
        missing_values = (None, "", "NA", "N/A", float("nan"))
        for index, missing_value in enumerate(missing_values):
            with (
                self.subTest(missing_value=missing_value),
                tempfile.TemporaryDirectory() as root,
                mock.patch.object(result_processor.os, "getcwd", return_value=root),
            ):
                case_name = f"gsm8k-evaluation-{index}"
                source = self._source_case(root, case_name=case_name)
                config = self._config(case_name, metrics="acc", baseline=0.8)
                result = self._result(config, success=False)
                if missing_value is not None:
                    result["benchmark_acc_result"] = missing_value
                result_processor.write_result(
                    result,
                    config,
                    task_path=str(source),
                )
                rows = self._sheet_rows(
                    Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
                )

                self.assertEqual(rows[0]["accuracy"], "-")

    def test_failure_categories_and_summary_counts(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            source = self._source_case(root)
            config = self._config("case-a", baseline=100)
            xllm_failure = self._result(config, success=False)
            xllm_failure.update(
                benchmark_success=False,
                benchmark_stderr="NPU process abnormal: card 0 process missing",
                message="Benchmark command failed with exit code -15",
            )
            baseline_failure = self._result(
                config, success=False, perf=FakePerfResult(50)
            )
            archive_failure = self._result(
                config, success=False, perf=FakePerfResult(100)
            )
            archive_failure["message"] = "Error in scheduler: archive permission denied"
            result_processor.write_result(
                xllm_failure, config, task_path=str(source)
            )
            result_processor.write_result(
                baseline_failure, config, task_path=str(source)
            )
            result_processor.write_result(
                archive_failure, config, task_path=str(source)
            )
            workbook_path = Path(root) / "outputs" / "result.xlsx"
            rows = self._sheet_rows(workbook_path, "qwen3_8b")
            summary_rows = self._sheet_rows(workbook_path, SUMMARY_SHEET)

        self.assertEqual(rows[0]["failure_category"], result_processor.FAILURE_XLLM)
        self.assertIn("NPU process abnormal", rows[0]["failure_reason"])
        self.assertEqual(
            rows[1]["failure_category"], result_processor.FAILURE_BASELINE
        )
        self.assertIn("output_tps_avg=50", rows[1]["failure_reason"])
        self.assertIn(
            "output_tps_avg: actual=50.0000",
            rows[1]["baseline_difference"],
        )
        self.assertIn("baseline=100.0000", rows[1]["baseline_difference"])
        self.assertIn("difference=-50.0000", rows[1]["baseline_difference"])
        self.assertIn("deviation=-50.00%", rows[1]["baseline_difference"])
        self.assertIn("below lower bound 90.0000", rows[1]["baseline_difference"])
        self.assertEqual(rows[0]["baseline_difference"], "NA")
        self.assertEqual(rows[2]["baseline_difference"], "NA")
        self.assertEqual(rows[2]["failure_category"], result_processor.FAILURE_OTHER)
        model_summary = next(
            row for row in summary_rows if row["model_name"] == "qwen3_8b"
        )
        self.assertEqual(model_summary["total_runs"], 3)
        self.assertEqual(model_summary["failed"], 3)
        self.assertEqual(model_summary[result_processor.FAILURE_XLLM], 1)
        self.assertEqual(model_summary[result_processor.FAILURE_BASELINE], 1)
        self.assertEqual(model_summary[result_processor.FAILURE_OTHER], 1)

    def test_failure_reason_with_control_characters_is_written(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            source = self._source_case(root)
            config = self._config("case-a")
            failure = self._result(config, success=False)
            failure.update(
                benchmark_success=False,
                benchmark_stderr=(
                    "Saving to: 'gsm8k.zip'\n"
                    "[ais_bench] [\x1b[32mINFO\x1b[0m] Loading dataset\n"
                    "wget progress\x08\x08 garbage"
                ),
                message="Benchmark command failed with exit code 1",
            )
            result_processor.write_result(failure, config, task_path=str(source))
            rows = self._sheet_rows(Path(root) / "outputs" / "result.xlsx", "qwen3_8b")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "fail")
        self.assertEqual(
            rows[0]["failure_category"], result_processor.FAILURE_BENCHMARK
        )
        reason = rows[0]["failure_reason"]
        self.assertIn("Benchmark command failed with exit code 1", reason)
        self.assertIn("[ais_bench] [INFO] Loading dataset", reason)
        self.assertNotIn("\x1b", reason)
        self.assertNotIn("\x08", reason)

    def test_generated_search_case_resolves_original_model_folder(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            self._source_case(root, model="deepseek_v3.2", case_name="search-case")
            generated = (
                Path(root)
                / "outputs"
                / "group"
                / "search-case_sg1"
                / "search-case_sg1.yaml"
            )
            generated.parent.mkdir(parents=True)
            generated.write_text("case_name: search-case_sg1\n", encoding="utf-8")
            config = self._config("search-case_sg1")
            result_processor.write_result(
                self._result(config, perf=FakePerfResult()),
                config,
                task_path=str(generated),
            )
            workbook = load_workbook(Path(root) / "outputs" / "result.xlsx")
            sheet_names = workbook.sheetnames
            workbook.close()

        self.assertIn("deepseek_v3.2", sheet_names)
        self.assertNotIn("search-case_sg1", sheet_names)

    def test_case_group_workbook_uses_timestamp_per_invocation(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ), mock.patch.object(
            result_processor,
            "_current_case_group_timestamp",
            side_effect=("20260730_101112", "20260730_131415"),
        ):
            source = self._source_case(root)
            first_args = SimpleNamespace(
                case_group="daily_group", assess_test_type="none"
            )
            for case_name in ("case-a", "case-b"):
                config = self._config(case_name)
                result_processor.write_result(
                    self._result(config, perf=FakePerfResult()),
                    config,
                    task_path=str(source),
                    args=first_args,
                )

            group_dir = Path(root) / "outputs" / "daily_group"
            first_group_path = group_dir / "daily_group_20260730_101112.xlsx"
            first_rows = self._sheet_rows(first_group_path, "qwen3_8b")

            second_args = SimpleNamespace(
                case_group="daily_group", assess_test_type="none"
            )
            config = self._config("case-c")
            result_processor.write_result(
                self._result(config, perf=FakePerfResult()),
                config,
                task_path=str(source),
                args=second_args,
            )
            second_group_path = group_dir / "daily_group_20260730_131415.xlsx"
            second_rows = self._sheet_rows(second_group_path, "qwen3_8b")
            history_rows = self._sheet_rows(
                Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
            )
            group_filenames = sorted(path.name for path in group_dir.glob("*.xlsx"))
            fixed_group_result_exists = (group_dir / "result.xlsx").exists()

        self.assertEqual([row["case_name"] for row in first_rows], ["case-a", "case-b"])
        self.assertEqual([row["case_name"] for row in second_rows], ["case-c"])
        self.assertEqual(
            group_filenames,
            [
                "daily_group_20260730_101112.xlsx",
                "daily_group_20260730_131415.xlsx",
            ],
        )
        self.assertFalse(fixed_group_result_exists)
        self.assertEqual(len(history_rows), 3)

    def test_output_path_uses_cli_host_and_benchmark_date_directory(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            source = self._source_case(root)
            config = self._config("case-a")
            result = self._result(config, perf=FakePerfResult())
            result["output_file_path"] = str(
                Path(root) / "outputs" / "daily_group" / "case-a"
            )
            result["benchmark_output_path"] = str(
                Path(result["output_file_path"]) / "20260712_195639"
            )
            args = SimpleNamespace(
                case_group="daily_group",
                assess_test_type="none",
                host_addr="11.87.191.99",
            )
            result_processor.write_result(
                result,
                config,
                task_path=str(source),
                args=args,
            )
            rows = self._sheet_rows(
                Path(root) / "outputs" / "result.xlsx", "qwen3_8b"
            )

        self.assertNotIn("host_addr", rows[0])
        self.assertEqual(
            rows[0]["output_path"],
            "11.87.191.99/daily_group/case-a/20260712_195639",
        )

    def test_existing_workbook_columns_are_pruned_and_reordered(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            result_processor.os, "getcwd", return_value=root
        ):
            source = self._source_case(root)
            config = self._config("case-a")
            result_processor.write_result(
                self._result(config, perf=FakePerfResult()),
                config,
                task_path=str(source),
            )
            workbook_path = Path(root) / "outputs" / "result.xlsx"
            workbook = load_workbook(workbook_path)
            worksheet = workbook["qwen3_8b"]
            worksheet.insert_cols(1)
            worksheet.cell(1, 1, "source_yaml")
            worksheet.cell(2, 1, "/legacy/case.yaml")
            worksheet.cell(1, worksheet.max_column + 1, "metadata.total_rows")
            worksheet.cell(2, worksheet.max_column, 8)
            worksheet.cell(1, worksheet.max_column + 1, "metadata.max_concurrency")
            worksheet.cell(2, worksheet.max_column, 4)
            worksheet.cell(1, worksheet.max_column + 1, "summary.ttft_p90")
            worksheet.cell(2, worksheet.max_column, 120.0)
            worksheet.cell(1, worksheet.max_column + 1, "summary_requests")
            worksheet.cell(2, worksheet.max_column, 9)
            worksheet.cell(
                1,
                worksheet.max_column + 1,
                "summary.output_token_throughput",
            )
            worksheet.cell(2, worksheet.max_column, 77.0)
            worksheet.cell(
                1,
                worksheet.max_column + 1,
                "parameters.ITL.total.average",
            )
            worksheet.cell(2, worksheet.max_column, 8.8)
            worksheet.cell(
                1,
                worksheet.max_column + 1,
                "parameters.TPOT.total.p90",
            )
            worksheet.cell(2, worksheet.max_column, 9.1)
            worksheet.cell(
                1,
                worksheet.max_column + 1,
                "parameters.OutputTokenThroughput.total.n",
            )
            worksheet.cell(2, worksheet.max_column, 8)
            headers = [cell.value for cell in worksheet[1]]
            worksheet.cell(
                2,
                headers.index("finished_at") + 1,
                "2026-07-28T11:51:26+00:00",
            )
            for field_name in (
                "failure_category",
                "failure_reason",
                "baseline_difference",
            ):
                worksheet.cell(2, headers.index(field_name) + 1, "")
            workbook.save(workbook_path)
            workbook.close()

            second_config = self._config("case-b")
            result_processor.write_result(
                self._result(second_config, perf=FakePerfResult()),
                second_config,
                task_path=str(source),
            )
            headers = self._sheet_headers(workbook_path, "qwen3_8b")
            rows = self._sheet_rows(workbook_path, "qwen3_8b")

        self.assertEqual(headers[:6], list(LEADING_RESULT_FIELDS))
        self.assertTrue(EXCLUDED_RESULT_FIELDS.isdisjoint(headers))
        self.assertFalse(any(header.startswith("metadata.") for header in headers))
        self.assertFalse(any(header == "summary.ttft_p90" for header in headers))
        self.assertNotIn("summary_requests", headers)
        self.assertNotIn("summary.output_token_throughput", headers)
        self.assertEqual(rows[0]["summary.total_requests"], 9)
        self.assertEqual(rows[0]["summary.output_tps_avg"], 77.0)
        self.assertFalse(any(header.endswith(".n") for header in headers))
        self.assertFalse(
            any(header.startswith("parameters.ITL.") for header in headers)
        )
        self.assertFalse(any("parameters." in header for header in headers))
        self.assertFalse(any("total" in header.split(".") for header in headers))
        self.assertEqual(rows[0]["TPOT.p90"], 9.1)
        self.assertEqual(rows[0]["failure_category"], "NA")
        self.assertEqual(rows[0]["failure_reason"], "NA")
        self.assertEqual(rows[0]["baseline_difference"], "NA")
        self.assertEqual(rows[0]["finished_at"], "2026-07-28T19:51:26+08:00")

    def test_workbook_failure_is_reported_without_raising(self):
        with mock.patch.object(
            result_processor, "append_result_workbook", return_value=False
        ):
            written = result_processor.write_result(
                {"success": True, "execution_time": 1},
                {"case_name": "case-one"},
            )
        self.assertFalse(written)


if __name__ == "__main__":
    unittest.main()
