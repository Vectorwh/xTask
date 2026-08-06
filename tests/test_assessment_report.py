import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from xtask.tasks.assessment_report import AssessmentReporter


def _perf_result(throughput, ttft, tpot, average_throughput=None):
    return SimpleNamespace(
        summary=SimpleNamespace(
            output_tps_total=throughput,
            output_tps_avg=(
                throughput if average_throughput is None else average_throughput
            ),
            ttft_p90=ttft,
            tpot_p90=tpot,
        )
    )


def _write_exhaustive_trial(root, base_name, index, batch_size, search_config):
    case_name = f"{base_name}_sg{index}"
    case_dir = root / case_name
    case_dir.mkdir(parents=True)
    config = {
        "case_name": case_name,
        "metrics": "perf",
        "benchmark": {"perf": {"batch_size": batch_size}},
        "assess_search": search_config,
    }
    task_file = case_dir / f"{case_name}.yaml"
    task_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    (case_dir / "assess_test_param.yaml").write_text(
        yaml.safe_dump(
            {
                "current_combination": {
                    "benchmark.perf.batch_size": batch_size,
                },
                "assess_test_param": {
                    "benchmark.perf.batch_size": [1, 2],
                },
            }
        ),
        encoding="utf-8",
    )
    return task_file, case_dir, config


class AssessmentReportTests(unittest.TestCase):
    def test_exhaustive_report_selects_best_feasible_trial(self):
        search_config = {
            "strategy": "exhaustive",
            "objective": {
                "metric": "output_tps_total",
                "direction": "maximize",
            },
            "constraints": {"tpot_p90": {"max": 50}},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "outputs"
            first = _write_exhaustive_trial(
                output_root, "search_case", 1, 1, search_config
            )
            second = _write_exhaustive_trial(
                output_root, "search_case", 2, 2, search_config
            )
            tasks = {str(first[0]): str(first[1]), str(second[0]): str(second[1])}
            args = SimpleNamespace(assess_test_type="para_set")
            reporter = AssessmentReporter(
                args,
                tasks,
                strategy="exhaustive",
                output_root=output_root,
            )

            reporter.record_result(
                str(first[0]),
                str(first[1]),
                {
                    "success": True,
                    "task_config": first[2],
                    "benchmark_perf_result": _perf_result(100, 900, 40),
                },
            )
            reporter.record_result(
                str(second[0]),
                str(second[1]),
                {
                    "success": True,
                    "task_config": second[2],
                    "benchmark_perf_result": _perf_result(200, 800, 60),
                },
            )
            reporter.finalize()

            report_dir = output_root / "search" / "search_case"
            summary = yaml.safe_load(
                (report_dir / "search_summary.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["planned_trials"], 2)
            self.assertEqual(summary["executed_trials"], 2)
            self.assertEqual(summary["constraint_failed_trials"], 1)
            self.assertEqual(summary["best_trial"]["case_name"], "search_case_sg1")
            self.assertEqual(summary["best_trial"]["objective_value"], 100.0)

            with open(
                report_dir / "search_trials.csv", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["batch_size"], "1")
            self.assertEqual(rows[1]["status"], "constraint_failed")
            self.assertEqual(rows[1]["output_tps_total"], "200.0")

    def test_exhaustive_report_without_objective_uses_total_throughput(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs"
            first = _write_exhaustive_trial(
                output_root,
                "search_case",
                1,
                1,
                {"strategy": "exhaustive"},
            )
            second = _write_exhaustive_trial(
                output_root,
                "search_case",
                2,
                2,
                {"strategy": "exhaustive"},
            )
            reporter = AssessmentReporter(
                SimpleNamespace(assess_test_type="para_set"),
                {str(first[0]): str(first[1]), str(second[0]): str(second[1])},
                strategy="exhaustive",
                output_root=output_root,
            )
            for trial, total, average in (
                (first, 100, 200),
                (second, 150, 50),
            ):
                reporter.record_result(
                    str(trial[0]),
                    str(trial[1]),
                    {
                        "success": True,
                        "task_config": trial[2],
                        "benchmark_perf_result": _perf_result(
                            total, 900, 40, average
                        ),
                    },
                )
            reporter.finalize()

            summary = yaml.safe_load(
                (
                    output_root
                    / "search"
                    / "search_case"
                    / "search_summary.yaml"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["objective"],
                {"metric": "output_tps_total", "direction": "maximize"},
            )
            self.assertEqual(
                summary["best_trial"]["case_name"], "search_case_sg2"
            )
            self.assertEqual(summary["best_trial"]["objective_value"], 150.0)

    def test_auto_assess_report_selects_largest_feasible_batch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs"
            args = SimpleNamespace(
                assess_test_type="auto_assess",
                target_ttft=1000,
                target_tpot=50,
            )
            reporter = AssessmentReporter(
                args,
                {},
                strategy="auto_assess",
                output_root=output_root,
            )

            for index, batch_size, ttft, tpot in (
                (1, 4, 900, 40),
                (2, 8, 1200, 45),
            ):
                case_name = f"auto_case_bs-{batch_size}"
                config = {
                    "case_name": case_name,
                    "metrics": "perf",
                    "benchmark": {"perf": {"batch_size": batch_size}},
                }
                reporter.record_result(
                    f"{case_name}.yaml",
                    str(output_root / case_name),
                    {
                        "success": True,
                        "task_config": config,
                        "benchmark_perf_result": _perf_result(
                            100 + index, ttft, tpot
                        ),
                    },
                    search_state={
                        "lower": 0,
                        "curr": batch_size,
                        "upper": 16,
                    },
                )
            reporter.finalize("interval_converged")

            report_dir = output_root / "search" / "auto_case"
            summary = yaml.safe_load(
                (report_dir / "search_summary.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["strategy"], "auto_assess")
            self.assertEqual(summary["stop_reason"], "interval_converged")
            self.assertEqual(summary["selected_batch_size"], 4)
            self.assertEqual(summary["best_trial"]["case_name"], "auto_case_bs-4")
            self.assertEqual(summary["constraint_failed_trials"], 1)
            self.assertEqual(
                summary["trials"][1]["search_bounds"],
                {"lower": 0, "curr": 8, "upper": 16},
            )

    def test_auto_assess_marks_scheduler_error_trial_failed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs"
            reporter = AssessmentReporter(
                SimpleNamespace(
                    assess_test_type="auto_assess",
                    target_ttft=1000,
                    target_tpot=50,
                ),
                {},
                strategy="auto_assess",
                output_root=output_root,
            )
            config = {
                "case_name": "auto_case_bs-4",
                "metrics": "perf",
                "benchmark": {"perf": {"batch_size": 4}},
            }

            reporter.record_result(
                "auto_case_bs-4.yaml",
                str(output_root / "auto_case_bs-4"),
                {
                    "success": False,
                    "scheduler_error": True,
                    "message": "archive permission denied",
                    "task_config": config,
                    "benchmark_perf_result": _perf_result(100, 900, 40),
                },
                search_state={"lower": 0, "curr": 4, "upper": 8},
                action="break",
            )
            reporter.finalize("scheduler_error")

            summary = yaml.safe_load(
                (
                    output_root
                    / "search"
                    / "auto_case"
                    / "search_summary.yaml"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["failed_trials"], 1)
            self.assertIsNone(summary["best_trial"])
            self.assertEqual(summary["trials"][0]["status"], "failed")
            self.assertEqual(
                summary["trials"][0]["error"], "archive permission denied"
            )


if __name__ == "__main__":
    unittest.main()
