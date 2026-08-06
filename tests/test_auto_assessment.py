import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.tasks.auto_assessment import process_auto_assessment


def _result(batch_size, ttft_p90, tpot_p90):
    return {
        "task_config": {
            "metrics": ["perf"],
            "benchmark": {
                "perf": {
                    "batch_size": batch_size,
                    "bs_assess": {"low_thrld": 0, "high_thrld": 6},
                }
            },
        },
        "benchmark_perf_result": SimpleNamespace(
            summary=SimpleNamespace(
                ttft_p90=ttft_p90,
                tpot_p90=tpot_p90,
            )
        ),
    }


class AutoAssessmentTests(unittest.TestCase):
    def test_feasible_trial_adds_higher_batch_task(self):
        args = SimpleNamespace(assess_test_type="auto_assess")
        config_para = {}
        thresholds = {"lower": 0, "curr": 0, "upper": 0}
        tasks = {"case-bs3.yaml": "output-bs3"}
        keys = ["case-bs3.yaml"]

        with mock.patch(
            "xtask.tasks.auto_assessment.get_tasks",
            return_value={"case-bs4.yaml": "output-bs4"},
        ) as get_tasks_mock:
            action = process_auto_assessment(
                args,
                _result(3, 100, 10),
                config_para,
                thresholds,
                1000,
                12,
                tasks,
                keys,
                {"config": True},
            )

        self.assertIsNone(action)
        self.assertEqual(config_para, {"batch_size": 4})
        self.assertEqual(
            thresholds, {"lower": 3, "curr": 4, "upper": 6}
        )
        self.assertEqual(keys, ["case-bs3.yaml", "case-bs4.yaml"])
        get_tasks_mock.assert_called_once_with(
            args, {"config": True}, {"batch_size": 4}
        )

    def test_trial_near_tpot_target_stops_without_new_task(self):
        thresholds = {"lower": 0, "curr": 0, "upper": 0}

        with mock.patch(
            "xtask.tasks.auto_assessment.get_tasks"
        ) as get_tasks_mock:
            action = process_auto_assessment(
                SimpleNamespace(assess_test_type="auto_assess"),
                _result(3, 100, 11.5),
                {},
                thresholds,
                1000,
                12,
                {},
                [],
                {},
            )

        self.assertEqual(action, "break")
        self.assertEqual(thresholds["stop_reason"], "tpot_target_reached")
        get_tasks_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
