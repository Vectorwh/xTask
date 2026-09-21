import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.tasks.execution_workflow import TaskExecutionWorkflow


class TaskExecutionWorkflowTests(unittest.TestCase):
    def test_auto_assess_case_group_uses_independent_sessions_and_state(self):
        tasks = {
            "/tmp/group/case_a_bs-1.yaml": "/tmp/group/case_a_bs-1",
            "/tmp/group/case_b_bs-2.yaml": "/tmp/group/case_b_bs-2",
        }
        executor_calls = []

        def execute(task_key, output_path, args, session):
            executor_calls.append((task_key, session))
            return {"task_config": {"metrics": []}}, f"session-{len(executor_calls)}"

        observed_states = []

        def assess(**kwargs):
            observed_states.append(
                (dict(kwargs["config_para"]), dict(kwargs["auto_assess_threshold"]))
            )
            kwargs["config_para"]["batch_size"] = 99
            kwargs["auto_assess_threshold"]["lower"] = 99
            return "break"

        reporter = mock.Mock()
        closer = mock.Mock()
        workflow = TaskExecutionWorkflow(
            args=SimpleNamespace(
                assess_test_type="auto_assess",
                target_ttft=1000,
                target_tpot=12,
            ),
            xtask_config={},
            openme_client=mock.Mock(),
            para_strategy=None,
            task_loader=lambda *args: tasks,
            task_executor=execute,
            auto_assessor=assess,
            reporter_factory=lambda *args, **kwargs: reporter,
            session_closer=closer,
        )

        workflow.run()

        self.assertEqual(
            executor_calls,
            [
                ("/tmp/group/case_a_bs-1.yaml", None),
                ("/tmp/group/case_b_bs-2.yaml", None),
            ],
        )
        self.assertEqual(
            observed_states,
            [
                ({}, {"lower": 0, "curr": 0, "upper": 0}),
                ({}, {"lower": 0, "curr": 0, "upper": 0}),
            ],
        )
        self.assertEqual(
            closer.call_args_list,
            [mock.call("session-1"), mock.call("session-2")],
        )
        self.assertEqual(reporter.record_result.call_count, 2)


def _make_workflow(args, tasks, execute, assess=lambda **kwargs: None):
    return TaskExecutionWorkflow(
        args=args,
        xtask_config={},
        openme_client=mock.Mock(),
        para_strategy=None,
        task_loader=lambda *loader_args: tasks,
        task_executor=execute,
        auto_assessor=assess,
        reporter_factory=lambda *args, **kwargs: None,
        session_closer=mock.Mock(),
    )


class TaskExecutionWorkflowStatusTests(unittest.TestCase):
    def test_normal_failed_task_returns_false(self):
        workflow = _make_workflow(
            args=SimpleNamespace(assess_test_type="normal"),
            tasks={"/tmp/case.yaml": "/tmp/case"},
            execute=lambda *args: ({"success": False}, None),
        )

        self.assertFalse(workflow.run())

    def test_normal_successful_task_returns_true(self):
        workflow = _make_workflow(
            args=SimpleNamespace(assess_test_type="normal"),
            tasks={"/tmp/case.yaml": "/tmp/case"},
            execute=lambda *args: ({"success": True}, None),
        )

        self.assertTrue(workflow.run())

    def test_result_without_success_field_keeps_legacy_success(self):
        workflow = _make_workflow(
            args=SimpleNamespace(assess_test_type="normal"),
            tasks={"/tmp/case.yaml": "/tmp/case"},
            execute=lambda *args: ({"task_config": {}}, None),
        )

        self.assertTrue(workflow.run())

    def test_executor_exception_returns_false(self):
        def execute(*args):
            raise RuntimeError("boom")

        workflow = _make_workflow(
            args=SimpleNamespace(assess_test_type="normal"),
            tasks={"/tmp/case.yaml": "/tmp/case"},
            execute=execute,
        )

        with self.assertLogs(level="ERROR"):
            self.assertFalse(workflow.run())

    def test_auto_assess_uses_latest_result_for_base_case(self):
        tasks = {
            "/tmp/case_bs-1.yaml": "/tmp/case_bs-1",
            "/tmp/case_bs-2.yaml": "/tmp/case_bs-2",
        }
        results = iter([{"success": False}, {"success": True}])

        def execute(*args):
            return next(results), None

        workflow = _make_workflow(
            args=SimpleNamespace(assess_test_type="auto_assess"),
            tasks=tasks,
            execute=execute,
        )

        self.assertTrue(workflow.run())


if __name__ == "__main__":
    unittest.main()
