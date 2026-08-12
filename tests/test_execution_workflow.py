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


if __name__ == "__main__":
    unittest.main()
