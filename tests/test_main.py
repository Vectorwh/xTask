import unittest
from types import SimpleNamespace
from unittest import mock

import main


def _make_args(assess_test_type=None, host_addr=None):
    kwargs = {}
    if assess_test_type is not None:
        kwargs["assess_test_type"] = assess_test_type
    if host_addr is not None:
        kwargs["host_addr"] = host_addr
    return SimpleNamespace(**kwargs)


class RunTaskTest(unittest.TestCase):
    """Tests for main.run_task lifecycle branches."""

    @mock.patch.object(main, "run_scheduler")
    def test_non_auto_assess_uses_run_scheduler(self, mock_run_scheduler):
        mock_run_scheduler.return_value = "scheduler_result"
        args = _make_args(assess_test_type="para_set")

        result, session = main.run_task("task_key", "/tmp/out", args)

        mock_run_scheduler.assert_called_once_with(
            "task_key", "/tmp/out", args
        )
        self.assertEqual(result, "scheduler_result")
        self.assertIsNone(session)

    @mock.patch.object(main, "run_scheduler_session_trial")
    @mock.patch.object(main, "start_scheduler_session")
    def test_auto_assess_starts_session_when_missing(
        self, mock_start, mock_trial
    ):
        mock_start.return_value = "new_session"
        mock_trial.return_value = "trial_result"
        args = _make_args(assess_test_type="auto_assess")

        result, session = main.run_task("task_key", "/tmp/out", args)

        mock_start.assert_called_once_with("task_key", "/tmp/out", args)
        mock_trial.assert_called_once_with(
            "new_session", "task_key", "/tmp/out"
        )
        self.assertEqual(result, "trial_result")
        self.assertEqual(session, "new_session")

    @mock.patch.object(main, "run_scheduler_session_trial")
    @mock.patch.object(main, "start_scheduler_session")
    def test_auto_assess_reuses_existing_session(self, mock_start, mock_trial):
        mock_trial.return_value = "trial_result"
        args = _make_args(assess_test_type="auto_assess")

        result, session = main.run_task(
            "task_key", "/tmp/out", args, scheduler_session="existing"
        )

        mock_start.assert_not_called()
        mock_trial.assert_called_once_with(
            "existing", "task_key", "/tmp/out"
        )
        self.assertEqual(result, "trial_result")
        self.assertEqual(session, "existing")


class RunTaskWorkflowTest(unittest.TestCase):
    """Tests for main.run_task_workflow wiring."""

    @mock.patch.object(main, "TaskExecutionWorkflow")
    def test_builds_workflow_and_runs(self, mock_workflow_cls):
        workflow = mock_workflow_cls.return_value
        workflow.run.return_value = "workflow_result"
        args = _make_args()
        client = object()

        result = main.run_task_workflow(
            args, {"cfg": 1}, client, para_strategy="grid"
        )

        mock_workflow_cls.assert_called_once()
        kwargs = mock_workflow_cls.call_args.kwargs
        self.assertIs(kwargs["args"], args)
        self.assertEqual(kwargs["xtask_config"], {"cfg": 1})
        self.assertIs(kwargs["openme_client"], client)
        self.assertEqual(kwargs["para_strategy"], "grid")
        self.assertIs(kwargs["task_executor"], main.run_task)
        workflow.run.assert_called_once_with()
        self.assertEqual(result, "workflow_result")


class RunValidatedWorkflowTest(unittest.TestCase):
    """Tests for the dispatch logic in main.run_validated_workflow."""

    def setUp(self):
        patcher_client = mock.patch.object(main, "OpenMeClient")
        patcher_resolve = mock.patch.object(
            main, "resolve_para_set_strategy"
        )
        patcher_tpe = mock.patch.object(main, "run_tpe_parameter_search")
        patcher_workflow = mock.patch.object(main, "run_task_workflow")
        self.mock_client_cls = patcher_client.start()
        self.mock_resolve = patcher_resolve.start()
        self.mock_tpe = patcher_tpe.start()
        self.mock_workflow = patcher_workflow.start()
        for patcher in (
            patcher_client,
            patcher_resolve,
            patcher_tpe,
            patcher_workflow,
        ):
            self.addCleanup(patcher.stop)
        self.client = self.mock_client_cls.return_value

    def test_default_host_addr_when_missing(self):
        args = _make_args()

        main.run_validated_workflow(args, {"cfg": 1})

        self.mock_client_cls.assert_called_once_with(
            "main", "A3-ARM", "0.0.0.0"
        )

    def test_custom_host_addr_is_used(self):
        args = _make_args(host_addr="10.0.0.5")

        main.run_validated_workflow(args, {"cfg": 1})

        self.mock_client_cls.assert_called_once_with(
            "main", "A3-ARM", "10.0.0.5"
        )

    def test_non_para_set_runs_task_workflow_without_strategy(self):
        args = _make_args(assess_test_type="normal")
        self.mock_workflow.return_value = False

        result = main.run_validated_workflow(args, {"cfg": 1})

        self.assertFalse(result)
        self.mock_resolve.assert_not_called()
        self.mock_tpe.assert_not_called()
        self.mock_workflow.assert_called_once_with(
            args, {"cfg": 1}, self.client, para_strategy=None
        )
        self.client.SendJUEMsgCard.assert_called_once_with(args=args)

    def test_para_set_tpe_dispatches_to_tpe_search(self):
        self.mock_resolve.return_value = "tpe"
        args = _make_args(assess_test_type="para_set")

        main.run_validated_workflow(args, {"cfg": 1})

        self.mock_resolve.assert_called_once_with(args)
        self.mock_tpe.assert_called_once_with(
            args=args,
            openme_client=self.client,
            run_trial=main.run_scheduler,
        )
        self.mock_workflow.assert_not_called()
        self.client.SendJUEMsgCard.assert_not_called()

    def test_para_set_non_tpe_runs_task_workflow_with_strategy(self):
        self.mock_resolve.return_value = "grid"
        args = _make_args(assess_test_type="para_set")

        main.run_validated_workflow(args, {"cfg": 1})

        self.mock_resolve.assert_called_once_with(args)
        self.mock_tpe.assert_not_called()
        self.mock_workflow.assert_called_once_with(
            args, {"cfg": 1}, self.client, para_strategy="grid"
        )
        self.client.SendJUEMsgCard.assert_called_once_with(args=args)


class ParseAndValidateArgumentsTest(unittest.TestCase):
    """Tests for main.parse_and_validate_arguments."""

    @mock.patch.object(main, "get_xtask_config_data")
    @mock.patch.object(main, "ArgumentChecker")
    def test_returns_args_result_and_config(
        self, mock_checker_cls, mock_get_config
    ):
        checker = mock_checker_cls.return_value
        checker.parse_arguments.return_value = "parsed_args"
        checker.validate_arguments.return_value = "validation"
        mock_get_config.return_value = {"cfg": 1}

        args, result, config = main.parse_and_validate_arguments()

        checker.parse_arguments.assert_called_once_with()
        checker.validate_arguments.assert_called_once_with("parsed_args")
        checker.print_validation_result.assert_called_once_with("validation")
        self.assertEqual(args, "parsed_args")
        self.assertEqual(result, "validation")
        self.assertEqual(config, {"cfg": 1})


class MainEntryTest(unittest.TestCase):
    """Tests for the main.main CLI entry point."""

    @mock.patch.object(main, "run_validated_workflow")
    @mock.patch.object(main, "parse_and_validate_arguments")
    @mock.patch.object(main, "init_logger")
    def test_runs_workflow_when_arguments_valid(
        self, mock_init, mock_parse, mock_run
    ):
        args = _make_args()
        mock_parse.return_value = (
            args,
            SimpleNamespace(is_valid=True),
            {"cfg": 1},
        )

        main.main()

        mock_init.assert_called_once_with()
        mock_run.assert_called_once_with(args, {"cfg": 1})

    @mock.patch.object(main, "run_validated_workflow")
    @mock.patch.object(main, "parse_and_validate_arguments")
    @mock.patch.object(main, "init_logger")
    def test_exits_when_arguments_invalid(
        self, mock_init, mock_parse, mock_run
    ):
        mock_parse.return_value = (
            _make_args(),
            SimpleNamespace(is_valid=False),
            {"cfg": 1},
        )

        with self.assertRaises(SystemExit) as ctx:
            main.main()

        self.assertEqual(ctx.exception.code, 1)
        mock_run.assert_not_called()

    @mock.patch.object(main, "run_validated_workflow")
    @mock.patch.object(main, "parse_and_validate_arguments")
    @mock.patch.object(main, "init_logger")
    def test_exits_when_workflow_fails(
        self, mock_init, mock_parse, mock_run
    ):
        mock_parse.return_value = (
            _make_args(),
            SimpleNamespace(is_valid=True),
            {"cfg": 1},
        )
        mock_run.return_value = False

        with self.assertRaises(SystemExit) as ctx:
            main.main()

        self.assertEqual(ctx.exception.code, 1)
        mock_run.assert_called_once()

    @mock.patch.object(main, "run_validated_workflow")
    @mock.patch.object(main, "parse_and_validate_arguments")
    @mock.patch.object(main, "init_logger")
    def test_does_not_exit_when_workflow_succeeds(
        self, mock_init, mock_parse, mock_run
    ):
        mock_parse.return_value = (
            _make_args(),
            SimpleNamespace(is_valid=True),
            {"cfg": 1},
        )
        mock_run.return_value = True

        self.assertIsNone(main.main())
        mock_run.assert_called_once()

    @mock.patch.object(main, "parse_and_validate_arguments")
    @mock.patch.object(main, "init_logger")
    def test_exits_on_unexpected_exception(self, mock_init, mock_parse):
        mock_parse.side_effect = RuntimeError("boom")

        with self.assertRaises(SystemExit) as ctx:
            main.main()

        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
