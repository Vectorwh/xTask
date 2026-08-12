import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.core.task_scheduler import (
    SchedulerDependencies,
    TaskExecutionContext,
    TaskScheduler,
)


def _dependencies(events, failing=None):
    def stage(name):
        def run(context):
            events.append(name)
            if name == failing:
                raise RuntimeError("boom")
        return run

    return SchedulerDependencies(
        load_task_and_engine=stage("load"),
        prepare_container=stage("container"),
        prepare_runtime_dependencies=stage("dependencies"),
        start_and_wait_service=stage("service"),
        run_benchmark_and_process=stage("benchmark"),
        finalize_result=stage("finalize"),
        cleanup=stage("cleanup"),
    )


class TaskSchedulerTests(unittest.TestCase):
    def test_task_config_load_normalizes_hierarchical_pd_instances(self):
        from xtask.core.scheduler import TaskConfig

        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "pd.yaml"
            task_path.write_text(
                """
deployment:
  mode: pd_disaggregated
  pd_instances:
    - name: prefill-0
      role: prefill
      nodes:
        - host: 10.0.0.1
          device_ids: [0, 1]
        - host: 10.0.0.2
          device_ids: [0, 1]
    - name: decode-0
      role: decode
      nodes:
        - host: 10.0.0.3
          device_ids: [0, 1]
""",
                encoding="utf-8",
            )
            config = TaskConfig.load_task_config(str(task_path))

        self.assertNotIn("pd_instances", config["deployment"])
        self.assertEqual(len(config["deployment"]["instances"]), 3)
        self.assertEqual(
            [item["pd_group"] for item in config["deployment"]["instances"]],
            ["prefill-0", "prefill-0", "decode-0"],
        )

    def test_task_config_load_rejects_flat_pd_authoring(self):
        from xtask.core.scheduler import TaskConfig

        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "flat-pd.yaml"
            task_path.write_text(
                """
deployment:
  mode: pd_disaggregated
  instances:
    - name: prefill-0
      role: prefill
      host: 10.0.0.1
    - name: decode-0
      role: decode
      host: 10.0.0.2
""",
                encoding="utf-8",
            )
            config = TaskConfig.load_task_config(str(task_path))

        self.assertEqual(config, {})

    def test_prepare_reuses_service_for_multiple_trials_and_cleans_once(self):
        events = []
        context = TaskExecutionContext("task.yaml", "output", result={})
        scheduler = TaskScheduler(_dependencies(events))

        scheduler.prepare(context)
        scheduler.run_trial(context)
        scheduler.finalize_result(context)
        scheduler.run_trial(context)
        scheduler.finalize_result(context)
        scheduler.cleanup(context)

        self.assertEqual(
            events,
            [
                "load",
                "container",
                "dependencies",
                "service",
                "benchmark",
                "finalize",
                "benchmark",
                "finalize",
                "cleanup",
            ],
        )

    def test_runs_real_stage_contract_in_order_and_cleans_up(self):
        events = []
        context = TaskExecutionContext("task.yaml", "output", result={"success": True})

        result = TaskScheduler(_dependencies(events)).run(context)

        self.assertEqual(
            events,
            ["load", "container", "dependencies", "service", "benchmark", "cleanup", "finalize"],
        )
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["execution_time"], 0)

    def test_stops_on_failure_and_still_cleans_up(self):
        events = []
        context = TaskExecutionContext("task.yaml", "output", result={})

        result = TaskScheduler(_dependencies(events, failing="container")).run(context)

        self.assertEqual(events, ["load", "container", "cleanup", "finalize"])
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Error in scheduler: boom")

    def test_real_load_stage_passes_state_to_container_and_early_failure_cleans_once(self):
        from xtask.core import scheduler

        task = {"served_model_name": "xllm", "benchmark": {"benchmark_type": "evalscope"}}
        engine_config = {"engine_type": "xllm", "serving_container_name": ""}
        fake_manager = mock.Mock()
        fake_manager.check_runtime_available.return_value = False
        fake_config = SimpleNamespace(runtime="docker", image_home="home", image_name="image", container_name="case")
        cleanup_calls = []
        dependencies = SchedulerDependencies(
            load_task_and_engine=scheduler._load_task_and_engine,
            prepare_container=scheduler._prepare_container,
            prepare_runtime_dependencies=mock.Mock(),
            start_and_wait_service=mock.Mock(),
            run_benchmark_and_process=mock.Mock(),
            finalize_result=mock.Mock(),
            cleanup=lambda context: cleanup_calls.append(context.state.get("container_manager")),
        )
        context = TaskExecutionContext("task.yaml", "out", result=scheduler._initial_result())

        with mock.patch.object(scheduler.TaskConfig, "load_task_config", return_value=task), \
             mock.patch.object(scheduler.Json_Reader, "load_config", return_value={"XLLM_CONFIG": {}}), \
             mock.patch.object(scheduler.Json_Reader, "get_container_config", return_value=engine_config), \
             mock.patch.object(scheduler, "generate_commands", return_value=("start", "bench")), \
             mock.patch.object(scheduler.ContainerConfigBuilder, "create_container_config", return_value=fake_config), \
             mock.patch.object(scheduler, "ContainerManager", return_value=fake_manager):
            result = TaskScheduler(dependencies).run(context)

        self.assertIs(context.state["task_config"], task)
        self.assertIs(context.state["engine_config"], engine_config)
        self.assertEqual(context.state["startup_cmd"], "start")
        self.assertIs(context.state["container_manager"], fake_manager)
        self.assertEqual(result["message"], "Container runtime docker not available")
        self.assertEqual(cleanup_calls, [fake_manager])
        dependencies.prepare_runtime_dependencies.assert_not_called()
        dependencies.finalize_result.assert_called_once_with(context)

    def test_multi_host_npu_guards_execute_inside_each_instance_container(self):
        from xtask.core import scheduler

        benchmark_manager = mock.Mock()
        benchmark_result = SimpleNamespace(
            success=True,
            exit_code=0,
            stdout="benchmark-ok",
            stderr="",
            execution_time=1.0,
        )
        benchmark_manager.exec_in_container.return_value = benchmark_result

        runtimes = []
        for instance_name, container_name, device_ids in (
            ("prefill-0", "prefill-container", [0, 1]),
            ("decode-0", "decode-container", [2, 3]),
        ):
            manager = mock.Mock()
            manager.exec_in_container.return_value = SimpleNamespace(
                success=True,
                exit_code=0,
                stdout="Process id:1 Process name:xllm Process memory(MB):1",
                stderr="",
            )
            runtimes.append(
                SimpleNamespace(
                    plan=SimpleNamespace(
                        instance_name=instance_name,
                        device_ids=device_ids,
                    ),
                    container_config=SimpleNamespace(container_name=container_name),
                    container_manager=manager,
                )
            )

        created_guards = []

        class FakeGuard:
            def __init__(self, **kwargs):
                self.command_executor = kwargs.get("command_executor")
                self.check_interval = kwargs.get("check_interval")
                self.aborted_reason = None
                created_guards.append(self)

            def start(self):
                self.command_executor("npu-smi info -t proc-mem -i 0 -c 0")

            def stop(self):
                pass

        with mock.patch(
            "xtask.utils.watchdog.NpuProcessGuard", FakeGuard
        ):
            result = scheduler._execute_benchmark_with_npu_guard(
                benchmark_manager,
                SimpleNamespace(container_name="prefill-container"),
                {},
                {"benchmark_timeout": 60},
                "run-benchmark",
                service_runtimes=runtimes,
            )

        self.assertIs(result, benchmark_result)
        self.assertEqual(len(created_guards), 2)
        self.assertTrue(
            all(guard.check_interval == 180.0 for guard in created_guards)
        )
        for runtime in runtimes:
            runtime.container_manager.exec_in_container.assert_called_once_with(
                runtime.container_config.container_name,
                "npu-smi info -t proc-mem -i 0 -c 0",
                timeout=10,
                capture_output=True,
            )
            runtime.container_manager.executor.run.assert_not_called()
        benchmark_manager.exec_in_container.assert_called_once()

    def test_multi_host_npu_guard_interval_can_be_overridden(self):
        from xtask.core import scheduler

        benchmark_manager = mock.Mock()
        benchmark_manager.exec_in_container.return_value = SimpleNamespace(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            execution_time=0.1,
        )
        runtime = SimpleNamespace(
            plan=SimpleNamespace(instance_name="master-0", device_ids=[0]),
            container_config=SimpleNamespace(container_name="master-container"),
            container_manager=mock.Mock(),
        )
        intervals = []

        class FakeGuard:
            aborted_reason = None

            def __init__(self, **kwargs):
                intervals.append(kwargs["check_interval"])

            def start(self):
                pass

            def stop(self):
                pass

        with mock.patch(
            "xtask.utils.watchdog.NpuProcessGuard", FakeGuard
        ):
            scheduler._execute_benchmark_with_npu_guard(
                benchmark_manager,
                SimpleNamespace(container_name="master-container"),
                {},
                {"benchmark_timeout": 60, "npu_guard_interval": 45},
                "run-benchmark",
                service_runtimes=[runtime],
            )

        self.assertEqual(intervals, [45])

    def test_pd_group_worker_readiness_checks_process_only(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(success=True)
        runtime = SimpleNamespace(
            plan=SimpleNamespace(
                role="decode",
                host="10.0.0.2",
                is_local=False,
                output_dir="/tmp/decode-worker",
                readiness={},
                instance_config={
                    "enable_disagg_pd": True,
                    "master_host": "10.0.0.1",
                },
            ),
            container_config=SimpleNamespace(container_name="decode-worker"),
            container_manager=manager,
        )

        self.assertTrue(scheduler._wait_instance_ready(runtime, 1, "xllm"))
        manager.exec_in_container.assert_called_once_with(
            "decode-worker",
            "pgrep -f '[x]llm'",
            timeout=30,
        )

    def test_pd_group_master_readiness_waits_for_registration(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(success=True)
        runtime = SimpleNamespace(
            plan=SimpleNamespace(
                role="decode",
                host="10.0.0.1",
                is_local=False,
                output_dir="/tmp/decode-master",
                readiness={},
                instance_config={
                    "enable_disagg_pd": True,
                    "master_host": "10.0.0.1",
                },
            ),
            container_config=SimpleNamespace(container_name="decode-master"),
            container_manager=manager,
        )

        self.assertTrue(scheduler._wait_instance_ready(runtime, 1, "xllm"))
        manager.exec_in_container.assert_called_once()
        container_name, command = manager.exec_in_container.call_args.args
        self.assertEqual(container_name, "decode-master")
        self.assertIn("Success register instance to etcd", command)
        self.assertIn("node_0.log", command)
        self.assertEqual(manager.exec_in_container.call_args.kwargs["timeout"], 350)

    def test_regular_multi_host_worker_readiness_waits_for_worker_log(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
        )
        runtime = SimpleNamespace(
            plan=SimpleNamespace(
                instance_name="worker-0",
                role="worker",
                host="10.0.0.2",
                is_local=False,
                output_dir="/tmp/worker-0",
                readiness={},
                instance_config={"enable_disagg_pd": False},
            ),
            container_config=SimpleNamespace(container_name="worker-container"),
            container_manager=manager,
        )

        self.assertTrue(scheduler._wait_instance_ready(runtime, 1, "xllm"))
        manager.exec_in_container.assert_called_once()
        container_name, command = manager.exec_in_container.call_args.args
        self.assertEqual(container_name, "worker-container")
        self.assertIn("DistributeWorker server started", command)
        self.assertIn("node_0.log", command)
        self.assertEqual(manager.exec_in_container.call_args.kwargs["timeout"], 350)

    def test_cleanup_stage_cleans_container_only_once(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        context = TaskExecutionContext("task.yaml", "out")
        context.state.update(
            engine_config={"serving_container_name": ""},
            container_manager=manager,
        )
        with mock.patch.object(
            scheduler,
            "get_xtask_config_data",
            return_value={"ENV_CONFIG": {"CLEAN_RESULTS": True}},
        ):
            scheduler._cleanup_scheduler(context)
        manager.cleanup.assert_called_once_with()

    def test_archives_service_group_logs_by_instance(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            local_log_dir = output_path / "xllm_log" / "master-0" / "log"
            local_log_dir.mkdir(parents=True)
            (local_log_dir / "node_0.log").write_text("ready\n", encoding="utf-8")
            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(local_log_dir.parent),
                    instance_name="master-0",
                    host="127.0.0.1",
                    is_local=True,
                ),
                container_manager=mock.Mock(),
            )

            scheduler._archive_service_group_logs(
                [runtime], str(output_path), "20260722_100000"
            )

            archived_log = (
                output_path
                / "20260722_100000"
                / "xllm_log"
                / "master-0"
                / "log"
                / "node_0.log"
            )
            self.assertEqual(archived_log.read_text(encoding="utf-8"), "ready\n")
            runtime.container_manager.executor.download.assert_not_called()

    def test_post_process_makes_container_created_log_archive_host_writable(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(success=True)

        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            scheduler, "_archive_service_group_logs"
        ):
            scheduler._post_process_benchmark_rst(
                manager,
                SimpleNamespace(container_name="benchmark-container"),
                {"case_name": "case"},
                temporary_directory,
                "20260727_174225",
                service_runtimes=[],
            )

        command = manager.exec_in_container.call_args.args[1]
        self.assertIn("chmod -R a+rwX", command)
        self.assertIn("20260727_174225/xllm_log", command)

    def test_archive_refreshes_remote_logs_before_copy(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            remote_log_dir = output_path / "xllm_log" / "worker-0" / "log"
            executor = mock.Mock()

            def download(_remote_path, local_path):
                local_log_path = Path(local_path)
                local_log_path.mkdir(parents=True, exist_ok=True)
                (local_log_path / "node_0.log").write_text(
                    "remote-ready\n", encoding="utf-8"
                )
                return True

            executor.download.side_effect = download
            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(remote_log_dir.parent),
                    instance_name="worker-0",
                    host="10.0.0.2",
                    is_local=False,
                ),
                container_manager=SimpleNamespace(executor=executor),
            )

            scheduler._archive_service_group_logs(
                [runtime], str(output_path), "20260722_100000"
            )

            executor.download.assert_called_once_with(
                str(remote_log_dir), str(remote_log_dir)
            )
            archived_log = (
                output_path
                / "20260722_100000"
                / "xllm_log"
                / "worker-0"
                / "log"
                / "node_0.log"
            )
            self.assertEqual(
                archived_log.read_text(encoding="utf-8"), "remote-ready\n"
            )

    def test_remote_runtime_download_makes_container_artifacts_readable(self):
        from xtask.core import scheduler

        executor = mock.Mock()
        executor.download.return_value = True
        manager = mock.Mock(executor=executor)
        manager.exec_in_container.return_value = SimpleNamespace(
            success=True,
            stdout="",
            stderr="",
        )
        runtime = SimpleNamespace(
            plan=SimpleNamespace(
                output_dir="/tmp/xllm_log/decode-0",
                instance_name="decode-0",
                host="10.0.0.2",
                is_local=False,
            ),
            container_config=SimpleNamespace(container_name="decode-container"),
            container_manager=manager,
        )

        self.assertTrue(
            scheduler._download_remote_runtime_path(
                runtime,
                "/tmp/xllm_log/decode-0/log",
                "/tmp/local-log",
            )
        )

        manager.exec_in_container.assert_called_once()
        container_name, command = manager.exec_in_container.call_args.args
        self.assertEqual(container_name, "decode-container")
        self.assertIn("chmod -R a+rwX", command)
        executor.download.assert_called_once_with(
            "/tmp/xllm_log/decode-0/log",
            "/tmp/local-log",
        )

    def test_completes_local_runtime_archive_without_overwriting_trial_log(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            instance_dir = output_path / "xllm_log" / "prefill-0"
            (instance_dir / "log").mkdir(parents=True)
            (instance_dir / "log" / "node_0.log").write_text(
                "latest-runtime\n", encoding="utf-8"
            )
            (instance_dir / "shutdown_log").mkdir()
            (instance_dir / "shutdown_log" / "node_0.log").write_text(
                "shutdown-output\n", encoding="utf-8"
            )
            (instance_dir / "kernel_meta").mkdir()
            (instance_dir / "kernel_meta" / "buildPidInfo.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (instance_dir / "start_xllm_serv_prefill-0.sh").write_text(
                "start-prefill\n", encoding="utf-8"
            )

            archived_instance = (
                output_path
                / "20260725_160250"
                / "xllm_log"
                / "prefill-0"
            )
            (archived_instance / "log").mkdir(parents=True)
            archived_log = archived_instance / "log" / "node_0.log"
            archived_log.write_text("trial-specific\n", encoding="utf-8")

            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(instance_dir),
                    instance_name="prefill-0",
                    host="127.0.0.1",
                    is_local=True,
                ),
                container_manager=mock.Mock(),
            )

            scheduler._archive_service_group_runtime_artifacts(
                [runtime], str(output_path), "20260725_160250"
            )

            self.assertEqual(
                archived_log.read_text(encoding="utf-8"), "trial-specific\n"
            )
            self.assertEqual(
                (
                    archived_instance
                    / "start_xllm_serv_prefill-0.sh"
                ).read_text(encoding="utf-8"),
                "start-prefill\n",
            )
            self.assertEqual(
                (
                    archived_instance / "shutdown_log" / "node_0.log"
                ).read_text(encoding="utf-8"),
                "shutdown-output\n",
            )
            self.assertTrue(
                (
                    archived_instance / "kernel_meta" / "buildPidInfo.json"
                ).is_file()
            )

    def test_completes_remote_runtime_archive_and_overlays_shutdown_log(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            instance_dir = output_path / "xllm_log" / "decode-0"
            instance_dir.mkdir(parents=True)
            (instance_dir / "start_xllm_serv_decode-0.sh").write_text(
                "local-start\n", encoding="utf-8"
            )
            (instance_dir / "shutdown_log").mkdir()
            (instance_dir / "shutdown_log" / "node_0.log").write_text(
                "remote-shutdown\n", encoding="utf-8"
            )

            archived_instance = (
                output_path
                / "20260725_160250"
                / "xllm_log"
                / "decode-0"
            )
            (archived_instance / "log").mkdir(parents=True)
            archived_log = archived_instance / "log" / "node_0.log"
            archived_log.write_text("trial-specific\n", encoding="utf-8")

            executor = mock.Mock()

            def download(_remote_path, local_path):
                snapshot = Path(local_path)
                (snapshot / "log").mkdir(parents=True)
                (snapshot / "log" / "node_0.log").write_text(
                    "final-runtime\n", encoding="utf-8"
                )
                (snapshot / "kernel_meta").mkdir()
                (snapshot / "kernel_meta" / "remote.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (snapshot / "start_xllm_serv_decode-0.sh").write_text(
                    "remote-start\n", encoding="utf-8"
                )
                return True

            executor.download.side_effect = download
            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(instance_dir),
                    instance_name="decode-0",
                    host="10.0.0.2",
                    is_local=False,
                ),
                container_manager=SimpleNamespace(executor=executor),
            )

            scheduler._archive_service_group_runtime_artifacts(
                [runtime], str(output_path), "20260725_160250"
            )

            self.assertEqual(executor.download.call_count, 1)
            self.assertEqual(
                executor.download.call_args.args[0], str(instance_dir)
            )
            self.assertEqual(
                archived_log.read_text(encoding="utf-8"), "trial-specific\n"
            )
            self.assertEqual(
                (
                    archived_instance / "shutdown_log" / "node_0.log"
                ).read_text(encoding="utf-8"),
                "remote-shutdown\n",
            )
            self.assertEqual(
                (
                    archived_instance / "start_xllm_serv_decode-0.sh"
                ).read_text(encoding="utf-8"),
                "local-start\n",
            )
            self.assertTrue(
                (archived_instance / "kernel_meta" / "remote.json").is_file()
            )

    def test_multi_host_cleanup_completes_every_recorded_archive_before_cleanup(self):
        from xtask.core import scheduler

        events = []
        runtime = SimpleNamespace(plan=SimpleNamespace(instance_name="prefill-0"))
        context = TaskExecutionContext("task.yaml", "/tmp/output")
        context.state.update(
            multi_host=True,
            service_runtimes=[runtime],
            service_group_archive_paths=["trial-1", "trial-2"],
            engine_config={"serving_container_name": ""},
            pd_control_service=None,
        )

        with mock.patch.object(
            scheduler,
            "get_xtask_config_data",
            return_value={"ENV_CONFIG": {"CLEAN_RESULTS": True}},
        ), mock.patch.object(
            scheduler, "_freeze_service_group_logs", return_value={}
        ), mock.patch.object(
            scheduler,
            "_separate_shutdown_logs",
            side_effect=lambda *_args: events.append("shutdown"),
        ), mock.patch.object(
            scheduler,
            "_archive_service_group_runtime_artifacts",
            side_effect=lambda _runtimes, _output, path: events.append(
                f"archive:{path}"
            ),
        ), mock.patch(
            "xtask.core.service_group.cleanup_service_group",
            side_effect=lambda *_args: events.append("cleanup"),
        ):
            scheduler._cleanup_scheduler(context)

        self.assertEqual(
            events,
            ["shutdown", "archive:trial-1", "archive:trial-2", "cleanup"],
        )

    def test_separates_local_shutdown_logs_and_restores_runtime_log(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            instance_dir = Path(temporary_directory) / "xllm_log" / "prefill-0"
            log_dir = instance_dir / "log"
            log_dir.mkdir(parents=True)
            log_file = log_dir / "node_0.log"
            log_file.write_text("runtime-ready\n", encoding="utf-8")
            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(instance_dir),
                    instance_name="prefill-0",
                    host="127.0.0.1",
                    is_local=True,
                ),
                container_manager=mock.Mock(),
            )

            snapshots = scheduler._freeze_service_group_logs([runtime])
            log_file.write_text(
                "runtime-ready\nException in thread Thread-1\nEOFError\n",
                encoding="utf-8",
            )
            scheduler._separate_shutdown_logs([runtime], snapshots)

            self.assertEqual(log_file.read_text(encoding="utf-8"), "runtime-ready\n")
            shutdown_log = instance_dir / "shutdown_log" / "node_0.log"
            self.assertIn(
                "Exception in thread Thread-1",
                shutdown_log.read_text(encoding="utf-8"),
            )
            self.assertFalse((instance_dir / ".log_before_shutdown").exists())

    def test_separates_remote_shutdown_logs_and_restores_remote_runtime_log(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            instance_dir = Path(temporary_directory) / "xllm_log" / "decode-0"
            log_dir = instance_dir / "log"
            executor = mock.Mock()
            download_count = 0

            def download(_remote_path, local_path):
                nonlocal download_count
                download_count += 1
                destination = Path(local_path)
                destination.mkdir(parents=True, exist_ok=True)
                content = "runtime-ready\n"
                if download_count == 2:
                    content += "TBE main process disappeared\n"
                (destination / "node_0.log").write_text(content, encoding="utf-8")
                return True

            executor.download.side_effect = download
            executor.upload.return_value = True
            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(instance_dir),
                    instance_name="decode-0",
                    host="10.0.0.2",
                    is_local=False,
                ),
                container_manager=SimpleNamespace(executor=executor),
            )

            snapshots = scheduler._freeze_service_group_logs([runtime])
            scheduler._separate_shutdown_logs([runtime], snapshots)

            self.assertEqual(
                (log_dir / "node_0.log").read_text(encoding="utf-8"),
                "runtime-ready\n",
            )
            self.assertIn(
                "main process disappeared",
                (instance_dir / "shutdown_log" / "node_0.log").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual(executor.download.call_count, 2)
            executor.upload.assert_not_called()

    def test_xllm_cleanup_does_not_kill_pd_control_process(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        manager.exec_in_container.side_effect = [
            SimpleNamespace(success=True),
            SimpleNamespace(success=True),
        ]
        runtime = SimpleNamespace(
            plan=SimpleNamespace(instance_name="prefill-0"),
            container_config=SimpleNamespace(container_name="prefill-container"),
            container_manager=manager,
        )

        with mock.patch.object(
            scheduler,
            "time",
            SimpleNamespace(sleep=mock.Mock()),
        ):
            scheduler._wait_xllm_processes_exit([runtime], grace_s=1)

        commands = [call.args[1] for call in manager.exec_in_container.call_args_list]
        self.assertIn("pkill -TERM -x xllm", commands[0])
        self.assertNotIn("pkill -TERM -f", commands[0])
        self.assertTrue(any("ps -eo stat=,comm=" in command for command in commands))
        wait_command = next(
            command for command in commands if "ps -eo stat=,comm=" in command
        )
        self.assertIn("grep -v '^[[:space:]]*Z'", wait_command)
        self.assertIn("\\$i", wait_command)
        self.assertIn("\\$((i+1))", wait_command)
        self.assertFalse(any("xllm_master_serving" in command for command in commands))

    def test_xllm_cleanup_uses_short_default_grace_and_skips_kill_on_exit(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        manager.exec_in_container.side_effect = [
            SimpleNamespace(success=True),
            SimpleNamespace(success=True),
        ]
        runtime = SimpleNamespace(
            plan=SimpleNamespace(instance_name="prefill-0"),
            container_config=SimpleNamespace(container_name="prefill-container"),
            container_manager=manager,
        )

        with mock.patch.object(
            scheduler,
            "time",
            SimpleNamespace(sleep=mock.Mock()),
        ):
            scheduler._wait_xllm_processes_exit([runtime])

        commands = [call.args[1] for call in manager.exec_in_container.call_args_list]
        wait_call = next(
            call
            for call in manager.exec_in_container.call_args_list
            if "ps -eo stat=,comm=" in call.args[1]
        )
        self.assertIn("-lt 15", wait_call.args[1])
        self.assertEqual(wait_call.kwargs["timeout"], 30)
        self.assertFalse(any("pkill -KILL -x xllm" in command for command in commands))

    def test_xllm_cleanup_kills_only_nodes_still_alive_after_grace(self):
        from xtask.core import scheduler

        manager = mock.Mock()
        manager.exec_in_container.side_effect = [
            SimpleNamespace(success=True),
            SimpleNamespace(success=False),
            SimpleNamespace(success=True),
        ]
        runtime = SimpleNamespace(
            plan=SimpleNamespace(instance_name="decode-0"),
            container_config=SimpleNamespace(container_name="decode-container"),
            container_manager=manager,
        )

        sleep = mock.Mock()
        with mock.patch.object(
            scheduler,
            "time",
            SimpleNamespace(sleep=sleep),
        ):
            scheduler._wait_xllm_processes_exit([runtime], grace_s=3)

        commands = [call.args[1] for call in manager.exec_in_container.call_args_list]
        self.assertEqual(
            sum("pkill -KILL -x xllm" in command for command in commands),
            1,
        )
        sleep.assert_called_once_with(0.2)

    def test_pd_cleanup_allows_shutdown_timeout_overrides(self):
        from xtask.core import scheduler

        runtime = SimpleNamespace(plan=SimpleNamespace(instance_name="prefill-0"))
        pd_control_service = mock.Mock()
        context = TaskExecutionContext("task.yaml", "/tmp/output")
        context.state.update(
            multi_host=True,
            service_runtimes=[runtime],
            service_group_archive_paths=[],
            engine_config={"serving_container_name": ""},
            task_config={
                "deployment": {
                    "shutdown_grace_seconds": 7,
                    "container_stop_timeout_seconds": 3,
                }
            },
            pd_control_service=pd_control_service,
        )

        with mock.patch.object(
            scheduler,
            "get_xtask_config_data",
            return_value={"ENV_CONFIG": {"CLEAN_RESULTS": True}},
        ), mock.patch.object(
            scheduler, "_freeze_service_group_logs", return_value={}
        ), mock.patch.object(
            scheduler, "_separate_shutdown_logs"
        ), mock.patch.object(
            scheduler, "_wait_xllm_processes_exit"
        ) as wait_exit, mock.patch(
            "xtask.core.service_group.cleanup_service_group"
        ) as cleanup_group:
            scheduler._cleanup_scheduler(context)

        wait_exit.assert_called_once_with([runtime], grace_s=7)
        pd_control_service.stop.assert_called_once_with()
        cleanup_group.assert_called_once_with(
            [runtime], True, stop_timeout=3
        )

    def test_run_scheduler_signature_and_stage_wiring(self):
        from xtask.core import scheduler

        events = []
        patches = []
        stage_names = [
            "_load_task_and_engine",
            "_prepare_container",
            "_prepare_runtime_dependencies",
            "_start_and_wait_service",
            "_run_benchmark_and_process",
            "_finalize_result",
            "_cleanup_scheduler",
        ]
        for name in stage_names:
            def stage(context, stage_name=name):
                events.append(stage_name)
            patches.append(mock.patch.object(scheduler, name, stage))
        for patcher in patches:
            patcher.start()
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patches)])

        result = scheduler.run_scheduler("task.yaml", "out", object())

        self.assertEqual(
            str(inspect.signature(scheduler.run_scheduler)),
            "(task_yaml_path: str, output_file_path: str, args=None) -> Dict[str, Any]",
        )
        self.assertEqual(
            events,
            stage_names[:5] + ["_cleanup_scheduler", "_finalize_result"],
        )
        self.assertIn("execution_time", result)

    def test_session_trial_accepts_benchmark_only_changes(self):
        from xtask.core import scheduler

        base_config = {
            "case_name": "case_bs-1",
            "served_model_name": "xllm",
            "num_devices": 8,
            "metrics": ["perf"],
            "benchmark": {
                "benchmark_type": "evalscope",
                "perf": {"batch_size": 1, "num_prompts": 4},
            },
        }
        trial_config = {
            "case_name": "case_bs-2",
            "served_model_name": "xllm",
            "num_devices": 8,
            "metrics": ["perf"],
            "benchmark": {
                "benchmark_type": "evalscope",
                "perf": {"batch_size": 2, "num_prompts": 8},
            },
        }
        context = TaskExecutionContext("base.yaml", "out")
        context.state["source_task_config"] = base_config

        with mock.patch.object(
            scheduler.TaskConfig,
            "load_task_config",
            return_value=trial_config,
        ):
            loaded = scheduler._load_session_trial_config(
                context, "trial.yaml"
            )

        self.assertIs(loaded, trial_config)

    def test_session_trial_rejects_service_or_deployment_changes(self):
        from xtask.core import scheduler

        base_config = {
            "case_name": "case_bs-1",
            "served_model_name": "xllm",
            "metrics": ["perf"],
            "benchmark": {"perf": {"batch_size": 1}},
            "deployment": {
                "instances": [{"name": "master-0", "device_ids": [0, 1]}]
            },
        }
        changed_config = copy.deepcopy(base_config)
        changed_config["case_name"] = "case_bs-2"
        changed_config["benchmark"]["perf"]["batch_size"] = 2
        changed_config["deployment"]["instances"][0]["device_ids"] = [2, 3]
        context = TaskExecutionContext("base.yaml", "out")
        context.state["source_task_config"] = base_config

        with mock.patch.object(
            scheduler.TaskConfig,
            "load_task_config",
            return_value=changed_config,
        ):
            with self.assertRaisesRegex(
                ValueError, "service or deployment configuration changed"
            ):
                scheduler._load_session_trial_config(context, "trial.yaml")

    def test_session_trial_finalizes_after_execution_time_is_recorded(self):
        from xtask.core import scheduler

        context = TaskExecutionContext(
            "base.yaml", "base-output", result={"execution_time": 0}
        )
        captured_times = []
        task_scheduler = mock.Mock()
        task_scheduler.finalize_result.side_effect = (
            lambda current: captured_times.append(current.result["execution_time"])
        )
        session = scheduler.SchedulerSession(
            scheduler=task_scheduler,
            context=context,
            base_output_file_path="base-output",
            prepared=True,
        )

        with mock.patch.object(
            scheduler, "_reset_session_trial"
        ), mock.patch.object(
            scheduler, "_check_session_service_health", return_value=True
        ), mock.patch.object(
            scheduler.time, "time", side_effect=[100.0, 103.5]
        ):
            result = scheduler.run_scheduler_session_trial(
                session, "trial.yaml", "trial-output"
            )

        self.assertEqual(result["execution_time"], 3.5)
        self.assertEqual(captured_times, [3.5])
        task_scheduler.finalize_result.assert_called_once_with(context)

    def test_failed_session_prepare_finalizes_one_failure_result(self):
        from xtask.core import scheduler

        task_scheduler = mock.Mock()

        def fail_prepare(context):
            context.complete = True
            context.result["message"] = "service startup failed"

        task_scheduler.prepare.side_effect = fail_prepare
        captured_times = []
        task_scheduler.finalize_result.side_effect = (
            lambda current: captured_times.append(current.result["execution_time"])
        )

        with mock.patch.object(
            scheduler, "_build_task_scheduler", return_value=task_scheduler
        ), mock.patch.object(
            scheduler.time, "time", side_effect=[10.0, 14.0]
        ):
            session = scheduler.start_scheduler_session(
                "trial.yaml", "trial-output"
            )

        self.assertFalse(session.prepared)
        self.assertTrue(session.closed)
        self.assertEqual(captured_times, [4.0])
        task_scheduler.cleanup.assert_called_once_with(session.context)
        task_scheduler.finalize_result.assert_called_once_with(session.context)

    def test_non_auto_task_dispatch_uses_full_lifecycle_scheduler(self):
        import main as main_module

        args = SimpleNamespace(assess_test_type="none")
        expected_result = {"success": True}

        with mock.patch.object(
            main_module, "run_scheduler", return_value=expected_result
        ) as run_scheduler_mock, mock.patch.object(
            main_module, "start_scheduler_session"
        ) as start_session_mock, mock.patch.object(
            main_module, "run_scheduler_session_trial"
        ) as run_trial_mock:
            result, session = main_module.run_task(
                "task.yaml", "output", args
            )

        self.assertIs(result, expected_result)
        self.assertIsNone(session)
        run_scheduler_mock.assert_called_once_with(
            "task.yaml", "output", args
        )
        start_session_mock.assert_not_called()
        run_trial_mock.assert_not_called()

    def test_auto_task_dispatch_starts_one_session_and_reuses_it(self):
        import main as main_module

        args = SimpleNamespace(assess_test_type="auto_assess")
        session = object()
        first_result = {"trial": 1}
        second_result = {"trial": 2}

        with mock.patch.object(
            main_module,
            "start_scheduler_session",
            return_value=session,
        ) as start_session_mock, mock.patch.object(
            main_module,
            "run_scheduler_session_trial",
            side_effect=[first_result, second_result],
        ) as run_trial_mock, mock.patch.object(
            main_module, "run_scheduler"
        ) as run_scheduler_mock:
            result_1, reused_session = main_module.run_task(
                "task-1.yaml", "output-1", args
            )
            result_2, reused_session = main_module.run_task(
                "task-2.yaml", "output-2", args, reused_session
            )

        self.assertIs(result_1, first_result)
        self.assertIs(result_2, second_result)
        self.assertIs(reused_session, session)
        start_session_mock.assert_called_once_with(
            "task-1.yaml", "output-1", args
        )
        self.assertEqual(run_trial_mock.call_count, 2)
        run_scheduler_mock.assert_not_called()

    def test_auto_assessment_stops_after_scheduler_session_error(self):
        import main as main_module

        state = {"lower": 3, "curr": 4, "upper": 6}
        tasks = {"case_bs-4.yaml": "output-4"}
        keys = ["case_bs-4.yaml"]
        result = {
            "scheduler_error": True,
            "message": "archive permission denied",
            "task_config": {
                "metrics": ["perf"],
                "benchmark": {"perf": {"batch_size": 4}},
            },
        }

        action = main_module.process_auto_assessment(
            args=SimpleNamespace(assess_test_type="auto_assess"),
            result=result,
            config_para={},
            auto_assess_threshold=state,
            target_ttft=1000,
            target_tpot=12,
            tasks=tasks,
            keys_to_process=keys,
            xtask_config={},
        )

        self.assertEqual(action, "break")
        self.assertEqual(state["stop_reason"], "scheduler_error")
        self.assertEqual(keys, ["case_bs-4.yaml"])

    def test_auto_main_loop_closes_reused_session_once_on_break(self):
        import main as main_module

        args = SimpleNamespace(
            assess_test_type="auto_assess",
            host_addr="0.0.0.0",
        )
        checker = mock.Mock()
        checker.parse_arguments.return_value = args
        checker.validate_arguments.return_value = SimpleNamespace(is_valid=True)
        tasks = {
            "case_bs-1.yaml": "output-1",
            "case_bs-2.yaml": "output-2",
        }
        session = object()
        result = {"task_config": {"metrics": []}}
        reporter = mock.Mock()

        with mock.patch.object(
            main_module, "init_logger"
        ), mock.patch.object(
            main_module, "ArgumentChecker", return_value=checker
        ), mock.patch.object(
            main_module, "get_xtask_config_data", return_value={}
        ), mock.patch.object(
            main_module, "OpenMeClient", return_value=mock.Mock()
        ), mock.patch.object(
            main_module, "create_assessment_reporter", return_value=reporter
        ), mock.patch.object(
            main_module, "get_tasks", return_value=tasks
        ), mock.patch.object(
            main_module, "start_scheduler_session", return_value=session
        ) as start_session_mock, mock.patch.object(
            main_module,
            "run_scheduler_session_trial",
            return_value=result,
        ) as run_trial_mock, mock.patch.object(
            main_module,
            "process_auto_assessment",
            side_effect=[None, "break"],
        ), mock.patch.object(
            main_module, "close_scheduler_session"
        ) as close_session_mock:
            main_module.main()

        start_session_mock.assert_called_once_with(
            "case_bs-1.yaml", "output-1", args
        )
        self.assertEqual(run_trial_mock.call_count, 2)
        close_session_mock.assert_called_once_with(session)
        self.assertEqual(reporter.record_result.call_count, 2)
        reporter.finalize.assert_called_once_with(None)

    def test_exhaustive_main_records_each_generated_trial(self):
        import main as main_module

        args = SimpleNamespace(
            assess_test_type="para_set",
            host_addr="0.0.0.0",
            case_group="search_case",
        )
        checker = mock.Mock()
        checker.parse_arguments.return_value = args
        checker.validate_arguments.return_value = SimpleNamespace(is_valid=True)
        tasks = {
            "search_case_sg1.yaml": "output-1",
            "search_case_sg2.yaml": "output-2",
        }
        result = {"task_config": {"metrics": []}}
        reporter = mock.Mock()

        with mock.patch.object(
            main_module, "init_logger"
        ), mock.patch.object(
            main_module, "ArgumentChecker", return_value=checker
        ), mock.patch.object(
            main_module, "get_xtask_config_data", return_value={}
        ), mock.patch.object(
            main_module, "OpenMeClient", return_value=mock.Mock()
        ), mock.patch.object(
            main_module, "resolve_para_set_strategy", return_value="exhaustive"
        ), mock.patch.object(
            main_module, "get_tasks", return_value=tasks
        ), mock.patch.object(
            main_module, "create_assessment_reporter", return_value=reporter
        ) as reporter_factory, mock.patch.object(
            main_module, "run_scheduler", return_value=result
        ) as scheduler_mock, mock.patch.object(
            main_module, "close_scheduler_session"
        ):
            main_module.main()

        reporter_factory.assert_called_once_with(
            args, tasks, para_strategy="exhaustive"
        )
        self.assertEqual(scheduler_mock.call_count, 2)
        self.assertEqual(reporter.record_result.call_count, 2)
        reporter.finalize.assert_called_once_with(None)

    def test_normal_main_does_not_create_assessment_reporter(self):
        import main as main_module

        args = SimpleNamespace(
            assess_test_type="none",
            host_addr="0.0.0.0",
        )
        checker = mock.Mock()
        checker.parse_arguments.return_value = args
        checker.validate_arguments.return_value = SimpleNamespace(is_valid=True)
        tasks = {"normal.yaml": "normal-output"}
        result = {"task_config": {"metrics": []}}

        with mock.patch.object(
            main_module, "init_logger"
        ), mock.patch.object(
            main_module, "ArgumentChecker", return_value=checker
        ), mock.patch.object(
            main_module, "get_xtask_config_data", return_value={}
        ), mock.patch.object(
            main_module, "OpenMeClient", return_value=mock.Mock()
        ), mock.patch.object(
            main_module, "get_tasks", return_value=tasks
        ), mock.patch.object(
            main_module, "run_scheduler", return_value=result
        ) as scheduler_mock, mock.patch.object(
            main_module, "create_assessment_reporter"
        ) as reporter_factory, mock.patch.object(
            main_module, "close_scheduler_session"
        ):
            main_module.main()

        scheduler_mock.assert_called_once_with(
            "normal.yaml", "normal-output", args
        )
        reporter_factory.assert_not_called()

    def test_tpe_main_uses_dynamic_search_instead_of_static_tasks(self):
        import main as main_module

        args = SimpleNamespace(
            assess_test_type="para_set",
            host_addr="0.0.0.0",
            case_group="search_case",
        )
        checker = mock.Mock()
        checker.parse_arguments.return_value = args
        checker.validate_arguments.return_value = SimpleNamespace(is_valid=True)
        openme = mock.Mock()

        with mock.patch.object(
            main_module, "init_logger"
        ), mock.patch.object(
            main_module, "ArgumentChecker", return_value=checker
        ), mock.patch.object(
            main_module, "get_xtask_config_data", return_value={}
        ), mock.patch.object(
            main_module, "OpenMeClient", return_value=openme
        ), mock.patch.object(
            main_module, "resolve_para_set_strategy", return_value="tpe"
        ) as strategy_mock, mock.patch.object(
            main_module, "run_tpe_parameter_search"
        ) as tpe_search_mock, mock.patch.object(
            main_module, "get_tasks"
        ) as get_tasks_mock, mock.patch.object(
            main_module, "create_assessment_reporter"
        ) as reporter_factory:
            main_module.main()

        strategy_mock.assert_called_once_with(args)
        tpe_search_mock.assert_called_once_with(
            args=args,
            openme_client=openme,
            run_trial=main_module.run_scheduler,
        )
        openme.SendJUEMsgCard.assert_called_once_with(args=args)
        get_tasks_mock.assert_not_called()
        reporter_factory.assert_not_called()

    def test_fallback_archive_creates_date_dir_when_benchmark_missing(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            log_dir = output_path / "log"
            log_dir.mkdir()
            (log_dir / "node_0.log").write_text("startup failed\n", encoding="utf-8")
            (output_path / "start_xllm_serv.sh").write_text(
                "#!/bin/bash\n", encoding="utf-8"
            )
            (output_path / "case_a.json").write_text("{}\n", encoding="utf-8")

            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state.update(
                task_config={"case_name": "case_a"},
                init_entries=["log", "start_xllm_serv.sh", "case_a.json"],
            )

            scheduler._ensure_benchmark_date_dir(context)

            benchmark_output = Path(context.result["benchmark_output_path"])
            self.assertEqual(benchmark_output.parent, output_path)
            self.assertRegex(benchmark_output.name, r"^\d{8}_\d{6}$")
            self.assertEqual(
                (benchmark_output / "xllm_log" / "node_0.log").read_text(
                    encoding="utf-8"
                ),
                "startup failed\n",
            )
            self.assertTrue((benchmark_output / "start_xllm_serv.sh").is_file())
            self.assertTrue((benchmark_output / "case_a.json").is_file())
            # The fallback archives copies; the case-root sources stay in place.
            self.assertTrue((log_dir / "node_0.log").is_file())
            self.assertTrue((output_path / "start_xllm_serv.sh").is_file())

    def test_fallback_archive_keeps_benchmark_generated_date_dir(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            benchmark_dir = output_path / "20260720_101010"
            benchmark_dir.mkdir()

            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["benchmark_output_path"] = str(benchmark_dir)
            context.state["init_entries"] = []

            scheduler._ensure_benchmark_date_dir(context)

            self.assertEqual(
                context.result["benchmark_output_path"], str(benchmark_dir)
            )
            self.assertEqual(
                [entry.name for entry in output_path.iterdir()],
                ["20260720_101010"],
            )

    def test_fallback_archive_reuses_uncaptured_benchmark_date_dir(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            benchmark_dir = output_path / "20260811_093000"
            benchmark_dir.mkdir()
            (benchmark_dir / "partial.txt").write_text("partial\n", encoding="utf-8")
            log_dir = output_path / "log"
            log_dir.mkdir()
            (log_dir / "node_0.log").write_text("crash\n", encoding="utf-8")

            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state["init_entries"] = ["log"]

            scheduler._ensure_benchmark_date_dir(context)

            self.assertEqual(
                context.result["benchmark_output_path"], str(benchmark_dir)
            )
            self.assertEqual(
                (benchmark_dir / "xllm_log" / "node_0.log").read_text(
                    encoding="utf-8"
                ),
                "crash\n",
            )
            self.assertTrue((benchmark_dir / "partial.txt").is_file())

    def test_fallback_archive_ignores_date_dirs_from_previous_runs(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            old_dir = output_path / "20260701_000000"
            old_dir.mkdir()

            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.state["init_entries"] = ["20260701_000000"]

            scheduler._ensure_benchmark_date_dir(context)

            benchmark_output = Path(context.result["benchmark_output_path"])
            self.assertNotEqual(benchmark_output.name, "20260701_000000")
            self.assertRegex(benchmark_output.name, r"^\d{8}_\d{6}$")
            self.assertEqual(list(old_dir.iterdir()), [])

    def test_fallback_archive_mirrors_service_group_layout(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            instance_dir = output_path / "xllm_log" / "master-0"
            log_dir = instance_dir / "log"
            log_dir.mkdir(parents=True)
            (log_dir / "node_0.log").write_text("ready\n", encoding="utf-8")
            (instance_dir / "start_xllm_serv_master-0.sh").write_text(
                "start\n", encoding="utf-8"
            )

            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(instance_dir),
                    instance_name="master-0",
                    host="127.0.0.1",
                    is_local=True,
                ),
                container_config=SimpleNamespace(
                    container_name="master-container"
                ),
                container_manager=mock.Mock(),
            )
            runtime.container_manager.container_is_running.return_value = False
            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state.update(
                task_config={"case_name": "case_a"},
                init_entries=["xllm_log"],
                service_runtimes=[runtime],
            )

            scheduler._ensure_benchmark_date_dir(context)

            benchmark_output = Path(context.result["benchmark_output_path"])
            self.assertRegex(benchmark_output.name, r"^\d{8}_\d{6}$")
            archived = benchmark_output / "xllm_log" / "master-0"
            self.assertEqual(
                (archived / "log" / "node_0.log").read_text(encoding="utf-8"),
                "ready\n",
            )
            self.assertTrue(
                (archived / "start_xllm_serv_master-0.sh").is_file()
            )
            self.assertEqual(
                context.state["service_group_archive_paths"],
                [str(benchmark_output)],
            )

    def test_fallback_archive_reuses_unwritable_date_dir_without_creating_another(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            root_like_dir = output_path / "20260811_201941"
            root_like_dir.mkdir()
            (root_like_dir / "configs").mkdir()
            # Simulate an aisbench-created directory the host user cannot write into.
            root_like_dir.chmod(0o555)
            log_dir = output_path / "log"
            log_dir.mkdir()
            (log_dir / "node_0.log").write_text("crash\n", encoding="utf-8")
            (output_path / "start_xllm_serv.sh").write_text(
                "#!/bin/bash\n", encoding="utf-8"
            )

            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state["init_entries"] = ["log", "start_xllm_serv.sh"]

            try:
                scheduler._ensure_benchmark_date_dir(context)
            finally:
                root_like_dir.chmod(0o755)

            # Prefer reuse even when host cannot write; do not mint another date dir.
            self.assertEqual(
                context.result["benchmark_output_path"], str(root_like_dir)
            )
            date_dirs = [
                entry
                for entry in output_path.iterdir()
                if entry.is_dir() and entry.name[:8].isdigit()
            ]
            self.assertEqual([entry.name for entry in date_dirs], ["20260811_201941"])
            self.assertTrue((root_like_dir / "configs").is_dir())

    def test_fallback_archive_copies_via_container_when_reused_not_writable(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            root_like_dir = output_path / "20260811_201941"
            root_like_dir.mkdir()
            root_like_dir.chmod(0o555)
            log_dir = output_path / "log"
            log_dir.mkdir()
            (log_dir / "node_0.log").write_text("crash\n", encoding="utf-8")
            (output_path / "start_xllm_serv.sh").write_text(
                "#!/bin/bash\n", encoding="utf-8"
            )

            manager = mock.Mock()
            manager.container_is_running.return_value = True
            # chmod claims success but does not make the directory writable, so
            # the fallback must archive through the container without minting
            # another date directory.
            manager.exec_in_container.return_value = SimpleNamespace(success=True)

            def _container_copy(_manager, _config, source, destination):
                root_like_dir.chmod(0o755)
                scheduler._copy_archive_entry(Path(source), Path(destination))
                root_like_dir.chmod(0o555)
                return True

            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state.update(
                init_entries=["log", "start_xllm_serv.sh"],
                container_manager=manager,
                container_config=SimpleNamespace(container_name="case-container"),
            )

            try:
                with mock.patch.object(
                    scheduler,
                    "_copy_entry_via_container",
                    side_effect=_container_copy,
                ), mock.patch(
                    "xtask.utils.container_manager.path_mapping_in_container",
                    side_effect=lambda path: path,
                ):
                    scheduler._ensure_benchmark_date_dir(context)
            finally:
                root_like_dir.chmod(0o755)

            self.assertEqual(
                context.result["benchmark_output_path"], str(root_like_dir)
            )
            self.assertEqual(
                (root_like_dir / "xllm_log" / "node_0.log").read_text(
                    encoding="utf-8"
                ),
                "crash\n",
            )
            self.assertTrue((root_like_dir / "start_xllm_serv.sh").is_file())
            date_dirs = [
                entry.name
                for entry in output_path.iterdir()
                if entry.is_dir() and entry.name[:8].isdigit()
            ]
            self.assertEqual(date_dirs, ["20260811_201941"])

    def test_fallback_archive_retries_unreadable_kernel_meta_via_container(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            date_dir = output_path / "20260812_100920"
            date_dir.mkdir()
            instance_dir = output_path / "xllm_log" / "master-0"
            log_dir = instance_dir / "log"
            log_dir.mkdir(parents=True)
            (log_dir / "node_0.log").write_text("ready\n", encoding="utf-8")
            (instance_dir / "start_xllm_serv_master-0.sh").write_text(
                "start\n", encoding="utf-8"
            )
            kernel_meta = instance_dir / "kernel_meta"
            kernel_meta.mkdir()
            locked = kernel_meta / "kernel_meta_temp_1" / "locked.bin"
            locked.parent.mkdir()
            locked.write_text("secret\n", encoding="utf-8")
            locked.chmod(0o000)

            manager = mock.Mock()
            manager.container_is_running.return_value = True
            manager.exec_in_container.return_value = SimpleNamespace(success=True)

            def _container_copy(_manager, _config, source, destination):
                # Simulate root-in-container being able to read/copy the locked tree.
                locked.chmod(0o644)
                scheduler._copy_archive_entry(Path(source), Path(destination))
                return True

            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    output_dir=str(instance_dir),
                    instance_name="master-0",
                    host="127.0.0.1",
                    is_local=True,
                ),
                container_config=SimpleNamespace(container_name="master-container"),
                container_manager=manager,
            )
            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state.update(
                task_config={"case_name": "case_a"},
                init_entries=["xllm_log"],
                service_runtimes=[runtime],
                container_manager=manager,
                container_config=SimpleNamespace(container_name="master-container"),
            )

            try:
                with mock.patch.object(
                    scheduler,
                    "_copy_entry_via_container",
                    side_effect=_container_copy,
                ), mock.patch(
                    "xtask.utils.container_manager.path_mapping_in_container",
                    side_effect=lambda path: path,
                ), mock.patch.object(
                    scheduler.logging, "warning"
                ) as warning_mock:
                    scheduler._ensure_benchmark_date_dir(context)
                archive_warnings = [
                    call
                    for call in warning_mock.call_args_list
                    if call.args and "Failed to archive" in str(call.args[0])
                ]
                self.assertEqual(archive_warnings, [])
            finally:
                locked.chmod(0o644)

            self.assertEqual(
                context.result["benchmark_output_path"], str(date_dir)
            )
            archived = date_dir / "xllm_log" / "master-0"
            self.assertEqual(
                (archived / "log" / "node_0.log").read_text(encoding="utf-8"),
                "ready\n",
            )
            self.assertTrue((archived / "start_xllm_serv_master-0.sh").is_file())
            self.assertEqual(
                (archived / "kernel_meta" / "kernel_meta_temp_1" / "locked.bin").read_text(
                    encoding="utf-8"
                ),
                "secret\n",
            )

    def test_fallback_archive_reuses_after_container_chmod(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            root_like_dir = output_path / "20260811_201941"
            root_like_dir.mkdir()
            root_like_dir.chmod(0o555)
            log_dir = output_path / "log"
            log_dir.mkdir()
            (log_dir / "node_0.log").write_text("crash\n", encoding="utf-8")

            manager = mock.Mock()
            manager.container_is_running.return_value = True
            manager.exec_in_container.return_value = SimpleNamespace(success=True)

            def _chmod_side_effect(*_args, **_kwargs):
                root_like_dir.chmod(0o755)
                return SimpleNamespace(success=True)

            manager.exec_in_container.side_effect = _chmod_side_effect

            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state.update(
                init_entries=["log"],
                container_manager=manager,
                container_config=SimpleNamespace(container_name="case-container"),
            )

            try:
                with mock.patch(
                    "xtask.utils.container_manager.path_mapping_in_container",
                    side_effect=lambda path: path,
                ):
                    scheduler._ensure_benchmark_date_dir(context)
            finally:
                root_like_dir.chmod(0o755)

            self.assertEqual(
                context.result["benchmark_output_path"], str(root_like_dir)
            )
            self.assertEqual(
                (root_like_dir / "xllm_log" / "node_0.log").read_text(
                    encoding="utf-8"
                ),
                "crash\n",
            )
            manager.exec_in_container.assert_called()

    def test_cleanup_prepares_fallback_date_dir_before_container_removal(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state["task_config"] = {"case_name": "case_a"}
            context.state["engine_config"] = {}
            container_manager = mock.Mock()
            context.state["container_manager"] = container_manager

            call_order = []

            def _ensure(ctx):
                call_order.append("ensure")
                ctx.result["benchmark_output_path"] = str(
                    output_path / "20260811_210000"
                )

            def _cleanup():
                call_order.append("cleanup")

            container_manager.cleanup.side_effect = _cleanup

            with mock.patch.object(
                scheduler, "_ensure_benchmark_date_dir", side_effect=_ensure
            ), mock.patch.object(
                scheduler,
                "get_xtask_config_data",
                return_value={"ENV_CONFIG": {"CLEAN_RESULTS": True}},
            ):
                scheduler._cleanup_scheduler(context)

            self.assertEqual(call_order, ["ensure", "cleanup"])
            self.assertEqual(
                context.result["benchmark_output_path"],
                str(output_path / "20260811_210000"),
            )

    def test_finalize_result_reports_fallback_date_dir(self):
        from xtask.core import scheduler

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            context = TaskExecutionContext("case_a.yaml", str(output_path))
            context.result["task_config"] = {"case_name": "case_a"}
            context.state["task_config"] = {"case_name": "case_a"}

            with mock.patch.object(scheduler, "write_result") as write_mock:
                scheduler._finalize_result(context)

            result_arg = write_mock.call_args.args[0]
            benchmark_output = Path(result_arg["benchmark_output_path"])
            self.assertEqual(benchmark_output.parent, output_path)
            self.assertRegex(benchmark_output.name, r"^\d{8}_\d{6}$")


if __name__ == "__main__":
    unittest.main()
