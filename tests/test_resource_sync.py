import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.utils.resource_sync import (
    ResourceSyncError,
    apply_dependence_package_mount,
    build_host_requirements,
    resolve_resource_sync_roots,
    sync_required_resources_to_hosts,
)
from xtask.core.task_scheduler import TaskExecutionContext


def _plan(host, *, local=False, runner=False):
    return SimpleNamespace(host=host, is_local=local, is_benchmark_runner=runner)


def _result(success=True, stdout="", stderr=""):
    return SimpleNamespace(success=success, stdout=stdout, stderr=stderr)


class _ResourceTree:
    def __init__(self, root):
        self.root = Path(root)
        self._write("numactl-2.0.16-9.vo2203sp4.aarch64.rpm", "rpm")
        self._write(
            "torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl",
            "wheel",
        )
        self._write("evalscope-1.5.0-py3-none-any.whl", "wheel")
        self._write("benchmark/setup.py", "setup")
        self._write("benchmark/ais_bench/benchmark/cli/main.py", "main")
        self._write(
            "benchmark/ais_bench/datasets/textvqa/textvqa_json/annotations.json",
            "{}",
        )
        self._write(
            "benchmark/ais_bench/datasets/textvqa/train_images/image.jpg",
            "image",
        )
        self._write("datasets/mmmu/Music/validation.parquet", "dataset")
        self._write(
            "datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
            "[]",
        )

    def _write(self, relative_path, content):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class ResourceSyncRootTests(unittest.TestCase):
    def test_source_and_target_fall_back_independently(self):
        with mock.patch(
            "xtask.utils.resource_sync.get_dependence_package_path",
            return_value="/export/home/models/xllm_test_060",
        ):
            self.assertEqual(
                resolve_resource_sync_roots({}),
                (
                    "/export/home/models/xllm_test_060",
                    "/export/home/models/xllm_test_060",
                ),
            )
            self.assertEqual(
                resolve_resource_sync_roots(
                    {"deployment": {"resource_sync": {"source": "/local/src"}}}
                ),
                ("/local/src", "/export/home/models/xllm_test_060"),
            )
            self.assertEqual(
                resolve_resource_sync_roots(
                    {"deployment": {"resource_sync": {"target": "/remote/dst"}}}
                ),
                ("/export/home/models/xllm_test_060", "/remote/dst"),
            )
            self.assertEqual(
                resolve_resource_sync_roots(
                    {
                        "deployment": {
                            "resource_sync": {
                                "source": "/local/src",
                                "target": "/remote/dst",
                            }
                        }
                    }
                ),
                ("/local/src", "/remote/dst"),
            )

    def test_apply_dependence_package_mount_replaces_previous_mapping(self):
        volumes = {
            "/export/home/models/": "/export/home/models",
            "/export/home/models/xllm_test_060": "/xllm_test_060",
        }
        apply_dependence_package_mount(volumes, "/data/xllm_test_060")
        self.assertEqual(volumes["/data/xllm_test_060"], "/xllm_test_060")
        self.assertNotIn("/export/home/models/xllm_test_060", volumes)
        self.assertEqual(volumes["/export/home/models/"], "/export/home/models")


class ResourceRequirementTests(unittest.TestCase):
    def test_remote_service_nodes_only_require_runtime_assets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            requirements = build_host_requirements(
                {
                    "benchmark": {"benchmark_type": "aisbench"},
                    "dataset_name": "gsm8k",
                },
                [
                    _plan("11.0.0.1", local=True, runner=True),
                    _plan("11.0.0.2"),
                    _plan("11.0.0.2"),
                ],
                temporary_directory,
            )

        self.assertEqual(list(requirements), ["11.0.0.2"])
        self.assertEqual(
            [component.name for component in requirements["11.0.0.2"]],
            ["runtime-numactl", "runtime-torch-npu"],
        )

    def test_benchmark_code_and_datasets_are_runner_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            requirements = build_host_requirements(
                {
                    "benchmark": {
                        "benchmark_type": "evalscope",
                        "acc": {
                            "dataset_args": {
                                "mmmu": {
                                    "dataset_id": "/xllm_test_060/datasets/mmmu"
                                }
                            }
                        },
                    },
                    "dataset_name": "mmmu",
                },
                [
                    _plan("11.0.0.1", runner=True),
                    _plan("11.0.0.2"),
                ],
                temporary_directory,
            )

        self.assertEqual(
            [component.name for component in requirements["11.0.0.2"]],
            ["runtime-numactl", "runtime-torch-npu"],
        )
        self.assertEqual(
            [component.relative_path for component in requirements["11.0.0.1"]],
            [
                "numactl-2.0.16-9.vo2203sp4.aarch64.rpm",
                "torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl",
                "evalscope-1.5.0-py3-none-any.whl",
                "datasets/mmmu",
            ],
        )

    def test_remote_aisbench_runner_excludes_unneeded_textvqa_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            requirements = build_host_requirements(
                {
                    "benchmark": {"benchmark_type": "aisbench"},
                    "dataset_name": "gsm8k",
                },
                [_plan("11.0.0.2", runner=True)],
                temporary_directory,
            )

        components = requirements["11.0.0.2"]
        self.assertEqual(
            [component.name for component in components],
            ["runtime-numactl", "runtime-torch-npu", "aisbench-core"],
        )
        self.assertIn(
            "ais_bench/datasets/textvqa/",
            next(item for item in components if item.name == "aisbench-core").excludes,
        )

    def test_textvqa_is_a_separate_on_demand_component(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            requirements = build_host_requirements(
                {
                    "benchmark": {"benchmark_type": "aisbench"},
                    "dataset_name": "textvqa",
                },
                [_plan("11.0.0.2", runner=True)],
                temporary_directory,
            )

        paths = [item.relative_path for item in requirements["11.0.0.2"]]
        self.assertIn("benchmark", paths)
        self.assertIn("benchmark/ais_bench/datasets/textvqa", paths)

    def test_evalscope_runner_gets_wheel_and_referenced_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            requirements = build_host_requirements(
                {
                    "benchmark": {
                        "benchmark_type": "evalscope",
                        "acc": {
                            "dataset_args": {
                                "mmmu": {
                                    "dataset_id": "/xllm_test_060/datasets/mmmu"
                                }
                            }
                        },
                    },
                    "dataset_name": "mmmu",
                },
                [_plan("11.0.0.2", runner=True)],
                temporary_directory,
            )

        paths = [item.relative_path for item in requirements["11.0.0.2"]]
        self.assertEqual(
            paths,
            [
                "numactl-2.0.16-9.vo2203sp4.aarch64.rpm",
                "torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl",
                "evalscope-1.5.0-py3-none-any.whl",
                "datasets/mmmu",
            ],
        )

    def test_host_style_dataset_path_is_synced_to_remote_aisbench_runner(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            requirements = build_host_requirements(
                {
                    "benchmark": {
                        "benchmark_type": "aisbench",
                        "perf": {
                            "dataset_path": (
                                "/export/home/models/xllm_test_060/datasets/sharegpt/"
                                "ShareGPT_V3_unfiltered_cleaned_split.json"
                            )
                        },
                    },
                    "dataset_name": "sharegpt",
                },
                [_plan("11.0.0.2", runner=True)],
                temporary_directory,
            )

        paths = [item.relative_path for item in requirements["11.0.0.2"]]
        self.assertEqual(
            paths,
            [
                "numactl-2.0.16-9.vo2203sp4.aarch64.rpm",
                "torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl",
                "benchmark",
                (
                    "datasets/sharegpt/"
                    "ShareGPT_V3_unfiltered_cleaned_split.json"
                ),
            ],
        )


class ResourceSynchronizationTests(unittest.TestCase):
    def _task(self, source, **resource_sync):
        return {
            "benchmark": {"benchmark_type": "aisbench"},
            "dataset_name": "gsm8k",
            "deployment": {
                "resource_sync": {
                    "source": source,
                    "target": "/remote/xllm_test_060",
                    **resource_sync,
                }
            },
        }

    def test_complete_remote_resources_are_skipped_with_one_probe(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            executor = mock.Mock()
            executor.host = "11.0.0.2"
            executor.run.return_value = _result(
                stdout=(
                    "PRESENT:runtime-numactl\n"
                    "PRESENT:runtime-torch-npu\n"
                    "PRESENT:aisbench-core\n"
                )
            )

            result = sync_required_resources_to_hosts(
                self._task(temporary_directory),
                [_plan("11.0.0.2", runner=True)],
                executor_factory=lambda _host: executor,
            )

        self.assertEqual(result["11.0.0.2"]["status"], "ready")
        self.assertEqual(executor.run.call_count, 1)
        executor.sync_path.assert_not_called()

    def test_missing_components_are_synced_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            executor = mock.Mock()
            executor.host = "11.0.0.2"

            def run(command, timeout=60):
                if "PRESENT:runtime-numactl" in command:
                    return _result(
                        stdout=(
                            "MISSING:runtime-numactl\n"
                            "MISSING:runtime-torch-npu\n"
                            "MISSING:aisbench-core\n"
                        )
                    )
                return _result()

            executor.run.side_effect = run
            executor.sync_path.return_value = True

            result = sync_required_resources_to_hosts(
                self._task(temporary_directory),
                [_plan("11.0.0.2", runner=True)],
                executor_factory=lambda _host: executor,
            )

        self.assertEqual(result["11.0.0.2"]["status"], "synced")
        self.assertEqual(
            result["11.0.0.2"]["synced"],
            ["runtime-numactl", "runtime-torch-npu", "aisbench-core"],
        )
        self.assertEqual(executor.sync_path.call_count, 3)
        aisbench_call = executor.sync_path.call_args_list[2]
        self.assertIn("ais_bench/datasets/textvqa/", aisbench_call.kwargs["excludes"])

    def test_sync_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            executor = mock.Mock()
            executor.host = "11.0.0.2"

            def run(command, timeout=60):
                if "PRESENT:runtime-numactl" in command:
                    return _result(
                        stdout=(
                            "MISSING:runtime-numactl\n"
                            "MISSING:runtime-torch-npu\n"
                            "MISSING:aisbench-core\n"
                        )
                    )
                return _result()

            executor.run.side_effect = run
            executor.sync_path.return_value = False

            with self.assertRaisesRegex(ResourceSyncError, "Failed to sync resource"):
                sync_required_resources_to_hosts(
                    self._task(temporary_directory),
                    [_plan("11.0.0.2", runner=True)],
                    executor_factory=lambda _host: executor,
                )

    def test_resource_sync_can_be_disabled(self):
        executor_factory = mock.Mock()
        result = sync_required_resources_to_hosts(
            {"deployment": {"resource_sync": False}},
            [_plan("11.0.0.2", runner=True)],
            executor_factory=executor_factory,
        )

        self.assertEqual(result, {})
        executor_factory.assert_not_called()

    def test_missing_target_falls_back_to_dependence_package_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _ResourceTree(temporary_directory)
            executor = mock.Mock()
            executor.host = "11.0.0.2"

            def run(command, timeout=60):
                if "PRESENT:runtime-numactl" in command:
                    return _result(
                        stdout=(
                            "MISSING:runtime-numactl\n"
                            "PRESENT:runtime-torch-npu\n"
                            "PRESENT:aisbench-core\n"
                        )
                    )
                return _result()

            executor.run.side_effect = run
            executor.sync_path.return_value = True

            with mock.patch(
                "xtask.utils.resource_sync.get_dependence_package_path",
                return_value="/dep/root",
            ):
                sync_required_resources_to_hosts(
                    {
                        "benchmark": {"benchmark_type": "aisbench"},
                        "dataset_name": "gsm8k",
                        "deployment": {
                            "resource_sync": {"source": temporary_directory}
                        },
                    },
                    [_plan("11.0.0.2", runner=True)],
                    executor_factory=lambda _host: executor,
                )

            remote_path = executor.sync_path.call_args.args[1]
            self.assertTrue(str(remote_path).startswith("/dep/root"))
            self.assertFalse(str(remote_path).startswith(temporary_directory))

    def test_scheduler_stops_before_workspace_or_container_on_sync_failure(self):
        from xtask.core import scheduler

        context = TaskExecutionContext(
            "task.yaml", "output", result=scheduler._initial_result()
        )
        context.state.update(
            engine_config={},
            task_config={"deployment": {}},
            service_plans=[_plan("11.0.0.2")],
        )

        with mock.patch(
            "xtask.utils.resource_sync.sync_required_resources_to_hosts",
            side_effect=ResourceSyncError("copy failed"),
        ), mock.patch(
            "xtask.utils.workspace_sync.sync_workspace_to_hosts"
        ) as sync_workspace, mock.patch(
            "xtask.core.service_group.create_runtime_for_plan"
        ) as create_runtime:
            scheduler._prepare_container_multi_host(context)

        self.assertTrue(context.complete)
        self.assertFalse(context.result["success"])
        self.assertIn("Multi-host resource sync failed", context.result["message"])
        sync_workspace.assert_not_called()
        create_runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
