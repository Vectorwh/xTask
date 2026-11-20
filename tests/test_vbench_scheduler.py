import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler


class VBenchSchedulerTests(unittest.TestCase):
    def _context(self, output_path):
        return SimpleNamespace(
            output_file_path=str(output_path),
            state={
                "task_config": {
                    "case_name": "vbench-case",
                    "benchmark": {
                        "benchmark_type": "vbench",
                        "vbench": {
                            "full_info": "/data/full_info.json",
                            "image_root": "/data/images",
                            "evaluation_script": "/workspace/vbench/evaluate.py",
                        },
                    },
                }
            },
            result={},
            complete=False,
        )

    def test_container_internal_evaluation_path_is_not_bound_from_host(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            scheduler, "get_xtask_config_data", return_value={"DOCKER_CONFIG": {"volumes": {}}}
        ):
            config = scheduler._vbench_container_config(
                self._context(Path(tmp) / "outputs" / "case"),
                SimpleNamespace(executor=object()),
                {"runtime": "docker", "network_mode": "host"},
            )
        self.assertNotIn("/workspace/vbench/evaluate.py", config.volumes)
        self.assertIn("/data/full_info.json", config.volumes)
        self.assertIn("/data/images", config.volumes)

    def test_default_cache_dir_points_to_shared_project_location(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            scheduler, "get_xtask_config_data", return_value={"DOCKER_CONFIG": {"volumes": {}}}
        ):
            config = scheduler._vbench_container_config(
                self._context(Path(tmp) / "outputs" / "case"),
                SimpleNamespace(executor=object()),
                {"runtime": "docker", "network_mode": "host"},
            )
        # The default cache is the on-demand symlink view rooted inside the
        # canonical asset tree; ensure_cache_layout() materializes it.
        self.assertTrue(
            config.environment["VBENCH_CACHE_DIR"].endswith(
                "/mmm_bench/Video/vbench2_beta_i2v/vbench_cache"
            )
        )
        # torch.hub-backed dimensions (camera_motion, dino) resolve through
        # TORCH_HOME, which must sit inside the same shared cache.
        self.assertEqual(
            config.environment["TORCH_HOME"],
            config.environment["VBENCH_CACHE_DIR"] + "/torch",
        )

    def test_explicit_cache_dir_is_mapped_into_container(self):
        context = self._context(Path(tempfile.gettempdir()) / "outputs" / "case")
        context.state["task_config"]["benchmark"]["vbench"]["cache_dir"] = "/opt/cache"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            scheduler, "get_xtask_config_data", return_value={"DOCKER_CONFIG": {"volumes": {}}}
        ), mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda value: value,
        ):
            config = scheduler._vbench_container_config(
                context,
                SimpleNamespace(executor=object()),
                {"runtime": "docker", "network_mode": "host"},
            )
        self.assertEqual(config.environment["VBENCH_CACHE_DIR"], "/opt/cache")
        self.assertIn("/opt/cache", config.volumes)

    def test_failed_readiness_cleans_partially_created_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = self._context(Path(tmp) / "outputs" / "case")
            fake_manager = mock.Mock()
            fake_manager.image_exists.return_value = True
            fake_manager.create_container.return_value = (True, "runner-id")
            fake_manager.wait_for_container.return_value = False
            with mock.patch.object(scheduler, "ContainerManager", return_value=fake_manager), mock.patch.object(
                scheduler,
                "get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                scheduler._prepare_vbench_container(
                    context,
                    SimpleNamespace(executor=object()),
                    {"runtime": "docker", "network_mode": "host"},
                )
        fake_manager.cleanup.assert_called_once_with()
        self.assertTrue(context.complete)
        self.assertNotIn("vbench_container_manager", context.state)


if __name__ == "__main__":
    unittest.main()
