import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core.scheduler import ContainerConfigBuilder
from xtask.utils.container_manager import ContainerConfig, ContainerManager


class ContainerManagerTests(unittest.TestCase):
    def _create_command(self, ipc_mode=None, shm_size=None):
        config = ContainerConfig(
            image_home="example/xllm",
            image_name="latest",
            container_name="runtime-0",
            volumes={},
            environment={},
            network_mode="host",
            ipc_mode=ipc_mode,
            shm_size=shm_size,
        )
        manager = ContainerManager(config)
        manager.image_exists = mock.Mock(return_value=True)
        manager.run_command = mock.Mock(
            return_value=SimpleNamespace(success=True, stdout="container-id", stderr="")
        )

        success, _container_id = manager.create_container()

        self.assertTrue(success)
        return manager.run_command.call_args.args[0]

    def test_default_ipc_mode_remains_host(self):
        command = self._create_command()

        self.assertIn("--network=host", command)
        self.assertIn("--ipc=host", command)
        self.assertNotIn("--shm-size=", command)

    def test_private_ipc_and_shm_size_are_rendered_independently(self):
        command = self._create_command(ipc_mode="private", shm_size="128g")

        self.assertIn("--network=host", command)
        self.assertIn("--ipc=private", command)
        self.assertIn("--shm-size=128g", command)

    def test_task_ipc_configuration_overrides_engine_defaults(self):
        with mock.patch(
            "xtask.core.scheduler.get_xtask_config_data",
            return_value={"DOCKER_CONFIG": {"volumes": {}}},
        ):
            config = ContainerConfigBuilder.create_container_config(
                {
                    "ipc_mode": "host",
                    "shm_size": "32g",
                    "network_mode": "host",
                },
                {
                    "case_name": "ipc-config",
                    "ipc_mode": "private",
                    "shm_size": "64g",
                },
                "task.yaml",
            )

        self.assertEqual(config.ipc_mode, "private")
        self.assertEqual(config.shm_size, "64g")


if __name__ == "__main__":
    unittest.main()
