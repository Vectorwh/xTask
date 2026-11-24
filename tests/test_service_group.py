import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core.service_group import (
    build_service_plans,
    cleanup_service_group,
    create_runtime_for_plan,
    is_multi_host_deployment,
    get_benchmark_endpoint,
    merge_instance_task_config,
    normalize_pd_deployment,
    normalize_service_routed_deployment,
    resolve_global_nnodes,
    resolve_master_host,
    validate_pd_deployment_authoring,
)


class ServiceGroupTests(unittest.TestCase):
    def test_service_control_config_uses_shared_ports_and_all_instance_hosts(self):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_service_control_config,
        )

        task = {
            "deployment": {
                "mode": "service_routed",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "etcd_client_port": 13389,
                    "etcd_peer_port": 13390,
                    "http_server_port": 13888,
                    "rpc_server_port": 13889,
                },
                "service_instances": [
                    {
                        "name": "default-0",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    },
                    {
                        "name": "default-1",
                        "nodes": [{"host": "10.0.0.2", "device_ids": [0]}],
                    },
                ],
            }
        }

        config = build_service_control_config(task)

        self.assertEqual(config.host, "10.0.0.1")
        self.assertEqual(config.etcd_client_port, 13389)
        self.assertEqual(config.http_server_port, 13888)
        self.assertEqual(config.output_subdir, "service_control")
        self.assertIn("10.0.0.2", config.no_proxy_hosts)
        runtime = SimpleNamespace()
        runtime.container_manager = mock.Mock()
        runtime.container_config = SimpleNamespace(container_name="service-control")
        # auto profile probes `xllm_master_serving --help`; simulate a legacy
        # binary so no --service_deployment_role flag is injected.
        runtime.container_manager.exec_in_container.return_value = SimpleNamespace(
            success=False, stdout="", stderr=""
        )
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            service = XllmControlService(
                runtime, config, "/models/Qwen3-8B", "/tmp/output"
            )
        command = service._build_start_command()
        self.assertIn("--etcd_addr=10.0.0.1:13389", command)
        self.assertIn("--http_server_port 13888", command)
        self.assertIn("--internal_api_token=" + config.internal_api_token, command)
        self.assertNotEqual(config.internal_api_token, "xtask-internal-api-token")
        self.assertNotIn("ENABLE_DECODE_RESPONSE_TO_SERVICE", command)

    def test_pd_control_start_command_includes_internal_api_token(self):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
        )

        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "internal_api_token": "pd-shared-secret",
                    "service_binary_profile": "refactor",
                },
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "nodes": [{"host": "10.0.0.2", "device_ids": [0]}],
                    },
                ],
            }
        }
        config = build_pd_control_config(task)
        runtime = SimpleNamespace()
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            service = XllmControlService(
                runtime, config, "/models/DeepSeek-V4", "/tmp/output"
            )
        command = service._build_start_command()
        self.assertIn("ENABLE_DECODE_RESPONSE_TO_SERVICE=true", command)
        self.assertIn("--internal_api_token=pd-shared-secret", command)
        self.assertIn("--service_deployment_role=COORDINATOR", command)

    def test_pd_control_legacy_profile_omits_deployment_role_flag(self):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
        )

        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "service_binary_profile": "legacy",
                },
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        config = build_pd_control_config(task)
        runtime = SimpleNamespace()
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            service = XllmControlService(
                runtime, config, "/models/DeepSeek-V4", "/tmp/output"
            )
        command = service._build_start_command()
        self.assertNotIn("service_deployment_role", command)

    def test_pd_control_auto_profile_uses_binary_help_detection(self):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
        )

        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1", "service_binary_profile": "auto"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        config = build_pd_control_config(task)
        runtime = SimpleNamespace()
        runtime.container_manager = mock.Mock()
        runtime.container_config = SimpleNamespace(container_name="pd-control")
        runtime.container_manager.exec_in_container.return_value = SimpleNamespace(
            success=True, stdout="", stderr=""
        )
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            service = XllmControlService(
                runtime, config, "/models/DeepSeek-V4", "/tmp/output"
            )
            command = service._build_start_command()

        self.assertIn("--service_deployment_role=COORDINATOR", command)
        commands = [
            call.args[1]
            for call in runtime.container_manager.exec_in_container.call_args_list
        ]
        self.assertEqual(len(commands), 2)
        self.assertIn("grep -q 'service_deployment_role'", commands[0])
        self.assertIn("grep -aq 'readyz'", commands[1])

    def test_pd_control_health_waits_for_master_key_or_readyz(self):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
        )

        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1", "master_key_wait_seconds": 5, "service_binary_profile": "refactor"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        config = build_pd_control_config(task)
        runtime = SimpleNamespace()
        runtime.container_manager = mock.Mock()
        runtime.container_config = SimpleNamespace(container_name="pd-control")
        runtime.container_manager.exec_in_container.side_effect = [
            SimpleNamespace(success=True, stdout="", stderr=""),
            SimpleNamespace(success=False, stdout="", stderr=""),
            SimpleNamespace(success=True, stdout="", stderr=""),
        ]
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ), mock.patch("xtask.core.pd_control_service.time.sleep", return_value=None):
            service = XllmControlService(
                runtime, config, "/models/DeepSeek-V4", "/tmp/output"
            )
            healthy, detail = service._check_health()

        self.assertTrue(healthy)
        self.assertEqual(detail, "ok")
        probe_commands = [
            call.args[1]
            for call in runtime.container_manager.exec_in_container.call_args_list
        ]
        self.assertIn("/v3/kv/range", probe_commands[1])
        self.assertIn("XLLM:SERVICE:MASTER", probe_commands[1])
        self.assertIn("/readyz", probe_commands[2])

    def test_pd_control_legacy_health_checks_master_key_only(self):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
        )

        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "master_key_wait_seconds": 5,
                    "service_binary_profile": "legacy",
                },
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        config = build_pd_control_config(task)
        runtime = SimpleNamespace()
        runtime.container_manager = mock.Mock()
        runtime.container_config = SimpleNamespace(container_name="pd-control")
        runtime.container_manager.exec_in_container.side_effect = [
            SimpleNamespace(success=True, stdout="", stderr=""),
            SimpleNamespace(success=True, stdout="", stderr=""),
        ]
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ), mock.patch("xtask.core.pd_control_service.time.sleep", return_value=None):
            service = XllmControlService(
                runtime, config, "/models/DeepSeek-V4", "/tmp/output"
            )
            healthy, detail = service._check_health()

        self.assertTrue(healthy)
        self.assertEqual(detail, "ok")
        probe_commands = [
            call.args[1]
            for call in runtime.container_manager.exec_in_container.call_args_list
        ]
        self.assertEqual(len(probe_commands), 2)
        self.assertIn("/v3/kv/range", probe_commands[1])
        self.assertNotIn("/readyz", probe_commands[1])

    def test_pd_control_legacy_anchor_failure_is_non_blocking(self):
        """Legacy binaries self-elect with a short lease; a timing-sensitive
        anchor probe must not block the historically-working basic health
        check (regression guard for the GitHub-compiled binary)."""
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
        )

        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "master_key_wait_seconds": 2,
                    "service_binary_profile": "legacy",
                },
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        config = build_pd_control_config(task)
        runtime = SimpleNamespace()
        runtime.container_manager = mock.Mock()
        runtime.container_config = SimpleNamespace(container_name="pd-control")
        # basic health passes, but the master-key anchor probe never finds
        # the key (simulating the 3s-lease expiry race).
        runtime.container_manager.exec_in_container.side_effect = [
            SimpleNamespace(success=True, stdout="", stderr=""),  # basic health
            SimpleNamespace(success=False, stdout="", stderr="key not found"),  # probe 1
            SimpleNamespace(success=False, stdout="", stderr="key not found"),  # probe 2
        ]
        # Drive the anchor loop for exactly two iterations: deadline = t0 + 2,
        # then while-checks return t0, t0+1, t0+2 (exit). The logging module
        # also calls time.time(), so the side_effect must not exhaust.
        time_seq = [0.0, 0.0, 1.0, 2.0]

        def fake_time():
            return time_seq.pop(0) if time_seq else 2.0

        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ), mock.patch(
            "xtask.core.pd_control_service.time.time", side_effect=fake_time
        ), mock.patch("xtask.core.pd_control_service.time.sleep", return_value=None):
            service = XllmControlService(
                runtime, config, "/models/DeepSeek-V4", "/tmp/output"
            )
            healthy, detail = service._check_health()

        # Legacy: anchor failure is advisory, so control plane is healthy.
        self.assertTrue(healthy)
        self.assertEqual(detail, "ok")

    def test_pd_control_refactor_anchor_failure_is_blocking(self):
        """The refactor binary may default to ROUTER and never elect a master;
        a missing anchor must fail startup (the regression this guard catches)."""
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
        )

        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "master_key_wait_seconds": 2,
                    "service_binary_profile": "refactor",
                },
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        config = build_pd_control_config(task)
        runtime = SimpleNamespace()
        runtime.container_manager = mock.Mock()
        runtime.container_config = SimpleNamespace(container_name="pd-control")
        runtime.container_manager.exec_in_container.side_effect = [
            SimpleNamespace(success=True, stdout="", stderr=""),  # basic health
            SimpleNamespace(success=False, stdout="", stderr="key not found"),  # master 1
            SimpleNamespace(success=False, stdout="", stderr="readyz not ready"),  # readyz 1
            SimpleNamespace(success=False, stdout="", stderr="key not found"),  # master 2
            SimpleNamespace(success=False, stdout="", stderr="readyz not ready"),  # readyz 2
        ]
        time_seq = [0.0, 0.0, 1.0, 2.0]

        def fake_time():
            return time_seq.pop(0) if time_seq else 2.0

        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ), mock.patch(
            "xtask.core.pd_control_service.time.time", side_effect=fake_time
        ), mock.patch("xtask.core.pd_control_service.time.sleep", return_value=None):
            service = XllmControlService(
                runtime, config, "/models/DeepSeek-V4", "/tmp/output"
            )
            healthy, detail = service._check_health()

        self.assertFalse(healthy)
        self.assertIn("control plane anchor not ready", detail)

    def test_service_control_start_command_uses_coordinator_role(self):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_service_control_config,
        )

        task = {
            "deployment": {
                "mode": "service_routed",
                "xllm_service": {"host": "10.0.0.1", "service_binary_profile": "refactor"},
                "service_instances": [
                    {
                        "name": "default-0",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        config = build_service_control_config(task)
        runtime = SimpleNamespace()
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container",
            side_effect=lambda path: path,
        ):
            service = XllmControlService(
                runtime, config, "/models/Qwen3-8B", "/tmp/output"
            )
        command = service._build_start_command()
        self.assertIn("--service_deployment_role=COORDINATOR", command)

    def test_service_routed_instances_are_flattened_into_independent_groups(self):
        task = {
            "deployment": {
                "mode": "service_routed",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "http_server_port": 38888,
                },
                "service_instances": [
                    {
                        "name": "default-0",
                        "progress_conn_port": 9590,
                        "nodes": [
                            {
                                "name": "default-0-node-0",
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            }
                        ],
                    },
                    {
                        "name": "default-1",
                        "progress_conn_port": 9690,
                        "nodes": [
                            {
                                "name": "default-1-node-0",
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48100,
                                "hccl_if_port": 43539,
                            }
                        ],
                    },
                ],
            }
        }

        normalize_service_routed_deployment(task)
        self.assertNotIn("service_instances", task["deployment"])
        self.assertEqual(
            [item["service_group"] for item in task["deployment"]["instances"]],
            ["default-0", "default-1"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)

        self.assertEqual([plan.instance_config["world_size"] for plan in plans], [2, 2])
        self.assertEqual(
            [plan.instance_config["progress_conn_port"] for plan in plans],
            [9590, 9690],
        )
        self.assertEqual(get_benchmark_endpoint(task, plans), "http://10.0.0.1:38888")

    def test_service_routed_requires_control_host_runtime_and_unique_ports(self):
        task = {
            "deployment": {
                "mode": "service_routed",
                "xllm_service": {"host": "10.0.0.9"},
                "service_instances": [
                    {
                        "name": "default-0",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "must match a service node host"):
                build_service_plans(task, tmp)

    def test_cleanup_service_group_stops_hosts_in_parallel(self):
        barrier = threading.Barrier(2)
        managers = []
        runtimes = []
        for index in range(2):
            manager = mock.Mock()
            manager.cleanup.side_effect = lambda **_kwargs: barrier.wait(timeout=2)
            managers.append(manager)
            runtimes.append(
                SimpleNamespace(
                    plan=SimpleNamespace(
                        instance_name=f"node-{index}",
                        host=f"10.0.0.{index + 1}",
                    ),
                    container_manager=manager,
                )
            )

        cleanup_service_group(runtimes, True, stop_timeout=5)

        for manager in managers:
            manager.cleanup.assert_called_once_with(stop_timeout=5)

    def test_hierarchical_pd_instances_are_flattened_per_logical_group(self):
        task = {
            "case_name": "pd-hierarchical",
            "deployment": {
                "mode": "pd_disaggregated",
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "group_master_host": "10.0.0.1",
                        "serving_extensiable_param": {
                            "enable_shm": True,
                            "max_seqs_per_batch": 10,
                        },
                        "nodes": [
                            {
                                "name": "prefill-0-node-0",
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                            },
                            {
                                "name": "prefill-0-node-1",
                                "host": "10.0.0.2",
                                "device_ids": [0, 1],
                                "serving_extensiable_param": {
                                    "enable_graph": True,
                                },
                            },
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "nodes": [
                            {
                                "host": "10.0.0.3",
                                "device_ids": [0, 1],
                            }
                        ],
                    },
                ],
            },
        }

        normalize_pd_deployment(task)

        self.assertNotIn("pd_instances", task["deployment"])
        self.assertEqual(
            [item["name"] for item in task["deployment"]["instances"]],
            ["prefill-0-node-0", "prefill-0-node-1", "decode-0-node-0"],
        )
        self.assertEqual(
            [item["pd_group"] for item in task["deployment"]["instances"]],
            ["prefill-0", "prefill-0", "decode-0"],
        )
        self.assertTrue(
            task["deployment"]["instances"][0]["serving_extensiable_param"][
                "enable_shm"
            ]
        )
        self.assertTrue(
            task["deployment"]["instances"][1]["serving_extensiable_param"][
                "enable_graph"
            ]
        )
        # Calling the normalizer again must not duplicate nodes.
        normalize_pd_deployment(task)
        self.assertEqual(len(task["deployment"]["instances"]), 3)

    def test_hierarchical_pd_same_host_nodes_and_group_ports_are_preserved(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "progress_conn_port": 9690,
                        "transfer_port_base": 27000,
                        "disagg_pd_port": 7797,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                                "hccl_if_port": 43539,
                            },
                            {
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48010,
                                "hccl_if_port": 43639,
                            },
                        ],
                    }
                ],
            }
        }

        normalize_pd_deployment(task)

        nodes = task["deployment"]["instances"]
        self.assertEqual([node["host"] for node in nodes], ["10.0.0.1"] * 2)
        self.assertEqual([node["hccl_if_port"] for node in nodes], [43539, 43639])
        for node in nodes:
            self.assertEqual(node["progress_conn_port"], 9690)
            self.assertEqual(node["transfer_port_base"], 27000)
            self.assertEqual(node["disagg_pd_port"], 7797)

    def test_hierarchical_pd_instances_reject_explicit_group_aliases(self):
        for field in ("pd_group", "group"):
            with self.subTest(field=field):
                task = {
                    "deployment": {
                        "pd_instances": [
                            {
                                "name": "decode-0",
                                field: "another-name",
                                "role": "decode",
                                "nodes": [{"host": "10.0.0.1"}],
                            }
                        ]
                    }
                }

                with self.assertRaisesRegex(ValueError, "use name"):
                    normalize_pd_deployment(task)

    def test_hierarchical_pd_nodes_reject_explicit_group_aliases(self):
        for field in ("pd_group", "group"):
            with self.subTest(field=field):
                task = {
                    "deployment": {
                        "pd_instances": [
                            {
                                "name": "decode-0",
                                "role": "decode",
                                "nodes": [
                                    {
                                        "host": "10.0.0.1",
                                        field: "decode-0",
                                    }
                                ],
                            }
                        ]
                    }
                }

                with self.assertRaisesRegex(ValueError, "internal field"):
                    normalize_pd_deployment(task)

    def test_flat_pd_authoring_is_rejected(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "instances": [
                    {
                        "name": "decode-0",
                        "host": "10.0.0.1",
                        "role": "decode",
                    }
                ]
            }
        }

        with self.assertRaisesRegex(ValueError, "deployment.pd_instances"):
            validate_pd_deployment_authoring(task)

    def test_non_pd_flat_authoring_remains_supported(self):
        task = {
            "deployment": {
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "role": "worker",
                    },
                ]
            }
        }

        validate_pd_deployment_authoring(task)

    def test_hierarchical_pd_plans_use_group_world_sizes_and_ranks(self):
        task = {
            "case_name": "pd-hierarchical",
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1", "http_server_port": 38888},
                "benchmark": {"runner_instance": "prefill-0-node-0"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [
                            {"host": "10.0.0.1", "device_ids": [0, 1]},
                            {"host": "10.0.0.2", "device_ids": [0, 1]},
                        ],
                    },
                    {
                        "name": "prefill-1",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.3", "device_ids": [0, 1]}],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "nodes": [
                            {"host": "10.0.0.4", "device_ids": [0, 1]},
                            {"host": "10.0.0.5", "device_ids": [0, 1]},
                            {"host": "10.0.0.6", "device_ids": [0, 1]},
                        ],
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)

        self.assertEqual(
            [
                (
                    plan.instance_name,
                    plan.rank_start,
                    plan.instance_config["world_size"],
                )
                for plan in plans
            ],
            [
                ("prefill-0-node-0", 0, 4),
                ("prefill-0-node-1", 2, 4),
                ("prefill-1-node-0", 0, 2),
                ("decode-0-node-0", 0, 6),
                ("decode-0-node-1", 2, 6),
                ("decode-0-node-2", 4, 6),
            ],
        )

    def test_same_host_non_pd_runtimes_use_disjoint_cards_and_ports(self):
        task = {
            "case_name": "same-host-mixed",
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0, 1],
                        "local_port": 8800,
                        "hccl_if_port": 43439,
                        "progress_conn_port": 9590,
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.1",
                        "role": "worker",
                        "device_ids": [2, 3],
                        "local_port": 8810,
                        "hccl_if_port": 43539,
                        "progress_conn_port": 9590,
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)

        self.assertEqual([plan.rank_start for plan in plans], [0, 2])
        self.assertEqual([plan.device_ids for plan in plans], [[0, 1], [2, 3]])
        self.assertEqual([plan.service_port for plan in plans], [8800, 8810])
        self.assertEqual(
            [plan.instance_config["hccl_if_port"] for plan in plans],
            [43439, 43539],
        )
        self.assertEqual(
            [plan.instance_config["progress_conn_port"] for plan in plans],
            [9590, 9590],
        )
        self.assertEqual([plan.isolate_ipc for plan in plans], [True, True])

    def test_same_host_pd_groups_and_same_group_nodes_are_supported(self):
        task = {
            "case_name": "same-host-pd",
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "progress_conn_port": 9590,
                        "transfer_port_base": 26000,
                        "disagg_pd_port": 7777,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            },
                            {
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48010,
                                "hccl_if_port": 43539,
                            },
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "progress_conn_port": 9790,
                        "transfer_port_base": 27000,
                        "disagg_pd_port": 7787,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [4, 5],
                                "local_port": 48100,
                                "hccl_if_port": 43639,
                            }
                        ],
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)

        self.assertEqual(
            [(plan.instance_config["pd_group"], plan.rank_start) for plan in plans],
            [("prefill-0", 0), ("prefill-0", 2), ("decode-0", 0)],
        )
        self.assertEqual(
            [plan.instance_config["transfer_port_base"] for plan in plans],
            [26000, 26000, 27000],
        )
        self.assertEqual(
            [plan.instance_config["hccl_if_port"] for plan in plans],
            [43439, 43539, 43639],
        )
        self.assertEqual(
            [plan.instance_config["progress_conn_port"] for plan in plans],
            [9590, 9590, 9790],
        )
        self.assertTrue(all(plan.isolate_ipc for plan in plans))

    def test_colocated_runtime_uses_host_ipc_by_default(self):
        task = {
            "case_name": "same-host-ipc",
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0],
                        "local_port": 8800,
                        "hccl_if_port": 43439,
                        "progress_conn_port": 9590,
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.1",
                        "role": "worker",
                        "device_ids": [1],
                        "local_port": 8810,
                        "hccl_if_port": 43539,
                        "progress_conn_port": 9590,
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)

        config = SimpleNamespace(ipc_mode=None, shm_size=None, container_name="")
        builder = mock.Mock()
        builder.create_container_config.return_value = config
        runtime = create_runtime_for_plan(plans[0], {}, "task.yaml", builder)

        self.assertEqual(runtime.container_config.ipc_mode, "host")
        self.assertIsNone(runtime.container_config.shm_size)

    def test_colocated_runtime_preserves_explicit_ipc_configuration(self):
        task = {
            "case_name": "same-host-ipc-explicit",
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0],
                        "local_port": 8800,
                        "ipc_mode": "private",
                        "shm_size": "64g",
                        "hccl_if_port": 43439,
                        "progress_conn_port": 9590,
                        "serving_extensiable_param": {"enable_shm": False},
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.1",
                        "role": "worker",
                        "device_ids": [1],
                        "local_port": 8810,
                        "hccl_if_port": 43539,
                        "progress_conn_port": 9590,
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_service_plans(task, tmp)[0]

        self.assertEqual(plan.instance_config["ipc_mode"], "private")
        self.assertEqual(plan.instance_config["shm_size"], "64g")
        config = SimpleNamespace(
            ipc_mode=plan.instance_config["ipc_mode"],
            shm_size=plan.instance_config["shm_size"],
            container_name="",
        )
        builder = mock.Mock()
        builder.create_container_config.return_value = config
        runtime = create_runtime_for_plan(plan, {}, "task.yaml", builder)

        self.assertEqual(runtime.container_config.ipc_mode, "private")
        self.assertEqual(runtime.container_config.shm_size, "64g")

    def test_colocated_same_group_rejects_private_ipc_with_shm(self):
        task = {
            "case_name": "same-host-private-shm",
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0],
                        "local_port": 8800,
                        "ipc_mode": "private",
                        "hccl_if_port": 43439,
                        "progress_conn_port": 9590,
                        "serving_extensiable_param": {"enable_shm": True},
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.1",
                        "role": "worker",
                        "device_ids": [1],
                        "local_port": 8810,
                        "hccl_if_port": 43539,
                        "progress_conn_port": 9590,
                        "serving_extensiable_param": {"enable_shm": True},
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "enable_shm=true.*ipc_mode=private"
        ):
            build_service_plans(task, tmp)

    def test_colocated_same_group_rejects_engine_private_ipc_with_shm(self):
        task = {
            "case_name": "same-host-engine-private-shm",
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0],
                        "local_port": 8800,
                        "hccl_if_port": 43439,
                        "serving_extensiable_param": {"enable_shm": True},
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.1",
                        "role": "worker",
                        "device_ids": [1],
                        "local_port": 8810,
                        "hccl_if_port": 43539,
                        "serving_extensiable_param": {"enable_shm": True},
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_service_plans(task, tmp)[0]

        config = SimpleNamespace(
            ipc_mode="private", shm_size="64g", container_name=""
        )
        builder = mock.Mock()
        builder.create_container_config.return_value = config

        with self.assertRaisesRegex(ValueError, "would isolate xLLM shared memory"):
            create_runtime_for_plan(plan, {}, "task.yaml", builder)

    def test_colocated_shm_groups_reject_duplicate_progress_port(self):
        task = {
            "case_name": "same-host-shm-groups",
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.4"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "group_master_host": "10.0.0.1",
                        "progress_conn_port": 9590,
                        "transfer_port_base": 26000,
                        "disagg_pd_port": 7777,
                        "serving_extensiable_param": {"enable_shm": True},
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            },
                            {
                                "host": "10.0.0.3",
                                "device_ids": [0],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            },
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "group_master_host": "10.0.0.2",
                        "progress_conn_port": 9590,
                        "transfer_port_base": 27000,
                        "disagg_pd_port": 7777,
                        "serving_extensiable_param": {"enable_shm": True},
                        "nodes": [
                            {
                                "host": "10.0.0.2",
                                "device_ids": [0],
                                "local_port": 48100,
                                "hccl_if_port": 43439,
                            },
                            {
                                "host": "10.0.0.3",
                                "device_ids": [1],
                                "local_port": 48100,
                                "hccl_if_port": 43539,
                            },
                        ],
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError,
            "Shared-memory progress_conn_port 9590.*prefill-0.*decode-0",
        ):
            build_service_plans(task, tmp)

    def test_same_host_rejects_overlapping_devices(self):
        task = {
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0, 1],
                        "local_port": 8800,
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.1",
                        "role": "worker",
                        "device_ids": [1, 2],
                        "local_port": 8810,
                    },
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "Device 1.*assigned to both"
        ):
            build_service_plans(task, tmp)

    def test_same_host_rejects_overlapping_local_port_ranges(self):
        task = {
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0, 1],
                        "local_port": 8800,
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.1",
                        "role": "worker",
                        "device_ids": [2, 3],
                        "local_port": 8801,
                    },
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "Port 8801.*local_port range"
        ):
            build_service_plans(task, tmp)

    def test_same_host_pd_nodes_reject_duplicate_hccl_port(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "progress_conn_port": 9590,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            },
                            {
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48010,
                                "hccl_if_port": 43439,  # Same port - should fail
                            },
                        ],
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "Port 43439.*is used by both"
        ):
            build_service_plans(task, tmp)

    def test_pd_group_rejects_inconsistent_progress_conn_port(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "progress_conn_port": 9590,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            },
                            {
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48010,
                                "hccl_if_port": 43539,
                                "progress_conn_port": 9690,
                            },
                        ],
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError,
            "Logical group prefill-0 must use one shared progress_conn_port",
        ):
            build_service_plans(task, tmp)

    def test_same_master_pd_groups_reject_duplicate_progress_conn_port(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "progress_conn_port": 9590,
                        "transfer_port_base": 26000,
                        "disagg_pd_port": 7777,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            }
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "progress_conn_port": 9590,
                        "transfer_port_base": 27000,
                        "disagg_pd_port": 7787,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48100,
                                "hccl_if_port": 43539,
                            }
                        ],
                    },
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "Port 9590.*progress_conn_port"
        ):
            build_service_plans(task, tmp)

    def test_cross_host_runtimes_may_reuse_hccl_port(self):
        task = {
            "deployment": {
                "master_host": "10.0.0.1",
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0, 1],
                        "local_port": 8800,
                        "hccl_if_port": 43439,
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "role": "worker",
                        "device_ids": [0, 1],
                        "local_port": 8800,
                        "hccl_if_port": 43439,
                    },
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)

        self.assertEqual(
            [plan.instance_config["hccl_if_port"] for plan in plans],
            [43439, 43439],
        )
        self.assertEqual(
            [plan.instance_config["progress_conn_port"] for plan in plans],
            [9590, 9590],
        )

    def test_same_host_pd_groups_reject_overlapping_transfer_ports(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "hccl_if_port": 43439,
                        "progress_conn_port": 9590,
                        "transfer_port_base": 26000,
                        "disagg_pd_port": 7777,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                            }
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "hccl_if_port": 43539,
                        "progress_conn_port": 9690,
                        "transfer_port_base": 26000,
                        "disagg_pd_port": 7787,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48100,
                            }
                        ],
                    },
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "Port 26000.*transfer ports"
        ):
            build_service_plans(task, tmp)

    def test_same_host_pd_groups_reject_overlapping_disagg_ports(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "hccl_if_port": 43439,
                        "progress_conn_port": 9590,
                        "transfer_port_base": 26000,
                        "disagg_pd_port": 7777,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                            }
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "hccl_if_port": 43539,
                        "progress_conn_port": 9690,
                        "transfer_port_base": 27000,
                        "disagg_pd_port": 7777,
                        "nodes": [
                            {
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48100,
                            }
                        ],
                    },
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "Port 7777.*disagg_pd_port"
        ):
            build_service_plans(task, tmp)

    def test_single_host_yaml_is_not_multi_host(self):
        self.assertFalse(is_multi_host_deployment({"case_name": "a"}))
        self.assertFalse(is_multi_host_deployment({"deployment": {}}))
        self.assertFalse(is_multi_host_deployment({"deployment": {"instances": []}}))

    def test_single_instance_deployment_uses_service_group_path(self):
        task = {
            "case_name": "single",
            "deployment": {
                "master_host": "10.0.0.1",
                "benchmark": {
                    "runner_instance": "master-0",
                },
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "rank_start": 0,
                        "device_ids": [0, 1],
                        "local_port": 8800,
                        "serving_extensiable_param": {"enable_shm": True},
                    }
                ],
            },
        }
        self.assertTrue(is_multi_host_deployment(task))
        self.assertEqual(resolve_master_host(task), "10.0.0.1")
        self.assertEqual(resolve_global_nnodes(task), 2)
        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)
            self.assertEqual(len(plans), 1)
            self.assertTrue(plans[0].is_benchmark_runner)
            self.assertFalse(plans[0].isolate_ipc)
            self.assertEqual(plans[0].instance_config["num_devices"], 2)
            self.assertEqual(plans[0].instance_config["world_size"], 2)
            self.assertEqual(plans[0].instance_config["master_host"], "10.0.0.1")
            self.assertTrue(
                plans[0].instance_config["serving_extensiable_param"]["enable_shm"]
            )

    def test_apply_host_addr_override_single_instance(self):
        from xtask.core.service_group import apply_host_addr_override

        task = {
            "deployment": {
                "master_host": "10.0.0.1",
                "benchmark": {"runner_instance": "master-0"},
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "local_port": 8800,
                    }
                ],
            }
        }
        apply_host_addr_override(task, "10.0.0.9")
        self.assertEqual(task["host_addr"], "10.0.0.9")
        self.assertEqual(task["deployment"]["master_host"], "10.0.0.9")
        self.assertEqual(task["deployment"]["instances"][0]["host"], "10.0.0.9")
        self.assertNotIn("endpoint", task["deployment"]["benchmark"])
        plans = build_service_plans(task, tempfile.mkdtemp())
        self.assertEqual(get_benchmark_endpoint(task, plans), "http://10.0.0.9:8800")

    def test_non_pd_endpoint_uses_master_instance_local_port(self):
        task = {
            "deployment": {
                "master_host": "10.0.0.1",
                "benchmark": {"runner_instance": "master-0"},
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0, 1],
                        "local_port": 8900,
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "role": "worker",
                        "device_ids": [0, 1],
                        "local_port": 8800,
                    },
                ],
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = build_service_plans(task, temporary_directory)
            self.assertEqual(
                get_benchmark_endpoint(task, plans), "http://10.0.0.1:8900"
            )

    def test_pd_endpoint_uses_pd_control_host_and_http_port(self):
        task = {
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.9",
                    "http_server_port": 38888,
                },
                "benchmark": {"runner_instance": "prefill-0"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [
                            {
                                "name": "prefill-0",
                                "host": "10.0.0.1",
                                "device_ids": [0],
                                "local_port": 48000,
                            }
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "nodes": [
                            {
                                "name": "decode-0",
                                "host": "10.0.0.2",
                                "device_ids": [0],
                                "local_port": 48000,
                            }
                        ],
                    },
                ],
            }
        }
        self.assertEqual(get_benchmark_endpoint(task, []), "http://10.0.0.9:38888")
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "must match a service node host"):
                build_service_plans(task, temporary_directory)

    def test_topology_ignores_legacy_explicit_endpoint(self):
        task = {
            "deployment": {
                "master_host": "10.0.0.1",
                "benchmark": {"endpoint": "http://wrong-host:9999"},
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": [0],
                        "local_port": 8800,
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = build_service_plans(task, temporary_directory)
            self.assertEqual(
                get_benchmark_endpoint(task, plans), "http://10.0.0.1:8800"
            )

    def test_endpoint_derivation_rejects_missing_master_instance(self):
        task = {
            "deployment": {
                "master_host": "10.0.0.9",
                "instances": [
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "device_ids": [0],
                        "local_port": 8800,
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = build_service_plans(task, temporary_directory)
            with self.assertRaisesRegex(ValueError, "matches master_host"):
                get_benchmark_endpoint(task, plans)

    def test_resolve_master_and_global_nnodes(self):
        task = {
            "deployment": {
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "device_ids": list(range(16)),
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "role": "worker",
                        "device_ids": list(range(16)),
                    },
                ]
            },
        }
        self.assertEqual(resolve_master_host(task), "10.0.0.1")
        self.assertEqual(resolve_global_nnodes(task), 32)

    def test_merge_instance_sets_rank_and_master(self):
        task = {
            "case_name": "demo",
            "port": 8800,
        }
        instance = {
            "name": "worker-0",
            "host": "10.0.0.2",
            "role": "worker",
            "rank_start": 16,
            "device_ids": list(range(16)),
            "local_port": 8800,
            "serving_extensiable_param": {
                "enable_shm": True,
                "rank_tablefile": "/tmp/rank.json",
            },
        }
        merged = merge_instance_task_config(task, instance, "10.0.0.1", 32)
        self.assertEqual(merged["host_addr"], "10.0.0.2")
        self.assertEqual(merged["master_host"], "10.0.0.1")
        self.assertEqual(merged["rank_start"], 16)
        self.assertEqual(merged["num_devices"], 16)
        self.assertEqual(merged["global_nnodes"], 32)
        self.assertEqual(
            merged["serving_extensiable_param"]["rank_tablefile"],
            "/tmp/rank.json",
        )
        self.assertTrue(merged["serving_extensiable_param"]["enable_shm"])

    def test_merge_instance_startup_environment_replaces_task_level_list(self):
        task = {"startup_environment": ["export TOP=1"], "port": 8800}
        inherited = {
            "name": "worker-0",
            "host": "10.0.0.2",
            "device_ids": [0],
            "local_port": 8800,
        }
        replaced = {
            **inherited,
            "startup_environment": ["export NODE=1"],
        }
        cleared = {**inherited, "startup_environment": []}

        self.assertEqual(
            merge_instance_task_config(task, inherited, "10.0.0.1", 1)[
                "startup_environment"
            ],
            ["export TOP=1"],
        )
        self.assertEqual(
            merge_instance_task_config(task, replaced, "10.0.0.1", 1)[
                "startup_environment"
            ],
            ["export NODE=1"],
        )
        self.assertEqual(
            merge_instance_task_config(task, cleared, "10.0.0.1", 1)[
                "startup_environment"
            ],
            [],
        )
        self.assertEqual(task["startup_environment"], ["export TOP=1"])

    def test_startup_environment_follows_each_node_in_pd_and_service_groups(self):
        pd_task = {
            "case_name": "startup-env-pd",
            "startup_environment": ["export TOP=1"],
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1", "http_server_port": 38888},
                "benchmark": {"runner_instance": "prefill-0-node-0"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "startup_environment": ["export GROUP=1"],
                        "nodes": [
                            {"host": "10.0.0.1", "device_ids": [0, 1]},
                            {
                                "host": "10.0.0.2",
                                "device_ids": [0, 1],
                                "startup_environment": ["export NODE=1"],
                            },
                        ],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "nodes": [
                            {"host": "10.0.0.4", "device_ids": [0, 1]},
                            {
                                "host": "10.0.0.5",
                                "device_ids": [0, 1],
                                "startup_environment": [],
                            },
                        ],
                    },
                ],
            },
        }
        service_task = {
            "case_name": "startup-env-service",
            "deployment": {
                "mode": "service_routed",
                "master_host": "10.0.0.1",
                "xllm_service": {
                    "host": "10.0.0.1",
                    "http_server_port": 13888,
                },
                "service_instances": [
                    {
                        "name": "default-0",
                        "progress_conn_port": 9590,
                        "startup_environment": ["export GROUP=1"],
                        "nodes": [
                            {
                                "name": "default-0-node-0",
                                "host": "10.0.0.1",
                                "device_ids": [0, 1],
                                "local_port": 48000,
                                "hccl_if_port": 43439,
                            },
                            {
                                "name": "default-0-node-1",
                                "host": "10.0.0.1",
                                "device_ids": [2, 3],
                                "local_port": 48010,
                                "hccl_if_port": 43449,
                                "startup_environment": ["export NODE=1"],
                            },
                        ],
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            pd_plans = {
                plan.instance_name: plan.instance_config.get("startup_environment")
                for plan in build_service_plans(pd_task, tmp)
            }
            service_plans = {
                plan.instance_name: plan.instance_config.get("startup_environment")
                for plan in build_service_plans(service_task, tmp)
            }

        self.assertEqual(
            pd_plans,
            {
                "prefill-0-node-0": ["export GROUP=1"],
                "prefill-0-node-1": ["export NODE=1"],
                "decode-0-node-0": ["export TOP=1"],
                "decode-0-node-1": [],
            },
        )
        self.assertEqual(
            service_plans,
            {
                "default-0-node-0": ["export GROUP=1"],
                "default-0-node-1": ["export NODE=1"],
            },
        )

    def test_build_service_plans_without_top_level_duplicates(self):
        task = {
            "case_name": "demo",
            "deployment": {
                "benchmark": {"runner_instance": "master-0"},
                "instances": [
                    {
                        "name": "master-0",
                        "host": "127.0.0.1",
                        "role": "master",
                        "rank_start": 0,
                        "device_ids": list(range(16)),
                        "local_port": 8800,
                        "serving_extensiable_param": {
                            "enable_shm": True,
                            "dp_size": 4,
                        },
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "role": "worker",
                        "rank_start": 16,
                        "device_ids": list(range(16)),
                        "local_port": 8800,
                        "serving_extensiable_param": {
                            "enable_shm": True,
                            "dp_size": 8,
                        },
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)
            self.assertEqual(len(plans), 2)
            self.assertEqual(plans[0].global_nnodes, 32)
            self.assertEqual(plans[0].instance_config["num_devices"], 16)
            self.assertEqual(plans[0].instance_config["host_addr"], "127.0.0.1")
            self.assertEqual(
                plans[0].instance_config["serving_extensiable_param"]["dp_size"], 4
            )
            self.assertEqual(
                plans[1].instance_config["serving_extensiable_param"]["dp_size"], 8
            )
            self.assertTrue(plans[0].is_benchmark_runner)
            self.assertFalse(plans[0].isolate_ipc)
            self.assertFalse(plans[1].isolate_ipc)

    def test_build_service_plans(self):
        task = {
            "case_name": "demo",
            "deployment": {
                "benchmark": {"runner_instance": "master-0"},
                "instances": [
                    {
                        "name": "master-0",
                        "host": "127.0.0.1",
                        "role": "master",
                        "rank_start": 0,
                        "device_ids": list(range(16)),
                        "local_port": 8800,
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "role": "worker",
                        "rank_start": 16,
                        "device_ids": list(range(16)),
                        "local_port": 8800,
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)
            self.assertEqual(len(plans), 2)
            self.assertTrue(plans[0].is_local)
            self.assertFalse(plans[1].is_local)
            self.assertTrue(plans[0].is_benchmark_runner)
            self.assertFalse(plans[1].is_benchmark_runner)
            self.assertEqual(plans[1].rank_start, 16)
            self.assertTrue(
                os.path.isdir(os.path.join(tmp, "xllm_log", "worker-0"))
            )


class GenerateXllmMultiHostTests(unittest.TestCase):
    def test_wait_accepts_readiness_written_before_watchdog_starts(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import wait_xllm_serv_start

        with tempfile.TemporaryDirectory() as tmp:
            log_file = os.path.join(tmp, "node_0.log")
            with open(log_file, "w", encoding="utf-8") as stream:
                stream.write("DistributeWorker server started on address 127.0.0.1\n")
            with mock.patch(
                "xtask.utils.watchdog.WatchdogManager"
            ) as watchdog_manager:
                ready = wait_xllm_serv_start(
                    {"device_ids": [0]},
                    1,
                    tmp,
                    log_file=log_file,
                    success_keywords=["DistributeWorker server started"],
                )

        self.assertTrue(ready)
        watchdog_manager.assert_not_called()

    def test_wait_xllm_serv_start_uses_serving_ready_timeout(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import wait_xllm_serv_start

        with tempfile.TemporaryDirectory() as tmp:
            log_file = os.path.join(tmp, "node_0.log")
            open(log_file, "w", encoding="utf-8").close()
            with mock.patch(
                "xtask.utils.watchdog.WatchdogConfig"
            ) as watchdog_config, mock.patch(
                "xtask.utils.watchdog.WatchdogManager"
            ) as watchdog_manager:
                wait_xllm_serv_start(
                    {"device_ids": [0], "serving_ready_timeout": 1800},
                    1,
                    tmp,
                    log_file=log_file,
                )

        watchdog_config.assert_called_once()
        self.assertEqual(watchdog_config.call_args.kwargs["timeout"], 1800)
        watchdog_manager.assert_called_once()

    def test_wait_xllm_serv_start_falls_back_to_serving_timeout(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import wait_xllm_serv_start

        with tempfile.TemporaryDirectory() as tmp:
            log_file = os.path.join(tmp, "node_0.log")
            open(log_file, "w", encoding="utf-8").close()
            with mock.patch(
                "xtask.utils.watchdog.WatchdogConfig"
            ) as watchdog_config, mock.patch(
                "xtask.utils.watchdog.WatchdogManager"
            ):
                wait_xllm_serv_start(
                    {"device_ids": [0], "serving_timeout": 900},
                    1,
                    tmp,
                    log_file=log_file,
                )

        self.assertEqual(watchdog_config.call_args.kwargs["timeout"], 900)

    def test_generated_script_owns_logs_with_xtask_process_uid(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "11.87.191.75",
            "device_ids": [0],
            "world_size": 1,
            "model_path": "/export/home/models/Qwen3-8B",
            "userid": 1048,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.serv_fmk.xllm_serv.generate_xllm_serv.os.getuid",
                return_value=1011,
            ), mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                name, _ = generate_xllm_script(config, tmp, "start_xllm_serv.sh")
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                script = fh.read()

        self.assertIn("USERID=1011", script)
        self.assertNotIn("USERID=1048", script)
        self.assertIn('touch "$LOG_FILE"', script)
        self.assertIn('chown "$USERID" "$LOG_DIR" "$LOG_FILE"', script)

    def test_generated_multi_card_script_owns_each_log_with_xtask_uid(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "11.87.191.75",
            "device_ids": [0, 1],
            "world_size": 2,
            "model_path": "/export/home/models/Qwen3-8B",
            "userid": 1048,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.serv_fmk.xllm_serv.generate_xllm_serv.os.getuid",
                return_value=1011,
            ), mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                name, _ = generate_xllm_script(config, tmp, "start_xllm_serv.sh")
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                script = fh.read()

        self.assertIn("USERID=1011", script)
        self.assertNotIn("USERID=1048", script)
        self.assertIn('touch "$LOG_FILE"', script)
        self.assertIn('chown "$USERID" "$LOG_FILE"', script)

    def test_pd_script_uses_group_ports_and_rank_offset_transfer_ports(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "11.87.191.75",
            "master_host": "11.87.191.75",
            "device_ids": [2, 3],
            "world_size": 4,
            "global_nnodes": 4,
            "rank_start": 2,
            "port": 48010,
            "hccl_if_port": 43539,
            "progress_conn_port": 9690,
            "enable_disagg_pd": True,
            "pd_role": "PREFILL",
            "pd_master_host": "11.87.191.75",
            "transfer_port_base": 27000,
            "disagg_pd_port": 7787,
            "model_path": "/export/home/models/Qwen3-8B",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                name, _ = generate_xllm_script(
                    config, tmp, "start_xllm_serv.sh", allow_large_world=True
                )
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                script = fh.read()

        self.assertIn("export HCCL_IF_BASE_PORT=43539", script)
        self.assertIn("PROGRESS_CONN_PORT=9690", script)
        self.assertIn("--transfer_listen_port=$((27000+2+i))", script)
        self.assertIn("--disagg_pd_port=7787", script)

    def test_single_host_script_keeps_local_master_and_rank0(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "11.87.191.75",
            "port": 8800,
            "start_device_id": 0,
            "num_devices": 2,
            "world_size": 2,
            "model_path": "/export/home/models/Qwen3-8B",
            "communication_backend": "hccl",
            "userid": 1000,
            "serving_extensiable_param": {"enable_shm": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.utils.container_manager.path_mapping_in_container",
                side_effect=lambda p: p,
            ), mock.patch(
                "xtask.serv_fmk.xllm_serv.generate_xllm_serv.path_mapping_in_container",
                side_effect=lambda p: p,
                create=True,
            ):
                # path_mapping imported inside function
                with mock.patch(
                    "xtask.utils.container_manager.get_xtask_config_data",
                    return_value={"DOCKER_CONFIG": {"volumes": {}}},
                ):
                    name, path = generate_xllm_script(config, tmp, "start_xllm_serv.sh")
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                script = fh.read()
            self.assertIn('MASTER_HOST=11.87.191.75', script)
            self.assertIn('LOCAL_IP=11.87.191.75', script)
            self.assertIn('RANK_START=0', script)
            self.assertIn('LOCAL_NNODES=2', script)
            self.assertIn('GLOBAL_NNODES=2', script)
            self.assertIn('export ASCEND_RT_VISIBLE_DEVICES=0,1', script)
            self.assertIn('--nnodes="$GLOBAL_NNODES"', script)
            self.assertIn('NODE_RANK=$((RANK_START + i))', script)
            self.assertNotIn("--device_id", script)
            self.assertNotIn("--devices", script)

    def test_single_card_script_uses_physical_device_and_rank_zero(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "11.87.191.75",
            "port": 8800,
            "start_device_id": 3,
            "device_ids": [3],
            "num_devices": 1,
            "world_size": 1,
            "model_path": "/export/home/models/Qwen3-8B",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                name, _ = generate_xllm_script(config, tmp, "start_xllm_serv.sh")
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                script = fh.read()

            self.assertIn('START_DEVICE=3', script)
            self.assertIn('export ASCEND_RT_VISIBLE_DEVICES=3', script)
            self.assertNotIn("--device_id", script)
            self.assertNotIn("--devices", script)
            self.assertIn('--node_rank=0', script)

    def test_non_contiguous_device_ids_are_preserved_in_multi_card_script(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "11.87.191.75",
            "port": 8800,
            "device_ids": [4, 7, 11, 15],
            "world_size": 4,
            "model_path": "/export/home/models/Qwen3-8B",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                name, _ = generate_xllm_script(config, tmp, "start_xllm_serv.sh")
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                script = fh.read()

            self.assertIn("START_DEVICE=4", script)
            self.assertIn("LOCAL_NNODES=4", script)
            self.assertIn("export ASCEND_RT_VISIBLE_DEVICES=4,7,11,15", script)
            self.assertIn('DEVICE="${DEVICE_IDS[$i]}"', script)
            self.assertNotIn("--device_id", script)
            self.assertNotIn("--devices", script)
            self.assertNotIn("DEVICE=$((START_DEVICE + i))", script)

    def test_multi_host_worker_script_uses_master_host_and_rank_offset(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import (
            generate_xllm_serv_commands,
        )

        config = {
            "host_addr": "11.87.191.121",
            "master_host": "11.87.191.75",
            "port": 8800,
            "start_device_id": 0,
            "num_devices": 16,
            "world_size": 32,
            "global_nnodes": 32,
            "rank_start": 16,
            "model_path": "/export/home/models/Qwen3-8B",
            "communication_backend": "hccl",
            "userid": 1000,
            "serving_extensiable_param": {
                "rank_tablefile": "/export/home/models/rank.json",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                cmd = generate_xllm_serv_commands(
                    config,
                    {"engine_type": "xllm"},
                    tmp,
                    instance={"name": "worker-0", "role": "worker"},
                )
            self.assertIn("start_xllm_serv_worker-0.sh", cmd)
            with open(
                os.path.join(tmp, "start_xllm_serv_worker-0.sh"), encoding="utf-8"
            ) as fh:
                script = fh.read()
            self.assertIn("LOCAL_IP=11.87.191.121", script)
            self.assertIn("MASTER_HOST=11.87.191.75", script)
            self.assertIn("RANK_START=16", script)
            self.assertIn("GLOBAL_NNODES=32", script)
            self.assertIn("LOCAL_NNODES=16", script)
            self.assertNotIn("--device_id", script)
            self.assertNotIn("--devices", script)
            self.assertIn("--rank_tablefile=/export/home/models/rank.json", script)

    def test_service_routed_script_registers_default_instance_with_shared_etcd(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "10.0.0.1",
            "master_host": "10.0.0.1",
            "port": 48000,
            "device_ids": [0, 1],
            "world_size": 2,
            "global_nnodes": 2,
            "progress_conn_port": 9590,
            "model_path": "/models/Qwen3-8B",
            "enable_service_routing": True,
            "service_control_host": "10.0.0.1",
            "service_etcd_port": 13389,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                name, _ = generate_xllm_script(
                    config, tmp, "start_xllm_serv.sh", allow_large_world=True
                )
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                script = fh.read()

        self.assertIn("--enable_service_routing=true", script)
        self.assertIn("--etcd_addr=10.0.0.1:13389", script)
        self.assertNotIn("--enable_disagg_pd=true", script)

    def _generate_script(self, config, allow_large_world=False):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "xtask.utils.container_manager.get_xtask_config_data",
                return_value={"DOCKER_CONFIG": {"volumes": {}}},
            ):
                name, _ = generate_xllm_script(
                    config,
                    tmp,
                    "start_xllm_serv.sh",
                    allow_large_world=allow_large_world,
                )
            with open(os.path.join(tmp, name), encoding="utf-8") as fh:
                return fh.read()

    def test_omitted_startup_environment_inserts_nothing(self):
        config = {
            "host_addr": "11.87.191.75",
            "device_ids": [0, 1],
            "world_size": 2,
            "model_path": "/export/home/models/Qwen3-8B",
        }
        absent = self._generate_script(config)
        empty = self._generate_script({**config, "startup_environment": []})

        single_config = {**config, "device_ids": [3], "world_size": 1}
        single_absent = self._generate_script(single_config)
        single_empty = self._generate_script({**single_config, "startup_environment": []})

        self.assertEqual(absent, empty)
        self.assertEqual(single_absent, single_empty)
        for script in (absent, single_absent):
            self.assertNotIn("{STARTUP_ENVIRONMENT}", script)
            self.assertEqual(script.count("export HCCL_CONNECT_TIMEOUT=7200"), 1)
            for variable in ("INF_NAN_MODE_ENABLE", "INF_NAN_MODE_FORCE_DISABLE", "PROFILING_MODE"):
                self.assertNotIn(f"export {variable}=", script)
        self.assertIn("fi\n\n# 启动服务节点", absent)
        self.assertIn('chown "$USERID" "$LOG_DIR"\nIFS=', absent)
        self.assertIn('LOG_DIR="log"\nLOG_FILE="$LOG_DIR/node_0.log"', single_absent)

    def test_qwen_environment_is_case_scoped(self):
        config = {
            "host_addr": "127.0.0.1", "device_ids": [0, 1], "world_size": 2,
            "model_path": "/models/Qwen3.5-27B",
        }
        lines = ["export INF_NAN_MODE_ENABLE=0", "export INF_NAN_MODE_FORCE_DISABLE=1",
                 'export PROFILING_MODE="dynamic"']
        explicit = self._generate_script({**config, "startup_environment": lines})
        unrelated = self._generate_script({**config, "model_path": "/models/OtherModel"})
        for line in lines:
            self.assertEqual(explicit.count(line), 1)
            self.assertLess(explicit.index(line), explicit.index("IFS=',' read -r -a DEVICE_IDS"))
            self.assertNotIn(line, unrelated)

    def test_startup_environment_is_inserted_after_template_defaults(self):
        config = {
            "host_addr": "11.87.191.75",
            "device_ids": [3],
            "world_size": 1,
            "model_path": "/export/home/models/Qwen3-8B",
            "startup_environment": [
                "export HCCL_CONNECT_TIMEOUT=1800",
                "export ASCEND_SLOG_PRINT_TO_STDOUT=1",
                "echo keep-{MODEL_PATH}",
            ],
        }
        single = self._generate_script(config)
        multi = self._generate_script(
            {**config, "device_ids": [0, 1], "world_size": 2}
        )

        for script in (single, multi):
            self.assertNotIn("{STARTUP_ENVIRONMENT}", script)
            self.assertLess(
                script.index("export HCCL_CONNECT_TIMEOUT=7200"),
                script.index("export HCCL_CONNECT_TIMEOUT=1800"),
            )
            self.assertLess(
                script.index("rm -rf core.*"),
                script.index("export ASCEND_SLOG_PRINT_TO_STDOUT=1"),
            )
            self.assertIn("echo keep-{MODEL_PATH}", script)
            self.assertNotIn("echo keep-/export/home/models/Qwen3-8B", script)
        self.assertLess(
            single.index("export ASCEND_RT_VISIBLE_DEVICES=3"),
            single.index("export HCCL_CONNECT_TIMEOUT=1800"),
        )
        self.assertLess(
            single.index("export HCCL_CONNECT_TIMEOUT=1800"),
            single.index('LOG_FILE="$LOG_DIR/node_0.log"'),
        )
        self.assertLess(
            multi.index("export HCCL_CONNECT_TIMEOUT=1800"),
            multi.index("IFS=',' read -r -a DEVICE_IDS"),
        )
        self.assertLess(
            multi.index("export HCCL_CONNECT_TIMEOUT=1800"),
            multi.index("for (( i=0; i<$LOCAL_NNODES; i++ ))"),
        )

    def test_startup_environment_rejects_non_command_entries(self):
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_script

        config = {
            "host_addr": "11.87.191.75",
            "device_ids": [0],
            "world_size": 1,
            "model_path": "/models/Qwen3-8B",
            "startup_environment": [{"export": "FOO=1"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "shell command string"):
                generate_xllm_script(config, tmp, "start_xllm_serv.sh")


if __name__ == "__main__":
    unittest.main()
