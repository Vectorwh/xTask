import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core.service_group import (
    build_service_plans,
    cleanup_service_group,
    is_multi_host_deployment,
    get_benchmark_endpoint,
    merge_instance_task_config,
    normalize_pd_deployment,
    resolve_global_nnodes,
    resolve_master_host,
    validate_pd_deployment_authoring,
)


class ServiceGroupTests(unittest.TestCase):
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
                "pd_control": {"host": "10.0.0.1", "http_server_port": 38888},
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
                "pd_control": {
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
        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = build_service_plans(task, temporary_directory)
            self.assertEqual(
                get_benchmark_endpoint(task, plans), "http://10.0.0.9:38888"
            )

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


if __name__ == "__main__":
    unittest.main()
