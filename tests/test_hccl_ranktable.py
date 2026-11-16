"""Unit tests for HCCL ranktable merge/split and multi-host grouping."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.core.ranktable_manager import (
    generate_and_merge_for_group,
    group_instances_for_ranktable,
    inject_rank_tablefile,
    prepare_multihost_ranktables,
)
from xtask.core.service_group import ServiceLaunchPlan, build_service_plans
from xtask.utils.hccl_ranktable import (
    load_ranktable,
    merge_ranktables,
    split_ranktable,
)

ROOT = Path(__file__).resolve().parents[1]
RANKTABLE_DIR = ROOT / "config" / "ranktable"


def _make_plan(
    name: str,
    host: str,
    role: str,
    rank_start: int,
    device_ids,
    *,
    rank_tablefile: str = None,
) -> ServiceLaunchPlan:
    ext = {}
    if rank_tablefile:
        ext["rank_tablefile"] = rank_tablefile
    return ServiceLaunchPlan(
        instance_name=name,
        role=role,
        host=host,
        is_local=True,
        rank_start=rank_start,
        device_ids=list(device_ids),
        service_port=8800,
        master_host=host,
        global_nnodes=len(device_ids),
        instance_config={
            "serving_extensiable_param": dict(ext),
            "instance_name": name,
            "role": role,
        },
        output_dir=f"/tmp/{name}",
    )


class HcclRanktableMergeSplitTests(unittest.TestCase):
    def test_generate_group_coalesces_same_host_members_and_preserves_ranks(self):
        from xtask.core import ranktable_manager

        members = [
            _make_plan("node-0", "10.0.0.1", "master", 0, [0, 1]),
            _make_plan("node-1", "10.0.0.1", "worker", 2, [2, 3]),
        ]
        generated_tables = {}

        def generate(aggregate_plan, _root, staging, runtime=None):
            self.assertIsNone(runtime)
            path = os.path.join(staging, f"{aggregate_plan.instance_name}.json")
            generated_tables[path] = {
                "version": "1.2",
                "server_count": "1",
                "server_list": [
                    {
                        "server_id": aggregate_plan.host,
                        "host_ip": aggregate_plan.host,
                        "container_ip": aggregate_plan.host,
                        "device": [
                            {
                                "device_id": str(device_id),
                                "device_ip": f"192.0.2.{device_id + 1}",
                                "rank_id": str(local_rank),
                                "super_device_id": str(device_id),
                            }
                            for local_rank, device_id in enumerate(
                                aggregate_plan.device_ids
                            )
                        ],
                    }
                ],
                "status": "completed",
            }
            return path

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            ranktable_manager,
            "_remote_generate_host_table",
            side_effect=generate,
        ) as generate_mock, mock.patch.object(
            ranktable_manager,
            "load_ranktable",
            side_effect=lambda path: generated_tables[path],
        ), mock.patch.object(ranktable_manager, "is_local_host", return_value=True):
            host_path, _ = generate_and_merge_for_group(
                members,
                output_dir=tmp,
                group_name="default",
                workspace_root=str(ROOT),
            )
            merged = load_ranktable(host_path)

        self.assertEqual(generate_mock.call_count, 1)
        aggregate_plan = generate_mock.call_args.args[0]
        self.assertEqual(aggregate_plan.device_ids, [0, 1, 2, 3])
        self.assertEqual(merged["server_count"], "1")
        self.assertEqual(
            [
                int(device["rank_id"])
                for device in merged["server_list"][0]["device"]
            ],
            [0, 1, 2, 3],
        )

    def test_copies_host_hccn_tool_when_runtime_image_does_not_have_it(self):
        from xtask.core import ranktable_manager

        manager = mock.Mock()
        manager.exec_in_container.return_value = SimpleNamespace(success=False)
        manager.executor.run.side_effect = [
            SimpleNamespace(success=True, stdout="/usr/bin/hccn_tool\n", stderr=""),
            SimpleNamespace(success=True, stdout="", stderr=""),
        ]
        runtime = SimpleNamespace(
            plan=SimpleNamespace(host="11.87.191.82"),
            container_config=SimpleNamespace(container_name="decode-container"),
            container_manager=manager,
        )

        ranktable_manager._ensure_hccn_tool_in_runtime(runtime)

        manager.exec_in_container.assert_called_once_with(
            "decode-container", "command -v hccn_tool", timeout=30
        )
        self.assertEqual(
            manager.executor.run.call_args_list[0].args[0],
            "command -v hccn_tool",
        )
        self.assertEqual(
            manager.executor.run.call_args_list[1].args[0],
            "docker cp /usr/bin/hccn_tool decode-container:/usr/bin/hccn_tool",
        )

    def test_generates_remote_ranktable_inside_prepared_container(self):
        from xtask.core import ranktable_manager

        plan = _make_plan(
            "decode-0-node-0",
            "11.87.191.82",
            "decode",
            0,
            range(16),
        )
        plan.is_local = False
        executor = mock.Mock()
        manager = mock.Mock()
        manager.executor = executor
        manager.exec_in_container.return_value = SimpleNamespace(
            success=True,
            stdout="",
            stderr="",
        )
        runtime = SimpleNamespace(
            plan=plan,
            container_config=SimpleNamespace(container_name="decode-container"),
            container_manager=manager,
        )

        def download(_remote_path, local_path):
            Path(local_path).write_text("{}\n", encoding="utf-8")
            return True

        executor.download.side_effect = download
        executor.run.return_value = SimpleNamespace(success=True)

        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            ranktable_manager,
            "path_mapping_in_container",
            side_effect=lambda path: path.replace(
                temporary_directory, "/workspace/ranktable/.staging_default"
            ).replace(str(ROOT), "/workspace"),
        ):
            generated = ranktable_manager._remote_generate_host_table(
                plan,
                str(ROOT),
                temporary_directory,
                runtime=runtime,
            )

        self.assertEqual(
            manager.exec_in_container.call_args.args[0], "decode-container"
        )
        command = manager.exec_in_container.call_args.args[1]
        self.assertIn("/workspace/tools/generate_hccl_ranktable.py", command)
        self.assertIn("--server-ip 11.87.191.82", command)
        self.assertIn("--device-ids 0-15", command)
        self.assertIn("chmod -R a+rwX /workspace/ranktable", command)
        self.assertIn("chmod 0644", command)
        executor.download.assert_called_once_with(
            generated, generated
        )
        executor.run.assert_called_once()

    def test_merge_75_121_16x2_matches_reference(self):
        t75 = load_ranktable(str(RANKTABLE_DIR / "ranktable_75_16.json"))
        t121 = load_ranktable(str(RANKTABLE_DIR / "ranktable_121_16.json"))
        ref = load_ranktable(str(RANKTABLE_DIR / "ranktable_75_121_16x2.json"))

        # Checked-in single-host samples omit super_pod_list; pass pod id like live gen.
        merged = merge_ranktables(
            [t75, t121],
            super_pod_id=ref["super_pod_list"][0]["super_pod_id"],
        )

        self.assertEqual(merged["version"], "1.2")
        self.assertEqual(merged["status"], "completed")
        self.assertEqual(merged["server_count"], "2")
        self.assertEqual(len(merged["server_list"]), 2)

        # Global rank_id continuity: 0..15 then 16..31
        ranks = [
            int(dev["rank_id"])
            for server in merged["server_list"]
            for dev in server["device"]
        ]
        self.assertEqual(ranks, list(range(32)))

        # Local device_id / device_ip / super_device_id preserved from each host
        for i, (got, expect) in enumerate(
            zip(merged["server_list"][0]["device"], t75["server_list"][0]["device"])
        ):
            self.assertEqual(got["device_id"], expect["device_id"], f"dev {i}")
            self.assertEqual(got["device_ip"], expect["device_ip"], f"dev {i}")
            self.assertEqual(got["super_device_id"], expect["super_device_id"], f"dev {i}")
            self.assertEqual(got["rank_id"], str(i))

        for i, (got, expect) in enumerate(
            zip(merged["server_list"][1]["device"], t121["server_list"][0]["device"])
        ):
            self.assertEqual(got["device_id"], expect["device_id"])
            self.assertEqual(got["device_ip"], expect["device_ip"])
            self.assertEqual(got["super_device_id"], expect["super_device_id"])
            self.assertEqual(got["rank_id"], str(16 + i))

        # super_pod_list covers both server_ids
        self.assertIn("super_pod_list", merged)
        pod = merged["super_pod_list"][0]
        self.assertEqual(pod["super_pod_id"], ref["super_pod_list"][0]["super_pod_id"])
        server_ids = [item["server_id"] for item in pod["server_list"]]
        self.assertEqual(server_ids, ["11.87.191.75", "11.87.191.121"])

        # Semantic match vs checked-in multi-host reference
        self.assertEqual(
            [s["server_id"] for s in merged["server_list"]],
            [s["server_id"] for s in ref["server_list"]],
        )
        ref_ranks = [
            int(dev["rank_id"])
            for server in ref["server_list"]
            for dev in server["device"]
        ]
        self.assertEqual(ranks, ref_ranks)

    def test_split_roundtrip_normalizes_local_ranks(self):
        t75 = load_ranktable(str(RANKTABLE_DIR / "ranktable_75_16.json"))
        t121 = load_ranktable(str(RANKTABLE_DIR / "ranktable_121_16.json"))
        merged = merge_ranktables([t75, t121])
        parts = split_ranktable(merged)

        self.assertEqual(len(parts), 2)
        for part, original in zip(parts, [t75, t121]):
            self.assertEqual(part["server_count"], "1")
            local_ranks = [int(d["rank_id"]) for d in part["server_list"][0]["device"]]
            self.assertEqual(local_ranks, list(range(16)))
            self.assertEqual(
                part["server_list"][0]["server_id"],
                original["server_list"][0]["server_id"],
            )
            # Re-merge of split tables yields same global ranks
        rematched = merge_ranktables(parts)
        rematch_ranks = [
            int(dev["rank_id"])
            for server in rematched["server_list"]
            for dev in server["device"]
        ]
        self.assertEqual(rematch_ranks, list(range(32)))

    def test_merge_2x2_renumbers(self):
        # Minimal 2-device hosts mimicking 2x2 sample layout
        a = {
            "version": "1.2",
            "server_count": "1",
            "status": "completed",
            "server_list": [
                {
                    "server_id": "11.87.191.75",
                    "host_ip": "11.87.191.75",
                    "container_ip": "11.87.191.75",
                    "device": [
                        {
                            "device_id": "0",
                            "device_ip": "1.1.1.1",
                            "rank_id": "0",
                            "super_device_id": "1",
                        },
                        {
                            "device_id": "1",
                            "device_ip": "1.1.1.2",
                            "rank_id": "1",
                            "super_device_id": "2",
                        },
                    ],
                }
            ],
            "super_pod_list": [
                {"super_pod_id": "2", "server_list": [{"server_id": "11.87.191.75"}]}
            ],
        }
        b = {
            "version": "1.2",
            "server_count": "1",
            "status": "completed",
            "server_list": [
                {
                    "server_id": "11.87.191.121",
                    "host_ip": "11.87.191.121",
                    "container_ip": "11.87.191.121",
                    "device": [
                        {
                            "device_id": "0",
                            "device_ip": "2.2.2.1",
                            "rank_id": "0",
                            "super_device_id": "3",
                        },
                        {
                            "device_id": "1",
                            "device_ip": "2.2.2.2",
                            "rank_id": "1",
                            "super_device_id": "4",
                        },
                    ],
                }
            ],
            "super_pod_list": [
                {"super_pod_id": "2", "server_list": [{"server_id": "11.87.191.121"}]}
            ],
        }
        merged = merge_ranktables([a, b])
        ranks = [
            int(d["rank_id"]) for s in merged["server_list"] for d in s["device"]
        ]
        self.assertEqual(ranks, [0, 1, 2, 3])
        self.assertEqual(merged["server_count"], "2")


class RanktableGroupingTests(unittest.TestCase):
    def test_non_pd_is_single_default_group(self):
        task = {
            "case_name": "mix",
            "host_addr": "10.0.0.1",
            "port": 8800,
            "world_size": 32,
            "deployment": {
                "instances": [
                    {
                        "name": "master-0",
                        "host": "10.0.0.1",
                        "role": "master",
                        "rank_start": 0,
                        "device_ids": list(range(16)),
                        "service": {"port": 8800},
                    },
                    {
                        "name": "worker-0",
                        "host": "10.0.0.2",
                        "role": "worker",
                        "rank_start": 16,
                        "device_ids": list(range(16)),
                        "service": {"port": 8800},
                    },
                ]
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)
            groups = group_instances_for_ranktable(task, plans)
        self.assertEqual(list(groups.keys()), ["default"])
        self.assertEqual(
            [p.instance_name for p in groups["default"]],
            ["master-0", "worker-0"],
        )

    def test_pd_groups_by_logical_instance_name(self):
        task = {
            "case_name": "pd",
            "host_addr": "10.0.0.1",
            "port": 8800,
            "world_size": 32,
            "deployment": {
                "mode": "pd_disaggregated",
                "xllm_service": {"host": "10.0.0.1"},
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [
                            {
                                "name": "prefill-0",
                                "host": "10.0.0.1",
                                "rank_start": 0,
                                "device_ids": list(range(16)),
                                "service": {"port": 8800},
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
                                "rank_start": 0,
                                "device_ids": list(range(16)),
                                "service": {"port": 8800},
                            }
                        ],
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)
            groups = group_instances_for_ranktable(task, plans)
        self.assertEqual(set(groups.keys()), {"prefill-0", "decode-0"})
        self.assertEqual(len(groups["prefill-0"]), 1)
        self.assertEqual(len(groups["decode-0"]), 1)
        self.assertEqual(groups["prefill-0"][0].instance_name, "prefill-0")
        self.assertEqual(groups["decode-0"][0].instance_name, "decode-0")

    def test_service_routed_groups_by_independent_default_instance(self):
        task = {
            "deployment": {
                "mode": "service_routed",
                "xllm_service": {"host": "10.0.0.1"},
                "service_instances": [
                    {
                        "name": "default-0",
                        "nodes": [
                            {"host": "10.0.0.1", "device_ids": [0, 1], "local_port": 48000},
                            {"host": "10.0.0.2", "device_ids": [0, 1], "local_port": 48000},
                        ],
                    },
                    {
                        "name": "default-1",
                        "nodes": [
                            {"host": "10.0.0.3", "device_ids": [0, 1], "local_port": 48000}
                        ],
                    },
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            plans = build_service_plans(task, tmp)
            groups = group_instances_for_ranktable(task, plans)

        self.assertEqual(list(groups), ["default-0", "default-1"])
        self.assertEqual(
            [plan.instance_name for plan in groups["default-0"]],
            ["default-0-node-0", "default-0-node-1"],
        )
        self.assertEqual(
            [plan.instance_name for plan in groups["default-1"]],
            ["default-1-node-0"],
        )

    def test_hierarchical_pd_groups_generate_ranktables_independently(self):
        task = {
            "case_name": "pd-hierarchical",
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "xllm_service": {"host": "10.0.0.1"},
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
                        ],
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "xtask.core.ranktable_manager.generate_and_merge_for_group",
            return_value=("/tmp/merged.json", "/workspace/merged.json"),
        ) as generate:
            plans = build_service_plans(task, tmp)
            injected = prepare_multihost_ranktables(task, plans, tmp)

        self.assertEqual(set(injected), {"prefill-0", "decode-0"})
        self.assertEqual(
            [call.kwargs["group_name"] for call in generate.call_args_list],
            ["prefill-0", "decode-0"],
        )
        self.assertEqual(
            {
                plan.instance_name
                for plan in plans
                if plan.instance_config.get("serving_extensiable_param", {}).get(
                    "rank_tablefile"
                )
            },
            {
                "prefill-0-node-0",
                "prefill-0-node-1",
                "decode-0-node-0",
                "decode-0-node-1",
            },
        )

    def test_inject_skips_when_manual_present_semantics(self):
        plan = _make_plan(
            "master-0",
            "10.0.0.1",
            "master",
            0,
            range(2),
            rank_tablefile="/manual/path.json",
        )
        # inject always writes; prepare_multihost skips calling it for manuals.
        inject_rank_tablefile(plan, "/auto/path.json")
        self.assertEqual(
            plan.instance_config["serving_extensiable_param"]["rank_tablefile"],
            "/auto/path.json",
        )


if __name__ == "__main__":
    unittest.main()
