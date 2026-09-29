"""xLLM fixed-port reservation against ephemeral allocation.

Does not start xLLM. Fake container manager; no host sysctl writes.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler


def _plan(**kwargs):
    defaults = dict(
        instance_name="decode-0-node-0",
        host="11.87.191.121",
        role="decode",
        rank_start=0,
        device_ids=list(range(16)),
        instance_config={
            "progress_conn_port": 8773,
            "world_size": 16,
            "communication_backend": "hccl",
            "hccl_if_port": 43432,
            "port": 18004,
            "disagg_pd_port": 7777,
            "transfer_port_base": 26000,
        },
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _task_config():
    return {
        "deployment": {
            "pd_control": {
                "host": "11.87.191.75",
                "etcd_client_port": 22389,
                "etcd_peer_port": 22380,
                "http_server_port": 28888,
                "rpc_server_port": 38889,
            }
        }
    }


class _FakeManager:
    def __init__(self, current="", read_ok=True, write_ok=True, error=None):
        self.current = current
        self.read_ok = read_ok
        self.write_ok = write_ok
        self.error = error
        self.calls = []

    def exec_in_container(self, cmd="", timeout=300, **kwargs):
        self.calls.append(cmd)
        if self.error is not None:
            raise self.error
        if cmd.startswith("cat "):
            return SimpleNamespace(
                success=self.read_ok,
                stdout=self.current if self.read_ok else "",
                stderr="" if self.read_ok else "unreadable",
            )
        return SimpleNamespace(
            success=self.write_ok,
            stdout="",
            stderr="" if self.write_ok else "write failed",
        )


def _runtime(plan=None, manager=None):
    return SimpleNamespace(
        plan=plan or _plan(),
        container_manager=manager or _FakeManager(),
    )


class ParseReservedPortsTests(unittest.TestCase):
    def test_empty_value(self):
        self.assertEqual(scheduler._parse_reserved_ports(""), set())
        self.assertEqual(scheduler._parse_reserved_ports(None), set())

    def test_single_ports_and_ranges(self):
        self.assertEqual(
            scheduler._parse_reserved_ports("7777,8773,18004-18006"),
            {7777, 8773, 18004, 18005, 18006},
        )

    def test_garbage_items_are_ignored(self):
        self.assertEqual(
            scheduler._parse_reserved_ports("abc,10,20-15,99999,-5,30000"),
            {10, 30000},
        )

    def test_whitespace_tolerant(self):
        self.assertEqual(
            scheduler._parse_reserved_ports(" 100 , 200-201 ,"),
            {100, 200, 201},
        )


class FormatReservedPortsTests(unittest.TestCase):
    def test_compacts_consecutive_runs(self):
        self.assertEqual(
            scheduler._format_reserved_ports({8773, 8774, 8775, 9000, 9002}),
            "8773-8775,9000,9002",
        )

    def test_roundtrip_with_parse(self):
        ports = {7777, 8773} | set(range(8774, 8870)) | set(range(26000, 26016))
        formatted = scheduler._format_reserved_ports(ports)
        self.assertEqual(scheduler._parse_reserved_ports(formatted), ports)


class ControlPlanePortsTests(unittest.TestCase):
    def test_extracts_pd_control_ports(self):
        self.assertEqual(
            scheduler._control_plane_ports(_task_config()),
            {22380, 22389, 28888, 38889},
        )

    def test_tolerates_missing_config(self):
        self.assertEqual(scheduler._control_plane_ports(None), set())
        self.assertEqual(scheduler._control_plane_ports({}), set())
        self.assertEqual(scheduler._control_plane_ports({"deployment": {}}), set())


class XllmFixedPortsTests(unittest.TestCase):
    def test_covers_all_port_families(self):
        ports = set(
            scheduler._xllm_fixed_ports(_plan(), _task_config())
        )
        # Group master + derived TCPStore window (the incident ports).
        self.assertIn(8773, ports)
        for port in range(8774, 8870):
            self.assertIn(port, ports)
        # PD disagg + transfer.
        self.assertIn(7777, ports)
        self.assertTrue(set(range(26000, 26016)) <= ports)
        # HCCL IF base + per-device.
        self.assertTrue(set(range(43432, 43448)) <= ports)
        # Node service ports.
        self.assertTrue(set(range(18004, 18020)) <= ports)
        # Control plane.
        self.assertTrue({22380, 22389, 28888, 38889} <= ports)

    def test_single_card_has_no_tcpstore_window(self):
        plan = _plan(
            device_ids=[0],
            instance_config={
                "progress_conn_port": 9590,
                "world_size": 1,
                "communication_backend": "hccl",
                "port": 14830,
            },
        )
        ports = set(scheduler._xllm_fixed_ports(plan, None))
        self.assertIn(9590, ports)
        self.assertNotIn(9591, ports)
        self.assertIn(14830, ports)

    def test_missing_plan_returns_empty(self):
        self.assertEqual(scheduler._xllm_fixed_ports(None), [])


class ReservePortsOnRuntimeTests(unittest.TestCase):
    def test_merges_with_existing_reservations(self):
        manager = _FakeManager(current="7777,8773,60000")
        scheduler._reserve_ports_on_runtime(
            _runtime(manager=manager), [8774, 8775, 8776]
        )
        writes = [c for c in manager.calls if "printf" in c]
        self.assertEqual(len(writes), 1)
        value = writes[0].split("printf '%s' ", 1)[1].split(" >")[0]
        parsed = scheduler._parse_reserved_ports(value)
        self.assertIn(60000, parsed)
        self.assertIn(8774, parsed)
        self.assertIn(7777, parsed)

    def test_skips_write_when_already_covered(self):
        manager = _FakeManager(current="8774-8870,7777")
        scheduler._reserve_ports_on_runtime(
            _runtime(manager=manager), [8775, 8869]
        )
        self.assertTrue(all(c.startswith("cat ") for c in manager.calls))

    def test_read_failure_never_writes(self):
        manager = _FakeManager(read_ok=False)
        with mock.patch.object(scheduler.logging, "warning") as warn:
            scheduler._reserve_ports_on_runtime(
                _runtime(manager=manager), [8774]
            )
        self.assertTrue(all(c.startswith("cat ") for c in manager.calls))
        self.assertTrue(
            any("skipping" in str(call) for call in warn.call_args_list),
            warn.call_args_list,
        )

    def test_write_failure_warns_and_proceeds(self):
        manager = _FakeManager(write_ok=False)
        with mock.patch.object(scheduler.logging, "warning") as warn:
            scheduler._reserve_ports_on_runtime(
                _runtime(manager=manager), [8774]
            )
        self.assertEqual(len(manager.calls), 2)
        self.assertTrue(
            any("Failed to reserve" in str(call) for call in warn.call_args_list),
            warn.call_args_list,
        )

    def test_exception_is_swallowed(self):
        manager = _FakeManager(error=RuntimeError("docker exploded"))
        with mock.patch.object(scheduler.logging, "warning") as warn:
            scheduler._reserve_ports_on_runtime(
                _runtime(manager=manager), [8774]
            )
        self.assertTrue(
            any("proceeding anyway" in str(call) for call in warn.call_args_list),
            warn.call_args_list,
        )

    def test_empty_ports_is_a_no_op(self):
        manager = _FakeManager()
        scheduler._reserve_ports_on_runtime(_runtime(manager=manager), [])
        self.assertEqual(manager.calls, [])


class ReserveXllmPortsBeforeStartTests(unittest.TestCase):
    def test_env_flag_disables(self):
        manager = _FakeManager()
        with mock.patch.dict(
            "os.environ", {"XTASK_RESERVE_XLLM_PORTS": "0"}
        ):
            scheduler._reserve_xllm_ports_before_start(
                {"task_config": _task_config()}, [_runtime(manager=manager)]
            )
        self.assertEqual(manager.calls, [])

    def test_same_host_runtimes_are_grouped_into_one_reservation(self):
        manager_a = _FakeManager()
        manager_b = _FakeManager()
        runtime_a = _runtime(
            plan=_plan(instance_name="decode-0"), manager=manager_a
        )
        runtime_b = _runtime(
            plan=_plan(instance_name="decode-1"), manager=manager_b
        )
        scheduler._reserve_xllm_ports_before_start(
            {"task_config": _task_config()}, [runtime_a, runtime_b]
        )
        # Only the first runtime on the host performs the read/write.
        self.assertEqual(len(manager_a.calls), 2)
        self.assertEqual(manager_b.calls, [])

    def test_different_hosts_each_reserve(self):
        manager_a = _FakeManager()
        manager_b = _FakeManager()
        runtime_a = _runtime(
            plan=_plan(host="11.87.191.75"), manager=manager_a
        )
        runtime_b = _runtime(
            plan=_plan(host="11.87.191.121"), manager=manager_b
        )
        scheduler._reserve_xllm_ports_before_start(
            {"task_config": _task_config()}, [runtime_a, runtime_b]
        )
        self.assertEqual(len(manager_a.calls), 2)
        self.assertEqual(len(manager_b.calls), 2)


if __name__ == "__main__":
    unittest.main()
