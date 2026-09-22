"""TCPStore listen-port wait: derive the window and poll until free.

Does not start xLLM. Fake executor + fake clock; no host ss/npu-smi.
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from xtask.core import scheduler


def _plan(**kwargs):
    defaults = dict(
        instance_name="decode-0",
        host="h1",
        role="decode",
        rank_start=0,
        device_ids=list(range(16)),
        instance_config={
            "progress_conn_port": 8773,
            "world_size": 16,
            "communication_backend": "hccl",
        },
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _runtime(plan=None, readings=None, error=None):
    seq = list(readings or [])

    class _Executor:
        def run(self, cmd, timeout=20):
            if error is not None:
                raise error
            if not seq:
                return SimpleNamespace(stdout="")
            v = seq.pop(0)
            if v is None:
                return SimpleNamespace(stdout="")
            return SimpleNamespace(stdout=str(v))

    return SimpleNamespace(
        plan=plan or _plan(),
        container_manager=SimpleNamespace(executor=_Executor()),
    )


class TcpStoreListenPortsTests(unittest.TestCase):
    def test_missing_plan_returns_no_ports(self):
        self.assertEqual(scheduler._tcpstore_listen_ports(None), [])

    def test_window_starts_after_progress_conn_port(self):
        plan = _plan(
            instance_config={
                "progress_conn_port": 8773,
                "world_size": 16,
                "communication_backend": "hccl",
            }
        )
        ports = scheduler._tcpstore_listen_ports(plan)
        # 2026-09-21 deepseek_v4 dp4/ep16 decode crash bound 8813
        # (progress_conn_port 8773 + 40); the window must cover it.
        self.assertEqual(ports[0], 8774)
        self.assertIn(8813, ports)
        self.assertEqual(ports[-1], 8773 + 5 * 16 + 16)

    def test_world_size_one_has_no_fixed_window(self):
        plan = _plan(
            device_ids=[0],
            instance_config={
                "progress_conn_port": 8773,
                "world_size": 1,
                "communication_backend": "hccl",
            },
        )
        self.assertEqual(scheduler._tcpstore_listen_ports(plan), [])

    def test_default_base_when_field_missing(self):
        plan = _plan(
            device_ids=[0, 1],
            instance_config={"world_size": 2, "communication_backend": "hccl"},
        )
        ports = scheduler._tcpstore_listen_ports(plan)
        self.assertEqual(ports[0], 9591)
        self.assertEqual(ports[-1], 9590 + 5 * 2 + 16)

    def test_world_size_falls_back_to_device_count(self):
        plan = _plan(
            device_ids=[0, 1, 2],
            instance_config={
                "progress_conn_port": 9000,
                "communication_backend": "hccl",
            },
        )
        ports = scheduler._tcpstore_listen_ports(plan)
        self.assertEqual(ports[0], 9001)
        self.assertEqual(ports[-1], 9000 + 5 * 3 + 16)

    def test_invalid_base_returns_no_ports(self):
        plan = _plan(
            instance_config={
                "progress_conn_port": "not-a-port",
                "world_size": 16,
                "communication_backend": "hccl",
            }
        )
        self.assertEqual(scheduler._tcpstore_listen_ports(plan), [])

    def test_window_clamps_to_65535(self):
        plan = _plan(
            instance_config={
                "progress_conn_port": 65530,
                "world_size": 16,
                "communication_backend": "hccl",
            }
        )
        ports = scheduler._tcpstore_listen_ports(plan)
        self.assertEqual(ports[0], 65531)
        self.assertEqual(ports[-1], 65535)


class WaitTcpStoreListenPortsReleasedTests(unittest.TestCase):
    def test_no_tcpstore_ports_does_not_probe(self):
        rt = _runtime(
            plan=_plan(device_ids=[0], instance_config={"world_size": 1}),
            error=RuntimeError("should not run"),
        )
        scheduler._wait_tcpstore_listen_ports_released(rt, timeout_s=30)

    def test_already_free_returns_without_sleep(self):
        rt = _runtime(readings=[""])
        clock = {"t": 0.0}

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_tcpstore_listen_ports_released(rt, timeout_s=30)
        self.assertEqual(clock["t"], 0.0)

    def test_waits_until_ports_free(self):
        rt = _runtime(readings=["8813", "8813", ""])
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_tcpstore_listen_ports_released(
                rt, timeout_s=30, interval_s=1
            )
        self.assertEqual(clock["t"], 2)

    def test_timeout_proceeds_anyway(self):
        rt = _runtime(readings=["8813"] * 100)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep), \
                mock.patch.object(scheduler.logging, "warning") as warn:
            scheduler._wait_tcpstore_listen_ports_released(
                rt, timeout_s=5, interval_s=1
            )
        self.assertGreaterEqual(clock["t"], 5)
        self.assertTrue(
            any("proceeding anyway" in str(call) for call in warn.call_args_list),
            warn.call_args_list,
        )

    def test_unreadable_skips_wait(self):
        rt = _runtime(readings=["UNREADABLE"])
        clock = {"t": 0.0}

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_tcpstore_listen_ports_released(rt, timeout_s=30)
        self.assertEqual(clock["t"], 0.0)

    def test_default_timeout_comes_from_env(self):
        rt = _runtime(readings=["8813"] * 100)
        clock = {"t": 0.0}

        def fake_mono():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += s

        with mock.patch.dict(
            os.environ, {"XTASK_TCPSTORE_PORT_RELEASE_TIMEOUT": "3"}
        ), mock.patch.object(scheduler.time, "monotonic", fake_mono), \
                mock.patch.object(scheduler.time, "sleep", fake_sleep):
            scheduler._wait_tcpstore_listen_ports_released(rt, interval_s=1)
        self.assertGreaterEqual(clock["t"], 3)
        self.assertLess(clock["t"], 600)


class WaitTcpStoreListenPortsBeforeStartTests(unittest.TestCase):
    def test_non_multi_host_does_not_wait(self):
        with mock.patch(
            "xtask.core.service_group.run_on_runtimes_parallel"
        ) as parallel:
            scheduler._wait_tcpstore_listen_ports_before_start(
                {"multi_host": False}, [_runtime()]
            )
        parallel.assert_not_called()

    def test_multi_host_waits_on_each_runtime(self):
        runtimes = [_runtime(), _runtime()]
        with mock.patch(
            "xtask.core.service_group.run_on_runtimes_parallel"
        ) as parallel:
            scheduler._wait_tcpstore_listen_ports_before_start(
                {"multi_host": True}, runtimes
            )
        parallel.assert_called_once_with(
            runtimes, scheduler._wait_tcpstore_listen_ports_released
        )


if __name__ == "__main__":
    unittest.main()
